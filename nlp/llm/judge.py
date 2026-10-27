"""Summary judge (a second Gemini model checks the summaries).

Reuses the sequential, spend-safe request loop of the classifier runner
(`nlp.llm.classify.run_queue`): reserve -> wait -> generate -> ledger.record -> cache.put.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import pandas as pd

from nlp.budget import Ledger, RateLimiter, estimate_run
from nlp.config import Config
from nlp.data import clean_text
from nlp.llm.classify import estimate_request_tokens, run_queue
from nlp.llm.client import GeminiClient
from nlp.store import JsonlCache

_BOOL = {"type": "boolean"}
_STR = {"type": "string"}
SUMMARY_FIELDS = ["faithful", "covers_main_issue", "invented_details"]
SUMMARY_JUDGE_PROMPT_VERSION = "sj-v1"


def _schema(fields: list[str]) -> dict:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"complaint_id": _STR, **{f: _BOOL for f in fields}},
            "required": ["complaint_id", *fields],
        },
    }


SUMMARY_SCHEMA = _schema(SUMMARY_FIELDS)


def _check_model(cfg: Config, model: str | None) -> str:
    model = model or cfg.judge_model
    if model == cfg.classifier_model:
        raise ValueError(f"judge model must differ from the classifier model ({model})")
    return model


def parse_flags(raw: str, batch_ids: list[str], fields: list[str]) -> tuple[list[dict], list[str]]:
    """Return (valid records, ids to retry). Fields must be real booleans."""
    try:
        items = json.loads(raw)
    except (ValueError, TypeError):
        return [], list(batch_ids)
    if isinstance(items, dict):
        items = [items]
    wanted = {str(i) for i in batch_ids}
    seen: set[str] = set()
    good: list[dict] = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        cid = str(it.get("complaint_id"))
        if cid not in wanted or cid in seen:
            continue
        seen.add(cid)
        if all(isinstance(it.get(f), bool) for f in fields):
            good.append({"complaint_id": cid, **{f: it[f] for f in fields}})
    ok = {g["complaint_id"] for g in good}
    return good, [str(i) for i in batch_ids if str(i) not in ok]


def _run(
    rows: pd.DataFrame,
    cfg: Config,
    client: GeminiClient,
    ledger: Ledger,
    cache: JsonlCache,
    *,
    tag: str,
    prompt_version: str,
    inputs: list[str],
    model: str,
    fields: list[str],
    schema: dict,
    system: str,
    render: Callable[[pd.Series, str], str],
    limiter: RateLimiter | None,
    sleep: Callable[[float], None],
) -> list[dict]:
    limiter = limiter or RateLimiter(cfg.rpm)

    def key(cid: str) -> str:
        return f"{cid}|{tag}|{model}|{prompt_version}"

    by_id: dict[str, pd.Series] = {}
    for _, r in rows.iterrows():
        by_id.setdefault(str(r["complaint_id"]), r)
    ids = list(by_id)

    def seen(cid: str) -> dict:
        return {f"in_{c}": str(by_id[cid][c]) for c in inputs}

    def fresh(cid: str) -> dict | None:
        """Cached verdict, only if it was judged on the same labels/summary as now."""
        rec = cache.get(key(cid))
        if rec is None or any(rec.get(k) != v for k, v in seen(cid).items()):
            return None
        return rec

    queue = [c for c in ids if fresh(c) is None]

    def request(batch_ids: list[str]) -> tuple[str, str]:
        parts = [
            render(by_id[c], clean_text(str(by_id[c]["narrative"]), cfg.max_chars))
            for c in batch_ids
        ]
        user = "Judge these items.\n\n" + "\n\n---\n\n".join(parts)
        return f"[prompt {prompt_version}] {system}", user

    def handle(raw: str, batch_ids: list[str]) -> list[str]:
        good, retry = parse_flags(raw, batch_ids, fields)
        for rec in good:
            cache.put(
                key(rec["complaint_id"]),
                {**rec, "model_id": model, **seen(rec["complaint_id"])},
            )
        return retry

    failed = run_queue(
        queue, batch_size=cfg.batch_size, client=client, ledger=ledger, limiter=limiter,
        sleep=sleep, model=model, thinking=cfg.thinking, schema=schema,
        request=request, handle=handle, tag=f"{tag}:{prompt_version}",
    )  # fmt: skip
    if failed:
        print(f"{len(failed)} row(s) failed {tag} validation after retries; left absent.")
    return [v for c in ids if (v := fresh(c)) is not None]


_SUMMARY_SYSTEM = (
    "You check summaries of US consumer financial complaints against the complaint text. "
    "For each item return faithful (every statement in the summary is supported by the "
    "complaint), covers_main_issue (the summary captures the consumer's main problem) and "
    "invented_details (the summary contains details not in the complaint). Return a JSON "
    "array with one object per item (complaint_id, faithful, covers_main_issue, "
    "invented_details), using the given complaint_id values unchanged."
)


def judge_summaries(
    rows: pd.DataFrame,
    cfg: Config,
    client: GeminiClient,
    ledger: Ledger,
    cache: JsonlCache,
    model: str | None = None,
    *,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """`rows` = complaint_id, narrative, summary -> complaint_id, faithful, ... (bools)."""
    model = _check_model(cfg, model)

    recs = _run(
        rows, cfg, client, ledger, cache, tag="summary_judge",
        prompt_version=SUMMARY_JUDGE_PROMPT_VERSION, inputs=["summary"], model=model, fields=SUMMARY_FIELDS,
        schema=SUMMARY_SCHEMA, system=_SUMMARY_SYSTEM, render=_summary_render, limiter=limiter,
        sleep=sleep,
    )  # fmt: skip
    return pd.DataFrame(recs, columns=["complaint_id", *SUMMARY_FIELDS])


SUMMARY_PLACEHOLDER = "x" * 450  # a 60-word summary, rounded up (summaries do not exist yet)


def _summary_render(r, text: str) -> str:
    return f"complaint_id: {r['complaint_id']}\nnarrative: {text}\nsummary: {r['summary']}"


def estimate_summary_judge_usd(
    cfg: Config, rows: pd.DataFrame, model: str | None = None
) -> float:
    """Worst-case USD for judging `rows` with the summary judge's own (short) prompt."""
    model = model or cfg.judge_model
    system = f"[prompt {SUMMARY_JUDGE_PROMPT_VERSION}] {_SUMMARY_SYSTEM}"
    ids = [str(c) for c in rows["complaint_id"]]
    texts = [clean_text(str(t), cfg.max_chars) for t in rows["narrative"]]
    total = 0.0
    for i in range(0, len(ids), cfg.batch_size):
        parts = [
            _summary_render({"complaint_id": c, "summary": SUMMARY_PLACEHOLDER}, t)
            for c, t in zip(ids[i : i + cfg.batch_size], texts[i : i + cfg.batch_size])
        ]
        user = "Judge these items.\n\n" + "\n\n---\n\n".join(parts)
        est_in, est_out = estimate_request_tokens(system, user, len(parts), cfg.thinking)
        total += estimate_run(1, est_in, est_out, model, cfg.prices)
    return total
