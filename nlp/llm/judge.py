"""JEB judge (a second model checks Gemini's classification) and summary judge.

Both reuse the sequential, spend-safe request loop of the classifier runner
(`nlp.llm.classify.run_queue`): reserve -> wait -> generate -> ledger.record -> cache.put.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import pandas as pd

from nlp.budget import Ledger, RateLimiter
from nlp.config import Config
from nlp.data import clean_text
from nlp.llm.classify import run_queue
from nlp.llm.client import GeminiClient
from nlp.schema import VALIDATOR_RESULTS_COLS
from nlp.store import JsonlCache

_BOOL = {"type": "boolean"}
_STR = {"type": "string"}
JEB_FIELDS = ["product_ok", "issue_ok", "sub_issue_ok"]
SUMMARY_FIELDS = ["faithful", "covers_main_issue", "invented_details"]
JEB_PROMPT_VERSION = "jeb-v1"
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


JEB_SCHEMA = _schema(JEB_FIELDS)
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
        request=request, handle=handle,
    )  # fmt: skip
    if failed:
        print(f"{len(failed)} row(s) failed {tag} validation after retries; left absent.")
    return [v for c in ids if (v := fresh(c)) is not None]


_JEB_SYSTEM = (
    "You are an independent judge of US consumer financial complaint classification. "
    "For each complaint you are shown another model's chosen Product, Issue and Sub-issue. "
    "Decide separately whether each chosen value is correct for the complaint text "
    "(product_ok, issue_ok, sub_issue_ok). Judge only against the complaint text. "
    "Return a JSON array with one object per complaint (complaint_id, product_ok, issue_ok, "
    "sub_issue_ok), using the given complaint_id values unchanged."
)


def judge_classifications(
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
    """JEB: `rows` = complaint_id, narrative, product, issue, sub_issue (Gemini's labels).

    Returns validator_results rows (validator `jeb`, score 1.0/0.0) for product, issue,
    sub_issue, and path (all three ok).
    """
    model = _check_model(cfg, model)

    def render(r: pd.Series, text: str) -> str:
        return (
            f"complaint_id: {r['complaint_id']}\nnarrative: {text}\n"
            f"chosen product: {r['product']}\nchosen issue: {r['issue']}\n"
            f"chosen sub_issue: {r['sub_issue']}"
        )

    recs = _run(
        rows, cfg, client, ledger, cache, tag="jeb", prompt_version=JEB_PROMPT_VERSION,
        inputs=["product", "issue", "sub_issue"], model=model, fields=JEB_FIELDS,
        schema=JEB_SCHEMA, system=_JEB_SYSTEM, render=render, limiter=limiter, sleep=sleep,
    )  # fmt: skip
    out = []
    for r in recs:
        flags = {
            "product": r["product_ok"],
            "issue": r["issue_ok"],
            "sub_issue": r["sub_issue_ok"],
        }
        flags["path"] = all(flags.values())
        for level, ok in flags.items():
            out.append((r["complaint_id"], "jeb", level, bool(ok), 1.0 if ok else 0.0))
    return pd.DataFrame(out, columns=VALIDATOR_RESULTS_COLS)


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

    def render(r: pd.Series, text: str) -> str:
        return f"complaint_id: {r['complaint_id']}\nnarrative: {text}\nsummary: {r['summary']}"

    recs = _run(
        rows, cfg, client, ledger, cache, tag="summary_judge",
        prompt_version=SUMMARY_JUDGE_PROMPT_VERSION, inputs=["summary"], model=model, fields=SUMMARY_FIELDS,
        schema=SUMMARY_SCHEMA, system=_SUMMARY_SYSTEM, render=render, limiter=limiter, sleep=sleep,
    )  # fmt: skip
    return pd.DataFrame(recs, columns=["complaint_id", *SUMMARY_FIELDS])
