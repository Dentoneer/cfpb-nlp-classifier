"""Batch classification with Gemini: prompt, parsing/validation, resumable budgeted runner."""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable

import pandas as pd

from nlp.budget import BudgetExceeded, DailyQuotaExhausted, Ledger, RateLimiter
from nlp.config import Config
from nlp.data import clean_text
from nlp.llm.client import GeminiClient, RateLimited
from nlp.schema import PREDICTIONS_COLS
from nlp.sheets import taxonomy_snapshot_hash
from nlp.store import JsonlCache
from nlp.taxonomy import normalize_path, taxonomy_prompt_text

_STR = {"type": "string"}
RESPONSE_SCHEMA: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "complaint_id": _STR,
            "product": _STR,
            "issue": _STR,
            "sub_issue": _STR,
            "summary": _STR,
            "confidence": {"type": "number"},
        },
        "required": ["complaint_id", "product", "issue", "sub_issue", "summary", "confidence"],
    },
}

MAX_RETRIES = 2  # re-queues per row after the first attempt
MAX_RATE_LIMIT_ATTEMPTS = 5
_PARSED_COLS = [
    "complaint_id", "product", "issue", "sub_issue", "summary", "confidence"
]  # fmt: skip


def build_prompt(batch: pd.DataFrame, taxonomy_text: str, prompt_version: str) -> tuple[str, str]:
    """Return (system, user). `batch` needs complaint_id and (already cleaned) narrative."""
    system = (
        f"[prompt {prompt_version}] You classify US consumer financial complaints.\n"
        "Classify each complaint into exactly one path (Product, Issue, Sub-issue) from the "
        "taxonomy below. Copy the Product, Issue and Sub-issue strings exactly as written; "
        'use the sub-issue "(none)" only where the taxonomy lists it.\n'
        "For each complaint also write a summary of at most 60 words that uses only facts "
        "stated in the complaint text (no guesses, no outside knowledge), and a confidence "
        "between 0 and 1 that the chosen path is correct.\n"
        "Return a JSON array with one object per complaint, following the response schema "
        "(complaint_id, product, issue, sub_issue, summary, confidence). Use the given "
        "complaint_id values unchanged and include every complaint exactly once.\n\n"
        f"TAXONOMY:\n{taxonomy_text}"
    )
    parts = [
        f"complaint_id: {cid}\nnarrative: {text}"
        for cid, text in zip(batch["complaint_id"], batch["narrative"])
    ]
    user = "Classify these complaints.\n\n" + "\n\n---\n\n".join(parts)
    return system, user


def estimate_request_tokens(system: str, user: str, n_rows: int, thinking: str) -> tuple[int, int]:
    """Conservative (est_in, est_out) for Ledger.reserve."""
    allowance = {"none": 0, "minimal": 0, "low": 512}.get(thinking, 2048)
    est_in = (len(system) + len(user)) // 4
    est_out = int(n_rows * (150 + allowance) * 1.5)
    return est_in, est_out


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=_PARSED_COLS)


def _confidence(v: object) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return None if math.isnan(f) else min(1.0, max(0.0, f))


def parse_and_validate(
    raw: str, batch_ids: list[str], taxonomy: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Return (valid rows with exact taxonomy strings, ids to retry).

    Invalid path or confidence, or missing id -> retry; duplicate id -> first valid-or-not
    occurrence wins; unknown id -> ignored. Unparseable output retries every id.
    """
    try:
        items = json.loads(raw)
    except (ValueError, TypeError):
        return _empty(), list(batch_ids)
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
        sub = it.get("sub_issue")
        path = normalize_path(
            taxonomy,
            it.get("product") or "",
            it.get("issue") or "",
            "(none)" if sub is None or str(sub).strip() == "" else sub,
        )
        conf = _confidence(it.get("confidence"))
        if path is None or conf is None:
            continue
        good.append(
            {
                "complaint_id": cid,
                "product": path[0],
                "issue": path[1],
                "sub_issue": path[2],
                "summary": str(it.get("summary") or "").strip(),
                "confidence": conf,
            }
        )
    ok = {g["complaint_id"] for g in good}
    retry = [str(i) for i in batch_ids if str(i) not in ok]
    return (pd.DataFrame(good, columns=_PARSED_COLS) if good else _empty()), retry


def _generate_with_backoff(
    client: GeminiClient,
    limiter: RateLimiter,
    sleep: Callable[[float], None],
    ledger: Ledger,
    model: str,
    est: tuple[int, int],
    args: tuple,
):
    """reserve -> limiter.wait -> generate, retrying non-daily 429s with exponential backoff."""
    for attempt in range(MAX_RATE_LIMIT_ATTEMPTS):
        ledger.reserve(model, *est)
        limiter.wait()
        try:
            return client.generate(model, *args)
        except RateLimited as e:
            if e.daily or attempt == MAX_RATE_LIMIT_ATTEMPTS - 1:
                raise
            sleep(max(e.retry_after or 0.0, 2.0 * 2**attempt))


def run_classification(
    rows: pd.DataFrame,
    cfg: Config,
    client: GeminiClient,
    ledger: Ledger,
    cache: JsonlCache,
    classifier_name: str = "gemini",
    model: str | None = None,
    prompt_version: str | None = None,
    thinking: str | None = None,
    *,
    taxonomy: pd.DataFrame,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Classify `rows` (complaint_id, narrative) sequentially; resumable via `cache`.

    Per request: reserve -> limiter.wait -> generate -> ledger.record -> cache.put per row.
    Returns every cached prediction for `rows`. On budget exhaustion stops and returns what
    is done; on a daily quota 429 raises DailyQuotaExhausted(remaining_rows).
    """
    model = model or cfg.classifier_model
    prompt_version = prompt_version or cfg.prompt_version
    thinking = thinking or cfg.thinking
    limiter = limiter or RateLimiter(cfg.rpm)
    tax_text = taxonomy_prompt_text(taxonomy)
    tax_hash = taxonomy_snapshot_hash(taxonomy)

    def key(cid: str) -> str:
        return f"{cid}|{classifier_name}|{model}|{prompt_version}"

    ids = [str(c) for c in rows["complaint_id"]]
    text = {
        str(c): clean_text(str(n), cfg.max_chars)
        for c, n in zip(rows["complaint_id"], rows["narrative"])
    }
    queue = [c for c in dict.fromkeys(ids) if cache.get(key(c)) is None]
    attempts: dict[str, int] = {}
    failed: set[str] = set()

    while queue:
        batch_ids, queue = queue[: cfg.batch_size], queue[cfg.batch_size :]
        batch = pd.DataFrame({"complaint_id": batch_ids, "narrative": [text[c] for c in batch_ids]})
        system, user = build_prompt(batch, tax_text, prompt_version)
        est = estimate_request_tokens(system, user, len(batch_ids), thinking)
        try:
            resp = _generate_with_backoff(
                client, limiter, sleep, ledger, model, est,
                (system, user, RESPONSE_SCHEMA, thinking, est[1]),
            )  # fmt: skip
        except BudgetExceeded as e:
            print(f"Budget cap reached ({e}); {len(batch_ids) + len(queue)} rows remaining.")
            break
        except RateLimited as e:
            if e.daily:  # everything finished so far is already cached
                raise DailyQuotaExhausted(len(batch_ids) + len(queue)) from e
            raise
        # Ledger first: a crash after a paid response must never lose the charge.
        # No usage metadata after a paid call: record the conservative estimate, never raise.
        in_tok = est[0] if resp.input_tokens is None else resp.input_tokens
        out_tok = est[1] if resp.output_tokens is None else resp.output_tokens
        ledger.record(model, uuid.uuid4().hex, in_tok, out_tok)
        parsed, retry = parse_and_validate(resp.text, batch_ids, taxonomy)
        run_ts = pd.Timestamp.now(tz="UTC").isoformat()
        for rec in parsed.to_dict("records"):
            cache.put(
                key(rec["complaint_id"]),
                {
                    **rec,
                    "classifier": classifier_name,
                    "model_id": model,
                    "prompt_version": prompt_version,
                    "taxonomy_hash": tax_hash,
                    "run_ts": run_ts,
                },
            )
        for cid in retry:
            attempts[cid] = attempts.get(cid, 0) + 1
            if attempts[cid] > MAX_RETRIES:
                failed.add(cid)
            else:
                queue.append(cid)

    if failed:
        print(f"{len(failed)} row(s) failed validation after {MAX_RETRIES} retries; left absent.")
    found = [v for c in dict.fromkeys(ids) if (v := cache.get(key(c))) is not None]
    return pd.DataFrame(found, columns=PREDICTIONS_COLS)
