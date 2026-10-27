"""Batch classification with Gemini: prompt, parsing/validation, resumable budgeted runner."""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable
from typing import TypeVar

import pandas as pd

from nlp.budget import BudgetExceeded, DailyQuotaExhausted, Ledger, RateLimiter
from nlp.config import Config
from nlp.data import clean_text
from nlp.llm.client import MAX_OUTPUT_TOKENS, GeminiClient, RateLimited, ServiceUnavailable
from nlp.schema import PREDICTIONS_COLS, split_label
from nlp.sheets import taxonomy_snapshot_hash
from nlp.store import JsonlCache
from nlp.taxonomy import normalize_label, taxonomy_prompt_text

T = TypeVar("T")
_STR = {"type": "string"}
RESPONSE_SCHEMA: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "complaint_id": _STR,
            "label": _STR,
            "summary": _STR,
            "confidence": {"type": "number"},
        },
        "required": ["complaint_id", "label", "summary", "confidence"],
    },
}

MAX_RETRIES = 2  # re-queues per row after the first attempt
MAX_RATE_LIMIT_ATTEMPTS = 5  # per request, for 429s and 5xx / timeouts alike
RESERVE_SAFETY = 1.5  # multiplier on the chars/4 input-token guess
_PARSED_COLS = [
    "complaint_id", "product", "issue", "sub_issue", "summary", "confidence"
]  # fmt: skip


def build_prompt(batch: pd.DataFrame, taxonomy_text: str, prompt_version: str) -> tuple[str, str]:
    """Return (system, user). `batch` needs complaint_id and (already cleaned) narrative.

    `taxonomy_text` is the category listing from `taxonomy_prompt_text` (one line per label
    with its description).
    """
    system = (
        f"[prompt {prompt_version}] You classify US consumer financial complaints.\n"
        "The CATEGORIES list below has one entry per category: a line \"- label: <label>\" "
        "(Product|Issue|Sub-issue) and an indented line \"description: ...\" saying what it "
        "covers and when to use it. Read the descriptions to tell similar categories apart.\n"
        "For each complaint choose exactly one label from the list and copy it verbatim: the "
        "whole text after \"- label: \" on its line, with the | separators exactly as written. "
        "Never invent or shorten a label.\n"
        "For each complaint also write a summary of at most 60 words that uses only facts "
        "stated in the complaint text (no guesses, no outside knowledge), and a confidence "
        "between 0 and 1 that the chosen label is correct.\n"
        "Return a JSON array with one object per complaint, following the response schema "
        "(complaint_id, label, summary, confidence). Use the given complaint_id values unchanged "
        "and include every complaint exactly once.\n\n"
        f"CATEGORIES:\n{taxonomy_text}"
    )
    parts = [
        f"complaint_id: {cid}\nnarrative: {text}"
        for cid, text in zip(batch["complaint_id"], batch["narrative"])
    ]
    user = "Classify these complaints.\n\n" + "\n\n---\n\n".join(parts)
    return system, user


def estimate_request_tokens(system: str, user: str, n_rows: int, thinking: str) -> tuple[int, int]:
    """Conservative (est_in, est_out) for Ledger.reserve (est_out is also max_output_tokens,
    clamped to the model's output limit so Google never rejects the request for size)."""
    allowance = {"none": 0, "minimal": 0, "low": 512}.get(thinking, 2048)
    est_in = int((len(system) + len(user)) / 4 * RESERVE_SAFETY)
    est_out = min(int(n_rows * (150 + allowance) * 1.5), MAX_OUTPUT_TOKENS)
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

    The answer's `label` must match a taxonomy label exactly or after normalization (case,
    whitespace, quotes); it is then split into product, issue and sub_issue. Invalid label or
    confidence, or missing id -> retry; duplicate id -> first valid-or-not
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
        label = normalize_label(taxonomy, it.get("label"))
        conf = _confidence(it.get("confidence"))
        if label is None or conf is None:
            continue
        path = split_label(label)
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
    """reserve -> limiter.wait -> generate, with bounded exponential backoff.

    Retried: non-daily 429s and ServiceUnavailable (5xx / timeouts; nothing is billed).
    """

    def attempt():
        ledger.reserve(model, *est)
        limiter.wait()
        return client.generate(model, *args)

    return with_backoff(attempt, sleep, "Gemini")


def with_backoff(call: Callable[[], T], sleep: Callable[[float], None], provider: str) -> T:
    """Run `call` up to MAX_RATE_LIMIT_ATTEMPTS times, sleeping max(retry_after, 2*2**n).

    A daily 429 is re-raised at once; a non-daily 429 after the last attempt is re-raised; a
    ServiceUnavailable after the last attempt is re-raised with a plain-English message.
    """
    for attempt in range(MAX_RATE_LIMIT_ATTEMPTS):
        try:
            return call()
        except RateLimited as e:
            if e.daily or attempt == MAX_RATE_LIMIT_ATTEMPTS - 1:
                raise
            sleep(max(e.retry_after or 0.0, 2.0 * 2**attempt))
        except ServiceUnavailable as e:
            if attempt == MAX_RATE_LIMIT_ATTEMPTS - 1:
                raise ServiceUnavailable(
                    f"{provider} is overloaded or not answering ({e}); gave up after "
                    f"{MAX_RATE_LIMIT_ATTEMPTS} attempts; nothing was charged for them."
                ) from e
            sleep(2.0 * 2**attempt)
    raise AssertionError("unreachable")


def run_queue(
    queue: list[str],
    *,
    batch_size: int,
    client: GeminiClient,
    ledger: Ledger,
    limiter: RateLimiter,
    sleep: Callable[[float], None],
    model: str,
    thinking: str,
    schema: dict,
    request: Callable[[list[str]], tuple[str, str]],
    handle: Callable[[str, list[str]], list[str]],
    tag: str | None = None,
) -> set[str]:
    """Shared sequential, spend-safe request loop (classification and judging).

    Per request: reserve -> limiter.wait -> generate(max_output_tokens=est_out) ->
    ledger.record (estimate if usage is missing) -> `handle` (parses, caches, returns ids to
    retry). Rows are re-queued up to MAX_RETRIES times; returns the ids that never validated.
    When the cap refuses a request, raises BudgetExceeded (naming the rows remaining) after
    every finished row is cached, so callers can never mistake a budget stop for a finished
    run; a daily quota 429 raises DailyQuotaExhausted.
    """
    attempts: dict[str, int] = {}
    failed: set[str] = set()
    while queue:
        batch_ids, queue = queue[:batch_size], queue[batch_size:]
        system, user = request(batch_ids)
        est = estimate_request_tokens(system, user, len(batch_ids), thinking)
        try:
            resp = _generate_with_backoff(
                client, limiter, sleep, ledger, model, est,
                (system, user, schema, thinking, est[1]),
            )  # fmt: skip
        except BudgetExceeded as e:  # everything finished so far is already cached
            raise BudgetExceeded(f"{e}; {len(batch_ids) + len(queue)} rows remaining") from e
        except RateLimited as e:
            if e.daily:  # everything finished so far is already cached
                raise DailyQuotaExhausted(len(batch_ids) + len(queue)) from e
            raise
        # Ledger first: a crash after a paid response must never lose the charge.
        # No usage metadata after a paid call: record the conservative estimate, never raise.
        in_tok = est[0] if resp.input_tokens is None else resp.input_tokens
        out_tok = est[1] if resp.output_tokens is None else resp.output_tokens
        ledger.record(model, uuid.uuid4().hex, in_tok, out_tok, tag=tag, rows=len(batch_ids))
        for cid in handle(resp.text, batch_ids):
            attempts[cid] = attempts.get(cid, 0) + 1
            if attempts[cid] > MAX_RETRIES:
                failed.add(cid)
            else:
                queue.append(cid)
    return failed


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
    descriptions: pd.DataFrame | None = None,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Classify `rows` (complaint_id, narrative) sequentially; resumable via `cache`.

    `descriptions` (label, description) default to the saved descriptions stage; without one,
    raises DescriptionsMissing ("Run Part 0 (descriptions) in notebook 03 first."). The
    recorded prompt_version (and cache key) is `<prompt_version>+<descriptions fingerprint>`.

    Per request: reserve -> limiter.wait -> generate -> ledger.record -> cache.put per row.
    Returns every cached prediction for `rows`. When the cap stops the run, raises
    BudgetExceeded (finished rows stay cached); on a daily quota 429 raises
    DailyQuotaExhausted(remaining_rows).
    """
    model = model or cfg.classifier_model
    prompt_version = prompt_version or cfg.prompt_version
    thinking = thinking or cfg.thinking
    limiter = limiter or RateLimiter(cfg.rpm)
    from nlp.llm.describe import load_descriptions, prompt_tag  # avoids an import cycle

    if descriptions is None:
        descriptions = load_descriptions(cfg, taxonomy)
    tax_text = taxonomy_prompt_text(taxonomy, descriptions)
    # The recorded version names the exact description wording, so answers given with older
    # (e.g. since edited) descriptions are never reused from the cache.
    prompt_version = prompt_tag(prompt_version, descriptions)
    tax_hash = taxonomy_snapshot_hash(taxonomy)

    def key(cid: str) -> str:
        return f"{cid}|{classifier_name}|{model}|{prompt_version}"

    ids = [str(c) for c in rows["complaint_id"]]
    text = {
        str(c): clean_text(str(n), cfg.max_chars)
        for c, n in zip(rows["complaint_id"], rows["narrative"])
    }
    queue = [c for c in dict.fromkeys(ids) if cache.get(key(c)) is None]

    def request(batch_ids: list[str]) -> tuple[str, str]:
        batch = pd.DataFrame({"complaint_id": batch_ids, "narrative": [text[c] for c in batch_ids]})
        return build_prompt(batch, tax_text, prompt_version)

    def handle(raw: str, batch_ids: list[str]) -> list[str]:
        parsed, retry = parse_and_validate(raw, batch_ids, taxonomy)
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
        return retry

    failed = run_queue(
        queue, batch_size=cfg.batch_size, client=client, ledger=ledger, limiter=limiter,
        sleep=sleep, model=model, thinking=thinking, schema=RESPONSE_SCHEMA,
        request=request, handle=handle, tag=prompt_version,
    )  # fmt: skip

    if failed:
        print(f"{len(failed)} row(s) failed validation after {MAX_RETRIES} retries; left absent.")
    found = [v for c in dict.fromkeys(ids) if (v := cache.get(key(c))) is not None]
    return pd.DataFrame(found, columns=PREDICTIONS_COLS)
