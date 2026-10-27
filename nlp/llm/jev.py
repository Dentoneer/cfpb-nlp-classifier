"""Jev (TypeSafe AI) validator: calibrated yes/no checks of Gemini's Product / Issue / Sub-issue.

Jev answers typed questions about a "state" (here: one complaint narrative) with a probability.
One request per complaint, three `noul` questions. Spend goes into the same ledger as Gemini
(input tokens only; output is free). The runner mirrors `nlp.llm.classify.run_queue`:
reserve -> limiter.wait -> system_one -> ledger.record -> cache.put, with 429 backoff.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from nlp.budget import BudgetExceeded, DailyQuotaExhausted, Ledger, RateLimiter
from nlp.config import Config
from nlp.data import clean_text
from nlp.llm.classify import with_backoff
from nlp.llm.client import RateLimited, ServiceUnavailable
from nlp.schema import VALIDATOR_RESULTS_COLS
from nlp.store import JsonlCache

JEV_PROMPT_VERSION = "jev-v2"
JEV_TAG = f"jev:{JEV_PROMPT_VERSION}"  # ledger tag of Jev requests
RESERVE_SAFETY = 1.5  # multiplier on the input-token guess, as for Gemini
JEV_FIELDS = ["product_ok", "issue_ok", "sub_issue_ok"]
_LEVEL_OF = {"product_ok": "product", "issue_ok": "issue", "sub_issue_ok": "sub_issue"}
_PLACEHOLDER = "x" * 40  # label stand-in for cost estimates before Gemini has labelled


@dataclass
class JevResponse:
    nouls: dict[str, float]  # question id -> P(yes) in [0, 1]
    input_tokens: int | None  # None -> no usage reported; runner records the estimate
    output_tokens: int | None
    model: str


class JevError(Exception):
    """Non-429 HTTP error or an unreadable response from Jev."""


class JevParseError(JevError):
    """A (possibly paid) 200 response that could not be read; carries any usage it reported."""

    def __init__(self, message: str, input_tokens: int | None = None):
        super().__init__(message)
        self.input_tokens = input_tokens


class JevClient(Protocol):
    def system_one(self, model: str, state: str, questions: dict[str, dict]) -> JevResponse: ...


def _noul_value(node: object) -> float | None:
    if isinstance(node, dict):
        node = node.get("noul")
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return None
    f = float(node)
    return f if 0.0 <= f <= 1.0 else None


def parse_jev_response(body: object, question_ids: list[str], model: str = "") -> JevResponse:
    """Extract {id: noul} from a Jev JSON body.

    The exact nesting is not confirmed by the docs, so accept the answers under a top-level
    "answers" or "nouls" map, or keyed by question id at the top level; each answer is either
    {"type": "noul", "noul": p, ...} or a bare number. Raises JevError naming the keys found.
    """
    if not isinstance(body, dict):
        raise JevParseError(f"Jev response is not a JSON object: {str(body)[:200]!r}")
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    found: dict[str, float] = {}
    for cont in (body.get("answers"), body.get("nouls"), body):
        if not isinstance(cont, dict):
            continue
        vals = {q: _noul_value(cont.get(q)) for q in question_ids}
        if all(v is not None for v in vals.values()):
            found = {q: float(v) for q, v in vals.items()}  # type: ignore[arg-type]
            break
    if not found:
        keys = {"top-level": sorted(body)}
        for k in ("answers", "nouls"):
            if isinstance(body.get(k), dict):
                keys[k] = sorted(body[k])
        raise JevParseError(
            f"could not find a noul in [0, 1] for each of {question_ids} in the Jev response; "
            f"keys found: {keys}",
            usage.get("input_tokens"),
        )
    return JevResponse(
        found,
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        str(body.get("model") or model),
    )


class HttpJevClient:
    """Raw httpx client for POST {base_url}/v1/systemone (no typesafe-sdk dependency)."""

    def __init__(
        self, api_key: str | None, base_url: str = "https://api.typesafe.ai", timeout: float = 60
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def system_one(self, model: str, state: str, questions: dict[str, dict]) -> JevResponse:
        import httpx

        if not self.api_key:
            raise JevError(
                "No Jev API key: add a TYPESAFE_API_KEY secret in Colab (key icon, Notebook "
                "access on), restart the session and rerun the setup cell."
            )
        try:
            r = httpx.post(
                f"{self.base_url}/v1/systemone",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "state": state, "questions": questions},
                timeout=self.timeout,
            )
        except httpx.TimeoutException as e:  # retried by the runner (bounded)
            raise ServiceUnavailable(f"Jev request timed out: {e}") from e
        except httpx.HTTPError as e:
            raise JevError(f"Jev request failed: {e}") from e
        if r.status_code == 429:
            raw = r.headers.get("retry-after")
            try:
                retry_after = float(raw) if raw is not None else None
            except ValueError:
                retry_after = None
            raise RateLimited(retry_after, daily=False)
        if r.status_code >= 500:  # overloaded / down: retried by the runner (bounded)
            raise ServiceUnavailable(f"Jev HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
            raise JevError(f"Jev HTTP {r.status_code}: {r.text}")
        try:
            body = r.json()
        except ValueError as e:
            raise JevParseError(f"Jev returned non-JSON: {r.text[:200]!r}") from e
        return parse_jev_response(body, list(questions), model)


class FakeJevClient:
    """Test double: `responder(model, state, questions) -> JevResponse`; records every call."""

    def __init__(self, responder: Callable[[str, str, dict], JevResponse]):
        self.responder = responder
        self.calls: list[tuple] = []

    def system_one(self, model: str, state: str, questions: dict[str, dict]) -> JevResponse:
        self.calls.append((model, state, questions))
        return self.responder(model, state, questions)


def build_jev_questions(product: str, issue: str, sub_issue: str) -> dict[str, dict]:
    """Testable claims about the complaint, one per hierarchy level.

    Every category has a real sub-issue, so all three claims are always asked. Sub-issue
    claims name the issue so they stand alone.
    """
    claims = {
        "product_ok": f'This complaint is about the financial product: "{product}".',
        "issue_ok": f'The main problem the consumer reports is: "{issue}".',
        "sub_issue_ok": (
            f'Within the issue "{issue}", this complaint is specifically about: "{sub_issue}".'
        ),
    }
    return {k: {"type": "noul", "instructions": v} for k, v in claims.items()}


def estimate_jev_tokens(state: str, questions: dict[str, dict]) -> int:
    """Conservative input-token estimate (chars/4 x RESERVE_SAFETY)."""
    return int(len(json.dumps({"state": state, "questions": questions})) / 4 * RESERVE_SAFETY)


def estimate_jev_usd(cfg: Config, rows: pd.DataFrame, model: str | None = None) -> float:
    """Worst-case USD for one request per row (labels unknown yet, so placeholders)."""
    model = model or cfg.jev_model
    q = build_jev_questions(_PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER)
    p_in, _ = cfg.prices[model]
    toks = sum(estimate_jev_tokens(clean_text(str(t), cfg.max_chars), q) for t in rows["narrative"])
    return toks * p_in / 1_000_000


def _call_with_backoff(client, limiter, sleep, ledger, model, est_in, state, questions):
    """reserve -> limiter.wait -> system_one; bounded backoff on 429s, 5xx and timeouts."""

    def attempt():
        ledger.reserve(model, est_in, 0)
        limiter.wait()
        return client.system_one(model, state, questions)

    return with_backoff(attempt, sleep, "Jev")


def _tokens(reported: object, estimate: int) -> int:
    """Reported token count if it is a usable number, else the reserve estimate."""
    if isinstance(reported, bool) or not isinstance(reported, (int, float)):
        return estimate
    if not math.isfinite(reported) or reported < 0:
        return estimate
    return int(reported)


def run_jev(
    rows: pd.DataFrame,
    cfg: Config,
    client: JevClient,
    ledger: Ledger,
    cache: JsonlCache,
    *,
    taxonomy: pd.DataFrame,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Ask Jev about each row (complaint_id, narrative, product, issue, sub_issue); resumable.

    Returns jev_results: complaint_id, product_ok, issue_ok, sub_issue_ok (noul P(yes); NaN if
    not asked),
    model_id - for every row with a fresh cached answer. A cached answer is stale (a miss)
    if the judged labels differ from the row's current labels. When the cap refuses a request,
    raises BudgetExceeded (finished rows stay cached); a daily-quota 429 raises
    DailyQuotaExhausted(remaining_rows).
    `taxonomy` is kept for interface stability (every category has a sub-issue now).
    """
    model = cfg.jev_model
    limiter = limiter or RateLimiter(cfg.rpm)

    def key(cid: str) -> str:
        return f"{cid}|jev|{model}|{JEV_PROMPT_VERSION}"

    by_id: dict[str, pd.Series] = {}
    for _, r in rows.iterrows():
        by_id.setdefault(str(r["complaint_id"]), r)
    ids = list(by_id)

    def seen(cid: str) -> dict:
        return {f"in_{c}": str(by_id[cid][c]) for c in ("product", "issue", "sub_issue")}

    def fresh(cid: str) -> dict | None:
        rec = cache.get(key(cid))
        if rec is None or any(rec.get(k) != v for k, v in seen(cid).items()):
            return None
        return rec

    queue = [c for c in ids if fresh(c) is None]
    for i, cid in enumerate(queue):
        r = by_id[cid]
        state = clean_text(str(r["narrative"]), cfg.max_chars)
        questions = build_jev_questions(r["product"], r["issue"], r["sub_issue"])
        est_in = estimate_jev_tokens(state, questions)
        try:
            resp = _call_with_backoff(
                client, limiter, sleep, ledger, model, est_in, state, questions
            )
        except BudgetExceeded as e:  # everything finished so far is already cached
            raise BudgetExceeded(f"{e}; {len(queue) - i} rows remaining") from e
        except RateLimited as e:
            if e.daily:  # everything finished so far is already cached
                raise DailyQuotaExhausted(len(queue) - i) from e
            raise
        except JevParseError as e:  # the call was paid even though we cannot read it
            ledger.record(model, uuid.uuid4().hex, _tokens(e.input_tokens, est_in), 0,
                          tag=JEV_TAG, rows=1)
            raise
        # Ledger first: a crash after a paid response must never lose the charge.
        in_tok = _tokens(resp.input_tokens, est_in)  # odd usage never blocks recording
        ledger.record(model, uuid.uuid4().hex, in_tok, 0, tag=JEV_TAG, rows=1)  # output is free
        missing = [f for f in questions if f not in resp.nouls]
        if missing:
            raise JevError(f"Jev response for {cid} lacks answers {missing}: {sorted(resp.nouls)}")
        cache.put(
            key(cid),
            {
                "complaint_id": cid,
                **{f: (float(resp.nouls[f]) if f in questions else None) for f in JEV_FIELDS},
                "model_id": model,
                **seen(cid),
            },
        )
    cols = ["complaint_id", *JEV_FIELDS, "model_id"]
    df = pd.DataFrame(
        [{c: v[c] for c in cols} for k in ids if (v := fresh(k)) is not None], columns=cols
    )
    return df.astype({f: float for f in JEV_FIELDS})  # None -> NaN


def v_jev(jev_results: pd.DataFrame) -> pd.DataFrame:
    """validator_results rows (validator `jev`): score = noul, verdict = score >= 0.5.

    Levels product / issue / sub_issue plus path (score = product of the noul values asked;
    a level that was not asked (NaN) emits no row and is left out of the path product).
    """
    out = []
    for r in jev_results.itertuples(index=False):
        scores = {
            lv: float(getattr(r, f))
            for f, lv in _LEVEL_OF.items()
            if not pd.isna(getattr(r, f))
        }
        scores["path"] = math.prod(scores.values())
        for level, s in scores.items():
            out.append((r.complaint_id, "jev", level, s >= 0.5, s))
    return pd.DataFrame(out, columns=VALIDATOR_RESULTS_COLS)
