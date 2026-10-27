"""Category descriptions written by Gemini, one request per product.

Each taxonomy label (`Product|Issue|Sub-issue`) gets a short description built from up to 15
training complaints per label, so the classifier sees what every category means and how it
differs from its siblings. Money path identical to classification: reserve -> limiter.wait ->
generate -> ledger.record -> cache.put (one cache entry per product).
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable

import pandas as pd

from nlp import store
from nlp.budget import BudgetExceeded, DailyQuotaExhausted, Ledger, RateLimiter, estimate_run
from nlp.config import Config
from nlp.data import clean_text
from nlp.errors import StageIncomplete
from nlp.llm.classify import RESERVE_SAFETY, with_backoff
from nlp.llm.client import GeminiClient, LLMResponse, RateLimited
from nlp.schema import DESCRIPTIONS_COLS
from nlp.store import JsonlCache
from nlp.taxonomy import _norm

DESCRIBE_VERSION = "d1"
STAGE_DESCRIPTIONS = "descriptions"
EVIDENCE_PER_LABEL = 15
EVIDENCE_CHARS = 600
MAX_WORDS = 45
DESCRIBE_THINKING = "low"
MAX_ATTEMPTS = 2  # first try + one automatic retry per product
MISSING_MSG = "Run Part 0 (descriptions) in notebook 03 first."

_STR = {"type": "string"}
DESCRIBE_SCHEMA: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"label": _STR, "description": _STR},
        "required": ["label", "description"],
    },
}


class DescriptionsMissing(StageIncomplete):
    """No (or out-of-date) descriptions stage: classification cannot start."""


class DescriptionError(StageIncomplete):
    """Gemini's descriptions for a product were unusable twice; nothing was saved."""


# --------------------------------------------------------------------------- prompt


def evidence_by_label(
    complaints: pd.DataFrame, splits: pd.DataFrame, taxonomy: pd.DataFrame, cfg: Config
) -> dict[str, list[str]]:
    """label -> up to 15 cleaned training narratives (train split only, seeded, ~600 chars)."""
    train = set(splits.loc[splits["split"] == "train", "complaint_id"].astype(str))
    c = complaints[complaints["complaint_id"].astype(str).isin(train)]
    c = c.assign(_label=[f"{p}|{i}|{s}" for p, i, s in zip(c["product"], c["issue"], c["sub_issue"])])
    c = c.sort_values("complaint_id")
    out: dict[str, list[str]] = {x: [] for x in taxonomy["label"]}
    for label, g in c.groupby("_label", sort=True):
        if label not in out:
            continue
        pick = g.sample(n=min(EVIDENCE_PER_LABEL, len(g)), random_state=cfg.seed)
        out[label] = [clean_text(str(t), EVIDENCE_CHARS) for t in pick["narrative"]]
    return out


def build_describe_prompt(
    product: str, labels: list[str], evidence: dict[str, list[str]]
) -> tuple[str, str]:
    """(system, user) for one product: every sibling label with its example complaints."""
    system = (
        f"[describe {DESCRIBE_VERSION}] You write short category descriptions for a classifier "
        "of US consumer financial complaints sent to the CFPB. Each category is a label "
        '"Product|Issue|Sub-issue". You get every category of one product, each with example '
        "complaints the CFPB filed under it.\n"
        f"For each label write a description of at most {MAX_WORDS} words in this shape: what "
        'the category covers. "Use when ..." (the signs in a complaint that point to it). '
        "\"Not when ... (use '<closest sibling label>' instead).\" Add the \"Not when\" "
        "sentence only when a confusable sibling label exists in this list, and copy that "
        "sibling label exactly.\n"
        "Base each description only on the example complaints and the category names; add no "
        "outside knowledge and no details that are not in them.\n"
        "Return a JSON array of objects {label, description}, with every label below exactly "
        "once. Copy each label exactly as written."
    )
    blocks = [f"Product: {product}"]
    for label in labels:
        ex = evidence.get(label) or []
        lines = [f"{k}. {t}" for k, t in enumerate(ex, start=1)] or ["(no examples)"]
        blocks.append(f"LABEL: {label}\nExamples:\n" + "\n".join(lines))
    return system, "\n\n".join(blocks)


def estimate_describe_tokens(system: str, user: str, n_labels: int) -> tuple[int, int]:
    """Conservative (est_in, est_out); est_out is also max_output_tokens (generous, so the
    JSON is never cut off)."""
    est_in = int((len(system) + len(user)) / 4 * RESERVE_SAFETY)
    est_out = n_labels * 300 + 4096
    return est_in, est_out


def _requests(
    cfg: Config, complaints: pd.DataFrame, splits: pd.DataFrame, taxonomy: pd.DataFrame
) -> list[tuple[str, list[str], str, str]]:
    ev = evidence_by_label(complaints, splits, taxonomy, cfg)
    out = []
    for product, g in taxonomy.groupby("product", sort=True):
        labels = sorted(g["label"])
        system, user = build_describe_prompt(str(product), labels, ev)
        out.append((str(product), labels, system, user))
    return out


def estimate_descriptions(
    cfg: Config,
    complaints: pd.DataFrame,
    splits: pd.DataFrame,
    taxonomy: pd.DataFrame,
    model: str | None = None,
) -> float:
    """Worst-case USD for writing every description (the same estimate the spend guard uses)."""
    model = model or cfg.describe_model
    total = 0.0
    for _p, labels, system, user in _requests(cfg, complaints, splits, taxonomy):
        est_in, est_out = estimate_describe_tokens(system, user, len(labels))
        total += estimate_run(1, est_in, est_out, model, cfg.prices)
    return total


# --------------------------------------------------------------------------- validation


def validate_descriptions(raw: str, labels: list[str]) -> list[dict]:
    """[{label, description}] in `labels` order, or ValueError saying what is wrong.

    Labels are matched exactly, else after normalization (case, whitespace, quotes) to exactly
    one wanted label. Missing, unknown or duplicated labels and empty descriptions are errors.
    """
    try:
        items = json.loads(raw)
    except (ValueError, TypeError) as e:
        raise ValueError(f"answer is not valid JSON ({e})") from None
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        raise ValueError("answer is not a JSON list")  # noqa: TRY004 - a bad answer, not a bug
    by_norm: dict[str, list[str]] = {}
    for x in labels:
        by_norm.setdefault(_norm(x), []).append(x)
    got: dict[str, str] = {}
    unknown, dup, empty = [], [], []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("label") or "")
        hit = name if name in labels else None
        if hit is None and len(by_norm.get(_norm(name), [])) == 1:
            hit = by_norm[_norm(name)][0]
        if hit is None:
            unknown.append(name)
            continue
        if hit in got:
            dup.append(hit)
            continue
        text = " ".join(str(it.get("description") or "").split())
        if not text:
            empty.append(hit)
        got[hit] = text
    missing = [x for x in labels if x not in got]
    problems = []
    if missing:
        problems.append(f"missing labels {missing[:3]}")
    if unknown:
        problems.append(f"unknown labels {unknown[:3]}")
    if dup:
        problems.append(f"duplicate labels {dup[:3]}")
    if empty:
        problems.append(f"empty descriptions for {empty[:3]}")
    if problems:
        raise ValueError("; ".join(problems))
    return [{"label": x, "description": got[x]} for x in labels]


# --------------------------------------------------------------------------- runner


def _rows(items: list[dict], model: str, version: str = DESCRIBE_VERSION) -> list[dict]:
    return [{**it, "model_id": model, "describe_version": version} for it in items]


def run_descriptions(
    complaints: pd.DataFrame,
    splits: pd.DataFrame,
    taxonomy: pd.DataFrame,
    cfg: Config,
    client: GeminiClient,
    ledger: Ledger,
    cache: JsonlCache,
    *,
    model: str | None = None,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Write every label's description (stage shape DESCRIPTIONS_COLS); resumable via `cache`.

    One request per product. An unusable answer is retried once; if it is still unusable a
    DescriptionError names the product and nothing is returned (finished products stay cached,
    so a rerun only pays for the rest). BudgetExceeded / DailyQuotaExhausted /
    ServiceUnavailable / RequestRejected propagate as in classification.
    """
    model = model or cfg.describe_model
    limiter = limiter or RateLimiter(cfg.rpm)
    reqs = _requests(cfg, complaints, splits, taxonomy)
    rows: list[dict] = []
    for n, (product, labels, system, user) in enumerate(reqs):
        key = f"{product}|describe|{model}|{DESCRIBE_VERSION}"
        hit = cache.get(key)
        if hit is not None and sorted(it["label"] for it in hit["items"]) == labels:
            rows += _rows(hit["items"], model)
            continue
        est = estimate_describe_tokens(system, user, len(labels))
        problem = ""
        for _attempt in range(MAX_ATTEMPTS):
            resp = _paid_call(client, ledger, limiter, sleep, model, est, system, user,
                              remaining=len(reqs) - n)  # fmt: skip
            try:
                items = validate_descriptions(resp.text, labels)
            except ValueError as e:
                problem = str(e)
                continue
            cache.put(key, {"product": product, "model_id": model,
                            "describe_version": DESCRIBE_VERSION, "items": items})  # fmt: skip
            rows += _rows(items, model)
            break
        else:
            raise DescriptionError(
                f"Gemini's descriptions for the product '{product}' were unusable twice "
                f"({problem}). Nothing was saved for this step; products already finished are "
                "kept, so rerunning this cell only pays for the rest. If it fails again, send "
                "this message to Claude."
            )
    return pd.DataFrame(rows, columns=DESCRIPTIONS_COLS)


def _paid_call(client, ledger, limiter, sleep, model, est, system, user, remaining) -> LLMResponse:
    """reserve -> limiter.wait -> generate (with backoff) -> ledger.record."""

    def attempt():
        ledger.reserve(model, *est)
        limiter.wait()
        return client.generate(model, system, user, DESCRIBE_SCHEMA, DESCRIBE_THINKING, est[1])

    try:
        resp = with_backoff(attempt, sleep, "Gemini")
    except BudgetExceeded as e:
        raise BudgetExceeded(f"{e}; descriptions for {remaining} product(s) still to write") from e
    except RateLimited as e:
        if e.daily:
            raise DailyQuotaExhausted(remaining) from e
        raise
    # Ledger first: a crash after a paid response must never lose the charge.
    in_tok = est[0] if resp.input_tokens is None else resp.input_tokens
    out_tok = est[1] if resp.output_tokens is None else resp.output_tokens
    ledger.record(model, uuid.uuid4().hex, in_tok, out_tok)
    return resp


# --------------------------------------------------------------------------- stage helpers


def descriptions_match(descriptions: pd.DataFrame, taxonomy: pd.DataFrame) -> bool:
    """True if there is exactly one non-empty description per taxonomy label."""
    labels = descriptions["label"].astype(str)
    return (
        sorted(labels) == sorted(taxonomy["label"].astype(str))
        and labels.is_unique
        and bool(descriptions["description"].astype(str).str.strip().ne("").all())
    )


def descriptions_fingerprint(descriptions: pd.DataFrame) -> str:
    """8 hex chars identifying the exact description wording (order independent)."""
    lines = sorted(f"{x}\t{d}" for x, d in zip(descriptions["label"], descriptions["description"]))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:8]


def load_descriptions(cfg: Config, taxonomy: pd.DataFrame) -> pd.DataFrame:
    """The saved descriptions stage, or DescriptionsMissing if absent or out of date."""
    if not store.exists(cfg, STAGE_DESCRIPTIONS):
        raise DescriptionsMissing(f"No category descriptions yet. {MISSING_MSG}")
    d = store.read(cfg, STAGE_DESCRIPTIONS)
    if not descriptions_match(d, taxonomy):
        raise DescriptionsMissing(
            f"The saved category descriptions do not match the current category list. {MISSING_MSG}"
        )
    return d


def apply_edits(current: pd.DataFrame, edits: pd.DataFrame) -> pd.DataFrame:
    """Descriptions with Marco's Sheet edits applied (labels must be unchanged, none empty)."""
    want = set(current["label"])
    got = list(edits["label"].astype(str).str.strip())
    missing, extra = sorted(want - set(got)), sorted(set(got) - want)
    dups = sorted({x for x in got if got.count(x) > 1})
    if missing or extra or dups:
        raise StageIncomplete(
            "The label column in the taxonomy Sheet was changed "
            f"(missing: {missing[:3]}, unknown: {extra[:3]}, repeated: {dups[:3]}). Labels must "
            "stay exactly as written - only edit the description column. Fix the Sheet (undo "
            "the label change) and rerun this cell. Nothing was saved."
        )
    new = dict(zip(got, (" ".join(str(t).split()) for t in edits["description"])))
    empty = sorted(x for x, t in new.items() if not t)
    if empty:
        raise StageIncomplete(
            f"{len(empty)} description(s) are empty in the taxonomy Sheet, e.g. {empty[:3]}. "
            "Write a description for each and rerun this cell. Nothing was saved."
        )
    out = current.copy()
    changed = [new[x] != d for x, d in zip(out["label"], out["description"])]
    out["description"] = [new[x] for x in out["label"]]
    out.loc[changed, "describe_version"] = [f"{v}-edited" if not str(v).endswith("-edited") else v
                                            for v in out.loc[changed, "describe_version"]]
    return out[DESCRIPTIONS_COLS]
