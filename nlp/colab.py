"""Glue for the Colab notebooks: setup, spend-safe staging, estimates, result summaries.

Notebooks stay thin: they call these helpers plus the `nlp.pipeline` functions. Nothing here
needs Colab except `setup()` (its Colab/gspread imports are lazy), so the rest is unit-tested.
"""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from nlp import pipeline as P
from nlp import store
from nlp.budget import BudgetExceeded, DailyQuotaExhausted, Ledger, RateLimiter, estimate_run
from nlp.config import Config
from nlp.data import clean_text, load_clean
from nlp.errors import StageIncomplete
from nlp.llm.classify import build_prompt, estimate_request_tokens, run_classification
from nlp.llm.client import GeminiClient, RequestRejected, ServiceUnavailable
from nlp.llm.describe import (
    STAGE_DESCRIPTIONS,
    DescriptionError,
    apply_edits,
    descriptions_match,
    load_descriptions,
    prompt_tag,
    run_descriptions,
)
from nlp.llm.jev import estimate_jev_usd
from nlp.llm.judge import SUMMARY_JUDGE_PROMPT_VERSION, estimate_summary_judge_usd
from nlp.review import count_verdicts
from nlp.sheets import (
    publish_taxonomy,
    read_description_edits,
    read_review_sheet,
    rewrite_taxonomy_sheet,
)
from nlp.store import JsonlCache
from nlp.taxonomy import taxonomy_prompt_text

__all__ = ["StageIncomplete"]  # defined in nlp.errors; notebooks use colab.StageIncomplete

REPO_URL = "https://github.com/Dentoneer/cfpb-nlp-classifier"
OVERRIDES_FILE = "config_overrides.json"
LEVELS4 = ["product", "issue", "sub_issue", "path"]
STAGE_FILTER_REPORT = "filter_report"
ALLOWED_OVERRIDES = ("classifier_model", "judge_model", "batch_size", "thinking")
MAX_MISSING_FRAC = 0.02  # stage_paid saves a stage missing at most this share of rows
DUCKDB_TMP = "/content/duckdb_tmp"  # local Colab disk DuckDB may spill to while filtering
MEASURED_SAFETY = 1.5
RESTART_MSG = "Restart session (Runtime menu \u2192 Restart session), then rerun this cell"
# Stand-in for a not-yet-written description when estimating cost (a 45-word description is
# about 300 characters; rounded up so the estimate stays conservative).
PLACEHOLDER_DESCRIPTION = "x" * 400


# --------------------------------------------------------------------------- config / setup


def read_overrides(artifacts_dir: Path) -> dict:
    """Saved config choices (only the allowed keys); unknown keys are an error."""
    f = Path(artifacts_dir) / OVERRIDES_FILE
    if not f.exists():
        return {}
    saved = json.loads(f.read_text(encoding="utf-8"))
    bad = set(saved) - set(ALLOWED_OVERRIDES)
    if bad:
        raise ValueError(f"{f} has settings that cannot be overridden: {sorted(bad)}")
    return saved


def load_config(config_path: Path, artifacts_dir: Path) -> Config:
    """Config from YAML with artifacts_dir replaced, then the saved choices applied."""
    cfg = dataclasses.replace(Config.load(Path(config_path)), artifacts_dir=Path(artifacts_dir))
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    return dataclasses.replace(cfg, **read_overrides(artifacts_dir))


def save_overrides(cfg: Config, **changes) -> Config:
    """Persist config choices (e.g. the bake-off winner) on Drive; every notebook applies them."""
    bad = set(changes) - set(ALLOWED_OVERRIDES)
    if bad:
        raise ValueError(
            f"not an overridable setting: {sorted(bad)}; allowed: {list(ALLOWED_OVERRIDES)}"
        )
    new = dataclasses.replace(cfg, **changes)
    for field in ("classifier_model", "judge_model"):
        if getattr(new, field) not in (cfg.prices or {}):
            raise ValueError(f"{field} {getattr(new, field)!r} has no price in config 'prices'")
    if new.classifier_model == new.judge_model:
        raise ValueError("judge model must differ from the classifier model")
    f = Path(cfg.artifacts_dir) / OVERRIDES_FILE
    merged = {**read_overrides(cfg.artifacts_dir), **changes}
    f.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return new


def apply_choices(cfg: Config, require_pick: bool = False, **choices) -> Config:
    """Save only the choices that are set (not None) and differ from the current config.

    With require_pick, a missing classifier_model is an error unless one was saved earlier.
    """
    if (
        require_pick
        and choices.get("classifier_model") is None
        and "classifier_model" not in read_overrides(cfg.artifacts_dir)
    ):
        raise ValueError("Set PICK to your chosen model (e.g. 'gemini-3.8-flash') and rerun.")
    changes = {k: v for k, v in choices.items() if v is not None and v != getattr(cfg, k, None)}
    if changes:
        cfg = save_overrides(cfg, **changes)
    print(
        f"classifier {cfg.classifier_model} | judge {cfg.judge_model} | "
        f"batch_size {cfg.batch_size} | thinking {cfg.thinking}"
    )
    return cfg


def _git_head(repo_dir: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=20,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


_HEAD_AT_IMPORT = _git_head(Path(__file__).resolve().parent.parent)


def needs_restart(head_at_import: str | None, head_now: str | None) -> bool:
    """True if the repo moved (git pull) after nlp was imported: loaded modules are stale."""
    return bool(head_at_import and head_now and head_at_import != head_now)


def check_code_fresh(head_at_import: str | None, head_now: str | None) -> None:
    if needs_restart(head_at_import, head_now):
        raise RuntimeError(RESTART_MSG)


def _secret(userdata, name: str) -> str | None:
    """The secret's value with surrounding spaces/newlines removed (None if blank)."""
    value = userdata.get(name)
    value = "" if value is None else str(value).strip()
    return value or None


def read_secrets(userdata) -> tuple[str, str | None]:
    """(Gemini key, Jev key or None). Gemini's secret is required; the Jev one is optional.

    Both are stripped: a space or line break pasted along with a key makes Google reject it.
    """
    missing = (
        "GEMINI_API_KEY not found. In Colab open the key icon (Secrets) on the left, add "
        "GEMINI_API_KEY and switch 'Notebook access' on."
    )
    try:
        api_key = _secret(userdata, "GEMINI_API_KEY")
    except Exception as e:  # secret missing or notebook access not granted
        raise RuntimeError(missing) from e
    if api_key is None:
        raise RuntimeError(missing + " (The secret is there but its value is empty.)")
    try:
        jev_key = _secret(userdata, "TYPESAFE_API_KEY")
    except Exception:  # noqa: BLE001 - Colab raises various errors for a missing secret
        jev_key = None
    if jev_key is None:
        print(
            "Note: no TYPESAFE_API_KEY secret - that is fine now; it is needed only for "
            "notebook 04's Jev step."
        )
    return api_key, jev_key


def _genai_client(api_key: str):
    from google import genai

    return genai.Client(api_key=api_key)


def _key_facts(api_key: str) -> str:
    """Length, first 3 characters and whether it has stray spaces - never the key itself."""
    raw = api_key or ""
    ws = "yes" if raw != raw.strip() or any(c.isspace() for c in raw) else "no"
    return f"length {len(raw.strip())}, starts with '{raw.strip()[:3]}', whitespace: {ws}"


def check_gemini_key(
    api_key: str, make_client: Callable | None = None, stop: bool = True
) -> bool:
    """Free check that Google accepts the Gemini key (lists models; nothing is generated).

    Prints the result. On failure raises StageIncomplete with what to do (README 'Key
    problems'), so the notebook stops there; with stop=False it prints that and returns False.
    """
    key = (api_key or "").strip()
    note = (
        " (Keys starting 'AQ.' are the new AI Studio format - that is normal.)"
        if key.startswith("AQ.")
        else ""
    )
    try:
        client = (make_client or _genai_client)(key)
        n = sum(1 for _ in client.models.list())
    except Exception as e:  # noqa: BLE001 - any failure here means the key does not work
        from nlp.llm.client import explain_rejection

        why = explain_rejection(e) if hasattr(e, "code") else f"{type(e).__name__}: {e}"
        msg = (
            f"PROBLEM: the Gemini key check failed: {why}\n"
            "What to do: open README.md, section 'What to do if a notebook stops' -> 'Key "
            "problems'. In short: in Colab Secrets (key icon) delete GEMINI_API_KEY, paste the "
            "key again from aistudio.google.com/apikey (no spaces), switch Notebook access on, "
            "then Runtime -> Restart session and rerun from the top."
        )
        if stop:
            raise StageIncomplete(msg) from None
        print(msg)
        return False
    print(f"Gemini key works ({n} models visible; {_key_facts(api_key)}).{note}")
    return True


DATA_VERSION_FILE = "data_version.txt"
# Results that belong to one data version. Never moved: ledger.jsonl (the spend cap must keep
# counting old spend), the Gemini / judge caches and config_overrides.json.
ARCHIVE_NAMES = ("models", "taxonomy_url.txt", "review_sheet_url.txt", "results.md")


def _free_name(dest: Path) -> Path:
    """`dest`, or `dest` with a numeric suffix if something already has that name."""
    if not dest.exists():
        return dest
    k = 2
    while (cand := dest.with_name(f"{dest.stem}_{k}{dest.suffix}")).exists():
        k += 1
    return cand


def archive_old_results(artifacts_dir: Path, data_version: str) -> list[str]:
    """Move results made with another data version into `archive_<old version>/`.

    Compares `data_version` with `data_version.txt` in the artifacts dir (missing = "v1").
    When they differ, moves every *.parquet plus models/, the Sheet URL files and results.md,
    then writes the new marker and prints what moved. Idempotent: once the marker matches,
    nothing is touched, so new results are never archived. Returns the moved names.
    """
    import shutil

    art = Path(artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    marker = art / DATA_VERSION_FILE
    old = marker.read_text(encoding="utf-8").strip() if marker.exists() else "v1"
    if marker.exists() and old == data_version:
        return []
    items = sorted(art.glob("*.parquet")) + [art / n for n in ARCHIVE_NAMES if (art / n).exists()]
    moved: list[str] = []
    if items and old != data_version:
        dest_dir = art / f"archive_{old}"
        dest_dir.mkdir(exist_ok=True)
        for p in items:
            shutil.move(str(p), str(_free_name(dest_dir / p.name)))
            moved.append(p.name)
        print(
            f"Data version changed ({old} -> {data_version}): moved {len(moved)} old result(s) "
            f"into {dest_dir.name}/: {', '.join(moved)}. Kept in place: ledger (spend so far), "
            "saved Gemini answers and your saved choices. Nothing to do - just carry on."
        )
    marker.write_text(data_version + "\n", encoding="utf-8")
    return moved


def setup(
    repo_dir: str = "/content/cfpb-nlp-classifier",
    drive_subdir: str = "cfpb-nlp-classifier/phase1",
    drive_root: str = "/content/drive",
):
    """Mount Drive, archive results of an older data version, load config, fetch the Gemini
    and Jev keys, authorise Sheets.

    Returns (cfg, gc, api_key, jev_key); jev_key is None if the optional TYPESAFE_API_KEY
    secret is not set (only notebook 04's Jev step needs it).
    """
    check_code_fresh(_HEAD_AT_IMPORT, _git_head(Path(repo_dir)))
    from google.colab import auth, drive, userdata

    drive.mount(drive_root)
    artifacts = Path(drive_root) / "MyDrive" / drive_subdir
    cfg = load_config(Path(repo_dir) / "config" / "phase1.yaml", artifacts)
    archive_old_results(artifacts, cfg.data_version)
    api_key, jev_key = read_secrets(userdata)
    auth.authenticate_user()
    import google.auth
    import gspread

    creds, _ = google.auth.default()
    gc = gspread.authorize(creds)
    print(f"Ready. Outputs go to {artifacts}")
    print(f"Saved choices applied: {read_overrides(artifacts) or 'none yet'}")
    print(f"Classifier {cfg.classifier_model} | judge {cfg.judge_model} | cap ${cfg.spend_cap_usd:.2f}")
    return cfg, gc, api_key, jev_key


def open_tools(cfg: Config) -> tuple[Ledger, JsonlCache, JsonlCache, RateLimiter]:
    """(ledger, gemini cache, judge cache, rate limiter) in the shared artifacts dir."""
    art = Path(cfg.artifacts_dir)
    ledger = Ledger(art / "ledger.jsonl", cfg.spend_cap_usd, cfg.prices)
    return (
        ledger,
        JsonlCache(art / "gemini_cache.jsonl"),
        JsonlCache(art / "judge_cache.jsonl"),
        RateLimiter(cfg.rpm),
    )


# --------------------------------------------------------------------------- staging


def stage_paid(
    cfg: Config,
    name: str,
    fn: Callable[[], pd.DataFrame],
    expected_ids: Iterable[str],
) -> pd.DataFrame:
    """`store.stage` for paid (Gemini / Jev) steps: saves only a (nearly) complete result.

    A run stopped by the budget cap (the runners raise BudgetExceeded), the daily quota or a
    provider outage is never saved, nor is a result missing more than 2% of rows - rerun
    later. Up to 2% missing for other reasons (rows Gemini keeps answering invalidly) is saved
    and the missing ids are printed.

    Paid responses are already in the JSONL cache, so rerunning the cell later only pays for
    the rows still missing.
    """
    if store.exists(cfg, name):
        return store.read(cfg, name)
    want = {str(i) for i in expected_ids}
    try:
        df = fn()
    except DailyQuotaExhausted as e:
        raise StageIncomplete(
            f"Gemini's daily request quota is used up ({e.remaining_rows} rows still to do). "
            "Everything finished is saved - rerun this cell tomorrow."
        ) from None
    except BudgetExceeded as e:
        raise StageIncomplete(
            f"Budget reached. Nothing more was spent. ({e}) Finished rows are saved."
        ) from None
    except ServiceUnavailable as e:
        raise StageIncomplete(f"{e} Finished rows are saved - rerun this cell later.") from None
    except RequestRejected as e:
        raise StageIncomplete(
            f"Google refused the request: {e.message} Nothing was charged for it. Finished "
            "rows are saved."
        ) from None
    got = set(df["complaint_id"].astype(str)) if len(df) else set()
    missing = sorted(want - got)
    if missing:
        frac = len(missing) / len(want)
        if frac > MAX_MISSING_FRAC:
            raise StageIncomplete(
                f"{len(want) - len(missing)} of {len(want)} complaints done for '{name}': "
                "many rows failed validation (see messages above). Finished rows are saved; "
                "rerun this cell to retry the rest."
            )
        print(
            f"{len(missing)} of {len(want)} complaints ({frac:.1%}) got no valid answer after "
            f"retries and are left out of '{name}': {missing[:50]}"
        )
    store.write(cfg, name, df)
    return df


def always_write(cfg: Config) -> Callable[[str, Callable[[], pd.DataFrame]], pd.DataFrame]:
    """Like `store.stage` but always recomputes and overwrites (for results that can change)."""

    def run(name: str, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        df = fn()
        store.write(cfg, name, df)
        return df

    return run


def publish_once(cfg: Config, name: str, create: Callable[[], str]) -> str:
    """Create a Google Sheet once; later runs reuse the saved URL instead of making duplicates."""
    f = Path(cfg.artifacts_dir) / f"{name}_url.txt"
    if f.exists():
        return f.read_text(encoding="utf-8").strip()
    url = create()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(url, encoding="utf-8")
    return url


def read_url(cfg: Config, name: str) -> str:
    f = Path(cfg.artifacts_dir) / f"{name}_url.txt"
    if not f.exists():
        raise StageIncomplete(f"No saved '{name}' sheet URL yet - run the earlier notebook first.")
    return f.read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------- money


def estimate_classification(
    cfg: Config,
    rows: pd.DataFrame,
    taxonomy: pd.DataFrame,
    *,
    model: str | None = None,
    batch_size: int | None = None,
    thinking: str | None = None,
    runs: int = 1,
    cache: JsonlCache | None = None,
    classifier_name: str | None = None,
) -> float:
    """Conservative USD estimate (the same token estimate the spend guard reserves with).

    Uses the saved category descriptions when they exist; before Part 0 has run, a
    conservative stand-in description per category. With `cache` and `classifier_name`, rows
    already answered for this run are left out, so a fully cached rerun estimates $0.
    """
    model = model or cfg.classifier_model
    bs = batch_size or cfg.batch_size
    thinking = thinking or cfg.thinking
    desc = _descriptions_for_estimate(cfg, taxonomy)
    if cache is not None and classifier_name is not None:
        rows = uncached_rows(cfg, rows, taxonomy, cache, classifier_name, model)
    tax = taxonomy_prompt_text(taxonomy, desc)
    text = [clean_text(str(t), cfg.max_chars) for t in rows["narrative"]]
    ids = [str(c) for c in rows["complaint_id"]]
    total = 0.0
    for i in range(0, len(ids), bs):
        batch = pd.DataFrame({"complaint_id": ids[i : i + bs], "narrative": text[i : i + bs]})
        system, user = build_prompt(batch, tax, cfg.prompt_version)
        est_in, est_out = estimate_request_tokens(system, user, len(batch), thinking)
        total += estimate_run(1, est_in, est_out, model, cfg.prices)
    return runs * total


def _descriptions_for_estimate(cfg: Config, taxonomy: pd.DataFrame):
    if store.exists(cfg, STAGE_DESCRIPTIONS):
        d = store.read(cfg, STAGE_DESCRIPTIONS)
        if descriptions_match(d, taxonomy):
            return d
    return {label: PLACEHOLDER_DESCRIPTION for label in taxonomy["label"]}


def current_prompt_tag(cfg: Config, taxonomy: pd.DataFrame) -> str | None:
    """The recorded prompt version classification uses now (None before Part 0)."""
    try:
        return prompt_tag(cfg.prompt_version, load_descriptions(cfg, taxonomy))
    except StageIncomplete:
        return None


def uncached_rows(
    cfg: Config,
    rows: pd.DataFrame,
    taxonomy: pd.DataFrame,
    cache: JsonlCache,
    classifier_name: str,
    model: str | None = None,
) -> pd.DataFrame:
    """The rows of `rows` that `classifier_name` has not answered yet with the current prompt."""
    tag = current_prompt_tag(cfg, taxonomy)
    if tag is None:
        return rows
    model = model or cfg.classifier_model
    done = [cache.get(f"{c}|{classifier_name}|{model}|{tag}") is not None
            for c in rows["complaint_id"].astype(str)]  # fmt: skip
    return rows[[not d for d in done]]


def _ledger_lines(ledger: Ledger) -> list[dict]:
    if not ledger.path.exists():
        return []
    out = []
    with ledger.path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                float(rec["usd"])
                out.append(rec)
            except (ValueError, KeyError, TypeError):
                continue
    return out


def measured_usd_per_request(ledger: Ledger, model: str) -> float | None:
    """Mean real USD per request for `model` from the ledger file (None if none recorded)."""
    usd = [float(r["usd"]) for r in _ledger_lines(ledger) if r.get("model") == model]
    return sum(usd) / len(usd) if usd else None


def measured_usd_per_row(ledger: Ledger, model: str, tag: str) -> float | None:
    """Real USD per complaint for `model` from ledger lines with this exact `tag` (the
    current prompt), so older, smaller prompts never lower the projection."""
    lines = [r for r in _ledger_lines(ledger)
             if r.get("model") == model and r.get("tag") == tag and r.get("rows")]  # fmt: skip
    n = sum(int(r["rows"]) for r in lines)
    return sum(float(r["usd"]) for r in lines) / n if n else None


def estimate_measured(
    cfg: Config,
    ledger: Ledger,
    rows: pd.DataFrame,
    *,
    model: str | None = None,
    batch_size: int | None = None,
    runs: int = 1,
    tag: str | None = None,
) -> float | None:
    """Realistic USD from real past requests x 1.5 safety.

    With `tag`: real $ per complaint of requests made with exactly that prompt. Without it:
    mean $ per request for the model x requests needed. None when nothing comparable has been
    recorded yet (use the reserve bound instead).
    """
    model = model or cfg.classifier_model
    if tag is not None:
        per_row = measured_usd_per_row(ledger, model, tag)
        return None if per_row is None else runs * len(rows) * per_row * MEASURED_SAFETY
    per = measured_usd_per_request(ledger, model)
    if per is None:
        return None
    n_requests = math.ceil(len(rows) / (batch_size or cfg.batch_size))
    return runs * n_requests * per * MEASURED_SAFETY


def check_fits(
    ledger: Ledger, estimate_usd: float, label: str, bound_usd: float | None = None
) -> None:
    """Print the estimate (and the worst-case bound) against the remaining cap.

    Raises StageIncomplete if `estimate_usd` will not fit. The ledger still refuses any single
    request that could cross the cap, whatever this says.
    """
    left = ledger.cap_usd - ledger.spent()
    extra = f" (worst-case bound ${bound_usd:.2f})" if bound_usd is not None else ""
    print(f"{label}: estimated ${estimate_usd:.2f}{extra}; remaining cap ${left:.2f}")
    if estimate_usd > left:
        raise StageIncomplete(
            f"{label} would need about ${estimate_usd:.2f} but only ${left:.2f} of the "
            f"${ledger.cap_usd:.2f} cap is left. Nothing was spent."
        )


def ledger_report(ledger: Ledger) -> str:
    spent = ledger.spent()
    return (
        f"Paid spend so far (Gemini + Jev): ${spent:.2f} of ${ledger.cap_usd:.2f} cap "
        f"(${ledger.cap_usd - spent:.2f} remaining)"
    )


# --------------------------------------------------------------------------- scoring helpers


def accuracy_by_level(pred: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    """Share of predicted complaints matching the CFPB label per level (and full path)."""
    cols = ["product", "issue", "sub_issue"]
    m = pred.drop_duplicates("complaint_id").merge(
        truth[["complaint_id", *cols]].drop_duplicates("complaint_id"),
        on="complaint_id",
        suffixes=("", "_t"),
    )
    if m.empty:
        return {lv: math.nan for lv in LEVELS4}
    ok = {lv: m[lv].astype(str) == m[f"{lv}_t"].astype(str) for lv in cols}
    out = {lv: float(ok[lv].mean()) for lv in cols}
    out["path"] = float((ok["product"] & ok["issue"] & ok["sub_issue"]).mean())
    return out


def _md_table(df: pd.DataFrame) -> str:
    def fmt(v) -> str:
        return f"{v:.3f}" if isinstance(v, float) and not pd.isna(v) else str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def results_markdown(scored: dict, spent_usd: float, cap_usd: float) -> str:
    """Markdown results summary: decision, adjusted accuracy, leaderboards, summary quality
    (when `scored` has it), spend."""
    d = scored["decision"]
    lines = [
        f"## Phase 1 result: {'GO' if d['go'] else 'NO-GO'}",
        "",
        *[f"- {r}" for r in d["reasons"]],
        "",
        "### Adjusted accuracy (Gemini vs your hand check, reweighted to the whole test set)",
        _md_table(scored["adjusted"]),
        "",
        "### Classifier leaderboard (vs CFPB labels)",
        _md_table(scored["classifier_board"]),
        "",
        "### Validator leaderboard",
        _md_table(scored["validator_board"]),
        "",
    ]
    if scored.get("summary_quality") is not None:
        lines += [
            "### Summary quality (judge on the hand-check rows, and your summary_ok answers)",
            _md_table(scored["summary_quality"]),
            "",
        ]
    lines += [
        f"**Total paid spend (Gemini + Jev): ${spent_usd:.2f}** of the ${cap_usd:.2f} cap.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- notebook steps


def load_complaints(
    cfg: Config, source: str, params, temp_directory: str | None = DUCKDB_TMP
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Staged `load_clean`: (complaints, filter_report); the report is stored beside it."""

    def fn() -> pd.DataFrame:
        df, report = load_clean(source, params, temp_directory=temp_directory)
        store.write(cfg, STAGE_FILTER_REPORT, report)
        return df

    complaints = store.stage(cfg, P.STAGE_COMPLAINTS, fn)
    return complaints, store.read(cfg, STAGE_FILTER_REPORT)


def split_summary(complaints: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Rows per Product x split (train / tuning / test) plus a total row."""
    m = splits.merge(complaints[["complaint_id", "product"]], on="complaint_id")
    t = m.pivot_table(
        index="product", columns="split", values="complaint_id", aggfunc="count", fill_value=0
    )
    t["total"] = t.sum(axis=1)
    t.loc["ALL"] = t.sum()
    return t.astype(int)


def run_classic(cfg: Config, name: str, make, train_df: pd.DataFrame, eval_df: pd.DataFrame):
    """Fit-or-load one non-LLM classifier; stage its predictions and probabilities."""
    if store.exists(cfg, P.stage_pred(name)) and store.exists(cfg, P.stage_probs(name)):
        return store.read(cfg, P.stage_pred(name))
    pred, probs = P.fit_predict_classifier(
        cfg, make(), train_df, eval_df[["complaint_id", "narrative"]], name
    )
    store.write(cfg, P.stage_pred(name), pred)
    store.write(cfg, P.stage_probs(name), probs)
    return pred


def compare_runs(
    runs: dict[str, Callable[[], pd.DataFrame]], truth: pd.DataFrame, ledger: Ledger
) -> pd.DataFrame:
    """Run each paid classification, report accuracy per level and the real $ (ledger delta).

    A run Google refuses (retired model, unsupported thinking level, ...) costs nothing: it is
    printed as "<run> skipped: <reason>", shown as skipped in the table, and the rest go on.
    """
    rows = []
    for name, fn in runs.items():
        before = ledger.spent()
        try:
            pred = fn()
        except RequestRejected as e:
            print(f"{name} skipped: {e.message}")
            rows.append(
                {"run": name, "status": f"skipped: {e.message}", "rows_done": 0,
                 "of": len(truth), **dict.fromkeys(LEVELS4, math.nan),
                 "usd": ledger.spent() - before}
            )  # fmt: skip
            continue
        except DailyQuotaExhausted as e:
            raise StageIncomplete(
                f"The daily Gemini request quota is used up during '{name}' "
                f"({e.remaining_rows} rows left). Finished rows are saved - rerun this cell "
                "tomorrow."
            ) from None
        except BudgetExceeded as e:
            raise StageIncomplete(
                f"Budget reached during '{name}'. Nothing more was spent. ({e})"
            ) from None
        except ServiceUnavailable as e:
            raise StageIncomplete(
                f"During '{name}': {e} Finished rows are saved - rerun this cell later."
            ) from None
        acc = accuracy_by_level(pred, truth)
        rows.append(
            {"run": name, "status": "ok", "rows_done": len(pred), "of": len(truth), **acc,
             "usd": ledger.spent() - before}
        )  # fmt: skip
    return pd.DataFrame(rows)


def cached_predictions(
    cache: JsonlCache, ids: Iterable[str], model: str, prompt_version: str
) -> tuple[str | None, pd.DataFrame]:
    """(run name, predictions) of the saved run for `model` and `prompt_version` that covers
    the most of `ids` (complaint_id, product, issue, sub_issue); (None, empty) if none."""
    want = {str(i) for i in ids}
    by_run: dict[str, list[dict]] = {}
    for key in sorted(cache.keys()):
        parts = key.split("|")
        if len(parts) != 4 or parts[2] != model or parts[3] != prompt_version or parts[0] not in want:
            continue
        v = cache.get(key) or {}
        if all(k in v for k in ("product", "issue", "sub_issue")):
            by_run.setdefault(parts[1], []).append({"complaint_id": parts[0], **v})
    if not by_run:
        return None, pd.DataFrame(columns=["complaint_id", "product", "issue", "sub_issue"])
    run = max(sorted(by_run), key=lambda r: len(by_run[r]))
    return run, pd.DataFrame(by_run[run])


V1_REFERENCE = (
    "For reference, the first run (prompt v1, gemini-3.8-flash, different complaints - a rough "
    "guide only), product/issue/sub-issue/path: bake-off on 50 tuning complaints "
    ".84/.62/.40/.38; 800-row test .820/.514/.371/.324."
)
MIN_OLD_OVERLAP = 30  # show the old-prompt comparison only with at least this many complaints
OTHER_PAID_USD = 0.15  # Jev + summary judge, for the rough projection after Part C


def project_rest_of_phase(cfg: Config, ledger: Ledger, taxonomy: pd.DataFrame) -> float | None:
    """Print the projected Gemini cost of Part D + notebook 04 from the current prompt's real
    cost per complaint (x1.5 safety); None if no request with this prompt is recorded yet."""
    tag = current_prompt_tag(cfg, taxonomy)
    per_row = None if tag is None else measured_usd_per_row(ledger, cfg.classifier_model, tag)
    if per_row is None:
        return None
    n = cfg.test_n + 2 * cfg.self_consistency_n
    usd = n * per_row * MEASURED_SAFETY
    left = ledger.cap_usd - ledger.spent()
    print(
        f"Projected cost for the rest of the phase: about ${usd + OTHER_PAID_USD:.2f} "
        f"(test run {cfg.test_n} + self-consistency 2 x {cfg.self_consistency_n} complaints at "
        f"${per_row:.4f} each, measured just now, x1.5 safety; plus ~${OTHER_PAID_USD:.2f} for "
        f"Jev and the summary judge). Remaining cap ${left:.2f}. This is an estimate; the spend "
        "guard still refuses any request that would cross the cap."
    )
    if usd + OTHER_PAID_USD > left:
        print(
            "That will not fit in what is left of the cap. Nothing more has been spent. Options: "
            "use batch size 25 (Part B; the category list is then sent 2.5x less often, which "
            "costs much less), or ask Claude. Do not start Part D until it fits."
        )
    return usd + OTHER_PAID_USD


def prompt_check(
    cfg: Config,
    tuning_df: pd.DataFrame,
    client: GeminiClient,
    ledger: Ledger,
    cache: JsonlCache,
    taxonomy: pd.DataFrame,
    limiter: RateLimiter | None = None,
    old_version: str = "v1",
) -> pd.DataFrame:
    """Part C: the current prompt on all tuning complaints vs TF-IDF on the same complaints.

    Also prints the first run's (v1) numbers for reference, an old-prompt row when at least 30
    of these complaints were answered by the old prompt, and a cost projection for the rest of
    the phase from this check's real cost. A fully cached rerun is free and never blocked.
    """
    name = f"tuning_{cfg.prompt_version}"
    check_fits(
        ledger,
        estimate_classification(cfg, tuning_df, taxonomy, cache=cache, classifier_name=name),
        "Tuning check",
    )
    got: dict[str, pd.DataFrame] = {}

    def run() -> pd.DataFrame:
        got["pred"] = run_classification(tuning_df, cfg, client, ledger, cache, name,
                                         taxonomy=taxonomy, limiter=limiter)  # fmt: skip
        return got["pred"]

    label = f"Gemini, new prompt ({cfg.prompt_version})"
    table = compare_runs({label: run}, tuning_df, ledger)
    new = got.get("pred")
    if new is None:
        print("The tuning check did not run (see the 'skipped' reason above); nothing to "
              "compare. Fix that first, then rerun this cell.")  # fmt: skip
        print(table.round(3).to_string(index=False))
        return table
    ids = set(tuning_df["complaint_id"].astype(str))
    extra = []
    if store.exists(cfg, P.stage_pred("tfidf")):
        tf = store.read(cfg, P.stage_pred("tfidf"))
        tf = tf[tf["complaint_id"].astype(str).isin(ids)]
        extra.append({"run": "TF-IDF (same complaints)", "status": "ok", "rows_done": len(tf),
                      "of": len(tuning_df), **accuracy_by_level(tf, tuning_df), "usd": 0.0})  # fmt: skip
    else:
        print("No TF-IDF predictions found (notebook 02), so no TF-IDF comparison.")
    old_run, old = cached_predictions(cache, ids, cfg.classifier_model, old_version)
    same = sorted(set(old["complaint_id"]) & set(new["complaint_id"].astype(str)))
    if len(same) >= MIN_OLD_OVERLAP:
        n = len(same)
        extra += [
            {"run": f"old prompt ({old_version}), same {n}", "status": f"saved run {old_run}",
             "rows_done": n, "of": n, **accuracy_by_level(old[old["complaint_id"].isin(same)],
             tuning_df), "usd": 0.0},
            {"run": f"{label}, same {n}", "status": "ok", "rows_done": n, "of": n,
             **accuracy_by_level(new[new["complaint_id"].astype(str).isin(same)], tuning_df),
             "usd": 0.0},
        ]  # fmt: skip
    elif same:
        print(f"Only {len(same)} of these complaints were answered by the {old_version} prompt "
              f"(fewer than {MIN_OLD_OVERLAP}), so no direct comparison with it.")  # fmt: skip
    if extra:
        table = pd.concat([table, pd.DataFrame(extra)], ignore_index=True)
    print(table.round(3).to_string(index=False))
    print(V1_REFERENCE)
    project_rest_of_phase(cfg, ledger, taxonomy)
    return table


def diagnose(
    cfg: Config,
    api_key: str,
    ledger: Ledger,
    candidates: Iterable[str] | None = None,
    make_client: Callable | None = None,
) -> None:
    """Free troubleshooting printout (no generate calls): settings, library, key, spend, saves."""
    try:
        from importlib.metadata import version

        genai_version = version("google-genai")
    except Exception:  # noqa: BLE001 - not installed / unknown
        genai_version = "not installed"
    print("== Troubleshooting (free: nothing is sent to Gemini except a model list) ==")
    print(f"Candidates in memory: {list(candidates) if candidates else '(not set in this session)'}")
    print(
        f"Classifier {cfg.classifier_model} | judge {cfg.judge_model} | describe "
        f"{cfg.describe_model} | thinking {cfg.thinking} | batch {cfg.batch_size} | prompt "
        f"{cfg.prompt_version}"
    )
    print(f"google-genai library version: {genai_version}")
    print(f"Gemini key: {_key_facts(api_key)}")
    check_gemini_key(api_key, make_client=make_client, stop=False)
    print(ledger_report(ledger))
    per_model: dict[str, int] = {}
    if ledger.path.exists():
        for line in ledger.path.read_text(encoding="utf-8").splitlines():
            try:
                m = json.loads(line)["model"]
            except (ValueError, KeyError, TypeError):
                continue
            per_model[m] = per_model.get(m, 0) + 1
    for m, n in sorted(per_model.items()):
        print(f"  {m}: {n} paid request(s)")
    cache = JsonlCache(Path(cfg.artifacts_dir) / "gemini_cache.jsonl")
    per_run: dict[str, int] = {}
    for key in sorted(cache.keys()):
        parts = key.split("|")
        run = " | ".join(parts[1:]) if len(parts) >= 4 else key
        per_run[run] = per_run.get(run, 0) + 1
    print(f"Saved Gemini answers: {sum(per_run.values())}")
    for run, n in sorted(per_run.items()):
        print(f"  {run}: {n}")


def phase_estimate(
    cfg: Config,
    ledger: Ledger,
    taxonomy: pd.DataFrame,
    test_df: pd.DataFrame,
    sc_rows: pd.DataFrame,
    cache: JsonlCache | None = None,
) -> pd.DataFrame:
    """Per remaining paid step: `bound_usd` (worst case) and `usd` (measured, <= bound).

    Gemini classification is measured only from requests made with the current prompt (its
    ledger tag), never from older, smaller prompts; until there are some, the estimate is the
    bound computed from the real prompt text. With `cache`, complaints already answered are
    left out of the classification steps.
    """
    judge = cfg.judge_model
    hc = test_df.head(cfg.hand_check_n)
    tag = current_prompt_tag(cfg, taxonomy)
    spec = [  # (stage, label, rows, model, classifier name)
        (P.STAGE_PRED_GEMINI, "Test classification", test_df, cfg.classifier_model, "gemini"),
        (P.STAGE_PRED_GEMINI_SC2, "Self-consistency run 2", sc_rows, cfg.classifier_model,
         "gemini_sc2"),
        (P.STAGE_PRED_GEMINI_SC3, "Self-consistency run 3", sc_rows, cfg.classifier_model,
         "gemini_sc3"),
        (P.STAGE_JEV, "Jev validator, full test", test_df, cfg.jev_model, None),
        (P.STAGE_SUMMARY_JUDGE, "Summary judge, hand-check rows", hc, judge, None),
    ]  # fmt: skip
    out = []
    for stage, label, rows, model, clf in spec:
        if model == cfg.jev_model:  # one request per complaint, input-only pricing
            bound = estimate_jev_usd(cfg, rows, model)
            meas = estimate_measured(cfg, ledger, rows, model=model, batch_size=1)
        elif stage == P.STAGE_SUMMARY_JUDGE:
            bound = estimate_summary_judge_usd(cfg, rows, model)
            meas = estimate_measured(cfg, ledger, rows, model=model,
                                     tag=f"summary_judge:{SUMMARY_JUDGE_PROMPT_VERSION}")  # fmt: skip
        else:
            if cache is not None:
                rows = uncached_rows(cfg, rows, taxonomy, cache, clf, model)
            bound = estimate_classification(cfg, rows, taxonomy, model=model)
            meas = None if tag is None else estimate_measured(cfg, ledger, rows, model=model,
                                                              tag=tag)  # fmt: skip
        out.append(
            (stage, label, bound, bound if meas is None else min(meas, bound),
             meas is not None, store.exists(cfg, stage))
        )  # fmt: skip
    df = pd.DataFrame(out, columns=["stage", "step", "bound_usd", "usd", "measured", "done"])
    show = df[["step", "bound_usd", "usd", "measured", "done"]].copy()
    show["bound_usd"] = show["bound_usd"].map("${:.2f}".format)
    show["usd"] = show["usd"].map("${:.2f}".format)
    print(show.rename(columns={"usd": "estimate"}).to_string(index=False))
    print(
        f"Still to spend: estimate ${pending_usd(df):.2f} "
        f"(worst-case bound ${pending_usd(df, column='bound_usd'):.2f}); {ledger_report(ledger)}"
    )
    return df


def pending_usd(
    est: pd.DataFrame, stages: Iterable[str] | None = None, column: str = "usd"
) -> float:
    """Sum of `column` over steps not yet done (optionally only the given stages)."""
    sel = est[~est["done"]]
    if stages is not None:
        sel = sel[sel["stage"].isin(list(stages))]
    return float(sel[column].sum())


def make_descriptions(
    cfg: Config,
    complaints: pd.DataFrame,
    splits: pd.DataFrame,
    taxonomy: pd.DataFrame,
    client: GeminiClient,
    ledger: Ledger,
    cache: JsonlCache,
    limiter: RateLimiter | None = None,
) -> pd.DataFrame:
    """Part 0: the saved descriptions stage, written by Gemini if missing or out of date.

    Every paid answer is cached per product, so a rerun never pays twice. Stops (saving
    nothing) with a plain-English StageIncomplete if the budget, the daily quota, an outage or
    a refused request gets in the way.
    """
    if store.exists(cfg, STAGE_DESCRIPTIONS):
        d = store.read(cfg, STAGE_DESCRIPTIONS)
        if descriptions_match(d, taxonomy):
            return d
    try:
        d = run_descriptions(complaints, splits, taxonomy, cfg, client, ledger, cache,
                             limiter=limiter)  # fmt: skip
    except DailyQuotaExhausted:
        raise StageIncomplete(
            "Gemini's daily request quota is used up. Finished products are saved - rerun this "
            "cell tomorrow."
        ) from None
    except BudgetExceeded as e:
        raise StageIncomplete(
            f"Budget reached. Nothing more was spent. ({e}) Finished products are saved."
        ) from None
    except ServiceUnavailable as e:
        raise StageIncomplete(f"{e} Finished products are saved - rerun this cell later.") from None
    except RequestRejected as e:
        raise StageIncomplete(
            f"Google refused the request: {e.message} Nothing was charged for it."
        ) from None
    except DescriptionError:
        raise
    store.write(cfg, STAGE_DESCRIPTIONS, d)
    print(f"{len(d)} category descriptions written. {ledger_report(ledger)}")
    return d


def _same_text(a: object, b: object) -> bool:
    return " ".join(str(a).split()) == " ".join(str(b).split())


def publish_descriptions(
    cfg: Config,
    gc,
    taxonomy: pd.DataFrame,
    descriptions: pd.DataFrame,
    overwrite_edits: bool = False,
) -> str:
    """Show the descriptions in the taxonomy Google Sheet (label, product, issue, sub_issue,
    description); creates the Sheet first if notebook 01 has not.

    Never wipes Marco's edits: if the Sheet already has descriptions that differ from the
    saved ones, the Sheet is left alone (unless `overwrite_edits`) and he is told what to do.
    """
    title = f"CFPB taxonomy ({cfg.data_version})"
    url = publish_once(cfg, "taxonomy", lambda: publish_taxonomy(gc, taxonomy, title))
    if not overwrite_edits:
        try:
            sheet = read_description_edits(gc, url)
        except ValueError:  # no description column yet: nothing to protect
            sheet = None
        if sheet is not None and len(sheet):
            saved = dict(zip(descriptions["label"], descriptions["description"]))
            differ = [x for x, t in zip(sheet["label"], sheet["description"])
                      if x in saved and not _same_text(t, saved[x])]  # fmt: skip
            if differ:
                print(
                    f"{len(differ)} description(s) in the Sheet differ from the saved ones (your "
                    f"edits?), e.g. {differ[:3]}. The Sheet was NOT changed. To keep them, run the "
                    "'use your own edits' cell below. To throw them away and show the saved "
                    "descriptions instead, set DISCARD_SHEET_EDITS = True in this cell and rerun it."
                )
                print("Taxonomy sheet:", url)
                return url
    rewrite_taxonomy_sheet(gc, url, taxonomy, descriptions)
    print("Taxonomy sheet with descriptions:", url)
    return url


def load_description_edits(cfg: Config, gc, taxonomy: pd.DataFrame) -> pd.DataFrame:
    """Read Marco's edits from the taxonomy Sheet and save them as the descriptions stage."""
    if not store.exists(cfg, STAGE_DESCRIPTIONS):
        raise StageIncomplete("No descriptions yet - run the Part 0 cells above first.")
    current = store.read(cfg, STAGE_DESCRIPTIONS)
    try:
        edits = read_description_edits(gc, read_url(cfg, "taxonomy"))
    except ValueError as e:
        raise StageIncomplete(str(e)) from None
    new = apply_edits(current, edits)
    changed = int((new["description"] != current["description"]).sum())
    store.write(cfg, STAGE_DESCRIPTIONS, new)
    print(
        f"{changed} description(s) changed and saved. Every later Gemini run uses them "
        "(answers given with the old wording are not reused)."
        if changed
        else "No description changed."
    )
    if changed and store.exists(cfg, P.STAGE_PRED_GEMINI):
        print(
            "WARNING: the 800-row test run (Part D) already used the old wording. Notebook 04's "
            "self-consistency reruns would now use the new wording, mixing two prompts. Load "
            "edits only before Part C/D; tell Claude if you want to redo the test run."
        )
    return new


def load_human_review(cfg: Config, gc, url: str, allow_partial: bool = False) -> pd.DataFrame:
    """Read the verdicts from the review sheet, print progress, store as Parquet."""
    human = read_review_sheet(gc, url)
    counts = count_verdicts(human)
    print(counts.to_string(index=False))
    blank_summary = int(((human["summary_ok"] == "") & (human["level"] == "product")).sum())
    n_blank = int(counts["n_blank"].sum())
    print(f"Blank verdict cells: {n_blank}; blank summary_ok cells: {blank_summary}")
    if n_blank and not allow_partial:
        raise StageIncomplete(
            f"{n_blank} verdict cells in the review sheet are still blank. Fill them in "
            "(use 'ambiguous' if you cannot tell), or set ALLOW_PARTIAL = True to score "
            "only what is filled in."
        )
    store.write(cfg, P.STAGE_HUMAN_REVIEW, human)
    return human


def show_results(cfg: Config, scored: dict, ledger: Ledger) -> str:
    """Save results.md next to the Parquet tables and display it as Markdown."""
    text = results_markdown(scored, ledger.spent(), ledger.cap_usd)
    (Path(cfg.artifacts_dir) / "results.md").write_text(text, encoding="utf-8")
    try:
        from IPython.display import Markdown, display

        display(Markdown(text))
    except ImportError:
        print(text)
    return text


def list_outputs(cfg: Config) -> list[str]:
    names = sorted(p.name for p in Path(cfg.artifacts_dir).glob("*.parquet"))
    print(f"{len(names)} Parquet tables in {cfg.artifacts_dir}:")
    print("  " + ", ".join(names))
    return names
