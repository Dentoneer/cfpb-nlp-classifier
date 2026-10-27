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
from nlp.llm.classify import build_prompt, estimate_request_tokens
from nlp.review import count_verdicts
from nlp.sheets import read_review_sheet
from nlp.store import JsonlCache
from nlp.taxonomy import taxonomy_prompt_text

REPO_URL = "https://github.com/Dentoneer/cfpb-nlp-classifier"
OVERRIDES_FILE = "config_overrides.json"
LEVELS4 = ["product", "issue", "sub_issue", "path"]
STAGE_FILTER_REPORT = "filter_report"
ALLOWED_OVERRIDES = ("classifier_model", "judge_model", "batch_size", "thinking")
MAX_MISSING_FRAC = 0.02  # stage_paid saves a stage missing at most this share of rows
BUDGET_FLOOR_USD = 0.10  # less than this left under the cap counts as "budget reached"
MEASURED_SAFETY = 1.5
RESTART_MSG = "Restart session (Runtime menu \u2192 Restart session), then rerun this cell"


class StageIncomplete(Exception):
    """A paid or manual step did not finish; the message tells Marco what to do next."""


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
        raise ValueError("Set PICK to your chosen model (e.g. 'gemini-2.5-flash') and rerun.")
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


def setup(
    repo_dir: str = "/content/cfpb-nlp-classifier",
    drive_subdir: str = "cfpb-nlp-classifier/phase1",
):
    """Mount Drive, load config, fetch the Gemini key, authorise Sheets -> (cfg, gc, api_key)."""
    check_code_fresh(_HEAD_AT_IMPORT, _git_head(Path(repo_dir)))
    from google.colab import auth, drive, userdata

    drive.mount("/content/drive")
    artifacts = Path("/content/drive/MyDrive") / drive_subdir
    cfg = load_config(Path(repo_dir) / "config" / "phase1.yaml", artifacts)
    try:
        api_key = userdata.get("GEMINI_API_KEY")
    except Exception as e:  # secret missing or notebook access not granted
        raise RuntimeError(
            "GEMINI_API_KEY not found. In Colab open the key icon (Secrets) on the left, add "
            "GEMINI_API_KEY and switch 'Notebook access' on."
        ) from e
    auth.authenticate_user()
    import google.auth
    import gspread

    creds, _ = google.auth.default()
    gc = gspread.authorize(creds)
    print(f"Ready. Outputs go to {artifacts}")
    print(f"Saved choices applied: {read_overrides(artifacts) or 'none yet'}")
    print(f"Classifier {cfg.classifier_model} | judge {cfg.judge_model} | cap ${cfg.spend_cap_usd:.2f}")
    return cfg, gc, api_key


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
    ledger: Ledger | None = None,
) -> pd.DataFrame:
    """`store.stage` for Gemini steps: saves only a (nearly) complete result.

    A shortfall caused by budget (under $0.10 left under the cap) or more than 2% of rows
    missing is not saved - rerun later. Up to 2% missing for other reasons (rows Gemini
    keeps answering invalidly) is saved and the missing ids are printed.

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
        raise StageIncomplete(f"Budget reached ({e}). Nothing more was spent.") from None
    got = set(df["complaint_id"].astype(str)) if len(df) else set()
    missing = sorted(want - got)
    if missing:
        frac = len(missing) / len(want)
        budget_hit = ledger is not None and ledger.cap_usd - ledger.spent() < BUDGET_FLOOR_USD
        if budget_hit or frac > MAX_MISSING_FRAC:
            raise StageIncomplete(
                f"{len(want) - len(missing)} of {len(want)} complaints done for '{name}'. "
                "Either the budget cap was reached (see the ledger line below) or many rows "
                "failed validation (see messages above). Finished rows are saved; rerun this "
                "cell to retry the rest."
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
) -> float:
    """Conservative USD estimate (the same token estimate the spend guard reserves with)."""
    model = model or cfg.classifier_model
    bs = batch_size or cfg.batch_size
    thinking = thinking or cfg.thinking
    tax = taxonomy_prompt_text(taxonomy)
    text = [clean_text(str(t), cfg.max_chars) for t in rows["narrative"]]
    ids = [str(c) for c in rows["complaint_id"]]
    total = 0.0
    for i in range(0, len(ids), bs):
        batch = pd.DataFrame({"complaint_id": ids[i : i + bs], "narrative": text[i : i + bs]})
        system, user = build_prompt(batch, tax, cfg.prompt_version)
        est_in, est_out = estimate_request_tokens(system, user, len(batch), thinking)
        total += estimate_run(1, est_in, est_out, model, cfg.prices)
    return runs * total


def measured_usd_per_request(ledger: Ledger, model: str) -> float | None:
    """Mean real USD per request for `model` from the ledger file (None if none recorded)."""
    if not ledger.path.exists():
        return None
    usd = []
    with ledger.path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec["model"] == model:
                    usd.append(float(rec["usd"]))
            except (ValueError, KeyError, TypeError):
                continue
    return sum(usd) / len(usd) if usd else None


def estimate_measured(
    cfg: Config,
    ledger: Ledger,
    rows: pd.DataFrame,
    *,
    model: str | None = None,
    batch_size: int | None = None,
    runs: int = 1,
) -> float | None:
    """Realistic USD: measured $/request for the model x requests needed x 1.5 safety.

    None when the ledger has no record for the model yet (use the reserve bound instead).
    """
    model = model or cfg.classifier_model
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
        f"Gemini spend so far: ${spent:.2f} of ${ledger.cap_usd:.2f} cap "
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
    """Markdown results summary: decision, adjusted accuracy, both leaderboards, spend."""
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
        f"**Total Gemini spend: ${spent_usd:.2f}** of the ${cap_usd:.2f} cap.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- notebook steps


def load_complaints(cfg: Config, source: str, params) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Staged `load_clean`: (complaints, filter_report); the report is stored beside it."""

    def fn() -> pd.DataFrame:
        df, report = load_clean(source, params)
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
    """Run each paid classification, report accuracy per level and the real $ (ledger delta)."""
    rows = []
    for name, fn in runs.items():
        before = ledger.spent()
        try:
            pred = fn()
        except DailyQuotaExhausted as e:
            raise StageIncomplete(
                f"The daily Gemini request quota is used up during '{name}' "
                f"({e.remaining_rows} rows left). Finished rows are saved - rerun this cell "
                "tomorrow."
            ) from None
        except BudgetExceeded as e:
            raise StageIncomplete(f"Budget reached during '{name}' ({e}).") from None
        acc = accuracy_by_level(pred, truth)
        rows.append(
            {"run": name, "rows_done": len(pred), "of": len(truth), **acc,
             "usd": ledger.spent() - before}
        )  # fmt: skip
    return pd.DataFrame(rows)


def phase_estimate(
    cfg: Config,
    ledger: Ledger,
    taxonomy: pd.DataFrame,
    test_df: pd.DataFrame,
    sc_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Per remaining paid step: `bound_usd` (worst case) and `usd` (measured, <= bound)."""
    judge = cfg.judge_model
    hc = test_df.head(cfg.hand_check_n)
    spec = [  # (stage, label, rows, model)
        (P.STAGE_PRED_GEMINI, "Test classification", test_df, cfg.classifier_model),
        (P.STAGE_PRED_GEMINI_SC2, "Self-consistency run 2", sc_rows, cfg.classifier_model),
        (P.STAGE_PRED_GEMINI_SC3, "Self-consistency run 3", sc_rows, cfg.classifier_model),
        (P.STAGE_JEB, "JEB judge, full test", test_df, judge),
        (P.STAGE_SUMMARY_JUDGE, "Summary judge, hand-check rows", hc, judge),
    ]
    out = []
    for stage, label, rows, model in spec:
        bound = estimate_classification(cfg, rows, taxonomy, model=model)
        meas = estimate_measured(cfg, ledger, rows, model=model)
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
