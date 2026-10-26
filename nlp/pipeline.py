"""Reusable pipeline orchestration shared by the smoke run and the Colab notebooks.

Plain functions over DataFrames; the caller wraps each output in `store.stage` so every step
stays resumable. Gemini-calling steps are separate functions so a notebook can stop and
resume between them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from nlp.budget import Ledger, RateLimiter
from nlp.classifiers.base import PathClassifier, fit_or_load, oof_proba, probs_long, to_predictions
from nlp.config import Config
from nlp.ensemble import ensemble_learned, ensemble_simple
from nlp.llm.classify import run_classification
from nlp.llm.client import GeminiClient
from nlp.llm.judge import judge_classifications, judge_summaries
from nlp.metrics import inverse_weights
from nlp.review import agree_path, is_correct
from nlp.schema import VALIDATOR_RESULTS_COLS, path_id
from nlp.scoring import (
    adjusted_accuracy,
    classifier_leaderboard,
    go_no_go,
    validator_leaderboard,
)
from nlp.store import JsonlCache
from nlp.validators import (
    v_cleanlab,
    v_hierarchy,
    v_label_match,
    v_second_opinion,
    v_self_confidence,
    v_self_consistency,
)

LEVELS3 = ["product", "issue", "sub_issue"]

STAGE_COMPLAINTS = "complaints"
STAGE_TAXONOMY = "taxonomy"
STAGE_SPLITS = "splits"
STAGE_PRED_GEMINI = "pred_gemini"
STAGE_PRED_GEMINI_SC2 = "pred_gemini_sc2"
STAGE_PRED_GEMINI_SC3 = "pred_gemini_sc3"
STAGE_HAND_CHECK = "hand_check"
STAGE_JEB = "validator_jeb"
STAGE_SUMMARY_JUDGE = "summary_judge"
STAGE_CLEANLAB = "validator_cleanlab"
STAGE_VALIDATORS = "validator_results"
STAGE_HUMAN_REVIEW = "human_review"
STAGE_ENSEMBLE_SIMPLE = "ensemble_simple"
STAGE_ENSEMBLE_LEARNED = "ensemble_learned"
STAGE_ROW_STATUS = "row_status"
STAGE_CLF_BOARD = "classifier_leaderboard"
STAGE_ADJUSTED = "adjusted_accuracy"
STAGE_VALIDATOR_BOARD = "validator_leaderboard"


def stage_pred(name: str) -> str:
    return f"pred_{name}"


def stage_probs(name: str) -> str:
    return f"probs_{name}"


def train_test_frames(
    complaints: pd.DataFrame, splits: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) frames of complaints with a `path_id` column (CFPB labels)."""
    full = complaints.assign(
        path_id=[path_id(*t) for t in zip(*(complaints[c] for c in LEVELS3))]
    )
    out = []
    for split in ("train", "test"):
        ids = splits.loc[splits["split"] == split, "complaint_id"]
        out.append(full[full["complaint_id"].isin(ids)].reset_index(drop=True))
    return out[0], out[1]


def split_rows(complaints: pd.DataFrame, splits: pd.DataFrame, mask) -> pd.DataFrame:
    """Complaints whose `splits` row satisfies boolean `mask` (e.g. the self-consistency subset)."""
    ids = splits.loc[mask, "complaint_id"]
    return complaints[complaints["complaint_id"].isin(ids)].reset_index(drop=True)


def fit_predict_classifier(
    cfg: Config,
    clf: PathClassifier,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit-or-load `clf` on train_df, predict eval_df -> (predictions, probs_long)."""
    clf = fit_or_load(cfg, clf, train_df)
    ids = eval_df["complaint_id"].tolist()
    classes, p = clf.predict_proba(eval_df["narrative"].tolist())
    model_id = getattr(clf, "model_id", clf.name)
    return to_predictions(name or clf.name, model_id, ids, classes, p), probs_long(ids, classes, p)


def run_gemini(
    cfg: Config,
    rows: pd.DataFrame,
    client: GeminiClient,
    ledger: Ledger,
    cache: JsonlCache,
    taxonomy: pd.DataFrame,
    classifier_name: str = "gemini",
    limiter: RateLimiter | None = None,
) -> pd.DataFrame:
    """Gemini classification of `rows` (complaint_id, narrative); resumable via `cache`."""
    return run_classification(
        rows, cfg, client, ledger, cache, classifier_name, taxonomy=taxonomy, limiter=limiter
    )


def _with_narrative(gemini: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    return gemini.merge(test_df[["complaint_id", "narrative"]], on="complaint_id")


def run_jeb(cfg, gemini, test_df, client, ledger, cache, limiter=None) -> pd.DataFrame:
    """JEB validator rows for every Gemini prediction (Gemini-calling step)."""
    return judge_classifications(
        _with_narrative(gemini, test_df), cfg, client, ledger, cache, limiter=limiter
    )


def run_summary_judge(
    cfg, gemini, test_df, selection, client, ledger, cache, limiter=None
) -> pd.DataFrame:
    """Summary-judge flags for the hand-check rows (Gemini-calling step)."""
    hc = gemini[gemini["complaint_id"].isin(selection["complaint_id"])]
    return judge_summaries(_with_narrative(hc, test_df), cfg, client, ledger, cache, limiter=limiter)


def cleanlab_validator(
    cfg: Config,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    gemini: pd.DataFrame,
    tfidf_probs: pd.DataFrame,
    make_tfidf: Callable[[], PathClassifier],
    n_splits: int = 5,
) -> pd.DataFrame:
    """cleanlab validator rows: OOF probs on the train pool, TF-IDF probs on the test set."""
    test_ids = test_df["complaint_id"].tolist()
    classes, oof = oof_proba(
        make_tfidf, train_df["narrative"].tolist(), train_df["path_id"].tolist(),
        n_splits=n_splits, seed=cfg.seed,
    )  # fmt: skip
    wide = tfidf_probs.pivot(index="complaint_id", columns="path_id", values="prob")
    test_probs = wide.reindex(index=test_ids, columns=classes).fillna(0.0).to_numpy()
    g = gemini.drop_duplicates("complaint_id").set_index("complaint_id").loc[test_ids]
    return v_cleanlab(
        [path_id(*t) for t in zip(g["product"], g["issue"], g["sub_issue"])],
        classes, oof, train_df["path_id"].tolist(),
        test_ids, test_df["path_id"].tolist(), test_probs,
    )  # fmt: skip


def build_validator_results(
    complaints: pd.DataFrame,
    taxonomy: pd.DataFrame,
    gemini: pd.DataFrame,
    sc_runs: Sequence[pd.DataFrame],
    probs: dict[str, pd.DataFrame],
    cleanlab_rows: pd.DataFrame,
    jeb: pd.DataFrame,
) -> pd.DataFrame:
    """Combine every validator into one validator_results frame (no API calls here).

    `sc_runs` = extra self-consistency runs (e.g. [sc2, sc3]); run 1 is Gemini's main
    prediction restricted to the complaints in the first extra run. `probs` maps classifier
    name -> probs_long frame (second-opinion validators).
    """
    parts = [
        v_label_match(gemini, complaints),
        *(v_second_opinion(gemini, p, name) for name, p in probs.items()),
        v_self_confidence(gemini),
        v_self_consistency(
            [gemini[gemini["complaint_id"].isin(sc_runs[0]["complaint_id"])], *sc_runs]
        ),
        v_hierarchy(gemini, taxonomy),
        cleanlab_rows,
        jeb,
    ]
    return pd.concat(parts, ignore_index=True)[VALIDATOR_RESULTS_COLS]


def human_labels(human_review: pd.DataFrame) -> pd.Series:
    """Series (index complaint_id): Gemini right at all three levels per Marco's verdicts.

    Complaints with an ambiguous or blank verdict, or a missing level, are excluded.
    """
    out = {}
    for cid, g in human_review.groupby("complaint_id", sort=True):
        if set(g["level"]) != set(LEVELS3) or (g["verdict"] == "").any():
            continue
        oks = [is_correct(v) for v in g["verdict"]]
        if None not in oks:
            out[cid] = all(oks)
    return pd.Series(
        list(out.values()), index=pd.Index(list(out), name="complaint_id"), name="correct",
        dtype=bool,
    )  # fmt: skip


def run_ensembles(
    vres: pd.DataFrame, human_review: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(ensemble_simple, ensemble_learned) from validator results and human verdicts."""
    labels = human_labels(human_review).reset_index()
    return ensemble_simple(vres), ensemble_learned(vres, labels)


def review_weights(
    gemini_test: pd.DataFrame, complaints: pd.DataFrame, selection: pd.DataFrame
) -> pd.Series:
    """Inverse-probability weights for the hand-checked rows, indexed by complaint_id.

    Strata = whether Gemini's path equals the CFPB path; the population is all scored test rows.
    """
    agree = agree_path(gemini_test, complaints)
    strata = pd.Series(np.where(agree.to_numpy(dtype=bool), "agree", "disagree"), index=agree.index)
    return inverse_weights(strata.reindex(selection["complaint_id"]), strata)


def all_validator_results(
    vres: pd.DataFrame, ens_simple: pd.DataFrame, ens_learned: pd.DataFrame
) -> pd.DataFrame:
    return pd.concat(
        [vres, ens_simple[VALIDATOR_RESULTS_COLS], ens_learned[VALIDATOR_RESULTS_COLS]],
        ignore_index=True,
    )


def run_scoring(
    cfg: Config,
    preds_all: pd.DataFrame,
    complaints: pd.DataFrame,
    test_ids: Sequence[str],
    gemini: pd.DataFrame,
    human_review: pd.DataFrame,
    selection: pd.DataFrame,
    validator_results: pd.DataFrame,
    costs: dict[str, dict] | None = None,
    n_boot: int = 1000,
    stage: Callable[[str, Callable[[], pd.DataFrame]], pd.DataFrame] | None = None,
) -> dict:
    """Boards + go/no-go. `stage(name, fn)` (e.g. a store.stage wrapper) makes boards resumable."""
    st = stage or (lambda _name, fn: fn())
    clf_board = st(
        STAGE_CLF_BOARD, lambda: classifier_leaderboard(preds_all, complaints, test_ids)
    )
    rate = float(agree_path(gemini, complaints).mean())
    adjusted = st(
        STAGE_ADJUSTED,
        lambda: adjusted_accuracy(human_review, selection, rate, n_boot=n_boot, seed=cfg.seed),
    )
    weights = review_weights(gemini, complaints, selection)
    val_board = st(
        STAGE_VALIDATOR_BOARD,
        lambda: validator_leaderboard(validator_results, human_review, weights, costs or {}),
    )
    return {
        "classifier_board": clf_board,
        "adjusted": adjusted,
        "validator_board": val_board,
        "decision": go_no_go(clf_board, adjusted),
    }
