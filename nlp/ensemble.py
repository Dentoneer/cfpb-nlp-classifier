"""Ensembles over validator results and the per-row status."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from nlp.config import Config
from nlp.schema import VALIDATOR_RESULTS_COLS, path_id

SEED = 20261026


def _base(results: pd.DataFrame) -> pd.DataFrame:
    """Drop earlier ensemble rows so ensembles never vote on themselves."""
    return results[~results["validator"].str.startswith("ensemble_")]


def ensemble_simple(results: pd.DataFrame) -> pd.DataFrame:
    """`ensemble_simple`: per complaint and level, the share of validators voting right.

    verdict = score >= 0.5; `dissent` = comma-separated validators voting wrong (sorted).
    """
    r = _base(results)
    rows = []
    for (cid, level), g in r.groupby(["complaint_id", "level"], sort=True):
        votes = g.drop_duplicates("validator")
        score = float(votes["verdict"].astype(bool).mean())
        wrong = sorted(votes.loc[~votes["verdict"].astype(bool), "validator"])
        rows.append((cid, "ensemble_simple", level, score >= 0.5, score, ",".join(wrong)))
    return pd.DataFrame(rows, columns=[*VALIDATOR_RESULTS_COLS, "dissent"])


def _features(results: pd.DataFrame, level: str) -> pd.DataFrame:
    """complaint_id x validator score matrix; missing scores -> column mean (0.5 if empty)."""
    r = _base(results)
    r = r[r["level"] == level]
    piv = r.pivot_table(index="complaint_id", columns="validator", values="score", aggfunc="mean")
    return piv.apply(lambda c: c.fillna(c.mean() if c.notna().any() else 0.5))


def ensemble_learned(
    results: pd.DataFrame,
    human_labels: pd.DataFrame,
    n_splits: int = 5,
    level: str = "path",
) -> pd.DataFrame:
    """`ensemble_learned`: logistic regression on the pivoted validator scores at `level`.

    `human_labels` = complaint_id, correct (bool: Gemini was right). Hand-checked rows get
    strictly out-of-fold scores (StratifiedKFold); all other rows are scored by a model fit on
    all hand-checked rows. The returned `fold` column is the fold a hand-checked row was held
    out in, or -1 for rows scored by the full-fit model.
    """
    X = _features(results, level)
    hl = human_labels.drop_duplicates("complaint_id").set_index("complaint_id")["correct"]
    hand = [c for c in hl.index if c in X.index]
    y = hl.loc[hand].astype(bool).to_numpy()
    minority = int(min(y.sum(), (~y).sum())) if len(y) else 0
    if minority < 2:
        raise ValueError("need at least 2 hand-checked rows of each class (right/wrong)")
    k = min(n_splits, minority)

    def fit(idx) -> LogisticRegression:
        return LogisticRegression(max_iter=1000).fit(X.loc[idx].to_numpy(), y_of(idx))

    ymap = dict(zip(hand, y))

    def y_of(idx) -> np.ndarray:
        return np.array([ymap[c] for c in idx])

    score: dict[str, float] = {}
    fold: dict[str, int] = {}
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    for f, (tr, te) in enumerate(skf.split(np.zeros(len(hand)), y)):
        model = fit([hand[i] for i in tr])
        te_ids = [hand[i] for i in te]
        for cid, p in zip(te_ids, model.predict_proba(X.loc[te_ids].to_numpy())[:, 1]):
            score[cid], fold[cid] = float(p), f
    rest = [c for c in X.index if c not in ymap]
    if rest:
        full = fit(hand)
        for cid, p in zip(rest, full.predict_proba(X.loc[rest].to_numpy())[:, 1]):
            score[cid], fold[cid] = float(p), -1
    ids = list(X.index)
    out = pd.DataFrame(
        {
            "complaint_id": ids,
            "validator": "ensemble_learned",
            "level": level,
            "verdict": [score[c] >= 0.5 for c in ids],
            "score": [score[c] for c in ids],
            "fold": [fold[c] for c in ids],
        }
    )
    return out[[*VALIDATOR_RESULTS_COLS, "fold"]]


def row_status(
    gemini: pd.DataFrame,
    complaints: pd.DataFrame,
    ensemble: pd.DataFrame,
    threshold: float | None = None,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """complaint_id, status in {verified, likely_correct, needs_review}.

    verified = Gemini's path equals the CFPB path; likely_correct = disagrees but the
    path-level ensemble score >= threshold (default cfg.likely_correct_threshold, else 0.6);
    otherwise needs_review (including rows with no ensemble score).
    """
    if threshold is None:
        threshold = cfg.likely_correct_threshold if cfg else 0.6
    cols = ["product", "issue", "sub_issue"]
    m = gemini[["complaint_id", *cols]].merge(
        complaints[["complaint_id", *cols]], on="complaint_id", suffixes=("", "_cfpb")
    )
    gp = [path_id(*t) for t in zip(*(m[c] for c in cols))]
    cp = [path_id(*t) for t in zip(*(m[f"{c}_cfpb"] for c in cols))]
    ens = ensemble[ensemble["level"] == "path"].drop_duplicates("complaint_id")
    es = m["complaint_id"].map(ens.set_index("complaint_id")["score"])
    status = []
    for g, c, s in zip(gp, cp, es):
        if g == c:
            status.append("verified")
        elif pd.notna(s) and s >= threshold:
            status.append("likely_correct")
        else:
            status.append("needs_review")
    return pd.DataFrame({"complaint_id": m["complaint_id"], "status": status})
