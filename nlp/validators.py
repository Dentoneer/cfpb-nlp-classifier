"""Validators: methods that judge whether Gemini's classification of each complaint is right.

Every function returns long-format `validator_results` rows
(complaint_id, validator, level, verdict, score); verdict True means "Gemini judged right",
a higher score means more likely right. A validator emits only the levels it covers.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from nlp.schema import LEVELS, VALIDATOR_RESULTS_COLS, path_id

_PATH_SEP = " || "
_KEYS = {
    "product": ["product"],
    "issue": ["product", "issue"],
    "sub_issue": ["sub_issue"],
    "path": ["product", "issue", "sub_issue"],
}
assert set(_KEYS) == set(LEVELS)


def _level_keys(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Per-level identity strings for the rows of a product/issue/sub_issue frame."""
    out = {}
    for level, cols in _KEYS.items():
        out[level] = df[cols[0]].astype(str)
        for c in cols[1:]:
            out[level] = out[level] + _PATH_SEP + df[c].astype(str)
    return out


def _frame(rows: list[pd.DataFrame]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=VALIDATOR_RESULTS_COLS)
    return pd.concat(rows, ignore_index=True)[VALIDATOR_RESULTS_COLS]


def _rows(ids, validator: str, level: str, verdict, score) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "complaint_id": list(ids),
            "validator": validator,
            "level": level,
            "verdict": np.asarray(verdict, dtype=bool),
            "score": np.asarray(score, dtype=float),
        }
    )


def v_label_match(gemini: pd.DataFrame, complaints: pd.DataFrame) -> pd.DataFrame:
    """`cfpb_label`: does Gemini agree with the CFPB label, per level (score 1.0/0.0)."""
    m = gemini[["complaint_id", "product", "issue", "sub_issue"]].merge(
        complaints[["complaint_id", "product", "issue", "sub_issue"]],
        on="complaint_id",
        suffixes=("", "_cfpb"),
    )
    g = m[["product", "issue", "sub_issue"]]
    c = m[["product_cfpb", "issue_cfpb", "sub_issue_cfpb"]].set_axis(g.columns, axis=1)
    gk, ck = _level_keys(g), _level_keys(c)
    parts = []
    for level in LEVELS:
        ok = (gk[level] == ck[level]).to_numpy()
        parts.append(_rows(m["complaint_id"], "cfpb_label", level, ok, ok.astype(float)))
    return _frame(parts)


def v_second_opinion(gemini: pd.DataFrame, other_probs: pd.DataFrame, name: str) -> pd.DataFrame:
    """Another classifier's view of Gemini's answer (`name` = tfidf, embed_lr, knn, modernbert).

    `other_probs` = complaint_id, path_id, prob. At path level the score is the probability of
    Gemini's path; at coarser levels it is the probability mass of paths sharing Gemini's
    product / (product, issue) / sub-issue. Verdict: the other classifier's top choice at that
    level equals Gemini's. Complaints absent from `other_probs` are skipped.
    """
    op = other_probs[["complaint_id", "path_id", "prob"]].copy()
    op[["product", "issue", "sub_issue"]] = op["path_id"].str.split(_PATH_SEP, n=2, expand=True, regex=False)
    keys = _level_keys(op)
    gem = gemini[["complaint_id", "product", "issue", "sub_issue"]]
    gk = _level_keys(gem)
    parts = []
    for level in LEVELS:
        mass = (
            op.assign(k=keys[level])
            .groupby(["complaint_id", "k"], sort=False)["prob"]
            .sum()
            .reset_index()
        )
        top = mass.loc[mass.groupby("complaint_id", sort=False)["prob"].idxmax()]
        top = top.set_index("complaint_id")["k"]
        look = mass.set_index(["complaint_id", "k"])["prob"]
        g = pd.DataFrame({"complaint_id": gem["complaint_id"], "k": gk[level]})
        g = g[g["complaint_id"].isin(top.index)]
        score = [float(look.get((c, k), 0.0)) for c, k in zip(g["complaint_id"], g["k"])]
        verdict = (g["complaint_id"].map(top) == g["k"]).to_numpy()
        parts.append(_rows(g["complaint_id"], name, level, verdict, score))
    return _frame(parts)


def v_self_confidence(gemini: pd.DataFrame) -> pd.DataFrame:
    """`gemini_confidence`: Gemini's verbalized confidence, same score at every level."""
    conf = gemini["confidence"].astype(float).to_numpy()
    return _frame(
        [
            _rows(gemini["complaint_id"], "gemini_confidence", lv, conf >= 0.5, conf)
            for lv in LEVELS
        ]
    )


def v_self_consistency(runs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """`self_consistency`: share of the runs agreeing with run 1, per level.

    Run 1 counts as agreeing with itself, so 3 runs give scores in {1/3, 2/3, 1}. Verdict:
    a majority of the runs agree (score >= 0.5). Only complaints present in every run.
    """
    if not runs:
        raise ValueError("need at least one run")
    cols = ["complaint_id", "product", "issue", "sub_issue"]
    base = runs[0][cols].drop_duplicates("complaint_id")
    ids = set(base["complaint_id"])
    for r in runs[1:]:
        ids &= set(r["complaint_id"])
    base = base[base["complaint_id"].isin(ids)].reset_index(drop=True)
    bk = _level_keys(base)
    agree = {lv: np.ones(len(base)) for lv in LEVELS}
    for r in runs[1:]:
        other = r[cols].drop_duplicates("complaint_id").set_index("complaint_id")
        other = other.loc[base["complaint_id"]].reset_index()
        ok = _level_keys(other)
        for lv in LEVELS:
            agree[lv] += (bk[lv] == ok[lv]).to_numpy()
    parts = []
    for lv in LEVELS:
        score = agree[lv] / len(runs)
        parts.append(_rows(base["complaint_id"], "self_consistency", lv, score >= 0.5, score))
    return _frame(parts)


def v_hierarchy(gemini: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    """`hierarchy` (path level only): is Gemini's path a real taxonomy path (1.0/0.0)."""
    valid = set(taxonomy["path_id"])
    ok = np.array(
        [
            path_id(p, i, s) in valid
            for p, i, s in zip(gemini["product"], gemini["issue"], gemini["sub_issue"])
        ]
    )
    return _frame([_rows(gemini["complaint_id"], "hierarchy", "path", ok, ok.astype(float))])


def v_cleanlab(
    gemini_paths: Sequence[str],
    classes: Sequence[str],
    train_probs_oof: np.ndarray,
    train_labels: Sequence[str],
    test_ids: Sequence[str],
    test_labels: Sequence[str],
    test_probs: np.ndarray,
) -> pd.DataFrame:
    """`cleanlab` (path level only): is the CFPB label itself suspect?

    `train_probs_oof` are out-of-fold TF-IDF probabilities on the train pool (columns follow
    `classes`; see `nlp.classifiers.base.oof_proba`). Per-class confident-learning thresholds
    come from the train pool. issue_probability of the CFPB label = 0 when the classifier's
    probability of it is >= its class threshold, else 1 - p(label) / threshold. The score is
    P(Gemini right): 1 - issue_probability when Gemini's path equals the CFPB label, else
    issue_probability; verdict = score >= 0.5. Complaints whose CFPB label is not in `classes`
    get no row. `gemini_paths`, `test_labels` and `test_probs` rows are aligned with `test_ids`.
    """
    from cleanlab.count import get_confident_thresholds

    classes = list(classes)
    col = {c: i for i, c in enumerate(classes)}
    train_probs_oof = np.asarray(train_probs_oof, dtype=float)
    test_probs = np.asarray(test_probs, dtype=float)
    keep = [i for i, lab in enumerate(train_labels) if lab in col]
    y = np.array([col[train_labels[i]] for i in keep])
    thr = np.asarray(get_confident_thresholds(y, train_probs_oof[keep]), dtype=float)
    valid = np.isfinite(thr) & (thr > 0)
    thr = np.where(valid, thr, thr[valid].mean() if valid.any() else 1.0)

    ids, scores = [], []
    for cid, gp, lab, p in zip(test_ids, gemini_paths, test_labels, test_probs):
        if lab not in col:
            continue  # label unseen in train: no cleanlab opinion
        j = col[lab]
        issue = 0.0 if p[j] >= thr[j] else float(min(1.0, max(0.0, 1.0 - p[j] / thr[j])))
        ids.append(cid)
        scores.append(1.0 - issue if gp == lab else issue)
    scores = np.array(scores)
    return _frame([_rows(ids, "cleanlab", "path", scores >= 0.5, scores)])
