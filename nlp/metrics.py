"""Evaluation metrics: per-level accuracy/macro-F1 with bootstrap CIs, McNemar, reweighting.

Scoring rule: each of ``product``, ``issue`` and ``sub_issue`` is scored on its own column
value (an issue is correct iff the issue strings are equal, regardless of product), except
``path``, which is correct only when product, issue and sub_issue all match.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, precision_recall_fscore_support

from nlp.schema import LEVELS, path_id

DEFAULT_SEED = 20261026


def _labels(df: pd.DataFrame, level: str) -> pd.Series:
    if level == "path":
        return pd.Series(
            [path_id(p, i, s) for p, i, s in zip(df["product"], df["issue"], df["sub_issue"])],
            index=df.index,
        )
    return df[level].astype(str)


def _align(pred: pd.DataFrame, truth: pd.DataFrame, level: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) aligned on complaint_id (inner join)."""
    missing = set(pred["complaint_id"]) - set(truth["complaint_id"])
    if missing:
        raise ValueError(f"{len(missing)} predicted complaint_ids missing from truth")
    merged = pred.merge(truth, on="complaint_id", suffixes=("_pred", "_true"))
    p = merged[["product_pred", "issue_pred", "sub_issue_pred"]]
    t = merged[["product_true", "issue_true", "sub_issue_true"]]
    p.columns = t.columns = ["product", "issue", "sub_issue"]
    return _labels(t, level).to_numpy(), _labels(p, level).to_numpy()


def _fast_macro_f1(t: np.ndarray, p: np.ndarray, k: int) -> float:
    tp = np.bincount(t[t == p], minlength=k)
    ct = np.bincount(t, minlength=k)
    cp = np.bincount(p, minlength=k)
    present = (ct + cp) > 0
    denom = (ct + cp).astype(float)
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1[present].mean())


def level_scores(
    pred: pd.DataFrame, truth: pd.DataFrame, n_boot: int = 1000, seed: int = DEFAULT_SEED
) -> pd.DataFrame:
    """Accuracy and macro-F1 per level with percentile 95% bootstrap CIs."""
    rng = np.random.default_rng(seed)
    rows = []
    for level in LEVELS:
        y_true, y_pred = _align(pred, truth, level)
        classes, inv = np.unique(np.concatenate([y_true, y_pred]), return_inverse=True)
        t, p = inv[: len(y_true)], inv[len(y_true) :]
        n, k = len(t), len(classes)
        acc = float((t == p).mean())
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        accs = np.empty(n_boot)
        f1s = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            accs[b] = (t[idx] == p[idx]).mean()
            f1s[b] = _fast_macro_f1(t[idx], p[idx], k)
        for metric, value, boots in (("accuracy", acc, accs), ("macro_f1", f1, f1s)):
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append((level, metric, value, float(lo), float(hi)))
    return pd.DataFrame(rows, columns=["level", "metric", "value", "ci_low", "ci_high"])


def per_class_report(pred: pd.DataFrame, truth: pd.DataFrame, level: str) -> pd.DataFrame:
    y_true, y_pred = _align(pred, truth, level)
    labels = sorted(set(y_true) | set(y_pred))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {"label": labels, "precision": prec, "recall": rec, "f1": f1, "support": sup}
    )


def confusion(pred: pd.DataFrame, truth: pd.DataFrame, level: str) -> pd.DataFrame:
    """Counts with true labels as rows and predicted labels as columns."""
    y_true, y_pred = _align(pred, truth, level)
    labels = sorted(set(y_true) | set(y_pred))
    table = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred"))
    return table.reindex(index=labels, columns=labels, fill_value=0)


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[float, float]:
    """McNemar test on paired correctness; returns (statistic, p-value)."""
    a = np.asarray(correct_a, dtype=bool)
    b_ = np.asarray(correct_b, dtype=bool)
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    n = b + c
    if n == 0:
        return 0.0, 1.0
    if n < 25:
        return float(min(b, c)), float(stats.binomtest(min(b, c), n, 0.5).pvalue)
    stat = (abs(b - c) - 1) ** 2 / n
    return float(stat), float(stats.chi2.sf(stat, 1))


def inverse_weights(sample_strata: pd.Series, population_strata: pd.Series) -> pd.Series:
    """Weight = population share / sample share per stratum, normalized to sum to len(sample)."""
    pop_share = population_strata.value_counts(normalize=True)
    samp_share = sample_strata.value_counts(normalize=True)
    absent = set(samp_share.index) - set(pop_share.index)
    if absent:
        raise ValueError(f"sample strata absent from population: {sorted(map(str, absent))}")
    w = sample_strata.map(pop_share / samp_share).astype(float)
    return w * (len(sample_strata) / w.sum())


def weighted_accuracy(
    correct: np.ndarray, weights: np.ndarray, n_boot: int = 1000, seed: int = DEFAULT_SEED
) -> tuple[float, float, float]:
    c = np.asarray(correct, dtype=float)
    w = np.asarray(weights, dtype=float)
    value = float((c * w).sum() / w.sum())
    rng = np.random.default_rng(seed)
    n = len(c)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = (c[idx] * w[idx]).sum() / w[idx].sum()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return value, float(lo), float(hi)
