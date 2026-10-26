"""Leaderboards, reweighted adjusted accuracy, and the go/no-go decision."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from nlp.metrics import DEFAULT_SEED, level_scores, mcnemar, weighted_accuracy
from nlp.review import is_correct
from nlp.schema import LEVELS

__all__ = ["adjusted_accuracy", "classifier_leaderboard", "go_no_go", "validator_leaderboard"]

GEMINI = "gemini"
TFIDF = "tfidf"
GO_PRODUCT_THRESHOLD = 0.85
GO_ALPHA = 0.05
_LEVELS3 = ["product", "issue", "sub_issue"]
_COST_KEYS = ["gemini_requests", "usd", "runtime_s"]


def _correct_at_level(pred: pd.DataFrame, truth: pd.DataFrame, level: str) -> pd.Series:
    """Bool Series indexed by complaint_id: prediction equals CFPB label at ``level``."""
    m = pred.merge(truth, on="complaint_id", suffixes=("_p", "_t"))
    if level == "path":
        ok = pd.Series(True, index=m.index)
        for lv in _LEVELS3:
            ok &= m[f"{lv}_p"].astype(str) == m[f"{lv}_t"].astype(str)
    else:
        ok = m[f"{level}_p"].astype(str) == m[f"{level}_t"].astype(str)
    ok.index = m["complaint_id"]
    return ok


def classifier_leaderboard(
    preds_all: pd.DataFrame, complaints: pd.DataFrame, test_ids
) -> pd.DataFrame:
    """Per classifier x level: accuracy / macro_f1 with CIs, plus McNemar p vs Gemini.

    Restricted to ``test_ids`` and to the ids every classifier has predicted, so all rows are
    scored on the same complaints (``n``). ``mcnemar_p`` is set on accuracy rows of every
    non-Gemini classifier; Gemini's own row and macro_f1 rows are NaN.
    """
    ids = set(test_ids)
    preds = preds_all[preds_all["complaint_id"].isin(ids)].drop_duplicates(
        ["classifier", "complaint_id"]
    )
    truth = complaints[["complaint_id", *_LEVELS3]].drop_duplicates("complaint_id")
    common: set | None = None
    for _, g in preds.groupby("classifier"):
        s = set(g["complaint_id"]) & set(truth["complaint_id"])
        common = s if common is None else common & s
    common = common or set()
    cols = ["classifier", "level", "metric", "value", "ci_low", "ci_high", "mcnemar_p", "n"]
    if not common:
        return pd.DataFrame(columns=cols)
    truth = truth[truth["complaint_id"].isin(common)]
    by_clf = {
        name: g[g["complaint_id"].isin(common)][["complaint_id", *_LEVELS3]]
        for name, g in preds.groupby("classifier")
    }
    gem_correct = (
        {lv: _correct_at_level(by_clf[GEMINI], truth, lv) for lv in LEVELS}
        if (GEMINI in by_clf)
        else {}
    )
    frames = []
    for name in sorted(by_clf):
        scores = level_scores(by_clf[name], truth)
        scores.insert(0, "classifier", name)
        scores["mcnemar_p"] = np.nan
        scores["n"] = len(common)
        if name != GEMINI and gem_correct:
            for lv in LEVELS:
                mine = _correct_at_level(by_clf[name], truth, lv)
                a = mine.reindex(gem_correct[lv].index)
                _, p = mcnemar(gem_correct[lv].to_numpy(), a.to_numpy())
                scores.loc[(scores.level == lv) & (scores.metric == "accuracy"), "mcnemar_p"] = p
        frames.append(scores)
    return pd.concat(frames, ignore_index=True)[cols]


def _level_correctness(human_review: pd.DataFrame) -> pd.DataFrame:
    """Wide frame indexed by complaint_id: per level True/False, or None (ambiguous/blank).

    Also returns ``blank_<level>`` flags. Path = correct at all three levels; any ambiguous or
    blank level makes path None.
    """
    rows: dict[str, dict] = {}
    for cid, lv, v in zip(
        human_review["complaint_id"], human_review["level"], human_review["verdict"]
    ):
        if lv not in _LEVELS3:
            continue
        r = rows.setdefault(cid, {})
        r[f"blank_{lv}"] = v == ""
        r[lv] = None if v == "" else is_correct(v)
    out = pd.DataFrame.from_dict(rows, orient="index").astype(object)
    out.index.name = "complaint_id"
    for lv in _LEVELS3:
        if lv not in out:
            out[lv] = None
        if f"blank_{lv}" not in out:
            out[f"blank_{lv}"] = True
        out[f"blank_{lv}"] = out[f"blank_{lv}"].fillna(True).astype(bool)
    path = []
    for _, r in out.iterrows():
        vals = [r[lv] for lv in _LEVELS3]
        path.append(None if any(v is None or pd.isna(v) for v in vals) else all(vals))
    out["path"] = path
    out["blank_path"] = out[[f"blank_{lv}" for lv in _LEVELS3]].any(axis=1)
    return out


def adjusted_accuracy(
    human_review: pd.DataFrame,
    selection: pd.DataFrame,
    test_agree_rate: float,
    n_boot: int = 1000,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Gemini accuracy per level from Marco's verdicts, reweighted to the test population.

    Strata are ``agree`` / ``disagree`` (``test_agree_rate`` = population share of agree).
    Weights = population share / share among the rows used at that level.
    """
    pop = {"agree": float(test_agree_rate), "disagree": 1.0 - float(test_agree_rate)}
    corr = _level_correctness(human_review)
    sel = selection[["complaint_id", "stratum"]].drop_duplicates("complaint_id")
    sel = sel.set_index("complaint_id")["stratum"]
    rows = []
    for lv in LEVELS:
        val = corr[lv].reindex(sel.index)
        blank = corr[f"blank_{lv}"].reindex(sel.index).fillna(True).astype(bool)
        amb = val.isna() & ~blank
        used = val.notna()
        n_used = int(used.sum())
        if n_used == 0:
            rows.append((lv, np.nan, np.nan, np.nan, 0, int(amb.sum()), int(blank.sum())))
            continue
        strata = sel[used]
        share = strata.value_counts(normalize=True)
        w = strata.map({k: pop[k] / share[k] for k in share.index}).astype(float)
        if w.sum() == 0:
            w = pd.Series(1.0, index=strata.index)
        v, lo, hi = weighted_accuracy(
            val[used].astype(bool).to_numpy(), w.to_numpy(), n_boot=n_boot, seed=seed
        )
        rows.append((lv, v, lo, hi, n_used, int(amb.sum()), int(blank.sum())))
    return pd.DataFrame(
        rows,
        columns=["level", "value", "ci_low", "ci_high", "n_used", "n_ambiguous", "n_blank"],
    )


def _wavg(mask_num: np.ndarray, mask_den: np.ndarray, w: np.ndarray) -> float:
    den = float(w[mask_den].sum())
    return float(w[mask_num & mask_den].sum() / den) if den > 0 else float("nan")


def validator_leaderboard(
    validator_results: pd.DataFrame,
    human_review: pd.DataFrame,
    weights: pd.Series,
    costs: dict[str, dict],
) -> pd.DataFrame:
    """Per validator x level: weighted catch rate, false-alarm rate, AUROC, plus costs.

    Truth = Gemini correct per Marco's verdict (``path`` = correct at all three levels).
    Rows with ambiguous/blank truth are excluded at that level. Sorted by path-level AUROC
    (descending, NaN last); levels keep the product/issue/sub_issue/path order.
    """
    corr = _level_correctness(human_review)
    rows = []
    for validator in sorted(validator_results["validator"].unique()):
        vdf = validator_results[validator_results["validator"] == validator]
        cost = costs.get(validator, {})
        for lv in LEVELS:
            g = vdf[vdf["level"] == lv].drop_duplicates("complaint_id").set_index("complaint_id")
            truth = corr[lv].reindex(g.index)
            w = weights.reindex(g.index)
            keep = truth.notna() & w.notna()
            g, truth, w = g[keep], truth[keep].astype(bool), w[keep].astype(float)
            t = truth.to_numpy()
            flagged = ~g["verdict"].astype(bool).to_numpy()
            wv = w.to_numpy()
            catch = _wavg(flagged, ~t, wv)
            false_alarm = _wavg(t, flagged, wv)
            if len(g) and len(set(t)) == 2:
                auroc = float(roc_auc_score(t, g["score"].to_numpy(), sample_weight=wv))
            else:
                auroc = float("nan")
            row = {
                "validator": validator,
                "level": lv,
                "n": len(g),
                "catch_rate": catch,
                "false_alarm_rate": false_alarm,
                "auroc": auroc,
            }
            for k in _COST_KEYS:
                row[k] = float(cost[k]) if k in cost else float("nan")
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    path_auc = out[out.level == "path"].set_index("validator")["auroc"]
    out["_k"] = out["validator"].map(path_auc).fillna(-1.0)
    out["_l"] = out["level"].map({lv: i for i, lv in enumerate(LEVELS)})
    out = out.sort_values(["_k", "validator", "_l"], ascending=[False, True, True])
    return out.drop(columns=["_k", "_l"]).reset_index(drop=True)


def _val(board: pd.DataFrame, clf: str, level: str, metric: str, col: str = "value") -> float:
    sel = board[(board.classifier == clf) & (board.level == level) & (board.metric == metric)]
    return float(sel[col].iloc[0]) if len(sel) else float("nan")


def go_no_go(clf_board: pd.DataFrame, adjusted: pd.DataFrame) -> dict:
    """Apply the Phase 1 rule; every failed condition is listed in plain words."""
    reasons: list[str] = []
    prod = adjusted.loc[adjusted["level"] == "product", "value"]
    prod_v = float(prod.iloc[0]) if len(prod) else float("nan")
    if not prod_v >= GO_PRODUCT_THRESHOLD:
        reasons.append(
            f"Adjusted product accuracy is {prod_v:.2f}, below the {GO_PRODUCT_THRESHOLD:.2f} bar."
        )
    if not (clf_board["classifier"] == TFIDF).any():
        reasons.append("No tfidf baseline rows on the classifier leaderboard; cannot compare.")
    elif not (clf_board["classifier"] == GEMINI).any():
        reasons.append("No gemini rows on the classifier leaderboard; cannot compare.")
    else:
        for lv in _LEVELS3:
            g_acc = _val(clf_board, GEMINI, lv, "accuracy")
            t_acc = _val(clf_board, TFIDF, lv, "accuracy")
            g_f1 = _val(clf_board, GEMINI, lv, "macro_f1")
            t_f1 = _val(clf_board, TFIDF, lv, "macro_f1")
            p = _val(clf_board, TFIDF, lv, "accuracy", "mcnemar_p")
            if not g_acc > t_acc:
                reasons.append(
                    f"At {lv}, Gemini accuracy {g_acc:.3f} does not beat TF-IDF {t_acc:.3f}."
                )
            if not g_f1 > t_f1:
                reasons.append(
                    f"At {lv}, Gemini macro-F1 {g_f1:.3f} does not beat TF-IDF {t_f1:.3f}."
                )
            if not p < GO_ALPHA:
                reasons.append(
                    f"At {lv}, McNemar p = {p:.3g} is not below {GO_ALPHA} "
                    "(Gemini vs TF-IDF accuracy difference not significant)."
                )
    if reasons:
        return {"go": False, "reasons": reasons}
    return {
        "go": True,
        "reasons": [
            (
                f"Adjusted product accuracy {prod_v:.2f} >= {GO_PRODUCT_THRESHOLD:.2f}, and Gemini "
                "beats TF-IDF on accuracy and macro-F1 at product, issue and sub_issue "
                f"(McNemar p < {GO_ALPHA} at each)."
            )
        ],
    }
