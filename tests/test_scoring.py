import copy
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from nlp.scoring import (
    adjusted_accuracy,
    classifier_leaderboard,
    go_no_go,
    validator_leaderboard,
)

LEVELS3 = ["product", "issue", "sub_issue"]


def _board(gemini=None, tfidf=None, p=None):
    """Classifier board with Gemini/TF-IDF rows at the three levels."""
    g = {"accuracy": 0.9, "macro_f1": 0.8}
    t = {"accuracy": 0.8, "macro_f1": 0.7}
    g.update(gemini or {})
    t.update(tfidf or {})
    rows = []
    for clf, vals in (("gemini", g), ("tfidf", t)):
        for lv in LEVELS3:
            for metric, v in vals.items():
                pv = float("nan")
                if clf == "tfidf" and metric == "accuracy":
                    pv = (p or {}).get(lv, 0.01)
                rows.append((clf, lv, metric, v, v - 0.05, v + 0.05, pv))
    return pd.DataFrame(
        rows,
        columns=["classifier", "level", "metric", "value", "ci_low", "ci_high", "mcnemar_p"],
    )


def _adj(product=0.9):
    return pd.DataFrame(
        {
            "level": ["product", "issue", "sub_issue", "path"],
            "value": [product, 0.8, 0.7, 0.6],
            "ci_low": 0.5,
            "ci_high": 0.95,
            "n_used": 100,
            "n_ambiguous": 0,
            "n_blank": 0,
        }
    )


def test_go_requires_all_conditions():
    res = go_no_go(_board(), _adj(0.9))
    assert res["go"] is True and res["reasons"]

    res = go_no_go(_board(), _adj(0.84))
    assert res["go"] is False
    assert any("0.84" in r and "product" in r.lower() for r in res["reasons"])

    # TF-IDF has higher macro-F1 at issue only
    board = _board()
    mask = (board.classifier == "tfidf") & (board.level == "issue") & (board.metric == "macro_f1")
    board.loc[mask, "value"] = 0.95
    res = go_no_go(board, _adj(0.9))
    assert res["go"] is False
    assert len(res["reasons"]) == 1 and "issue" in res["reasons"][0]

    res = go_no_go(_board(p={"sub_issue": 0.2}), _adj(0.9))
    assert res["go"] is False
    assert any("sub_issue" in r and "0.2" in r for r in res["reasons"])


def test_go_no_go_missing_tfidf_and_boundary():
    board = _board()
    res = go_no_go(board[board.classifier != "tfidf"], _adj(0.9))
    assert res["go"] is False and any("tfidf" in r.lower() for r in res["reasons"])
    assert go_no_go(_board(), _adj(0.85))["go"] is True


def _vr_inputs(weights=None):
    ids = [f"c{i}" for i in range(6)]
    truth = [True, True, True, False, False, False]
    verdicts = [True, True, False, False, False, True]
    hr = pd.DataFrame(
        {
            "complaint_id": ids,
            "level": "product",
            "verdict": ["both OK" if t else "both wrong" for t in truth],
            "summary_ok": "yes",
        }
    )
    vr = pd.DataFrame(
        {
            "complaint_id": ids,
            "validator": "v1",
            "level": "product",
            "verdict": verdicts,
            "score": [0.9, 0.8, 0.4, 0.3, 0.2, 0.6],
        }
    )
    return vr, hr, pd.Series(1.0, index=ids)


def test_catch_rate_and_false_alarm_known():
    vr, hr, w = _vr_inputs()
    costs = {"v1": {"gemini_requests": 3, "usd": 0.5, "runtime_s": 12.0}}
    out = validator_leaderboard(vr, hr, w, costs)
    row = out[(out.validator == "v1") & (out.level == "product")].iloc[0]
    assert row.catch_rate == pytest.approx(2 / 3)
    assert row.false_alarm_rate == pytest.approx(1 / 3)
    assert row.gemini_requests == 3 and row.usd == 0.5 and row.runtime_s == 12.0

    # weight c5 (a missed error) double
    w2 = w.copy()
    w2["c5"] = 2.0
    row = validator_leaderboard(vr, hr, w2, costs).query("level == 'product'").iloc[0]
    assert row.catch_rate == pytest.approx(2 / 4)

    # missing costs -> NaN
    row = validator_leaderboard(vr, hr, w, {}).query("level == 'product'").iloc[0]
    assert math.isnan(row.usd) and math.isnan(row.runtime_s)


def test_auroc_weighted():
    vr, hr, w = _vr_inputs()
    w2 = w.copy()
    w2["c2"] = 3.0
    w2["c3"] = 0.5
    truth = np.array([True, True, True, False, False, False])
    expected = roc_auc_score(truth, vr["score"], sample_weight=w2.to_numpy())
    row = validator_leaderboard(vr, hr, w2, {}).query("level == 'product'").iloc[0]
    assert row.auroc == pytest.approx(expected)

    # single class -> NaN, no raise
    hr1 = hr.copy()
    hr1["verdict"] = "both OK"
    row = validator_leaderboard(vr, hr1, w, {}).query("level == 'product'").iloc[0]
    assert math.isnan(row.auroc)


def test_validator_path_level_and_exclusions_and_sort():
    ids = ["a", "b", "c", "d"]
    hr_rows = []
    verd = {
        "a": ("both OK", "both OK", "both OK"),
        "b": ("both OK", "both wrong", "both OK"),
        "c": ("both OK", "ambiguous", "both OK"),
        "d": ("both OK", "both OK", ""),
    }
    for cid, vs in verd.items():
        for lv, v in zip(LEVELS3, vs):
            hr_rows.append((cid, lv, v, "yes"))
    hr = pd.DataFrame(hr_rows, columns=["complaint_id", "level", "verdict", "summary_ok"])
    recs = []
    for name, scores in (("bad", [0.1, 0.9, 0.5, 0.5]), ("good", [0.9, 0.1, 0.5, 0.5])):
        for lv in ["product", "issue", "sub_issue", "path"]:
            for cid, s in zip(ids, scores):
                recs.append((cid, name, lv, s > 0.4, s))
    vr = pd.DataFrame(recs, columns=["complaint_id", "validator", "level", "verdict", "score"])
    w = pd.Series(1.0, index=ids)
    out = validator_leaderboard(vr, hr, w, {})
    # path truth only for a (True) and b (False): c ambiguous, d blank excluded
    assert out.query("validator == 'good' and level == 'path'").iloc[0].auroc == 1.0
    assert out.query("validator == 'bad' and level == 'path'").iloc[0].auroc == 0.0
    # sorted by path-level AUROC descending: good rows first
    assert list(out.validator[:4]) == ["good"] * 4


def _human(sel_ids, verdict_by_id):
    rows = []
    for cid in sel_ids:
        for lv in LEVELS3:
            rows.append((cid, lv, verdict_by_id[cid], "yes"))
    return pd.DataFrame(rows, columns=["complaint_id", "level", "verdict", "summary_ok"])


def test_adjusted_accuracy_reweights_strata():
    # 10 agree (all correct), 10 disagree (all wrong); population is 80% agree
    sel = pd.DataFrame(
        {
            "complaint_id": [f"a{i}" for i in range(10)] + [f"d{i}" for i in range(10)],
            "stratum": ["agree"] * 10 + ["disagree"] * 10,
        }
    )
    v = {c: ("both OK" if c.startswith("a") else "both wrong") for c in sel.complaint_id}
    out = adjusted_accuracy(_human(sel.complaint_id, v), sel, 0.8).set_index("level")
    assert list(out.index) == ["product", "issue", "sub_issue", "path"]
    for lv in out.index:
        assert out.loc[lv, "value"] == pytest.approx(0.8)
        assert out.loc[lv, "n_used"] == 20
        assert out.loc[lv, "ci_low"] <= 0.8 <= out.loc[lv, "ci_high"]


def test_adjusted_accuracy_counts_ambiguous_and_blank():
    sel = pd.DataFrame(
        {"complaint_id": ["a0", "a1", "a2", "d0"], "stratum": ["agree"] * 3 + ["disagree"]}
    )
    verdicts = {"a0": "both OK", "a1": "ambiguous", "a2": "", "d0": "label right"}
    out = adjusted_accuracy(_human(sel.complaint_id, verdicts), sel, 0.5).set_index("level")
    for lv in out.index:
        assert out.loc[lv, "n_used"] == 2
        assert out.loc[lv, "n_ambiguous"] == 1
        assert out.loc[lv, "n_blank"] == 1
    # a0 correct (agree), d0 wrong (disagree); equal pop shares -> 0.5
    assert out.loc["product", "value"] == pytest.approx(0.5)


def _preds(clf, correct_ids, all_ids):
    rows = []
    for cid in all_ids:
        ok = cid in correct_ids
        rows.append(
            {
                "complaint_id": cid,
                "classifier": clf,
                "product": "P" if ok else "Q",
                "issue": "I" if ok else "J",
                "sub_issue": "S" if ok else "T",
            }
        )
    return pd.DataFrame(rows)


def _comp(ids):
    return pd.DataFrame({"complaint_id": ids, "product": "P", "issue": "I", "sub_issue": "S"})


def test_classifier_leaderboard_basic():
    ids = [f"c{i}" for i in range(40)]
    g = _preds("gemini", set(ids[:36]), ids)
    t = _preds("tfidf", set(ids[:20]), ids)
    extra = _preds("tfidf", set(), ["zz_train"])  # outside test_ids
    board = classifier_leaderboard(pd.concat([g, t, extra]), _comp(ids), ids)
    acc = board[board.metric == "accuracy"].set_index(["classifier", "level"])
    assert acc.loc[("gemini", "product"), "value"] == pytest.approx(0.9)
    assert acc.loc[("tfidf", "product"), "value"] == pytest.approx(0.5)
    assert math.isnan(acc.loc[("gemini", "product"), "mcnemar_p"])
    assert acc.loc[("tfidf", "product"), "mcnemar_p"] < 0.05
    assert {"path"} <= set(board.level)
    assert {"ci_low", "ci_high", "mcnemar_p"} <= set(board.columns)


def test_classifier_leaderboard_uses_intersection_of_ids():
    ids = [f"c{i}" for i in range(40)]
    g = _preds("gemini", set(ids), ids)
    t = _preds("tfidf", set(ids[:10]), ids[:20])  # predicted only 20
    with pytest.warns(UserWarning, match="tfidf"):
        board = classifier_leaderboard(pd.concat([g, t]), _comp(ids), ids)
    assert (board["n"] == 20).all()
    assert (board["coverage"] == 0.5).all()
    acc = board[(board.metric == "accuracy") & (board.level == "product")].set_index("classifier")
    assert acc.loc["tfidf", "value"] == pytest.approx(0.5)
    assert acc.loc["gemini", "value"] == pytest.approx(1.0)


def test_inputs_not_mutated():
    b = _board()
    before = copy.deepcopy(b)
    go_no_go(b, _adj())
    pd.testing.assert_frame_equal(b, before)


def test_zero_overlap_raises_naming_classifier():
    ids = [f"c{i}" for i in range(10)]
    g = _preds("gemini", set(ids), ids[:5])
    t = _preds("tfidf", set(ids), ids[5:])
    with pytest.raises(ValueError, match="gemini.*tfidf|tfidf.*gemini"):
        classifier_leaderboard(pd.concat([g, t]), _comp(ids), ids)


def test_full_coverage_no_warning():
    import warnings

    ids = [f"c{i}" for i in range(20)]
    g = _preds("gemini", set(ids), ids)
    t = _preds("tfidf", set(ids[:5]), ids)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        board = classifier_leaderboard(pd.concat([g, t]), _comp(ids), ids)
    assert (board["coverage"] == 1.0).all()


def test_go_no_go_low_coverage_reason():
    board = _board()
    board["n"] = 20
    board["coverage"] = 0.5
    res = go_no_go(board, _adj(0.9))
    assert res["go"] is False
    assert any("Only 20 of 40" in r and "50%" in r for r in res["reasons"])
    board["coverage"] = 1.0
    assert go_no_go(board, _adj(0.9))["go"] is True
