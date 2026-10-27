import numpy as np
import pandas as pd
import pytest

from nlp.metrics import (
    confusion,
    inverse_weights,
    level_scores,
    mcnemar,
    per_class_report,
    weighted_accuracy,
)


def _frames():
    truth_p = ["A", "A", "A", "B", "B", "C"]
    pred_p = ["A", "A", "B", "B", "C", "C"]
    ids = [f"c{i}" for i in range(6)]
    truth = pd.DataFrame(
        {"complaint_id": ids, "product": truth_p, "issue": ["x"] * 6, "sub_issue": ["s"] * 6}
    )
    pred = pd.DataFrame(
        {"complaint_id": ids[::-1], "product": pred_p[::-1], "issue": ["x"] * 6,
         "sub_issue": ["s"] * 6}
    )
    return pred, truth


def test_accuracy_and_macro_f1_known_values():
    pred, truth = _frames()
    out = level_scores(pred, truth)
    assert list(out.columns) == ["level", "metric", "value", "ci_low", "ci_high"]
    assert set(out["level"]) == {"product", "issue", "sub_issue", "path"}
    row = lambda lvl, m: out[(out.level == lvl) & (out.metric == m)].iloc[0]
    assert row("product", "accuracy").value == pytest.approx(4 / 6)
    # A: 0.8, B: 0.5, C: 2/3
    assert row("product", "macro_f1").value == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
    assert row("issue", "accuracy").value == 1.0
    assert row("path", "accuracy").value == pytest.approx(4 / 6)
    for _, r in out.iterrows():
        assert r.ci_low <= r.value <= r.ci_high


def test_level_scores_deterministic_and_unknown_ids_raise():
    pred, truth = _frames()
    pd.testing.assert_frame_equal(level_scores(pred, truth), level_scores(pred, truth))
    bad = pred.copy()
    bad.loc[0, "complaint_id"] = "zzz"
    with pytest.raises(ValueError):
        level_scores(bad, truth)


def test_per_class_and_confusion():
    pred, truth = _frames()
    rep = per_class_report(pred, truth, "product").set_index("label")
    assert rep.loc["A", "precision"] == pytest.approx(1.0)
    assert rep.loc["A", "recall"] == pytest.approx(2 / 3)
    assert rep.loc["A", "support"] == 3
    conf = confusion(pred, truth, "product")
    assert conf.loc["A", "B"] == 1
    assert conf.loc["B", "B"] == 1
    assert conf.values.sum() == 6


def test_mcnemar_identical_is_p1():
    a = np.array([True, False, True, True])
    assert mcnemar(a, a.copy()) == (0.0, 1.0)


def test_mcnemar_known():
    a = np.array([True] * 10 + [True] * 5)
    b = np.array([False] * 10 + [True] * 5)
    stat, p = mcnemar(a, b)
    assert stat == 0.0
    assert p < 0.01


def test_mcnemar_large_uses_chi_square():
    a = np.array([True] * 40 + [False] * 10)
    b = np.array([False] * 40 + [True] * 10)
    stat, p = mcnemar(a, b)
    assert stat == pytest.approx((30 - 1) ** 2 / 50)
    assert p < 0.001


def test_inverse_weights_validation():
    with pytest.raises(ValueError):
        inverse_weights(pd.Series(["a", "z"]), pd.Series(["a", "b"]))


def test_reweighting_recovers_population():
    pop = pd.Series(["agree"] * 90 + ["disagree"] * 10)
    sample = pd.Series(["agree"] * 10 + ["disagree"] * 10)
    correct = np.array([True] * 10 + [True] * 5 + [False] * 5)
    w = inverse_weights(sample, pop)
    assert w.sum() == pytest.approx(20)
    val, lo, hi = weighted_accuracy(correct, w.to_numpy())
    assert val == pytest.approx(0.95, abs=1e-9)
    assert lo <= val <= hi
