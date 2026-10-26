import pandas as pd
import pytest

from nlp.pipeline import human_labels, review_weights


def _hr(rows):
    return pd.DataFrame(rows, columns=["complaint_id", "level", "verdict", "summary_ok"])


def test_human_labels_all_levels_and_exclusions():
    lv = ["product", "issue", "sub_issue"]
    rows = []
    for cid, verdicts in {
        "a": ["both OK", "model right", "both OK"],
        "b": ["both OK", "label right", "both OK"],
        "c": ["both OK", "ambiguous", "both OK"],
        "d": ["both OK", "", "both OK"],
        "e": ["both wrong", "both wrong", "both wrong"],
    }.items():
        rows += [(cid, level, v, "yes") for level, v in zip(lv, verdicts)]
    rows.append(("f", "product", "both OK", "yes"))  # missing levels
    out = human_labels(_hr(rows))
    assert out.to_dict() == {"a": True, "b": False, "e": False}
    assert out.index.name == "complaint_id" and out.dtype == bool


def test_review_weights_inverse_to_sampling():
    cols = ["product", "issue", "sub_issue"]
    ids = [f"c{i}" for i in range(10)]
    complaints = pd.DataFrame({"complaint_id": ids, **{c: "x" for c in cols}})
    gem = complaints.copy()
    for i in (0, 1):  # 2 of 10 disagree
        gem.loc[i, "product"] = "y"
    selection = pd.DataFrame(
        {"complaint_id": ["c0", "c1", "c5", "c6"], "stratum": ["disagree"] * 2 + ["agree"] * 2}
    )
    w = review_weights(gem, complaints, selection)
    assert list(w.index) == ["c0", "c1", "c5", "c6"]
    # population 80% agree / 20% disagree, sample 50/50 -> agree weight 4x disagree weight
    assert w["c5"] == pytest.approx(4 * w["c0"])
    assert w.sum() == pytest.approx(4)
