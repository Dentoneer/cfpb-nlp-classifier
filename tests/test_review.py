import inspect

import pandas as pd
import pytest

from nlp.review import agree_path, count_verdicts, is_correct, review_rows, select_hand_check

WIDE = [
    "complaint_id",
    "narrative",
    "cfpb_product",
    "cfpb_issue",
    "cfpb_sub_issue",
    "gemini_product",
    "gemini_issue",
    "gemini_sub_issue",
    "gemini_summary",
    "verdict_product",
    "verdict_issue",
    "verdict_sub_issue",
    "summary_ok",
]


def _frames(n_agree=30, n_dis=30):
    comp, pred = [], []
    for i in range(n_agree + n_dis):
        cid = f"c{i}"
        comp.append(
            {
                "complaint_id": cid,
                "product": "P",
                "issue": "I",
                "sub_issue": "S",
                "narrative": f"text {i}",
            }
        )
        agree = i < n_agree
        pred.append(
            {
                "complaint_id": cid,
                "classifier": "gemini",
                "product": "P",
                "issue": "I" if agree else "I2",
                "sub_issue": "S" if agree else "S2",
                "summary": f"sum {i}",
                "confidence": 0.9,
            }
        )
    return pd.DataFrame(pred), pd.DataFrame(comp)


def test_selection_ignores_validator_outputs():
    assert list(inspect.signature(select_hand_check).parameters) == [
        "gemini_test",
        "complaints",
        "n",
        "seed",
    ]
    pred, comp = _frames()
    sel = select_hand_check(pred, comp, n=20, seed=1)
    assert list(sel.columns) == ["complaint_id", "stratum"]
    assert set(sel["stratum"]) == {"agree", "disagree"}
    assert (sel["stratum"] == "agree").sum() == 10
    assert (sel["stratum"] == "disagree").sum() == 10
    assert sel.equals(select_hand_check(pred, comp, n=20, seed=1))


def test_selection_few_disagreements_fills_with_agree():
    pred, comp = _frames(n_agree=50, n_dis=3)
    sel = select_hand_check(pred, comp, n=20, seed=1)
    assert len(sel) == 20
    assert (sel["stratum"] == "disagree").sum() == 3
    assert len(select_hand_check(pred, comp, n=500, seed=1)) == 53


def test_prefill_only_where_agree():
    pred, comp = _frames(2, 2)
    sel = pd.DataFrame({"complaint_id": ["c0", "c3"], "stratum": ["agree", "disagree"]})
    rows = review_rows(sel, pred, comp)
    assert list(rows.columns) == WIDE
    r0 = rows[rows.complaint_id == "c0"].iloc[0]
    assert (r0.verdict_product, r0.verdict_issue, r0.verdict_sub_issue) == ("both OK",) * 3
    r3 = rows[rows.complaint_id == "c3"].iloc[0]
    assert r3.verdict_product == "both OK"
    assert r3.verdict_issue == "" and r3.verdict_sub_issue == ""
    assert (rows["summary_ok"] == "").all()
    assert r3.narrative == "text 3" and r3.gemini_summary == "sum 3"


def test_counting_rules():
    assert is_correct("model right") is True
    assert is_correct("both OK") is True
    assert is_correct("label right") is False
    assert is_correct("both wrong") is False
    assert is_correct("ambiguous") is None
    for bad in ["", "yes", "Model right"]:
        with pytest.raises(ValueError):
            is_correct(bad)
    hr = pd.DataFrame(
        {
            "complaint_id": list("abcdef"),
            "level": ["product"] * 6,
            "verdict": ["model right", "both OK", "label right", "both wrong", "ambiguous", ""],
            "summary_ok": [""] * 6,
        }
    )
    out = count_verdicts(hr)
    assert list(out.columns) == ["level", "n_correct", "n_error", "n_ambiguous", "n_blank"]
    row = out[out.level == "product"].iloc[0]
    assert (row.n_correct, row.n_error, row.n_ambiguous, row.n_blank) == (2, 2, 1, 1)


def test_agree_path_requires_all_three_levels():
    comp = pd.DataFrame(
        {
            "complaint_id": ["a", "b", "c", "d"],
            "product": ["P"] * 4,
            "issue": ["I"] * 4,
            "sub_issue": ["S"] * 4,
            "narrative": ["t"] * 4,
        }
    )
    pred = pd.DataFrame(
        {
            "complaint_id": ["a", "b", "c", "d"],
            "product": ["P", "P", "P", "Q"],
            "issue": ["I", "I", "J", "J"],
            "sub_issue": ["S", "X", "S", "X"],
            "summary": ["s"] * 4,
        }
    )
    ap = agree_path(pred, comp)
    assert ap.to_dict() == {"a": True, "b": False, "c": False, "d": False}
    assert ap.mean() == 0.25
    sel = select_hand_check(pred, comp, n=4, seed=1)
    assert dict(zip(sel.complaint_id, sel.stratum)) == {
        "a": "agree",
        "b": "disagree",
        "c": "disagree",
        "d": "disagree",
    }
