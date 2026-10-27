import numpy as np
import pandas as pd
import pytest

from nlp.classifiers.base import probs_long
from nlp.schema import VALIDATOR_RESULTS_COLS, path_id
from nlp.taxonomy import build_taxonomy
from nlp.validators import (
    v_cleanlab,
    v_hierarchy,
    v_label_match,
    v_second_opinion,
    v_self_confidence,
    v_self_consistency,
)

IDS = ["a", "b", "c", "d"]


def pred(products, issues, subs, conf=(0.9, 0.8, 0.4, 0.5)):
    return pd.DataFrame(
        {
            "complaint_id": IDS,
            "product": products,
            "issue": issues,
            "sub_issue": subs,
            "confidence": list(conf),
        }
    )


@pytest.fixture
def complaints():
    df = pred(["P1", "P1", "P2", "P2"], ["I1", "I2", "I3", "I3"], ["s1", "s2", "s3", "s4"])
    return df[["complaint_id", "product", "issue", "sub_issue"]]


@pytest.fixture
def gemini():
    # a: all right; b: product+issue right, sub wrong; c: product wrong; d: all right
    return pred(["P1", "P1", "P1", "P2"], ["I1", "I2", "I9", "I3"], ["s1", "sX", "s3", "s4"])


def get(df, cid, validator, level):
    r = df[(df.complaint_id == cid) & (df.validator == validator) & (df.level == level)]
    assert len(r) == 1, (cid, validator, level)
    return r.iloc[0]


def test_label_match(gemini, complaints):
    out = v_label_match(gemini, complaints)
    assert list(out.columns) == VALIDATOR_RESULTS_COLS
    assert len(out) == 16
    assert get(out, "b", "cfpb_label", "issue").verdict
    assert not get(out, "b", "cfpb_label", "sub_issue").verdict
    assert not get(out, "b", "cfpb_label", "path").verdict
    assert not get(out, "c", "cfpb_label", "product").verdict
    r = get(out, "a", "cfpb_label", "path")
    assert r.verdict and r.score == 1.0


def test_second_opinion(gemini):
    pa, pb = path_id("P1", "I1", "s1"), path_id("P1", "I2", "sX")
    pc, pd_ = path_id("P2", "I3", "s3"), path_id("P2", "I3", "s4")
    classes = [pa, pb, pc, pd_]
    probs = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],  # a: agrees
            [0.1, 0.2, 0.3, 0.4],  # b: other says P2/I3/s4
            [0.4, 0.4, 0.1, 0.1],  # c: gemini path P1/I9/s3 not in classes
            [0.1, 0.1, 0.2, 0.6],  # d: agrees
        ]
    )
    other = probs_long(IDS, classes, probs)
    out = v_second_opinion(gemini, other, "tfidf")
    assert set(out.validator) == {"tfidf"}
    a = get(out, "a", "tfidf", "path")
    assert a.verdict and a.score == pytest.approx(0.7)
    b = get(out, "b", "tfidf", "path")
    assert not b.verdict and b.score == pytest.approx(0.2)
    # product level aggregates mass over P1 paths
    assert get(out, "b", "tfidf", "product").score == pytest.approx(0.3)
    assert not get(out, "b", "tfidf", "product").verdict
    assert get(out, "c", "tfidf", "path").score == 0.0


def test_self_confidence(gemini):
    out = v_self_confidence(gemini)
    assert set(out.validator) == {"gemini_confidence"}
    assert len(out) == 16
    assert get(out, "a", "gemini_confidence", "issue").score == pytest.approx(0.9)
    assert not get(out, "c", "gemini_confidence", "path").verdict
    assert get(out, "d", "gemini_confidence", "path").verdict


def test_self_consistency(gemini):
    r2 = gemini.copy()
    r2.loc[r2.complaint_id == "a", "sub_issue"] = "other"  # differs at sub_issue/path
    r3 = gemini.copy()
    r3.loc[r3.complaint_id == "a", ["issue", "sub_issue"]] = ["Iz", "other"]
    out = v_self_consistency([gemini, r2, r3])
    assert set(out.validator) == {"self_consistency"}
    assert get(out, "a", "self_consistency", "product").score == pytest.approx(1.0)
    assert get(out, "a", "self_consistency", "issue").score == pytest.approx(2 / 3)
    p = get(out, "a", "self_consistency", "path")
    assert p.score == pytest.approx(1 / 3) and not p.verdict
    assert get(out, "b", "self_consistency", "path").verdict


def test_hierarchy(gemini, complaints):
    tax = build_taxonomy(complaints)
    out = v_hierarchy(gemini, tax)
    assert set(out.level) == {"path"}
    assert get(out, "a", "hierarchy", "path").verdict
    assert not get(out, "b", "hierarchy", "path").verdict  # P1/I2/sX not in taxonomy
    assert get(out, "b", "hierarchy", "path").score == 0.0


def test_cleanlab():
    classes = ["x", "y", "z"]
    rng = np.random.default_rng(0)
    train_labels = ["x"] * 30 + ["y"] * 30 + ["z"] * 30
    tp = np.vstack([np.tile(np.eye(3)[i], (30, 1)) for i in range(3)]) * 0.8 + 0.1
    tp = tp + rng.uniform(0, 0.01, tp.shape)
    tp = tp / tp.sum(axis=1, keepdims=True)
    # a: label ok, gemini agrees; b: label wrong (probs say y), gemini says y;
    # c: label ok (y), gemini differs; d: label wrong (probs say z), gemini agrees w/ label
    test_labels = ["x", "x", "y", "x"]
    test_probs = np.array(
        [[0.9, 0.05, 0.05], [0.02, 0.96, 0.02], [0.1, 0.8, 0.1], [0.02, 0.02, 0.96]]
    )
    out = v_cleanlab(
        ["x", "y", "x", "x"], classes, tp, train_labels, IDS, test_labels, test_probs
    )
    assert set(out.validator) == {"cleanlab"} and set(out.level) == {"path"}
    a = get(out, "a", "cleanlab", "path")  # agrees, label fine
    assert a.verdict and a.score > 0.9
    b = get(out, "b", "cleanlab", "path")  # disagrees, label very likely wrong
    assert b.verdict and b.score > 0.8
    c = get(out, "c", "cleanlab", "path")  # disagrees, label fine
    assert not c.verdict and c.score < 0.5
    d = get(out, "d", "cleanlab", "path")  # agrees with a likely-wrong label
    assert not d.verdict and d.score < 0.2


def test_cleanlab_skips_unseen_label():
    classes = ["x", "y"]
    tp = np.array([[0.9, 0.1]] * 5 + [[0.1, 0.9]] * 5)
    out = v_cleanlab(
        ["x", "q"], classes, tp, ["x"] * 5 + ["y"] * 5, ["a", "b"], ["x", "q"],
        np.array([[0.9, 0.1], [0.5, 0.5]]),
    )  # fmt: skip
    assert list(out.complaint_id) == ["a"]
