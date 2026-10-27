import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from nlp.ensemble import ensemble_learned, ensemble_simple, row_status


def res(cid, validator, level, score, verdict=None):
    return {
        "complaint_id": cid,
        "validator": validator,
        "level": level,
        "verdict": score >= 0.5 if verdict is None else verdict,
        "score": score,
    }


def test_ensemble_simple():
    r = pd.DataFrame(
        [
            res("a", "v1", "path", 1.0),
            res("a", "v2", "path", 0.9),
            res("a", "v3", "path", 0.1),
            res("a", "v1", "product", 1.0),
            res("b", "v1", "path", 0.0),
            res("b", "v2", "path", 0.2),
            res("a", "ensemble_simple", "path", 0.0),  # must be ignored
        ]
    )
    out = ensemble_simple(r)
    assert set(out.validator) == {"ensemble_simple"}
    a = out[(out.complaint_id == "a") & (out.level == "path")].iloc[0]
    assert a.score == pytest.approx(2 / 3) and a.verdict and a.dissent == "v3"
    b = out[(out.complaint_id == "b") & (out.level == "path")].iloc[0]
    assert b.score == 0.0 and not b.verdict and b.dissent == "v1,v2"
    assert out[(out.complaint_id == "a") & (out.level == "product")].iloc[0].dissent == ""


def _learned_inputs(n=40):
    rng = np.random.default_rng(1)
    ids = [f"c{i}" for i in range(n)]
    label = np.array([i % 2 == 0 for i in range(n)])
    rows = []
    for i, cid in enumerate(ids):
        rows.append(res(cid, "leak", "path", float(label[i])))  # equals the label
        rows.append(res(cid, "noise", "path", float(rng.uniform())))
    hand = pd.DataFrame({"complaint_id": ids[:30], "correct": label[:30]})
    return pd.DataFrame(rows), hand


def test_learned_ensemble_is_out_of_fold():
    results, hand = _learned_inputs()
    out = ensemble_learned(results, hand, n_splits=5)
    assert set(out.validator) == {"ensemble_learned"}
    assert len(out) == 40
    in_hand = out.complaint_id.isin(hand.complaint_id)
    h = out[in_hand]
    assert (h.fold >= 0).all() and (out[~in_hand].fold == -1).all()
    assert sorted(h.fold.unique()) == [0, 1, 2, 3, 4]
    piv = results.pivot(index="complaint_id", columns="validator", values="score")
    y = hand.set_index("complaint_id")["correct"]
    hand_ids = list(hand.complaint_id)
    for f in range(5):
        test_ids = list(h[h.fold == f].complaint_id)
        train_ids = [c for c in hand_ids if c not in test_ids]
        m = LogisticRegression(max_iter=1000).fit(piv.loc[train_ids], y.loc[train_ids])
        want = m.predict_proba(piv.loc[test_ids])[:, 1]
        got = h.set_index("complaint_id").loc[test_ids, "score"].to_numpy()
        assert got == pytest.approx(want)
    # non-hand rows come from a model fit on all hand rows
    m = LogisticRegression(max_iter=1000).fit(piv.loc[hand_ids], y.loc[hand_ids])
    rest = out[~in_hand].set_index("complaint_id")
    assert rest["score"].to_numpy() == pytest.approx(m.predict_proba(piv.loc[rest.index])[:, 1])


def test_learned_needs_both_classes():
    results, hand = _learned_inputs()
    hand["correct"] = True
    with pytest.raises(ValueError):
        ensemble_learned(results, hand)


def test_row_status_rules():
    gem = pd.DataFrame(
        {
            "complaint_id": list("abcd"),
            "product": "P",
            "issue": "I",
            "sub_issue": ["s", "s", "s", "s"],
        }
    )
    cfpb = gem.assign(sub_issue=["s", "x", "x", "x"])
    ens = pd.DataFrame(
        {
            "complaint_id": list("abc"),
            "validator": "ensemble_simple",
            "level": "path",
            "verdict": True,
            "score": [0.1, 0.6, 0.59],  # a verified regardless; b on threshold; c below
        }
    )
    out = row_status(gem, cfpb, ens)
    s = out.set_index("complaint_id")["status"]
    assert s["a"] == "verified"
    assert s["b"] == "likely_correct"
    assert s["c"] == "needs_review"
    assert s["d"] == "needs_review"  # no ensemble score
    assert row_status(gem, cfpb, ens, threshold=0.5).set_index("complaint_id")["status"]["c"] == (
        "likely_correct"
    )


# ---- final review fixes ------------------------------------------------------------------


def test_learned_ensemble_uses_sample_weights():
    results, hand = _learned_inputs()
    w = pd.Series(np.linspace(0.5, 3.0, len(hand)), index=hand.complaint_id)
    out = ensemble_learned(results, hand, n_splits=5, sample_weight=w)
    piv = results.pivot(index="complaint_id", columns="validator", values="score")
    y = hand.set_index("complaint_id")["correct"]
    hand_ids = list(hand.complaint_id)
    m = LogisticRegression(max_iter=1000).fit(
        piv.loc[hand_ids], y.loc[hand_ids], sample_weight=w.loc[hand_ids].to_numpy()
    )
    rest = out[~out.complaint_id.isin(hand_ids)].set_index("complaint_id")
    assert rest["score"].to_numpy() == pytest.approx(m.predict_proba(piv.loc[rest.index])[:, 1])
    unweighted = ensemble_learned(results, hand, n_splits=5)
    assert not np.allclose(unweighted["score"], out["score"])
