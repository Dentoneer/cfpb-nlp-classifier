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


# ---- final review fixes ------------------------------------------------------------------


def _cleanlab_inputs():
    from nlp.classifiers.base import probs_long
    from nlp.schema import path_id

    paths = [("P1", "I1", "S1"), ("P2", "I2", "S2")]
    pids = [path_id(*p) for p in paths]
    words = ["alpha beta gamma", "delta epsilon zeta"]
    train = pd.DataFrame(
        {
            "complaint_id": [f"t{i}" for i in range(20)],
            "narrative": [f"{words[i % 2]} row{i}" for i in range(20)],
            "path_id": [pids[i % 2] for i in range(20)],
        }
    )
    test_ids = [f"x{i}" for i in range(6)]
    test = pd.DataFrame(
        {
            "complaint_id": test_ids,
            "narrative": [words[i % 2] for i in range(6)],
            "path_id": [pids[i % 2] for i in range(6)],
        }
    )
    probs = probs_long(test_ids, pids, [[0.9, 0.1] if i % 2 == 0 else [0.2, 0.8] for i in range(6)])
    gem = pd.DataFrame(
        {
            "complaint_id": test_ids,
            "product": [paths[(i + (i == 3)) % 2][0] for i in range(6)],
            "issue": [paths[(i + (i == 3)) % 2][1] for i in range(6)],
            "sub_issue": [paths[(i + (i == 3)) % 2][2] for i in range(6)],
        }
    )
    return train, test, gem, probs


def test_cleanlab_validator_tolerates_missing_gemini_rows(tmp_cfg):
    from nlp.classifiers.tfidf import TfidfLR
    from nlp.pipeline import cleanlab_validator

    train, test, gem, probs = _cleanlab_inputs()
    full = cleanlab_validator(tmp_cfg, train, test, gem, probs, TfidfLR, n_splits=2)
    part = cleanlab_validator(
        tmp_cfg, train, test, gem[gem["complaint_id"] != "x2"], probs, TfidfLR, n_splits=2
    )
    assert "x2" not in set(part["complaint_id"])
    assert sorted(part["complaint_id"]) == sorted(set(full["complaint_id"]) - {"x2"})
    # labels / probabilities stay aligned with the remaining ids
    pd.testing.assert_frame_equal(
        part.set_index("complaint_id").sort_index(),
        full[full["complaint_id"] != "x2"].set_index("complaint_id").sort_index(),
    )


def test_summary_quality_rates_and_agreement():
    from nlp.pipeline import summary_quality

    flags = pd.DataFrame(
        {
            "complaint_id": ["a", "b", "c", "d"],
            "faithful": [True, True, False, True],
            "covers_main_issue": [True, False, True, True],
            "invented_details": [False, False, True, False],
        }
    )
    lv = ["product", "issue", "sub_issue"]
    hr = _hr(
        [(c, level, "both OK", s) for c, s in [("a", "yes"), ("b", "no"), ("c", "no"), ("e", "yes")]
         for level in lv]
        + [("d", level, "both OK", "") for level in lv]
    )  # fmt: skip
    q = summary_quality(flags, hr).set_index("measure")
    assert q.loc["judge: faithful", "rate"] == pytest.approx(0.75)
    assert q.loc["judge: faithful", "n"] == 4
    assert q.loc["judge: covers main issue", "rate"] == pytest.approx(0.75)
    assert q.loc["judge: no invented details", "rate"] == pytest.approx(0.75)
    assert q.loc["Marco: summary OK (yes)", "rate"] == pytest.approx(0.5)
    assert q.loc["Marco: summary OK (yes)", "n"] == 4  # a, b, c, e (d blank)
    # judge OK = faithful & covers & not invented: a yes, b no, c no; human a yes, b no, c no
    assert q.loc["judge-human agreement", "rate"] == pytest.approx(1.0)
    assert q.loc["judge-human agreement", "n"] == 3


def test_run_scoring_reports_summary_quality(tmp_cfg, monkeypatch):
    import nlp.pipeline as P

    for name in ("classifier_leaderboard", "adjusted_accuracy", "validator_leaderboard"):
        monkeypatch.setattr(P, name, lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(P, "go_no_go", lambda *a: {"go": False, "reasons": []})
    monkeypatch.setattr(P, "agree_path", lambda *a: pd.Series([True]))
    monkeypatch.setattr(P, "review_weights", lambda *a: pd.Series(dtype=float))
    flags = pd.DataFrame(
        {"complaint_id": ["a"], "faithful": [True], "covers_main_issue": [True],
         "invented_details": [False]}
    )  # fmt: skip
    hr = _hr([("a", lv, "both OK", "yes") for lv in ("product", "issue", "sub_issue")])
    scored = P.run_scoring(
        tmp_cfg, pd.DataFrame(), pd.DataFrame(), [], pd.DataFrame(), hr, pd.DataFrame(),
        pd.DataFrame(), summary_flags=flags,
    )  # fmt: skip
    assert "judge: faithful" in set(scored["summary_quality"]["measure"])
    assert "summary_quality" not in P.run_scoring(
        tmp_cfg, pd.DataFrame(), pd.DataFrame(), [], pd.DataFrame(), hr, pd.DataFrame(),
        pd.DataFrame(),
    )


def test_run_ensembles_passes_review_weights(monkeypatch):
    import nlp.pipeline as P

    seen = {}

    def fake_learned(vres, labels, sample_weight=None):
        seen["w"] = sample_weight
        return pd.DataFrame()

    monkeypatch.setattr(P, "ensemble_learned", fake_learned)
    monkeypatch.setattr(P, "ensemble_simple", lambda v: pd.DataFrame())
    hr = _hr([("a", lv, "both OK", "yes") for lv in ("product", "issue", "sub_issue")])
    w = pd.Series([2.0], index=pd.Index(["a"], name="complaint_id"))
    P.run_ensembles(pd.DataFrame(), hr, weights=w)
    assert seen["w"] is w
    P.run_ensembles(pd.DataFrame(), hr)
    assert seen["w"] is None
