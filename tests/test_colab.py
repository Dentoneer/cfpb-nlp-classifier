import json
from pathlib import Path

import pandas as pd
import pytest

from nlp import colab, store
from nlp.budget import BudgetExceeded, DailyQuotaExhausted, Ledger
from nlp.config import Config
from nlp.taxonomy import build_taxonomy

CONFIG = Path(__file__).resolve().parent.parent / "config" / "phase1.yaml"


def test_load_config_overrides_artifacts_dir(tmp_path):
    cfg = colab.load_config(CONFIG, tmp_path / "phase1")
    assert cfg.artifacts_dir == tmp_path / "phase1"
    assert (tmp_path / "phase1").is_dir()
    assert cfg.spend_cap_usd == 4.00


def test_overrides_roundtrip_and_validation(tmp_path):
    cfg = colab.load_config(CONFIG, tmp_path)
    colab.save_overrides(cfg, classifier_model="gemini-3.8-flash", batch_size=25)
    again = colab.load_config(CONFIG, tmp_path)
    assert again.classifier_model == "gemini-3.8-flash" and again.batch_size == 25
    assert again.judge_model == cfg.judge_model
    with pytest.raises(ValueError):
        colab.save_overrides(cfg, not_a_field=1)
    with pytest.raises(ValueError, match="judge"):
        colab.save_overrides(cfg, classifier_model=cfg.judge_model)


def test_publish_once_reuses_url(tmp_path):
    cfg = colab.load_config(CONFIG, tmp_path)
    calls = []
    url = colab.publish_once(cfg, "taxonomy", lambda: calls.append(1) or "https://x/1")
    assert colab.publish_once(cfg, "taxonomy", lambda: "https://x/2") == url == "https://x/1"
    assert colab.read_url(cfg, "taxonomy") == url and len(calls) == 1
    with pytest.raises(colab.StageIncomplete):
        colab.read_url(cfg, "nothing")


def _cfg(tmp_path) -> Config:
    return colab.load_config(CONFIG, tmp_path)


def test_stage_paid_writes_only_complete_results(tmp_path):
    cfg = _cfg(tmp_path)
    part = pd.DataFrame({"complaint_id": ["a"]})
    full = pd.DataFrame({"complaint_id": ["a", "b"]})
    with pytest.raises(colab.StageIncomplete, match="1 of 2"):
        colab.stage_paid(cfg, "s", lambda: part, ["a", "b"])
    assert not store.exists(cfg, "s")
    assert len(colab.stage_paid(cfg, "s", lambda: full, ["a", "b"])) == 2
    assert store.exists(cfg, "s")


def test_stage_paid_translates_quota_and_budget(tmp_path):
    cfg = _cfg(tmp_path)

    def quota():
        raise DailyQuotaExhausted(7)

    def budget():
        raise BudgetExceeded("cap")

    with pytest.raises(colab.StageIncomplete, match="tomorrow"):
        colab.stage_paid(cfg, "q", quota, ["a"])
    with pytest.raises(colab.StageIncomplete, match="Budget reached"):
        colab.stage_paid(cfg, "b", budget, ["a"])


def test_always_write_refreshes(tmp_path):
    cfg = _cfg(tmp_path)
    w = colab.always_write(cfg)
    w("t", lambda: pd.DataFrame({"x": [1]}))
    w("t", lambda: pd.DataFrame({"x": [2]}))
    assert store.read(cfg, "t")["x"].tolist() == [2]


def _tax_rows():
    return pd.DataFrame(
        {
            "complaint_id": ["1", "2", "3", "4"],
            "product": ["P1", "P1", "P2", "P2"],
            "issue": ["I1", "I2", "I3", "I3"],
            "sub_issue": ["(none)"] * 4,
            "narrative": ["some text " * 20] * 4,
        }
    )


def test_accuracy_by_level():
    truth = _tax_rows()
    pred = truth.copy()
    pred.loc[1, "issue"] = "I1"  # wrong issue, right product
    pred.loc[2, "product"] = "P1"
    acc = colab.accuracy_by_level(pred, truth)
    assert acc == {"product": 0.75, "issue": 0.75, "sub_issue": 1.0, "path": 0.5}


def test_estimate_scales_with_runs_and_model(tmp_path):
    cfg = _cfg(tmp_path)
    rows = _tax_rows()
    tax = build_taxonomy(rows)
    one = colab.estimate_classification(cfg, rows, tax)
    assert one > 0
    assert colab.estimate_classification(cfg, rows, tax, runs=3) == pytest.approx(3 * one)
    cheap = colab.estimate_classification(cfg, rows, tax, model="gemini-2.5-flash-lite")
    assert cheap < colab.estimate_classification(cfg, rows, tax, model="gemini-3.8-flash")


def test_check_fits_and_ledger_report(tmp_path):
    cfg = _cfg(tmp_path)
    ledger = Ledger(tmp_path / "ledger.jsonl", 1.00, cfg.prices)
    colab.check_fits(ledger, 0.50, "thing")
    with pytest.raises(colab.StageIncomplete, match="thing"):
        colab.check_fits(ledger, 1.50, "thing")
    ledger.record("gemini-2.5-flash", "r", 1_000_000, 0)
    text = colab.ledger_report(ledger)
    assert "$0.30" in text and "$0.70" in text


def test_results_markdown_mentions_everything():
    scored = {
        "classifier_board": pd.DataFrame(
            {"classifier": ["gemini"], "level": ["product"], "metric": ["accuracy"], "value": [0.9]}
        ),
        "validator_board": pd.DataFrame({"validator": ["jeb"], "level": ["path"], "auroc": [0.8]}),
        "adjusted": pd.DataFrame({"level": ["product"], "value": [0.91]}),
        "decision": {"go": False, "reasons": ["because reasons"]},
    }
    md = colab.results_markdown(scored, 1.2345, 4.0)
    for needle in ("NO-GO", "because reasons", "0.91", "jeb", "gemini", "$1.23"):
        assert needle in md
    scored["decision"] = {"go": True, "reasons": ["fine"]}
    assert "GO" in colab.results_markdown(scored, 0.0, 4.0)


def test_overrides_file_is_json(tmp_path):
    cfg = _cfg(tmp_path)
    colab.save_overrides(cfg, prompt_version="v2")
    data = json.loads((tmp_path / "config_overrides.json").read_text())
    assert data == {"prompt_version": "v2"}


# ---- notebook-step helpers -------------------------------------------------------------


def test_save_overrides_rejects_unpriced_model(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="no price"):
        colab.save_overrides(cfg, classifier_model="CHANGE_ME")


def _synthetic(n=60):
    from nlp.schema import COMPLAINTS_COLS

    rows = []
    for i in range(n):
        k = i % 3
        rows.append(
            (f"c{i:03d}", f"2026-01-{i % 28 + 1:02d}", f"P{k % 2}", f"I{k}", "(none)",
             f"word{k} " * 40 + f"extra{i}", "Co", "TX")
        )  # fmt: skip
    return pd.DataFrame(rows, columns=COMPLAINTS_COLS)


def test_split_summary_and_run_classic(tmp_path):
    import dataclasses

    from nlp import pipeline as P
    from nlp.classifiers.tfidf import TfidfLR
    from nlp.sampling import make_splits

    cfg = dataclasses.replace(
        _cfg(tmp_path), train_max=40, tuning_n=6, test_n=12, test_floor=6, self_consistency_n=3
    )
    comp = _synthetic()
    splits = make_splits(comp, cfg)
    summ = colab.split_summary(comp, splits)
    assert summ.loc["ALL", "total"] == len(splits)
    assert {"train", "tuning", "test"} <= set(summ.columns)

    train_df, test_df = P.train_test_frames(comp, splits)
    pred = colab.run_classic(cfg, "tfidf", lambda: TfidfLR(max_features=500), train_df, test_df)
    assert len(pred) == len(test_df) and store.exists(cfg, P.stage_probs("tfidf"))
    again = colab.run_classic(cfg, "tfidf", lambda: 1 / 0, train_df, test_df)  # staged: no refit
    assert len(again) == len(pred)


def test_compare_runs_reports_accuracy_and_dollars(tmp_path):
    cfg = _cfg(tmp_path)
    ledger = Ledger(tmp_path / "l.jsonl", 4.0, cfg.prices)
    truth = _tax_rows()
    good = truth.assign(classifier="x")

    def paid():
        ledger.record("gemini-2.5-flash", "r", 1_000_000, 0)
        return good

    out = colab.compare_runs({"a": paid, "b": lambda: good.iloc[:2]}, truth, ledger)
    assert out["run"].tolist() == ["a", "b"] and out["path"].tolist() == [1.0, 1.0]
    assert out["usd"].tolist() == [pytest.approx(0.30), 0.0]
    assert out["rows_done"].tolist() == [4, 2]

    def quota():
        raise DailyQuotaExhausted(3)

    with pytest.raises(colab.StageIncomplete, match="tomorrow"):
        colab.compare_runs({"q": quota}, truth, ledger)


def test_phase_estimate_marks_done_stages(tmp_path, capsys):
    from nlp import pipeline as P

    cfg = _cfg(tmp_path)
    ledger = Ledger(tmp_path / "l.jsonl", 4.0, cfg.prices)
    rows = _tax_rows()
    tax = build_taxonomy(rows)
    store.write(cfg, P.STAGE_JEB, pd.DataFrame({"x": [1]}))
    est = colab.phase_estimate(cfg, ledger, tax, rows, rows.head(2))
    assert est.set_index("stage").loc[P.STAGE_JEB, "done"]
    assert colab.pending_usd(est, [P.STAGE_JEB]) == 0.0
    assert colab.pending_usd(est, [P.STAGE_PRED_GEMINI]) > 0
    assert "Still to spend" in capsys.readouterr().out


class _FakeSheet:
    def __init__(self, records):
        self.sheet1 = self
        self._r = records

    def get_all_records(self, **kw):
        return self._r


class _FakeGC:
    def __init__(self, records):
        self._r = records

    def open_by_url(self, url):
        return _FakeSheet(self._r)


def test_load_human_review_blocks_blank_unless_partial(tmp_path):
    cfg = _cfg(tmp_path)
    rec = {"complaint_id": "1", "verdict_product": "both OK", "verdict_issue": "",
           "verdict_sub_issue": "both OK", "summary_ok": "yes"}  # fmt: skip
    with pytest.raises(colab.StageIncomplete, match="blank"):
        colab.load_human_review(cfg, _FakeGC([rec]), "u")
    df = colab.load_human_review(cfg, _FakeGC([rec]), "u", allow_partial=True)
    assert len(df) == 3 and store.exists(cfg, "human_review")


def test_show_results_and_list_outputs(tmp_path):
    cfg = _cfg(tmp_path)
    ledger = Ledger(tmp_path / "l.jsonl", 4.0, cfg.prices)
    scored = {
        "classifier_board": pd.DataFrame({"classifier": ["g"]}),
        "validator_board": pd.DataFrame({"validator": ["v"]}),
        "adjusted": pd.DataFrame({"level": ["product"], "value": [0.9]}),
        "decision": {"go": True, "reasons": ["ok"]},
    }
    colab.show_results(cfg, scored, ledger)
    assert (tmp_path / "results.md").read_text(encoding="utf-8").startswith("## Phase 1 result")
    store.write(cfg, "x", pd.DataFrame({"a": [1]}))
    assert colab.list_outputs(cfg) == ["x.parquet"]
