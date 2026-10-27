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
        "validator_board": pd.DataFrame({"validator": ["jev"], "level": ["path"], "auroc": [0.8]}),
        "adjusted": pd.DataFrame({"level": ["product"], "value": [0.91]}),
        "decision": {"go": False, "reasons": ["because reasons"]},
    }
    md = colab.results_markdown(scored, 1.2345, 4.0)
    for needle in ("NO-GO", "because reasons", "0.91", "jev", "gemini", "$1.23"):
        assert needle in md
    scored["decision"] = {"go": True, "reasons": ["fine"]}
    assert "GO" in colab.results_markdown(scored, 0.0, 4.0)


def test_overrides_file_is_json(tmp_path):
    cfg = _cfg(tmp_path)
    colab.save_overrides(cfg, thinking="medium")
    data = json.loads((tmp_path / "config_overrides.json").read_text())
    assert data == {"thinking": "medium"}


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
    store.write(cfg, P.STAGE_JEV, pd.DataFrame({"x": [1]}))
    est = colab.phase_estimate(cfg, ledger, tax, rows, rows.head(2))
    assert est.set_index("stage").loc[P.STAGE_JEV, "done"]
    assert colab.pending_usd(est, [P.STAGE_JEV]) == 0.0
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


# ---- fix round 1 ------------------------------------------------------------------------


def test_overrides_restricted_to_four_settings(tmp_path):
    cfg = _cfg(tmp_path)
    for bad in ("prompt_version", "spend_cap_usd", "rpm", "seed"):
        with pytest.raises(ValueError, match="not an overridable"):
            colab.save_overrides(cfg, **{bad: 1})
    (tmp_path / "config_overrides.json").write_text('{"spend_cap_usd": 99}', encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be overridden"):
        colab.load_config(CONFIG, tmp_path)


def test_apply_choices_is_rerun_safe(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="PICK"):
        colab.apply_choices(cfg, require_pick=True, classifier_model=None)
    cfg = colab.apply_choices(cfg, require_pick=True, classifier_model="gemini-3.8-flash")
    f = tmp_path / "config_overrides.json"
    stamp = f.stat().st_mtime_ns
    # rerun with None everywhere: keeps the saved pick, writes nothing
    cfg2 = colab.load_config(CONFIG, tmp_path)
    assert cfg2.classifier_model == "gemini-3.8-flash"
    same = colab.apply_choices(cfg2, require_pick=True, classifier_model=None, batch_size=None)
    assert same == cfg2 and f.stat().st_mtime_ns == stamp
    # equal value: no write; different value: saved
    colab.apply_choices(cfg2, classifier_model="gemini-3.8-flash", batch_size=cfg2.batch_size)
    assert f.stat().st_mtime_ns == stamp
    assert colab.apply_choices(cfg2, batch_size=25).batch_size == 25
    assert colab.load_config(CONFIG, tmp_path).batch_size == 25
    assert "classifier gemini-3.8-flash" in capsys.readouterr().out


def test_stage_paid_tolerates_two_percent_but_not_budget(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    ids = [f"c{i}" for i in range(100)]
    ledger = Ledger(tmp_path / "l.jsonl", 4.0, cfg.prices)
    one_short = pd.DataFrame({"complaint_id": ids[:99]})
    two_short = pd.DataFrame({"complaint_id": ids[:98]})
    three_short = pd.DataFrame({"complaint_id": ids[:97]})
    # >2% missing: not saved
    with pytest.raises(colab.StageIncomplete, match="97 of 100"):
        colab.stage_paid(cfg, "a", lambda: three_short, ids, ledger)
    assert not store.exists(cfg, "a")
    # <=2% missing and budget fine: saved, ids printed
    out = colab.stage_paid(cfg, "a", lambda: two_short, ids, ledger)
    assert len(out) == 98 and store.exists(cfg, "a")
    printed = capsys.readouterr().out
    assert "c98" in printed and "c99" in printed and "2 of 100" in printed
    # budget exhausted: even 1 missing row is not saved
    ledger.record("gemini-2.5-flash", "r", 0, 1_600_000)  # $4.00 spent
    with pytest.raises(colab.StageIncomplete, match="99 of 100"):
        colab.stage_paid(cfg, "b", lambda: one_short, ids, ledger)
    assert not store.exists(cfg, "b")


def test_measured_estimate_from_ledger(tmp_path):
    cfg = _cfg(tmp_path)
    ledger = Ledger(tmp_path / "l.jsonl", 4.0, cfg.prices)
    rows = _tax_rows()  # 4 rows
    assert colab.measured_usd_per_request(ledger, "gemini-2.5-flash") is None
    assert colab.estimate_measured(cfg, ledger, rows) is None
    ledger.record("gemini-2.5-flash", "a", 1_000_000, 0)  # $0.30
    ledger.record("gemini-2.5-flash", "b", 3_000_000, 0)  # $0.90
    ledger.record("gemini-3.1-flash-lite", "c", 1_000_000, 0)  # other model ignored
    assert colab.measured_usd_per_request(ledger, "gemini-2.5-flash") == pytest.approx(0.60)
    # 4 rows at batch 2 -> 2 requests x $0.60 x 1.5 safety, x2 runs
    got = colab.estimate_measured(cfg, ledger, rows, batch_size=2, runs=2)
    assert got == pytest.approx(2 * 2 * 0.60 * 1.5)


def test_phase_estimate_uses_measured_when_available(tmp_path, capsys):
    from nlp import pipeline as P

    cfg = _cfg(tmp_path)
    ledger = Ledger(tmp_path / "l.jsonl", 4.0, cfg.prices)
    rows = _tax_rows()
    tax = build_taxonomy(rows)
    before = colab.phase_estimate(cfg, ledger, tax, rows, rows.head(2))
    assert not before["measured"].any()
    assert (before["usd"] == before["bound_usd"]).all()
    ledger.record(cfg.classifier_model, "a", 1000, 100)
    after = colab.phase_estimate(cfg, ledger, tax, rows, rows.head(2)).set_index("stage")
    assert after.loc[P.STAGE_PRED_GEMINI, "measured"]
    assert after.loc[P.STAGE_PRED_GEMINI, "usd"] < after.loc[P.STAGE_PRED_GEMINI, "bound_usd"]
    out = capsys.readouterr().out
    assert "worst-case bound" in out


def test_check_fits_prints_bound(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    ledger = Ledger(tmp_path / "l.jsonl", 1.0, cfg.prices)
    colab.check_fits(ledger, 0.2, "x", bound_usd=0.9)
    assert "worst-case bound $0.90" in capsys.readouterr().out


def test_restart_detection_logic():
    assert not colab.needs_restart("abc", "abc")
    assert not colab.needs_restart(None, "abc")  # git unavailable: do not block
    assert not colab.needs_restart("abc", None)
    assert colab.needs_restart("abc", "def")
    colab.check_code_fresh("abc", "abc")
    with pytest.raises(RuntimeError, match="Restart session"):
        colab.check_code_fresh("abc", "def")


def test_phase_estimate_prices_jev_step_per_request(tmp_path, capsys):
    from nlp import pipeline as P

    cfg = _cfg(tmp_path)
    ledger = Ledger(tmp_path / "l.jsonl", 4.0, cfg.prices)
    rows = _tax_rows()
    tax = build_taxonomy(rows)
    est = colab.phase_estimate(cfg, ledger, tax, rows, rows.head(2)).set_index("stage")
    assert est.loc[P.STAGE_JEV, "step"] == "Jev validator, full test"
    assert 0 < est.loc[P.STAGE_JEV, "bound_usd"] < 0.01  # $0.042 per 1M input tokens
    ledger.record(cfg.jev_model, "a", 1000, 0)
    after = colab.phase_estimate(cfg, ledger, tax, rows, rows.head(2)).set_index("stage")
    assert after.loc[P.STAGE_JEV, "measured"]
    # measured: one request per complaint, x1.5 safety
    assert after.loc[P.STAGE_JEV, "usd"] == pytest.approx(
        min(len(rows) * (1000 * 0.042 / 1e6) * colab.MEASURED_SAFETY, after.loc[P.STAGE_JEV, "bound_usd"])
    )
    capsys.readouterr()


def test_read_secrets_jev_key_is_optional(capsys):
    class UD:
        def __init__(self, d):
            self.d = d

        def get(self, name):
            if name not in self.d:
                raise KeyError(name)
            return self.d[name]

    assert colab.read_secrets(UD({"GEMINI_API_KEY": "g", "TYPESAFE_API_KEY": "t"})) == ("g", "t")
    assert capsys.readouterr().out == ""
    assert colab.read_secrets(UD({"GEMINI_API_KEY": "g"})) == ("g", None)
    assert "notebook 04" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        colab.read_secrets(UD({"TYPESAFE_API_KEY": "t"}))
