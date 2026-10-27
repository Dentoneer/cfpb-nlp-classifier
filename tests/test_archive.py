"""Old results on Drive are archived automatically when data_version changes (design v2)."""

import json
from pathlib import Path

import pandas as pd

from nlp import colab, store

CONFIG = Path(__file__).resolve().parent.parent / "config" / "phase1.yaml"
KEPT = ["ledger.jsonl", "gemini_cache.jsonl", "judge_cache.jsonl", "config_overrides.json"]
V1_STAGES = ["complaints", "taxonomy", "splits", "filter_report", "pred_tfidf", "probs_tfidf",
             "pred_gemini"]  # fmt: skip


def _live_v1_drive(art: Path) -> None:
    """What the Drive folder held after the first (v1) run."""
    art.mkdir(parents=True)
    for name in V1_STAGES:
        pd.DataFrame({"x": [1]}).to_parquet(art / f"{name}.parquet")
    (art / "models" / "tfidf_abc").mkdir(parents=True)
    (art / "models" / "tfidf_abc" / "model.joblib").write_text("m", encoding="utf-8")
    (art / "taxonomy_url.txt").write_text("https://sheet/1", encoding="utf-8")
    (art / "ledger.jsonl").write_text(
        json.dumps({"model": "gemini-3.8-flash", "usd": 1.21}) + "\n", encoding="utf-8"
    )
    (art / "gemini_cache.jsonl").write_text('{"key": "k", "value": {}}\n', encoding="utf-8")
    (art / "config_overrides.json").write_text(
        json.dumps({"classifier_model": "gemini-3.8-flash", "batch_size": 10, "thinking": "low"}),
        encoding="utf-8",
    )


def test_v1_results_are_archived_and_money_files_kept(tmp_path, capsys):
    art = tmp_path / "phase1"
    _live_v1_drive(art)
    moved = colab.archive_old_results(art, "v2")
    out = capsys.readouterr().out
    arch = art / "archive_v1"
    for name in V1_STAGES:
        assert not (art / f"{name}.parquet").exists() and (arch / f"{name}.parquet").exists()
    assert (arch / "models" / "tfidf_abc" / "model.joblib").exists() and not (art / "models").exists()
    assert (arch / "taxonomy_url.txt").exists() and not (art / "taxonomy_url.txt").exists()
    for f in ("ledger.jsonl", "gemini_cache.jsonl", "config_overrides.json"):
        assert (art / f).exists(), f
    assert (art / "data_version.txt").read_text(encoding="utf-8").strip() == "v2"
    assert "taxonomy_url.txt" in moved and "models" in moved
    assert "archive_v1" in out and "ledger" in out.lower()


def test_rerun_with_zero_manual_steps_after_archive(tmp_path):
    """Setup on the live v1 folder leaves exactly what the v2 notebooks expect."""
    art = tmp_path / "phase1"
    _live_v1_drive(art)
    colab.archive_old_results(art, "v2")
    cfg = colab.load_config(CONFIG, art)  # overrides still apply
    assert cfg.classifier_model == "gemini-3.8-flash" and cfg.batch_size == 10
    ledger, *_ = colab.open_tools(cfg)
    assert ledger.spent() == 1.21  # the spend cap still counts the v1 money
    for name in V1_STAGES:
        assert not store.exists(cfg, name)  # 01 rebuilds them
    try:
        colab.read_url(cfg, "taxonomy")
        raise AssertionError("old taxonomy sheet URL must not be reused")
    except colab.StageIncomplete:
        pass


def test_archive_is_idempotent_and_never_touches_new_results(tmp_path, capsys):
    art = tmp_path / "phase1"
    _live_v1_drive(art)
    colab.archive_old_results(art, "v2")
    capsys.readouterr()
    # v2 notebooks produce new results ...
    pd.DataFrame({"x": [2]}).to_parquet(art / "complaints.parquet")
    (art / "models").mkdir()
    (art / "taxonomy_url.txt").write_text("https://sheet/2", encoding="utf-8")
    # ... and the next setup must not archive them.
    assert colab.archive_old_results(art, "v2") == []
    assert (art / "complaints.parquet").exists() and (art / "models").exists()
    assert (art / "taxonomy_url.txt").read_text(encoding="utf-8") == "https://sheet/2"
    assert capsys.readouterr().out == ""


def test_fresh_folder_just_gets_the_marker(tmp_path):
    art = tmp_path / "phase1"
    art.mkdir()
    (art / "ledger.jsonl").write_text("", encoding="utf-8")
    assert colab.archive_old_results(art, "v2") == []
    assert (art / "data_version.txt").read_text(encoding="utf-8").strip() == "v2"
    assert not any(p.name.startswith("archive_") for p in art.iterdir())


def test_interrupted_archive_finishes_on_rerun(tmp_path):
    art = tmp_path / "phase1"
    _live_v1_drive(art)
    (art / "archive_v1").mkdir()
    pd.DataFrame({"x": [0]}).to_parquet(art / "archive_v1" / "complaints.parquet")  # half done
    moved = colab.archive_old_results(art, "v2")
    assert "complaints.parquet" in moved and not (art / "complaints.parquet").exists()
    assert len(list((art / "archive_v1").glob("complaints*.parquet"))) == 2  # nothing overwritten


def test_later_version_change_archives_under_old_marker(tmp_path):
    art = tmp_path / "phase1"
    art.mkdir()
    (art / "data_version.txt").write_text("v2\n", encoding="utf-8")
    pd.DataFrame({"x": [1]}).to_parquet(art / "splits.parquet")
    colab.archive_old_results(art, "v3")
    assert (art / "archive_v2" / "splits.parquet").exists()


def test_setup_archives_before_anything_reads_stages(tmp_path, monkeypatch, capsys):
    """setup() on the live v1 folder: archive happens automatically, money files kept."""
    import sys
    import types

    import google.auth

    drive_root = tmp_path / "drive"
    art = drive_root / "MyDrive" / "cfpb-nlp-classifier" / "phase1"
    _live_v1_drive(art)

    class UD:
        @staticmethod
        def get(name):
            return {"GEMINI_API_KEY": "AQ.key\n", "TYPESAFE_API_KEY": "t"}[name]

    fake = types.ModuleType("google.colab")
    fake.auth = types.SimpleNamespace(authenticate_user=lambda: None)
    fake.drive = types.SimpleNamespace(mount=lambda path: None)
    fake.userdata = UD
    monkeypatch.setitem(sys.modules, "google.colab", fake)
    monkeypatch.setitem(sys.modules, "gspread", types.SimpleNamespace(authorize=lambda c: "gc"))
    monkeypatch.setattr(google.auth, "default", lambda: ("creds", "proj"))
    monkeypatch.setattr(colab, "_git_head", lambda d: None)
    repo = Path(__file__).resolve().parent.parent
    cfg, gc, key, _jev = colab.setup(repo_dir=str(repo), drive_root=str(drive_root))
    assert key == "AQ.key" and gc == "gc" and cfg.artifacts_dir == art
    assert (art / "archive_v1" / "complaints.parquet").exists()
    assert (art / "ledger.jsonl").exists() and cfg.classifier_model == "gemini-3.8-flash"
    assert "archive_v1" in capsys.readouterr().out
