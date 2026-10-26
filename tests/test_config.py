from pathlib import Path

from nlp.config import Config
from nlp.schema import PREDICTIONS_COLS

DEFAULT = Path(__file__).resolve().parent.parent / "config" / "phase1.yaml"


def test_config_loads_default():
    cfg = Config.load(DEFAULT)
    assert cfg.spend_cap_usd == 4.00 and cfg.judge_model != cfg.classifier_model
    assert cfg.prices["gemini-3.8-flash"] == (0.75, 3.75)
    assert cfg.rpd is None and cfg.thinking == "low"
    assert cfg.likely_correct_threshold == 0.6
    assert isinstance(cfg.artifacts_dir, Path)


def test_schema_predictions_cols():
    assert PREDICTIONS_COLS[0] == "complaint_id" and "taxonomy_hash" in PREDICTIONS_COLS
