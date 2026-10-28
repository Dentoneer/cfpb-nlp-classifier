from pathlib import Path

import pytest

from nlp.config import Config
from nlp.schema import PREDICTIONS_COLS

DEFAULT = Path(__file__).resolve().parent.parent / "config" / "phase1.yaml"


def test_config_loads_default():
    cfg = Config.load(DEFAULT)
    assert cfg.spend_cap_usd == 1000.00 and cfg.judge_model != cfg.classifier_model
    assert cfg.prices["gemini-3.8-flash"] == (0.75, 3.75)
    assert cfg.rpd is None and cfg.thinking == "low"
    assert cfg.likely_correct_threshold == 0.6
    assert isinstance(cfg.artifacts_dir, Path)


def test_schema_predictions_cols():
    assert PREDICTIONS_COLS[0] == "complaint_id" and "taxonomy_hash" in PREDICTIONS_COLS


# ---- final review fixes ------------------------------------------------------------------

def _yaml(tmp_path, extra: str) -> Path:
    base = DEFAULT.read_text(encoding="utf-8")
    p = tmp_path / "c.yaml"
    p.write_text(base + "\n" + extra, encoding="utf-8")
    return p


@pytest.mark.parametrize(
    "extra, match",
    [
        ("  bad-model: [-0.1, 1.0]", "price"),
        ("  bad-model: [.nan, 1.0]", "price"),
        ("  bad-model: [0.1, .inf]", "price"),
    ],
)
def test_config_rejects_bad_prices(tmp_path, extra, match):
    with pytest.raises(ValueError, match=match):
        Config.load(_yaml(tmp_path, extra))


@pytest.mark.parametrize("cap", ["0", "-1", ".nan", ".inf"])
def test_config_rejects_bad_cap(tmp_path, cap):
    text = DEFAULT.read_text(encoding="utf-8").replace("spend_cap_usd: 1000.00", f"spend_cap_usd: {cap}")
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="spend_cap_usd"):
        Config.load(p)
