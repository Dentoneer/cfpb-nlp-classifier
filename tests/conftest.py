from dataclasses import replace
from pathlib import Path

import pytest

from nlp.config import Config

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "phase1.yaml"


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> Config:
    return replace(Config.load(DEFAULT_CONFIG), artifacts_dir=tmp_path / "artifacts")
