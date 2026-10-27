"""Experiment configuration, loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    classifier_model: str
    judge_model: str
    prompt_version: str
    rpm: int
    artifacts_dir: Path = Path("artifacts")
    seed: int = 20261026
    train_max: int = 20000
    tuning_n: int = 200
    test_n: int = 800
    test_floor: int = 400
    hand_check_n: int = 200
    self_consistency_n: int = 150
    max_chars: int = 6000
    batch_size: int = 10
    spend_cap_usd: float = 4.00
    prices: dict[str, tuple[float, float]] | None = None
    rpd: int | None = None
    thinking: str = "low"
    embed_model: str = "BAAI/bge-base-en-v1.5"
    likely_correct_threshold: float = 0.6
    jev_model: str = "jev-1.14.0"

    @classmethod
    def load(cls, path: Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if "artifacts_dir" in raw:
            raw["artifacts_dir"] = Path(raw["artifacts_dir"])
        prices = raw.get("prices") or {}
        raw["prices"] = {m: (float(p[0]), float(p[1])) for m, p in prices.items()}
        return cls(**raw)
