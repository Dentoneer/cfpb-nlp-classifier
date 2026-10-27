"""Resumable artifact store: Parquet tables plus an append-only JSONL cache."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from nlp.config import Config


def _path(cfg: Config, name: str) -> Path:
    return Path(cfg.artifacts_dir) / f"{name}.parquet"


def exists(cfg: Config, name: str) -> bool:
    return _path(cfg, name).exists()


def read(cfg: Config, name: str) -> pd.DataFrame:
    return pd.read_parquet(_path(cfg, name))


def write(cfg: Config, name: str, df: pd.DataFrame) -> None:
    """Write atomically: temp file in the same directory, then rename."""
    final = _path(cfg, name)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(final.name + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)


def stage(cfg: Config, name: str, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Return the stored table if present, else compute, store, and return it."""
    if exists(cfg, name):
        return read(cfg, name)
    df = fn()
    write(cfg, name, df)
    return df


class JsonlCache:
    """Key -> dict cache backed by an append-only JSONL file (last write wins)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._data[rec["key"]] = rec["value"]
                    except (ValueError, KeyError, TypeError):
                        continue  # truncated or corrupt line (e.g. crash mid-write)

    def get(self, key: str) -> dict | None:
        return self._data.get(key)

    def put(self, key: str, value: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"key": key, "value": value}) + "\n"
        # A crash can leave a truncated last line with no newline; start a fresh line.
        prefix = "\n" if self._needs_newline() else ""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(prefix + line)
            f.flush()
            os.fsync(f.fileno())
        self._data[key] = value

    def _needs_newline(self) -> bool:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return False
        with self.path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            return f.read(1) != b"\n"

    def keys(self) -> set[str]:
        return set(self._data)
