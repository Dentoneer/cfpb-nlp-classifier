"""Spend guard for paid Gemini calls: JSONL ledger, cost math, rate limiter.

Google budget alerts do not stop spending; this module is the real protection.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path

_EPS = 1e-9  # float slack so spend exactly at the cap is allowed


class BudgetExceeded(Exception):
    """A request would push cumulative spend past the cap."""


class DailyQuotaExhausted(Exception):
    """The provider's daily request quota is used up."""

    def __init__(self, remaining_rows: int):
        super().__init__(f"daily quota exhausted; {remaining_rows} rows remaining")
        self.remaining_rows = remaining_rows


def _price(prices: dict, model: str) -> tuple[float, float]:
    return prices[model]  # unknown model -> KeyError, never priced as zero


def _usd(prices: dict, model: str, in_tok: int, out_tok: int) -> float:
    p_in, p_out = _price(prices, model)
    if in_tok < 0 or out_tok < 0:
        raise ValueError("token counts must be non-negative")
    return (in_tok * p_in + out_tok * p_out) / 1_000_000


def estimate_run(n_requests: int, avg_in: float, avg_out: float, model: str, prices: dict) -> float:
    """Estimated USD for n requests; avg_out includes thinking tokens."""
    return n_requests * _usd(prices, model, avg_in, avg_out)


class Ledger:
    """Append-only JSONL spend ledger; the file is the source of truth."""

    def __init__(self, path: Path, cap_usd: float, prices: dict):
        self.path = Path(path)
        self.cap_usd = cap_usd
        self.prices = prices

    def spent(self) -> float:
        """Total USD over all lines, re-read from disk; skips corrupt/truncated lines."""
        if not self.path.exists():
            return 0.0
        amounts = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    amounts.append(float(json.loads(line)["usd"]))
                except (ValueError, KeyError, TypeError):
                    continue
        return math.fsum(amounts)

    def cost(self, model: str, in_tok: int, out_tok: int) -> float:
        return _usd(self.prices, model, in_tok, out_tok)

    def reserve(self, model: str, est_in: int, est_out: int) -> None:
        """Pre-check only: raise if spent + estimated cost would exceed the cap."""
        est = self.cost(model, est_in, est_out)
        spent = self.spent()
        if spent + est > self.cap_usd + _EPS:
            raise BudgetExceeded(
                f"spent ${spent:.4f} + est ${est:.4f} would exceed cap ${self.cap_usd:.2f}"
            )

    def record(self, model: str, request_id: str, in_tok: int, out_tok: int) -> float:
        """Durably log actual spend (never refuses) and return its USD."""
        usd = self.cost(model, in_tok, out_tok)
        line = json.dumps(
            {
                "ts": time.time(),
                "model": model,
                "request_id": request_id,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "usd": usd,
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        prefix = "\n" if self._needs_newline() else ""  # after a crash mid-write
        with self.path.open("a", encoding="utf-8") as f:
            f.write(prefix + line + "\n")
            f.flush()
            os.fsync(f.fileno())
        return usd

    def _needs_newline(self) -> bool:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return False
        with self.path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            return f.read(1) != b"\n"


class RateLimiter:
    """Enforce a minimum spacing of 60/rpm seconds between wait() returns."""

    def __init__(
        self,
        rpm: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if rpm <= 0:
            raise ValueError("rpm must be positive")
        self.interval = 60.0 / rpm
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self._last + self.interval - self._clock()
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()
