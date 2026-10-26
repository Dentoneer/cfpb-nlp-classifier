import json

import pytest

from nlp.budget import (
    BudgetExceeded,
    DailyQuotaExhausted,
    Ledger,
    RateLimiter,
    estimate_run,
)
from nlp.store import JsonlCache

PRICES = {"gemini-2.5-flash": (0.30, 2.50), "dollar": (1.0, 0.0)}


def _ledger(tmp_path, cap=4.00):
    return Ledger(tmp_path / "ledger.jsonl", cap, PRICES)


def test_cost_math(tmp_path):
    assert _ledger(tmp_path).cost("gemini-2.5-flash", 1_000_000, 1_000_000) == pytest.approx(2.80)


def test_unknown_model_raises(tmp_path):
    led = _ledger(tmp_path)
    with pytest.raises(KeyError):
        led.cost("nope", 1, 1)
    with pytest.raises(KeyError):
        led.reserve("nope", 1, 1)
    with pytest.raises(KeyError):
        led.record("nope", "r1", 1, 1)
    assert led.spent() == 0.0


def test_cap_blocks_request_that_would_cross(tmp_path):
    led = _ledger(tmp_path)
    led.record("dollar", "r1", 3_990_000, 0)  # $3.99
    assert led.spent() == pytest.approx(3.99)
    with pytest.raises(BudgetExceeded):
        led.reserve("dollar", 20_000, 0)  # +$0.02 -> 4.01
    led.reserve("dollar", 10_000, 0)  # exactly $4.00 is allowed


def test_reserve_writes_nothing(tmp_path):
    led = _ledger(tmp_path)
    led.reserve("dollar", 1000, 0)
    assert not (tmp_path / "ledger.jsonl").exists()
    assert led.spent() == 0.0


def test_record_always_counts_even_over_cap(tmp_path):
    led = _ledger(tmp_path, cap=1.0)
    usd = led.record("dollar", "r1", 2_000_000, 0)
    assert usd == pytest.approx(2.0)
    assert led.spent() == pytest.approx(2.0)


def test_ledger_persists_across_instances(tmp_path):
    a = _ledger(tmp_path)
    a.record("gemini-2.5-flash", "r1", 1_000_000, 0)
    a.record("gemini-2.5-flash", "r2", 0, 1_000_000)
    b = _ledger(tmp_path)
    assert b.spent() == pytest.approx(2.80)
    line = json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[0])
    assert set(line) == {"ts", "model", "request_id", "input_tokens", "output_tokens", "usd"}


def test_spent_sees_other_instance_writes(tmp_path):
    a, b = _ledger(tmp_path), _ledger(tmp_path)
    a.record("dollar", "r1", 1_000_000, 0)
    assert b.spent() == pytest.approx(1.0)


def test_truncated_last_line_tolerated(tmp_path):
    led = _ledger(tmp_path)
    led.record("dollar", "r1", 1_000_000, 0)
    with (tmp_path / "ledger.jsonl").open("a") as f:
        f.write('{"ts": 1, "model": "dol')  # crash mid-write
    assert _ledger(tmp_path).spent() == pytest.approx(1.0)
    led.record("dollar", "r2", 1_000_000, 0)  # must start a fresh line
    assert _ledger(tmp_path).spent() == pytest.approx(2.0)


def test_ledger_persists_before_cache_append(tmp_path, monkeypatch):
    led = _ledger(tmp_path)
    cache = JsonlCache(tmp_path / "cache.jsonl")

    def boom(*a, **k):
        raise OSError("crash")

    monkeypatch.setattr(JsonlCache, "put", boom)
    with pytest.raises(OSError):
        led.record("gemini-2.5-flash", "r1", 1_000_000, 0)
        cache.put("r1", {"x": 1})
    assert _ledger(tmp_path).spent() == pytest.approx(0.30)


def test_estimate_run():
    assert estimate_run(1000, 1000, 500, "gemini-2.5-flash", PRICES) == pytest.approx(
        1000 * (1000 * 0.30 + 500 * 2.50) / 1e6
    )
    with pytest.raises(KeyError):
        estimate_run(1, 1, 1, "nope", PRICES)


def test_daily_quota_exhausted_carries_remaining():
    e = DailyQuotaExhausted(42)
    assert e.remaining_rows == 42
    assert isinstance(e, Exception)


def test_rate_limiter_spaces_calls():
    t = [100.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        t[0] += s

    rl = RateLimiter(60, clock=lambda: t[0], sleep=sleep)
    for _ in range(3):
        rl.wait()
    assert sum(sleeps) >= 2.0
