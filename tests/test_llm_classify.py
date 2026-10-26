import json
import os
import re
from dataclasses import replace

import pandas as pd
import pytest

from nlp.budget import DailyQuotaExhausted, Ledger, RateLimiter
from nlp.llm.classify import (
    RESPONSE_SCHEMA,
    build_prompt,
    estimate_request_tokens,
    parse_and_validate,
    run_classification,
)
from nlp.llm.client import FakeClient, LLMResponse, RateLimited
from nlp.schema import PREDICTIONS_COLS
from nlp.sheets import taxonomy_snapshot_hash
from nlp.store import JsonlCache
from nlp.taxonomy import build_taxonomy

PRICES = {"m": (1.0, 1.0)}  # $1 per 1M in and out


@pytest.fixture
def taxonomy():
    df = pd.DataFrame(
        {
            "product": ["Credit card", "Credit card", "Mortgage"],
            "issue": ["Billing dispute", "Fraud", "Escrow"],
            "sub_issue": ["(none)", "Stolen card", "Wrong amount"],
        }
    )
    return build_taxonomy(df)


@pytest.fixture
def cfg(tmp_cfg):
    return replace(tmp_cfg, classifier_model="m", prices=PRICES, batch_size=3, spend_cap_usd=4.0)


def rows(n):
    return pd.DataFrame(
        {"complaint_id": [f"c{i}" for i in range(n)], "narrative": [f"text {i}" for i in range(n)]}
    )


def item(cid, product="Credit card", issue="Billing dispute", sub="(none)", conf=0.9):
    return {
        "complaint_id": cid,
        "product": product,
        "issue": issue,
        "sub_issue": sub,
        "summary": "A short summary.",
        "confidence": conf,
    }


def good_responder(in_tok=100, out_tok=100):
    def responder(model, system, user, schema, thinking):
        ids = re.findall(r"complaint_id: (\S+)", user)
        return LLMResponse(json.dumps([item(i) for i in ids]), in_tok, out_tok, None)

    return responder


def make(tmp_path, responder, cap=4.0):
    ledger = Ledger(tmp_path / "ledger.jsonl", cap, PRICES)
    cache = JsonlCache(tmp_path / "cache.jsonl")
    client = FakeClient(responder)
    limiter = RateLimiter(10_000, sleep=lambda s: None)
    return client, ledger, cache, limiter


def run(cfg, client, ledger, cache, limiter, taxonomy, n=7, sleep=lambda s: None):
    return run_classification(
        rows(n), cfg, client, ledger, cache, taxonomy=taxonomy, limiter=limiter, sleep=sleep
    )


def test_parse_normalizes_case_and_whitespace(taxonomy):
    raw = json.dumps([item("a", "  credit CARD ", "billing   dispute", "(NONE)")])
    df, retry = parse_and_validate(raw, ["a"], taxonomy)
    assert retry == []
    r = df.iloc[0]
    assert (r["product"], r["issue"], r["sub_issue"]) == (
        "Credit card",
        "Billing dispute",
        "(none)",
    )


def test_parse_handles_missing_duplicate_unknown_ids(taxonomy):
    first = item("a", sub="(none)")
    dup = item("a", "Mortgage", "Escrow", "Wrong amount")
    raw = json.dumps([first, dup, item("zzz")])  # b missing, zzz unknown
    df, retry = parse_and_validate(raw, ["a", "b"], taxonomy)
    assert list(df["complaint_id"]) == ["a"]
    assert df.iloc[0]["product"] == "Credit card"  # first kept
    assert retry == ["b"]


def test_parse_rejects_invalid_path(taxonomy):
    raw = json.dumps([item("a", "Credit card", "Escrow", "Wrong amount"), item("b")])
    df, retry = parse_and_validate(raw, ["a", "b"], taxonomy)
    assert list(df["complaint_id"]) == ["b"]
    assert retry == ["a"]


def test_parse_garbage_retries_everything(taxonomy):
    df, retry = parse_and_validate("not json", ["a", "b"], taxonomy)
    assert df.empty and retry == ["a", "b"]


def test_parse_bad_confidence(taxonomy):
    raw = json.dumps([item("a", conf="high"), item("b", conf=7)])
    df, retry = parse_and_validate(raw, ["a", "b"], taxonomy)
    assert retry == ["a"]
    assert df.iloc[0]["confidence"] == 1.0  # clamped


def test_estimate_request_tokens():
    est_in, est_out = estimate_request_tokens("a" * 400, "b" * 400, 10, "none")
    assert est_in == 200
    assert est_out == int(10 * 150 * 1.5)
    assert estimate_request_tokens("", "", 10, "low")[1] == int(10 * (150 + 512) * 1.5)
    assert estimate_request_tokens("", "", 10, "high")[1] == int(10 * (150 + 2048) * 1.5)
    assert estimate_request_tokens("", "", 1, "minimal")[1] == int(150 * 1.5)


def test_build_prompt_contents():
    system, user = build_prompt(rows(2), "TAXTEXT", "v1")
    assert "TAXTEXT" in system and "60 words" in system and "exactly one" in system
    assert "complaint_id: c0" in user and "complaint_id: c1" in user
    assert RESPONSE_SCHEMA["type"] == "array"


def test_every_prediction_has_model_and_prompt_version(tmp_path, cfg, taxonomy):
    client, ledger, cache, limiter = make(tmp_path, good_responder())
    out = run(cfg, client, ledger, cache, limiter, taxonomy)
    assert len(out) == 7
    assert set(PREDICTIONS_COLS) <= set(out.columns)
    assert (out["model_id"] == "m").all() and (out["prompt_version"] == cfg.prompt_version).all()
    assert (out["classifier"] == "gemini").all()
    assert (out["taxonomy_hash"] == taxonomy_snapshot_hash(taxonomy)).all()
    assert out["run_ts"].notna().all()
    assert len(client.calls) == 3  # 3+3+1
    assert ledger.spent() == pytest.approx(3 * 200 / 1e6)


def test_cached_rows_not_resent_or_charged(tmp_path, cfg, taxonomy):
    client, ledger, cache, limiter = make(tmp_path, good_responder())
    first = run(cfg, client, ledger, cache, limiter, taxonomy)
    spent, calls = ledger.spent(), len(client.calls)
    cache2 = JsonlCache(tmp_path / "cache.jsonl")  # fresh process
    second = run(cfg, client, ledger, cache2, limiter, taxonomy)
    assert len(client.calls) == calls
    assert ledger.spent() == spent
    assert len(second) == len(first)


def test_budget_exhaustion_stops_cleanly(tmp_path, cfg, taxonomy, capsys):
    # Each request records $0.0002 actual; the conservative reserve estimate (~$0.0033) is far
    # larger, so this cap lets the first requests through and then stops without raising.
    client, ledger, cache, limiter = make(tmp_path, good_responder(), cap=0.0036)
    out = run(cfg, client, ledger, cache, limiter, taxonomy, n=9)
    assert 0 < len(out) < 9
    assert ledger.spent() <= 0.0036
    assert "remaining" in capsys.readouterr().out.lower()


def test_budget_exhausted_at_start_returns_empty(tmp_path, cfg, taxonomy):
    client, ledger, cache, limiter = make(tmp_path, good_responder(), cap=0.0)
    out = run(cfg, client, ledger, cache, limiter, taxonomy)
    assert out.empty and client.calls == []
    assert list(out.columns) == PREDICTIONS_COLS


def test_daily_quota_raises_with_remaining(tmp_path, cfg, taxonomy):
    good = good_responder()

    def responder(model, system, user, schema, thinking):
        if len(client.calls) > 1:
            raise RateLimited(None, True)
        return good(model, system, user, schema, thinking)

    client, ledger, cache, limiter = make(tmp_path, responder)
    with pytest.raises(DailyQuotaExhausted) as e:
        run(cfg, client, ledger, cache, limiter, taxonomy)
    assert e.value.remaining_rows == 4
    assert len(cache.keys()) == 3  # first request already saved


def test_non_daily_rate_limit_backs_off_then_succeeds(tmp_path, cfg, taxonomy):
    good = good_responder()
    n = {"k": 0}

    def responder(*a):
        n["k"] += 1
        if n["k"] <= 2:
            raise RateLimited(None, False)
        return good(*a)

    client, ledger, cache, limiter = make(tmp_path, responder)
    sleeps = []
    out = run(cfg, client, ledger, cache, limiter, taxonomy, n=2, sleep=sleeps.append)
    assert len(out) == 2
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]


def test_non_daily_rate_limit_gives_up_after_5(tmp_path, cfg, taxonomy):
    def responder(*a):
        raise RateLimited(1.0, False)

    client, ledger, cache, limiter = make(tmp_path, responder)
    with pytest.raises(RateLimited):
        run(cfg, client, ledger, cache, limiter, taxonomy, n=2)
    assert len(client.calls) == 5


def test_missing_rows_requeued_then_dropped_after_two_retries(tmp_path, cfg, taxonomy, capsys):
    def responder(model, system, user, schema, thinking):
        ids = [i for i in re.findall(r"complaint_id: (\S+)", user) if i != "c1"]
        return LLMResponse(json.dumps([item(i) for i in ids]), 10, 10, None)

    client, ledger, cache, limiter = make(tmp_path, responder)
    out = run(cfg, client, ledger, cache, limiter, taxonomy, n=3)
    assert sorted(out["complaint_id"]) == ["c0", "c2"]
    assert len(client.calls) == 3  # initial + 2 retries of c1
    assert "1 row(s) failed" in capsys.readouterr().out


def test_spend_recorded_before_cache(tmp_path, cfg, taxonomy):
    seen = {}
    client, ledger, cache, limiter = make(tmp_path, good_responder())
    orig = cache.put

    def put(key, value):
        seen.setdefault("spent_at_first_put", ledger.spent())
        orig(key, value)

    cache.put = put
    run(cfg, client, ledger, cache, limiter, taxonomy, n=1)
    assert seen["spent_at_first_put"] > 0


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="needs GEMINI_API_KEY")
def test_real_gemini_one_batch(tmp_path, tmp_cfg, taxonomy):
    from nlp.llm.client import GenAIClient

    cfg = replace(
        tmp_cfg,
        classifier_model="gemini-2.5-flash-lite",
        batch_size=2,
        rpm=10,
        prices={"gemini-2.5-flash-lite": (0.10, 0.40)},
    )
    r = pd.DataFrame(
        {
            "complaint_id": ["r1", "r2"],
            "narrative": [
                "Someone stole my credit card and made charges I did not authorize.",
                "My mortgage escrow payment amount is wrong after the annual analysis.",
            ],
        }
    )
    ledger = Ledger(tmp_path / "l.jsonl", 0.05, cfg.prices)
    cache = JsonlCache(tmp_path / "c.jsonl")
    out = run_classification(
        r, cfg, GenAIClient(os.environ["GEMINI_API_KEY"]), ledger, cache, taxonomy=taxonomy
    )
    assert len(out) == 2
    paths = set(zip(taxonomy["product"], taxonomy["issue"], taxonomy["sub_issue"]))
    for _, x in out.iterrows():
        assert (x["product"], x["issue"], x["sub_issue"]) in paths
