import json
import os
from dataclasses import replace

import pandas as pd
import pytest

from nlp.budget import BudgetExceeded, DailyQuotaExhausted, Ledger, RateLimiter
from nlp.llm.client import RateLimited
from nlp.llm.jev import (
    JEV_PROMPT_VERSION,
    FakeJevClient,
    HttpJevClient,
    JevError,
    JevResponse,
    build_jev_questions,
    parse_jev_response,
    run_jev,
    v_jev,
)
from nlp.store import JsonlCache

MODEL = "jev-1.14.0"
PRICES = {MODEL: (0.042, 0.0), "gemini-2.5-flash": (0.30, 2.50)}


@pytest.fixture
def cfg(tmp_cfg):
    return replace(tmp_cfg, jev_model=MODEL, prices=PRICES)


def rows(n=3):
    return pd.DataFrame(
        {
            "complaint_id": [f"c{i}" for i in range(n)],
            "narrative": [f"text {i}" for i in range(n)],
            "product": "P",
            "issue": "I",
            "sub_issue": "S",
        }
    )


def responder(model, state, questions):
    bad = "text 1" in state  # c1: issue not ok
    return JevResponse(
        {"product_ok": 0.9, "issue_ok": 0.2 if bad else 0.8, "sub_issue_ok": 0.5},
        1000,
        7,
        model,
    )


def make(tmp_path, resp=responder, cap=4.0):
    return (
        FakeJevClient(resp),
        Ledger(tmp_path / "l.jsonl", cap, PRICES),
        JsonlCache(tmp_path / "c.jsonl"),
        RateLimiter(10_000, sleep=lambda s: None),
    )


def test_questions_shape():
    q = build_jev_questions("Mortgage", "Trouble paying", "Foreclosure")
    assert list(q) == ["product_ok", "issue_ok", "sub_issue_ok"]
    assert all(v["type"] == "noul" for v in q.values())
    assert "'Mortgage'" in q["product_ok"]["instructions"]
    assert "'Trouble paying'" in q["issue_ok"]["instructions"]
    assert "'Foreclosure'" in q["sub_issue_ok"]["instructions"]
    none = build_jev_questions("Mortgage", "Trouble paying", "(none)")
    assert "(none)" not in none["sub_issue_ok"]["instructions"]
    assert "no more specific" in none["sub_issue_ok"]["instructions"].lower() or (
        "does not fit any more specific" in none["sub_issue_ok"]["instructions"]
    )


def test_request_shape_and_results(cfg, tmp_path):
    client, ledger, cache, limiter = make(tmp_path)
    out = run_jev(rows(), cfg, client, ledger, cache, limiter=limiter)
    assert len(client.calls) == 3  # one request per complaint
    model, state, questions = client.calls[0]
    assert model == MODEL and state == "text 0"
    assert set(questions) == {"product_ok", "issue_ok", "sub_issue_ok"}
    assert "'P'" in questions["product_ok"]["instructions"]
    assert len(out) == 3
    assert out.set_index("complaint_id").loc["c1", "issue_ok"] == 0.2
    assert cache.get(f"c0|jev|{MODEL}|{JEV_PROMPT_VERSION}") is not None
    # 1000 input tokens x 3 requests at $0.042/1M, output free
    assert ledger.spent() == pytest.approx(3 * 1000 * 0.042 / 1_000_000)


def test_cache_hit_is_free(cfg, tmp_path):
    client, ledger, cache, limiter = make(tmp_path)
    run_jev(rows(), cfg, client, ledger, cache, limiter=limiter)
    n, spent = len(client.calls), ledger.spent()
    out = run_jev(rows(), cfg, client, ledger, cache, limiter=limiter)
    assert len(client.calls) == n and ledger.spent() == spent and len(out) == 3


def test_label_change_is_rejudged(cfg, tmp_path):
    client, ledger, cache, limiter = make(tmp_path)
    r = rows()
    run_jev(r, cfg, client, ledger, cache, limiter=limiter)
    n = len(client.calls)
    r2 = r.copy()
    r2.loc[1, "issue"] = "Other issue"
    out = run_jev(r2, cfg, client, ledger, cache, limiter=limiter)
    assert len(client.calls) == n + 1
    assert client.calls[-1][1] == "text 1"
    assert "'Other issue'" in client.calls[-1][2]["issue_ok"]["instructions"]
    assert len(out) == 3


def test_spend_recorded_before_cache_put(cfg, tmp_path):
    client, ledger, cache, limiter = make(tmp_path)
    seen = []
    real_put = cache.put

    def put(key, value):
        seen.append(ledger.spent())
        real_put(key, value)

    cache.put = put
    run_jev(rows(1), cfg, client, ledger, cache, limiter=limiter)
    assert seen and seen[0] > 0


def test_missing_usage_records_estimate(cfg, tmp_path):
    client, ledger, cache, limiter = make(
        tmp_path, lambda m, s, q: JevResponse({k: 0.5 for k in q}, None, None, m)
    )
    run_jev(rows(1), cfg, client, ledger, cache, limiter=limiter)
    assert ledger.spent() > 0


def test_budget_stop_is_clean(cfg, tmp_path):
    client, _, cache, limiter = make(tmp_path)
    tiny = Ledger(tmp_path / "tiny.jsonl", 1e-12, PRICES)
    out = run_jev(rows(), cfg, client, tiny, cache, limiter=limiter)
    assert out.empty and client.calls == []
    assert list(out.columns) == ["complaint_id", "product_ok", "issue_ok", "sub_issue_ok", "model_id"]
    assert BudgetExceeded  # same exception family as run_queue


def test_budget_stops_midway(cfg, tmp_path):
    client, _, cache, limiter = make(tmp_path)
    one = 1000 * 0.042 / 1_000_000
    ledger = Ledger(tmp_path / "mid.jsonl", one * 2.05, PRICES)
    out = run_jev(rows(), cfg, client, ledger, cache, limiter=limiter)
    assert len(client.calls) == 2 and len(out) == 2  # third would cross the cap
    assert ledger.spent() <= ledger.cap_usd


def test_429_backoff_honours_retry_after(cfg, tmp_path):
    attempts = []

    def flaky(model, state, questions):
        attempts.append(1)
        if len(attempts) <= 2:
            raise RateLimited(30.0)
        return responder(model, state, questions)

    client, ledger, cache, limiter = make(tmp_path, flaky)
    sleeps = []
    out = run_jev(rows(1), cfg, client, ledger, cache, limiter=limiter, sleep=sleeps.append)
    assert len(out) == 1 and len(attempts) == 3
    assert sleeps == [30.0, 30.0]  # max(retry_after, 2*2**attempt)


def test_429_without_retry_after_backs_off_exponentially(cfg, tmp_path):
    attempts = []

    def flaky(model, state, questions):
        attempts.append(1)
        if len(attempts) <= 2:
            raise RateLimited(None)
        return responder(model, state, questions)

    client, ledger, cache, limiter = make(tmp_path, flaky)
    sleeps = []
    run_jev(rows(1), cfg, client, ledger, cache, limiter=limiter, sleep=sleeps.append)
    assert sleeps == [2.0, 4.0]


def test_429_gives_up_after_max_attempts(cfg, tmp_path):
    def always(model, state, questions):
        raise RateLimited(0.0)

    client, ledger, cache, limiter = make(tmp_path, always)
    with pytest.raises(RateLimited):
        run_jev(rows(1), cfg, client, ledger, cache, limiter=limiter, sleep=lambda s: None)
    assert ledger.spent() == 0.0


def test_daily_quota_raises(cfg, tmp_path):
    def daily(model, state, questions):
        raise RateLimited(None, daily=True)

    client, ledger, cache, limiter = make(tmp_path, daily)
    with pytest.raises(DailyQuotaExhausted) as e:
        run_jev(rows(), cfg, client, ledger, cache, limiter=limiter)
    assert e.value.remaining_rows == 3


def test_v_jev_scores_verdicts_and_path_product():
    res = pd.DataFrame(
        {
            "complaint_id": ["a", "b"],
            "product_ok": [0.9, 0.4],
            "issue_ok": [0.8, 0.9],
            "sub_issue_ok": [0.5, 0.9],
            "model_id": MODEL,
        }
    )
    out = v_jev(res)
    assert set(out.validator) == {"jev"} and len(out) == 8
    g = out.set_index(["complaint_id", "level"])
    assert g.loc[("a", "product"), "score"] == pytest.approx(0.9)
    assert g.loc[("a", "sub_issue"), "verdict"]  # 0.5 counts as yes
    assert g.loc[("a", "path"), "score"] == pytest.approx(0.9 * 0.8 * 0.5)
    assert not g.loc[("a", "path"), "verdict"]  # 0.36 < 0.5
    assert not g.loc[("b", "product"), "verdict"]
    assert g.loc[("b", "path"), "score"] == pytest.approx(0.4 * 0.9 * 0.9)
    assert list(out.columns) == ["complaint_id", "validator", "level", "verdict", "score"]


QIDS = ["product_ok", "issue_ok", "sub_issue_ok"]


@pytest.mark.parametrize(
    "body",
    [
        {k: {"type": "noul", "noul": 0.7} for k in QIDS},
        {"answers": {k: {"type": "noul", "noul": 0.7} for k in QIDS}},
        {"nouls": {k: {"noul": 0.7} for k in QIDS}},
        {"nouls": {k: 0.7 for k in QIDS}},
    ],
)
def test_parse_accepts_layouts(body):
    body = {"model": MODEL, **body, "usage": {"input_tokens": 12, "output_tokens": 3}}
    r = parse_jev_response(body, QIDS)
    assert r.nouls == {k: 0.7 for k in QIDS}
    assert (r.input_tokens, r.output_tokens, r.model) == (12, 3, MODEL)


def test_parse_error_names_keys():
    with pytest.raises(JevError, match="weird_key"):
        parse_jev_response({"weird_key": 1, "model": MODEL}, QIDS)
    with pytest.raises(JevError):
        parse_jev_response({k: {"noul": 1.5} for k in QIDS}, QIDS)  # out of range
    with pytest.raises(JevError):
        parse_jev_response([1, 2], QIDS)


class _Resp:
    def __init__(self, status, body=None, headers=None, text=""):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, text

    def json(self):
        return self._body


def test_http_client_request_and_errors(monkeypatch):
    import httpx

    sent = {}

    def post(url, headers, json, timeout):
        sent.update(url=url, headers=headers, json=json, timeout=timeout)
        return _Resp(200, {"model": MODEL, **{k: {"type": "noul", "noul": 0.6} for k in QIDS},
                           "usage": {"input_tokens": 5, "output_tokens": 0}})  # fmt: skip

    monkeypatch.setattr(httpx, "post", post)
    c = HttpJevClient("KEY")
    q = build_jev_questions("P", "I", "S")
    r = c.system_one(MODEL, "state", q)
    assert sent["url"] == "https://api.typesafe.ai/v1/systemone"
    assert sent["headers"]["Authorization"] == "Bearer KEY"
    assert sent["headers"]["Content-Type"] == "application/json"
    assert sent["json"] == {"model": MODEL, "state": "state", "questions": q}
    assert r.nouls["issue_ok"] == 0.6 and r.input_tokens == 5

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _Resp(429, headers={"retry-after": "7"})
    )
    with pytest.raises(RateLimited) as e:
        c.system_one(MODEL, "s", q)
    assert e.value.retry_after == 7.0 and not e.value.daily

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp(429))
    with pytest.raises(RateLimited) as e:
        c.system_one(MODEL, "s", q)
    assert e.value.retry_after is None

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp(401, text="bad key"))
    with pytest.raises(JevError, match="401.*bad key"):
        c.system_one(MODEL, "s", q)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp(503, text="down"))
    with pytest.raises(JevError, match="503.*down"):
        c.system_one(MODEL, "s", q)


@pytest.mark.slow
def test_live_jev_call():
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        pytest.skip("TYPESAFE_API_KEY not set")
    q = build_jev_questions("Mortgage", "Trouble during payment process", "(none)")
    state = "My mortgage servicer misapplied my last two payments and then charged late fees."
    r = HttpJevClient(key).system_one(MODEL, state, q)
    assert set(r.nouls) == set(QIDS) and all(0 <= v <= 1 for v in r.nouls.values())
    assert r.nouls["product_ok"] > 0.5
    json.dumps(r.nouls)
