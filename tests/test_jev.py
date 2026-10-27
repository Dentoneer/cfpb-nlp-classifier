import json
import os
from dataclasses import replace

import pandas as pd
import pytest

from nlp.budget import BudgetExceeded, DailyQuotaExhausted, Ledger, RateLimiter
from nlp.llm.client import RateLimited, ServiceUnavailable
from nlp.llm.jev import (
    JEV_PROMPT_VERSION,
    FakeJevClient,
    HttpJevClient,
    JevError,
    JevParseError,
    JevResponse,
    build_jev_questions,
    parse_jev_response,
    run_jev,
    v_jev,
)
from nlp.store import JsonlCache

MODEL = "jev-1.14.0"
TAX = pd.DataFrame(
    {"product": ["P", "P"], "issue": ["I", "I2"], "sub_issue": ["S", "S2"], "path_id": ["a", "b"]}
)
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


def test_questions_three_claim_cases():
    q = build_jev_questions("Mortgage", "Trouble paying", "Foreclosure")
    assert list(q) == ["product_ok", "issue_ok", "sub_issue_ok"]
    assert all(v["type"] == "noul" for v in q.values())
    assert '"Mortgage"' in q["product_ok"]["instructions"]
    assert '"Trouble paying"' in q["issue_ok"]["instructions"]
    assert q["sub_issue_ok"]["instructions"] == (
        'Within the issue "Trouble paying", this complaint is specifically about: "Foreclosure".'
    )
    assert "(none)" not in json.dumps(q) and "no more specific" not in json.dumps(q)


def test_run_jev_always_asks_all_three_claims(cfg, tmp_path):
    client, ledger, cache, limiter = make(tmp_path)
    r = rows(2)
    r["issue"], r["sub_issue"] = "I2", "S2"
    out = run_jev(r, cfg, client, ledger, cache, taxonomy=TAX, limiter=limiter)
    assert all(set(c[2]) == {"product_ok", "issue_ok", "sub_issue_ok"} for c in client.calls)
    assert out["sub_issue_ok"].notna().all() and len(out) == 2


def test_v_jev_missing_level_path_product():
    res = pd.DataFrame(
        {
            "complaint_id": ["a"],
            "product_ok": [0.9],
            "issue_ok": [0.8],
            "sub_issue_ok": [float("nan")],
            "model_id": MODEL,
        }
    )
    out = v_jev(res)
    assert sorted(out.level) == ["issue", "path", "product"]
    assert out.set_index("level").loc["path", "score"] == pytest.approx(0.72)


def test_unparseable_paid_response_is_still_charged(cfg, tmp_path):
    def bad(model, state, questions):
        raise JevParseError("unreadable", 800)

    client, ledger, cache, limiter = make(tmp_path, bad)
    with pytest.raises(JevError):
        run_jev(rows(1), cfg, client, ledger, cache, taxonomy=TAX, limiter=limiter)
    assert ledger.spent() == pytest.approx(800 * 0.042 / 1_000_000)
    assert cache.keys() == set()

    def bad_no_usage(model, state, questions):
        raise JevParseError("unreadable")

    client, ledger2, cache, limiter = make(tmp_path / "x", bad_no_usage)
    with pytest.raises(JevError):
        run_jev(rows(1), cfg, client, ledger2, cache, taxonomy=TAX, limiter=limiter)
    assert ledger2.spent() > 0  # estimate recorded


def test_http_unreadable_200_carries_usage(monkeypatch):
    import httpx

    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: _Resp(200, {"model": MODEL, "weird": 1, "usage": {"input_tokens": 321}}),
    )  # fmt: skip
    with pytest.raises(JevParseError) as e:
        HttpJevClient("K").system_one(MODEL, "s", build_jev_questions("P", "I", "S"))
    assert e.value.input_tokens == 321 and "weird" in str(e.value)


def test_http_client_without_key_raises_clear_error():
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        HttpJevClient(None).system_one(MODEL, "s", build_jev_questions("P", "I", "S"))


def test_reserve_uses_safety_factor():
    from nlp.llm.jev import RESERVE_SAFETY, estimate_jev_tokens

    q = build_jev_questions("P", "I", "S")
    raw = len(json.dumps({"state": "abc", "questions": q})) / 4
    assert RESERVE_SAFETY == 1.5 and estimate_jev_tokens("abc", q) == int(raw * 1.5)


def test_request_shape_and_results(cfg, tmp_path):
    client, ledger, cache, limiter = make(tmp_path)
    out = run_jev(rows(), cfg, client, ledger, cache, taxonomy=TAX, limiter=limiter)
    assert len(client.calls) == 3  # one request per complaint
    model, state, questions = client.calls[0]
    assert model == MODEL and state == "text 0"
    assert set(questions) == {"product_ok", "issue_ok", "sub_issue_ok"}
    assert '"P"' in questions["product_ok"]["instructions"]
    assert len(out) == 3
    assert out.set_index("complaint_id").loc["c1", "issue_ok"] == 0.2
    assert cache.get(f"c0|jev|{MODEL}|{JEV_PROMPT_VERSION}") is not None
    # 1000 input tokens x 3 requests at $0.042/1M, output free
    assert ledger.spent() == pytest.approx(3 * 1000 * 0.042 / 1_000_000)


def test_cache_hit_is_free(cfg, tmp_path):
    client, ledger, cache, limiter = make(tmp_path)
    run_jev(rows(), cfg, client, ledger, cache, taxonomy=TAX, limiter=limiter)
    n, spent = len(client.calls), ledger.spent()
    out = run_jev(rows(), cfg, client, ledger, cache, taxonomy=TAX, limiter=limiter)
    assert len(client.calls) == n and ledger.spent() == spent and len(out) == 3


def test_label_change_is_rejudged(cfg, tmp_path):
    client, ledger, cache, limiter = make(tmp_path)
    r = rows()
    run_jev(r, cfg, client, ledger, cache, taxonomy=TAX, limiter=limiter)
    n = len(client.calls)
    r2 = r.copy()
    r2.loc[1, "issue"] = "Other issue"
    out = run_jev(r2, cfg, client, ledger, cache, taxonomy=TAX, limiter=limiter)
    assert len(client.calls) == n + 1
    assert client.calls[-1][1] == "text 1"
    assert '"Other issue"' in client.calls[-1][2]["issue_ok"]["instructions"]
    assert len(out) == 3


def test_spend_recorded_before_cache_put(cfg, tmp_path):
    client, ledger, cache, limiter = make(tmp_path)
    seen = []
    real_put = cache.put

    def put(key, value):
        seen.append(ledger.spent())
        real_put(key, value)

    cache.put = put
    run_jev(rows(1), cfg, client, ledger, cache, taxonomy=TAX, limiter=limiter)
    assert seen and seen[0] > 0


def test_missing_usage_records_estimate(cfg, tmp_path):
    client, ledger, cache, limiter = make(
        tmp_path, lambda m, s, q: JevResponse({k: 0.5 for k in q}, None, None, m)
    )
    run_jev(rows(1), cfg, client, ledger, cache, taxonomy=TAX, limiter=limiter)
    assert ledger.spent() > 0


def test_budget_stop_is_explicit(cfg, tmp_path):
    client, _, cache, limiter = make(tmp_path)
    tiny = Ledger(tmp_path / "tiny.jsonl", 1e-12, PRICES)
    with pytest.raises(BudgetExceeded, match="3 rows remaining"):
        run_jev(rows(), cfg, client, tiny, cache, taxonomy=TAX, limiter=limiter)
    assert client.calls == []


def test_budget_stops_midway_after_caching(cfg, tmp_path):
    client, _, cache, limiter = make(tmp_path)
    one = 1000 * 0.042 / 1_000_000
    ledger = Ledger(tmp_path / "mid.jsonl", one * 2.05, PRICES)
    with pytest.raises(BudgetExceeded, match="1 rows remaining"):
        run_jev(rows(), cfg, client, ledger, cache, taxonomy=TAX, limiter=limiter)
    assert len(client.calls) == 2  # third would cross the cap
    assert ledger.spent() <= ledger.cap_usd
    out = run_jev(rows(), cfg, client, make(tmp_path / "x")[1], cache, taxonomy=TAX,
                  limiter=limiter)  # fmt: skip
    assert len(out) == 3 and len(client.calls) == 3  # rerun pays only for the missing row


def test_429_backoff_honours_retry_after(cfg, tmp_path):
    attempts = []

    def flaky(model, state, questions):
        attempts.append(1)
        if len(attempts) <= 2:
            raise RateLimited(30.0)
        return responder(model, state, questions)

    client, ledger, cache, limiter = make(tmp_path, flaky)
    sleeps = []
    out = run_jev(rows(1), cfg, client, ledger, cache, taxonomy=TAX, limiter=limiter, sleep=sleeps.append)
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
    run_jev(rows(1), cfg, client, ledger, cache, taxonomy=TAX, limiter=limiter, sleep=sleeps.append)
    assert sleeps == [2.0, 4.0]


def test_429_gives_up_after_max_attempts(cfg, tmp_path):
    def always(model, state, questions):
        raise RateLimited(0.0)

    client, ledger, cache, limiter = make(tmp_path, always)
    with pytest.raises(RateLimited):
        run_jev(rows(1), cfg, client, ledger, cache, taxonomy=TAX, limiter=limiter, sleep=lambda s: None)
    assert ledger.spent() == 0.0


def test_daily_quota_raises(cfg, tmp_path):
    def daily(model, state, questions):
        raise RateLimited(None, daily=True)

    client, ledger, cache, limiter = make(tmp_path, daily)
    with pytest.raises(DailyQuotaExhausted) as e:
        run_jev(rows(), cfg, client, ledger, cache, taxonomy=TAX, limiter=limiter)
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
    with pytest.raises(ServiceUnavailable, match="503.*down"):  # retried by the runner
        c.system_one(MODEL, "s", q)


@pytest.mark.slow
def test_live_jev_call():
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        pytest.skip("TYPESAFE_API_KEY not set")
    q = build_jev_questions("Mortgage", "Trouble during payment process", "Problem with additional add-on products or services purchased with the loan")
    state = "My mortgage servicer misapplied my last two payments and then charged late fees."
    r = HttpJevClient(key).system_one(MODEL, state, q)
    assert set(r.nouls) == set(QIDS) and all(0 <= v <= 1 for v in r.nouls.values())
    assert r.nouls["product_ok"] > 0.5
    json.dumps(r.nouls)


# ---- final review fixes ------------------------------------------------------------------


def test_jev_server_error_retried_then_succeeds(cfg, tmp_path):
    attempts = []

    def flaky(model, state, questions):
        attempts.append(1)
        if len(attempts) == 1:
            raise ServiceUnavailable("Jev HTTP 503")
        return responder(model, state, questions)

    client, ledger, cache, limiter = make(tmp_path, flaky)
    sleeps = []
    out = run_jev(rows(1), cfg, client, ledger, cache, taxonomy=TAX, limiter=limiter,
                  sleep=sleeps.append)  # fmt: skip
    assert len(out) == 1 and len(attempts) == 2 and sleeps == [2.0]


def test_jev_server_error_bounded(cfg, tmp_path):
    def down(model, state, questions):
        raise ServiceUnavailable("Jev HTTP 503")

    client, ledger, cache, limiter = make(tmp_path, down)
    with pytest.raises(ServiceUnavailable, match="5 attempts"):
        run_jev(rows(1), cfg, client, ledger, cache, taxonomy=TAX, limiter=limiter,
                sleep=lambda s: None)  # fmt: skip
    assert len(client.calls) == 5 and ledger.spent() == 0.0


@pytest.mark.parametrize("kind", ["503", "timeout"])
def test_http_jev_client_maps_5xx_and_timeouts(monkeypatch, kind):
    import httpx

    def post(*a, **k):
        if kind == "timeout":
            raise httpx.ReadTimeout("timed out")
        return _Resp(503, text="overloaded")

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(ServiceUnavailable):
        HttpJevClient("key").system_one(MODEL, "s", build_jev_questions("P", "I", "S"))


@pytest.mark.parametrize("bad", ["12", None, -5, float("nan"), True])
def test_non_numeric_usage_falls_back_to_estimate(cfg, tmp_path, bad):
    def weird(model, state, questions):
        r = responder(model, state, questions)
        r.input_tokens = bad
        return r

    client, ledger, cache, limiter = make(tmp_path, weird)
    out = run_jev(rows(1), cfg, client, ledger, cache, taxonomy=TAX, limiter=limiter)
    assert len(out) == 1
    rec = json.loads((tmp_path / "l.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(rec["input_tokens"], int) and rec["input_tokens"] > 0
