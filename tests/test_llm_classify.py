import json
import os
import re
from dataclasses import replace

import pandas as pd
import pytest

from nlp.budget import BudgetExceeded, DailyQuotaExhausted, Ledger, RateLimiter
from nlp.llm.classify import (
    RESPONSE_SCHEMA,
    build_prompt,
    estimate_request_tokens,
    parse_and_validate,
    run_classification,
)
from nlp.llm.client import FakeClient, LLMResponse, RateLimited, ServiceUnavailable
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
            "sub_issue": ["Late fee", "Stolen card", "Wrong amount"],
        }
    )
    return build_taxonomy(df)


def descs(tax):
    return pd.DataFrame({"label": tax["label"], "description": [f"About {x}." for x in tax["label"]],
                         "model_id": "dm", "describe_version": "d1"})  # fmt: skip


@pytest.fixture
def cfg(tmp_cfg):
    return replace(tmp_cfg, classifier_model="m", prices=PRICES, batch_size=3, spend_cap_usd=4.0)


def rows(n):
    return pd.DataFrame(
        {"complaint_id": [f"c{i}" for i in range(n)], "narrative": [f"text {i}" for i in range(n)]}
    )


def item(cid, product="Credit card", issue="Billing dispute", sub="Late fee", conf=0.9):
    return {
        "complaint_id": cid,
        "label": f"{product}|{issue}|{sub}",
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
        rows(n), cfg, client, ledger, cache, taxonomy=taxonomy, descriptions=descs(taxonomy),
        limiter=limiter, sleep=sleep,
    )  # fmt: skip


def test_parse_normalizes_case_and_whitespace(taxonomy):
    raw = json.dumps([item("a", "  credit CARD", "billing   dispute", "LATE fee ")])
    df, retry = parse_and_validate(raw, ["a"], taxonomy)
    assert retry == []
    r = df.iloc[0]
    assert (r["product"], r["issue"], r["sub_issue"]) == (
        "Credit card",
        "Billing dispute",
        "Late fee",
    )


def test_parse_exact_label_is_split_into_levels(taxonomy):
    raw = json.dumps([{"complaint_id": "a", "label": "Mortgage|Escrow|Wrong amount",
                       "summary": "s", "confidence": 0.5}])  # fmt: skip
    df, retry = parse_and_validate(raw, ["a"], taxonomy)
    assert retry == [] and tuple(df.iloc[0][["product", "issue", "sub_issue"]]) == (
        "Mortgage", "Escrow", "Wrong amount")


@pytest.mark.parametrize("label", ["Mortgage|Escrow", "Mortgage|Escrow|Nope", "", None,
                                   "Mortgage / Escrow / Wrong amount"])
def test_parse_invalid_label_is_retried(taxonomy, label):
    raw = json.dumps([{"complaint_id": "a", "label": label, "summary": "s", "confidence": 0.5}])
    df, retry = parse_and_validate(raw, ["a"], taxonomy)
    assert df.empty and retry == ["a"]


def test_parse_handles_missing_duplicate_unknown_ids(taxonomy):
    first = item("a", sub="Late fee")
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
    assert est_in == 300  # chars/4 x 1.5 safety
    assert est_out == int(10 * 150 * 1.5)
    assert estimate_request_tokens("", "", 10, "low")[1] == int(10 * (150 + 512) * 1.5)
    assert estimate_request_tokens("", "", 10, "high")[1] == int(10 * (150 + 2048) * 1.5)
    assert estimate_request_tokens("", "", 1, "minimal")[1] == int(150 * 1.5)


def test_build_prompt_contents():
    system, user = build_prompt(rows(2), "TAXTEXT", "v2")
    assert "TAXTEXT" in system and "60 words" in system and "exactly one" in system
    assert "verbatim" in system and "label" in system and "(none)" not in system
    assert RESPONSE_SCHEMA["items"]["required"] == ["complaint_id", "label", "summary", "confidence"]
    assert "complaint_id: c0" in user and "complaint_id: c1" in user
    assert RESPONSE_SCHEMA["type"] == "array"


def test_every_prediction_has_model_and_prompt_version(tmp_path, cfg, taxonomy):
    client, ledger, cache, limiter = make(tmp_path, good_responder())
    out = run(cfg, client, ledger, cache, limiter, taxonomy)
    assert len(out) == 7
    assert set(PREDICTIONS_COLS) <= set(out.columns)
    assert (out["model_id"] == "m").all()
    assert out["prompt_version"].str.fullmatch(cfg.prompt_version + r"\+[0-9a-f]{8}").all()
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


def test_budget_exhaustion_raises_after_caching_finished_rows(tmp_path, cfg, taxonomy):
    # Each request records $0.0002 actual; the conservative reserve estimate is far larger, so
    # this cap lets the first requests through, then stops with an explicit BudgetExceeded.
    client, ledger, cache, limiter = make(tmp_path, good_responder(), cap=0.0036)
    with pytest.raises(BudgetExceeded, match="remaining"):
        run(cfg, client, ledger, cache, limiter, taxonomy, n=9)
    assert 0 < len(client.calls) < 3
    assert ledger.spent() <= 0.0036
    done = run(cfg, client, Ledger(tmp_path / "big.jsonl", 4.0, PRICES), cache, limiter,
               taxonomy, n=9)  # fmt: skip
    assert len(done) == 9  # finished rows were cached; the rerun completes the rest


def test_budget_exhausted_at_start_raises(tmp_path, cfg, taxonomy):
    client, ledger, cache, limiter = make(tmp_path, good_responder(), cap=0.0)
    with pytest.raises(BudgetExceeded):
        run(cfg, client, ledger, cache, limiter, taxonomy)
    assert client.calls == []


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


def test_classifier_name_is_part_of_cache_key(tmp_path, cfg, taxonomy):
    client, ledger, cache, limiter = make(tmp_path, good_responder())
    run(cfg, client, ledger, cache, limiter, taxonomy, n=3)
    assert len(client.calls) == 1
    out = run_classification(
        rows(3), cfg, client, ledger, cache, classifier_name="sc",
        taxonomy=taxonomy, descriptions=descs(taxonomy), limiter=limiter, sleep=lambda s: None,
    )  # fmt: skip
    assert len(client.calls) == 2
    assert (out["classifier"] == "sc").all() and len(out) == 3


def test_max_output_tokens_equals_reserve_estimate(tmp_path, cfg, taxonomy):
    client, ledger, cache, limiter = make(tmp_path, good_responder())
    run(cfg, client, ledger, cache, limiter, taxonomy, n=3)
    _model, system, user, _, thinking, max_out = client.calls[0]
    assert max_out == estimate_request_tokens(system, user, 3, thinking)[1]


def test_missing_usage_records_conservative_estimate(tmp_path, cfg, taxonomy):
    good = good_responder()

    def responder(model, system, user, schema, thinking):
        r = good(model, system, user, schema, thinking)
        return LLMResponse(r.text, None, None, None)

    client, ledger, cache, limiter = make(tmp_path, responder)
    out = run(cfg, client, ledger, cache, limiter, taxonomy, n=3)
    _, system, user, _, thinking, _ = client.calls[0]
    est = estimate_request_tokens(system, user, 3, thinking)
    assert len(out) == 3
    assert ledger.spent() == pytest.approx(ledger.cost("m", *est))


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
        r, cfg, GenAIClient(os.environ["GEMINI_API_KEY"]), ledger, cache, taxonomy=taxonomy,
        descriptions=descs(taxonomy),
    )
    assert len(out) == 2
    paths = set(zip(taxonomy["product"], taxonomy["issue"], taxonomy["sub_issue"]))
    for _, x in out.iterrows():
        assert (x["product"], x["issue"], x["sub_issue"]) in paths


# ---- v2: descriptions ---------------------------------------------------------------------


def test_classification_needs_descriptions(tmp_path, cfg, taxonomy):
    from nlp.llm.describe import DescriptionsMissing

    client, ledger, cache, limiter = make(tmp_path, good_responder())
    with pytest.raises(DescriptionsMissing, match=r"Run Part 0 \(descriptions\) in notebook 03 first"):
        run_classification(rows(2), cfg, client, ledger, cache, taxonomy=taxonomy, limiter=limiter)
    assert client.calls == [] and ledger.spent() == 0.0


def test_classification_reads_saved_descriptions(tmp_path, cfg, taxonomy):
    from nlp import store
    from nlp.llm.describe import STAGE_DESCRIPTIONS

    store.write(cfg, STAGE_DESCRIPTIONS, descs(taxonomy))
    client, ledger, cache, limiter = make(tmp_path, good_responder())
    out = run_classification(rows(2), cfg, client, ledger, cache, taxonomy=taxonomy,
                             limiter=limiter, sleep=lambda s: None)  # fmt: skip
    assert len(out) == 2
    system = client.calls[0][1]
    assert "- Credit card|Fraud|Stolen card: About Credit card|Fraud|Stolen card." in system


def test_reserve_grows_with_description_length(tmp_path, cfg, taxonomy):
    short, long = descs(taxonomy), descs(taxonomy)
    long["description"] = long["description"] + " word" * 40
    seen = {}
    for name, d in (("short", short), ("long", long)):
        client, ledger, cache, limiter = make(tmp_path / name, good_responder())
        run_classification(rows(3), cfg, client, ledger, cache, taxonomy=taxonomy,
                           descriptions=d, limiter=limiter, sleep=lambda s: None)  # fmt: skip
        _, system, user, _, thinking, _ = client.calls[0]
        seen[name] = estimate_request_tokens(system, user, 3, thinking)[0]
        assert seen[name] >= int(len(system) / 4 * 1.5)
    assert seen["long"] > seen["short"]


def test_edited_descriptions_are_not_answered_from_cache(tmp_path, cfg, taxonomy):
    client, ledger, cache, limiter = make(tmp_path, good_responder())
    run(cfg, client, ledger, cache, limiter, taxonomy, n=3)
    assert len(client.calls) == 1
    edited = descs(taxonomy)
    edited.loc[0, "description"] = "Better wording."
    run_classification(rows(3), cfg, client, ledger, cache, taxonomy=taxonomy,
                       descriptions=edited, limiter=limiter, sleep=lambda s: None)  # fmt: skip
    assert len(client.calls) == 2  # new wording -> asked again
    run(cfg, client, ledger, cache, limiter, taxonomy, n=3)
    assert len(client.calls) == 2  # original wording still cached


# ---- final review fixes ------------------------------------------------------------------


def test_server_error_retried_with_bounded_backoff(tmp_path, cfg, taxonomy):
    good = good_responder()
    fails = []

    def flaky(model, system, user, schema, thinking):
        if not fails:
            fails.append(1)
            raise ServiceUnavailable("503 overloaded")
        return good(model, system, user, schema, thinking)

    client, ledger, cache, limiter = make(tmp_path, flaky)
    sleeps = []
    out = run(cfg, client, ledger, cache, limiter, taxonomy, n=3, sleep=sleeps.append)
    assert len(out) == 3 and len(client.calls) == 2 and sleeps == [2.0]


def test_server_error_gives_up_with_clear_message(tmp_path, cfg, taxonomy):
    def down(model, system, user, schema, thinking):
        raise ServiceUnavailable("503 overloaded")

    client, ledger, cache, limiter = make(tmp_path, down)
    with pytest.raises(ServiceUnavailable, match="5 attempts") as e:
        run(cfg, client, ledger, cache, limiter, taxonomy, n=3)
    assert "nothing was charged" in str(e.value).lower()
    assert len(client.calls) == 5 and ledger.spent() == 0.0


def _genai_with(exc):
    from types import SimpleNamespace

    from nlp.llm.client import GenAIClient

    def boom(**kw):
        raise exc

    c = GenAIClient.__new__(GenAIClient)
    c._client = SimpleNamespace(models=SimpleNamespace(generate_content=boom))
    c._quota_logged = False
    return c


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # google-genai on 3.14
@pytest.mark.parametrize("kind", ["server", "timeout"])
def test_genai_client_maps_server_errors_and_timeouts(kind):
    import httpx
    from google.genai import errors

    exc = (
        errors.ServerError(503, {"error": {"code": 503, "message": "overloaded"}})
        if kind == "server"
        else httpx.ReadTimeout("timed out")
    )
    with pytest.raises(ServiceUnavailable):
        _genai_with(exc).generate("gemini-2.5-flash", "s", "u", RESPONSE_SCHEMA, "low", 10)


def test_reserve_estimate_has_input_safety_factor():
    est_in, est_out = estimate_request_tokens("s" * 400, "u" * 400, 2, "low")
    assert est_in == int(800 / 4 * 1.5)
    assert est_out == int(2 * (150 + 512) * 1.5)
