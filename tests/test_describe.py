import json
import re
from dataclasses import replace

import pandas as pd
import pytest

from nlp import store
from nlp.budget import BudgetExceeded, DailyQuotaExhausted, Ledger, RateLimiter
from nlp.errors import StageIncomplete
from nlp.llm.client import FakeClient, LLMResponse, RateLimited, RequestRejected
from nlp.llm.describe import (
    DESCRIBE_SCHEMA,
    DESCRIBE_VERSION,
    STAGE_DESCRIPTIONS,
    DescriptionError,
    DescriptionsMissing,
    apply_edits,
    build_describe_prompt,
    estimate_descriptions,
    evidence_by_label,
    load_descriptions,
    run_descriptions,
    validate_descriptions,
)
from nlp.schema import DESCRIPTIONS_COLS
from nlp.store import JsonlCache
from nlp.taxonomy import build_taxonomy

PRICES = {"dm": (1.0, 1.0)}
PATHS = [
    ("Credit card", "Fees", "Late fee"),
    ("Credit card", "Fees", "Annual fee"),
    ("Credit card", "Fraud", "Stolen card"),
    ("Mortgage", "Escrow", "Wrong amount"),
]


def _data(per_path=20):
    rows, splits = [], []
    n = 0
    for p, i, s in PATHS:
        for k in range(per_path):
            cid = f"c{n:04d}"
            rows.append({"complaint_id": cid, "product": p, "issue": i, "sub_issue": s,
                         "narrative": f"{s} story {k} " + "detail " * 200})  # fmt: skip
            split = "train" if k < 17 else ("tuning" if k < 19 else "test")
            splits.append({"complaint_id": cid, "split": split})
            n += 1
    complaints = pd.DataFrame(rows)
    return complaints, pd.DataFrame(splits), build_taxonomy(complaints)


@pytest.fixture
def cfg(tmp_cfg):
    return replace(tmp_cfg, describe_model="dm", prices=PRICES, spend_cap_usd=5.0)


def good_responder(calls=None):
    def responder(model, system, user, schema, thinking):
        labels = re.findall(r"^LABEL: (.+)$", user, re.MULTILINE)
        items = [{"label": x, "description": f"Covers {x}. Use when it fits."} for x in labels]
        return LLMResponse(json.dumps(items), 1000, 200)

    return responder


def tools(tmp_path, responder, cap=5.0):
    return (
        FakeClient(responder),
        Ledger(tmp_path / "ledger.jsonl", cap, PRICES),
        JsonlCache(tmp_path / "gemini_cache.jsonl"),
        RateLimiter(10_000, sleep=lambda s: None),
    )


def run(cfg, client, ledger, cache, limiter, data=None):
    complaints, splits, tax = data or _data()
    return run_descriptions(complaints, splits, tax, cfg, client, ledger, cache,
                            limiter=limiter, sleep=lambda s: None)  # fmt: skip


def test_evidence_train_only_capped_and_seeded(cfg):
    complaints, splits, tax = _data()
    ev = evidence_by_label(complaints, splits, tax, cfg)
    train = set(splits.loc[splits["split"] == "train", "complaint_id"])
    assert set(ev) == set(tax["label"])
    for texts in ev.values():
        assert len(texts) == 15  # 17 train rows, capped at 15
        assert all(len(t) <= 600 for t in texts)
    # never tuning/test text (their story numbers are 17-19)
    assert len(train) == 4 * 17
    used = {t.split(" story ")[1].split()[0] for ts in ev.values() for t in ts}
    assert not used & {"17", "18", "19"}
    assert ev == evidence_by_label(complaints, splits, tax, cfg)  # deterministic
    other = evidence_by_label(complaints, splits, tax, replace(cfg, seed=1))
    assert other != ev


def test_prompt_one_request_per_product_lists_all_siblings():
    system, user = build_describe_prompt(
        "Credit card", ["Credit card|Fees|Late fee", "Credit card|Fees|Annual fee"],
        {"Credit card|Fees|Late fee": ["late text"], "Credit card|Fees|Annual fee": []},
    )
    assert "45 words" in system and "Use when" in system and "Not when" in system
    assert "instead" in system and "exactly once" in system
    assert "LABEL: Credit card|Fees|Late fee" in user and "late text" in user
    assert DESCRIBE_SCHEMA["items"]["required"] == ["label", "description"]


def test_run_descriptions_one_call_per_product_and_stage_shape(tmp_path, cfg):
    client, ledger, cache, limiter = tools(tmp_path, good_responder())
    out = run(cfg, client, ledger, cache, limiter)
    assert list(out.columns) == DESCRIPTIONS_COLS
    assert len(client.calls) == 2  # Credit card, Mortgage
    assert sorted(out["label"]) == sorted(build_taxonomy(_data()[0])["label"])
    assert (out["model_id"] == "dm").all() and (out["describe_version"] == DESCRIBE_VERSION).all()
    assert ledger.spent() == pytest.approx(2 * 1200 / 1e6)
    # cached per product: rerun is free
    again = run(cfg, client, ledger, JsonlCache(tmp_path / "gemini_cache.jsonl"), limiter)
    assert len(client.calls) == 2 and again.equals(out)
    keys = JsonlCache(tmp_path / "gemini_cache.jsonl").keys()
    assert f"Credit card|describe|dm|{DESCRIBE_VERSION}" in keys


def test_reserve_before_generate_and_ledger_before_cache(tmp_path, cfg):
    order = []
    client, ledger, cache, limiter = tools(tmp_path, good_responder())
    orig_reserve, orig_record, orig_put = ledger.reserve, ledger.record, cache.put
    ledger.reserve = lambda *a: (order.append("reserve"), orig_reserve(*a))[1]
    ledger.record = lambda *a, **k: (order.append("record"), orig_record(*a, **k))[1]
    cache.put = lambda *a: (order.append("put"), orig_put(*a))[1]
    run(cfg, client, ledger, cache, limiter)
    assert order[:3] == ["reserve", "record", "put"]


def test_invalid_answer_retried_once_then_plain_error(tmp_path, cfg):
    def missing_one(model, system, user, schema, thinking):
        labels = re.findall(r"^LABEL: (.+)$", user, re.MULTILINE)
        items = [{"label": x, "description": "d"} for x in labels[1:]]
        return LLMResponse(json.dumps(items), 10, 10)

    client, ledger, cache, limiter = tools(tmp_path, missing_one)
    with pytest.raises(DescriptionError) as e:
        run(cfg, client, ledger, cache, limiter)
    msg = str(e.value)
    assert "Credit card" in msg and "missing" in msg
    assert isinstance(e.value, StageIncomplete)
    assert len(client.calls) == 2  # first try + one automatic retry
    assert ledger.spent() > 0  # both paid calls recorded
    assert cache.keys() == set()


def test_retry_succeeds(tmp_path, cfg):
    good = good_responder()
    n = {"k": 0}

    def flaky(*a):
        n["k"] += 1
        return LLMResponse("not json", 10, 10) if n["k"] == 1 else good(*a)

    client, ledger, cache, limiter = tools(tmp_path, flaky)
    out = run(cfg, client, ledger, cache, limiter)
    assert len(out) == 4 and len(client.calls) == 3


@pytest.mark.parametrize(
    "items, why",
    [
        ([{"label": "A|b|c", "description": "x"}], "missing"),
        ([{"label": "A|b|c", "description": "x"}, {"label": "A|b|d", "description": "x"},
          {"label": "Z|z|z", "description": "x"}], "unknown"),
        ([{"label": "A|b|c", "description": "x"}, {"label": "A|b|c", "description": "y"},
          {"label": "A|b|d", "description": "x"}], "duplicate"),
        ([{"label": "A|b|c", "description": " "}, {"label": "A|b|d", "description": "x"}],
         "empty"),
    ],
)
def test_validate_descriptions_errors(items, why):
    with pytest.raises(ValueError, match=why):
        validate_descriptions(json.dumps(items), ["A|b|c", "A|b|d"])


def test_validate_maps_near_miss_label_to_exact():
    got = validate_descriptions(
        json.dumps([{"label": " a|B|c ", "description": " x "}, {"label": "A|b|d", "description": "y"}]),
        ["A|b|c", "A|b|d"],
    )
    assert got == [{"label": "A|b|c", "description": "x"}, {"label": "A|b|d", "description": "y"}]


def test_budget_and_quota_stop(tmp_path, cfg):
    client, ledger, cache, limiter = tools(tmp_path, good_responder(), cap=0.0)
    with pytest.raises(BudgetExceeded):
        run(cfg, client, ledger, cache, limiter)
    assert client.calls == []

    def quota(*a):
        raise RateLimited(None, True)

    client, ledger, cache, limiter = tools(tmp_path / "q", quota)
    with pytest.raises(DailyQuotaExhausted):
        run(cfg, client, ledger, cache, limiter)


def test_rejected_request_is_not_billed(tmp_path, cfg):
    def rejected(*a):
        raise RequestRejected(404, "model retired, remove it from the list")

    client, ledger, cache, limiter = tools(tmp_path, rejected)
    with pytest.raises(RequestRejected):
        run(cfg, client, ledger, cache, limiter)
    assert ledger.spent() == 0.0 and len(client.calls) == 1


def test_estimate_is_positive_and_scales_with_price(cfg):
    complaints, splits, tax = _data()
    one = estimate_descriptions(cfg, complaints, splits, tax)
    assert one > 0
    pricier = replace(cfg, prices={"dm": (2.0, 2.0)})
    assert estimate_descriptions(pricier, complaints, splits, tax) == pytest.approx(2 * one)


def _stage(tax):
    return pd.DataFrame({"label": tax["label"], "description": "d", "model_id": "dm",
                         "describe_version": DESCRIBE_VERSION})  # fmt: skip


def test_load_descriptions_missing_or_stale(cfg):
    _, _, tax = _data()
    with pytest.raises(DescriptionsMissing, match=r"Run Part 0 \(descriptions\) in notebook 03 first"):
        load_descriptions(cfg, tax)
    store.write(cfg, STAGE_DESCRIPTIONS, _stage(tax).iloc[1:])
    with pytest.raises(DescriptionsMissing, match="Part 0"):
        load_descriptions(cfg, tax)
    store.write(cfg, STAGE_DESCRIPTIONS, _stage(tax))
    assert len(load_descriptions(cfg, tax)) == len(tax)


def test_apply_edits_validates_and_marks_changes():
    _, _, tax = _data()
    cur = _stage(tax)
    edits = cur[["label", "description"]].copy()
    edits.loc[0, "description"] = "Marco's better text."
    out = apply_edits(cur, edits)
    assert list(out.columns) == DESCRIPTIONS_COLS
    assert out.set_index("label").loc[cur.loc[0, "label"], "description"] == "Marco's better text."
    assert (out["describe_version"] == f"{DESCRIBE_VERSION}-edited").sum() == 1

    renamed = edits.copy()
    renamed.loc[1, "label"] = "Renamed|x|y"
    with pytest.raises(StageIncomplete, match="label"):
        apply_edits(cur, renamed)
    blank = edits.copy()
    blank.loc[2, "description"] = ""
    with pytest.raises(StageIncomplete, match="empty"):
        apply_edits(cur, blank)


def test_estimate_is_zero_once_every_product_is_cached(tmp_path, cfg):
    complaints, splits, tax = _data()
    client, ledger, cache, limiter = tools(tmp_path, good_responder())
    assert estimate_descriptions(cfg, complaints, splits, tax, cache=cache) > 0
    run(cfg, client, ledger, cache, limiter)
    assert estimate_descriptions(cfg, complaints, splits, tax, cache=cache) == 0.0
