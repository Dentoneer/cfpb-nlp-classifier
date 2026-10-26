import json
import re
from dataclasses import replace

import pandas as pd
import pytest

from nlp.budget import Ledger, RateLimiter
from nlp.llm.client import FakeClient, LLMResponse
from nlp.llm.judge import judge_classifications, judge_summaries
from nlp.store import JsonlCache

PRICES = {"m": (1.0, 1.0), "j": (1.0, 1.0)}


@pytest.fixture
def cfg(tmp_cfg):
    return replace(tmp_cfg, classifier_model="m", judge_model="j", prices=PRICES, batch_size=2)


def rows(n=3):
    return pd.DataFrame(
        {
            "complaint_id": [f"c{i}" for i in range(n)],
            "narrative": [f"text {i}" for i in range(n)],
            "product": "P",
            "issue": "I",
            "sub_issue": "S",
            "summary": "sum",
        }
    )


def make(tmp_path, responder):
    return (
        FakeClient(responder),
        Ledger(tmp_path / "l.jsonl", 4.0, PRICES),
        JsonlCache(tmp_path / "c.jsonl"),
        RateLimiter(10_000, sleep=lambda s: None),
    )


def jeb_responder(model, system, user, schema, thinking):
    ids = re.findall(r"complaint_id: (\S+)", user)
    items = [
        {"complaint_id": i, "product_ok": True, "issue_ok": i != "c1", "sub_issue_ok": False}
        for i in ids
    ]
    return LLMResponse(json.dumps(items), 50, 50)


def test_jeb_refuses_same_model(cfg, tmp_path):
    client, ledger, cache, limiter = make(tmp_path, jeb_responder)
    with pytest.raises(ValueError):
        judge_classifications(rows(), cfg, client, ledger, cache, model="m", limiter=limiter)
    bad = replace(cfg, judge_model="m")
    with pytest.raises(ValueError):
        judge_classifications(rows(), bad, client, ledger, cache, limiter=limiter)
    assert client.calls == []


def test_jeb_results_and_cache(cfg, tmp_path):
    client, ledger, cache, limiter = make(tmp_path, jeb_responder)
    out = judge_classifications(rows(), cfg, client, ledger, cache, limiter=limiter)
    assert set(out.validator) == {"jeb"}
    assert len(out) == 12  # 3 rows x 4 levels
    g = out.set_index(["complaint_id", "level"])
    assert g.loc[("c0", "product"), "score"] == 1.0 and g.loc[("c0", "product"), "verdict"]
    assert g.loc[("c1", "issue"), "score"] == 0.0
    assert g.loc[("c0", "path"), "score"] == 0.0  # sub_issue not ok -> path wrong
    assert cache.get(f"c0|jeb|j|{cfg.prompt_version}") is not None
    assert ledger.spent() > 0
    n = len(client.calls)
    judge_classifications(rows(), cfg, client, ledger, cache, limiter=limiter)
    assert len(client.calls) == n  # fully cached


def test_jeb_max_output_tokens_and_estimate_fallback(cfg, tmp_path):
    def responder(model, system, user, schema, thinking):
        ids = re.findall(r"complaint_id: (\S+)", user)
        items = [
            {"complaint_id": i, "product_ok": True, "issue_ok": True, "sub_issue_ok": True}
            for i in ids
        ]
        return LLMResponse(json.dumps(items), None, None)

    client, ledger, cache, limiter = make(tmp_path, responder)
    judge_classifications(rows(2), cfg, client, ledger, cache, limiter=limiter)
    assert client.calls[0][5] > 0
    assert ledger.spent() > 0  # estimate recorded when usage is missing


def test_jeb_budget_stop(cfg, tmp_path):
    client, _, cache, limiter = make(tmp_path, jeb_responder)
    tiny = Ledger(tmp_path / "tiny.jsonl", 0.0000001, PRICES)
    out = judge_classifications(rows(), cfg, client, tiny, cache, limiter=limiter)
    assert out.empty and client.calls == []


def test_judge_summaries(cfg, tmp_path):
    def responder(model, system, user, schema, thinking):
        ids = re.findall(r"complaint_id: (\S+)", user)
        items = [
            {
                "complaint_id": i,
                "faithful": True,
                "covers_main_issue": i != "c0",
                "invented_details": False,
            }
            for i in ids
        ]
        return LLMResponse(json.dumps(items), 10, 10)

    client, ledger, cache, limiter = make(tmp_path, responder)
    out = judge_summaries(rows(), cfg, client, ledger, cache, limiter=limiter)
    assert list(out.columns) == ["complaint_id", "faithful", "covers_main_issue", "invented_details"]
    assert len(out) == 3
    assert not out.set_index("complaint_id").loc["c0", "covers_main_issue"]
    assert cache.get(f"c0|summary_judge|j|{cfg.prompt_version}") is not None
    with pytest.raises(ValueError):
        judge_summaries(rows(), cfg, client, ledger, cache, model="m", limiter=limiter)
