import json
import re
from dataclasses import replace

import pandas as pd
import pytest

from nlp.budget import Ledger, RateLimiter
from nlp.llm.client import FakeClient, LLMResponse
from nlp.llm.judge import judge_summaries
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
    assert cache.get("c0|summary_judge|j|sj-v1") is not None
    with pytest.raises(ValueError):
        judge_summaries(rows(), cfg, client, ledger, cache, model="m", limiter=limiter)


def test_summary_changed_is_rejudged(cfg, tmp_path):
    def responder(model, system, user, schema, thinking):
        ids = re.findall(r"complaint_id: (\S+)", user)
        return LLMResponse(
            json.dumps(
                [
                    {"complaint_id": i, "faithful": True, "covers_main_issue": True,
                     "invented_details": False}
                    for i in ids
                ]
            ),
            10,
            10,
        )

    client, ledger, cache, limiter = make(tmp_path, responder)
    r = rows(3)
    judge_summaries(r, cfg, client, ledger, cache, limiter=limiter)
    n = len(client.calls)
    r.loc[2, "summary"] = "new summary"
    judge_summaries(r, cfg, client, ledger, cache, limiter=limiter)
    assert len(client.calls) == n + 1


# ---- final review fixes ------------------------------------------------------------------


def test_judge_budget_stop_is_explicit(cfg, tmp_path):
    from nlp.budget import BudgetExceeded

    client, _, cache, limiter = make(tmp_path, lambda *a: LLMResponse("[]", 1, 1))
    tiny = Ledger(tmp_path / "tiny.jsonl", 0.0, PRICES)
    with pytest.raises(BudgetExceeded):
        judge_summaries(rows(), cfg, client, tiny, cache, limiter=limiter)
    assert client.calls == []
