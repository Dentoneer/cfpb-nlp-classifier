import json
from pathlib import Path

from nlp.llm.client import FakeClient
from nlp.smoke import make_fake_client, run_smoke


def test_smoke_end_to_end(tmp_path: Path):
    art = tmp_path / "artifacts"
    client = make_fake_client()
    out = run_smoke(art, client=client)

    assert isinstance(out["go"], bool)
    assert out["n_classifiers"] == 4 and out["n_levels"] == 4
    assert set(out["classifiers"]) == {"gemini", "tfidf", "embed_lr", "knn"}
    assert out["validator_board_shape"][0] > 0
    assert 0 < out["spend_usd"] <= 4.00
    assert len(client.calls) > 0
    assert out["gemini_path_accuracy"] > 0.8
    assert out["jeb_path_catch_rate"] > 0.5
    assert set(out["adjusted"]) == {"product", "issue", "sub_issue", "path"}
    assert {m for m, *_ in client.calls} == {"gemini-2.5-flash", "gemini-3.1-flash-lite"}

    # Resume: everything is read back from Parquet / caches, zero new Gemini calls.
    before = len(client.calls)
    again = run_smoke(art, client=client)
    assert len(client.calls) == before
    assert again["spend_usd"] == out["spend_usd"]
    assert json.dumps(again["go"]) == json.dumps(out["go"])


def test_smoke_cache_alone_prevents_calls(tmp_path: Path):
    """Even with the Parquet stages deleted, the JSONL caches make a rerun free."""
    art = tmp_path / "artifacts"
    client = make_fake_client()
    run_smoke(art, client=client)
    before = len(client.calls)
    for p in Path(art).glob("*.parquet"):
        p.unlink()
    run_smoke(art, client=client)
    assert len(client.calls) == before


def test_fake_client_type():
    assert isinstance(make_fake_client(), FakeClient)


def test_fake_responder_rejects_drifted_prompts():
    import pytest

    client = make_fake_client()
    schema = {"items": {"properties": {"product": {}}}}
    with pytest.raises(ValueError):
        client.generate("m", "sys", "no ids here", schema, "low", 100)
    jeb = {"items": {"properties": {"product_ok": {}}}}
    with pytest.raises(ValueError):
        client.generate("m", "sys", "complaint_id: c00001\nnarrative: x", jeb, "low", 100)
