import json
from pathlib import Path

from nlp.llm.client import FakeClient
from nlp.smoke import make_fake_client, make_fake_jev_client, run_smoke


def test_smoke_end_to_end(tmp_path: Path):
    art = tmp_path / "artifacts"
    client = make_fake_client()
    jev = make_fake_jev_client()
    out = run_smoke(art, client=client, jev_client=jev)

    assert isinstance(out["go"], bool)
    assert out["n_classifiers"] == 4 and out["n_levels"] == 4
    assert set(out["classifiers"]) == {"gemini", "tfidf", "embed_lr", "knn"}
    assert out["validator_board_shape"][0] > 0
    assert 0 < out["spend_usd"] <= 5.00
    assert out["n_descriptions"] == 40
    assert len(client.calls) > 0
    assert out["gemini_path_accuracy"] > 0.8
    assert out["jev_path_catch_rate"] > 0.5
    assert len(jev.calls) == 400 and {m for m, *_ in jev.calls} == {"jev-1.14.0"}  # 1 per test row
    assert set(out["adjusted"]) == {"product", "issue", "sub_issue", "path"}
    assert {m for m, *_ in client.calls} == {"gemini-3.8-flash", "gemini-3.1-flash-lite"}
    describe_calls = [c for c in client.calls if "description" in c[3]["items"]["properties"]]
    assert len(describe_calls) == 6  # one request per product
    classify_calls = [c for c in client.calls if "confidence" in c[3]["items"]["properties"]]
    assert classify_calls and all("CATEGORIES:\n- " in c[1] for c in classify_calls)
    assert "judge-human agreement" in out["summary_quality"]  # judge results are reported
    assert 0.0 <= out["summary_quality"]["judge: faithful"] <= 1.0

    # Resume: everything is read back from Parquet / caches, zero new Gemini calls.
    before, jbefore = len(client.calls), len(jev.calls)
    again = run_smoke(art, client=client, jev_client=jev)
    assert len(client.calls) == before and len(jev.calls) == jbefore
    assert again["spend_usd"] == out["spend_usd"]
    assert json.dumps(again["go"]) == json.dumps(out["go"])


def test_smoke_cache_alone_prevents_calls(tmp_path: Path):
    """Even with the Parquet stages deleted, the JSONL caches make a rerun free."""
    art = tmp_path / "artifacts"
    client, jev = make_fake_client(), make_fake_jev_client()
    run_smoke(art, client=client, jev_client=jev)
    before, jbefore = len(client.calls), len(jev.calls)
    for p in Path(art).glob("*.parquet"):
        p.unlink()
    run_smoke(art, client=client, jev_client=jev)
    assert len(client.calls) == before and len(jev.calls) == jbefore


def test_fake_client_type():
    assert isinstance(make_fake_client(), FakeClient)


def test_fake_responder_rejects_drifted_prompts():
    import pytest

    client = make_fake_client()
    schema = {"items": {"properties": {"summary": {}}}}
    with pytest.raises(ValueError):
        client.generate("m", "sys", "no ids here", schema, "low", 100)
    with pytest.raises(ValueError, match="description listing"):
        client.generate("m", "no listing", "complaint_id: c00001", {"items": {"properties": {"label": {}}}}, "low", 100)
    with pytest.raises(ValueError, match="no labels"):
        client.generate("m", "s", "nothing", {"items": {"properties": {"description": {}}}}, "low", 100)
    jev = make_fake_jev_client()
    with pytest.raises(ValueError):
        jev.system_one("jev-1.14.0", "unknown narrative", {})


def test_smoke_has_no_placeholder_sub_issue():
    from nlp.smoke import synthetic_data

    data, _ = synthetic_data()
    assert "(none)" not in set(data["sub_issue"]) and data["sub_issue"].str.strip().ne("").all()
