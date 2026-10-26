import pandas as pd

from nlp import store


def test_stage_runs_once(tmp_cfg):
    calls = []

    def fn():
        calls.append(1)
        return pd.DataFrame({"a": [1]})

    store.stage(tmp_cfg, "x", fn)
    out = store.stage(tmp_cfg, "x", fn)
    assert calls == [1]
    assert out["a"].tolist() == [1]
    assert store.exists(tmp_cfg, "x")
    assert not list(tmp_cfg.artifacts_dir.glob("*.tmp"))


def test_jsonl_cache_survives_reopen(tmp_path):
    c = store.JsonlCache(tmp_path / "c.jsonl")
    c.put("k", {"v": 1})
    assert store.JsonlCache(tmp_path / "c.jsonl").get("k") == {"v": 1}


def test_jsonl_cache_ignores_truncated_last_line(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"key":"a","value":{}}\n{"key":"b","val')
    assert store.JsonlCache(p).keys() == {"a"}


def test_jsonl_cache_put_after_truncated_line(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"key":"a","value":{}}\n{"key":"b","val')
    c = store.JsonlCache(p)
    c.put("c", {"v": 2})
    assert store.JsonlCache(p).keys() == {"a", "c"}
