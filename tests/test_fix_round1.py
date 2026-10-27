"""Review fix round 1 (v2): realistic cost projection, Part C vs TF-IDF, Sheet edits safety, ..."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from nlp import colab, store
from nlp import pipeline as P
from nlp.budget import Ledger, RateLimiter
from nlp.llm.classify import estimate_request_tokens, run_classification
from nlp.llm.client import MAX_OUTPUT_TOKENS, FakeClient, LLMResponse, RequestRejected
from nlp.llm.describe import (
    DESCRIBE_VERSION,
    estimate_describe_tokens,
    prompt_tag,
)
from nlp.store import JsonlCache
from nlp.taxonomy import build_taxonomy, normalize_label, taxonomy_prompt_text
from nlp.tools.check_notebooks import code_cells

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "phase1.yaml"
NB = ROOT / "notebooks"


def _cfg(tmp_path):
    return colab.load_config(CONFIG, tmp_path)


def _rows(n=40, prefix="t"):
    subs = ["S1", "S2"]
    return pd.DataFrame({
        "complaint_id": [f"{prefix}{i}" for i in range(n)],
        "product": ["P1"] * n,
        "issue": ["I1"] * n,
        "sub_issue": [subs[i % 2] for i in range(n)],
        "narrative": [f"complaint {i} " * 30 for i in range(n)],
    })  # fmt: skip


def _tax():
    return build_taxonomy(_rows(2))


def _descs(tax, text="d"):
    return pd.DataFrame({"label": tax["label"], "description": text,
                         "model_id": "gemini-3.8-flash", "describe_version": DESCRIBE_VERSION})  # fmt: skip


def _answer(truth):
    t = truth.set_index("complaint_id")

    def responder(model, system, user, schema, thinking):
        ids = re.findall(r"complaint_id: (\S+)", user)
        items = [{"complaint_id": i, "label": f"{t.at[i, 'product']}|{t.at[i, 'issue']}|"
                  f"{t.at[i, 'sub_issue']}", "summary": "s", "confidence": 0.9} for i in ids]
        return LLMResponse(json.dumps(items), 20_000, 2_000)  # a big v2 request

    return responder


def _lim():
    return RateLimiter(10_000, sleep=lambda s: None)


# ---- 1. cost projection uses v2 requests only ----------------------------------------------


def test_ledger_lines_carry_optional_tag_and_rows(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", 5.0, {"m": (1.0, 1.0)})
    led.record("m", "old", 1_000_000, 0)  # v1-style line, no tag
    led.record("m", "new", 1_000_000, 0, tag="v2+abc", rows=10)
    lines = [json.loads(x) for x in led.path.read_text(encoding="utf-8").splitlines()]
    assert "tag" not in lines[0] and lines[1]["tag"] == "v2+abc" and lines[1]["rows"] == 10
    assert led.spent() == pytest.approx(2.0)


def test_classification_ledger_lines_are_tagged_with_prompt_version(tmp_path):
    cfg = _cfg(tmp_path)
    tax = _tax()
    truth = _rows(12)
    led = Ledger(tmp_path / "l.jsonl", 5.0, cfg.prices)
    run_classification(truth, cfg, FakeClient(_answer(truth)), led, JsonlCache(tmp_path / "c.jsonl"),
                       taxonomy=tax, descriptions=_descs(tax), limiter=_lim(),
                       sleep=lambda s: None)  # fmt: skip
    lines = [json.loads(x) for x in led.path.read_text(encoding="utf-8").splitlines()]
    tag = prompt_tag(cfg.prompt_version, _descs(tax))
    assert {x["tag"] for x in lines} == {tag} and [x["rows"] for x in lines] == [10, 2]


def test_measured_estimate_ignores_cheap_v1_lines(tmp_path):
    cfg = _cfg(tmp_path)
    model = cfg.classifier_model
    led = Ledger(tmp_path / "l.jsonl", 5.0, cfg.prices)
    for _ in range(50):  # the first run: cheap v1 requests, untagged
        led.record(model, "v1", 2_000, 1_000)
    rows = _rows(100)
    assert colab.estimate_measured(cfg, led, rows, tag="v2+x") is None  # no v2 lines yet
    led.record(model, "v2", 20_000, 2_000, tag="v2+x", rows=10)
    per_row = led.cost(model, 20_000, 2_000) / 10
    got = colab.estimate_measured(cfg, led, rows, tag="v2+x")
    assert got == pytest.approx(100 * per_row * colab.MEASURED_SAFETY)


def test_phase_estimate_falls_back_to_v2_bound_not_v1_history(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    tax = _tax()
    store.write(cfg, P.STAGE_DESCRIPTIONS, _descs(tax, "a long description " * 20))
    led = Ledger(tmp_path / "l.jsonl", 5.0, cfg.prices)
    for _ in range(50):
        led.record(cfg.classifier_model, "v1", 2_000, 1_000)  # v1 history only
    rows = _rows(40)
    est = colab.phase_estimate(cfg, led, tax, rows, rows.head(10)).set_index("stage")
    g = est.loc[P.STAGE_PRED_GEMINI]
    assert not g["measured"] and g["usd"] == g["bound_usd"]
    tag = prompt_tag(cfg.prompt_version, _descs(tax, "a long description " * 20))
    led.record(cfg.classifier_model, "v2", 20_000, 2_000, tag=tag, rows=10)
    est = colab.phase_estimate(cfg, led, tax, rows, rows.head(10)).set_index("stage")
    assert est.loc[P.STAGE_PRED_GEMINI, "measured"]
    capsys.readouterr()


def test_summary_judge_bound_uses_judge_prompt(tmp_path, capsys):
    from nlp.llm.judge import estimate_summary_judge_usd

    cfg = _cfg(tmp_path)
    tax = _tax()
    store.write(cfg, P.STAGE_DESCRIPTIONS, _descs(tax, "a long description " * 40))
    led = Ledger(tmp_path / "l.jsonl", 5.0, cfg.prices)
    rows = _rows(40)
    est = colab.phase_estimate(cfg, led, tax, rows, rows.head(10)).set_index("stage")
    hc = rows.head(cfg.hand_check_n)
    want = estimate_summary_judge_usd(cfg, hc)
    assert est.loc[P.STAGE_SUMMARY_JUDGE, "bound_usd"] == pytest.approx(want)
    assert want != pytest.approx(colab.estimate_classification(cfg, hc, tax, model=cfg.judge_model))
    capsys.readouterr()


# ---- 2. Part C: v2 vs TF-IDF on the same tuning complaints ---------------------------------


def _tfidf_pred(rows, right_frac=0.5):
    n = int(len(rows) * right_frac)
    sub = [s if i < n else ("S2" if s == "S1" else "S1") for i, s in enumerate(rows["sub_issue"])]
    return pd.DataFrame({"complaint_id": rows["complaint_id"], "classifier": "tfidf",
                         "product": rows["product"], "issue": rows["issue"], "sub_issue": sub})  # fmt: skip


def _part_c_setup(tmp_path, n=40):
    cfg = _cfg(tmp_path)
    tax = _tax()
    store.write(cfg, P.STAGE_DESCRIPTIONS, _descs(tax))
    tune = _rows(n, prefix="u")
    store.write(cfg, P.stage_pred("tfidf"), pd.concat([_tfidf_pred(tune), _tfidf_pred(_rows(5, "x"))]))
    return cfg, tax, tune, Ledger(tmp_path / "l.jsonl", 5.0, cfg.prices), JsonlCache(tmp_path / "c.jsonl")


def test_part_c_compares_with_tfidf_and_prints_v1_reference(tmp_path, capsys):
    cfg, tax, tune, led, cache = _part_c_setup(tmp_path)
    out = colab.prompt_check(cfg, tune, FakeClient(_answer(tune)), led, cache, tax, _lim())
    t = out.set_index("run")
    assert t.loc[f"Gemini, new prompt ({cfg.prompt_version})", "path"] == 1.0
    assert t.loc["TF-IDF (same complaints)", "path"] == pytest.approx(0.5)
    assert t.loc["TF-IDF (same complaints)", "rows_done"] == len(tune)
    printed = capsys.readouterr().out
    assert ".84/.62/.40/.38" in printed and ".820/.514/.371/.324" in printed
    assert "old prompt" not in printed.lower() or "fewer than 30" in printed
    assert "rest of the phase" in printed  # projection from Part C's real cost


def test_part_c_old_prompt_row_only_with_30_overlapping(tmp_path, capsys):
    cfg, tax, tune, led, cache = _part_c_setup(tmp_path)
    model = cfg.classifier_model
    for k, cid in enumerate(tune["complaint_id"][:30]):
        cache.put(f"{cid}|gemini|{model}|v1", {"product": "P1", "issue": "I1",
                                                "sub_issue": "S1" if k % 2 == 0 else "S9"})  # fmt: skip
    out = colab.prompt_check(cfg, tune, FakeClient(_answer(tune)), led, cache, tax, _lim())
    assert "old prompt (v1), same 30" in set(out["run"])
    capsys.readouterr()

    cfg2, tax2, tune2, led2, cache2 = _part_c_setup(tmp_path / "b")
    for cid in tune2["complaint_id"][:29]:
        cache2.put(f"{cid}|gemini|{model}|v1", {"product": "P1", "issue": "I1", "sub_issue": "S1"})
    out2 = colab.prompt_check(cfg2, tune2, FakeClient(_answer(tune2)), led2, cache2, tax2, _lim())
    assert not any("old prompt" in r for r in out2["run"])
    capsys.readouterr()


def test_part_c_skipped_run_says_why(tmp_path, capsys):
    cfg, tax, tune, led, cache = _part_c_setup(tmp_path)

    def rejected(*a):
        raise RequestRejected(404, "model retired, remove it from the list")

    out = colab.prompt_check(cfg, tune, FakeClient(rejected), led, cache, tax, _lim())
    printed = capsys.readouterr().out
    assert "skipped: model retired" in printed
    assert "No answers" not in printed and "rest of the phase" not in printed
    assert out.set_index("run").loc[f"Gemini, new prompt ({cfg.prompt_version})", "status"].startswith(
        "skipped")


# ---- 3. Sheet edits are never wiped ----------------------------------------------------------


class _WS:
    id = 0

    def __init__(self):
        self.values = []
        self.clears = 0

    def update(self, values, range_name=None):
        self.values = [list(r) for r in values]

    def clear(self):
        self.clears += 1
        self.values = []

    def get_all_records(self, numericise_ignore=()):
        if not self.values:
            return []
        head, *body = self.values
        return [dict(zip(head, r)) for r in body]


class _GC:
    def __init__(self):
        self.books = {}

    def create(self, title):
        ss = SimpleNamespace(url=f"https://fake/{len(self.books)}", title=title, sheet1=_WS())
        self.books[ss.url] = ss
        return ss

    def open_by_url(self, url):
        return self.books[url]


def test_publish_descriptions_keeps_sheet_edits(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    tax = _tax()
    d = _descs(tax)
    gc = _GC()
    url = colab.publish_descriptions(cfg, gc, tax, d)
    ws = gc.open_by_url(url).sheet1
    ws.values[1][-1] = "Marco's edit"
    colab.publish_descriptions(cfg, gc, tax, d)  # e.g. rerunning the Part 0 cell
    assert ws.values[1][-1] == "Marco's edit"  # not wiped
    out = capsys.readouterr().out
    assert "1 description(s) in the Sheet differ" in out and "use your own edits" in out
    assert "DISCARD_SHEET_EDITS" in out
    colab.publish_descriptions(cfg, gc, tax, d, overwrite_edits=True)
    assert ws.values[1][-1] == "d"


def test_first_publish_over_plain_taxonomy_sheet_writes(tmp_path):
    from nlp.sheets import publish_taxonomy

    cfg = _cfg(tmp_path)
    tax = _tax()
    gc = _GC()
    colab.publish_once(cfg, "taxonomy", lambda: publish_taxonomy(gc, tax))  # notebook 01
    url = colab.publish_descriptions(cfg, gc, tax, _descs(tax))
    assert gc.open_by_url(url).sheet1.values[0][-1] == "description"


# ---- 4. fully cached reruns are free and never blocked --------------------------------------


def test_estimate_counts_only_uncached_rows(tmp_path):
    cfg = _cfg(tmp_path)
    tax = _tax()
    store.write(cfg, P.STAGE_DESCRIPTIONS, _descs(tax))
    truth = _rows(20)
    cache = JsonlCache(tmp_path / "c.jsonl")
    full = colab.estimate_classification(cfg, truth, tax, cache=cache, classifier_name="x")
    led = Ledger(tmp_path / "l.jsonl", 5.0, cfg.prices)
    run_classification(truth.head(10), cfg, FakeClient(_answer(truth)), led, cache, "x",
                       taxonomy=tax, limiter=_lim(), sleep=lambda s: None)  # fmt: skip
    half = colab.estimate_classification(cfg, truth, tax, cache=cache, classifier_name="x")
    assert 0 < half < full
    run_classification(truth, cfg, FakeClient(_answer(truth)), led, cache, "x",
                       taxonomy=tax, limiter=_lim(), sleep=lambda s: None)  # fmt: skip
    assert colab.estimate_classification(cfg, truth, tax, cache=cache, classifier_name="x") == 0.0


def test_cached_part_c_rerun_is_free_even_at_the_cap(tmp_path, capsys):
    cfg, tax, tune, led, cache = _part_c_setup(tmp_path)
    client = FakeClient(_answer(tune))
    colab.prompt_check(cfg, tune, client, led, cache, tax, _lim())
    calls = len(client.calls)
    full = Ledger(tmp_path / "l.jsonl", led.spent(), cfg.prices)  # no money left at all
    out = colab.prompt_check(cfg, tune, client, full, cache, tax, _lim())
    assert len(client.calls) == calls and len(out) >= 2
    capsys.readouterr()


def test_phase_estimate_counts_only_uncached_rows(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    tax = _tax()
    store.write(cfg, P.STAGE_DESCRIPTIONS, _descs(tax))
    test = _rows(20)
    cache = JsonlCache(tmp_path / "c.jsonl")
    led = Ledger(tmp_path / "l.jsonl", 5.0, cfg.prices)
    before = colab.phase_estimate(cfg, led, tax, test, test.head(5), cache=cache).set_index("stage")
    run_classification(test, cfg, FakeClient(_answer(test)), led, cache, "gemini",
                       taxonomy=tax, limiter=_lim(), sleep=lambda s: None)  # fmt: skip
    after = colab.phase_estimate(cfg, led, tax, test, test.head(5), cache=cache).set_index("stage")
    assert before.loc[P.STAGE_PRED_GEMINI, "bound_usd"] > 0
    assert after.loc[P.STAGE_PRED_GEMINI, "bound_usd"] == 0.0
    assert after.loc[P.STAGE_PRED_GEMINI, "usd"] == 0.0
    capsys.readouterr()


def test_notebook_gates_pass_the_cache():
    code = "\n".join(code_cells(NB / "03_gemini.ipynb"))
    assert code.count("cache=gem_cache") >= 3
    assert "phase_estimate(cfg, ledger, taxonomy, test_df, sc_rows, cache=gem_cache)" in code
    code4 = "\n".join(code_cells(NB / "04_validators.ipynb"))
    assert "cache=gem_cache" in code4


# ---- 5./6. labels in the prompt ------------------------------------------------------------


def test_normalize_label_tolerates_spaces_around_pipes():
    tax = _tax()
    assert normalize_label(tax, "P1 | I1 |S1") == "P1|I1|S1"
    assert normalize_label(tax, " p1 |  i1 | s2 ") == "P1|I1|S2"


def test_listing_is_unambiguous_when_names_contain_colons():
    df = pd.DataFrame({"product": ["Debt: collection", "Debt: collection"],
                       "issue": ["Fees: late", "Fees: late"],
                       "sub_issue": ["Charged: twice", "Other"]})  # fmt: skip
    tax = build_taxonomy(df)
    txt = taxonomy_prompt_text(tax, {x: f"About {x}." for x in tax["label"]})
    labels = re.findall(r"^- label: (.+)$", txt, re.MULTILINE)
    assert labels == sorted(tax["label"])
    assert re.findall(r"^  description: (.+)$", txt, re.MULTILINE)[0].startswith("About ")


# ---- 7. Sheet titles name the data version -------------------------------------------------


def test_sheet_titles_include_data_version(tmp_path):
    cfg = _cfg(tmp_path)
    tax = _tax()
    gc = _GC()
    url = colab.publish_descriptions(cfg, gc, tax, _descs(tax))
    assert cfg.data_version in gc.open_by_url(url).title
    nb1 = "\n".join(code_cells(NB / "01_data_and_splits.ipynb"))
    nb4 = "\n".join(code_cells(NB / "04_validators.ipynb"))
    assert 'f"CFPB taxonomy ({cfg.data_version})"' in nb1
    assert 'f"CFPB hand check ({cfg.data_version})"' in nb4


# ---- 8. a failed key check stops notebook 00 -----------------------------------------------


def test_key_check_failure_stops_by_default():
    from google.genai import errors

    bad = errors.ClientError(400, {"error": {"code": 400, "message": "API key not valid.",
                                             "status": "INVALID_ARGUMENT"}})  # fmt: skip

    def boom(key):
        def lst(**kw):
            raise bad

        return SimpleNamespace(models=SimpleNamespace(list=lst))

    with pytest.raises(colab.StageIncomplete, match="Key problems") as e:
        colab.check_gemini_key("AIzaBAD", make_client=boom)
    assert "AIzaBAD" not in str(e.value)
    assert colab.check_gemini_key("AIzaBAD", make_client=boom, stop=False) is False


# ---- 9. max_output_tokens never exceeds the model limit ------------------------------------


def test_output_tokens_clamped_to_model_limit():
    assert MAX_OUTPUT_TOKENS == 65536
    assert estimate_request_tokens("", "", 25, "high")[1] == MAX_OUTPUT_TOKENS
    assert estimate_request_tokens("", "", 10, "low")[1] < MAX_OUTPUT_TOKENS
    assert estimate_describe_tokens("", "", 400)[1] == MAX_OUTPUT_TOKENS


# ---- 11. edits after Part D warn --------------------------------------------------------------


def test_loading_edits_after_test_run_warns(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    tax = _tax()
    d = _descs(tax)
    store.write(cfg, P.STAGE_DESCRIPTIONS, d)
    gc = _GC()
    url = colab.publish_descriptions(cfg, gc, tax, d)
    gc.open_by_url(url).sheet1.values[1][-1] = "Edited."
    store.write(cfg, P.STAGE_PRED_GEMINI, pd.DataFrame({"complaint_id": ["a"]}))
    colab.load_description_edits(cfg, gc, tax)
    assert "self-consistency" in capsys.readouterr().out
    md = "\n".join(c["source"] for c in json.loads((NB / "03_gemini.ipynb").read_text(
        encoding="utf-8"))["cells"] if c["cell_type"] == "markdown")
    assert "before Part C" in md and "self-consistency" in md


def test_projection_warns_when_rest_of_phase_will_not_fit(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    tax = _tax()
    store.write(cfg, P.STAGE_DESCRIPTIONS, _descs(tax))
    tag = colab.current_prompt_tag(cfg, tax)
    led = Ledger(tmp_path / "l.jsonl", 5.0, cfg.prices)
    led.record(cfg.classifier_model, "a", 4_000, 400, tag=tag, rows=10)  # cheap: fits
    colab.project_rest_of_phase(cfg, led, tax)
    assert "will not fit" not in capsys.readouterr().out
    led.record(cfg.classifier_model, "b", 900_000, 90_000, tag=tag, rows=10)  # expensive
    colab.project_rest_of_phase(cfg, led, tax)
    out = capsys.readouterr().out
    assert "will not fit" in out and "batch size 25" in out
