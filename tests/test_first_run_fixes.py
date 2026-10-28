"""One test (or more) per failure Marco hit on the first live run (design v2, "First-run fixes").

Each test reproduces the failure as it happened, then checks the fixed behaviour.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from nlp import colab
from nlp.budget import Ledger, RateLimiter
from nlp.config import Config
from nlp.llm.classify import RESPONSE_SCHEMA, run_classification
from nlp.llm.client import FakeClient, GenAIClient, RequestRejected
from nlp.store import JsonlCache
from nlp.taxonomy import build_taxonomy
from nlp.tools.check_notebooks import code_cells

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "phase1.yaml"
NB = ROOT / "notebooks"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")  # google-genai on 3.14


def _code(name: str) -> str:
    return "\n".join(code_cells(NB / name))


def _genai_with(exc):
    def boom(**kw):
        raise exc

    c = GenAIClient.__new__(GenAIClient)
    c._client = SimpleNamespace(models=SimpleNamespace(generate_content=boom))
    c._quota_logged = False
    return c


def _client_error(code, status, message, reason=None):
    from google.genai import errors

    err = {"code": code, "status": status, "message": message}
    if reason:
        err["details"] = [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason}]
    return errors.ClientError(code, {"error": err})


RETIRED = _client_error(
    404, "NOT_FOUND",
    "This model models/gemini-2.5-flash is no longer available to new users. Please update your "
    "code to use a newer model for the latest features and improvements.",
)  # fmt: skip
NO_MINIMAL = _client_error(
    400, "INVALID_ARGUMENT", "Thinking level MINIMAL is not supported for this model. Please "
    "retry with other thinking level"
)  # fmt: skip
BAD_KEY = _client_error(
    400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.", "API_KEY_INVALID"
)


# ---- 1. Retired model ----------------------------------------------------------------------


def test_fix1_retired_model_not_default_nor_candidate():
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert raw["classifier_model"] == "gemini-3.8-flash"
    assert "gemini-2.5-flash" in raw["prices"]  # price kept for old ledger lines
    m = re.search(r"CANDIDATES = (\[.*?\])", _code("03_gemini.ipynb"))
    assert json.loads(m.group(1)) == [
        "gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"
    ]  # fmt: skip
    for nb in NB.glob("*.ipynb"):
        assert '"gemini-2.5-flash"' not in "\n".join(code_cells(nb)), nb.name


def test_fix1_pick_hint_names_a_live_model(tmp_path):
    cfg = colab.load_config(CONFIG, tmp_path)
    with pytest.raises(ValueError) as e:
        colab.apply_choices(cfg, require_pick=True, classifier_model=None)
    assert "gemini-2.5-flash" not in str(e.value)


# ---- 2. Rejected requests ------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc, needle",
    [
        (RETIRED, "remove it from the list"),
        (NO_MINIMAL, "does not support that thinking level"),
        (BAD_KEY, "Gemini rejected the API key. See README 'Key problems'"),
        (_client_error(403, "PERMISSION_DENIED", "Method doesn't allow unregistered callers."),
         "Gemini rejected the API key"),
        (_client_error(400, "INVALID_ARGUMENT", "Request contains an invalid argument."),
         "invalid argument"),
    ],
)
def test_fix2_genai_client_maps_4xx_to_request_rejected(exc, needle):
    with pytest.raises(RequestRejected) as e:
        _genai_with(exc).generate("gemini-2.5-flash", "s", "u", RESPONSE_SCHEMA, "minimal", 10)
    assert needle in str(e.value)
    assert e.value.status == exc.code


def test_fix2_429_still_rate_limited():
    from nlp.llm.client import RateLimited

    exc = _client_error(429, "RESOURCE_EXHAUSTED", "quota")
    with pytest.raises(RateLimited):
        _genai_with(exc).generate("gemini-3.8-flash", "s", "u", RESPONSE_SCHEMA, "low", 10)


@pytest.mark.parametrize("code", [402, 429])
def test_prepaid_credits_depleted_is_plain_english_not_retried(code):
    exc = _client_error(code, "RESOURCE_EXHAUSTED", "Your prepayment credits are depleted. "
                        "Please go to AI Studio at https://ai.studio/projects to manage your "
                        "project and billing.")  # fmt: skip
    with pytest.raises(RequestRejected) as e:
        _genai_with(exc).generate("gemini-3.8-flash", "s", "u", RESPONSE_SCHEMA, "low", 10)
    assert "prepaid credits are used up" in str(e.value)
    assert "rerun this cell" in str(e.value)


def _tax():
    return build_taxonomy(pd.DataFrame({"product": ["P"], "issue": ["I"], "sub_issue": ["S"]}))


def _descs(tax):
    return pd.DataFrame({"label": tax["label"], "description": "d", "model_id": "m",
                         "describe_version": "d1"})  # fmt: skip


def test_fix2_rejected_request_is_never_billed_or_retried(tmp_path, tmp_cfg):
    def rejected(*a):
        raise RequestRejected(404, "retired")

    client = FakeClient(rejected)
    ledger = Ledger(tmp_path / "l.jsonl", 5.0, tmp_cfg.prices)
    rows = pd.DataFrame({"complaint_id": ["a"], "narrative": ["text"]})
    tax = _tax()
    with pytest.raises(RequestRejected):
        run_classification(rows, tmp_cfg, client, ledger, JsonlCache(tmp_path / "c.jsonl"),
                           taxonomy=tax, descriptions=_descs(tax),
                           limiter=RateLimiter(10_000, sleep=lambda s: None),
                           sleep=lambda s: None)  # fmt: skip
    assert len(client.calls) == 1 and ledger.spent() == 0.0
    assert not ledger.path.exists()


def test_fix2_compare_runs_skips_a_rejected_run_and_continues(tmp_path, capsys):
    cfg = colab.load_config(CONFIG, tmp_path)
    ledger = Ledger(tmp_path / "l.jsonl", 5.0, cfg.prices)
    truth = pd.DataFrame({"complaint_id": ["1", "2"], "product": ["P", "P"], "issue": ["I", "I"],
                          "sub_issue": ["S", "S"]})  # fmt: skip

    def retired():
        raise RequestRejected(404, "Gemini model 'gemini-2.5-flash' is retired (Google says it "
                                   "is no longer available): remove it from the list.")

    out = colab.compare_runs({"old": retired, "new": lambda: truth}, truth, ledger)
    printed = capsys.readouterr().out
    assert "old skipped: Gemini model 'gemini-2.5-flash' is retired" in printed
    assert out["run"].tolist() == ["old", "new"]
    row = out.set_index("run")
    assert row.loc["old", "status"].startswith("skipped") and row.loc["new", "status"] == "ok"
    assert pd.isna(row.loc["old", "path"]) and row.loc["new", "path"] == 1.0


def test_fix2_stage_paid_turns_rejection_into_plain_english(tmp_path):
    cfg = colab.load_config(CONFIG, tmp_path)

    def bad_key():
        raise RequestRejected(400, "Gemini rejected the API key. See README 'Key problems'.")

    with pytest.raises(colab.StageIncomplete, match="Key problems") as e:
        colab.stage_paid(cfg, "s", bad_key, ["a"])
    assert "Nothing was charged" in str(e.value)


# ---- 3. Thinking levels --------------------------------------------------------------------


def test_fix3_thinking_check_uses_low_and_medium_only():
    code = _code("03_gemini.ipynb")
    m = re.search(r"LEVELS = (\[.*?\])", code)
    assert json.loads(m.group(1)) == ["low", "medium"]
    assert '"minimal"' not in code


# ---- 4. Keys with stray whitespace ---------------------------------------------------------


class _UD:
    def __init__(self, d):
        self.d = d

    def get(self, name):
        if name not in self.d:
            raise KeyError(name)
        return self.d[name]


def test_fix4_read_secrets_strips_whitespace():
    got = colab.read_secrets(_UD({"GEMINI_API_KEY": " AQ.abc123\n", "TYPESAFE_API_KEY": "\tts-1 "}))
    assert got == ("AQ.abc123", "ts-1")
    assert colab.read_secrets(_UD({"GEMINI_API_KEY": "k", "TYPESAFE_API_KEY": "  "}))[1] is None
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        colab.read_secrets(_UD({"GEMINI_API_KEY": "  \n"}))


# ---- 5. Key check in notebook 00 -----------------------------------------------------------


class _Models:
    def __init__(self, exc=None):
        self.exc = exc

    def list(self, **kw):
        if self.exc:
            raise self.exc
        return iter([SimpleNamespace(name="models/gemini-3.8-flash")])


def test_fix5_key_check_passes_for_aq_key(capsys):
    ok = colab.check_gemini_key("AQ.Ab8RN6Kx", make_client=lambda key: SimpleNamespace(models=_Models()))
    out = capsys.readouterr().out
    assert ok is True and "works" in out
    assert "AQ." in out and "normal" in out


def test_fix5_key_check_failure_prints_readme_fix(capsys):
    ok = colab.check_gemini_key(
        "AIzaWRONG", make_client=lambda key: SimpleNamespace(models=_Models(BAD_KEY)), stop=False
    )
    out = capsys.readouterr().out
    assert ok is False
    assert "Key problems" in out and "README" in out
    assert "AIzaWRONG" not in out  # never print the key


def test_fix5_notebook_00_checks_the_key():
    assert "colab.check_gemini_key(api_key)" in _code("00_setup.ipynb")


# ---- 6. Never pip-install Google libraries -------------------------------------------------


def test_fix6_google_genai_floor_satisfied_by_colab():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"google-genai>=2.12"' in text


def test_fix6_notebooks_never_upgrade_or_install_google_libraries():
    for nb in NB.glob("*.ipynb"):
        for src in code_cells(nb):
            for line in src.splitlines():
                if line.lstrip().startswith(("%pip", "!pip")):
                    assert "#" not in line, (nb.name, "pip would read a comment as arguments")
                if "pip" in line:
                    low = line.lower()
                    assert "-u " not in low and "--upgrade" not in low, (nb.name, line)
                    assert "google-genai" not in low and "google-auth" not in low, (nb.name, line)
                    assert "[colab]" not in low and ",colab]" not in low, (nb.name, line)


# ---- 7. TF-IDF convergence -----------------------------------------------------------------


def test_fix7_tfidf_logistic_regression_max_iter_1000(monkeypatch):
    import nlp.classifiers.tfidf as T

    seen = {}
    real = T.LogisticRegression

    def spy(**kw):
        seen.update(kw)
        return real(**kw)

    monkeypatch.setattr(T, "LogisticRegression", spy)
    m = T.TfidfLR(max_features=50, ngram_range=(1, 1))
    m.fit(["a b c", "d e f", "a b x", "d e y"], ["p1", "p2", "p1", "p2"])
    assert seen["max_iter"] == 1000


# ---- 8. CI on Colab's Python ---------------------------------------------------------------


def test_fix8_ci_runs_python_311_and_313():
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    assert ci["jobs"]["test"]["strategy"]["matrix"]["python-version"] == ["3.11", "3.13"]
    gate = ci["jobs"]["lint-and-test"]  # the check name the protect-main ruleset requires
    assert gate["needs"] == "test" and gate["if"] == "always()"
    assert "needs.test.result" in gate["steps"][0]["run"]


# ---- 9. Diagnostics ------------------------------------------------------------------------


def test_fix9_diagnose_is_free_and_reports_everything(tmp_path, capsys):
    cfg = colab.load_config(CONFIG, tmp_path)
    ledger = Ledger(tmp_path / "ledger.jsonl", cfg.spend_cap_usd, cfg.prices)
    ledger.record("gemini-3.8-flash", "a", 1000, 100)
    ledger.record("gemini-3.8-flash", "b", 1000, 100)
    cache = JsonlCache(tmp_path / "gemini_cache.jsonl")
    cache.put("c1|bakeoff_gemini-3.8-flash|gemini-3.8-flash|v1", {})
    cache.put("c2|bakeoff_gemini-3.8-flash|gemini-3.8-flash|v1", {})
    cache.put("Credit card|describe|gemini-3.8-flash|d1", {})
    calls = []

    def make_client(key):
        calls.append(key)
        return SimpleNamespace(models=_Models())

    colab.diagnose(cfg, " AQ.secretkey\n", ledger, ["gemini-3.8-flash", "gemini-3.1-flash-lite"],
                   make_client=make_client)  # fmt: skip
    out = capsys.readouterr().out
    for needle in ("gemini-3.8-flash", "gemini-3.1-flash-lite", "google-genai",
                   "length 12", "starts with 'AQ.'", "whitespace: yes", "works",
                   "$", "2 paid request(s)", "bakeoff_gemini-3.8-flash", "describe"):
        assert needle in out, needle
    assert "secretkey" not in out
    assert calls == ["AQ.secretkey"]  # stripped key; only the free model listing is called


def test_fix9_notebook_03_has_troubleshoot_cell():
    assert "colab.diagnose(cfg, api_key, ledger" in _code("03_gemini.ipynb")


def test_config_class_has_v2_keys():
    cfg = Config.load(CONFIG)
    assert cfg.describe_model == "gemini-3.8-flash"
    assert cfg.data_version == "v2" and cfg.prompt_version == "v2"


def test_readme_troubleshooting_covers_first_run_problems():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for needle in ("Key problems", "AQ.", "API key not valid", "Never pip-install Google libraries",
                   "credential propagation was unsuccessful", "KeyboardInterrupt",
                   "Disconnect and delete runtime", "Restart session", "Save in GitHub",
                   "Run that edited cell", "Part 0", "prepaid credits are used up"):
        assert needle in text, needle
    assert "$4.00" not in text and "Four Gemini models" not in text


def test_cap_is_off_everywhere_marco_reads_it():
    # Marco 2026-10-28: Google prepaid credits are the limit; the in-code cap is effectively off.
    assert Config.load(CONFIG).spend_cap_usd == 1000.00
    for f in [ROOT / "README.md", ROOT / "notebooks" / "README.md", ROOT / "CLAUDE.md",
              *NB.glob("*.ipynb")]:
        text = f.read_text(encoding="utf-8")
        assert "$4.00" not in text and "$5.00 cap" not in text and "capped at $5" not in text, f.name
