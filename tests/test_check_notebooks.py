import json
import subprocess
import sys
from pathlib import Path

from nlp.tools.check_notebooks import check_notebooks, code_cells, refs_in_source

ROOT = Path(__file__).resolve().parent.parent


def _nb(path: Path, *sources: str) -> Path:
    cells = [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
              "source": s} for s in sources]  # fmt: skip
    path.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    return path


def test_refs_attribute_and_from_import():
    src = "!pip x\n%load_ext y\nfrom nlp.store import stage, nope_x\nimport nlp.pipeline as P\nP.STAGE_JEB\nnlp.budget.Ledger"
    refs = refs_in_source(src)
    assert ("nlp.store", "stage") in refs and ("nlp.store", "nope_x") in refs
    assert ("nlp.pipeline", "STAGE_JEB") in refs and ("nlp.budget", "Ledger") in refs


def test_check_flags_missing(tmp_path):
    _nb(tmp_path / "a.ipynb", "from nlp.store import stage\nfrom nlp.store import nope_x\n")
    problems = check_notebooks(tmp_path)
    assert len(problems) == 1 and "nope_x" in problems[0]


def test_check_flags_syntax_error_and_missing_module(tmp_path):
    _nb(tmp_path / "a.ipynb", "def (:\n", "from nlp.nosuchmod import y\n")
    problems = check_notebooks(tmp_path)
    assert any("syntax" in p.lower() for p in problems)
    assert any("nosuchmod" in p for p in problems)
    assert len(code_cells(tmp_path / "a.ipynb")) == 2


def test_real_notebooks_resolve():
    r = subprocess.run(
        [sys.executable, "-m", "nlp.tools.check_notebooks", "notebooks/"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr


def test_cli_exits_nonzero_on_missing(tmp_path):
    _nb(tmp_path / "a.ipynb", "from nlp.store import nope_x\n")
    r = subprocess.run(
        [sys.executable, "-m", "nlp.tools.check_notebooks", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert r.returncode == 1 and "nope_x" in r.stdout


def test_aliases_are_shared_across_cells(tmp_path):
    _nb(
        tmp_path / "a.ipynb",
        "from nlp import colab, pipeline as P\n",
        "colab.setup\nP.STAGE_JEB\n",
        "colab.no_such_helper\nP.NO_SUCH_STAGE\n",
    )
    problems = check_notebooks(tmp_path)
    assert len(problems) == 2
    assert all("cell 3" in p for p in problems)
    assert any("no_such_helper" in p for p in problems)
    r = subprocess.run(
        [sys.executable, "-m", "nlp.tools.check_notebooks", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert r.returncode == 1 and "NO_SUCH_STAGE" in r.stdout
