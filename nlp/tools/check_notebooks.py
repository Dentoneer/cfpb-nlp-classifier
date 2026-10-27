"""CI check: every `nlp.` name a notebook's code cells reference actually exists.

Usage: python -m nlp.tools.check_notebooks notebooks/

Each code cell is parsed with `ast` (lines starting with `!` or `%` are Colab shell/magics and
are skipped). We collect `import nlp.x as y`, `from nlp import x`, `from nlp.x import y` and
attribute chains rooted at those names, then resolve each with importlib/getattr.
"""

from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path


def code_cells(path: Path) -> list[str]:
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = cell.get("source", "")
            out.append("".join(src) if isinstance(src, list) else src)
    return out


def _strip_magics(source: str) -> str:
    return "\n".join(
        "" if line.lstrip().startswith(("!", "%")) else line for line in source.splitlines()
    )


def refs_in_source(source: str) -> set[tuple[str, str]]:
    """(module-ish prefix, last name) pairs for every `nlp.` reference in `source`."""
    tree = ast.parse(_strip_magics(source))
    alias: dict[str, str] = {}
    dotted: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "nlp" or a.name.startswith("nlp."):
                    dotted.add(a.name)
                    if a.asname:
                        alias[a.asname] = a.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mod = node.module or ""
            if mod == "nlp" or mod.startswith("nlp."):
                for a in node.names:
                    full = f"{mod}.{a.name}"
                    dotted.add(full)
                    alias[a.asname or a.name] = full
    inner = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or id(node) in inner:
            continue
        parts = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name) and (cur.id in alias or cur.id == "nlp"):
            root = alias.get(cur.id, "nlp")
            dotted.add(".".join([root, *reversed(parts)]))
    return {tuple(d.rsplit(".", 1)) for d in dotted if "." in d}  # type: ignore[misc]


def resolve(dotted: str) -> str | None:
    """None if `dotted` resolves; otherwise a short reason."""
    parts = dotted.split(".")
    for k in range(len(parts), 0, -1):
        prefix = ".".join(parts[:k])
        try:
            obj = importlib.import_module(prefix)
        except ModuleNotFoundError as e:
            if e.name and (prefix == e.name or prefix.startswith(e.name + ".")):
                continue
            return f"imports a missing dependency ({e.name})"
        for attr in parts[k:]:
            if not hasattr(obj, attr):
                return f"{prefix} has no attribute {attr!r}"
            obj = getattr(obj, attr)
            prefix = f"{prefix}.{attr}"
        return None
    return "module not found"


def check_notebooks(root: Path) -> list[str]:
    """Problems found in every *.ipynb under `root` (empty list = all good)."""
    root = Path(root)
    files = sorted(root.rglob("*.ipynb")) if root.is_dir() else [root]
    problems: list[str] = []
    for nb in files:
        for n, src in enumerate(code_cells(nb), start=1):
            try:
                refs = refs_in_source(src)
            except SyntaxError as e:
                problems.append(f"{nb.name} code cell {n}: syntax error: {e.msg} (line {e.lineno})")
                continue
            for mod, attr in sorted(refs):
                why = resolve(f"{mod}.{attr}")
                if why:
                    problems.append(f"{nb.name} code cell {n}: {mod}.{attr}: {why}")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m nlp.tools.check_notebooks <notebooks dir>")
        return 2
    problems = check_notebooks(Path(args[0]))
    if problems:
        print("Notebook references that do not resolve:")
        for p in problems:
            print(f"  {p}")
        return 1
    n = len(list(Path(args[0]).rglob("*.ipynb")))
    print(f"OK: {n} notebook(s), every nlp.* reference resolves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
