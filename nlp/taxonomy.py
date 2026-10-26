"""Product / Issue / Sub-issue taxonomy: build, normalize, render, and window choice."""

from __future__ import annotations

import re
from datetime import date

import pandas as pd

from nlp.schema import TAXONOMY_COLS, path_id

__all__ = [
    "build_taxonomy",
    "choose_window",
    "normalize_path",
    "path_id",
    "taxonomy_prompt_text",
]

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def build_taxonomy(complaints: pd.DataFrame) -> pd.DataFrame:
    """Unique (product, issue, sub_issue) paths, sorted, with their path_id."""
    t = (
        complaints[["product", "issue", "sub_issue"]]
        .drop_duplicates()
        .sort_values(["product", "issue", "sub_issue"])
        .reset_index(drop=True)
    )
    t["path_id"] = [path_id(p, i, s) for p, i, s in zip(t["product"], t["issue"], t["sub_issue"])]
    return t[TAXONOMY_COLS]


def _norm(s: object) -> str:
    return re.sub(r"\s+", " ", str(s).translate(_QUOTES)).casefold().strip()


def normalize_path(
    taxonomy: pd.DataFrame, product: str, issue: str, sub_issue: str
) -> tuple[str, str, str] | None:
    """Map a near-miss path to the exact taxonomy strings, or None if not exactly one match."""
    want = (_norm(product), _norm(issue), _norm(sub_issue))
    hits = [
        (p, i, s)
        for p, i, s in zip(taxonomy["product"], taxonomy["issue"], taxonomy["sub_issue"])
        if (_norm(p), _norm(i), _norm(s)) == want
    ]
    hits = list(dict.fromkeys(hits))
    return hits[0] if len(hits) == 1 else None


def taxonomy_prompt_text(taxonomy: pd.DataFrame) -> str:
    """Deterministic nested listing: Product, then indented Issues, then Sub-issues."""
    tree: dict[str, dict[str, set[str]]] = {}
    for p, i, s in zip(taxonomy["product"], taxonomy["issue"], taxonomy["sub_issue"]):
        tree.setdefault(p, {}).setdefault(i, set()).add(s)
    lines: list[str] = []
    for p in sorted(tree):
        lines.append(f"Product: {p}")
        for i in sorted(tree[p]):
            lines.append(f"  Issue: {i}")
            for s in sorted(tree[p][i]):
                lines.append(f"    Sub-issue: {s}")
    return "\n".join(lines)


def _differs(a: dict[tuple, float], b: dict[tuple, float], tol: float) -> bool:
    """True if more than `tol` of either month's volume sits in pairs absent from the other."""
    for x, y in ((a, b), (b, a)):
        total = sum(x.values())
        if total and sum(v for k, v in x.items() if k not in y) / total > tol:
            return True
    return False


def choose_window(counts_by_month: pd.DataFrame, months: int = 12) -> tuple[date, date]:
    """Latest window of up to `months` months with a stable (product, issue) set.

    Walks back from the latest month and stops at the first month whose pair set
    differs from the latest month's on more than 10% of volume. Returns
    (start, end) with `end` exclusive (first day after the last included month).
    """
    df = counts_by_month
    by_month = {
        m: {(p, i): float(n) for p, i, n in zip(g["product"], g["issue"], g["n"])}
        for m, g in df.groupby("month")
    }
    ordered = sorted(by_month)
    ref = by_month[ordered[-1]]
    stable = [ordered[-1]]
    for m in reversed(ordered[:-1]):
        if len(stable) >= months or _differs(by_month[m], ref, 0.10):
            break
        stable.append(m)
    first, last = stable[-1], stable[0]
    y, mo = map(int, first.split("-"))
    ly, lm = map(int, last.split("-"))
    end = date(ly + (lm == 12), lm % 12 + 1, 1)
    return date(y, mo, 1), end
