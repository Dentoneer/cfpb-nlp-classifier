"""Product / Issue / Sub-issue taxonomy: build, normalize, render, and window choice."""

from __future__ import annotations

import re
from datetime import date

import pandas as pd

from nlp.schema import LABEL_SEP, TAXONOMY_COLS, label_of, path_id, split_label

__all__ = [
    "build_taxonomy",
    "choose_window",
    "label_of",
    "normalize_label",
    "normalize_path",
    "path_id",
    "split_label",
    "taxonomy_prompt_text",
]

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def build_taxonomy(complaints: pd.DataFrame) -> pd.DataFrame:
    """Unique (product, issue, sub_issue) paths, sorted, with their path_id and label.

    Raises ValueError if a name contains `|` (the label separator), since the label could then
    not be split back into its three parts.
    """
    for c in ("product", "issue", "sub_issue"):
        bad = sorted({str(v) for v in complaints[c] if LABEL_SEP in str(v)})
        if bad:
            raise ValueError(
                f"{c} names may not contain '{LABEL_SEP}' (it separates the label parts): {bad[:5]}"
            )
    t = (
        complaints[["product", "issue", "sub_issue"]]
        .drop_duplicates()
        .sort_values(["product", "issue", "sub_issue"])
        .reset_index(drop=True)
    )
    parts = list(zip(t["product"], t["issue"], t["sub_issue"]))
    t["path_id"] = [path_id(*x) for x in parts]
    t["label"] = [label_of(*x) for x in parts]
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


def normalize_label(taxonomy: pd.DataFrame, label: object) -> str | None:
    """The exact taxonomy label for `label`: exact match, else the one label equal after
    normalization (quotes, whitespace, case - as in `normalize_path`), else None."""
    labels = list(taxonomy["label"])
    text = "" if label is None else str(label)
    if text in set(labels):
        return text
    want = _norm(text)
    if not want:
        return None
    hits = list(dict.fromkeys(x for x in labels if _norm(x) == want))
    return hits[0] if len(hits) == 1 else None


def _description_map(descriptions) -> dict[str, str]:
    if isinstance(descriptions, pd.DataFrame):
        return dict(zip(descriptions["label"].astype(str), descriptions["description"].astype(str)))
    return {str(k): str(v) for k, v in dict(descriptions).items()}


def taxonomy_prompt_text(taxonomy: pd.DataFrame, descriptions) -> str:
    """One line per taxonomy row, sorted by label: `- <label>: <description>`.

    `descriptions` is the descriptions table (label, description) or a label -> text mapping;
    every taxonomy label needs a non-empty description (ValueError otherwise).
    """
    desc = _description_map(descriptions)
    labels = sorted(set(taxonomy["label"]))
    missing = [x for x in labels if not desc.get(x, "").strip()]
    if missing:
        raise ValueError(f"{len(missing)} taxonomy label(s) have no description, e.g. {missing[:3]}")
    return "\n".join(f"- {x}: {' '.join(desc[x].split())}" for x in labels)


def _differs(month: dict[tuple, float], ref: dict[tuple, float], tol: float) -> bool:
    """True if more than `tol` of this month's volume is in pairs absent from the reference."""
    total = sum(month.values())
    return bool(total) and sum(v for k, v in month.items() if k not in ref) / total > tol


def choose_window(
    counts_by_month: pd.DataFrame, data_end: date, months: int = 12
) -> tuple[date, date]:
    """Latest window of up to `months` full months with a stable (product, issue) set.

    If `data_end` is not the last calendar day of the latest month present, that month is
    partial and is dropped. The reference is the latest FULL month. Walking back from it,
    each earlier month is compared to the reference (not to its neighbour); the walk stops
    at the first month whose (product, issue) pairs outside the reference's set carry more
    than 10% of that month's volume. Returns (start, end) with `end` exclusive (first day
    after the reference month). The window may be shorter than `months` if the stable
    stretch is short. Raises ValueError if no full month remains.
    """
    by_month = {
        m: {(p, i): float(n) for p, i, n in zip(g["product"], g["issue"], g["n"])}
        for m, g in counts_by_month.groupby("month")
    }
    ordered = sorted(by_month)
    if ordered:
        y, mo = map(int, ordered[-1].split("-"))
        month_end = (pd.Timestamp(y, mo, 1) + pd.offsets.MonthEnd(0)).date()
        if data_end < month_end:
            ordered.pop()
    if not ordered:
        raise ValueError("no full months available to choose a window from")
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
