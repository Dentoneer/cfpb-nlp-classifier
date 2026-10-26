from datetime import date

import pandas as pd
import pytest

from nlp.schema import TAXONOMY_COLS
from nlp.taxonomy import (
    build_taxonomy,
    choose_window,
    normalize_path,
    path_id,
    taxonomy_prompt_text,
)


def _complaints():
    rows = [
        ("Credit reporting", "Incorrect information", "Wrong name"),
        ("Credit reporting", "Incorrect information", "Wrong name"),
        ("Credit reporting", "Incorrect information", "(none)"),
        ("Debt collection", "Attempts to collect", "(none)"),
        ("Debt collection", "Attempts to collect", "Not owed"),
    ]
    return pd.DataFrame(rows, columns=["product", "issue", "sub_issue"])


def test_build_taxonomy_unique_sorted():
    t = build_taxonomy(_complaints())
    assert list(t.columns) == TAXONOMY_COLS
    assert len(t) == 4
    assert t["path_id"].is_unique
    keys = list(zip(t["product"], t["issue"], t["sub_issue"]))
    assert keys == sorted(keys)
    assert t["path_id"].iloc[0] == path_id(*keys[0])
    assert " || " in t["path_id"].iloc[0]


def test_normalize_parse_variants():
    t = build_taxonomy(
        pd.DataFrame(
            [
                ("Credit reporting", "Can't verify", "(none)"),
                ("Debt collection", "Attempts", "Not owed"),
                ("Debt collection", "attempts", "Not owed"),
            ],
            columns=["product", "issue", "sub_issue"],
        )
    )
    got = normalize_path(t, " credit REPORTING… ".replace("…", ""), "Can’t  verify ", "(NONE)")
    assert got == ("Credit reporting", "Can't verify", "(none)")
    assert normalize_path(t, "Debt collection", "ATTEMPTS", "not owed") is None  # ambiguous
    assert normalize_path(t, "Nope", "x", "y") is None


def test_prompt_text_deterministic():
    t = build_taxonomy(_complaints())
    txt = taxonomy_prompt_text(t.sample(frac=1, random_state=1))
    assert txt == taxonomy_prompt_text(t)
    assert txt.count("Credit reporting") == 1
    assert txt.count("Incorrect information") == 1
    assert "Wrong name" in txt and "(none)" in txt


def _counts(n_months=20, change_back=8):
    months = [f"{2025 + (i // 12)}-{i % 12 + 1:02d}" for i in range(n_months)]
    rows = []
    for i, m in enumerate(months):
        old = i < n_months - change_back
        rows.append((m, "Credit", "Old issue" if old else "New issue", 900))
        rows.append((m, "Debt", "Same", 100))
    return pd.DataFrame(rows, columns=["month", "product", "issue", "n"])


def test_choose_window_stops_at_taxonomy_change():
    start, end = choose_window(_counts(), date(2026, 8, 31), months=12)
    assert start == date(2026, 1, 1)  # first month with the new issue set (index 12)
    assert end == date(2026, 9, 1)  # exclusive, after last month 2026-08


def test_choose_window_full_when_stable():
    start, end = choose_window(_counts(change_back=0), date(2026, 8, 31), months=12)
    assert (start, end) == (date(2025, 9, 1), date(2026, 9, 1))


def test_choose_window_drops_partial_final_month():
    c = _counts(change_back=0)  # full months 2025-01..2026-08
    partial = pd.DataFrame(
        [("2026-09", "Credit", "Old issue", 5), ("2026-09", "Rare", "Brand new", 5)],
        columns=["month", "product", "issue", "n"],
    )
    c = pd.concat([c, partial], ignore_index=True)
    start, end = choose_window(c, date(2026, 9, 6), months=12)
    assert (start, end) == (date(2025, 9, 1), date(2026, 9, 1))


def test_choose_window_no_full_month_raises():
    c = pd.DataFrame([("2026-07", "A", "b", 3)], columns=["month", "product", "issue", "n"])
    with pytest.raises(ValueError):
        choose_window(c, date(2026, 7, 6))
