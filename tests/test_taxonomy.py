from datetime import date

import pandas as pd
import pytest

from nlp.schema import TAXONOMY_COLS
from nlp.taxonomy import (
    build_taxonomy,
    choose_window,
    label_of,
    normalize_label,
    normalize_path,
    path_id,
    split_label,
    taxonomy_prompt_text,
)


def _complaints():
    rows = [
        ("Credit reporting", "Incorrect information", "Wrong name"),
        ("Credit reporting", "Incorrect information", "Wrong name"),
        ("Credit reporting", "Incorrect information", "Wrong address"),
        ("Debt collection", "Attempts to collect", "Debt is not yours"),
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
    assert t["label"].iloc[0] == "|".join(keys[0])
    assert t["label"].is_unique


def test_label_roundtrip():
    assert label_of("P", "I", "S") == "P|I|S"
    assert split_label("P|I|S") == ("P", "I", "S")


def test_build_taxonomy_rejects_pipe_in_names():
    df = pd.DataFrame([("Credit|card", "Fees", "Late fee")], columns=["product", "issue", "sub_issue"])
    with pytest.raises(ValueError, match=r"\|"):
        build_taxonomy(df)


def test_normalize_parse_variants():
    t = build_taxonomy(
        pd.DataFrame(
            [
                ("Credit reporting", "Can't verify", "Old debt"),
                ("Debt collection", "Attempts", "Not owed"),
                ("Debt collection", "attempts", "Not owed"),
            ],
            columns=["product", "issue", "sub_issue"],
        )
    )
    got = normalize_path(t, " credit REPORTING… ".replace("…", ""), "Can’t  verify ", "OLD debt")
    assert got == ("Credit reporting", "Can't verify", "Old debt")
    assert normalize_path(t, "Debt collection", "ATTEMPTS", "not owed") is None  # ambiguous
    assert normalize_path(t, "Nope", "x", "y") is None


def _descriptions(t):
    return pd.DataFrame({"label": t["label"], "description": [f"About {x}." for x in t["sub_issue"]]})


def test_prompt_text_one_line_per_label_sorted():
    t = build_taxonomy(_complaints())
    d = _descriptions(t)
    txt = taxonomy_prompt_text(t.sample(frac=1, random_state=1), d.sample(frac=1, random_state=2))
    assert txt == taxonomy_prompt_text(t, d)
    lines = txt.splitlines()
    assert len(lines) == len(t)
    assert lines == sorted(lines)
    assert "- Credit reporting|Incorrect information|Wrong name: About Wrong name." in lines
    assert "(none)" not in txt


def test_prompt_text_needs_every_description():
    t = build_taxonomy(_complaints())
    with pytest.raises(ValueError, match="description"):
        taxonomy_prompt_text(t, _descriptions(t).iloc[1:])


def test_normalize_label_exact_and_near_miss():
    t = build_taxonomy(_complaints())
    lab = "Credit reporting|Incorrect information|Wrong name"
    assert normalize_label(t, lab) == lab
    assert normalize_label(t, " credit REPORTING|Incorrect  information|wrong name ") == lab
    assert normalize_label(t, "Credit reporting|Incorrect information|Nope") is None
    assert normalize_label(t, "") is None


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
