import itertools

import pandas as pd
import pytest

from nlp.data import (
    HF_SOURCE,
    FilterParams,
    clean_text,
    load_clean,
    max_date_received,
    product_counts_by_month,
)
from nlp.schema import COMPLAINTS_COLS

_ids = itertools.count(1)


def _text(i: int, n: int = 260) -> str:
    base = f"Unique complaint body number {i} describing a distinct billing problem in detail. "
    return (base * 10)[:n]


def row(**kw):
    i = next(_ids)
    r = {
        "complaint_id": f"{i:06d}",
        "date_received": pd.Timestamp("2025-03-01", tz="UTC"),
        "product": "Credit card",
        "sub_product": None,
        "issue": "Billing",
        "sub_issue": "Late fee",
        "consumer_complaint_narrative": _text(i),
        "company": "Acme",
        "state": "TX",
    }
    r.update(kw)
    return r


def write(tmp_path, rows):
    p = tmp_path / "c.parquet"
    pd.DataFrame(rows).to_parquet(p)
    return str(p)


PARAMS = FilterParams(min_path_rows=1)


def test_filter_nulls_and_window(tmp_path):
    rows = [
        row(),
        row(consumer_complaint_narrative=None),
        row(product=None),
        row(issue=None),
        row(date_received=pd.Timestamp("2023-12-31", tz="UTC")),
        row(date_received=pd.Timestamp("2024-01-01", tz="UTC")),
        row(date_received=pd.Timestamp("2026-07-06", tz="UTC")),
        row(date_received=pd.Timestamp("2026-07-07", tz="UTC")),
    ]
    df, _ = load_clean(write(tmp_path, rows), PARAMS)
    assert list(df.columns) == COMPLAINTS_COLS
    assert len(df) == 3
    assert df["date_received"].min() == pd.Timestamp("2024-01-01")
    assert df["date_received"].max() == pd.Timestamp("2026-07-06")


def test_mass_template_dropped_entirely(tmp_path):
    t = "I want to dispute this charge. " * 10
    rows = [row(consumer_complaint_narrative=t if k % 2 else t.upper() + "  ") for k in range(5)]
    rows.append(row())
    df, _ = load_clean(write(tmp_path, rows), PARAMS)
    assert len(df) == 1


def test_small_duplicate_keeps_earliest(tmp_path):
    t = "Duplicate narrative text for testing purposes. " * 6
    rows = [
        row(complaint_id="000300", date_received=pd.Timestamp("2025-05-01", tz="UTC"),
            consumer_complaint_narrative=t),
        row(complaint_id="000200", date_received=pd.Timestamp("2025-02-01", tz="UTC"),
            consumer_complaint_narrative=t),
        row(complaint_id="000100", date_received=pd.Timestamp("2025-02-01", tz="UTC"),
            consumer_complaint_narrative=t),
    ]
    df, _ = load_clean(write(tmp_path, rows), PARAMS)
    assert len(df) == 1
    assert df.iloc[0]["complaint_id"] == "000100"
    assert df.iloc[0]["narrative"] == t


def test_short_and_redacted_dropped(tmp_path):
    def pad(n, x=0):
        return "X" * x + ("a" * (n - x))

    rows = [
        row(consumer_complaint_narrative=pad(199)),
        row(consumer_complaint_narrative=pad(200)),
        row(consumer_complaint_narrative=pad(200, 62)),  # 31% X
        row(consumer_complaint_narrative=pad(200, 60)),  # 30% X kept
    ]
    df, _ = load_clean(write(tmp_path, rows), PARAMS)
    assert sorted(len(n) for n in df["narrative"]) == [200, 200]
    assert sorted(n.count("X") for n in df["narrative"]) == [0, 60]


def test_null_sub_issue_rules(tmp_path):
    rows = [row(issue="Has subs", sub_issue=f"s{k % 2}") for k in range(19)]
    rows.append(row(issue="Has subs", sub_issue=None))  # 95% non-null -> dropped
    rows += [row(issue="No subs", sub_issue=None) for _ in range(3)]
    df, _ = load_clean(write(tmp_path, rows), PARAMS)
    assert len(df[df["issue"] == "Has subs"]) == 19
    assert df["sub_issue"][df["issue"] == "Has subs"].notna().all()
    assert (df["sub_issue"][df["issue"] == "No subs"] == "(none)").all()
    assert (df["issue"] == "No subs").sum() == 3


def test_rare_paths_dropped(tmp_path):
    rows = [row(sub_issue="common") for _ in range(3)] + [row(sub_issue="rare") for _ in range(2)]
    df, _ = load_clean(write(tmp_path, rows), FilterParams(min_path_rows=3))
    assert len(df) == 3
    assert set(df["sub_issue"]) == {"common"}


def test_filter_report_counts_add_up(tmp_path):
    t = "Template text repeated many times in the data. " * 6
    rows = [row() for _ in range(6)]
    rows += [row(consumer_complaint_narrative=None)]
    rows += [row(date_received=pd.Timestamp("2020-01-01", tz="UTC"))]
    rows += [row(consumer_complaint_narrative=t) for _ in range(5)]
    rows += [row(consumer_complaint_narrative="short")]
    df, rep = load_clean(write(tmp_path, rows), PARAMS)
    assert list(rep.columns) == ["step", "rows_before", "rows_removed", "rows_after"]
    assert list(rep["step"]) == [
        "nulls", "window", "templates", "low_signal", "sub_issue_consistency", "rare_paths",
    ]
    assert rep["rows_before"].iloc[0] == len(rows)
    assert (rep["rows_before"] - rep["rows_removed"] == rep["rows_after"]).all()
    assert list(rep["rows_before"].iloc[1:]) == list(rep["rows_after"].iloc[:-1])
    assert rep["rows_after"].iloc[-1] == len(df) == 6
    assert list(rep["rows_removed"])[:4] == [1, 1, 5, 1]


def test_clean_text_truncates_and_keeps_redactions():
    out = clean_text("a  XXXX\n\nb" * 1000, 50)
    assert len(out) <= 50
    assert "XXXX" in out
    assert "  " not in out


def test_product_counts_by_month(tmp_path):
    rows = [
        row(date_received=pd.Timestamp("2025-03-05", tz="UTC")),
        row(date_received=pd.Timestamp("2025-03-20", tz="UTC")),
        row(date_received=pd.Timestamp("2025-04-01", tz="UTC")),
        row(consumer_complaint_narrative=None),
    ]
    out = product_counts_by_month(write(tmp_path, rows), PARAMS)
    assert list(out.columns) == ["month", "product", "issue", "n"]
    assert dict(zip(out["month"], out["n"])) == {"2025-03": 2, "2025-04": 1}


def test_max_date_received_ignores_rows_without_narrative(tmp_path):
    rows = [
        row(date_received=pd.Timestamp("2025-03-05", tz="UTC")),
        row(date_received=pd.Timestamp("2025-04-09", tz="UTC")),
        row(date_received=pd.Timestamp("2025-05-01", tz="UTC"), consumer_complaint_narrative=None),
    ]
    assert str(max_date_received(write(tmp_path, rows))) == "2025-04-09"


@pytest.mark.slow
def test_hf_source_reads_one_row():
    import duckdb

    r = duckdb.sql(f"SELECT complaint_id FROM read_parquet('{HF_SOURCE}') LIMIT 1").fetchall()
    assert len(r) == 1
