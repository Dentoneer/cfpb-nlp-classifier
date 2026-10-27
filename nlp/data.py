"""Load the CFPB complaint mirror with DuckDB and apply the noise filters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

HF_SOURCE = "hf://datasets/Bob-68/cfpb-consumer-complaints/data/*.parquet"

REPORT_COLS = ["step", "rows_before", "rows_removed", "rows_after"]


@dataclass
class FilterParams:
    start: str = "2024-01-01"
    end: str = "2026-07-07"  # exclusive
    template_min_copies: int = 5
    min_chars: int = 200
    max_x_frac: float = 0.30
    sub_issue_required_frac: float = 0.90
    min_path_rows: int = 30


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    return con


def _quote(source: str) -> str:
    return "'" + source.replace("'", "''") + "'"


def load_clean(
    source: str, params: FilterParams, temp_directory: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (complaints frame, filter_report frame) after the six noise filters.

    Memory-safe for the full 16M-row mirror on a Colab VM: the null and window filters run
    inside the source scan (their report counts come from one aggregate, nothing is copied),
    each step table is dropped once the next one exists, and DuckDB may spill to
    `temp_directory` when given.
    """
    con = _connect()
    try:
        return _load_clean(con, _quote(source), params, temp_directory)
    finally:
        con.close()


_SOURCE_COLS = """complaint_id, CAST(date_received AS DATE) AS date_received, product, issue,
               sub_issue, consumer_complaint_narrative AS narrative, company, state"""
_NOT_NULL = """consumer_complaint_narrative IS NOT NULL AND product IS NOT NULL
               AND issue IS NOT NULL"""
_IN_WINDOW = """CAST(date_received AS DATE) >= CAST($start AS DATE)
               AND CAST(date_received AS DATE) < CAST($end AS DATE)"""


def _load_clean(
    con: duckdb.DuckDBPyConnection, src: str, params: FilterParams, temp_directory: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    con.execute("SET preserve_insertion_order = false")
    if temp_directory:
        Path(temp_directory).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = {_quote(str(temp_directory))}")
    window = {"start": params.start, "end": params.end}
    total, not_null, kept = con.execute(
        f"""SELECT count(*), count(*) FILTER (WHERE {_NOT_NULL}),
                   count(*) FILTER (WHERE {_NOT_NULL} AND {_IN_WINDOW})
            FROM read_parquet({src})""",
        window,
    ).fetchone()
    report = [
        ("nulls", total, total - not_null, not_null),
        ("window", not_null, not_null - kept, kept),
    ]
    con.execute(
        f"""CREATE TEMP TABLE s0 AS SELECT {_SOURCE_COLS} FROM read_parquet({src})
            WHERE {_NOT_NULL} AND {_IN_WINDOW}""",
        window,
    )
    steps: list[tuple[str, str]] = [
        (
            "templates",
            r"""
            WITH n AS (
                SELECT *, lower(trim(regexp_replace(narrative, '\s+', ' ', 'g'))) AS norm
                FROM {prev}
            ), r AS (
                SELECT *, count(*) OVER (PARTITION BY norm) AS copies,
                       row_number() OVER (PARTITION BY norm
                                          ORDER BY date_received, complaint_id) AS rn
                FROM n
            )
            SELECT complaint_id, date_received, product, issue, sub_issue, narrative,
                   company, state
            FROM r WHERE copies < $copies AND rn = 1
            """,
        ),
        (
            "low_signal",
            """
            SELECT * FROM {prev}
            WHERE length(narrative) >= $min_chars
              AND (length(narrative) - length(replace(narrative, 'X', '')))
                  / CAST(length(narrative) AS DOUBLE) <= $max_x""",
        ),
        (
            "sub_issue_consistency",
            """
            WITH f AS (
                SELECT product, issue FROM {prev}
                GROUP BY product, issue
                HAVING count(sub_issue) / CAST(count(*) AS DOUBLE) >= $req_frac
            )
            SELECT p.complaint_id, p.date_received, p.product, p.issue,
                   coalesce(p.sub_issue, '(none)') AS sub_issue, p.narrative, p.company, p.state
            FROM {prev} p LEFT JOIN f USING (product, issue)
            WHERE NOT (f.issue IS NOT NULL AND p.sub_issue IS NULL)""",
        ),
        (
            "rare_paths",
            """
            SELECT * FROM {prev} WHERE (product, issue, sub_issue) IN (
                SELECT product, issue, sub_issue FROM {prev}
                GROUP BY ALL HAVING count(*) >= $min_rows)""",
        ),
    ]
    binds = {
        "copies": params.template_min_copies,
        "min_chars": params.min_chars,
        "max_x": params.max_x_frac,
        "req_frac": params.sub_issue_required_frac,
        "min_rows": params.min_path_rows,
    }
    prev, before = "s0", kept
    for i, (name, sql) in enumerate(steps, start=1):
        cur = f"s{i}"
        used = {k: v for k, v in binds.items() if f"${k}" in sql}
        con.execute(f"CREATE TEMP TABLE {cur} AS {sql.format(prev=prev)}", used)
        con.execute(f"DROP TABLE {prev}")
        after = con.execute(f"SELECT count(*) FROM {cur}").fetchone()[0]
        report.append((name, before, before - after, after))
        prev, before = cur, after
    df = con.execute(f"SELECT * FROM {prev} ORDER BY date_received, complaint_id").df()
    return df, pd.DataFrame(report, columns=REPORT_COLS)


def clean_text(s: str, max_chars: int) -> str:
    """Collapse whitespace (keeping XXXX redactions) and truncate to max_chars."""
    return re.sub(r"\s+", " ", s).strip()[:max_chars].rstrip()


def product_counts_by_month(source: str, params: FilterParams) -> pd.DataFrame:
    """Rows with a narrative, counted by month/product/issue (no other filters).

    `params` is accepted for interface symmetry; counts are unfiltered so that the
    window can be chosen from them.
    """
    con = _connect()
    df = con.execute(
        f"""
        SELECT strftime(CAST(date_received AS DATE), '%Y-%m') AS month, product, issue,
               count(*) AS n
        FROM read_parquet({_quote(source)})
        WHERE consumer_complaint_narrative IS NOT NULL
        GROUP BY ALL ORDER BY month, product, issue
        """
    ).df()
    con.close()
    return df


def max_date_received(source: str) -> date:
    """Latest `date_received` among rows with a narrative (the data's end date)."""
    con = _connect()
    d = con.execute(
        f"""SELECT max(CAST(date_received AS DATE)) FROM read_parquet({_quote(source)})
            WHERE consumer_complaint_narrative IS NOT NULL"""
    ).fetchone()[0]
    con.close()
    return d
