"""Load the CFPB complaint mirror with DuckDB and apply the noise filters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

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


def load_clean(source: str, params: FilterParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (complaints frame, filter_report frame) after the six noise filters."""
    con = _connect()
    src = _quote(source)
    con.execute(
        f"""
        CREATE TEMP TABLE s0 AS
        SELECT complaint_id, CAST(date_received AS DATE) AS date_received, product, issue,
               sub_issue, consumer_complaint_narrative AS narrative, company, state
        FROM read_parquet({src})
        """
    )
    steps: list[tuple[str, str]] = [
        (
            "nulls",
            """SELECT * FROM {prev}
               WHERE narrative IS NOT NULL AND product IS NOT NULL AND issue IS NOT NULL""",
        ),
        (
            "window",
            """SELECT * FROM {prev} WHERE date_received >= CAST($start AS DATE)
               AND date_received < CAST($end AS DATE)""",
        ),
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
        "start": params.start,
        "end": params.end,
        "copies": params.template_min_copies,
        "min_chars": params.min_chars,
        "max_x": params.max_x_frac,
        "req_frac": params.sub_issue_required_frac,
        "min_rows": params.min_path_rows,
    }
    report = []
    prev = "s0"
    for i, (name, sql) in enumerate(steps, start=1):
        cur = f"s{i}"
        used = {k: v for k, v in binds.items() if f"${k}" in sql}
        con.execute(f"CREATE TEMP TABLE {cur} AS {sql.format(prev=prev)}", used)
        before = con.execute(f"SELECT count(*) FROM {prev}").fetchone()[0]
        after = con.execute(f"SELECT count(*) FROM {cur}").fetchone()[0]
        report.append((name, before, before - after, after))
        prev = cur
    df = con.execute(
        f"SELECT * FROM {prev} ORDER BY date_received, complaint_id"
    ).df()
    con.close()
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
