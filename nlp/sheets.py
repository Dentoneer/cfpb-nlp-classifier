"""Google Sheets I/O: publish/read the taxonomy and the human review sheet.

gspread is only in the `colab` extra, so it is never imported here; every function takes an
already-authorised gspread client `gc`.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from nlp.schema import HUMAN_REVIEW_COLS, TAXONOMY_COLS, VERDICTS, path_id

__all__ = [
    "publish_taxonomy",
    "read_review_sheet",
    "read_taxonomy",
    "taxonomy_snapshot_hash",
    "write_review_sheet",
]

REVIEW_LEVELS = ["product", "issue", "sub_issue"]
SUMMARY_VALUES = ["yes", "no"]


def taxonomy_snapshot_hash(taxonomy: pd.DataFrame) -> str:
    """sha256 of the sorted path_ids (first 12 hex chars); independent of row order."""
    joined = "\n".join(sorted(taxonomy["path_id"].astype(str)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def _cell(v: object) -> object:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else v


def _values(df: pd.DataFrame) -> list[list[object]]:
    return [list(df.columns)] + [[_cell(v) for v in row] for row in df.itertuples(index=False)]


def publish_taxonomy(gc, taxonomy: pd.DataFrame, title: str = "CFPB taxonomy (phase1)") -> str:
    ss = gc.create(title)
    ss.sheet1.update(_values(taxonomy[TAXONOMY_COLS]), range_name="A1")
    return ss.url


def read_taxonomy(gc, url: str) -> pd.DataFrame:
    records = gc.open_by_url(url).sheet1.get_all_records()
    df = pd.DataFrame(records, columns=TAXONOMY_COLS[:3])
    for c in TAXONOMY_COLS[:3]:
        df[c] = df[c].astype(str).str.strip()
    blank = (df["product"] == "") | (df["issue"] == "")
    if blank.any():
        raise ValueError(f"taxonomy has blank product/issue in {int(blank.sum())} row(s)")
    df["path_id"] = [
        path_id(p, i, s) for p, i, s in zip(df["product"], df["issue"], df["sub_issue"])
    ]
    dup = df["path_id"].duplicated()
    if dup.any():
        raise ValueError(f"taxonomy has duplicate paths: {df.loc[dup, 'path_id'].tolist()[:5]}")
    return df[TAXONOMY_COLS]


def _validation_request(sheet_id: int, col: int, n_rows: int, values: list[str]) -> dict:
    return {
        "setDataValidation": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": 1,
                "endRowIndex": n_rows + 1,
                "startColumnIndex": col,
                "endColumnIndex": col + 1,
            },
            "rule": {
                "condition": {
                    "type": "ONE_OF_LIST",
                    "values": [{"userEnteredValue": v} for v in values],
                },
                "showCustomUi": True,
                "strict": False,
            },
        }
    }


def write_review_sheet(gc, rows: pd.DataFrame, title: str) -> str:
    """Write the wide review frame and add dropdowns on verdict_* and summary_ok."""
    ss = gc.create(title)
    ws = ss.sheet1
    ws.update(_values(rows), range_name="A1")
    cols = list(rows.columns)
    reqs = [
        _validation_request(ws.id, cols.index(f"verdict_{lv}"), len(rows), VERDICTS)
        for lv in REVIEW_LEVELS
    ]
    reqs.append(_validation_request(ws.id, cols.index("summary_ok"), len(rows), SUMMARY_VALUES))
    ss.batch_update({"requests": reqs})
    return ss.url


def read_review_sheet(gc, url: str) -> pd.DataFrame:
    """Long `human_review` frame: one row per complaint x level; blank verdicts kept as ''."""
    records = gc.open_by_url(url).sheet1.get_all_records()
    out = []
    for r in records:
        summary_ok = str(r.get("summary_ok", "")).strip()
        for lv in REVIEW_LEVELS:
            out.append(
                {
                    "complaint_id": r["complaint_id"],
                    "level": lv,
                    "verdict": str(r.get(f"verdict_{lv}", "")).strip(),
                    "summary_ok": summary_ok,
                }
            )
    return pd.DataFrame(out, columns=HUMAN_REVIEW_COLS)
