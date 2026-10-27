"""Google Sheets I/O: publish/read the taxonomy and the human review sheet.

gspread is only in the `colab` extra, so it is never imported here; every function takes an
already-authorised gspread client `gc`.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from nlp.schema import HUMAN_REVIEW_COLS, TAXONOMY_COLS, VERDICTS, label_of, path_id

__all__ = [
    "TAXONOMY_SHEET_COLS",
    "publish_taxonomy",
    "read_description_edits",
    "read_review_sheet",
    "read_taxonomy",
    "rewrite_taxonomy_sheet",
    "taxonomy_snapshot_hash",
    "write_review_sheet",
]

REVIEW_LEVELS = ["product", "issue", "sub_issue"]
SUMMARY_VALUES = ["yes", "no"]
TAXONOMY_SHEET_COLS = ["label", "product", "issue", "sub_issue"]


def taxonomy_snapshot_hash(taxonomy: pd.DataFrame) -> str:
    """sha256 of the sorted path_ids (first 12 hex chars); independent of row order."""
    joined = "\n".join(sorted(taxonomy["path_id"].astype(str)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def _cell(v: object) -> object:
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        v = v.item()
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else v


def _values(df: pd.DataFrame) -> list[list[object]]:
    return [list(df.columns)] + [[_cell(v) for v in row] for row in df.itertuples(index=False)]


def _taxonomy_sheet(taxonomy: pd.DataFrame, descriptions: pd.DataFrame | None) -> pd.DataFrame:
    t = taxonomy.sort_values("label")[TAXONOMY_SHEET_COLS].reset_index(drop=True)
    if descriptions is None:
        return t
    d = dict(zip(descriptions["label"], descriptions["description"]))
    return t.assign(description=[d.get(x, "") for x in t["label"]])


def publish_taxonomy(
    gc,
    taxonomy: pd.DataFrame,
    title: str = "CFPB taxonomy (phase1)",
    descriptions: pd.DataFrame | None = None,
) -> str:
    """New Sheet with one row per category: label, product, issue, sub_issue (+ description)."""
    ss = gc.create(title)
    ss.sheet1.update(_values(_taxonomy_sheet(taxonomy, descriptions)), range_name="A1")
    return ss.url


def rewrite_taxonomy_sheet(
    gc, url: str, taxonomy: pd.DataFrame, descriptions: pd.DataFrame
) -> str:
    """Replace the taxonomy Sheet's contents with label, product, issue, sub_issue, description."""
    ws = gc.open_by_url(url).sheet1
    ws.clear()
    ws.update(_values(_taxonomy_sheet(taxonomy, descriptions)), range_name="A1")
    return url


def read_description_edits(gc, url: str) -> pd.DataFrame:
    """(label, description) as currently typed in the taxonomy Sheet, whitespace-trimmed."""
    records = gc.open_by_url(url).sheet1.get_all_records(numericise_ignore=["all"])
    if records and ("label" not in records[0] or "description" not in records[0]):
        raise ValueError(
            "The taxonomy Sheet has no 'label' and 'description' columns. Run Part 0 in "
            "notebook 03 first (it writes them)."
        )
    df = pd.DataFrame(records, columns=["label", "description"])
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    return df


def read_taxonomy(gc, url: str) -> pd.DataFrame:
    records = gc.open_by_url(url).sheet1.get_all_records(numericise_ignore=["all"])
    df = pd.DataFrame(records, columns=TAXONOMY_COLS[:3])
    for c in TAXONOMY_COLS[:3]:
        df[c] = df[c].astype(str).str.strip()
    blank = (df["product"] == "") | (df["issue"] == "")
    if blank.any():
        raise ValueError(f"taxonomy has blank product/issue in {int(blank.sum())} row(s)")
    if (df["sub_issue"] == "").any():
        raise ValueError(
            f"taxonomy has a blank sub-issue in {int((df['sub_issue'] == '').sum())} row(s)"
        )
    parts = list(zip(df["product"], df["issue"], df["sub_issue"]))
    df["path_id"] = [path_id(*x) for x in parts]
    df["label"] = [label_of(*x) for x in parts]
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
    records = gc.open_by_url(url).sheet1.get_all_records(numericise_ignore=["all"])
    out = []
    for r in records:
        summary_ok = str(r.get("summary_ok", "")).strip()
        for lv in REVIEW_LEVELS:
            out.append(
                {
                    "complaint_id": str(r["complaint_id"]).strip(),
                    "level": lv,
                    "verdict": str(r.get(f"verdict_{lv}", "")).strip(),
                    "summary_ok": summary_ok,
                }
            )
    return pd.DataFrame(out, columns=HUMAN_REVIEW_COLS)
