import pandas as pd
import pytest

from nlp.schema import HUMAN_REVIEW_COLS, TAXONOMY_COLS, VERDICTS
from nlp.sheets import (
    publish_taxonomy,
    read_review_sheet,
    read_taxonomy,
    taxonomy_snapshot_hash,
    write_review_sheet,
)
from nlp.taxonomy import build_taxonomy


class FakeWorksheet:
    id = 0

    def __init__(self):
        self.values = []

    def update(self, values, range_name=None):
        self.values = [list(r) for r in values]

    def get_all_records(self, numericise_ignore=()):
        if not self.values:
            return []
        head, *body = self.values
        ignore = "all" in numericise_ignore

        def conv(v):
            return int(v) if not ignore and isinstance(v, str) and v.isdigit() else v

        return [{h: conv(v) for h, v in zip(head, r, strict=False)} for r in body]


class FakeSpreadsheet:
    def __init__(self, title, url):
        self.title, self.url = title, url
        self.sheet1 = FakeWorksheet()
        self.batches = []

    def batch_update(self, body):
        self.batches.append(body)


class FakeGC:
    def __init__(self):
        self.books = {}

    def create(self, title):
        ss = FakeSpreadsheet(title, f"https://fake/{len(self.books)}")
        self.books[ss.url] = ss
        return ss

    def open_by_url(self, url):
        return self.books[url]


def _tax():
    df = pd.DataFrame(
        {
            "product": ["A", "A", "B"],
            "issue": ["i1", "i2", "i3"],
            "sub_issue": ["(none)", "s", "(none)"],
        }
    )
    return build_taxonomy(df)


def test_publish_read_taxonomy_roundtrip():
    gc = FakeGC()
    url = publish_taxonomy(gc, _tax())
    out = read_taxonomy(gc, url)
    assert list(out.columns) == TAXONOMY_COLS
    assert out.equals(_tax())


def test_read_taxonomy_rejects_duplicates():
    gc = FakeGC()
    url = publish_taxonomy(gc, _tax())
    ws = gc.books[url].sheet1
    ws.values.append(list(ws.values[1]))
    with pytest.raises(ValueError, match="duplicate"):
        read_taxonomy(gc, url)


def test_read_taxonomy_rejects_blank_product_or_issue():
    gc = FakeGC()
    url = publish_taxonomy(gc, _tax())
    gc.books[url].sheet1.values[1][0] = "  "
    with pytest.raises(ValueError, match="blank"):
        read_taxonomy(gc, url)


def test_snapshot_hash_order_independent():
    t = _tax()
    h = taxonomy_snapshot_hash(t)
    assert len(h) == 12
    assert h == taxonomy_snapshot_hash(t.iloc[::-1].reset_index(drop=True))
    assert h != taxonomy_snapshot_hash(t.iloc[:2])


def _wide():
    return pd.DataFrame(
        {
            "complaint_id": ["c1", "c2"],
            "narrative": ["n1", "n2"],
            "cfpb_product": ["A", "B"],
            "cfpb_issue": ["i1", "i3"],
            "cfpb_sub_issue": ["(none)", "(none)"],
            "gemini_product": ["A", "A"],
            "gemini_issue": ["i1", "i2"],
            "gemini_sub_issue": ["(none)", "s"],
            "gemini_summary": ["s1", "s2"],
            "verdict_product": ["model right", ""],
            "verdict_issue": ["both OK", "ambiguous"],
            "verdict_sub_issue": ["label right", "both wrong"],
            "summary_ok": ["yes", "no"],
        }
    )


def test_review_sheet_roundtrip_with_fake_gc():
    gc = FakeGC()
    url = write_review_sheet(gc, _wide(), "review")
    ss = gc.books[url]
    reqs = [r["setDataValidation"] for b in ss.batches for r in b["requests"]]
    lists = [[v["userEnteredValue"] for v in r["rule"]["condition"]["values"]] for r in reqs]
    assert lists == [VERDICTS, VERDICTS, VERDICTS, ["yes", "no"]]
    assert all(r["rule"]["condition"]["type"] == "ONE_OF_LIST" for r in reqs)
    assert len(reqs) == 4
    out = read_review_sheet(gc, url)
    assert list(out.columns) == HUMAN_REVIEW_COLS
    assert len(out) == 6
    assert out.groupby("complaint_id")["level"].apply(list).to_dict() == {
        "c1": ["product", "issue", "sub_issue"],
        "c2": ["product", "issue", "sub_issue"],
    }
    got = out.set_index(["complaint_id", "level"])["verdict"].to_dict()
    assert got[("c1", "product")] == "model right"
    assert got[("c2", "product")] == ""
    assert got[("c2", "sub_issue")] == "both wrong"
    assert out.loc[out.complaint_id == "c2", "summary_ok"].eq("no").all()


def test_review_ids_keep_leading_zeros():
    gc = FakeGC()
    w = _wide()
    w["complaint_id"] = ["0012345", "0000007"]
    out = read_review_sheet(gc, write_review_sheet(gc, w, "r"))
    assert set(out["complaint_id"]) == {"0012345", "0000007"}


def test_written_values_are_plain_python_types():
    import numpy as np

    gc = FakeGC()
    w = _wide()
    w["verdict_product"] = np.array([1, 2], dtype=np.int64)
    w["gemini_summary"] = np.array([0.5, np.nan])
    w["summary_ok"] = np.array([True, False])
    url = write_review_sheet(gc, w, "r")
    vals = [v for row in gc.books[url].sheet1.values for v in row]
    assert all(type(v) in (str, int, float, bool) for v in vals)
