"""Hand-check selection, review-sheet rows, and verdict counting."""

from __future__ import annotations

import pandas as pd

from nlp.schema import VERDICTS

__all__ = ["agree_path", "count_verdicts", "is_correct", "review_rows", "select_hand_check"]

_LEVELS = ["product", "issue", "sub_issue"]
_WIDE_COLS = (
    ["complaint_id", "narrative"]
    + [f"cfpb_{lv}" for lv in _LEVELS]
    + [f"gemini_{lv}" for lv in _LEVELS]
    + ["gemini_summary"]
    + [f"verdict_{lv}" for lv in _LEVELS]
    + ["summary_ok"]
)


def _joined(gemini: pd.DataFrame, complaints: pd.DataFrame) -> pd.DataFrame:
    g = gemini[["complaint_id", *_LEVELS]].rename(columns={lv: f"gemini_{lv}" for lv in _LEVELS})
    c = complaints[["complaint_id", *_LEVELS]].rename(columns={lv: f"cfpb_{lv}" for lv in _LEVELS})
    return g.drop_duplicates("complaint_id").merge(c, on="complaint_id", how="inner")


def agree_path(gemini: pd.DataFrame, complaints: pd.DataFrame) -> pd.Series:
    """Bool Series indexed by complaint_id: True iff product, issue and sub_issue all match CFPB."""
    df = _joined(gemini, complaints)
    same = pd.Series(True, index=df.index)
    for lv in _LEVELS:
        same &= df[f"gemini_{lv}"] == df[f"cfpb_{lv}"]
    same.index = df["complaint_id"]
    same.name = "agree_path"
    return same


def select_hand_check(
    gemini_test: pd.DataFrame, complaints: pd.DataFrame, n: int = 200, seed: int = 20261026
) -> pd.DataFrame:
    """Pick up to n test complaints, 50/50 by whether Gemini's full path equals CFPB's.

    Uses only Gemini and CFPB labels (no validator outputs). If disagreements are fewer than
    n/2, all of them are taken and the rest filled with agreements.
    """
    df = _joined(gemini_test, complaints).sort_values("complaint_id")
    same = df["complaint_id"].map(agree_path(gemini_test, complaints))
    agree = df.loc[same, "complaint_id"]
    dis = df.loc[~same, "complaint_id"]
    n_dis = min(len(dis), n - n // 2)
    n_agree = min(len(agree), n - n_dis)
    n_dis = min(len(dis), n - n_agree)  # agreements ran short: top up with disagreements
    parts = [
        pd.DataFrame(
            {
                "complaint_id": dis.sample(n=n_dis, random_state=seed).to_numpy(),
                "stratum": "disagree",
            }
        ),
        pd.DataFrame(
            {
                "complaint_id": agree.sample(n=n_agree, random_state=seed).to_numpy(),
                "stratum": "agree",
            }
        ),
    ]
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values("complaint_id").reset_index(drop=True)


def review_rows(
    selection: pd.DataFrame, gemini: pd.DataFrame, complaints: pd.DataFrame
) -> pd.DataFrame:
    """Wide sheet rows; verdict pre-filled `both OK` per level where Gemini == CFPB."""
    j = _joined(gemini, complaints)
    j = j.merge(
        gemini[["complaint_id", "summary"]].drop_duplicates("complaint_id"),
        on="complaint_id",
        how="left",
    )
    j = j.merge(
        complaints[["complaint_id", "narrative"]].drop_duplicates("complaint_id"),
        on="complaint_id",
        how="left",
    )
    j = j.rename(columns={"summary": "gemini_summary"})
    out = selection[["complaint_id"]].merge(j, on="complaint_id", how="left")
    for lv in _LEVELS:
        out[f"verdict_{lv}"] = (out[f"gemini_{lv}"] == out[f"cfpb_{lv}"]).map(
            {True: "both OK", False: ""}
        )
    out["summary_ok"] = ""
    return out[_WIDE_COLS].fillna("").reset_index(drop=True)


def is_correct(verdict: str) -> bool | None:
    """model right/both OK -> True; label right/both wrong -> False; ambiguous -> None."""
    if verdict in ("model right", "both OK"):
        return True
    if verdict in ("label right", "both wrong"):
        return False
    if verdict == "ambiguous":
        return None
    raise ValueError(f"invalid verdict {verdict!r}; expected one of {VERDICTS}")


def count_verdicts(human_review: pd.DataFrame) -> pd.DataFrame:
    """Per level: n_correct, n_error, n_ambiguous, n_blank (blank = unreviewed, excluded)."""
    rows = []
    for lv in _LEVELS:
        counts = {"n_correct": 0, "n_error": 0, "n_ambiguous": 0, "n_blank": 0}
        for v in human_review.loc[human_review["level"] == lv, "verdict"]:
            if v == "":
                counts["n_blank"] += 1
                continue
            ok = is_correct(v)
            key = "n_ambiguous" if ok is None else ("n_correct" if ok else "n_error")
            counts[key] += 1
        rows.append({"level": lv, **counts})
    return pd.DataFrame(rows)
