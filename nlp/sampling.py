"""Stratified sampling and train/tuning/test split construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nlp.config import Config
from nlp.schema import SPLITS_COLS, path_id


def _allocate(sizes: pd.Series, n: int, min_per_class: int) -> pd.Series:
    """Proportional allocation with a per-class floor; never exceeds class size."""
    total = int(sizes.sum())
    if n >= total:
        return sizes.copy()
    base = sizes.clip(upper=min_per_class)
    target = n * sizes / total
    alloc = np.maximum(base, np.floor(target)).astype(int)
    alloc = np.minimum(alloc, sizes)
    while alloc.sum() < n:
        room = (alloc < sizes)
        if not room.any():
            break
        alloc[(target - alloc)[room].idxmax()] += 1
    while alloc.sum() > n:
        room = alloc > base
        if not room.any():
            break
        alloc[(target - alloc)[room].idxmin()] -= 1
    return alloc


def stratified_sample(
    complaints: pd.DataFrame, n: int, by: str, seed: int, min_per_class: int
) -> pd.DataFrame:
    """Proportional sample of n rows by `by`; each class first gets min(min_per_class, size)."""
    sizes = complaints.groupby(by).size().sort_index()
    alloc = _allocate(sizes, n, min_per_class)
    parts = [
        g.sample(n=int(alloc[k]), random_state=seed)
        for k, g in complaints.groupby(by, sort=True)
        if alloc[k] > 0
    ]
    if not parts:
        return complaints.iloc[0:0]
    return pd.concat(parts).sort_values("complaint_id", kind="stable")


def make_splits(complaints: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Disjoint tuning / test / train assignment plus the self-consistency subset of test."""
    df = complaints.copy()
    df["path_id"] = [path_id(p, i, s) for p, i, s in zip(df["product"], df["issue"], df["sub_issue"])]
    tuning = stratified_sample(df, cfg.tuning_n, "product", cfg.seed, min_per_class=3)
    rest = df[~df["complaint_id"].isin(tuning["complaint_id"])]
    test = stratified_sample(rest, cfg.test_n, "product", cfg.seed, min_per_class=10)
    rest = rest[~rest["complaint_id"].isin(test["complaint_id"])]
    train = stratified_sample(rest, min(cfg.train_max, len(rest)), "path_id", cfg.seed, 0)
    sc = stratified_sample(test, cfg.self_consistency_n, "product", cfg.seed, min_per_class=1)
    out = pd.concat(
        [
            pd.DataFrame({"complaint_id": tuning["complaint_id"].values, "split": "tuning"}),
            pd.DataFrame({"complaint_id": test["complaint_id"].values, "split": "test"}),
            pd.DataFrame({"complaint_id": train["complaint_id"].values, "split": "train"}),
        ],
        ignore_index=True,
    )
    out["hand_check"] = False
    out["self_consistency"] = out["complaint_id"].isin(sc["complaint_id"])
    return out[SPLITS_COLS].sort_values("complaint_id", kind="stable").reset_index(drop=True)
