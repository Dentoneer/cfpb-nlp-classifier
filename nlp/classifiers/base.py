"""Shared classifier protocol and helpers. Every classifier predicts a full path_id."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from nlp.config import Config
from nlp.schema import PREDICTIONS_COLS

PATH_SEP = " || "


class PathClassifier(Protocol):
    name: str

    def fingerprint(self) -> dict: ...

    def fit(self, texts: list[str], path_ids: list[str]) -> None: ...

    def predict_proba(self, texts: list[str]) -> tuple[list[str], np.ndarray]: ...


def one_class_proba(classes: list[str], n: int) -> np.ndarray:
    """Degenerate case: a single training class gets probability 1."""
    return np.ones((n, len(classes)))


def to_predictions(
    classifier_name: str,
    model_id: str,
    complaint_ids: Sequence[str],
    classes: list[str],
    probs: np.ndarray,
) -> pd.DataFrame:
    probs = np.asarray(probs)
    best = probs.argmax(axis=1)
    run_ts = datetime.now(UTC).isoformat()
    rows = []
    for cid, b, p in zip(complaint_ids, best, probs, strict=True):
        product, issue, sub_issue = classes[b].split(PATH_SEP, 2)
        rows.append(
            {
                "complaint_id": cid,
                "classifier": classifier_name,
                "model_id": model_id,
                "prompt_version": "n/a",
                "product": product,
                "issue": issue,
                "sub_issue": sub_issue,
                "summary": None,
                "confidence": float(p[b]),
                "taxonomy_hash": None,
                "run_ts": run_ts,
            }
        )
    return pd.DataFrame(rows, columns=PREDICTIONS_COLS)


def probs_long(complaint_ids: Sequence[str], classes: list[str], probs: np.ndarray) -> pd.DataFrame:
    """One row per complaint x class: complaint_id, path_id, prob."""
    probs = np.asarray(probs)
    ids = list(complaint_ids)
    return pd.DataFrame(
        {
            "complaint_id": np.repeat(np.asarray(ids, dtype=object), len(classes)),
            "path_id": np.tile(np.asarray(classes, dtype=object), len(ids)),
            "prob": probs.reshape(-1),
        }
    )


def oof_proba(
    make_clf: Callable[[], PathClassifier],
    texts: Sequence[str],
    path_ids: Sequence[str],
    n_splits: int = 5,
    seed: int = 20261026,
) -> tuple[list[str], np.ndarray]:
    """Out-of-fold probabilities over one global class list (sorted unique path_ids).

    Classes absent from a fold's training data get probability 0 in that fold.
    """
    texts = list(texts)
    path_ids = list(path_ids)
    n = len(texts)
    classes = sorted(set(path_ids))
    col = {c: i for i, c in enumerate(classes)}
    out = np.zeros((n, len(classes)))
    if n < 2:
        return classes, out
    n_splits = max(2, min(n_splits, n))

    # Stratify on classes that can fill every fold; pool the rest into one bucket.
    counts = pd.Series(path_ids).value_counts()
    strat = [p if counts[p] >= n_splits else "__rare__" for p in path_ids]
    n_rare = strat.count("__rare__")
    if 0 < n_rare < n_splits:
        strat = [counts.index[0] if s == "__rare__" else s for s in strat]
    try:
        folds = list(
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
                np.zeros(n), strat
            )
        )
    except ValueError:
        folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(texts))

    for tr, te in folds:
        clf = make_clf()
        clf.fit([texts[i] for i in tr], [path_ids[i] for i in tr])
        fold_classes, p = clf.predict_proba([texts[i] for i in te])
        out[np.ix_(te, [col[c] for c in fold_classes])] = p
    return classes, out


def model_dir(cfg: Config, clf: PathClassifier) -> Path:
    """models/<name>-<8 hex of fingerprint>, so changed hyperparameters never reuse a model."""
    fp = json.dumps(clf.fingerprint(), sort_keys=True, default=str)
    return Path(cfg.artifacts_dir) / "models" / f"{clf.name}-{hashlib.sha1(fp.encode()).hexdigest()[:8]}"


def fit_or_load(cfg: Config, clf: PathClassifier, train_df: pd.DataFrame) -> PathClassifier:
    """Fit on train_df (complaint_id, narrative, path_id) or load a persisted model."""
    d = model_dir(cfg, clf)
    texts, labels = train_df["narrative"].tolist(), train_df["path_id"].tolist()
    if hasattr(clf, "save_pretrained"):
        if (d / "meta.json").exists():
            clf.load_pretrained(d)
            return clf
        clf.fit(texts, labels)
        d.mkdir(parents=True, exist_ok=True)
        clf.save_pretrained(d)
        return clf
    f = d / "model.joblib"
    if f.exists():
        return joblib.load(f)
    clf.fit(texts, labels)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, f)
    return clf
