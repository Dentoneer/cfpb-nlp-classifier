"""TF-IDF + logistic regression baseline."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nlp.classifiers.base import one_class_proba

# Keeps single-character tokens (fallback for degenerate corpora).
TOKEN_ANY = r"(?u)" + chr(92) + "w+"


def fit_lr(X, path_ids: list[str], C: float):
    """LR fit that tolerates a single training class. Returns (classes, model or None)."""
    classes = sorted(set(path_ids))
    if len(classes) == 1:
        return classes, None
    model = LogisticRegression(C=C, max_iter=300)
    model.fit(X, path_ids)
    return list(model.classes_), model


class TfidfLR:
    name = "tfidf_lr"

    def __init__(self, max_features: int = 100_000, ngram_range=(1, 2), C: float = 4.0):
        self.max_features = max_features
        self.ngram_range = tuple(ngram_range)
        self.C = C
        self.model_id = "tfidf+logreg"

    def fingerprint(self) -> dict:
        return {
            "model_id": self.model_id,
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "C": self.C,
        }

    def fit(self, texts: list[str], path_ids: list[str]) -> None:
        kw = {"max_features": self.max_features, "ngram_range": self.ngram_range}
        self.vec = TfidfVectorizer(sublinear_tf=True, **kw)
        try:
            X = self.vec.fit_transform(texts)
        except ValueError:  # empty vocabulary (e.g. only 1-char tokens): keep them
            self.vec = TfidfVectorizer(sublinear_tf=True, token_pattern=TOKEN_ANY, **kw)
            X = self.vec.fit_transform(texts)
        self.classes_, self.lr = fit_lr(X, list(path_ids), self.C)

    def predict_proba(self, texts: list[str]) -> tuple[list[str], np.ndarray]:
        X = self.vec.transform(texts)
        if self.lr is None:
            return self.classes_, one_class_proba(self.classes_, len(texts))
        return self.classes_, self.lr.predict_proba(X)
