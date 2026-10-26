"""Sentence-embedding classifiers (LR and kNN) sharing an embedding cache."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from nlp.classifiers.base import one_class_proba
from nlp.classifiers.tfidf import fit_lr

_TOKEN = re.compile(r"[a-z0-9]+")

# Process-wide default cache {(encoder_name, text): vector}, shared by EmbedLR/EmbedKNN.
DEFAULT_CACHE: dict[tuple[str, str], np.ndarray] = {}


class FakeEncoder:
    """Deterministic hashed bag-of-words, L2-normalised. No torch; for tests/CI."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.name = f"fake-{dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            for tok in _TOKEN.findall(t.lower()):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


class SentenceTransformerEncoder:
    """Lazily loads a sentence-transformers model on first encode."""

    def __init__(self, model_name: str):
        self.name = model_name
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name)
        return np.asarray(
            self._model.encode(texts, batch_size=64, normalize_embeddings=True), dtype=float
        )

    def __getstate__(self):
        return {"name": self.name, "_model": None}


def embed_texts(
    model_name: str,
    texts: list[str],
    cache: dict | None = None,
    encoder=None,
) -> np.ndarray:
    """Normalised embeddings (n x d), encoding only texts missing from the cache."""
    cache = DEFAULT_CACHE if cache is None else cache
    encoder = encoder or SentenceTransformerEncoder(model_name)
    key = encoder.name
    missing = list(dict.fromkeys(t for t in texts if (key, t) not in cache))
    if missing:
        vecs = np.asarray(encoder.encode(missing), dtype=float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.where(norms == 0, 1.0, norms)
        for t, v in zip(missing, vecs, strict=True):
            cache[(key, t)] = v
    return np.vstack([cache[(key, t)] for t in texts]) if texts else np.zeros((0, 0))


class _EmbedBase:
    def __init__(self, model_name: str, encoder=None, cache: dict | None = None):
        self.model_name = model_name
        self.model_id = model_name
        self.encoder = encoder
        self.cache = cache

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.encoder is None:
            self.encoder = SentenceTransformerEncoder(self.model_name)
        return embed_texts(self.model_name, texts, self.cache, self.encoder)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["cache"] = None  # never persist embeddings with the model
        if isinstance(state.get("encoder"), SentenceTransformerEncoder):
            state["encoder"] = None  # reloaded lazily
        return state


class EmbedLR(_EmbedBase):
    name = "embed_lr"

    def __init__(self, model_name: str, encoder=None, cache=None, C: float = 10.0):
        super().__init__(model_name, encoder, cache)
        self.C = C

    def fit(self, texts: list[str], path_ids: list[str]) -> None:
        self.classes_, self.lr = fit_lr(self._embed(texts), list(path_ids), self.C)

    def predict_proba(self, texts: list[str]) -> tuple[list[str], np.ndarray]:
        X = self._embed(texts)
        if self.lr is None:
            return self.classes_, one_class_proba(self.classes_, len(texts))
        return self.classes_, self.lr.predict_proba(X)


class EmbedKNN(_EmbedBase):
    name = "embed_knn"

    def __init__(self, model_name: str, k: int = 15, encoder=None, cache=None):
        super().__init__(model_name, encoder, cache)
        self.k = k

    def fit(self, texts: list[str], path_ids: list[str]) -> None:
        self.X = self._embed(texts)
        self.classes_ = sorted(set(path_ids))
        col = {c: i for i, c in enumerate(self.classes_)}
        self.y = np.array([col[p] for p in path_ids])

    def predict_proba(self, texts: list[str]) -> tuple[list[str], np.ndarray]:
        Q = self._embed(texts)
        k = max(1, min(self.k, len(self.y)))
        out = np.zeros((len(texts), len(self.classes_)))
        for start in range(0, len(texts), 512):
            sims = Q[start : start + 512] @ self.X.T
            top = np.argsort(-sims, axis=1)[:, :k]
            for r, idx in enumerate(top):
                w = np.clip(sims[r, idx], 0, None)
                if w.sum() == 0:
                    w = np.ones_like(w)
                np.add.at(out[start + r], self.y[idx], w)
                out[start + r] /= out[start + r].sum()
        return self.classes_, out
