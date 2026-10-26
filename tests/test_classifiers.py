import numpy as np
import pandas as pd
import pytest

from nlp.classifiers.base import (
    fit_or_load,
    oof_proba,
    probs_long,
    to_predictions,
)
from nlp.classifiers.embed import EmbedKNN, EmbedLR, FakeEncoder
from nlp.classifiers.tfidf import TfidfLR
from nlp.config import Config
from nlp.schema import PREDICTIONS_COLS, path_id

A = path_id("Credit reporting", "Incorrect information", "(none)")
B = path_id("Debt collection", "Not my debt", "Never owed")
C = path_id("Mortgage", "Payment trouble", "(none)")

TEXTS = (
    ["wrong account on my credit report error"] * 4
    + ["collector says I owe debt that is not mine"] * 4
    + ["mortgage payment servicer escrow trouble"] * 2
)
LABELS = [A] * 4 + [B] * 4 + [C] * 2


def _cfg(tmp_path):
    return Config(
        classifier_model="m", judge_model="j", prompt_version="v", rpm=1, artifacts_dir=tmp_path
    )


def test_tfidf_learns_toy():
    clf = TfidfLR()
    clf.fit(TEXTS, LABELS)
    classes, p = clf.predict_proba(TEXTS)
    pred = [classes[i] for i in p.argmax(axis=1)]
    assert np.mean([a == b for a, b in zip(pred, LABELS, strict=True)]) == 1.0
    assert np.allclose(p.sum(axis=1), 1.0)


def test_knn_k_larger_than_class():
    knn = EmbedKNN("fake", k=15, encoder=FakeEncoder(), cache={})
    knn.fit(TEXTS, LABELS)  # class C has 2 examples, k=15 > 10 examples
    classes, p = knn.predict_proba(["mortgage escrow payment"])
    assert classes[int(p.argmax())] == C
    assert np.isclose(p.sum(), 1.0)


def test_classifiers_handle_rare_and_unseen_paths():
    texts = TEXTS + ["unique rare complaint zebra"]
    labels = LABELS + [path_id("Other", "Rare", "(none)")]
    unseen = ["totally new text about satellites", ""]
    enc = FakeEncoder()
    for clf in [
        TfidfLR(),
        EmbedLR("fake", encoder=enc, cache={}),
        EmbedKNN("fake", k=15, encoder=enc, cache={}),
    ]:
        clf.fit(texts, labels)
        classes, p = clf.predict_proba(unseen)
        assert set(classes) <= set(labels)
        assert p.shape == (2, len(classes))
        df = to_predictions(clf.name, "m", ["x", "y"], classes, p)
        assert df["product"].notna().all()
        assert {path_id(r.product, r.issue, r.sub_issue) for r in df.itertuples()} <= set(labels)


def test_single_class_training_does_not_crash():
    for clf in [TfidfLR(), EmbedLR("fake", encoder=FakeEncoder(), cache={})]:
        clf.fit(["a b", "c d"], [A, A])
        classes, p = clf.predict_proba(["a"])
        assert classes == [A] and np.allclose(p, 1.0)


def test_to_predictions_schema():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    df = to_predictions("tfidf_lr", "m1", ["c1", "c2"], [A, B, C], probs)
    assert list(df.columns) == PREDICTIONS_COLS
    assert df.loc[0, "product"] == "Credit reporting"
    assert df.loc[1, "sub_issue"] == "(none)"
    assert df.loc[1, "issue"] == "Payment trouble"
    assert df["confidence"].tolist() == [0.7, 0.7]
    assert (df["prompt_version"] == "n/a").all()
    assert df["summary"].isna().all() and df["taxonomy_hash"].isna().all()
    assert df["run_ts"].notna().all()


def test_probs_long_shape():
    df = probs_long(["c1", "c2"], [A, B], np.array([[0.6, 0.4], [0.1, 0.9]]))
    assert list(df.columns) == ["complaint_id", "path_id", "prob"]
    assert len(df) == 4
    assert df[(df.complaint_id == "c2") & (df.path_id == B)]["prob"].iloc[0] == 0.9


def test_oof_proba_global_classes_and_rare():
    rare = path_id("Other", "Rare", "(none)")
    classes, p = oof_proba(TfidfLR, TEXTS + ["zebra complaint"], LABELS + [rare], n_splits=3, seed=1)
    assert classes == sorted(set(LABELS + [rare]))
    assert p.shape == (11, len(classes))
    # the singleton class is never in the training data of its own held-out fold
    assert p[-1, classes.index(rare)] == 0.0


def test_fit_or_load_roundtrip(tmp_path):
    cfg = _cfg(tmp_path)
    df = pd.DataFrame({"complaint_id": range(10), "narrative": TEXTS, "path_id": LABELS})
    clf = fit_or_load(cfg, TfidfLR(), df)
    assert len(list((tmp_path / "models").glob("tfidf_lr-*/model.joblib"))) == 1

    class Boom(TfidfLR):
        def fit(self, *a, **k):
            raise AssertionError("should load, not fit")

    loaded = fit_or_load(cfg, Boom(), df)
    c1, p1 = clf.predict_proba(TEXTS[:2])
    c2, p2 = loaded.predict_proba(TEXTS[:2])
    assert c1 == c2 and np.allclose(p1, p2)


def test_embed_lr_persists_without_cache(tmp_path):
    cfg = _cfg(tmp_path)
    df = pd.DataFrame({"complaint_id": range(10), "narrative": TEXTS, "path_id": LABELS})
    fit_or_load(cfg, EmbedLR("fake", encoder=FakeEncoder(), cache={}), df)
    loaded = fit_or_load(cfg, EmbedLR("fake"), df)
    _, p = loaded.predict_proba(TEXTS[:1])
    assert np.isclose(p.sum(), 1.0)


def test_fit_or_load_refits_when_hyperparameters_change(tmp_path):
    cfg = _cfg(tmp_path)
    df = pd.DataFrame({"complaint_id": range(10), "narrative": TEXTS, "path_id": LABELS})
    fit_or_load(cfg, TfidfLR(C=1.0), df)
    fit_or_load(cfg, TfidfLR(C=1.0), df)  # same config: reuses
    assert len(list((tmp_path / "models").glob("tfidf_lr-*"))) == 1
    fit_or_load(cfg, TfidfLR(C=2.0), df)  # changed: new dir
    assert len(list((tmp_path / "models").glob("tfidf_lr-*"))) == 2
    fit_or_load(cfg, EmbedKNN("m1", encoder=FakeEncoder(), cache={}), df)
    fit_or_load(cfg, EmbedKNN("m2", encoder=FakeEncoder(), cache={}), df)
    assert len(list((tmp_path / "models").glob("embed_knn-*"))) == 2


@pytest.mark.slow
def test_modernbert_save_load_roundtrip(tmp_path):
    from nlp.classifiers.modernbert import ModernBertClassifier

    cfg = _cfg(tmp_path)
    texts, labels = TEXTS[:8], LABELS[:8]  # two classes
    df = pd.DataFrame({"complaint_id": range(8), "narrative": texts, "path_id": labels})
    clf = fit_or_load(cfg, ModernBertClassifier(max_len=64, epochs=1, batch_size=4), df)
    c1, p1 = clf.predict_proba(texts)
    assert len(c1) == 2 and p1.shape == (8, 2)
    loaded = fit_or_load(cfg, ModernBertClassifier(max_len=64, epochs=1, batch_size=4), df)
    c2, p2 = loaded.predict_proba(texts)
    assert c1 == c2 and np.allclose(p1, p2, atol=1e-4)
    assert (p1.argmax(axis=1) == p2.argmax(axis=1)).all()


@pytest.mark.slow
def test_modernbert_single_class():
    from nlp.classifiers.modernbert import ModernBertClassifier

    clf = ModernBertClassifier()
    clf.fit(["a", "b"], [A, A])
    classes, p = clf.predict_proba(["x"])
    assert classes == [A] and np.allclose(p, 1.0)


@pytest.mark.slow
def test_real_encoder_embeds():
    from nlp.classifiers.embed import embed_texts

    v = embed_texts("BAAI/bge-base-en-v1.5", ["hello world"], {})
    assert np.isclose(np.linalg.norm(v[0]), 1.0)


@pytest.mark.slow
def test_modernbert_fits_toy():
    from nlp.classifiers.modernbert import ModernBertClassifier

    clf = ModernBertClassifier(max_len=64, epochs=1, batch_size=4)
    clf.fit(TEXTS, LABELS)
    _, p = clf.predict_proba(TEXTS[:2])
    assert p.shape == (2, 3)
