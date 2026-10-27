import numpy as np
import pandas as pd

from nlp.config import Config
from nlp.sampling import make_splits, stratified_sample
from nlp.schema import SPLITS_COLS


def _cfg(**kw):
    return Config(classifier_model="a", judge_model="b", prompt_version="v", rpm=10, **kw)


def _frame(n=30000, seed=0):
    rng = np.random.default_rng(seed)
    prods = rng.choice(["A", "B", "C", "D"], size=n, p=[0.5, 0.3, 0.19, 0.01])
    issues = rng.choice(["i1", "i2", "i3"], size=n)
    subs = rng.choice(["s1", "s2", "(none)"], size=n)
    return pd.DataFrame(
        {
            "complaint_id": [f"c{i}" for i in range(n)],
            "product": prods,
            "issue": issues,
            "sub_issue": subs,
        }
    )


def test_sampling_rare_classes():
    df = _frame(500)
    df.loc[:1, "product"] = "RARE"
    df.loc[2:, "product"] = df.loc[2:, "product"].replace("RARE", "A")
    out = stratified_sample(df, 100, by="product", seed=1, min_per_class=10)
    assert len(out) == 100
    assert (out["product"] == "RARE").sum() == 2


def test_sampling_proportional_and_capped():
    df = _frame(3000)
    out = stratified_sample(df, 300, by="product", seed=1, min_per_class=0)
    assert len(out) == 300
    assert stratified_sample(df, 10**6, by="product", seed=1, min_per_class=0).shape[0] == 3000


def test_splits_disjoint_and_deterministic():
    df = _frame(5000)
    cfg = _cfg(train_max=1000, tuning_n=50, test_n=100, self_consistency_n=20)
    a = make_splits(df, cfg)
    b = make_splits(df, cfg)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == SPLITS_COLS
    assert a["complaint_id"].is_unique
    assert not a["hand_check"].any()
    sc = a[a["self_consistency"]]
    assert len(sc) == 20 and (sc["split"] == "test").all()


def test_splits_sizes():
    df = _frame(30000)
    s = make_splits(df, _cfg())
    counts = s["split"].value_counts()
    assert counts["tuning"] == 200
    assert counts["test"] == 800
    assert counts["train"] <= 20000
    assert s["self_consistency"].sum() == 150
