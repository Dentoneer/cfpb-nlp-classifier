"""Synthetic end-to-end smoke run: every pipeline stage, fake Gemini, fake embeddings.

`run_smoke(artifacts_dir)` generates 3,000 synthetic complaints (6 products / 25 issues /
40 sub-issues, keyword-bearing text) and runs the real pipeline code on them: splits ->
TF-IDF / embed-LR / kNN -> Gemini classification (+ self-consistency) -> validators and judges
-> simulated human verdicts -> ensembles -> leaderboards -> go/no-go. Every stage goes through
`store.stage` and every paid call through the JSONL caches, so a rerun makes zero API calls.
ModernBERT is skipped. Needs only the `.[dev]` dependencies (no torch).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from nlp import store
from nlp.budget import Ledger, RateLimiter
from nlp.classifiers.base import fit_or_load, oof_proba, probs_long, to_predictions
from nlp.classifiers.embed import EmbedKNN, EmbedLR, FakeEncoder
from nlp.classifiers.tfidf import TfidfLR
from nlp.config import Config
from nlp.ensemble import ensemble_learned, ensemble_simple, row_status
from nlp.llm.classify import run_classification
from nlp.llm.client import FakeClient, GeminiClient, LLMResponse
from nlp.llm.judge import judge_classifications, judge_summaries
from nlp.metrics import inverse_weights
from nlp.review import agree_path, is_correct, select_hand_check
from nlp.sampling import make_splits
from nlp.schema import COMPLAINTS_COLS, HUMAN_REVIEW_COLS, VALIDATOR_RESULTS_COLS, path_id
from nlp.scoring import (
    adjusted_accuracy,
    classifier_leaderboard,
    go_no_go,
    validator_leaderboard,
)
from nlp.taxonomy import build_taxonomy
from nlp.validators import (
    v_cleanlab,
    v_hierarchy,
    v_label_match,
    v_second_opinion,
    v_self_confidence,
    v_self_consistency,
)

N_COMPLAINTS = 3000
SEED = 20261026
LEVELS3 = ["product", "issue", "sub_issue"]
PRODUCTS = [
    "Credit reporting", "Debt collection", "Mortgage",
    "Checking account", "Credit card", "Student loan",
]  # fmt: skip
ISSUES_PER_PRODUCT = [5, 4, 4, 4, 4, 4]  # 25 issues
N_TWO_SUB_ISSUES = 15  # 15 issues x 2 sub-issues + 10 issues x "(none)" = 40 paths
_REQ_ID = re.compile(r"^complaint_id: (\S+)$", re.MULTILINE)
_CHOSEN = re.compile(
    r"chosen product: (.*)\nchosen issue: (.*)\nchosen sub_issue: (.*)", re.MULTILINE
)  # fmt: skip


def smoke_config(artifacts_dir: Path) -> Config:
    """Small sizes for 3,000 rows; limiter effectively off; prices for both models."""
    return Config(
        classifier_model="gemini-2.5-flash",
        judge_model="gemini-3.1-flash-lite",
        prompt_version="smoke-v1",
        rpm=6_000_000,
        artifacts_dir=Path(artifacts_dir),
        seed=SEED,
        train_max=2000,
        tuning_n=100,
        test_n=400,
        test_floor=400,
        hand_check_n=100,
        self_consistency_n=60,
        batch_size=10,
        spend_cap_usd=4.00,
        prices={"gemini-2.5-flash": (0.30, 2.50), "gemini-3.1-flash-lite": (0.25, 1.50)},
    )


def synthetic_data() -> tuple[pd.DataFrame, dict[str, tuple[str, str, str]]]:
    """(complaints with CFPB labels, true path by complaint_id). Deterministic."""
    rng = random.Random(SEED)
    paths: list[tuple[str, str, str]] = []
    issue_no = 0
    for p, n_issues in enumerate(ISSUES_PER_PRODUCT):
        for _ in range(n_issues):
            subs = (
                [f"Sub-issue {issue_no}a", f"Sub-issue {issue_no}b"]
                if issue_no < N_TWO_SUB_ISSUES
                else ["(none)"]
            )
            paths += [(PRODUCTS[p], f"Issue {issue_no}", s) for s in subs]
            issue_no += 1
    assert len(paths) == 40 and len({q[1] for q in paths}) == 25
    kw = {q: (f"prodkw{PRODUCTS.index(q[0])}", f"issuekw{q[1][6:]}", f"subkw{i}")
          for i, q in enumerate(paths)}  # fmt: skip
    vocab = [f"word{i}" for i in range(300)]
    weights = [0.3 + rng.random() for _ in paths]
    rows, truth = [], {}
    for n in range(N_COMPLAINTS):
        true = paths[n] if n < len(paths) else rng.choices(paths, weights)[0]
        words = list(kw[true]) * 3 + rng.choices(vocab, k=25)
        rng.shuffle(words)
        label = true
        if rng.random() < 0.08:  # CFPB label noise
            pool = [q for q in paths if q != true and (rng.random() < 0.5) == (q[0] == true[0])]
            label = rng.choice(pool or [q for q in paths if q != true])
        cid = f"c{n:05d}"
        truth[cid] = true
        rows.append(
            (cid, f"2026-01-{n % 28 + 1:02d}", *label, " ".join(words), f"Co{n % 17}", "TX")
        )
    cols = ["complaint_id", "date_received", "product", "issue", "sub_issue",
            "narrative", "company", "state"]  # fmt: skip
    assert cols == COMPLAINTS_COLS
    return pd.DataFrame(rows, columns=cols), truth


def make_fake_client(error_rate: float = 0.10) -> FakeClient:
    """FakeClient whose responder plays classifier (90% correct), JEB judge, summary judge."""
    _, truth = synthetic_data()
    all_paths = sorted(set(truth.values()))
    seen_count: dict[str, int] = {}  # per-complaint classification count -> varied reruns

    def classify_one(cid: str) -> dict:
        n = seen_count.get(cid, 0)
        seen_count[cid] = n + 1
        r = random.Random(f"{cid}|cls|{n}")
        true = truth[cid]
        if r.random() >= error_rate:
            pick, conf = true, r.uniform(0.7, 0.99)
        else:
            same = [q for q in all_paths if q != true and q[0] == true[0]]
            other = [q for q in all_paths if q != true]
            pick = r.choice(same if same and r.random() < 0.5 else other)
            conf = r.uniform(0.3, 0.85)
        return {
            "complaint_id": cid, "product": pick[0], "issue": pick[1], "sub_issue": pick[2],
            "summary": f"Consumer reports a problem with {pick[0].lower()} ({pick[1]}).",
            "confidence": round(conf, 3),
        }  # fmt: skip

    def jeb_one(cid: str, block: str) -> dict:
        r = random.Random(f"{cid}|jeb")
        m = _CHOSEN.search(block)
        chosen = [g.strip() for g in m.groups()] if m else ["", "", ""]
        flags = [c == t for c, t in zip(chosen, truth[cid])]
        flags = [f != (r.random() < 0.08) for f in flags]  # judge is right ~92% of the time
        return dict(zip(["complaint_id", "product_ok", "issue_ok", "sub_issue_ok"], [cid, *flags]))

    def summary_one(cid: str) -> dict:
        r = random.Random(f"{cid}|sj")
        return {
            "complaint_id": cid,
            "faithful": r.random() < 0.9,
            "covers_main_issue": r.random() < 0.9,
            "invented_details": r.random() < 0.1,
        }

    def responder(model, system, user, schema, thinking) -> LLMResponse:
        props = schema["items"]["properties"]
        blocks = user.split("\n\n---\n\n")
        items = []
        for b in blocks:
            m = _REQ_ID.search(b)
            if not m:
                continue
            cid = m.group(1)
            if "product_ok" in props:
                items.append(jeb_one(cid, b))
            elif "faithful" in props:
                items.append(summary_one(cid))
            else:
                items.append(classify_one(cid))
        return LLMResponse(
            json.dumps(items),
            input_tokens=(len(system) + len(user)) // 4,
            output_tokens=60 * len(items) + 100,  # candidates + thinking
            logprobs=None,
        )

    return FakeClient(responder)


def simulate_human_review(
    selection: pd.DataFrame,
    gemini: pd.DataFrame,
    complaints: pd.DataFrame,
    truth: dict[str, tuple[str, str, str]],
) -> pd.DataFrame:
    """Long human_review frame from synthetic ground truth, plus a few `ambiguous` cells."""
    g = gemini.drop_duplicates("complaint_id").set_index("complaint_id")
    c = complaints.drop_duplicates("complaint_id").set_index("complaint_id")
    rng = random.Random(SEED)
    rows = []
    for cid in selection["complaint_id"]:
        summary_ok = "yes" if rng.random() < 0.9 else "no"
        for k, lv in enumerate(LEVELS3):
            g_ok, c_ok = g.at[cid, lv] == truth[cid][k], c.at[cid, lv] == truth[cid][k]
            if rng.random() < 0.04:
                verdict = "ambiguous"
            elif g_ok:
                verdict = "both OK" if c_ok else "model right"
            else:
                verdict = "label right" if c_ok else "both wrong"
            rows.append((cid, lv, verdict, summary_ok))
    return pd.DataFrame(rows, columns=HUMAN_REVIEW_COLS)


def _human_labels(human_review: pd.DataFrame) -> pd.DataFrame:
    """complaint_id, correct: Gemini right at all three levels; ambiguous rows dropped."""
    out = []
    for cid, g in human_review.groupby("complaint_id", sort=True):
        oks = [is_correct(v) for v in g["verdict"]]
        if None not in oks:
            out.append((cid, all(oks)))
    return pd.DataFrame(out, columns=["complaint_id", "correct"])


def run_smoke(artifacts_dir: Path, client: GeminiClient | None = None) -> dict:
    """Run every stage on synthetic data; resumable. Returns go/no-go plus board shapes."""
    cfg = smoke_config(Path(artifacts_dir))
    art = Path(cfg.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    client = client or make_fake_client()
    ledger = Ledger(art / "ledger.jsonl", cfg.spend_cap_usd, cfg.prices)
    gem_cache = store.JsonlCache(art / "gemini_cache.jsonl")
    judge_cache = store.JsonlCache(art / "judge_cache.jsonl")
    limiter = RateLimiter(cfg.rpm)

    def S(name, fn):
        return store.stage(cfg, name, fn)

    data, truth = synthetic_data()
    complaints = S("complaints", lambda: data)
    taxonomy = S("taxonomy", lambda: build_taxonomy(complaints))
    splits = S("splits", lambda: make_splits(complaints, cfg))

    def rows_of(mask) -> pd.DataFrame:
        ids = splits.loc[mask, "complaint_id"]
        return complaints[complaints["complaint_id"].isin(ids)].reset_index(drop=True)

    full = complaints.assign(
        path_id=[path_id(*t) for t in zip(*(complaints[c] for c in LEVELS3))]
    )
    train_df = full[full["complaint_id"].isin(splits.loc[splits.split == "train", "complaint_id"])]
    test_df = full[full["complaint_id"].isin(splits.loc[splits.split == "test", "complaint_id"])]
    test_df = test_df.reset_index(drop=True)
    test_ids = test_df["complaint_id"].tolist()
    sc_rows = rows_of(splits["self_consistency"])

    # --- classic classifiers (fit/predict once, shared by the pred and probs stages) ---
    encoder = FakeEncoder()
    makers = {
        "tfidf": lambda: TfidfLR(max_features=5000, ngram_range=(1, 1)),
        "embed_lr": lambda: EmbedLR("fake", encoder=encoder, cache={}),
        "knn": lambda: EmbedKNN("fake", encoder=encoder, cache={}),
    }
    preds: dict[str, pd.DataFrame] = {}
    probs: dict[str, pd.DataFrame] = {}
    for name, make in makers.items():
        memo: dict = {}

        def compute(make=make, memo=memo):
            if not memo:
                clf = fit_or_load(cfg, make(), train_df)
                memo["model_id"] = clf.model_id
                memo["classes"], memo["p"] = clf.predict_proba(test_df["narrative"].tolist())
            return memo

        preds[name] = S(
            f"pred_{name}",
            lambda name=name, compute=compute: to_predictions(
                name, compute()["model_id"], test_ids, compute()["classes"], compute()["p"]
            ),
        )
        probs[name] = S(
            f"probs_{name}",
            lambda compute=compute: probs_long(test_ids, compute()["classes"], compute()["p"]),
        )

    # --- Gemini classification + self-consistency runs ---
    def gemini(name: str, rows: pd.DataFrame) -> pd.DataFrame:
        return run_classification(
            rows, cfg, client, ledger, gem_cache, name, taxonomy=taxonomy, limiter=limiter
        )

    gem = S("pred_gemini", lambda: gemini("gemini", test_df))
    sc2 = S("pred_gemini_sc2", lambda: gemini("gemini_sc2", sc_rows))
    sc3 = S("pred_gemini_sc3", lambda: gemini("gemini_sc3", sc_rows))

    # --- hand-check selection (test only, from Gemini vs CFPB labels) ---
    selection = S(
        "hand_check", lambda: select_hand_check(gem, complaints, cfg.hand_check_n, cfg.seed)
    )

    # --- validators ---
    def judged() -> pd.DataFrame:
        rows = gem.merge(test_df[["complaint_id", "narrative"]], on="complaint_id")
        return judge_classifications(rows, cfg, client, ledger, judge_cache, limiter=limiter)

    jeb = S("validator_jeb", judged)

    def summary_judged() -> pd.DataFrame:
        hc = gem[gem["complaint_id"].isin(selection["complaint_id"])]
        rows = hc.merge(test_df[["complaint_id", "narrative"]], on="complaint_id")
        return judge_summaries(rows, cfg, client, ledger, judge_cache, limiter=limiter)

    summary_flags = S("summary_judge", summary_judged)

    def cleanlab_rows() -> pd.DataFrame:
        classes, oof = oof_proba(
            lambda: TfidfLR(max_features=5000, ngram_range=(1, 1)),
            train_df["narrative"].tolist(), train_df["path_id"].tolist(),
            n_splits=3, seed=cfg.seed,
        )  # fmt: skip
        wide = probs["tfidf"].pivot(index="complaint_id", columns="path_id", values="prob")
        tp = wide.reindex(index=test_ids, columns=classes).fillna(0.0).to_numpy()
        g = gem.drop_duplicates("complaint_id").set_index("complaint_id").loc[test_ids]
        return v_cleanlab(
            [path_id(*t) for t in zip(g["product"], g["issue"], g["sub_issue"])],
            classes, oof, train_df["path_id"].tolist(),
            test_ids, test_df["path_id"].tolist(), tp,
        )  # fmt: skip

    def all_validators() -> pd.DataFrame:
        parts = [
            v_label_match(gem, complaints),
            *(v_second_opinion(gem, probs[n], n) for n in makers),
            v_self_confidence(gem),
            v_self_consistency([gem[gem["complaint_id"].isin(sc2["complaint_id"])], sc2, sc3]),
            v_hierarchy(gem, taxonomy),
            cleanlab_rows(),
            jeb,
        ]
        return pd.concat(parts, ignore_index=True)[VALIDATOR_RESULTS_COLS]

    vres = S("validator_results", all_validators)

    # --- simulated human verdicts, ensembles, row status ---
    human = S("human_review", lambda: simulate_human_review(selection, gem, complaints, truth))
    ens_simple = S("ensemble_simple", lambda: ensemble_simple(vres))
    ens_learned = S("ensemble_learned", lambda: ensemble_learned(vres, _human_labels(human)))
    status = S("row_status", lambda: row_status(gem, complaints, ens_learned, cfg=cfg))

    # --- leaderboards and decision ---
    all_preds = pd.concat([gem, *preds.values()], ignore_index=True)
    clf_board = S("classifier_leaderboard", lambda: classifier_leaderboard(
        all_preds, complaints, test_ids))  # fmt: skip
    agree = agree_path(gem, complaints)
    adjusted = S("adjusted_accuracy", lambda: adjusted_accuracy(
        human, selection, float(agree.mean()), n_boot=200, seed=cfg.seed))  # fmt: skip
    strata = pd.Series(
        np.where(agree.reindex(test_ids).to_numpy(dtype=bool), "agree", "disagree"),
        index=test_ids,
    )
    weights = inverse_weights(strata[selection["complaint_id"]], strata)
    weights.index = selection["complaint_id"]
    all_vres = pd.concat(
        [vres, ens_simple[VALIDATOR_RESULTS_COLS], ens_learned[VALIDATOR_RESULTS_COLS]],
        ignore_index=True,
    )
    val_board = S("validator_leaderboard", lambda: validator_leaderboard(
        all_vres, human, weights, {}))  # fmt: skip
    decision = go_no_go(clf_board, adjusted)

    return {
        **decision,
        "classifiers": sorted(clf_board["classifier"].unique()),
        "n_classifiers": int(clf_board["classifier"].nunique()),
        "n_levels": int(clf_board["level"].nunique()),
        "classifier_board_shape": tuple(clf_board.shape),
        "validator_board_shape": tuple(val_board.shape),
        "n_validators": int(val_board["validator"].nunique()),
        "adjusted": adjusted.set_index("level")["value"].to_dict(),
        "status_counts": status["status"].value_counts().to_dict(),
        "summary_ok_share": float(summary_flags["faithful"].mean()),
        "spend_usd": ledger.spent(),
    }
