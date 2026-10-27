"""Synthetic end-to-end smoke run: every pipeline stage, fake Gemini, fake Jev, fake embeddings.

`run_smoke(artifacts_dir)` generates 3,000 synthetic complaints (6 products / 25 issues /
40 sub-issues, keyword-bearing text) and runs the real pipeline code on them: splits ->
TF-IDF / embed-LR / kNN -> Gemini classification (+ self-consistency) -> validators (Jev) and summary judge
-> simulated human verdicts -> ensembles -> leaderboards -> go/no-go. Every stage goes through
`store.stage` and every paid call through the JSONL caches, so a rerun makes zero API calls.
ModernBERT is skipped. Needs only the `.[dev]` dependencies (no torch).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd

from nlp import pipeline as P
from nlp import store
from nlp.budget import Ledger, RateLimiter
from nlp.classifiers.embed import EmbedKNN, EmbedLR, FakeEncoder
from nlp.classifiers.tfidf import TfidfLR
from nlp.config import Config
from nlp.data import clean_text
from nlp.ensemble import row_status
from nlp.llm.client import FakeClient, GeminiClient, LLMResponse
from nlp.llm.jev import FakeJevClient, JevClient, JevResponse
from nlp.review import select_hand_check
from nlp.sampling import make_splits
from nlp.schema import COMPLAINTS_COLS, HUMAN_REVIEW_COLS
from nlp.taxonomy import build_taxonomy

N_COMPLAINTS = 3000
SEED = 20261026
LEVELS3 = P.LEVELS3
PRODUCTS = [
    "Credit reporting", "Debt collection", "Mortgage",
    "Checking account", "Credit card", "Student loan",
]  # fmt: skip
ISSUES_PER_PRODUCT = [5, 4, 4, 4, 4, 4]  # 25 issues
N_TWO_SUB_ISSUES = 15  # 15 issues x 2 sub-issues + 10 issues x "(none)" = 40 paths
_REQ_ID = re.compile(r"^complaint_id: (\S+)$", re.MULTILINE)
_QUOTED = re.compile(r'"([^"]*)"')


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
        prices={
            "gemini-2.5-flash": (0.30, 2.50),
            "gemini-3.1-flash-lite": (0.25, 1.50),
            "jev-1.14.0": (0.042, 0.0),
        },
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
    """FakeClient whose responder plays classifier (90% correct) and summary judge."""
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
                continue  # preamble block has no complaint id
            cid = m.group(1)
            if "faithful" in props:
                items.append(summary_one(cid))
            else:
                items.append(classify_one(cid))
        if not items:
            raise ValueError(f"fake responder found no complaint ids; prompt drifted: {user!r}")
        return LLMResponse(
            json.dumps(items),
            input_tokens=(len(system) + len(user)) // 4,
            output_tokens=60 * len(items) + 100,  # candidates + thinking
            logprobs=None,
        )

    return FakeClient(responder)


def make_fake_jev_client() -> FakeJevClient:
    """FakeJevClient: answers each claim from synthetic ground truth, right ~92% of the time."""
    data, truth = synthetic_data()
    by_text = {clean_text(n, 6000): c for c, n in zip(data["complaint_id"], data["narrative"])}
    assert len(by_text) == len(data), "synthetic narratives must be unique"
    index = {"product_ok": 0, "issue_ok": 1, "sub_issue_ok": 2}

    def responder(model: str, state: str, questions: dict) -> JevResponse:
        if state not in by_text:
            raise ValueError(f"fake Jev got an unknown narrative; prompt drifted: {state[:80]!r}")
        cid = by_text[state]
        nouls = {}
        for qid, q in questions.items():
            text = q["instructions"]
            if "no more specific sub-issue" in text:
                claimed = "(none)"
            else:
                claimed = _QUOTED.findall(text)[-1]
            yes = claimed == truth[cid][index[qid]]
            r = random.Random(f"{cid}|jev|{qid}")
            yes = yes != (r.random() < 0.08)  # judge is wrong ~8% of the time
            nouls[qid] = round(r.uniform(0.75, 0.98) if yes else r.uniform(0.02, 0.25), 3)
        tokens = len(json.dumps({"state": state, "questions": questions})) // 4
        return JevResponse(nouls, tokens, 0, model)

    return FakeJevClient(responder)


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


def run_smoke(
    artifacts_dir: Path,
    client: GeminiClient | None = None,
    jev_client: JevClient | None = None,
) -> dict:
    """Run every stage on synthetic data; resumable. Returns go/no-go plus board shapes."""
    cfg = smoke_config(Path(artifacts_dir))
    art = Path(cfg.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    client = client or make_fake_client()
    jev_client = jev_client or make_fake_jev_client()
    ledger = Ledger(art / "ledger.jsonl", cfg.spend_cap_usd, cfg.prices)
    gem_cache = store.JsonlCache(art / "gemini_cache.jsonl")
    judge_cache = store.JsonlCache(art / "judge_cache.jsonl")
    limiter = RateLimiter(cfg.rpm)

    def S(name, fn):
        return store.stage(cfg, name, fn)

    data, truth = synthetic_data()
    complaints = S(P.STAGE_COMPLAINTS, lambda: data)
    taxonomy = S(P.STAGE_TAXONOMY, lambda: build_taxonomy(complaints))
    splits = S(P.STAGE_SPLITS, lambda: make_splits(complaints, cfg))
    train_df, test_df = P.train_test_frames(complaints, splits)
    test_ids = test_df["complaint_id"].tolist()
    sc_rows = P.split_rows(complaints, splits, splits["self_consistency"])

    # --- classic classifiers (fit once; the pred and probs stages share the result) ---
    encoder = FakeEncoder()

    def tfidf():
        return TfidfLR(max_features=5000, ngram_range=(1, 1))

    makers = {
        "tfidf": tfidf,
        "embed_lr": lambda: EmbedLR("fake", encoder=encoder, cache={}),
        "knn": lambda: EmbedKNN("fake", encoder=encoder, cache={}),
    }
    preds: dict[str, pd.DataFrame] = {}
    probs: dict[str, pd.DataFrame] = {}
    for name, make in makers.items():
        memo: list = []

        def fitted(name=name, make=make, memo=memo):
            if not memo:
                memo.append(P.fit_predict_classifier(cfg, make(), train_df, test_df, name))
            return memo[0]

        preds[name] = S(P.stage_pred(name), lambda fitted=fitted: fitted()[0])
        probs[name] = S(P.stage_probs(name), lambda fitted=fitted: fitted()[1])

    # --- Gemini classification + self-consistency runs ---
    def gem_run(name: str, rows: pd.DataFrame) -> pd.DataFrame:
        return P.run_gemini(cfg, rows, client, ledger, gem_cache, taxonomy, name, limiter)

    gem = S(P.STAGE_PRED_GEMINI, lambda: gem_run("gemini", test_df))
    sc2 = S(P.STAGE_PRED_GEMINI_SC2, lambda: gem_run("gemini_sc2", sc_rows))
    sc3 = S(P.STAGE_PRED_GEMINI_SC3, lambda: gem_run("gemini_sc3", sc_rows))
    selection = S(
        P.STAGE_HAND_CHECK, lambda: select_hand_check(gem, complaints, cfg.hand_check_n, cfg.seed)
    )

    # --- validators ---
    jev = S(
        P.STAGE_JEV,
        lambda: P.run_jev(
            cfg, gem, test_df, taxonomy, jev_client, ledger, judge_cache, limiter
        ),
    )
    summary_flags = S(
        P.STAGE_SUMMARY_JUDGE,
        lambda: P.run_summary_judge(
            cfg, gem, test_df, selection, client, ledger, judge_cache, limiter
        ),
    )
    clean = S(
        P.STAGE_CLEANLAB,
        lambda: P.cleanlab_validator(cfg, train_df, test_df, gem, probs["tfidf"], tfidf, 3),
    )
    vres = S(
        P.STAGE_VALIDATORS,
        lambda: P.build_validator_results(complaints, taxonomy, gem, [sc2, sc3], probs, clean, jev),
    )

    # --- simulated human verdicts, ensembles, row status ---
    human = S(
        P.STAGE_HUMAN_REVIEW, lambda: simulate_human_review(selection, gem, complaints, truth)
    )
    ens: list = []

    def ensembles(i: int) -> pd.DataFrame:
        if not ens:
            weights = P.review_weights(gem, complaints, selection)
            ens.extend(P.run_ensembles(vres, human, weights=weights))
        return ens[i]

    ens_simple = S(P.STAGE_ENSEMBLE_SIMPLE, lambda: ensembles(0))
    ens_learned = S(P.STAGE_ENSEMBLE_LEARNED, lambda: ensembles(1))
    status = S(P.STAGE_ROW_STATUS, lambda: row_status(gem, complaints, ens_learned, cfg=cfg))

    # --- leaderboards and decision ---
    scored = P.run_scoring(
        cfg, pd.concat([gem, *preds.values()], ignore_index=True), complaints, test_ids, gem,
        human, selection, P.all_validator_results(vres, ens_simple, ens_learned),
        n_boot=200, stage=S, summary_flags=summary_flags,
    )  # fmt: skip
    clf_board, adjusted = scored["classifier_board"], scored["adjusted"]
    val_board = scored["validator_board"]
    jev_path = val_board[(val_board["validator"] == "jev") & (val_board["level"] == "path")]
    return {
        **scored["decision"],
        "classifiers": sorted(clf_board["classifier"].unique()),
        "n_classifiers": int(clf_board["classifier"].nunique()),
        "n_levels": int(clf_board["level"].nunique()),
        "classifier_board_shape": tuple(clf_board.shape),
        "validator_board_shape": tuple(val_board.shape),
        "n_validators": int(val_board["validator"].nunique()),
        "adjusted": adjusted.set_index("level")["value"].to_dict(),
        "gemini_path_accuracy": _board_value(clf_board, "gemini", "path", "accuracy"),
        "jev_path_catch_rate": float(jev_path["catch_rate"].iloc[0]),
        "status_counts": status["status"].value_counts().to_dict(),
        "summary_ok_share": float(summary_flags["faithful"].mean()),
        "summary_quality": scored["summary_quality"].set_index("measure")["rate"].to_dict(),
        "spend_usd": ledger.spent(),
    }


def _board_value(board: pd.DataFrame, clf: str, level: str, metric: str) -> float:
    sel = board[(board.classifier == clf) & (board.level == level) & (board.metric == metric)]
    return float(sel["value"].iloc[0])
