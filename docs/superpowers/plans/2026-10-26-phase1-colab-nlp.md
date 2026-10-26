# Phase 1 — Colab NLP Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tested `nlp/` Python package plus thin Colab notebooks that classify a stratified CFPB sample with Gemini and four non-LLM classifiers, run eleven validators, collect Marco's hand-check verdicts, and produce classifier + validator leaderboards with a go/no-go call — resumable at every step and hard-capped at $4 of Gemini spend.

**Architecture:** All logic lives in the installable `nlp/` package (pure functions over pandas DataFrames with fixed column schemas). Every pipeline stage writes Parquet to an artifacts directory (Google Drive in Colab, a temp dir in tests) and is skipped on rerun if its output exists. Gemini calls go through one client wrapper that enforces rate limit, response cache, and spend ledger. A synthetic end-to-end smoke run with a fake Gemini client runs in CI.

**Tech Stack:** Python 3.11, pandas + pyarrow, duckdb, scikit-learn, sentence-transformers, transformers + torch (ModernBERT), cleanlab, google-genai SDK, gspread (+ google-auth in Colab), pytest, ruff, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-26-cfpb-nlp-classifier-design.md`

## Global Constraints

- Hierarchy: `group = Product`, `subgroup = Issue`, `detail = Sub-issue`; missing Sub-issue → the literal `"(none)"`, a valid value.
- Source: HF mirror `Bob-68/cfpb-consumer-complaints` via DuckDB; noise filters exactly as spec "Noise filters" (window 2024-01-01→2026-07-06, templates ≥5 copies dropped, <200 chars or >30% X dropped, sub-issue rule at 90%, paths <30 rows dropped).
- Split targets: train pool ≤ 20,000 (no Gemini); tuning ~200; frozen test ~800 (floor 400); hand-check ~200 drawn from test; self-consistency ~150 drawn from test. Fixed seed `20261026`.
- Test set never used for prompt tuning.
- Gemini spend hard cap **$4.00** total for Phase 1, enforced in code from API usage metadata × configured prices; output price includes thinking tokens.
- Verified prices (2026-10-26, USD per 1M tokens, input/output): `gemini-3.8-flash` 0.75/3.75 (through 2026-12-31); `gemini-2.5-flash` 0.30/2.50; `gemini-3.5-flash-lite` 0.30/2.50; `gemini-3.1-flash-lite` 0.25/1.50; `gemini-2.5-flash-lite` 0.10/0.40. Store in config, not code.
- JEB judge model must differ from the classifier model.
- Every prediction row carries `model_id` and `prompt_version`.
- Every stage resumable; Gemini responses appended to cache as each request completes.
- No API keys, raw data, Parquet, or model files in git (`.gitignore` already covers these). Repo is public.
- All logic in `nlp/`; notebooks only call package functions.
- Hand-check verdict values exactly: `model right`, `label right`, `both OK`, `both wrong`, `ambiguous`; summary column `yes`/`no`. Counting: `model right`/`both OK` = correct; `label right`/`both wrong` = error; `ambiguous` excluded and counted.
- Go/no-go: reweighted adjusted Product accuracy ≥ 0.85 (point estimate) AND Gemini > TF-IDF on accuracy and macro-F1 at every level vs CFPB labels with McNemar p < 0.05 on accuracy.
- CI must stay green; tests needing GPU/network/API are marked `@pytest.mark.slow` and skipped in CI.

## Review Focus

1. **Gemini returns a category that is a near-miss string** (different case, trailing space, curly quote) — expect normalization to an exact taxonomy value when unambiguous, otherwise rejection; never a silently invalid path. → Task 6 test `test_parse_normalizes_case_and_whitespace`.
2. **Batch response with missing, duplicated, or unknown complaint IDs** — expect matched rows saved, missing rows re-queued, unknown IDs ignored. → Task 6 test `test_parse_handles_missing_duplicate_unknown_ids`.
3. **Colab dies between a paid response and the cache write** — expect the ledger never under-counts spend (charge recorded before or atomically with the cache append) so the cap still holds after restart. → Task 5 test `test_ledger_persists_before_cache_append`.
4. **Very long narratives / redaction markers (`XXXX`)** — expect truncation to a configured max chars before prompting and tokenizing, with no crash and the same truncation for every classifier. → Task 2 test `test_clean_text_truncates_and_keeps_redactions`.
5. **Rare categories (1–2 train examples) and test paths unseen in train** — expect stratifier, kNN (k > class size), and LR not to crash; taxonomy built from the full filtered window, not the train pool. → Task 3 test `test_sampling_rare_classes`, Task 7 test `test_classifiers_handle_rare_and_unseen_paths`.

---

## File Structure

```
pyproject.toml                    # package + deps + ruff + pytest config
nlp/__init__.py
nlp/config.py                     # Config dataclass: paths, sizes, seed, models, prices, cap
nlp/schema.py                     # column-name constants for every table
nlp/store.py                      # artifacts dir, stage skip-if-exists, Parquet I/O, JSONL cache
nlp/data.py                       # download + load + clean CFPB narratives
nlp/taxonomy.py                   # build taxonomy, validate/normalize paths, prompt text
nlp/sheets.py                     # Google Sheets publish/read (taxonomy + hand-check)
nlp/sampling.py                   # window choice, stratified sample, splits
nlp/budget.py                     # spend ledger, cap, cost estimate, rate limiter
nlp/llm/client.py                 # GeminiClient protocol + real google-genai impl + FakeClient
nlp/llm/classify.py               # prompt, schema, parse/validate, resumable batch runner
nlp/llm/judge.py                  # JEB binary judge + summary judge
nlp/classifiers/base.py           # PathClassifier protocol + label encoding of paths
nlp/classifiers/tfidf.py
nlp/classifiers/embed.py          # sentence-transformers embeddings + LR, and kNN
nlp/classifiers/modernbert.py
nlp/validators.py                 # all non-LLM validators → long-format results
nlp/ensemble.py                   # simple + learned ensemble, row status
nlp/review.py                     # hand-check selection, sheet rows, verdict counting
nlp/metrics.py                    # classifier metrics, bootstrap CI, McNemar, reweighting
nlp/scoring.py                    # validator leaderboard, go/no-go
nlp/smoke.py                      # synthetic end-to-end run with FakeClient
notebooks/00_setup.ipynb ... 05_results.ipynb
tests/... (one test file per module)
```

## Table schemas (`nlp/schema.py`, used everywhere)

| Table | Columns |
|---|---|
| `complaints` | `complaint_id:str, date_received:date, product, issue, sub_issue, narrative, company, state` |
| `taxonomy` | `product, issue, sub_issue, path_id:str ("product \|\| issue \|\| sub_issue")` |
| `splits` | `complaint_id, split ∈ {train, tuning, test}, hand_check:bool, self_consistency:bool` |
| `predictions` | `complaint_id, classifier, model_id, prompt_version, product, issue, sub_issue, summary (nullable), confidence:float, taxonomy_hash, run_ts` |
| `validator_results` | `complaint_id, validator, level ∈ {product, issue, sub_issue, path}, verdict:bool (Gemini judged right), score:float (higher = more likely right)` |
| `human_review` | `complaint_id, level, verdict (one of the five values), summary_ok (yes/no, path level only)` |
| `metrics` | `table, classifier_or_validator, level, metric, value, ci_low, ci_high` |

---

### Task 1: Package skeleton, config, schema, artifact store

**Files:**
- Create: `pyproject.toml`, `nlp/__init__.py`, `nlp/config.py`, `nlp/schema.py`, `nlp/store.py`
- Test: `tests/test_store.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `Config` (frozen dataclass) with fields `artifacts_dir: Path`, `seed: int = 20261026`, `train_max: int = 20000`, `tuning_n: int = 200`, `test_n: int = 800`, `test_floor: int = 400`, `hand_check_n: int = 200`, `self_consistency_n: int = 150`, `max_chars: int = 6000`, `classifier_model: str`, `judge_model: str`, `prompt_version: str`, `batch_size: int = 10`, `spend_cap_usd: float = 4.00`, `prices: dict[str, tuple[float, float]]` (model → (input, output) per 1M), `rpm: int`, `rpd: int | None`. `Config.load(path: Path) -> Config` from YAML; default file `config/phase1.yaml` committed.
  - `store.stage(cfg, name: str, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame` — returns `read(name)` if `artifacts_dir/name.parquet` exists, else runs `fn`, writes atomically (temp file + rename), returns it.
  - `store.read(cfg, name) -> pd.DataFrame`, `store.write(cfg, name, df) -> None`, `store.exists(cfg, name) -> bool`.
  - `store.JsonlCache(path)` with `.get(key: str) -> dict | None`, `.put(key: str, value: dict) -> None` (append + flush + fsync per call), `.keys() -> set[str]`.
  - `schema.py` constants for every table above, e.g. `PREDICTIONS_COLS: list[str]`.

- [ ] **Step 1: Write failing tests**

```python
def test_stage_runs_once(tmp_cfg):
    calls = []
    fn = lambda: calls.append(1) or pd.DataFrame({"a": [1]})
    store.stage(tmp_cfg, "x", fn); store.stage(tmp_cfg, "x", fn)
    assert calls == [1]

def test_jsonl_cache_survives_reopen(tmp_path):
    c = store.JsonlCache(tmp_path / "c.jsonl"); c.put("k", {"v": 1})
    assert store.JsonlCache(tmp_path / "c.jsonl").get("k") == {"v": 1}

def test_jsonl_cache_ignores_truncated_last_line(tmp_path):
    p = tmp_path / "c.jsonl"; p.write_text('{"key":"a","value":{}}\n{"key":"b","val')
    assert store.JsonlCache(p).keys() == {"a"}

def test_config_loads_default():
    cfg = Config.load(Path("config/phase1.yaml"))
    assert cfg.spend_cap_usd == 4.00 and cfg.judge_model != cfg.classifier_model
```

- [ ] **Step 2: Run `pytest tests/test_store.py tests/test_config.py -v`** — expect FAIL (module not found).
- [ ] **Step 3: Implement.** `pyproject.toml`: project `cfpb-nlp`, packages `nlp*`, extras `dev = [pytest, ruff]`, `ml = [sentence-transformers, transformers, torch, cleanlab]`, `colab = [gspread, google-auth]`; core deps `pandas, pyarrow, scikit-learn, pyyaml, google-genai, scipy`. Ruff line-length 100. Pytest marker `slow`, `addopts = "-m 'not slow'"`. `config/phase1.yaml`: `classifier_model: gemini-2.5-flash` (placeholder until Task 13 bake-off), `judge_model: gemini-3.1-flash-lite`, `prompt_version: v1`, prices from Global Constraints, `rpm: 60`, `rpd: null`. Fixture `tmp_cfg` in `tests/conftest.py`.
- [ ] **Step 4: Run tests** — PASS. Run `ruff check .` — clean.
- [ ] **Step 5: Commit** on a branch, push, PR, CI green, merge (admin bypass): `feat: package skeleton, config, artifact store`.

---

### Task 2: CFPB data load + noise filters (DuckDB over the Hugging Face mirror)

The official CFPB files no longer carry narratives (removed 2026-08-14). Source = HF mirror `Bob-68/cfpb-consumer-complaints`. Columns (verified): `date_received` (timestamp UTC), `product`, `sub_product`, `issue`, `sub_issue`, `consumer_complaint_narrative`, `company_public_response`, `company`, `state`, `zip_code`, `tags`, `submitted_via`, `date_sent_to_company`, `company_response_to_consumer`, `timely_response`, `complaint_id` (string).

**Files:**
- Create: `nlp/data.py`
- Modify: `pyproject.toml` (add core dep `duckdb`)
- Test: `tests/test_data.py` (fixture Parquet built inside the test with ~40 hand-made rows exercising every filter)

**Interfaces:**
- Produces:
  - `HF_SOURCE = "hf://datasets/Bob-68/cfpb-consumer-complaints/data/*.parquet"`
  - `FilterParams` dataclass: `start="2024-01-01"`, `end="2026-07-07"` (exclusive), `template_min_copies=5`, `min_chars=200`, `max_x_frac=0.30`, `sub_issue_required_frac=0.90`, `min_path_rows=30`.
  - `load_clean(source: str, params: FilterParams) -> tuple[pd.DataFrame, pd.DataFrame]` → (`complaints` schema frame, `filter_report` frame `step, rows_before, rows_removed, rows_after`). One DuckDB query pipeline, steps in this exact order: (1) null narrative/product/issue; (2) date window; (3) normalized text = lower + whitespace-collapsed; group count ≥ `template_min_copies` → drop all, else keep the row with earliest `date_received` then lowest `complaint_id`; (4) `length < min_chars` or X-fraction `> max_x_frac`; (5) per issue, share of rows with non-null sub_issue ≥ `sub_issue_required_frac` → drop that issue's null-sub_issue rows; remaining nulls → `"(none)"`; (6) drop paths with `< min_path_rows`. Report counts each step. Narrative renamed to `narrative`; `date_received` → date.
  - `clean_text(s: str, max_chars: int) -> str` — collapse whitespace, keep `XXXX` redactions, truncate to `max_chars`.
  - `product_counts_by_month(source: str, params: FilterParams) -> pd.DataFrame` — `month, product, issue, n` over rows with narrative (Task 3's `choose_window` input).

- [ ] **Step 1: Write failing tests** — `test_filter_nulls_and_window`, `test_mass_template_dropped_entirely` (5 copies → 0 kept), `test_small_duplicate_keeps_earliest` (3 copies → 1 kept, earliest date), `test_short_and_redacted_dropped` (199 chars dropped, 200 kept; 31% X dropped), `test_null_sub_issue_rules` (issue with 95% sub_issue → null row dropped; issue with 0% → `"(none)"`), `test_rare_paths_dropped` (with `min_path_rows=3` in test params), `test_filter_report_counts_add_up`, `test_clean_text_truncates_and_keeps_redactions` (`clean_text("a  XXXX\n\nb"*1000, 50)` has len ≤ 50, contains `"XXXX"`, no double spaces). Tests pass a local Parquet path as `source`. One `slow` test reads 1 row from `HF_SOURCE`.
- [ ] **Step 2: Run** `pytest tests/test_data.py -v` — FAIL.
- [ ] **Step 3: Implement** per interfaces.
- [ ] **Step 4: Run** — PASS; `ruff check .` clean.
- [ ] **Step 5: Commit/PR/merge** `feat: CFPB data loading`.

---

### Task 3: Taxonomy, window choice, stratified sample, splits

**Files:**
- Create: `nlp/taxonomy.py`, `nlp/sampling.py`
- Test: `tests/test_taxonomy.py`, `tests/test_sampling.py`

**Interfaces:**
- Consumes: `complaints` DataFrame (Task 2), `Config`.
- Produces:
  - `build_taxonomy(complaints) -> pd.DataFrame` (`taxonomy` schema, unique paths, sorted).
  - `path_id(product, issue, sub_issue) -> str` joined by `" || "`.
  - `normalize_path(taxonomy, product, issue, sub_issue) -> tuple[str, str, str] | None` — exact match after casefold + whitespace collapse + straight quotes; `None` if no unique match.
  - `taxonomy_prompt_text(taxonomy) -> str` — compact nested listing (Product → Issue → Sub-issues) for the prompt.
  - `choose_window(counts_by_month, months: int = 12) -> tuple[date, date]` — latest `months`-long window ending at the last full month in which the set of `(product, issue)` pairs is identical month to month for ≥ 90% of volume (algorithm: walk back from latest month, stop at first month whose pair set differs on >10% of volume).
  - `stratified_sample(complaints, n, by: str, seed, min_per_class: int) -> pd.DataFrame` — proportional by `by`, each class gets `min(min_per_class, available)` first.
  - `make_splits(complaints, cfg) -> pd.DataFrame` (`splits` schema): draw tuning (stratified by product, `min_per_class=3`), test (stratified by product, `min_per_class=10`), train from the remainder (stratified by `path_id`, cap `train_max`); `self_consistency` = seeded stratified subset of test. `hand_check` set to False here — chosen later in Task 10 after Gemini runs.

- [ ] **Step 1: Write failing tests** — `test_normalize_parse_variants` (`" credit REPORTING… "` with curly apostrophe → exact taxonomy tuple; ambiguous → None), `test_splits_disjoint_and_deterministic` (same seed → identical frame; tuning∩test∩train empty), `test_splits_sizes` (on synthetic 30k rows: tuning=200, test=800, train ≤ 20000), `test_sampling_rare_classes` (a class with 2 rows doesn't raise and appears in at most the available count), `test_choose_window_stops_at_taxonomy_change` (synthetic counts with a change 8 months back → window starts after it).
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit/PR/merge** `feat: taxonomy and splits`.

---

### Task 4: Google Sheets I/O

**Files:**
- Create: `nlp/sheets.py`
- Test: `tests/test_sheets.py` (pure functions only; gspread calls are `slow`)

**Interfaces:**
- Produces:
  - `publish_taxonomy(gc, taxonomy, title="CFPB taxonomy (phase1)") -> str` (sheet URL); `read_taxonomy(gc, url) -> pd.DataFrame` (validated against `taxonomy` schema; duplicates/blank product or issue → `ValueError`).
  - `taxonomy_snapshot_hash(taxonomy) -> str` (sha256 of sorted path_ids, first 12 chars) — recorded on every Gemini run.
  - `write_review_sheet(gc, rows: pd.DataFrame, title) -> str` with data-validation dropdowns on verdict columns (five values) and `summary_ok` (yes/no); `read_review_sheet(gc, url) -> pd.DataFrame` (`human_review` schema, long format).
- [ ] **Step 1: Failing tests** — `test_read_taxonomy_rejects_duplicates` (via a fake worksheet object returning records), `test_snapshot_hash_order_independent`, `test_review_sheet_roundtrip_with_fake_gc` (fake gc stores values in memory; read returns the same verdicts, long format, 3 levels per row).
- [ ] **Steps 2–5:** run FAIL → implement (gspread `worksheet.update`, `set_data_validation_for_cell_range` or batch_update with `ONE_OF_LIST`) → PASS → commit/PR/merge `feat: sheets io`.

---

### Task 5: Budget ledger, cost estimate, rate limiter

**Files:**
- Create: `nlp/budget.py`
- Test: `tests/test_budget.py`

**Interfaces:**
- Produces:
  - `class BudgetExceeded(Exception)`, `class DailyQuotaExhausted(Exception)` (carries `remaining_rows: int`).
  - `Ledger(path: Path, cap_usd: float, prices: dict)`: `.spent() -> float`, `.cost(model, in_tok, out_tok) -> float` (out_tok includes thinking), `.reserve(model, est_in, est_out) -> None` raises `BudgetExceeded` if `spent + est > cap`, `.record(model, request_id, in_tok, out_tok) -> float` appends a JSONL line with fsync and returns spend. Unknown model → `KeyError` (never price as zero).
  - `estimate_run(n_requests, avg_in, avg_out, model, prices) -> float`.
  - `RateLimiter(rpm: int, clock=time.monotonic, sleep=time.sleep)` with `.wait() -> None`.
- [ ] **Step 1: Failing tests** — `test_cap_blocks_request_that_would_cross` (cap 4.00, spent 3.99, reserve est 0.02 → raises), `test_ledger_persists_across_instances`, `test_cost_math` (`gemini-2.5-flash`, 1,000,000 in / 1,000,000 out → 2.80), `test_unknown_model_raises`, `test_ledger_persists_before_cache_append` (simulate crash: `record` succeeds, cache put raises → new Ledger instance still reports the spend), `test_rate_limiter_spaces_calls` (fake clock: 3 calls at rpm=60 → sleeps total ≥ 2s).
- [ ] **Steps 2–5:** FAIL → implement → PASS → commit/PR/merge `feat: spend ledger and rate limiter`.

---

### Task 6: Gemini client + classification runner

**Files:**
- Create: `nlp/llm/__init__.py`, `nlp/llm/client.py`, `nlp/llm/classify.py`
- Test: `tests/test_llm_classify.py`

**Interfaces:**
- Consumes: `Ledger`, `RateLimiter`, `JsonlCache`, `normalize_path`, `taxonomy_prompt_text`, `clean_text`.
- Produces:
  - `LLMResponse` dataclass: `text: str, input_tokens: int, output_tokens: int (incl. thinking), logprobs: list | None`.
  - `GeminiClient` protocol: `generate(model: str, system: str, user: str, response_schema: dict, thinking: str) -> LLMResponse`; raises `RateLimited(retry_after: float | None, daily: bool)`.
  - `GenAIClient(api_key)` real impl (google-genai SDK; structured output via `response_mime_type="application/json"` + schema; maps 429 RESOURCE_EXHAUSTED to `RateLimited`, `daily=True` when the error's quota id contains `PerDay`; logs the quota id/value the first time). `FakeClient(responder: Callable)` for tests.
  - `build_prompt(batch: pd.DataFrame, taxonomy_text: str, prompt_version: str) -> tuple[str, str]` (system, user). Asks for per complaint: `complaint_id, product, issue, sub_issue, summary (≤ 60 words, only facts in the text), confidence (0–1)`.
  - `RESPONSE_SCHEMA: dict` (array of those objects).
  - `parse_and_validate(raw: str, batch_ids: list[str], taxonomy) -> tuple[pd.DataFrame, list[str]]` → (valid predictions in `predictions` schema minus run metadata, ids to retry). Invalid path, missing id → retry list; duplicate id → first kept; unknown id → ignored.
  - `run_classification(rows, cfg, client, ledger, cache, classifier_name="gemini", model=None, prompt_version=None, thinking="low") -> pd.DataFrame` — batches of `cfg.batch_size`, skips rows whose cache key `f"{cid}|{model}|{prompt_version}"` exists, per request: `ledger.reserve` (estimate from prompt length/4 + 150 out tokens per row) → limiter.wait → generate → `ledger.record` → cache.put per row → retry list re-queued (max 2 retries per row, then left absent and counted). Backoff on non-daily `RateLimited`; on daily → raise `DailyQuotaExhausted(remaining_rows)` after state is saved. On `BudgetExceeded` → stop, return what's done, print remaining count. Every prediction records `taxonomy_hash` (Task 4 `taxonomy_snapshot_hash`). Returns all cached predictions for `rows`.
- [ ] **Step 1: Failing tests (FakeClient)** — `test_parse_normalizes_case_and_whitespace`, `test_parse_handles_missing_duplicate_unknown_ids`, `test_parse_rejects_invalid_path`, `test_cached_rows_not_resent_or_charged` (second run → zero generate calls, ledger unchanged), `test_budget_exhaustion_stops_cleanly` (cap reached mid-run → returns partial, no exception, remaining reported), `test_daily_quota_raises_with_remaining`, `test_every_prediction_has_model_and_prompt_version`.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — PASS. Plus one `slow` test `test_real_gemini_one_batch` (skipped without `GEMINI_API_KEY`) that classifies 2 fixture rows with `gemini-2.5-flash-lite` and asserts a valid path.
- [ ] **Step 5: Commit/PR/merge** `feat: gemini classification runner`.

---

### Task 7: Non-LLM classifiers (TF-IDF, sentence-transformer LR, kNN, ModernBERT)

**Files:**
- Create: `nlp/classifiers/__init__.py`, `base.py`, `tfidf.py`, `embed.py`, `modernbert.py`
- Test: `tests/test_classifiers.py`

**Interfaces:**
- Produces:
  - `PathClassifier` protocol: `name: str`, `fit(texts: list[str], path_ids: list[str]) -> None`, `predict_proba(texts) -> tuple[list[str], np.ndarray]` (classes, probs n×k). Predicting full `path_id` as one label guarantees valid paths (builder's choice from spec, taken here).
  - `to_predictions(classifier_name, model_id, complaint_ids, classes, probs) -> pd.DataFrame` — argmax → split path_id into product/issue/sub_issue; `confidence` = max prob; `summary` = None; `prompt_version` = `"n/a"`.
  - `TfidfLR(max_features=100_000, ngram_range=(1,2), C=4.0)`; `EmbedLR(model_name)` and `EmbedKNN(model_name, k=15)` sharing an embedding cache (`embed_texts(model_name, texts, cache) -> np.ndarray`, normalized); `ModernBertClassifier(model_name="answerdotai/ModernBERT-base", max_len=512, epochs=2, lr=5e-5, batch_size=16)`.
  - Embedding model for Phase 1: `BAAI/bge-base-en-v1.5` (my call: strong, 110M params, fast on T4; swap via config `embed_model`).
  - `fit_or_load(cfg, clf, train_df) -> PathClassifier` stores fitted models under `artifacts_dir/models/` (joblib; ModernBERT `save_pretrained`).
- [ ] **Step 1: Failing tests** — on a 3-class toy corpus: `test_tfidf_learns_toy` (accuracy 1.0 on train), `test_knn_k_larger_than_class` (class with 2 examples, k=15 → no error), `test_classifiers_handle_rare_and_unseen_paths` (predictions only from fitted classes; output paths valid), `test_to_predictions_schema`. Embedding/ModernBERT tests use a `FakeEncoder` for CI; real-model tests marked `slow`.
- [ ] **Steps 2–5:** FAIL → implement → PASS → commit/PR/merge `feat: non-LLM classifiers`.

---

### Task 8: Metrics

**Files:**
- Create: `nlp/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `level_scores(pred, truth) -> pd.DataFrame` — per level (product, issue, sub_issue, path): accuracy, macro_f1, with bootstrap 95% CI (`n_boot=1000`, seeded).
  - `per_class_report(pred, truth, level) -> pd.DataFrame`, `confusion(pred, truth, level) -> pd.DataFrame`.
  - `mcnemar(correct_a: np.ndarray[bool], correct_b) -> tuple[float, float]` (statistic, p; exact binomial when discordant < 25).
  - `inverse_weights(sample_strata: pd.Series, population_strata: pd.Series) -> pd.Series`.
  - `weighted_accuracy(correct: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]` (value, ci_low, ci_high; weighted bootstrap).
- [ ] **Step 1: Failing tests** — `test_accuracy_and_macro_f1_known_values` (hand-computed 6-row example), `test_mcnemar_identical_is_p1`, `test_mcnemar_known` (b=10, c=0 → p < 0.01), `test_reweighting_recovers_population` (oversampled disagreements: weighted accuracy equals true population accuracy within 1e-9).
- [ ] **Steps 2–5:** FAIL → implement (scikit-learn + scipy.stats.binomtest) → PASS → commit/PR/merge `feat: metrics`.

---

### Task 9: Validators, JEB judge, ensemble

**Files:**
- Create: `nlp/validators.py`, `nlp/llm/judge.py`, `nlp/ensemble.py`
- Test: `tests/test_validators.py`, `tests/test_judge.py`, `tests/test_ensemble.py`

**Interfaces:**
- Consumes: `predictions` (Gemini + others), `complaints`, `taxonomy`, classifier probabilities, Task 6 runner pieces.
- Produces (each returns `validator_results` rows):
  - `v_label_match(gemini, complaints)` — validator `cfpb_label`, per level.
  - `v_second_opinion(gemini, other_preds, name)` — `tfidf`, `embed_lr`, `knn`, `modernbert`; score = other classifier's probability of Gemini's path (needs `other_probs` frame `complaint_id, path_id, prob`).
  - `v_self_confidence(gemini)` — `gemini_confidence`; adds `gemini_logprob` rows only if logprobs present.
  - `v_self_consistency(runs: list[pd.DataFrame])` — `self_consistency`; score = share of 3 runs matching run 1, per level.
  - `v_hierarchy(gemini, taxonomy)` — `hierarchy`, path level.
  - `v_cleanlab(train_probs_oof, train_labels, test_labels, test_probs)` — `cleanlab`; score = `1 - label_issue_probability` of the CFPB label, verdict = Gemini agrees with CFPB OR label flagged as issue. Out-of-fold probs from TF-IDF 5-fold on train.
  - `judge_classifications(rows, cfg, client, ledger, cache, model=cfg.judge_model)` (in `judge.py`) — JEB; prompt shows complaint + Gemini's three levels; schema `{complaint_id, product_ok, issue_ok, sub_issue_ok}` booleans → validator `jeb`, score = 1.0/0.0. Raises if `model == cfg.classifier_model`.
  - `judge_summaries(rows, ...)` → `summary_judge` frame `complaint_id, faithful, covers_main_issue, invented_details` (bools).
  - `ensemble_simple(results) -> pd.DataFrame` — validator `ensemble_simple`; score = share of validators voting right; `dissent` = comma list.
  - `ensemble_learned(results, human_labels, n_splits=5) -> pd.DataFrame` — validator `ensemble_learned`; logistic regression on pivoted validator scores; scores for hand-checked rows are strictly out-of-fold; others from model fit on all hand-checked rows.
  - `row_status(gemini, complaints, ensemble) -> pd.DataFrame` — `complaint_id, status ∈ {verified, likely_correct, needs_review}`: verified = matches CFPB at path; likely_correct = disagrees but ensemble score ≥ 0.6; else needs_review. Threshold in config.
- [ ] **Step 1: Failing tests** — one per validator on hand-made 4-row frames asserting verdict/score; `test_jeb_refuses_same_model`; `test_learned_ensemble_is_out_of_fold` (a validator column equal to the label must not yield perfect in-fold-like scores: verify each hand row's score came from a model not trained on it via fold id); `test_row_status_rules`.
- [ ] **Steps 2–5:** FAIL → implement → PASS → commit/PR/merge `feat: validators, judge, ensemble`.

---

### Task 10: Hand-check selection and verdict counting

**Files:**
- Create: `nlp/review.py`
- Test: `tests/test_review.py`

**Interfaces:**
- Produces:
  - `select_hand_check(gemini_test, complaints, n=200, seed) -> pd.DataFrame` — stratified by `agree_path` (Gemini path == CFPB path) only, 50/50 agree/disagree (or all available disagreements if fewer); returns ids + `stratum`.
  - `review_rows(selection, gemini, complaints) -> pd.DataFrame` — wide sheet rows: text, CFPB 3 levels, Gemini 3 levels + summary, three verdict columns pre-filled `both OK` where Gemini == CFPB at that level, else blank; `summary_ok` blank.
  - `count_verdicts(human_review) -> pd.DataFrame` — per level: `n_correct, n_error, n_ambiguous`; `is_correct(verdict) -> bool | None` per Global Constraints.
- [ ] **Step 1: Failing tests** — `test_selection_ignores_validator_outputs` (function signature takes no validator frame; strata only agree/disagree), `test_prefill_only_where_agree`, `test_counting_rules` (each of the five values maps exactly as specified; ambiguous excluded).
- [ ] **Steps 2–5:** FAIL → implement → PASS → commit/PR/merge `feat: hand-check review`.

---

### Task 11: Leaderboards and go/no-go

**Files:**
- Create: `nlp/scoring.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `metrics.*`, `review.count_verdicts`, `validator_results`, `human_review`, ledger, run timings.
- Produces:
  - `classifier_leaderboard(preds_all, complaints, test_ids) -> pd.DataFrame` — per classifier × level: accuracy, macro_f1 with CIs, McNemar p vs Gemini.
  - `adjusted_accuracy(human_review, selection, test_agree_rate) -> pd.DataFrame` — reweighted per level with CI.
  - `validator_leaderboard(validator_results, human_review, weights, costs: dict[str, dict]) -> pd.DataFrame` — per validator × level: catch_rate, false_alarm_rate, auroc (weighted; `sklearn.metrics.roc_auc_score` with `sample_weight`), gemini_requests, usd, runtime_s; sorted by AUROC at path level.
  - `go_no_go(clf_board, adjusted) -> dict` — `{"go": bool, "reasons": [str, ...]}` applying the Global Constraints rule exactly.
- [ ] **Step 1: Failing tests** — `test_go_requires_all_conditions` (four cases: all pass → go; Product adj 0.84 → no-go; TF-IDF higher macro-F1 at issue → no-go; p = 0.2 → no-go), `test_catch_rate_and_false_alarm_known`, `test_auroc_weighted`.
- [ ] **Steps 2–5:** FAIL → implement → PASS → commit/PR/merge `feat: leaderboards and go/no-go`.

---

### Task 12: Synthetic end-to-end smoke run in CI

**Files:**
- Create: `nlp/smoke.py`, `tests/test_smoke.py`
- Modify: `.github/workflows/ci.yml` (install `.[dev]` only; smoke uses FakeClient and FakeEncoder)

**Interfaces:**
- Produces: `run_smoke(artifacts_dir: Path) -> dict` — generates 3,000 synthetic complaints over 6 products / 25 issues / 40 sub-issues with keyword-bearing text; runs every stage of the real pipeline (splits → TF-IDF → fake-encoder LR/kNN → FakeClient Gemini (90% correct, emits logprobs=None) → validators → simulated human verdicts → ensembles → leaderboards → go/no-go), using `stage()` so it is resumable; returns the go/no-go dict and leaderboard shapes.
- [ ] **Step 1: Failing test** — `test_smoke_end_to_end(tmp_path)`: returns dict with `go` key; classifier board has 4 classifiers (ModernBERT skipped in smoke) × 4 levels; validator board non-empty; ledger spend > 0 and ≤ 4.00; **second call performs zero FakeClient calls** (resume works).
- [ ] **Steps 2–5:** FAIL → implement → PASS locally and in CI → commit/PR/merge `test: end-to-end smoke run`.

---

### Task 13: Colab notebooks

**Files:**
- Create: `notebooks/00_setup.ipynb`, `01_data_and_splits.ipynb`, `02_baselines.ipynb`, `03_gemini.ipynb`, `04_validators.ipynb`, `05_results.ipynb`, `notebooks/README.md`

Each notebook starts with the same setup cell (from `00_setup`): mount Drive at `/content/drive`, `artifacts_dir = /content/drive/MyDrive/cfpb-nlp-classifier/phase1`, `git clone`/`git pull` the public repo, `pip install -e ".[ml,colab]"`, read `GEMINI_API_KEY` from `google.colab.userdata`, `google.colab.auth.authenticate_user()` + gspread client. Cells only call `nlp.*` functions wrapped in `store.stage`.

- `01`: `product_counts_by_month(HF_SOURCE)` → `choose_window` (print; if it starts later than 2024-01-01 use it as `FilterParams.start`) → `load_clean` (stage `complaints_clean` + `filter_report`, print the report) → `build_taxonomy` → `publish_taxonomy` (print URL) → `make_splits` → print split sizes and Product distribution.
- `02` (GPU runtime): TF-IDF, EmbedLR, EmbedKNN, ModernBERT fit_or_load → predictions on tuning + test → TF-IDF 5-fold OOF probs for cleanlab.
- `03`: **model bake-off** — 50 tuning rows × candidates `gemini-3.8-flash`, `gemini-2.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite` (print accuracy per level and actual $ each; total ≤ ~$0.30); Marco picks → set `classifier_model` in config (commit). **Batch size check** on tuning: 10 vs 25 per request. **Thinking level check**: lowest level that holds tuning accuracy (thinking is billed as output). **Prompt iterations** on tuning (bump `prompt_version` each). Print `estimate_run` for the test set + JEB + self-consistency and the remaining cap **before** the paid test run; then classify test.
- `04`: JEB on full test, summary judge on 100 test rows, self-consistency (2 extra runs on the 150 subset), non-LLM validators, `select_hand_check` → `write_review_sheet` (print URL). **Stop: Marco reviews.**
- `05`: `read_review_sheet` → learned ensemble → row status → both leaderboards → adjusted accuracy → `go_no_go` → write all tables to Parquet → print a results summary (markdown) and ledger total.

- [ ] **Step 1:** Write notebooks; each cell ≤ ~10 lines of package calls.
- [ ] **Step 2: Verify locally:** `jupyter nbconvert --to notebook --execute` is not possible without Drive/keys; instead verify each notebook's code cells import-resolve: `python -m nlp.tools.check_notebooks notebooks/` (small helper: parse cells, `ast.parse` each, confirm every `nlp.` attribute referenced exists). Add to CI.
- [ ] **Step 3: Commit/PR/merge** `feat: colab notebooks`.

---

### Task 14: Docs and handoff

**Files:**
- Modify: `README.md` (how to run Phase 1 in Colab, step by step for Marco and his sister), `CLAUDE.md`, `STATE.md`

- [ ] **Step 1:** README "Run Phase 1" section: GCP project + key in Colab Secrets, open `00_setup` from GitHub in Colab, run 01→05, where outputs land on Drive, what to do at the review stop, how the $4 cap and resume work.
- [ ] **Step 2:** Update `STATE.md` (status, decisions, next: Marco runs notebooks; Phase 2 design after go/no-go). Add project to `sites-hub` per Marco's convention.
- [ ] **Step 3: Commit/PR/merge** `docs: phase 1 run guide`.

---

## What happens after the build (Marco, in Colab)

Runs 01 → 04 (~1–2 hours plus model bake-off decision), reviews ~200 rows in the Sheet (~2 hours), runs 05. Results feed the Phase 2 design.
