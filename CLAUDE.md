# cfpb-nlp-classifier — primer

**Purpose:** Rebuild (to state-of-the-art standard) a friend's Gemini-based case classifier:
complaint text + a category sheet (group/subgroup/detail) → category + summary. Marco's goal is
to become expert in NLP classification and take it to production. His sister (GitHub
`Karoltrojacek`, Write access) collaborates.

**Data:** CFPB Consumer Complaint Database (CC0). group/subgroup/detail = Product → Issue →
Sub-issue. Since v2 every category has a real sub-issue: complaints without one are dropped (filter
step `no_sub_issue`); nothing produces a `"(none)"` placeholder. The CFPB stopped publishing
narratives on 2026-08-14, so the source is the HF mirror `Bob-68/cfpb-consumer-complaints`
(narratives to 2026-07-06) read via DuckDB, with noise filters: window 2024-01→2026-07, mass
templates (≥5 copies) dropped, <200 chars or >30% `XXXX` dropped, no sub-issue dropped, paths
with <30 rows dropped. Credit reporting dominates (~70%), so samples are stratified.

**Phases:** Phase 1 = Google Colab experiment (classifier + validator leaderboards, go/no-go at
≥85% adjusted Product accuracy AND Gemini beats TF-IDF at every level, McNemar p<0.05).
Phase 2 = same `nlp/` package as a Docker image on Cloud Run Jobs + BigQuery + dbt + Airflow
(designed after go/no-go). Design: `docs/superpowers/specs/2026-10-26-cfpb-nlp-classifier-design.md`, v2 changes
`docs/superpowers/specs/2026-10-27-descriptions-v2-design.md`;
plan: `docs/superpowers/plans/2026-10-26-phase1-colab-nlp.md`. Run guide: `README.md`.

**Architecture (`nlp/`, all logic here):**
- `config.py` (Config from `config/phase1.yaml` + Drive `config_overrides.json`), `schema.py`
  (constants; `label_of`/`split_label`), `store.py` (Parquet stage store, resumable), `errors.py`
  (`StageIncomplete`: plain-English stop), `colab.py` (notebook setup/helpers: archive on
  `data_version` change, key check, `diagnose`, description stage + Sheet, `prompt_check`)
- `data.py` (load + filters), `taxonomy.py` (taxonomy with `label` = `Product|Issue|Sub-issue`,
  unique per row; names may not contain `|`; prompt listing `- label: <label>` +
  indented `description: ...` so names with `:` stay unambiguous),
  `sampling.py` (stratified splits, seed 20261026)
- `classifiers/` — `tfidf.py`, `embed.py` (embeddings+LR, kNN, `FakeEncoder`), `modernbert.py`,
  `base.py`; `llm/` — `describe.py` (one Gemini request per product writes every label's
  description from ≤15 train complaints; stage `descriptions`, `DESCRIBE_VERSION`), `classify.py`
  (Gemini prompt v2: answers one `label`, exact or normalized match, split into the 3 levels),
  `client.py` (`RequestRejected` for non-429 4xx), `judge.py` (summary judge), `jev.py` (TypeSafe
  Jev validator)
- `budget.py` (spend ledger, cap, rate limiter), `validators.py`, `ensemble.py`, `review.py` +
  `sheets.py` (hand-check Google Sheet), `metrics.py`, `scoring.py` (leaderboards, go/no-go),
  `pipeline.py` (orchestration reused by notebooks and smoke test), `smoke.py`,
  `tools/check_notebooks.py`
- `notebooks/00`–`05` are thin: they only call `nlp.*`. Full table: `notebooks/README.md`.

**Conventions / rules:**
- Notebooks stay thin; `python -m nlp.tools.check_notebooks notebooks/` (also in CI) verifies
  every `nlp.*` name they use exists.
- Heavy libs (torch, transformers, sentence_transformers, cleanlab) are imported lazily inside
  functions. Tests needing GPU/network/keys are `@pytest.mark.slow` (skipped by default).
- Python 3.11 compatible; CI runs 3.11 and 3.13 (Colab's), behind the required `lint-and-test`
  gate job. Colab ships google-genai 2.12.1: keep `google-genai>=2.12` satisfiable by it, and
  notebooks never pip-install Google libraries (only `.[ml]`; upgrading genai broke google-auth).
- **Money path:** strictly sequential reserve → call → record in ledger → write cache. Ledger is
  written before the cache so a crash never under-counts. Hard cap **$5.00** (`spend_cap_usd`,
  raised from $4 by Marco on 2026-10-27),
  covers Gemini and Jev (same ledger). Prices live in config, not code. Output price includes
  thinking tokens. Jev is input-only priced.
- Gemini cache key = complaint_id|classifier_name|model|prompt_version, where the recorded
  prompt_version is `<cfg.prompt_version>+<descriptions fingerprint>` so edited descriptions are
  never answered from the cache; description cache key = `product|describe|model|DESCRIBE_VERSION`.
  Every prediction row carries `model_id` and `prompt_version`. Judge model must differ from
  classifier model. Classification needs the descriptions stage (notebook 03 Part 0).
- Ledger lines may carry `tag` (classification: the recorded prompt version; `describe:d1`,
  `summary_judge:sj-v1`, `jev:<ver>`) and `rows`; cost projections for classification use only
  lines with the current tag (per complaint), never older smaller prompts. Estimate gates skip
  cached rows, so a fully cached rerun costs $0 and is never blocked. `max_output_tokens` is
  clamped to `MAX_OUTPUT_TOKENS` (65536).
- `data_version` (config) vs Drive `data_version.txt`: `setup()` archives old `*.parquet`,
  `models/`, Sheet URL files, `results.md` into `archive_<old>/`; never moves the ledger, caches or
  `config_overrides.json`. Bump `data_version` whenever stored stages become incompatible.
- Messages Marco sees are plain English and say what to do next (`StageIncomplete`); Google 4xx
  are mapped in `client.explain_rejection` (retired model, thinking level, API key).
- Test set is never used for tuning. Caches, ledger, models, Parquet live on Drive
  (`MyDrive/cfpb-nlp-classifier/phase1`), never in git. Public repo: no keys/data/models committed.
- Validators: CFPB label, second opinions, Gemini confidence, self-consistency, hierarchy rules,
  Jev (replaced the Gemini "JEB" judge), cleanlab, simple + learned ensembles.

**Workflow:** `main` protected by ruleset `protect-main` (PR + CI `lint-and-test` + 1 approval;
repo admin Marco bypasses). Sister's PRs need Marco's approval. CI = ruff + pytest + notebook
check (Python 3.11 and 3.13). Venv: `.venv`; `pytest -q`, `ruff check .`.
