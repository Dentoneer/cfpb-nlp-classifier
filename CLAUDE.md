# cfpb-nlp-classifier — primer

**Purpose:** Rebuild (to state-of-the-art standard) a friend's Gemini-based case classifier:
complaint text + a category sheet (group/subgroup/detail) → category + summary. Marco's goal is
to become expert in NLP classification and take it to production. His sister (GitHub
`Karoltrojacek`, Write access) collaborates.

**Data:** CFPB Consumer Complaint Database (CC0). group/subgroup/detail = Product → Issue →
Sub-issue; missing Sub-issue = the literal `"(none)"` (a valid value). The CFPB stopped publishing
narratives on 2026-08-14, so the source is the HF mirror `Bob-68/cfpb-consumer-complaints`
(narratives to 2026-07-06) read via DuckDB, with noise filters: window 2024-01→2026-07, mass
templates (≥5 copies) dropped, <200 chars or >30% `XXXX` dropped, sub-issue consistency rule,
paths with <30 rows dropped. Credit reporting dominates (~70%), so samples are stratified.

**Phases:** Phase 1 = Google Colab experiment (classifier + validator leaderboards, go/no-go at
≥85% adjusted Product accuracy AND Gemini beats TF-IDF at every level, McNemar p<0.05).
Phase 2 = same `nlp/` package as a Docker image on Cloud Run Jobs + BigQuery + dbt + Airflow
(designed after go/no-go). Design: `docs/superpowers/specs/2026-10-26-cfpb-nlp-classifier-design.md`;
plan: `docs/superpowers/plans/2026-10-26-phase1-colab-nlp.md`. Run guide: `README.md`.

**Architecture (`nlp/`, all logic here):**
- `config.py` (Config from `config/phase1.yaml` + Drive `config_overrides.json`), `schema.py`
  (constants), `store.py` (Parquet stage store, resumable), `colab.py` (notebook setup/helpers)
- `data.py` (load + filters), `taxonomy.py`, `sampling.py` (stratified splits, seed 20261026)
- `classifiers/` — `tfidf.py`, `embed.py` (embeddings+LR, kNN, `FakeEncoder`), `modernbert.py`,
  `base.py`; `llm/` — `classify.py` (Gemini prompt/parse), `client.py`, `judge.py` (summary
  judge), `jev.py` (TypeSafe Jev validator)
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
- Python 3.11 compatible (CI); no 3.12+-only syntax.
- **Money path:** strictly sequential reserve → call → record in ledger → write cache. Ledger is
  written before the cache so a crash never under-counts. Hard cap **$4.00** (`spend_cap_usd`),
  covers Gemini and Jev (same ledger). Prices live in config, not code. Output price includes
  thinking tokens. Jev is input-only priced.
- Gemini cache key = complaint_id|classifier_name|model|prompt_version; every prediction row
  carries `model_id` and `prompt_version`. Judge model must differ from classifier model.
- Test set is never used for tuning. Caches, ledger, models, Parquet live on Drive
  (`MyDrive/cfpb-nlp-classifier/phase1`), never in git. Public repo: no keys/data/models committed.
- Validators: CFPB label, second opinions, Gemini confidence, self-consistency, hierarchy rules,
  Jev (replaced the Gemini "JEB" judge), cleanlab, simple + learned ensembles.

**Workflow:** `main` protected by ruleset `protect-main` (PR + CI `lint-and-test` + 1 approval;
repo admin Marco bypasses). Sister's PRs need Marco's approval. CI = ruff + pytest + notebook
check (Python 3.11). Venv: `.venv`; `pytest -q`, `ruff check .`.
