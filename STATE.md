# STATE

_Last updated: 2026-10-27_

## Current status
- **Phase 1 code complete** on branch `feat/phase1-nlp` (14 tasks + Jev task, 216 tests passing).
  Final whole-branch review done ("with fixes"); all 12 fixes committed in `fff1a90` (data load
  filters inside DuckDB, cleanlab missing-row crash, explicit budget-stop exception, 5xx/timeout
  retries, summary-judge section in results, Jev ids in nb 04, model fingerprint includes
  training rows, config price validation). Scoped re-review clean (all 12 addressed). Next: one PR
  to `main` (awaiting Marco's go-ahead to push).
  Notebooks clone `main`, so the PR must be merged before Marco can run them.
- Repo is public: https://github.com/Dentoneer/cfpb-nlp-classifier (public because rulesets need
  GitHub Pro on private repos). Ruleset `protect-main` (id 25913186): PR + 1 approval + the
  `lint-and-test` check; repo admin (Marco) bypasses. Secret scanning + push protection on.
  CI = ruff + pytest + notebook check. Sister (Karoltrojacek) invited with Write — acceptance
  not confirmed.
- Run guide for Marco and his sister is the "Run Phase 1" section in `README.md`;
  `notebooks/README.md` has the notebook table. Expected cost ~$2-3 of the $4 cap.
- Not yet run live: nothing has been executed in Colab against real keys.

## Recent decisions
- **Data source (2026-10-26):** CFPB stopped publishing narratives on 2026-08-14. Using HF mirror
  `Bob-68/cfpb-consumer-complaints` (3.8M rows with narrative, to 2026-07-06) via DuckDB with
  Marco-approved noise filters (window 2024-01→2026-07, templates ≥5 copies dropped — one letter
  appeared 30,102×, <200 chars / >30% XXXX dropped, sub-issue consistency rule, paths <30 rows
  dropped). Expect ~0.9-1M clean rows.
- **Classifiers:** Gemini (main), TF-IDF+LR, embeddings+LR, kNN, ModernBERT. BERTopic dropped.
  Gemini model chosen by a bake-off in notebook 03 (saved to Drive `config_overrides.json`).
  Prices verified 2026-10-26 (per 1M in/out): 3.8-flash 0.75/3.75, 2.5-flash 0.30/2.50,
  3.5-flash-lite 0.30/2.50, 3.1-flash-lite 0.25/1.50.
- **Validators:** CFPB label, second opinions, Gemini confidence, self-consistency, hierarchy
  rules, Jev, cleanlab, simple + learned ensembles. **"JEB" = Jev** (TypeSafe AI's yes/no model):
  replaced the Gemini judge as validator `jev` (3 questions per complaint, $0.042/1M input,
  output free, same ledger/cap). Gemini remains only as the summary judge.
- **Billing:** dedicated GCP project, Tier 1, $5 budget alert, **$4 hard cap in code** (Google
  alerts do not stop spending). Free tier (5 RPM / 20 RPD) too slow.
- **Operationalization (Phase 2):** Colab for Phase 1 experiments only. Production = same `nlp/`
  package as a Docker image (CI builds it) on **Cloud Run Jobs**, Airflow to sequence (Docker
  locally to start), BigQuery for storage, dbt for merge/tests. Vertex AI Pipelines and Cloud
  Composer rejected as overkill; BigQuery in-SQL Gemini rejected (loses spend cap, cache,
  validation); JupyterHub not needed.
- Workflow: one feature branch, commit per task, one PR at the end.

## Open threads / next steps
- Finish: push `feat/phase1-nlp`, open PR → `main`; Marco merges.
- Known, accepted limits: a timeout after Gemini already billed may under-count the ledger by
  ≤1 request; a 429 that persists through 5 retries surfaces as a raw `RateLimited` error;
  peak memory of notebook 01 on the full mirror is unmeasured (DuckDB spills to
  `/content/duckdb_tmp`) — watch the first run.
- Add this project to sites-hub (controller/Marco).
- **Marco before running:** create GCP project + Gemini API key + billing + $5 budget alert; get a
  Jev key from console.typesafe.ai; add `GEMINI_API_KEY` and `TYPESAFE_API_KEY` to Colab Secrets
  (Notebook access on).
- **Then run notebooks 00 → 05** (02 needs a T4 GPU). Stop after 04 for ~2 hours of hand review of
  the ~200-row Google Sheet, then run 05 for the leaderboards and go/no-go.
- **Confirm Jev response JSON nesting** on the first live call (slow test `test_live_jev_call`,
  or notebook 04's Jev step); adjust the parser if it differs.
- Sister: confirm she accepted the collaborator invite.
- After go/no-go: design Phase 2 (Docker image, Cloud Run Jobs, BigQuery, dbt, Airflow).
