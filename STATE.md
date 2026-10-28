# STATE

_Last updated: 2026-10-28_

## Current status
- **Phase 1 v1 merged** (PR #2) and **run live once (2026-10-27)**. Notebooks 00-03 Part D done.
  Baselines (test, vs CFPB labels) product/issue/sub_issue/path: TF-IDF .771/.542/.458/.440,
  embed_lr .795/.511/.451/.429, kNN .784/.507/.430/.406, ModernBERT .808/.538/.440/.414.
  Bake-off (50 tuning): 3.8-flash .84/.62/.40/.38 (picked), 3.5-flash-lite .80/.60/.40/.40,
  3.1-flash-lite .78/.54/.38/.36. Picks saved: 3.8-flash, batch 10, thinking low.
  **Gemini v1 test: .820/.514/.371/.324** -> beats TF-IDF only on product. Spend ~$1.21 of $4.
- **v2 in progress** on branch `feat/descriptions-v2` (design:
  `docs/superpowers/specs/2026-10-27-descriptions-v2-design.md`): drop no-sub-issue categories,
  Gemini outputs one `Product|Issue|Sub-issue` label, auto-written per-category descriptions
  (Marco: descriptions only, no examples yet), auto-archive of v1 Drive results (ledger kept),
  and fixes for every first-run failure. Implemented (commits 8b5d193..fa4bfb2, 292 tests, verified
  on Python 3.13 + google-genai 2.12.1). **Spend cap raised to $5.00 (Marco, 2026-10-27).** Independent
  review: "ready with fixes" (v2-only cost projection — test likely ~$2 not $1.2-1.5; Part C
  compares v2 vs TF-IDF on tuning; don't overwrite Sheet edits; + 8 minors). All fixed in `5bd70f8` (312 tests). **PR #3 merged 2026-10-27** (117e2f8); Marco is rerunning 00->05. **v2 rerun progress:** 00 ok ($1.20 spent);
  01 = 319,016 complaints, 231 paths, window 2025-06..2026-06 (no_sub_issue removed 27,729);
  02 v2 baselines (test) product/issue/sub_issue/path: TF-IDF .810/.581/.482/.470, embed_lr
  .842/.558/.430/.421, kNN .821/.516/.409/.402, ModernBERT .861/.576/.466/.461.
  03 Parts 0, A, B, C all run (A/B re-run on v2 by mistake, ~+$0.5); spend $2.52. Part D estimate
  said $3.72 (x1.5 safety; ~$2.60 real) vs $2.48 left -> stopped by the code cap, nothing spent.
  **Marco 2026-10-28: remove the code cap** - he uses Google *prepaid credits* (no budget alert), so
  the credits are the real limit. `spend_cap_usd` -> 1000 (effectively off); credits-depleted
  (402/429 "prepayment credits are depleted") now -> `RequestRejected` with "add credit at
  aistudio.google.com/billing, rerun" instead of being retried as a rate limit. PR
  `fix/remove-code-cap`. Next: Marco merges, reruns 03 setup + load + Part C (free) + Part D. CI now: `test` matrix (3.11, 3.13) +
  `lint-and-test` gate job (name required by ruleset) — confirm on first PR run.
- Repo is public: https://github.com/Dentoneer/cfpb-nlp-classifier (public because rulesets need
  GitHub Pro on private repos). Ruleset `protect-main` (id 25913186): PR + 1 approval + the
  `lint-and-test` check; repo admin (Marco) bypasses. Secret scanning + push protection on.
  CI = ruff + pytest + notebook check. Sister (Karoltrojacek) invited with Write — acceptance
  not confirmed.
- Run guide for Marco and his sister is the "Run Phase 1" section in `README.md`;
  `notebooks/README.md` has the notebook table. Expected cost ~$2-3 of the $4 cap.

## Recent decisions
- **First-run lessons (2026-10-27):** new AI Studio keys start `AQ.` (normal; work with Colab's
  stock google-genai 2.12.1). Never pip-install Google libs in Colab (upgrading genai broke
  google-auth: `CredentialsWithRegionalAccessBoundary`). gemini-2.5-flash is retired for new users
  (404). 3.8-flash rejects thinking `minimal`. Drive "credential propagation was unsuccessful" =
  permission box not ticked. Edited notebook cells must be re-run before later cells.
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
- Finish v2 rerun: 03 Part D -> 04 (+2h hand review) -> 05. Limit = Google prepaid credits;
  Marco to report his remaining balance (aistudio.google.com/billing). Part C tuning table not yet seen.
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
