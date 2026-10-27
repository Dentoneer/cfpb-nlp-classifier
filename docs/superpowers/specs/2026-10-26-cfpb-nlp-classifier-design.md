# CFPB NLP Classifier — Phase 1 (Colab experiment) Design

_Date: 2026-10-26 · Owner: Marco Coronel_

## Intent and the why

Marco saw a friend's NLP pipeline that takes support-case text plus a Google Sheet of
categories (group / subgroup / detail) and uses Gemini to assign each case a category and
write a summary. He wants to rebuild that process himself, to state-of-the-art standard, as a
way to become genuinely expert in NLP classification — and then take it all the way to a
production pipeline (BigQuery + dbt + Airflow + GitHub).

The work is split in two phases:

1. **Phase 1 (this document): prove the NLP in Google Colab.** Build the classifiers, score
   them against annotated data, and — just as importantly — score several *validation
   methods* to learn which ones actually catch classifier mistakes.
2. **Phase 2 (separate design later): productionize** with BigQuery tables, dbt models and
   tests, Airflow daily schedule, larger sample, and a simulated daily "replay" of new data.

Phase 1 must be built so its code and outputs carry straight into Phase 2 without a rewrite.

## Data

**Source:** CFPB Consumer Complaint Database (public domain, US government). Use only
complaints that have a consumer narrative (the free text).

**Update 2026-10-26 (approved by Marco):** the CFPB stopped publishing narratives on
2026-08-14; the official bulk file and API no longer contain them. Source is therefore the
Hugging Face mirror `Bob-68/cfpb-consumer-complaints` (Parquet, 16.56M rows, 3.8M with
narrative, dates to 2026-07-06), read with DuckDB directly from `hf://` so only the needed
columns/rows are fetched. Phase 2's daily feed is a replay of held-back rows from this snapshot.

**Noise filters (approved by Marco), applied once; a report records rows removed per filter:**
1. Drop rows with null narrative, Product, or Issue.
2. Window 2024-01-01 → 2026-07-06 (after CFPB's 2023 category rename; window check still runs).
3. Mass templates: normalized narrative (lowercase, whitespace collapsed) appearing ≥ 5 times →
   all copies dropped; appearing 2–4 times → keep one copy (earliest `date_received`, then
   lowest `complaint_id`).
4. Low signal: narrative < 200 characters, or > 30% of characters are `X` (redactions).
5. Inconsistent labels: null Sub-issue on an Issue that has a Sub-issue in ≥ 90% of its rows →
   drop; null Sub-issue on other Issues → `(none)`.
6. Rare paths: Product→Issue→Sub-issue paths with < 30 rows after filters 1–5 → drop.

Expected ≈ 0.9–1M clean complaints across 11 Products (~79% credit reporting).

**Category hierarchy (Marco's group / subgroup / detail):**
`group = Product`, `subgroup = Issue`, `detail = Sub-issue`.
_My call:_ Sub-product was skipped because it is often blank and describes a product variant,
not what the complaint is about. Some Issues have no Sub-issues; `detail` is then a
`(none)` value, which counts as a valid answer.

**Time window:** restrict to a recent window in which the Product/Issue/Sub-issue taxonomy is
stable (CFPB has revised it over the years). Builder determines the exact window from the data.

**Dataset criteria (Marco's, must hold for the sample):** annotated categories, ≥1,000
entries, free text, >20 annotated categories (the Issue level alone exceeds this).

**Known caveat:** CFPB labels are picked by consumers on a form, not by experts, so some
"ground truth" labels are wrong. The validation design below exists largely because of this.

## Sample — hard constraints

Marco's build criterion: **the Colab phase must use a small sample so that all Gemini work
finishes in under 2 hours and stays inside the Gemini free-tier daily limits.** Bigger
samples are for Phase 2.

Key point: only rows sent to Gemini cost quota. The non-LLM classifiers can train on a much
larger pool for free.

| Split | Size (target) | Touches Gemini? | Purpose |
|---|---|---|---|
| Train pool | up to ~20,000, stratified | No | Train TF-IDF, sentence-transformer, kNN, ModernBERT; cleanlab |
| Tuning set | ~200 | Yes | Iterate the Gemini prompt (look as often as you like) |
| Frozen test set | ~800 | Yes | Score every classifier and validator once per candidate |
| Hand-check set | ~200, drawn from the test set | (already scored) | Marco's manual verdicts — the only true ground truth |
| Self-consistency subset | ~150, drawn from the test set | Yes (×2 extra runs) | Self-consistency validator |

Rough Gemini budget, batching ~10 complaints per request: ~4 prompt iterations on tuning
(~80 requests) + test classification (~80) + JEB judge (~80) + self-consistency (~30)
≈ **~270 requests**. Builder must:

- **First verify the current free-tier model name and limits** (requests/minute, requests/day,
  tokens/minute) from Google's official page — web sources disagreed in Oct 2026
  (most said ~10 RPM / ~1,500 RPD for Flash).
- Keep total planned requests ≤ ~60% of the verified daily limit, leaving room for retries.
- Rate-limit client-side, retry with backoff, and **cache every response to disk** keyed by
  `(complaint_id, model_id, prompt_version)` so a Colab disconnect or rerun never re-spends
  quota.
- Shrink sizes if verified limits are tighter than assumed; the ratios matter more than the
  absolute numbers.

**Billing decision (Marco, 2026-10-26): paid Tier 1 with a $5 total budget, so Phase 1 runs in
one session instead of ~2 weeks.** Marco's free-tier limits turned out to be 5 RPM / 250K TPM /
20 RPD (Gemini 2.5 Flash). Rules:
- Dedicated Google Cloud project `cfpb-nlp-classifier` (own API key, own quotas), linked to
  Marco's existing billing account, with a project-scoped budget alert at $5. Google budget
  alerts only notify — they do **not** stop spending — so the code enforces the cap itself.
- **Hard spend cap in code:** the runner tracks input/output (and thinking) tokens per request
  from the API's usage metadata × the chosen model's published price, keeps a running total on
  Drive, and stops before a new request would push total Phase 1 spend past **$4.00** (leaving
  margin under $5). Prices live in config, verified from Google's pricing page at build time.
- Before any paid run, print a cost estimate (requests × expected tokens × price) and require it
  to fit inside the remaining cap.
- Keep thinking/reasoning set to the lowest level that holds accuracy (thinking tokens are
  billed as output) — compare on the tuning set.
- Rough estimate: ~270 requests × ~10K input tokens + modest output ≈ a few dollars at Flash
  prices; if the verified price makes it tighter, cut self-consistency first, then prompt
  iterations, never the test set below ~400.
- Pick the model by quality per dollar among current Flash / Flash-Lite models (verify prices
  and Tier 1 limits at build time).

**Save as you go / resume anywhere (Marco's requirement).** The notebook must never have to
start from scratch after a Colab crash, a disconnect, or hitting the daily limit:
- Every stage (sample, splits, each classifier's predictions, each validator's results,
  metrics) writes its output to Drive when it finishes, and on start skips work whose output
  already exists.
- Gemini responses are appended to the on-disk cache **as each request completes**, not at the
  end of a loop.
- On a daily-quota error (429 RESOURCE_EXHAUSTED that persists after backoff), the runner saves
  state, reports how many rows remain, and stops cleanly. Rerunning the same cell the next day
  (quota resets at midnight Pacific; limits are per Google Cloud project, not per key) continues
  where it left off.
- So **spreading Gemini work over several days is acceptable** (each session ≤ 2 h, ≤ that day's
  quota). The test set is not shrunk below ~400 rows to fit one day.
- Google's docs page lists no free-tier numbers (only "view in AI Studio"), so limits are
  config values the builder sets from Marco's AI Studio Rate Limit page. Gemini 4 Argon was
  announced 2026-10-26; check whether it or a newer Flash is on the free tier when choosing the
  model.

Splits are drawn once with a fixed seed, stratified by Product (with a minimum floor for small
categories), and saved as a file of `complaint_id → split`. The test set is never used for
prompt tuning.

**Hand-check selection:** draw the 200 rows stratified by *whether Gemini agreed with the CFPB
label* (a mix of agree and disagree), **not** by any validator's output — otherwise the
validator scores would be biased.

## Classifiers

Each classifier predicts the full path (Product → Issue → Sub-issue) and must output a path that
exists in the taxonomy. Each also doubles as a "second opinion" validator for Gemini.

| Classifier | Notes |
|---|---|
| **Gemini (main)** | Latest free-tier Flash model; exact model id pinned in config. Input: batch of complaints + taxonomy. Output (structured JSON): group, subgroup, detail, summary, confidence (0–1). Outputs not in the taxonomy are rejected/retried. Token log-probabilities used as an extra confidence signal **only if** the API exposes them for that model. |
| TF-IDF + logistic regression | The floor every method must beat. |
| Sentence-transformers embeddings + logistic regression | Builder picks a current, strong open embedding model that runs on Colab's free GPU. |
| k-nearest neighbours | On the same sentence-transformer embeddings; predicts the majority label of the most similar train complaints. |
| ModernBERT (fine-tuned) | `ModernBERT-base` fine-tuned on the train pool on Colab's free GPU; truncate long narratives. Keep training time reasonable (~1 h); reduce train size if needed. |

Out: BERTopic (Marco removed it).

## Validators (methods that judge whether Gemini's answer is right)

| # | Validator | Output per row |
|---|---|---|
| 1 | vs CFPB label | match / no match, per level |
| 2 | TF-IDF second opinion | agrees with Gemini? (+ probability) |
| 3 | Sentence-transformer + LR second opinion | agrees? (+ probability) |
| 4 | kNN second opinion | share of neighbours with Gemini's label |
| 5 | ModernBERT second opinion | agrees? (+ probability) |
| 6 | Gemini's own confidence | 0–1 (+ log-prob based score if available) |
| 7 | Self-consistency | agreement across 3 runs (subset only) |
| 8 | Hierarchy rules | path valid / invalid |
| 9 | **JEB** — GenAI binary judge | yes / no "is this classification correct", per level. Use a different model than the classifier where the free tier allows (models tend to approve their own answers). |
| 10 | cleanlab | probability the *CFPB label* itself is wrong |
| 11 | **Ensemble vote** | consensus score 0–1, ✅/🟡/🔴 status, list of dissenting validators |

**Ensemble — two versions, both scored:**
- *Simple:* count of validators saying "right", plus who dissented.
- *Learned:* a small model weighting each validator, trained on the hand-check rows with
  cross-validation (never scored on rows it learned from).

**Row status** (from the ensemble): ✅ verified, 🟡 likely correct (disagrees with CFPB
label but validators side with the model), 🔴 needs review.

During Phase 1, **every validator runs on the full test set** (self-consistency on its
subset) so they are compared on the same rows.

## Summaries

Gemini's summary has no answer key. Check it on ~100 test rows with the JEB judge (faithful?
covers the main issue? anything invented?) and include a "summary OK?" column in Marco's
hand-check sheet.

## Human review (Marco's part)

One-time, ~200 rows, ~2 hours. Delivered as a **Google Sheet** with the complaint text, CFPB
label, Gemini's answer and summary, and **one verdict dropdown per level** (Product, Issue,
Sub-issue): *model right / label right / both OK / both wrong / ambiguous*, plus *summary OK?
yes/no*. _My call:_ at any level where Gemini and the CFPB label already agree, the verdict is
pre-filled as "both OK" (Marco can override) to keep review time near 2 hours. The notebook
reads the verdicts back.

**Counting rules (_my call_):** "model right" and "both OK" = Gemini correct; "label right" and
"both wrong" = Gemini error; "ambiguous" rows are excluded from accuracy and catch-rate math and
their count is reported separately.

## Metrics and outputs

**Classifier leaderboard** (Gemini, TF-IDF, sentence-transformer, kNN, ModernBERT), per level and
full path: accuracy, macro-F1, per-category precision/recall, confusion tables, 95% intervals
(bootstrap), McNemar test for Gemini vs each other classifier. Reported both vs CFPB labels
(full test set) and as **adjusted accuracy** vs Marco's verdicts (hand-check set; "model right"
or "both OK" counts as correct).

**Validator leaderboard:** scored against Marco's verdicts: catch rate (recall on Gemini's
real errors), false-alarm rate, AUROC, and cost (Gemini requests, runtime).

**Weighting (_my call_):** the hand-check set over-samples Gemini/label disagreements, so
adjusted accuracy and validator false-alarm rates / AUROC are **reweighted back to the
agree/disagree mix of the full test set** (inverse sampling weights). 95% intervals are reported
alongside every number; with ~200 rows expect roughly ±5–7 points.

**Go / no-go for Phase 2 (Marco's bar):**
- Gemini's reweighted **adjusted accuracy at the Product level ≥ 85%** (point estimate; the
  interval is reported so Marco can judge how solid it is — _my call_ not to require the lower
  bound to clear 85%, which would need a much larger hand-check).
- **"Clearly beats TF-IDF at every level" (_my call_):** measured against CFPB labels on the full
  test set; Gemini is higher on both accuracy and macro-F1 at each level, and the accuracy gap is
  significant (McNemar p < 0.05). Adjusted-accuracy comparison is reported too, since noisy labels
  can flatter TF-IDF.

Final notebook section: a short written results summary with both leaderboards and the
go/no-go call.

## Built for Phase 2 from day one

- **One GitHub repo, `cfpb-nlp-classifier`, private**, at
  `C:\Users\coron\Documents\github\cfpb-nlp-classifier`.
  _My call (Marco asked whether to split data vs data-science repos):_ one repo. Raw data does
  not belong in git — it will live in BigQuery — so a "data repo" would hold only code and small
  reference files. One repo keeps taxonomy, prompt versions, NLP code, dbt and Airflow changing
  together. Planned layout:
  - `nlp/` — installable Python package (data loading, sampling, taxonomy, each classifier,
    each validator, metrics). All real logic lives here, not in notebook cells.
  - `notebooks/` — thin Colab notebooks that import `nlp/`, split into resumable stages
    (data & sample → baselines → Gemini → validators → results) so a Colab disconnect doesn't
    lose work.
  - `tests/` — pytest for parsing/validation of LLM output, taxonomy checks, metrics.
  - `dbt/`, `airflow/` — Phase 2, not built now.
- **Persistent storage in Phase 1:** Google Drive (mounted in Colab) for data, caches and
  outputs. Secrets (Gemini API key, GitHub token for cloning the private repo) via Colab
  Secrets — never in git.
- **Outputs shaped like future BigQuery tables**, saved as Parquet:
  `complaints_sample`, `taxonomy`, `splits`, `predictions` (complaint_id, classifier,
  model_id, prompt_version, group, subgroup, detail, summary, confidence, run_ts),
  `validator_results` (long format: complaint_id, validator, verdict, score), `human_review`,
  `metrics`. Phase 2 loads these directly — the classified rows become the "history" half.
- **Taxonomy as a Google Sheet** (mirrors the friend's setup): generated from the data, published
  to a Sheet, and read by the classifier; each run records the taxonomy snapshot it used.
- Every prediction carries `model_id` and `prompt_version`.
- Marco's global convention: repo gets `CLAUDE.md` + `STATE.md`; add the project to `sites-hub`.

## Phase 2 sketch (not built now — for orientation only)

BigQuery (billing linked; sandbox reportedly lacks the MERGE that dbt incremental models need —
verify; set a budget alert), datasets for raw / staging / nlp / marts. Larger stratified sample
split 50/50: history (bulk-classified) + future (released ~daily by a replay table keyed on
assigned arrival date). Daily Airflow DAG (local Docker, not Cloud Composer, for cost): release
rows for the date → dbt staging → classify new rows only → dbt incremental merge of NLP columns
back onto the original complaint by `complaint_id` → dbt tests + daily accuracy gate. Validators
kept in production are whichever win the Phase 1 leaderboard on value for cost.

## Goals / non-goals / anti-goals

**Goals:** classifier leaderboard; validator leaderboard; go/no-go vs the 85% bar; reusable
code and Phase-2-shaped outputs.

**Non-goals (Phase 1):** BigQuery, dbt, Airflow implementation; BERTopic; any UI.

**Anti-goals (failure even if it "works"):** spending more than the $5 budget (code cap at $4); tuning on the test set; logic trapped in notebook cells; API keys or raw data in git;
validator scores biased by how the hand-check rows were chosen.

## Left to the builder

Exact time window; exact sentence-transformer model; batch size (compare 1 vs ~10 per request
on the tuning set if quota allows — batching can cost accuracy); ModernBERT hyperparameters and
train size; whether non-LLM classifiers predict the full path as one label or per level (output
must be a valid path); rare-category handling; JEB model choice within the free tier.

## Update 2026-10-26 — JEB is Jev (approved by Marco)

"JEB" in Marco's request meant **Jev**, TypeSafe AI's discriminative (non-generative) model that
answers typed questions with calibrated probabilities. It replaces the Gemini yes/no judge as the
binary validator. Gemini remains the classifier and the summary judge.
- API: `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <TYPESAFE_API_KEY>`,
  body `{model, state, questions:{id:{type:"noul", instructions}}}`; response `nouls[id].noul`
  (P(yes)) and `usage.input_tokens/output_tokens`. Model `jev-1.14.0`. Price $0.042 / 1M input
  tokens, output free. One request per complaint (state = narrative), three noul questions
  (Product / Issue / Sub-issue fit). Spend recorded in the same ledger under the same $4 cap.
- Validator name `jev`; per-level score = P(yes); path score = product of the three.
- Key stored in Colab Secrets as `TYPESAFE_API_KEY`.
