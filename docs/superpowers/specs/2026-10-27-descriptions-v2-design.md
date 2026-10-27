# Phase 1 v2: descriptions, sub-issue-only taxonomy, and first-run fixes

_2026-10-27. Builds on `2026-10-26-cfpb-nlp-classifier-design.md`; everything there still holds
unless changed here._

## Why

On the first live run, Gemini 3.8-flash (zero-shot, prompt v1) scored this on the 800 test
complaints (vs CFPB labels):

| | Product | Issue | Sub-issue | Path |
|---|---|---|---|---|
| Gemini 3.8-flash | 82.0% | 51.4% | 37.1% | 32.4% |
| TF-IDF | 77.1% | 54.2% | 45.8% | 44.0% |

Gemini only sees category names, while TF-IDF learned the CFPB's labelling habits from 20,000
examples. Gemini loses on issue and sub-issue. Marco's friend's approach is a description per
category row. Marco chose **descriptions only (no examples yet)**, to measure what they do alone.
The run also exposed several operational failures (listed under "First-run fixes"). Marco's bar:
**the next run must not hit any of these errors again.**

## Decisions (Marco's)

1. **Only categories with a real sub-issue.** Complaints whose CFPB sub-issue is empty are dropped
   from the data, taxonomy, splits and every model. No "(none)" and no "Other".
2. **Gemini outputs one label**, the exact string `Product|Issue|Sub-issue` (single `|`, no
   spaces around it, CFPB wording unchanged). That string is unique per taxonomy row. No renaming of
   sub-issues and no lookup step: product, issue and sub-issue are recovered by splitting on `|`.
3. **Descriptions, done well.** Every taxonomy row gets a description, written automatically and
   visible in the taxonomy Google Sheet.
4. **No examples in the prompt for now.**

## Design

### Data (`nlp/data.py`)

- Replace the `sub_issue_consistency` step with **`no_sub_issue`**: drop rows where `sub_issue IS
  NULL` (also drop empty or whitespace-only strings). `FilterParams.sub_issue_required_frac` is
  removed. The filter report keeps one row per step: `nulls, window, templates, low_signal,
  no_sub_issue, rare_paths`.
- Nothing may produce the literal `"(none)"` any more.

### Taxonomy (`nlp/taxonomy.py`, `nlp/schema.py`)

- `TAXONOMY_COLS` gains `label` (`f"{product}|{issue}|{sub_issue}"`). Keep `path_id` as is.
  Downstream code keys on it.
- `build_taxonomy` raises a clear `ValueError` if any product, issue or sub-issue contains `|`.
- The prompt listing renders one line per row: `- <label>: <description>`, sorted by label.

### Descriptions (new `nlp/llm/describe.py`)

- **One Gemini request per product.** Group all of a product's taxonomy rows together, so the model
  sees every sibling issue and sub-issue and can contrast them.
- **Evidence:** up to 15 training-split complaints per label. Seeded with `cfg.seed`, taken from
  `split == "train"` only (never tuning/test), each narrative passed through `clean_text` and capped
  at about 600 characters.
- **Instructions:** for each label, write at most 45 words in this shape: what it covers; "Use
  when …"; "Not when … (use '<closest sibling label>' instead)" when a confusable sibling exists.
  Base it only on the evidence and the category names. Return JSON `[{label, description}]` with
  every label of that product exactly once. Use the given labels verbatim.
- **Model:** new config key `describe_model` (default `gemini-3.8-flash`, the best available model).
  New constant `DESCRIBE_VERSION = "d1"`.
- **Money path identical to classification:** reserve (estimate x1.5) → limiter.wait → generate →
  ledger.record → cache.put. Cache key `f"{product}|describe|{model}|{DESCRIBE_VERSION}"`, in the
  existing Gemini cache. Uses `with_backoff` for 429/5xx/timeouts. `BudgetExceeded` and
  `DailyQuotaExhausted` behave as in classification.
- **Validation:** labels that are missing, unknown or duplicated, or descriptions that are empty, are
  an error naming the product. One automatic retry for that product. If it still fails, raise a
  plain-English error. Never silently save a partial description table.
- Stored as stage `STAGE_DESCRIPTIONS = "descriptions"` (`label, description, model_id,
  describe_version`).
- An estimate function gives the notebook a cost preview before spending.
- **Sheet:** after the stage exists, the taxonomy Sheet is rewritten with columns `label, product,
  issue, sub_issue, description`. An optional notebook cell reads Marco's edits back from the Sheet
  and overwrites the stage. It validates that the label set is unchanged and that no description is
  empty.

### Classification prompt v2 (`nlp/llm/classify.py`)

- `prompt_version: v2` in config.
- The system prompt contains the description listing and asks for exactly one label per complaint,
  copied verbatim from the list. Response schema per complaint: `complaint_id, label, summary,
  confidence`.
- **Parse:** exact label match, else a normalized match (same normalization as `normalize_path`,
  applied to the whole label), else invalid. Invalid answers are handled exactly as invalid paths are
  today. Valid labels are split into `product, issue, sub_issue`, so the `PREDICTIONS_COLS` shape and
  everything downstream (validators, Jev, review sheet, scoring) are unchanged.
- The token estimate accounts for the longer system prompt (it is computed from the real text
  already; verify).
- Classification needs the descriptions stage. If it is missing, fail with "Run Part 0
  (descriptions) in notebook 03 first."

### Old results on Drive (automatic, no manual step)

- New config key `data_version: v2`. `setup()` compares it with `data_version.txt` in the artifacts
  dir. If they differ, it moves every `*.parquet`, `models/`, `taxonomy_url.txt` and
  `review_sheet_url.txt` into `archive_<old version or "v1">/`, writes the new marker and prints what
  it moved.
- **Never moved:** `ledger.jsonl` (the spend cap must count the ~$1.21 already spent),
  `gemini_cache.jsonl`, `judge_cache.jsonl`, `config_overrides.json`.
- Idempotent: running it again is a no-op. Covered by tests, including "rerun after new results
  exist does not archive them".

### First-run fixes (each needs a test reproducing the failure)

1. **Retired model.** Remove `gemini-2.5-flash` from config (`classifier_model` default becomes
   `gemini-3.8-flash`; keep the price entries) and from the bake-off list. Notebook 03 candidates:
   `gemini-3.8-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`.
2. **Any request Google rejects (4xx other than 429) must not crash a comparison.** `GenAIClient`
   maps them to a new `RequestRejected(status, message)`, which is never billed, so nothing is
   recorded. `compare_runs` catches it per run, prints `"<run> skipped: <short reason>"` and
   continues; the table shows the run as skipped. `stage_paid` turns it into a plain-English
   `StageIncomplete`. Special messages:
   - 404 "no longer available": "model retired, remove it from the list".
   - 400 thinking level not supported: "this model does not support that thinking level".
   - 400/401/403 API key: "Gemini rejected the API key. See README 'Key problems'".
3. **Thinking levels.** Notebook 03 uses `["low", "medium"]` (3.8-flash rejects `minimal`). Item 2
   covers any other unsupported level.
4. **Keys:** `read_secrets` strips whitespace from both keys.
5. **Key check in notebook 00:** a free call (list models) proves the Gemini key works. The new
   `AQ.`-prefixed AI Studio keys are normal. On failure it prints the README fix.
6. **Never pip-install Google libraries in Colab.** Upgrading `google-genai` mid-session broke
   `google-auth`. The stock Colab `google-genai` 2.12.1 works with `AQ.` keys. `pyproject` gets the
   floor `google-genai>=2.12` (satisfied by Colab, so pip never upgrades it). Verify the code imports
   and builds requests under google-genai 2.12.1 (scratch venv) and under the CI version.
7. **TF-IDF convergence:** `LogisticRegression(max_iter=1000)` (was 300).
8. **CI also runs on Python 3.13** (Colab's version) alongside 3.11.
9. **A diagnostics helper** `colab.diagnose(cfg, api_key, ledger, candidates)`. It is free: no
   generate calls. It prints the candidates in memory, the classifier/judge, the library version, the
   key length, prefix and whitespace, a key check, the ledger total, paid requests per model and
   saved answers per run. Notebook 03 gets an optional "Troubleshoot" cell calling it.

### Notebooks

- **00:** key check (fix 5).
- **01:** unchanged calls; the filter table shows `no_sub_issue`. The taxonomy Sheet now has a
  `label` column.
- **02:** unchanged (retrains automatically because the training data changed).
- **03:**
  - New **Part 0, descriptions**: estimate, run, Sheet link, and an optional "load my edits" cell.
  - Part A candidates per fix 1. Text: "skip Part A/B if you already picked; your saved picks are
    kept".
  - Part B levels per fix 3.
  - **Part C becomes "check the new prompt on all 200 tuning complaints"**: free to read, about
    $0.25. It compares with the v1 tuning numbers if cached.
  - Troubleshoot cell (fix 9).
  - Part D unchanged.
- **04/05:** unchanged unless the label change requires it.

### Docs

- README:
  - "What to do if a notebook stops" gains: key problems (`AQ.` keys normal; "API key not valid"
    steps); never pip-install Google libraries; Drive "credential propagation was unsuccessful"
    (tick every permission box, allow pop-ups); `KeyboardInterrupt` (rerun, nothing lost);
    "Restart session" vs "Disconnect and delete runtime"; the "Save in GitHub" banner (ignore, do not
    click); editing a notebook cell (run the edited cell before the ones after it).
  - The step-by-step guide covers Part 0.
- CLAUDE.md and STATE.md are updated.

## Budget

About $1.21 has been spent of the $4.00 cap; the ledger is kept. Rough remaining costs:

| Step | Cost |
|---|---|
| Descriptions | ~$0.3 |
| Tuning check | ~$0.25 |
| Test (longer prompt) | ~$1.2-1.5 |
| Self-consistency x2 (150 rows) | ~$0.5 |
| Jev + summary judge | ~$0.15 |

That totals about $2.4-2.7. It fits, but tightly. The existing phase-estimate gate stops cleanly if
it would not fit. Raising `spend_cap_usd` is Marco's decision only; it is not changed here.

## Non-goals

Examples in the prompt; a shortlist of candidate categories; Phase 2.
