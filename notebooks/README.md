# Colab notebooks (Phase 1)

Six thin notebooks that run Phase 1 on the real CFPB data. They only call functions in the `nlp/`
package; each one starts with the same setup cell (clone/pull the repo, install, mount Drive, read the
`GEMINI_API_KEY` and `TYPESAFE_API_KEY` Colab secrets (the second is optional until notebook 04), authorise Google Sheets).

Before the first run: in Colab add a secret named `GEMINI_API_KEY` (key icon, left bar) with
"Notebook access" on. Notebook 04's Jev validator also needs a `TYPESAFE_API_KEY` secret set the same
way; notebooks 00-03 run without it (setup just prints a note).

| # | Notebook | Runtime | Time | Paid cost |
|---|----------|---------|------|-------------|
| 00 | `00_setup.ipynb` - checks Drive, key, Sheets | CPU | 3 min | $0 |
| 01 | `01_data_and_splits.ipynb` - load, filter, taxonomy, splits | CPU | 5-15 min | $0 |
| 02 | `02_baselines.ipynb` - TF-IDF, embeddings, kNN, ModernBERT | **GPU (T4)** | 45-90 min | $0 |
| 03 | `03_gemini.ipynb` - bake-off, settings, 800-row test run | CPU | 30-90 min | ~$1.5-2.5 |
| 04 | `04_validators.ipynb` - judges, validators, review sheet | CPU | 30-60 min | ~$1-2 |
| 05 | `05_results.ipynb` - ensembles, leaderboards, go/no-go | CPU | 5 min | $0 |

Run them in order. After 04, **stop and fill in the review sheet** (link printed at the end of 04),
then run 05. Total paid spend (Gemini + Jev) is capped at $4.00 in code (`spend_cap_usd`); every paid cell prints an
estimate first and refuses to start if it will not fit in the remaining cap.

If a paid cell says the daily quota is exhausted, rerun that cell tomorrow; finished rows are cached.

## Where outputs land

Google Drive: `MyDrive/cfpb-nlp-classifier/phase1/`

- `*.parquet` - every table (complaints, taxonomy, splits, predictions, validator results, human review,
  ensembles, leaderboards, adjusted accuracy)
- `ledger.jsonl` - every paid charge (Gemini and Jev) (shared by all notebooks); `gemini_cache.jsonl`,
  `judge_cache.jsonl` - cached responses, so reruns are free
- `models/` - trained classifiers; `config_overrides.json` - your bake-off pick, batch size and
  thinking level (applied on top of `config/phase1.yaml` by every notebook)
- `taxonomy_url.txt`, `review_sheet_url.txt` - the Google Sheet links; `results.md` - final summary

## Checking the notebooks

`python -m nlp.tools.check_notebooks notebooks/` verifies every `nlp.*` name used in the code cells
exists (runs in CI and in `tests/test_check_notebooks.py`).

## Rerunning after a code change

If the repo changes while a session is open (for example a new prompt version), the setup cell stops with
"Restart session (Runtime menu → Restart session), then rerun this cell". Do that; saved results and
choices on Drive are kept.
