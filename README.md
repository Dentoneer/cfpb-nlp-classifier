# CFPB NLP Classifier

Classifies CFPB consumer complaints into a three-level category hierarchy
(**Product → Issue → Sub-issue**) and writes a short summary of each complaint, using Gemini
as the main classifier. Classic and transformer models (TF-IDF, sentence-transformers,
kNN, ModernBERT) are scored alongside it, and several validation methods are compared on how
well they catch classification mistakes.

## Phases

1. **Phase 1 — Colab experiment (in progress).** Prove the NLP on a small stratified sample,
   produce a classifier leaderboard and a validator leaderboard, and make a go/no-go call
   (≥85% adjusted accuracy at the Product level). Gemini spend is capped at $4 in code.
2. **Phase 2 — Production.** BigQuery tables, dbt models + tests, a daily Airflow DAG that
   classifies newly arriving complaints incrementally and merges results back onto the source
   data.

Design: [docs/superpowers/specs/2026-10-26-cfpb-nlp-classifier-design.md](docs/superpowers/specs/2026-10-26-cfpb-nlp-classifier-design.md)

## Data

[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
(public domain, CC0). Raw data is never committed to this repo.

## How we work

- `main` is protected: all changes go through a pull request, CI must pass, and one approval
  is required before merging.
- CI (GitHub Actions) runs lint (`ruff`) and tests (`pytest`) on every pull request.
- Secrets (Gemini API key, etc.) live in Colab Secrets / GitHub Secrets — never in code.
