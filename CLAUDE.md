# cfpb-nlp-classifier — primer

**Purpose:** Rebuild (to state-of-the-art standard) a friend's Gemini-based case classifier:
complaint text + a category sheet (group/subgroup/detail) → category + summary. Marco's goal is
to become expert in NLP classification and take it to production.

**Data:** CFPB Consumer Complaint Database (CC0). group/subgroup/detail = Product → Issue →
Sub-issue. Bulk file: `https://files.consumerfinance.gov/ccdb/complaints.csv.zip` (~360 MB,
updated daily). Search API: `https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/`
(fields: product, sub_product, issue, sub_issue, complaint_id, date_received, …). Credit
reporting dominates (~70% of all complaints) — stratify samples.

**Phases:** Phase 1 = Google Colab experiment (classifier + validator leaderboards, go/no-go at
≥85% adjusted Product accuracy). Phase 2 = BigQuery + dbt + Airflow (local Docker) daily
incremental replay. Full design: `docs/superpowers/specs/2026-10-26-cfpb-nlp-classifier-design.md`.

**Key constraints:**
- Gemini spend: paid Tier 1, $5 budget, **hard cap $4.00 enforced in code**.
- Save-as-you-go: every stage resumable; Gemini responses cached per request.
- One repo (private) for NLP + dbt + Airflow. Raw data / artifacts never in git (Drive in
  Phase 1, BigQuery in Phase 2).
- All logic in the `nlp/` package; notebooks are thin.

**Workflow:** `main` protected — PR + passing CI + 1 approval. CI = ruff + pytest (GitHub Actions).
