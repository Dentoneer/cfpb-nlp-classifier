# STATE

_Last updated: 2026-10-26_

## Current status
- Phase 1 design approved by Marco (2026-10-26).
- Repo live and **public**: https://github.com/Dentoneer/cfpb-nlp-classifier (public because
  rulesets need GitHub Pro on private repos). Ruleset `protect-main` (id 25913186): PR + 1
  approval required, repo admin (Marco) bypasses. Secret scanning + push protection on.
- CI workflow (`.github/workflows/ci.yml`, ruff + pytest) written but **not yet pushed** — Marco
  must run `gh auth refresh -h github.com -s workflow` first.
- No NLP code yet. Implementation plan is next.

## Recent decisions
- Dataset: CFPB complaints; hierarchy Product → Issue → Sub-issue.
- Classifiers: Gemini (main), TF-IDF+LR, sentence-transformers+LR, kNN, ModernBERT. BERTopic dropped.
- Validators: CFPB label, second opinions, Gemini confidence, self-consistency, hierarchy rules,
  JEB binary judge, cleanlab, ensemble (simple + learned).
- Billing: dedicated GCP project `cfpb-nlp-classifier`, Tier 1, $5 budget, $4 code cap.
  Free tier was 5 RPM / 20 RPD (Gemini 2.5 Flash) — too slow.
- One repo; collaborator access for Marco's sister; PR approvals required.

## Open threads / next steps
- Write implementation plan (`docs/superpowers/plans/`).
- Marco: create GCP project + API key + billing + $5 budget alert.
- Verify Gemini model choice + prices at build time.
- Add sister as collaborator (Write role): Marco to invite via Settings → Collaborators using her
  email, or get her username.
