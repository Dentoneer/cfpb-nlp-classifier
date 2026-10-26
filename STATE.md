# STATE

_Last updated: 2026-10-26_

## Current status
- Phase 1 design approved by Marco (2026-10-26).
- Repo live and **public**: https://github.com/Dentoneer/cfpb-nlp-classifier (public because
  rulesets need GitHub Pro on private repos). Ruleset `protect-main` (id 25913186): PR + 1
  approval required, repo admin (Marco) bypasses. Secret scanning + push protection on.
- CI live (`.github/workflows/ci.yml`, ruff + pytest) via PR #1; ruleset now also requires the
  `lint-and-test` check. Sister (Karoltrojacek) invited with Write — pending acceptance.
- Implementation plan written: `docs/superpowers/plans/2026-10-26-phase1-colab-nlp.md`
  (14 tasks). Awaiting Marco's review + choice of execution method.
- Verified Gemini prices 2026-10-26 (per 1M in/out): 3.8-flash 0.75/3.75, 2.5-flash 0.30/2.50,
  3.5-flash-lite 0.30/2.50, 3.1-flash-lite 0.25/1.50. 3.8-flash likely too pricey for the $4 cap
  on the full run → small bake-off decides.
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
- Marco: create GCP project + API key + billing + $5 budget alert.
- Verify Gemini model choice + prices at build time.
- Add sister as collaborator (Write role): Marco to invite via Settings → Collaborators using her
  email, or get her username.
