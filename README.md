# CFPB NLP Classifier

Classifies CFPB consumer complaints into a three-level category hierarchy
(**Product → Issue → Sub-issue**) and writes a short summary of each complaint, using Gemini
as the main classifier. Classic and transformer models (TF-IDF, sentence-transformers,
kNN, ModernBERT) are scored alongside it, and several validation methods are compared on how
well they catch classification mistakes.

## Phases

1. **Phase 1 — Colab experiment (code complete, ready to run).** Prove the NLP on a small stratified sample,
   produce a classifier leaderboard and a validator leaderboard, and make a go/no-go call
   (≥85% adjusted accuracy at the Product level). Gemini spend is capped at $5 in code.
2. **Phase 2 — Production.** The same `nlp/` code packaged as a Docker image and run on
   Google Cloud Run Jobs, with BigQuery tables, dbt models + tests, and an Airflow DAG that
   classifies newly arriving complaints incrementally. Designed after the Phase 1 go/no-go.

Design: [docs/superpowers/specs/2026-10-26-cfpb-nlp-classifier-design.md](docs/superpowers/specs/2026-10-26-cfpb-nlp-classifier-design.md),
v2 changes (category descriptions, sub-issue-only categories, first-run fixes):
[docs/superpowers/specs/2026-10-27-descriptions-v2-design.md](docs/superpowers/specs/2026-10-27-descriptions-v2-design.md)

## Data

[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
(public domain, CC0). The CFPB stopped publishing complaint narratives on 2026-08-14, so Phase 1
reads the Hugging Face mirror `Bob-68/cfpb-consumer-complaints` (narratives up to 2026-07-06) and
removes noisy rows (copy-pasted templates, very short or mostly redacted text, complaints without a
sub-issue, rare categories). Every category is a real Product → Issue → Sub-issue path.
Raw data is never committed to this repo.

## Run Phase 1 (step by step)

You do not need to know Python. You click "run" on a few cells in Google Colab (a free notebook
website from Google). Everything the notebooks do is saved to your Google Drive, so if something
stops you can come back later and nothing is lost or paid for twice.

**What you get at the end:** a scoreboard of five classifiers, a scoreboard of the checks that
catch Gemini's mistakes, and a clear **go / no-go** answer (is Gemini accurate enough, at least 85%
on the Product level, to build the real thing?).

**Total time:** about 4-6 hours spread over a few sittings, including about 2 hours of hand review.
**Total cost:** about $3.5-4 of the $5.00 hard cap, counting the ~$1.21 already spent on the first
run (about $2.4-2.7 still to spend). The code refuses to spend past $5.00.

> The notebooks download the code from the `main` branch on GitHub. Make sure the Phase 1 pull
> request has been merged before you start.

**Only one person and one notebook should spend money at a time.** The $5.00 cap is tracked in a
ledger file on Drive, so two notebooks running paid cells at once could both see the same money left.

### Before you start (one time, about 30 minutes)

1. **Google Cloud project for Gemini.** Marco creates a project (for example
   `cfpb-nlp-classifier`) at [console.cloud.google.com](https://console.cloud.google.com), turns
   on billing, and sets a **$5 budget alert** (Billing > Budgets & alerts). Google's alert only
   emails you; it does not stop spending. The real stop is the $5.00 cap in our code.
2. **Gemini API key.** At [aistudio.google.com/apikey](https://aistudio.google.com/apikey) create a
   key in that project. (The free tier is too slow for this job, so billing must be on.)
3. **Jev API key.** Jev is a small AI checker from TypeSafe AI that answers yes/no questions about
   each result. Create a key at [console.typesafe.ai](https://console.typesafe.ai). It costs cents.
4. **Put both keys in Colab Secrets.** Open any Colab notebook, click the **key icon** in the left
   bar, then "Add new secret":
   - name `GEMINI_API_KEY`, value = your Gemini key
   - name `TYPESAFE_API_KEY`, value = your Jev key
   - switch **Notebook access** on for each.

   Paste only the key itself (no spaces or line breaks). New Gemini keys start with `AQ.` - that is
   normal. Never paste a key into a notebook cell, a chat, or GitHub.
5. **Open the first notebook.** Go to
   [github.com/Dentoneer/cfpb-nlp-classifier](https://github.com/Dentoneer/cfpb-nlp-classifier), open
   the `notebooks` folder and click `00_setup.ipynb`, or go straight to
   `https://colab.research.google.com/github/Dentoneer/cfpb-nlp-classifier/blob/main/notebooks/00_setup.ipynb`.
   Open the other notebooks the same way (change `00_setup` in the address) when their turn comes.

**Whose Drive and keys?** Whoever runs the notebooks uses their own Colab Secrets and their own
Google Drive, and all outputs land in that person's Drive. To work together, the runner shares the
Drive folder `cfpb-nlp-classifier` and the review Sheet with the other person.

**How to run a notebook:** click the play button on each cell and wait for the green check before
the next one. Never add a `pip install` line yourself (see "What to do if a notebook stops"). Do this one cell at a time for 03 and 04, so you read the cost estimate before the
money is spent. The first time, Colab asks to access your Google Drive and Google Sheets. Click
**Allow**.

### Notebook by notebook

Run them in order. More detail and the same table are in [notebooks/README.md](notebooks/README.md).

| # | Notebook | Time | Cost |
|---|----------|------|------|
| 00 | Setup check | 3 min | $0 |
| 01 | Data and splits | 5-15 min | $0 |
| 02 | Baselines (needs GPU) | 45-90 min | $0 |
| 03 | Gemini | 30-90 min | ~$1.8-2.1 |
| 04 | Validators + review sheet | 30-60 min | ~$0.6-0.7 |
| 05 | Results | 5 min | $0 |

**00 Setup check.** Connects Drive, checks your keys and Google Sheets. *Look for:* a line starting
`Ready.`, then `Gemini key works` (a free check) and the spend ledger (about $1.21 after the first run,
$0.00 on a brand-new setup). *If a key is missing or the check prints `PROBLEM`:* see "Key problems"
below.

**After a project update** the setup cell of whichever notebook you open first may say it *moved old
results into an `archive_v1` folder*. That is expected: results from the old data version are put
aside automatically, while the spending ledger, saved Gemini answers and your saved choices stay.
Then run the notebooks 01 → 05 again in order.

**01 Data and splits.** Downloads the complaints, removes noisy rows (including complaints with no
sub-issue: the `no_sub_issue` row), builds the category list (the "taxonomy") and splits the data into
train / tuning / test sets. *Look at:* the filter table (each row says how many complaints a filter
removed; several thousand should be left) and the printed Google Sheet link for the taxonomy. Each
row there has a `label` like `Mortgage|Trouble during payment process|Problem with fees` - the exact
text Gemini answers with. Safe to rerun; finished steps are skipped.

**02 Baselines.** Trains four non-Gemini classifiers that Gemini has to beat. **Needs a GPU:**
before running anything, choose **Runtime > Change runtime type > T4 GPU**. *Look at:* the accuracy
table at the end. Remember the TF-IDF numbers; Gemini must beat them. If Colab disconnects,
reconnect, rerun the first cell and the cell that was running; finished models are loaded from
Drive.

**03 Gemini (money is spent here).**
- *Part 0, category descriptions (about $0.30, once).* Gemini writes a short description of every
  category (what it covers, when to use it, which similar category to use instead) from example
  training complaints. Run the estimate cell, then the cell that writes them. They appear in a new
  `description` column of the taxonomy Google Sheet. *Optional:* if you want to improve some, edit
  only the `description` column in the Sheet (never the `label` column) and run the "use your own
  edits" cell; it checks your edits and saves them. Skip that cell if you made no edits.
- *Part A, model bake-off (up to ~$0.30).* **Skip Parts A and B if you already picked on the first
  run: your saved picks are kept.** Otherwise three Gemini models try the same 50 complaints. Look at
  the `path` column (all three levels right) and `usd`. Type the winner into the pick cell and run it.
  If Google refuses a model, its row says `skipped` with the reason and nothing is charged.
- *Part B, batch size and thinking (~$0.05-0.15).* Keep batch size 10 unless 25 is just as accurate. Pick
  the lowest thinking level that keeps accuracy (`low` or `medium`).
- *Part C, tuning check (about $0.25).* Classifies all 200 tuning complaints with the new prompt and
  compares with the old prompt's saved answers on the same complaints. Look at the `path` column of
  the "same" rows: is the new prompt better?
- *Troubleshoot (optional, free).* Prints your settings, the library version, facts about your key
  (never the key), a free key check, the spend so far and saved answers per run. Send its output to
  Claude if something looks wrong.
- *Part D, the test run (about $1.2-1.5, 15-40 min).* Classifies 800 test complaints. Look at
  accuracy per level and the spend line.

Every paid cell prints an **estimate first**. Read it before continuing.

**04 Validators and the review sheet.** Runs the checks that try to catch Gemini's mistakes (Jev,
repeated runs, agreement with the other models, label-noise detection) and builds your review
Google Sheet. Paid steps cost about $0.65 in total (self-consistency reruns ~$0.5, Jev and the
summary judge ~$0.15). At the end it prints the **link to the
review sheet**. Then you stop (see below).

**05 Results.** Reads your review, combines everything, prints the scoreboards and the go/no-go
decision. Free, a few minutes.

### What to do if a notebook stops

- **"Budget reached ..." or "would need about $X but only $Y of the $5.00 cap is left".** Stop. The $5.00 cap did its job. Nothing is lost.
  Tell Claude what the ledger says; do not try to get around it.
- **"daily request quota is used up" (or "daily Gemini request quota is used up").** Google limits how many requests you can send per day. Rerun the same
  cell tomorrow. Finished rows are cached, so you do not pay again.
- **"... is overloaded or not answering ... rerun this cell later".** Gemini or Jev was busy and
  the code gave up after a few tries. Nothing was charged for those tries. Wait 15-60 minutes and
  rerun the same cell; finished rows are saved.
- **"Restart session (Runtime menu > Restart session), then rerun this cell".** Do exactly that.
  The code on GitHub changed while your session was open. Saved results and choices on Drive are kept.
- **Colab disconnected or timed out.** Reconnect, rerun the first (setup) cell, then the cell that
  was running. Everything finished is already on Drive.
- **Key problems** (`Gemini rejected the API key`, `API key not valid`, or the key check in 00 prints
  `PROBLEM`). Keys starting with `AQ.` are the normal new format - that is not the problem. Fix:
  1. Open [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and copy the key again
     (use the copy button; make sure it is the key of the project with billing on).
  2. In Colab click the **key icon**, delete the `GEMINI_API_KEY` secret, add it again and paste
     the key with no spaces or line breaks. Switch **Notebook access** on.
  3. **Runtime → Restart session**, then run the notebook from the top. Notebook 00's key check
     should say `Gemini key works`.
- **"... skipped: model retired, remove it from the list" / "does not support that thinking
  level".** Google refused that one model or setting; nothing was charged and the other rows still
  ran. Ignore that row (or tell Claude so the list gets updated).
- **"Run Part 0 (descriptions) in notebook 03 first."** Open notebook 03 and run Part 0 (the
  category descriptions), then come back.
- **Never pip-install Google libraries in Colab** (no `pip install google-genai`, `-U`,
  `--upgrade`, and do not click "update" suggestions). Colab's own versions work; upgrading one in
  the middle of a session broke Google's login library on the first run. If you did: **Runtime →
  Disconnect and delete runtime**, then start the notebook again from the top.
- **Drive says "credential propagation was unsuccessful".** When Colab asks for Google access, tick
  **every** permission box (Drive and Sheets), and allow pop-ups for colab.research.google.com. Then
  rerun the setup cell.
- **`KeyboardInterrupt`.** A cell was stopped (you clicked stop, or Colab did). Nothing is lost:
  rerun the same cell; finished work is reused, not paid again.
- **"Restart session" vs "Disconnect and delete runtime".** *Restart session* (Runtime menu) clears
  Python's memory but keeps installed files - use it when a message asks for it or after code
  changes. *Disconnect and delete runtime* gives you a brand-new machine - use it only if a library
  install went wrong; then rerun from the setup cell. Your results on Drive are kept either way.
- **The "Save in GitHub" / "save a copy" banner.** Ignore it; do not click it. The notebooks run
  straight from GitHub and save everything to Drive; nothing needs saving to GitHub.
- **You edited a cell** (for example typed your pick). Run that edited cell before running the cells
  after it; later cells only see an edit once its cell has run.
- **A red error you do not understand.** Run notebook 03's Troubleshoot cell (free), then copy its
  output and the last 10 lines of the error and send them to Claude. Do not rerun paid cells over
  and over.

### The review-sheet stop (about 2 hours)

After notebook 04, open the Google Sheet link it printed. It has about 200 complaints. For each:

1. Read the complaint, the CFPB's category, and Gemini's category.
2. In the verdict dropdown for each level choose: `model right`, `label right`, `both OK`,
   `both wrong`, or `ambiguous` (cannot tell).
3. Mark whether Gemini's summary is OK: `yes` or `no`.

Rows where Gemini and the CFPB agree are pre-filled with `both OK`; check a few anyway. Leave no
blank cells. Your answers matter because CFPB labels are not perfect: sometimes Gemini is right
and the label is wrong. Your review turns that into a fair accuracy number. You can stop and
continue later. Then run notebook 05.

### Where the results are

Google Drive: `MyDrive/cfpb-nlp-classifier/phase1/`

- `results.md` - the final summary (start here)
- `*.parquet` - every table, including the two leaderboards and adjusted accuracy
- `ledger.jsonl` - every paid charge; this is how the $5.00 cap is enforced
- `gemini_cache.jsonl`, `judge_cache.jsonl` - saved answers, so reruns are free
- `models/` and `config_overrides.json` - trained models and your bake-off choices
- `taxonomy_url.txt`, `review_sheet_url.txt` - the Google Sheet links
- `descriptions.parquet` - the category descriptions Gemini wrote (and your edits)
- `data_version.txt` and `archive_v1/` - which data version the results belong to, and the
  first run's results, set aside automatically

### How to read the results and the go/no-go

Notebook 05 prints a decision with reasons. It is **GO** only if both are true:

1. **Adjusted Product accuracy is at least 85%.** "Adjusted" means Gemini's accuracy after your
   hand review, scaled up to represent all 800 test complaints.
2. **Gemini beats TF-IDF** (the simple word-count baseline) at every level, on accuracy and
   macro-F1 (accuracy that treats rare categories fairly), and the accuracy gap is statistically
   real (McNemar test, p < 0.05).

The **classifier leaderboard** ranks the five models. The **validator leaderboard** ranks the
checks by how well each catches Gemini's mistakes. The best validators feed the final
`verified` / `likely_correct` / `needs_review` labels. A NO-GO is still a useful result: it tells us
what to fix before Phase 2. Share `results.md` with Claude to plan Phase 2.

## For developers

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; on Mac/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                        # fast tests only, no keys needed
pytest -m slow                   # slow tests: need API keys and/or torch
ruff check .                     # lint
python -m nlp.tools.check_notebooks notebooks/   # every nlp.* name used in notebooks exists
```

- All logic lives in the `nlp/` package; notebooks only call it.
- Slow tests (`-m slow`) call real services or load models. `test_live_jev_call` needs
  `TYPESAFE_API_KEY` and confirms the shape of Jev's response on the first live call.
- Work on a branch and open a pull request. CI runs ruff, pytest and the notebook check. Merging
  needs one approval from Marco (the admin can bypass).
- Never commit keys, raw data, Parquet or model files.

## How we work

- `main` is protected by the `protect-main` ruleset: collaborators work on a branch and open a
  pull request, which needs one approval from the repo owner before merging. The repo admin can
  bypass. No force-pushes or branch deletion on `main`.
- CI (GitHub Actions) runs lint (`ruff`) and tests (`pytest`) on every pull request and every
  push to `main`.
- This repo is **public**. Secrets (Gemini API key, etc.) live in Colab Secrets / GitHub
  Secrets — never in code. GitHub secret scanning with push protection is on and blocks pushes
  containing keys.
