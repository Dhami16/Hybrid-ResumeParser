# Hybrid AI Resume Parser

**Live demo: [hybrid-resume-parser.streamlit.app](https://hybrid-resume-parser.streamlit.app/)**

A resume parser built around a five-tier confidence hierarchy — a custom-trained
spaCy NER model backed by an EntityRuler, deterministic regex extraction,
positional/keyword heuristics, and an LLM (Gemini) fallback for the fields
local extraction can't resolve — plus a lightweight rule-based job-description
matcher, all served through a Streamlit dashboard.

## Architecture

Each tier runs in order; a tier is skipped for a field once a higher-confidence
tier has already filled it, so most resumes never need to touch the LLM tier
at all.

| Tier | What it does |
|---|---|
| 1. EntityRuler | Verified name/institution patterns loaded from `entity_patterns.json` — data-driven, no hard-coded identities |
| 2. spaCy NER | Custom-trained model (`files/model-best`) predicting NAME and UNIVERSITY from context |
| 3. Regex | Deterministic extraction: email, phone, GPA, graduation year, GitHub/LinkedIn URLs, degree |
| 4. Heuristic | Positional (name is usually the first line) and keyword-based (university line contains "Institute"/"University"/etc.) fallback |
| 5. LLM (Gemini) | Called only when NAME or UNIVERSITY is still unresolved after tiers 1–4 — one API call per resume, at most |

Skill detection runs a whole-word, alias-aware match (e.g. "ml"/"deep learning"
→ "Machine Learning") against a categorized skill catalogue, scoped to the
resume's Skills/Projects sections where section detection succeeds.

The job-matching feature (`files/job_matcher.py`) parses a pasted job
description into Required/Bonus/Nice-to-have skill tiers, scores the
candidate's extracted skills against it with tier-weighted overlap, and
separately surfaces JD terms outside the skill catalogue so the score's
coverage limits stay visible rather than silently hidden.

## Running it

```bash
cd files
python -m venv ../.venv          # if you don't already have one
../.venv/Scripts/Activate.ps1    # Windows PowerShell; use `source ../.venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

The full trained NER model (`files/model-best`) isn't in this repo (see below).
A smaller model, `files/model-best-lite` (74MB, `en_core_web_md` vectors
instead of `en_core_web_lg`), *is* committed specifically so the [live
demo](https://hybrid-resume-parser.streamlit.app/) has real NER instead of
running degraded — `app.py` prefers `model-best` when present locally and
falls back to `model-best-lite` otherwise. You'll get slightly better
NAME/UNIVERSITY accuracy training your own full-size model:

```bash
python prepare_data.py                    # generates train.spacy / dev.spacy
python -m spacy train config.cfg --output ./output \
    --paths.train ./train.spacy --paths.dev ./dev.spacy
cp -r output/model-best files/model-best
```

Gemini API key (optional, enables the Tier 5 fallback): paste it into the
sidebar at runtime, or set `GEMINI_API_KEY` in `.streamlit/secrets.toml`
(see `files/secrets.toml.template`).

## What's not in this repo

A few things are intentionally excluded (see `.gitignore`):

- **Trained model artifacts** (`files/model-best/`, `output*/`) — some
  individual files exceed GitHub's 100MB limit, and all of it is reproducible
  from the scripts above.
- **Real annotated resume data** (`external_data/`, `train.spacy`, `dev.spacy`)
  — a subset of training/eval data comes from a real-world annotated dataset
  containing actual people's names, emails, and phone numbers (see
  `DATA_SOURCES.md` for provenance and licensing notes). Regenerate via
  `convert_dataturks.py` + `build_combined_dataset.py` if you have your own
  copy of that dataset.
- `archive/` — superseded development iterations and model backups, kept
  locally for rollback, not needed to run or understand the current app.

## Project layout

```
files/               the actual app: app.py, parser_engine.py, job_matcher.py
resumes/              sample resumes used for manual testing
job_descriptions/      sample job descriptions used for testing the matcher
prepare_data.py         synthetic NER training data generator
convert_dataturks.py    converts the real annotated dataset (see DATA_SOURCES.md)
build_combined_dataset.py  merges real + synthetic data into train/dev sets
eval_model.py           spaCy-level NER precision/recall/F1 against dev.spacy
eval_pipeline_segmentation.py  full-pipeline eval (exercises parser_engine.py, not just the raw model)
trace_pipeline.py       shows which tier resolved each field, per sample resume
test_job_matching.py    sanity-checks the job matcher against sample JDs
```
