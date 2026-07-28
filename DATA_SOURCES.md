# External data sources

## DataTurks "Resume Entities for NER" (`external_data/dataturks_resume_ner/`)

- **What**: 220 manually-annotated resumes (`traindata.json` = 200, `testdata.json` = 20),
  JSON-lines format, entities including Name, College Name, Skills, Designation,
  Companies worked at, Degree, Graduation Year, Email Address, Location, Years of
  Experience.
- **Origin**: Originally published by DataTurks (dataturks.com, now defunct) as a
  Kaggle dataset (`dataturks/resume-entities-for-ner`) and mirrored in DataTurks'
  own GitHub org
  ([DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy](https://github.com/DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy)).
  Downloaded directly from the GitHub mirror (no Kaggle authentication required —
  the files are identical to the Kaggle listing).
- **License: UNSPECIFIED.** Kaggle's own page metadata lists the license as
  `"Unknown"` and the GitHub source repo has no LICENSE file. Used here for
  non-commercial academic/research model training only. If this repository is
  ever made public, the raw `traindata.json`/`testdata.json` files should not be
  redistributed given the unclear licensing — regenerate or omit them rather than
  committing them to a public remote.
- **Used for**: augmenting/validating the custom spaCy NER model's NAME and
  UNIVERSITY labels (only the "Name" and "College Name" annotations are used;
  other label types are ignored since they're already handled by regex/heuristic
  tiers in `parser_engine.py`).
