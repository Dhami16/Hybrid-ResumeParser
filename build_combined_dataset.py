"""
build_combined_dataset.py — combine the converted DataTurks real-resume
examples with our synthetic generator to produce three DocBin datasets:

  1. train.spacy         — real_train + synthetic, synthetic capped at ~70%
  2. dev.spacy            — real_dev only (real data "primarily into dev" so
                             it actually validates generalization to real
                             resumes, not just our own synthetic templates)
  3. real_only_train.spacy — real_train only, no synthetic (for the
                             real-data-only model variant)

Real examples are split 25% train / 75% dev: most of the real data goes to
dev (where it matters most for measuring real-world generalization), while
still keeping some in train so the model sees real phrasing during training.

Usage:
    python build_combined_dataset.py
"""

import json
import pathlib
import random

import spacy
from spacy.tokens import DocBin

from prepare_data import NAMES, UNIVERSITIES, NEGATIVE_POOLS, _split_pool, generate_dataset

ROOT = pathlib.Path(__file__).parent
REAL_EXAMPLES_PATH = ROOT / "external_data" / "dataturks_converted.json"
REAL_TRAIN_RATIO = 0.25
SEED = 42
MAX_SYNTHETIC_FRACTION = 0.70


def to_docbin(examples, nlp):
    db = DocBin()
    skipped = 0
    for text, entities in examples:
        doc = nlp.make_doc(text)
        spans = []
        for start, end, label in entities:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                skipped += 1
                continue
            spans.append(span)
        try:
            doc.ents = spans
        except ValueError:
            skipped += 1
            continue
        db.add(doc)
    return db, skipped


def main():
    rng = random.Random(SEED)

    with open(REAL_EXAMPLES_PATH, encoding="utf-8") as fh:
        real_examples = json.load(fh)
    real_examples = [(text, [tuple(e) for e in ents]) for text, ents in real_examples]

    # The source dataset has one exact-duplicate resume; drop the repeat so
    # the same resume can't end up in both train and dev (which would leak
    # the "held-out" dev score and defeat the point of evaluating on it).
    seen_texts = set()
    deduped = []
    for text, ents in real_examples:
        if text in seen_texts:
            continue
        seen_texts.add(text)
        deduped.append((text, ents))
    real_examples = deduped

    rng.shuffle(real_examples)

    cut = int(len(real_examples) * REAL_TRAIN_RATIO)
    real_train = real_examples[:cut]
    real_dev = real_examples[cut:]

    # Cap synthetic at MAX_SYNTHETIC_FRACTION of the final train set:
    # synthetic / (synthetic + real_train) <= MAX_SYNTHETIC_FRACTION
    # => synthetic <= real_train * MAX_SYNTHETIC_FRACTION / (1 - MAX_SYNTHETIC_FRACTION)
    synthetic_count = int(len(real_train) * MAX_SYNTHETIC_FRACTION / (1 - MAX_SYNTHETIC_FRACTION))

    # Reuse the same held-out name/university/negative split Phase 3 used
    # (same seed) so this stays consistent with the existing synthetic pipeline.
    train_names, _ = _split_pool(NAMES, 0.8, random.Random(SEED))
    train_unis, _ = _split_pool(UNIVERSITIES, 0.8, random.Random(SEED))
    train_negatives = {}
    for key, pool in NEGATIVE_POOLS.items():
        tr, _ = _split_pool(pool, 0.8, random.Random(SEED))
        train_negatives[key] = tr

    synthetic_examples = generate_dataset(train_names, train_unis, train_negatives, synthetic_count, rng)

    combined_train = real_train + synthetic_examples
    rng.shuffle(combined_train)

    nlp = spacy.blank("en")

    train_db, train_skipped = to_docbin(combined_train, nlp)
    dev_db, dev_skipped = to_docbin(real_dev, nlp)
    real_only_db, real_only_skipped = to_docbin(real_train, nlp)

    train_db.to_disk(ROOT / "train.spacy")
    dev_db.to_disk(ROOT / "dev.spacy")
    real_only_db.to_disk(ROOT / "real_only_train.spacy")

    synthetic_fraction = len(synthetic_examples) / len(combined_train)

    print("=" * 60)
    print("  Combined dataset build report")
    print("=" * 60)
    print(f"  Real examples total:        {len(real_examples)}")
    print(f"  Real -> train:              {len(real_train)}  ({REAL_TRAIN_RATIO:.0%} of real)")
    print(f"  Real -> dev:                {len(real_dev)}  ({1 - REAL_TRAIN_RATIO:.0%} of real)")
    print(f"  Synthetic examples added:   {len(synthetic_examples)}")
    print(f"  Combined train.spacy total: {len(combined_train)}  (synthetic = {synthetic_fraction:.1%})")
    print(f"  dev.spacy (real only):      {len(real_dev)}")
    print(f"  real_only_train.spacy:      {len(real_train)}")
    print(f"  Entity spans skipped (alignment): train={train_skipped} dev={dev_skipped} real_only={real_only_skipped}")


if __name__ == "__main__":
    main()
