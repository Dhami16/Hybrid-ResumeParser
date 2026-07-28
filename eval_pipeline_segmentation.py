"""
eval_pipeline_segmentation.py — pipeline-level A/B eval of section segmentation

eval_model.py only measures the frozen spaCy model's raw NER accuracy on
dev.spacy's text -- it never calls IndustrialParser.parse() or
segment_resume(), so it can't detect any effect from the new section-scoping
logic in parser_engine.py. This script runs the FULL pipeline (all five
tiers, with vs. without segmentation) against dev.spacy's 165 real resumes
and scores the final NAME/UNIVERSITY output against the gold entities,
using exact-match precision/recall/F1 per field:

    precision = (# correct predictions) / (# non-empty predictions)
    recall    = (# correct predictions) / (# docs with a gold entity)

Usage (from project root):
    python eval_pipeline_segmentation.py
"""

import sys
import pathlib

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "files"))

import spacy                                   # noqa: E402
from spacy.tokens import DocBin                # noqa: E402

import parser_engine                           # noqa: E402
from parser_engine import IndustrialParser      # noqa: E402


def load_gold_examples():
    nlp_blank = spacy.blank("en")
    db = DocBin().from_disk(str(ROOT / "dev.spacy"))
    docs = list(db.get_docs(nlp_blank.vocab))

    examples = []
    for doc in docs:
        gold_name, gold_uni = None, None
        for ent in doc.ents:
            if ent.label_ == "NAME" and gold_name is None:
                gold_name = ent.text.strip()
            if ent.label_ == "UNIVERSITY" and gold_uni is None:
                gold_uni = ent.text.strip()
        examples.append((doc.text, gold_name, gold_uni))
    return examples


def score_field(rows, gold_idx, pred_idx):
    correct = predicted = gold_count = 0
    for row in rows:
        gold, pred = row[gold_idx], row[pred_idx]
        if gold is not None:
            gold_count += 1
        if pred is not None:
            predicted += 1
            if gold is not None and pred.strip() == gold.strip():
                correct += 1
    precision = correct / predicted if predicted else 0.0
    recall = correct / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def run_pipeline(parser, examples, segmentation_enabled):
    original_segment_resume = parser_engine.segment_resume
    if not segmentation_enabled:
        parser_engine.segment_resume = lambda text: {}  # forces full-document fallback everywhere

    rows = []
    try:
        for text, gold_name, gold_uni in examples:
            result = parser.parse(text)
            pred_name = result["PERSONAL"]["NAME"]["value"]
            pred_uni = result["PERSONAL"]["UNIVERSITY"]["value"]
            pred_name = None if pred_name == "Not Found" else pred_name
            pred_uni = None if pred_uni == "Not Found" else pred_uni
            rows.append((gold_name, pred_name, gold_uni, pred_uni))
    finally:
        parser_engine.segment_resume = original_segment_resume

    return rows


def main():
    examples = load_gold_examples()
    parser = IndustrialParser(
        model_path=str(ROOT / "files" / "model-best"),
        entity_patterns_path=str(ROOT / "files" / "entity_patterns.json"),
    )

    print(f"Evaluating full IndustrialParser.parse() pipeline on {len(examples)} real resumes (dev.spacy)\n")

    for label, enabled in [("WITHOUT segmentation (pre-change behavior)", False),
                            ("WITH segmentation (new behavior)", True)]:
        rows = run_pipeline(parser, examples, enabled)
        name_p, name_r, name_f = score_field(rows, 0, 1)
        uni_p, uni_r, uni_f = score_field(rows, 2, 3)

        print(f"=== {label} ===")
        print(f"  {'LABEL':<14}{'PRECISION':>10}{'RECALL':>10}{'F1':>10}")
        print(f"  {'NAME':<14}{name_p:>10.4f}{name_r:>10.4f}{name_f:>10.4f}")
        print(f"  {'UNIVERSITY':<14}{uni_p:>10.4f}{uni_r:>10.4f}{uni_f:>10.4f}")
        print()


if __name__ == "__main__":
    main()
