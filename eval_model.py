"""
eval_model.py — Phase 2 evaluation baseline

Runs spaCy's built-in evaluator against the trained model in files/model-best
using dev.spacy, and reports precision/recall/F1 per entity label (NAME,
UNIVERSITY) plus the overall NER score.

Usage (from project root):
    python eval_model.py
"""

import json
import pathlib

from spacy.cli.evaluate import evaluate

ROOT = pathlib.Path(__file__).parent
MODEL_PATH = ROOT / "files" / "model-best"
DEV_DATA = ROOT / "dev.spacy"


def main():
    scores = evaluate(
        model=str(MODEL_PATH),
        data_path=DEV_DATA,
        silent=True,
    )

    print("=" * 60)
    print("  Phase 2 — Evaluation Baseline")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Dev data: {DEV_DATA}")
    print("=" * 60)

    print(f"\nOverall NER — P: {scores['ents_p']:.4f}  R: {scores['ents_r']:.4f}  F: {scores['ents_f']:.4f}")

    print("\nPer-label breakdown:")
    print(f"  {'LABEL':<14}{'PRECISION':>10}{'RECALL':>10}{'F1':>10}")
    for label, m in scores["ents_per_type"].items():
        print(f"  {label:<14}{m['p']:>10.4f}{m['r']:>10.4f}{m['f']:>10.4f}")

    out_path = ROOT / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "ents_p": scores["ents_p"],
                "ents_r": scores["ents_r"],
                "ents_f": scores["ents_f"],
                "ents_per_type": scores["ents_per_type"],
            },
            fh,
            indent=2,
        )
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
