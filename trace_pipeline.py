"""
trace_pipeline.py — Phase 2 pipeline diagnostic

Runs the full IndustrialParser.parse() five-tier pipeline against every
sample resume in resumes/ and prints which tier (EntityRuler / SpaCy-NER /
Regex / Heuristic / LLM-Gemini / unresolved) resolved each field, so weak
spots in the current pipeline are visible at a glance.

Usage (from project root):
    python trace_pipeline.py
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).parent
FILES_DIR = ROOT / "files"
RESUMES_DIR = ROOT / "resumes"

sys.path.insert(0, str(FILES_DIR))
from parser_engine import IndustrialParser, extract_text_from_path  # noqa: E402

SOURCE_TAG = {
    "EntityRuler": "[ENTITYRULER]",
    "SpaCy-NER":   "[NER]",
    "Regex":       "[REGEX]",
    "Heuristic":   "[HEURISTIC]",
    "LLM-Gemini":  "[LLM-GEMINI]",
    "—":           "[UNRESOLVED]",
}


def main():
    parser = IndustrialParser(
        model_path=str(FILES_DIR / "model-best"),
        entity_patterns_path=str(FILES_DIR / "entity_patterns.json"),
    )

    if parser.model_error:
        print(f"WARNING: {parser.model_error}\n")

    sample_files = sorted(
        f for f in RESUMES_DIR.iterdir() if f.suffix.lower() in (".pdf", ".txt")
    )
    if not sample_files:
        print(f"No .pdf/.txt files found in {RESUMES_DIR}")
        return

    tier_counts: dict[str, int] = {}

    for path in sample_files:
        print("=" * 72)
        print(f"  {path.name}")
        print("=" * 72)

        try:
            text = extract_text_from_path(str(path))
        except ValueError as exc:
            print(f"  SKIPPED — {exc}\n")
            continue

        result = parser.parse(text)

        for section in ("PERSONAL", "CONTACT"):
            for field, obj in result[section].items():
                tag = SOURCE_TAG.get(obj["source"], SOURCE_TAG["—"])
                print(f"  {field:<12} {obj['value']:<40} {tag}")
                tier_counts[obj["source"]] = tier_counts.get(obj["source"], 0) + 1

        matched_skills = sum(len(v) for v in result["SKILLS"].values())
        print(f"  {'SKILLS':<12} {matched_skills} matched across {len(result['SKILLS'])} categories")

        if result["_meta"]["llm_used"]:
            print("  -> Gemini LLM tier was invoked for this resume")
        print()

    print("=" * 72)
    print("  Tier usage summary (across all PERSONAL/CONTACT fields, all resumes)")
    print("=" * 72)
    total = sum(tier_counts.values())
    for source, tag in SOURCE_TAG.items():
        count = tier_counts.get(source, 0)
        pct = (count / total * 100) if total else 0
        print(f"  {tag:<16}{count:>4}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
