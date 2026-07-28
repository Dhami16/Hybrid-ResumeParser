"""
test_job_matching.py — job-description match scoring test cases

Runs the weighted skill-overlap job matcher (files/job_matcher.py) against
resumes/test_resume.txt for three job descriptions of varying expected fit
(good/partial/poor), as a sanity check that the scoring behaves as expected.

Usage (from project root):
    python test_job_matching.py
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "files"))

from parser_engine import IndustrialParser, extract_text_from_path  # noqa: E402
from job_matcher import parse_job_description, score_match          # noqa: E402

JD_FILES = ["jd_good_match.txt", "jd_partial_match.txt", "jd_poor_match.txt"]
RESUME_PATH = ROOT / "resumes" / "test_resume.txt"
JD_DIR = ROOT / "job_descriptions"


def main():
    parser = IndustrialParser(
        model_path=str(ROOT / "files" / "model-best"),
        entity_patterns_path=str(ROOT / "files" / "entity_patterns.json"),
    )

    resume_text = extract_text_from_path(str(RESUME_PATH))
    result = parser.parse(resume_text)
    resume_skills = {skill for cat in result["SKILLS"].values() for skill in cat}

    print(f"Candidate: {RESUME_PATH.name}")
    print(f"Extracted skills ({len(resume_skills)}): {', '.join(sorted(resume_skills))}\n")

    for jd_file in JD_FILES:
        jd_path = JD_DIR / jd_file
        jd_text = jd_path.read_text(encoding="utf-8")
        jd_tiers = parse_job_description(jd_text)
        outcome = score_match(resume_skills, jd_tiers)

        print("=" * 72)
        print(f"  {jd_file}")
        print(f"  Score: {outcome['score']:.1%}  ({outcome['earned_points']}/{outcome['max_points']} weighted points)")
        print("=" * 72)
        for tier in ("required", "bonus", "nice_to_have"):
            print(f"  {tier.upper():<14} matched: {', '.join(outcome['matched'][tier]) or '(none)'}")
            print(f"  {'':<14} missing: {', '.join(outcome['missing'][tier]) or '(none)'}")
            unrecognized = jd_tiers[tier]["unrecognized_terms"]
            print(f"  {'':<14} unrecognized (not evaluated): {', '.join(unrecognized) or '(none)'}")
        print()


if __name__ == "__main__":
    main()
