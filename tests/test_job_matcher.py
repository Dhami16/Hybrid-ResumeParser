"""Tests for job_matcher.py: JD tier parsing, catalog-gap detection, and scoring."""

import pytest

from job_matcher import parse_job_description, score_match


def test_parse_job_description_good_match(job_descriptions_dir):
    text = (job_descriptions_dir / "jd_good_match.txt").read_text(encoding="utf-8")
    tiers = parse_job_description(text)

    assert {"AWS", "Docker", "Python", "SQL"}.issubset(tiers["required"]["skills"])
    assert {"Kubernetes", "Redis"}.issubset(tiers["bonus"]["skills"])
    assert "Git" in tiers["nice_to_have"]["skills"]
    # AWS sub-services aren't in the catalog as separate entries -- should be
    # surfaced as unrecognized, not silently dropped.
    assert "Lambda" in tiers["required"]["unrecognized_terms"] or "S3" in tiers["required"]["unrecognized_terms"]


def test_parse_job_description_partial_match_catches_react(job_descriptions_dir):
    """
    Regression test: before the skill catalog was expanded to include
    Frontend terms, "React" wasn't in SKILL_CATEGORIES at all and was
    invisible to the scorer -- it wouldn't appear as matched, missing, OR
    unrecognized. It must now appear as a recognized (if unmatched) skill.
    """
    text = (job_descriptions_dir / "jd_partial_match.txt").read_text(encoding="utf-8")
    tiers = parse_job_description(text)
    assert "React" in tiers["required"]["skills"]
    assert "REST APIs" in tiers["required"]["unrecognized_terms"]


def test_parse_job_description_poor_match(job_descriptions_dir):
    text = (job_descriptions_dir / "jd_poor_match.txt").read_text(encoding="utf-8")
    tiers = parse_job_description(text)
    assert "Communication" in tiers["required"]["skills"]
    assert {"Photoshop", "Illustrator", "Figma", "InDesign"}.issubset(tiers["required"]["skills"])


def test_score_match_weights_tiers_correctly():
    jd_tiers = {
        "required":     {"skills": {"Python", "SQL"}, "unrecognized_terms": []},
        "bonus":        {"skills": {"Docker"}, "unrecognized_terms": []},
        "nice_to_have": {"skills": {"Git"}, "unrecognized_terms": []},
    }
    # Candidate has everything except the nice-to-have.
    resume_skills = {"Python", "SQL", "Docker"}
    result = score_match(resume_skills, jd_tiers)

    # required: 2 skills * 3 = 6, bonus: 1 * 2 = 2, nice_to_have: 1 * 1 = 1 -> max 9
    assert result["max_points"] == 9
    assert result["earned_points"] == 8
    assert result["score"] == pytest.approx(8 / 9)
    assert result["matched"]["required"] == ["Python", "SQL"]
    assert result["missing"]["nice_to_have"] == ["Git"]


def test_score_match_handles_jd_with_no_recognized_skills():
    """No division-by-zero crash when a JD has zero catalog-recognized skills."""
    jd_tiers = {
        "required":     {"skills": set(), "unrecognized_terms": ["SomeNicheTool"]},
        "bonus":        {"skills": set(), "unrecognized_terms": []},
        "nice_to_have": {"skills": set(), "unrecognized_terms": []},
    }
    result = score_match({"Python"}, jd_tiers)
    assert result["score"] == 0.0
    assert result["max_points"] == 0
