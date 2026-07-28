"""
Regression tests for parser_engine.py. Several of these pin down bugs found
and fixed during development -- see the comment on each for what it guards
against.
"""

import io

import pytest

from parser_engine import (
    IndustrialParser,
    SKILL_MAP,
    _extract_page_text_column_aware,
    _looks_like_bad_name,
    _looks_like_contact_info,
    _looks_like_university,
    extract_text_from_bytes,
    extract_text_from_path,
    segment_resume,
)


# ── Full-pipeline regression tests on real sample files ────────────────────

def test_entityruler_resolves_seeded_patterns(parser, resumes_dir):
    result = parser.parse(extract_text_from_path(str(resumes_dir / "test_resume.txt")))
    assert result["PERSONAL"]["NAME"] == {"value": "RAHUL V.", "source": "EntityRuler"}
    assert result["PERSONAL"]["UNIVERSITY"] == {"value": "Thapar Institute", "source": "EntityRuler"}


def test_regex_tier_extracts_contact_fields(parser, resumes_dir):
    result = parser.parse(extract_text_from_path(str(resumes_dir / "test_resume.txt")))
    contact = result["CONTACT"]
    assert contact["EMAIL"]["value"] == "rahul.v_test@email.com"
    assert contact["GITHUB"]["value"] == "https://github.com/rahul-v-dev"
    assert contact["LINKEDIN"]["value"] == "https://linkedin.com/in/rahulv"
    assert result["PERSONAL"]["DEGREE"]["value"] == "B.Tech"


def test_ner_resolves_name_on_real_resume(parser, resumes_dir):
    result = parser.parse(extract_text_from_path(str(resumes_dir / "demo_meera_resume.txt")))
    assert result["PERSONAL"]["NAME"]["value"] == "Meera Chandrasekaran"
    assert result["PERSONAL"]["UNIVERSITY"]["value"] == "Vellore Institute of Technology"


@pytest.mark.parametrize(
    "filename,expected_name",
    [
        ("resume.pdf", "Jiaming Chen"),
        ("computer-science-resume-example.pdf", "BELLA TREVINO"),
    ],
)
def test_pdf_extraction_puts_name_on_first_line(resumes_dir, filename, expected_name):
    """
    Regression test for the column-detection straddle-check: a single-column
    PDF with a centered name header (varying x0 from indentation, not real
    columns) must not have its reading order scrambled. Without the check,
    resume.pdf's name previously ended up ~75 lines into the extracted text
    instead of on line 1.
    """
    path = resumes_dir / filename
    if not path.exists():
        pytest.skip(f"{filename} not present (may be gitignored)")
    text = extract_text_from_bytes(path.read_bytes(), filename)
    first_line = next(ln for ln in text.split("\n") if ln.strip())
    assert first_line.strip() == expected_name


def test_multicolumn_pdf_university_not_garbled(parser, resumes_dir):
    """
    Regression test for the original bug this fix targeted: fitz's sort=True
    interleaved "Chicago, IL" (sidebar column) directly into "University of
    Illinois Chicago" (main column), producing a garbled UNIVERSITY value.
    Column-aware block reordering should keep them as separate clean lines.
    """
    path = resumes_dir / "computer-science-resume-example.pdf"
    if not path.exists():
        pytest.skip("computer-science-resume-example.pdf not present")
    text = extract_text_from_bytes(path.read_bytes(), path.name)
    result = parser.parse(text)
    assert result["PERSONAL"]["UNIVERSITY"]["value"] == "University of Illinois Chicago"


# ── Section segmentation ────────────────────────────────────────────────────

def test_segment_resume_splits_known_sections(resumes_dir):
    text = extract_text_from_path(str(resumes_dir / "test_resume.txt"))
    sections = segment_resume(text)
    assert "Thapar Institute" in sections["Education"]
    assert "AI Intern" in sections["Experience"]
    assert "Resume Parser Project" in sections["Projects"]
    assert "Python" in sections["Skills"]


def test_segment_resume_allcaps_name_not_mistaken_for_header(resumes_dir):
    """
    Regression test: an all-caps candidate name on the first line ("RAHUL
    V.") was originally misdetected as an unrecognized section header,
    emptying the Header section entirely.
    """
    text = extract_text_from_path(str(resumes_dir / "test_resume.txt"))
    sections = segment_resume(text)
    assert sections["Header"].startswith("RAHUL V.")


def test_segment_resume_unknown_header_routes_to_other():
    """
    An all-caps header with no matching keyword (e.g. "CERTIFICATIONS")
    still starts a section boundary once we're past the intro -- it just
    routes to "Other" instead of corrupting whichever section preceded it.
    """
    text = (
        "John Smith\nEDUCATION\nMassachusetts Institute of Technology\n"
        "CERTIFICATIONS\nAWS Certified Solutions Architect"
    )
    sections = segment_resume(text)
    assert "Massachusetts Institute of Technology" in sections["Education"]
    assert "AWS Certified Solutions Architect" in sections["Other"]


# ── Shield / plausibility helpers ───────────────────────────────────────────

@pytest.mark.parametrize("value", ["WORK EXPERIENCE", "California,", "Docker", "linkedin.com/in/x"])
def test_looks_like_bad_name_rejects_non_names(value):
    assert _looks_like_bad_name(value) is True


def test_looks_like_bad_name_accepts_real_name():
    assert _looks_like_bad_name("Jiaming Chen") is False


@pytest.mark.parametrize("value", ["Thapar Institute", "MIT University", "Some College"])
def test_looks_like_university_accepts_institution_keywords(value):
    assert _looks_like_university(value) is True


def test_looks_like_university_rejects_plain_text():
    assert _looks_like_university("Web Developer") is False


@pytest.mark.parametrize(
    "value",
    ["test@example.com", "https://github.com/x", "linkedin.com/in/x", "123-456-7890"],
)
def test_looks_like_contact_info(value):
    assert _looks_like_contact_info(value) is True


# ── Skill alias normalization ────────────────────────────────────────────────

def test_skill_map_covers_common_aliases():
    assert "ml" in SKILL_MAP["Machine Learning"]
    assert "postgres" in SKILL_MAP["PostgreSQL"]
    assert "cpp" in SKILL_MAP["C++"]


def test_pipeline_normalizes_skill_aliases(parser):
    result = parser.parse("Test Person\nExperienced in ml and postgres and cpp.\n")
    all_skills = {s for cat in result["SKILLS"].values() for s in cat}
    assert "Machine Learning" in all_skills
    assert "PostgreSQL" in all_skills
    assert "C++" in all_skills


# ── File extraction edge cases ───────────────────────────────────────────────

def test_extract_text_from_bytes_txt_utf8():
    text = extract_text_from_bytes("Hello résumé".encode("utf-8"), "test.txt")
    assert text == "Hello résumé"


def test_extract_text_from_bytes_txt_latin1_fallback():
    raw = "Hello café".encode("latin-1")
    text = extract_text_from_bytes(raw, "test.txt")
    assert "Hello caf" in text


def test_extract_text_from_bytes_image_only_pdf_raises():
    import fitz

    doc = fitz.open()
    doc.new_page()  # blank page, no text
    blank_pdf_bytes = doc.tobytes()
    doc.close()

    with pytest.raises(ValueError, match="image-only"):
        extract_text_from_bytes(blank_pdf_bytes, "blank.pdf")


def test_hyperlink_only_urls_are_still_extracted():
    """
    Regression test: a resume where "LinkedIn"/"GitHub" are the visible
    hyperlink anchor text (common in resume templates), with the actual URL
    only present as a PDF link annotation, not as visible text. Plain text
    extraction alone can't find a URL to match against -- the hyperlink URIs
    must be pulled in separately and appended to the extracted text.
    """
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Contact: LinkedIn | GitHub")
    page.insert_link({
        "kind": fitz.LINK_URI, "from": fitz.Rect(100, 40, 150, 60),
        "uri": "https://linkedin.com/in/johndoe",
    })
    page.insert_link({
        "kind": fitz.LINK_URI, "from": fitz.Rect(160, 40, 210, 60),
        "uri": "https://github.com/johndoe",
    })
    pdf_bytes = doc.tobytes()
    doc.close()

    text = extract_text_from_bytes(pdf_bytes, "test.pdf")
    assert "linkedin.com/in/johndoe" in text
    assert "github.com/johndoe" in text


def test_column_aware_extraction_falls_back_on_single_column():
    """
    A single-column page (all blocks span nearly the full width) must be
    read top-to-bottom, not reordered as if it had two columns.
    """
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "First Line")
    page.insert_text((50, 100), "Second Line")
    page.insert_text((50, 150), "Third Line")
    text = _extract_page_text_column_aware(page)
    doc.close()

    lines = [ln for ln in text.split("\n") if ln.strip()]
    assert lines == ["First Line", "Second Line", "Third Line"]
