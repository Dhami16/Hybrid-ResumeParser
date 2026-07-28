"""
job_matcher.py — lightweight weighted skill-overlap job-description matcher

Parses a job description into Required / Bonus / Nice-to-have skill tiers
using simple text heuristics matched to common JD phrasing, extracts which
known catalog skills (SKILL_CATEGORIES / SKILL_MAP, the same catalog
parser_engine.py's skill-detection tier uses) are mentioned in each tier, and
scores a candidate's extracted skill set against it: matched skills are
weighted by tier and summed as a fraction of the JD's maximum possible score.

Grounding the match in the existing skill catalog (rather than free-text NLP
over the JD) keeps this comparable apples-to-apples with what the resume
parser itself already recognizes -- no new extraction logic to maintain.
"""

import re

from parser_engine import SKILL_CATEGORIES, SKILL_MAP

TIER_WEIGHTS: dict[str, int] = {"required": 3, "bonus": 2, "nice_to_have": 1}

_ALL_CATALOG_SKILLS: list[str] = [
    skill for skills in SKILL_CATEGORIES.values() for skill in skills
]

_SECTION_HEADERS = {"requirements", "responsibilities", "qualifications"}
_BONUS_PATTERN = re.compile(r'^bonus:?\s*(.*)', re.IGNORECASE)
_NICE_TO_HAVE_PATTERN = re.compile(r'^nice to have:?\s*(.*)', re.IGNORECASE)
_BULLET_PATTERN = re.compile(r'^[-*•]\s*')


def _find_catalog_skills(text: str) -> set[str]:
    """Returns the set of canonical catalog skill names mentioned in text."""
    text_lower = text.lower()
    found = set()
    for skill in _ALL_CATALOG_SKILLS:
        search_terms = [skill.lower()] + SKILL_MAP.get(skill, [])
        for term in search_terms:
            pattern = re.escape(term)
            if re.search(rf'(?<!\w){pattern}(?!\w)', text_lower):
                found.add(skill)
                break
    return found


# ── Catalog-gap safety net ──────────────────────────────────────────────────
# Flags capitalized/tech-looking terms that AREN'T in SKILL_CATEGORIES, so a
# JD requirement our catalog doesn't recognize is surfaced instead of
# silently dropped. Deliberately a rough heuristic, not an NLP model.
_CAMEL_CASE_PATTERN = re.compile(r'\b[A-Za-z]*[a-z][A-Z][A-Za-z]*\b')
_TRIGGER_PHRASE_PATTERN = re.compile(
    r'(?:experience (?:with|in)|proficiency in|proficient in|knowledge of|familiarity with)\s+([^.\n]+)',
    re.IGNORECASE,
)
_COLON_LIST_PATTERN = re.compile(r'^[A-Za-z][\w /&]{1,30}:\s*(.+)$')
_NON_TERM_WORDS = {"and", "or", "the", "a", "an", "with", "in", "of", "for", "to", "on"}


def _split_candidate_list(fragment: str) -> list[str]:
    fragment = re.sub(r'\s+(?:and|or)\s+', ',', fragment, flags=re.IGNORECASE)
    return [p.strip(" .()") for p in fragment.split(",") if p.strip(" .()")]


def _looks_like_tech_term(term: str) -> bool:
    if not term or len(term) < 2 or len(term) > 30:
        return False
    if term.lower() in _NON_TERM_WORDS:
        return False
    if len(term.split()) > 4:
        return False
    # Proper-noun-shaped (capitalized) or contains a digit/version marker --
    # filters out generic sentence fragments like "a role where" or "years".
    return term[0].isupper() or any(c.isdigit() for c in term)


def _find_unrecognized_terms(text: str, known_skills: set[str]) -> list[str]:
    """Candidate tech terms in `text` not covered by `known_skills` (already-
    matched catalog skills for this tier). Not scored -- just surfaced."""
    candidates: set[str] = set()

    for m in _CAMEL_CASE_PATTERN.finditer(text):
        candidates.add(m.group(0))

    for m in _TRIGGER_PHRASE_PATTERN.finditer(text):
        candidates.update(_split_candidate_list(m.group(1)))

    for line in text.split("\n"):
        m = _COLON_LIST_PATTERN.match(line.strip())
        if m:
            candidates.update(_split_candidate_list(m.group(1)))

    known_lower = {s.lower() for s in known_skills}
    for skill in known_skills:
        known_lower.update(alias.lower() for alias in SKILL_MAP.get(skill, []))

    unrecognized = set()
    for term in candidates:
        term = term.strip()
        if not _looks_like_tech_term(term):
            continue
        if term.lower() in known_lower:
            continue
        if _find_catalog_skills(term):
            continue  # candidate text itself contains/is a recognized skill
        unrecognized.add(term)

    return sorted(unrecognized)


def parse_job_description(text: str) -> dict[str, dict]:
    """
    Returns {
        "required":     {"skills": {...}, "unrecognized_terms": [...]},
        "bonus":        {"skills": {...}, "unrecognized_terms": [...]},
        "nice_to_have": {"skills": {...}, "unrecognized_terms": [...]},
    }
    "skills" is the set of canonical catalog skill names mentioned in that
    tier. "unrecognized_terms" is a best-effort list of capitalized/tech-
    looking terms in that tier NOT covered by the catalog -- surfaced (not
    scored) so catalog gaps are visible instead of silently dropped.

    Heuristics (matched to the JD phrasing this is designed for):
      - A short line ending in ":" that reads "Requirements"/"Responsibilities"/
        "Qualifications" sets the current section.
      - Within "Requirements", a bullet starting with "Bonus:" is a bonus-tier
        skill line; every other bullet in that section is required-tier.
      - A "Nice to have: ..." line is recognized anywhere in the document,
        not just inside a bulleted section (it commonly appears as a trailing
        standalone line).
    """
    current_section = None
    required_lines: list[str] = []
    bonus_lines: list[str] = []
    nice_to_have_lines: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        header = stripped.rstrip(":").strip().lower()
        if header in _SECTION_HEADERS:
            current_section = header
            continue

        nice_match = _NICE_TO_HAVE_PATTERN.match(stripped)
        if nice_match:
            nice_to_have_lines.append(nice_match.group(1))
            continue

        if current_section == "requirements":
            bullet = _BULLET_PATTERN.sub("", stripped)
            bonus_match = _BONUS_PATTERN.match(bullet)
            if bonus_match:
                bonus_lines.append(bonus_match.group(1))
            else:
                required_lines.append(bullet)

    tier_texts = {
        "required":     "\n".join(required_lines),
        "bonus":        "\n".join(bonus_lines),
        "nice_to_have": "\n".join(nice_to_have_lines),
    }

    result = {}
    for tier, tier_text in tier_texts.items():
        skills = _find_catalog_skills(tier_text)
        result[tier] = {
            "skills": skills,
            "unrecognized_terms": _find_unrecognized_terms(tier_text, skills),
        }
    return result


def score_match(resume_skills: set[str], jd_tiers: dict[str, dict]) -> dict:
    """
    Returns {
        "score": float,          # earned_points / max_points, 0.0 if JD has no recognized skills
        "earned_points": int,
        "max_points": int,
        "matched": {tier: [skill, ...]},
        "missing": {tier: [skill, ...]},
    }
    `jd_tiers` is parse_job_description()'s output; "unrecognized_terms" is
    intentionally not scored here -- we can't know if the candidate has an
    unrecognized skill, so it's surfaced separately for transparency instead.
    """
    max_points = 0
    earned_points = 0
    matched: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}

    for tier, weight in TIER_WEIGHTS.items():
        tier_skills = jd_tiers.get(tier, {}).get("skills", set())
        max_points += len(tier_skills) * weight
        hit = tier_skills & resume_skills
        miss = tier_skills - resume_skills
        earned_points += len(hit) * weight
        matched[tier] = sorted(hit)
        missing[tier] = sorted(miss)

    score = earned_points / max_points if max_points else 0.0

    return {
        "score": score,
        "earned_points": earned_points,
        "max_points": max_points,
        "matched": matched,
        "missing": missing,
    }
