"""
parser_engine.py  —  Week 8 FINAL | Hybrid AI Resume Parser
============================================================
Architecture (confidence hierarchy, highest → lowest):

  Tier 1 — EntityRuler   : Verified patterns loaded from entity_patterns.json
  Tier 2 — SpaCy-NER     : Custom-trained neural NER model
  Tier 3 — Regex         : Deterministic rule-based extraction
  Tier 4 — Heuristic     : Positional / keyword fallback
  Tier 5 — LLM-Gemini    : Gemini API, called ONLY when Tiers 1-4 all fail
                           (cost-efficient: API is the last resort, not the first)

Week 8 upgrades over Week 7:
  - LLM fallback layer via google-genai (Gemini 2.5 Flash-Lite)
  - Expanded SKILL_CATEGORIES: Cloud, Databases, DevOps, Soft Skills
  - Edge-case hardening: multi-column PDF sort, image-only PDF guard,
    50k-char truncation guard for very large resumes
  - LinkedIn extraction added to Regex tier
  - Whole-word skill matching (avoids "C" matching inside "CI/CD")
  - _meta now exposes llm_calls_made counter for cost transparency
"""

import re
import io
import os
import json
import logging
import pathlib
from typing import Optional

import fitz          # PyMuPDF
import spacy

logger = logging.getLogger(__name__)

# EntityRuler seed patterns live in a JSON config next to this module so the
# ruler is data-driven — swap in your own verified names/institutions without
# touching code. See entity_patterns.json for the format.
DEFAULT_ENTITY_PATTERNS_PATH = pathlib.Path(__file__).with_name("entity_patterns.json")


def _load_entity_patterns(path) -> list[dict]:
    path = pathlib.Path(path)
    if not path.exists():
        logger.warning(
            "Entity pattern file not found at '%s' — EntityRuler will start with no seed patterns.",
            path,
        )
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:           # noqa: BLE001
        logger.warning("Failed to load entity patterns from '%s': %s", path, exc)
        return []

# Optional LLM import — degrades gracefully if SDK not installed.
# Uses google-genai (the current SDK) rather than the legacy
# google-generativeai package, whose support ended 2025-11-30.
try:
    from google import genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SKILL CATALOGUE  (expanded for Week 8)
# ═══════════════════════════════════════════════════════════════════════════════

SKILL_MAP: dict[str, list[str]] = {
    "Python":           ["python", "python3", "py3"],
    "Machine Learning": ["ml", "machine learning", "deep learning"],
    "NLP":              ["natural language processing", "nlp"],
    "C++":              ["c++", "cpp"],
    "TypeScript":       ["typescript", "ts"],
    "PostgreSQL":       ["postgresql", "postgres"],
    "React":            ["react", "reactjs", "react.js"],
    "Node.js":          ["node.js", "nodejs", "node js", "node"],
    "Vue":              ["vue", "vuejs", "vue.js"],
    "Angular":          ["angular", "angularjs"],
    "Next.js":          ["next.js", "nextjs", "next js"],
    "Tailwind":         ["tailwind", "tailwindcss", "tailwind css"],
    "C#":               ["c#", "csharp", "c sharp"],
    "Photoshop":        ["photoshop", "adobe photoshop"],
    "Illustrator":      ["illustrator", "adobe illustrator"],
    "Adobe XD":         ["adobe xd"],
    "InDesign":         ["indesign", "adobe indesign"],
}

SKILL_CATEGORIES: dict[str, list[str]] = {
    "Languages": [
        "Python", "Java", "C++", "C", "SQL", "JavaScript", "TypeScript",
        "Bash", "HTML", "CSS", "Go", "Rust", "R", "MATLAB",
        "Swift", "Kotlin", "C#", "PHP", "Ruby",
    ],
    "AI/ML Frameworks": [
        "spaCy", "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Machine Learning",
        "NLP", "OpenCV", "Hugging Face", "LangChain", "XGBoost",
    ],
    "Frontend": [
        "React", "Vue", "Angular", "Next.js", "Node.js", "HTML", "CSS",
        "Tailwind", "Redux",
    ],
    "Design & Product": [
        "Figma", "Sketch", "Adobe XD", "Photoshop", "Illustrator", "InDesign",
    ],
    "Cloud & Infrastructure": [
        "AWS", "GCP", "Azure", "Heroku", "Vercel", "Netlify",
        "Firebase", "Cloudflare", "DigitalOcean",
    ],
    "Databases": [
        "MySQL", "PostgreSQL", "MongoDB", "Redis", "SQLite",
        "Cassandra", "Elasticsearch", "DynamoDB", "Supabase",
    ],
    "DevOps & Tools": [
        "Git", "GitHub", "Docker", "Kubernetes", "Jenkins", "Linux",
        "VS Code", "Jupyter", "Postman", "Terraform", "Ansible", "CI/CD",
    ],
    "Soft Skills": [
        "Leadership", "Communication", "Teamwork", "Problem Solving",
        "Agile", "Scrum", "Project Management", "Public Speaking",
    ],
}

# Shield — skill tokens that must NEVER be classified as a person's name
ALL_SKILLS_SHIELD: set[str] = {
    token.lower()
    for cat in SKILL_CATEGORIES.values()
    for token in cat
}

# Shield — contact-info-shaped strings (URLs, emails, phone numbers) that must
# NEVER be classified as a NAME or UNIVERSITY, however confidently a tier
# predicts them. Guards against e.g. a GitHub/LinkedIn URL sitting near the
# top of a resume being mistaken for the university line by an under-trained
# NER model — without this, a wrong-but-confident Tier 2 prediction blocks
# Tier 4's correct keyword-based match from ever running.
_CONTACT_INFO_PATTERN = re.compile(
    r'(https?://|www\.|github\.com|linkedin\.com|@[\w.\-]+\.\w{2,}|\b\d{3}[\s.\-]?\d{3}[\s.\-]?\d{4}\b)',
    re.IGNORECASE,
)


def _looks_like_contact_info(value: str) -> bool:
    return bool(_CONTACT_INFO_PATTERN.search(value))


# Keywords a real institution name plausibly contains. Shared by Tier 2's
# acceptance check and Tier 4's keyword scan so both tiers agree on what
# "looks like a university" means — an under-trained NER model otherwise
# fires UNIVERSITY on arbitrary capitalized noise (skills, job titles,
# section headers), which blocks Tier 4's reliable match from ever running.
UNIVERSITY_KEYWORDS: tuple[str, ...] = (
    "University", "Institute", "College", "School", "Academy", "IIT", "NIT",
)


def _looks_like_university(value: str) -> bool:
    return any(kw in value for kw in UNIVERSITY_KEYWORDS)


# Words that mark a resume section header, never a real person's name.
# Shared by Tier 2's acceptance check and Tier 4a's positional heuristic so
# both tiers agree on what "looks like a name" means — an under-trained NER
# model otherwise fires NAME on headers like "WORK EXPERIENCE" just as
# readily as on an actual name, and (being non-"Not Found") blocks Tier 4a's
# reliable first-line match from ever running.
SECTION_HEADER_WORDS: set[str] = {
    "RESUME", "CURRICULUM", "VITAE", "EDUCATION", "CONTACT", "PROFILE", "PAGE", "CV",
    "WORK", "EXPERIENCE", "SKILLS", "TECHNICAL", "PROJECTS", "CERTIFICATIONS",
    "OBJECTIVE", "CAREER", "SUMMARY", "ACADEMIC", "REFERENCES", "PERSONAL", "DETAILS",
}

# A bare word ending in a comma ("California,") is a location fragment from a
# contact-info line, not a name.
_LOCATION_FRAGMENT_PATTERN = re.compile(r'^[A-Z][a-zA-Z.\-]*,$')


def _looks_like_bad_name(value: str) -> bool:
    value = value.strip()
    return (
        value.lower() in ALL_SKILLS_SHIELD
        or _looks_like_contact_info(value)
        or any(w in value.upper() for w in SECTION_HEADER_WORDS)
        or bool(_LOCATION_FRAGMENT_PATTERN.match(value))
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1b. SECTION SEGMENTATION  (rule-based, line-based — no ML)
# ═══════════════════════════════════════════════════════════════════════════════

# Section header keyword -> canonical section name. Matching is case-insensitive
# and exact (after stripping surrounding decoration like ":", "-", "="), since
# real resume headers are almost always a standalone short line rather than a
# phrase embedded in a sentence.
SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Education":  ("education", "academic background", "qualifications"),
    "Experience": ("experience", "work experience", "employment history", "professional experience"),
    "Skills":     ("skills", "technical skills", "core competencies"),
    "Projects":   ("projects", "personal projects"),
}

_SECTION_HEADER_MAX_LEN = 40
_HEADER_DECORATION_PATTERN = re.compile(r'^[\s\-=_•*:#]+|[\s\-=_•*:#]+$')


def _normalize_header_candidate(line: str) -> str:
    return _HEADER_DECORATION_PATTERN.sub("", line.strip()).strip()


def _match_known_section(normalized_lower: str) -> Optional[str]:
    for section, keywords in SECTION_KEYWORDS.items():
        if normalized_lower in keywords:
            return section
    return None


def _is_all_caps_header_shape(normalized: str) -> bool:
    letters = [c for c in normalized if c.isalpha()]
    return bool(letters) and normalized.upper() == normalized


def segment_resume(text: str) -> dict[str, str]:
    """
    Rule-based line segmentation into resume sections — deliberately lightweight
    (no ML): splits on short (<40 char) header-shaped lines.

      - A line matching a known keyword (see SECTION_KEYWORDS) starts that
        named section: Education, Experience, Skills, or Projects.
      - Everything before the first recognized header is "Header" (this is
        where NAME almost always lives).
      - A short ALL-CAPS line that *doesn't* match a known keyword (e.g.
        "CERTIFICATIONS", "AWARDS") still starts a section boundary, just
        routed to "Other" — so it doesn't get silently absorbed into
        whichever named section preceded it.
      - Anything else is body content in the current section.

    Returns a dict with keys Header/Education/Experience/Skills/Projects/Other,
    each a whitespace-stripped string of that section's lines (possibly empty).
    """
    sections: dict[str, list[str]] = {
        "Header": [], "Education": [], "Experience": [],
        "Skills": [], "Projects": [], "Other": [],
    }
    current = "Header"
    seen_known_header = False

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and len(stripped) < _SECTION_HEADER_MAX_LEN:
            normalized = _normalize_header_candidate(stripped)
            known = _match_known_section(normalized.lower())
            if known:
                current = known
                seen_known_header = True
                continue
            # Only treat an unrecognized all-caps line as a section boundary
            # once we're past the intro area -- otherwise an all-caps NAME
            # ("RAHUL V.", "BELLA TREVINO") gets mistaken for a header and
            # the real Header section ends up empty.
            if seen_known_header and _is_all_caps_header_shape(normalized):
                current = "Other"
                continue
        sections[current].append(line)

    return {name: "\n".join(ls).strip() for name, ls in sections.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LLM FALLBACK  (Gemini 2.5 Flash-Lite — cheapest tier, ~1s latency)
# ═══════════════════════════════════════════════════════════════════════════════

class GeminiFallback:
    """
    Wraps Gemini 2.5 Flash-Lite.  Only instantiated when an API key is present.
    Called at most ONCE per resume, only when NAME or UNIVERSITY is still
    "Not Found" after all local tiers have been exhausted.
    This is the cost-efficiency principle: local is free, LLM costs money.

    Model name is a class constant rather than buried in a method call --
    Gemini's model lineup turns over reasonably often (gemini-1.5-flash, the
    model this class originally used, is fully retired as of this writing).
    Check https://ai.google.dev/gemini-api/docs/models if queries start
    failing outright.
    """

    _MODEL = "gemini-2.5-flash-lite"

    _PROMPT = """You are a resume data extractor. Read the resume text below and respond with
ONLY a JSON object — no explanation, no markdown fences.

Extract exactly two fields:
  "name"       : The candidate's full name (string, or null if not found)
  "university" : Their primary educational institution (string, or null if not found)

Resume text (first 1500 chars):
\"\"\"{text}\"\"\"

Respond with valid JSON only. Example: {{"name": "Jane Doe", "university": "IIT Delhi"}}"""

    def __init__(self, api_key: str):
        if not _GEMINI_AVAILABLE:
            raise RuntimeError(
                "google-genai is not installed. "
                "Run: pip install google-genai"
            )
        self._client = genai.Client(api_key=api_key)

    def query(self, text: str) -> dict:
        """Returns {"name": str|None, "university": str|None}. Never raises."""
        try:
            response = self._client.models.generate_content(
                model=self._MODEL,
                contents=self._PROMPT.format(text=text[:1500]),
            )
            raw = response.text.strip()
            # Strip accidental markdown fences the model sometimes adds
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            data = json.loads(raw)
            return {
                "name":       data.get("name") or None,
                "university": data.get("university") or None,
            }
        except Exception as exc:           # noqa: BLE001
            logger.warning("Gemini query failed: %s", exc)
            return {"name": None, "university": None}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class IndustrialParser:
    """
    Five-tier hybrid extraction engine.

    Usage:
        parser = IndustrialParser(
            model_path    = "./model-best",
            gemini_api_key= "YOUR_KEY",   # omit to disable LLM tier
        )
        result = parser.parse(text)
    """

    def __init__(
        self,
        model_path: str = "./model-best",
        gemini_api_key: Optional[str] = None,
        entity_patterns_path: Optional[str] = None,
    ):
        self.model_error: Optional[str] = None
        self.nlp = None
        self._gemini: Optional[GeminiFallback] = None
        self._ruler_patterns_by_label: dict[str, set[str]] = {}

        # ── Load spaCy model ──────────────────────────────────────────────
        try:
            self.nlp = spacy.load(model_path)

            # EntityRuler sits BEFORE ner in the pipeline — verified patterns win
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            patterns = _load_entity_patterns(entity_patterns_path or DEFAULT_ENTITY_PATTERNS_PATH)
            if patterns:
                ruler.add_patterns(patterns)
            for p in patterns:
                self._ruler_patterns_by_label.setdefault(p["label"], set()).add(p["pattern"])

        except OSError:
            self.model_error = (
                f"spaCy model not found at '{model_path}'. "
                "NER will fall back to Heuristics and LLM."
            )
        except Exception as exc:           # noqa: BLE001
            self.model_error = f"Model load error: {exc}"

        # ── Load Gemini (optional) ────────────────────────────────────────
        key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        if key:
            try:
                self._gemini = GeminiFallback(api_key=key)
            except Exception as exc:       # noqa: BLE001
                logger.warning("Gemini init failed (LLM tier disabled): %s", exc)

    # ── Tiers 1+2 helper: section-scoped EntityRuler + SpaCy-NER ──────────────

    def _extract_entity_from_scope(self, scope_text: str, label: str, accept_fn):
        """
        Runs EntityRuler + SpaCy-NER on `scope_text` only, returns the first
        accepted (value, source) for `label`, or (None, None). Scoping the
        input text (rather than post-filtering full-document entities) is
        what lets NAME look only at the Header section and UNIVERSITY look
        only at the Education section.
        """
        if not self.nlp or not scope_text.strip():
            return None, None

        doc = self.nlp(scope_text)
        value, source = None, None

        for ent in doc.ents:
            if ent.label_ != label:
                continue
            val = ent.text.strip()
            if value is None and accept_fn(val):
                value, source = val, "SpaCy-NER"

        # Promote EntityRuler-matched spans to their own badge (see the Tier 1
        # docstring note in __init__ for why text-matching is used here).
        for ent in doc.ents:
            if ent.label_ == label and ent.text.strip() in self._ruler_patterns_by_label.get(label, ()):
                value, source = ent.text.strip(), "EntityRuler"

        return value, source

    # ── Tier 3: Regex ────────────────────────────────────────────────────────

    def _extract_metadata(self, text: str) -> dict:
        """Returns {field: (value, source_tag)} for all Regex-extractable fields."""
        email_m    = re.search(r'[\w.\-+]+@[\w.\-]+\.\w{2,}', text)
        phone_m    = re.search(r'\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}', text)
        gpa_m      = re.search(
            r'\b(?:GPA|CGPA|CPI|Score)[:\s]*(\d(?:\.\d{1,2})?)', text, re.IGNORECASE
        )
        years      = re.findall(r'\b(20\d{2})\b', text)
        github_m   = re.search(r'github\.com/([\w\-]+)', text, re.IGNORECASE)
        linkedin_m = re.search(r'linkedin\.com/in/([\w\-]+)', text, re.IGNORECASE)
        degree_m   = re.search(
            r'\b(B\.?Tech|M\.?Tech|B\.?E\.?|B\.?S\.?|Bachelor|Master|Ph\.?D)\b',
            text, re.IGNORECASE,
        )

        return {
            "EMAIL":    (email_m.group(0)   if email_m    else "N/A", "Regex"),
            "PHONE":    (phone_m.group(0)   if phone_m    else "N/A", "Regex"),
            "GPA":      (gpa_m.group(1)     if gpa_m      else "N/A", "Regex"),
            "BATCH":    (max(years)          if years       else "N/A", "Regex"),
            "GITHUB":   (
                f"https://github.com/{github_m.group(1)}" if github_m else "N/A",
                "Regex",
            ),
            "LINKEDIN": (
                f"https://linkedin.com/in/{linkedin_m.group(1)}" if linkedin_m else "N/A",
                "Regex",
            ),
            "DEGREE":   (degree_m.group(0)  if degree_m   else "N/A", "Regex"),
        }

    # ── Full parse pipeline ──────────────────────────────────────────────────

    def parse(self, text: str) -> dict:
        """
        Runs all five tiers in order.  Returns:
        {
            "PERSONAL": { field: {"value": str, "source": str}, ... },
            "CONTACT":  { ... },
            "SKILLS":   { category: [skill, ...], ... },
            "_meta": {
                "model_status":   "ok" | "degraded",
                "model_error":    str | None,
                "llm_used":       bool,
                "llm_calls_made": int,
                "field_sources":  { field: source, ... }
            }
        }
        """
        meta      = self._extract_metadata(text)          # regex tier: always full document
        lines     = [ln.strip() for ln in text.split("\n") if ln.strip()]
        llm_calls = 0

        # ── Section segmentation (rule-based) ─────────────────────────────
        # Used to scope skill matching to Skills+Projects (deterministic
        # keyword matching only gets more precise from a smaller, relevant
        # scope). NAME and UNIVERSITY were also scoped to Header/Education in
        # an earlier version of this pipeline, but a pipeline-level A/B eval
        # against dev.spacy showed that regressed UNIVERSITY F1 (0.565 ->
        # 0.527): the NER model was trained on full-document context and
        # sometimes mislabels the same span (e.g. tags a university as NAME)
        # when given only an isolated Education-section snippet, and the
        # keyword-heuristic fallback can't tell which of several institutions
        # mentioned in a section is the right one. So NAME/UNIVERSITY stay
        # scoped to the full document; only Skills is section-scoped.
        try:
            sections = segment_resume(text)
        except Exception as exc:           # noqa: BLE001
            logger.warning("Section segmentation failed, falling back to whole-document: %s", exc)
            sections = {}

        skills_section_used = "Skills+Projects"
        skills_scope = "\n".join(
            p for p in (sections.get("Skills", ""), sections.get("Projects", "")) if p.strip()
        ).strip()
        if not skills_scope:
            skills_scope = text
            skills_section_used = "Full Document (fallback)"

        # ── Result skeleton ───────────────────────────────────────────────
        personal: dict[str, dict] = {
            "NAME":       {"value": "Not Found", "source": "—"},
            "UNIVERSITY": {"value": "Not Found", "source": "—"},
            "DEGREE":     {"value": meta["DEGREE"][0],   "source": meta["DEGREE"][1]},
            "BATCH":      {"value": meta["BATCH"][0],    "source": meta["BATCH"][1]},
        }
        contact: dict[str, dict] = {
            "EMAIL":    {"value": meta["EMAIL"][0],    "source": meta["EMAIL"][1]},
            "PHONE":    {"value": meta["PHONE"][0],    "source": meta["PHONE"][1]},
            "GPA":      {"value": meta["GPA"][0],      "source": meta["GPA"][1]},
            "GITHUB":   {"value": meta["GITHUB"][0],   "source": meta["GITHUB"][1]},
            "LINKEDIN": {"value": meta["LINKEDIN"][0], "source": meta["LINKEDIN"][1]},
        }
        skills: dict[str, list[str]] = {cat: [] for cat in SKILL_CATEGORIES}

        # ── Tiers 1 + 2: EntityRuler + SpaCy NER (full document -- see note above) ─
        name_val, name_src = self._extract_entity_from_scope(
            text, "NAME", lambda v: not _looks_like_bad_name(v)
        )
        if name_val:
            personal["NAME"] = {"value": name_val, "source": name_src}

        uni_val, uni_src = self._extract_entity_from_scope(
            text, "UNIVERSITY",
            lambda v: not _looks_like_contact_info(v) and _looks_like_university(v),
        )
        if uni_val:
            personal["UNIVERSITY"] = {"value": uni_val, "source": uni_src}

        # ── Tier 4a: Positional heuristic for NAME ────────────────────────
        if personal["NAME"]["value"] == "Not Found" and lines:
            candidate = lines[0]
            if len(candidate) < 45 and not _looks_like_bad_name(candidate):
                personal["NAME"] = {"value": candidate, "source": "Heuristic"}

        # ── Tier 4b: Keyword heuristic for UNIVERSITY ─────────────────────
        if personal["UNIVERSITY"]["value"] == "Not Found":
            for line in lines:
                if _looks_like_university(line):
                    personal["UNIVERSITY"] = {"value": line, "source": "Heuristic"}
                    break

        # ── Tier 5: LLM-Gemini fallback ───────────────────────────────────
        # Fires only if a critical field is still missing AND Gemini is configured
        needs_llm = (
            personal["NAME"]["value"] == "Not Found"
            or personal["UNIVERSITY"]["value"] == "Not Found"
        )
        if needs_llm and self._gemini:
            llm_result = self._gemini.query(text)
            llm_calls += 1

            if personal["NAME"]["value"] == "Not Found" and llm_result.get("name"):
                personal["NAME"] = {"value": llm_result["name"], "source": "LLM-Gemini"}

            if personal["UNIVERSITY"]["value"] == "Not Found" and llm_result.get("university"):
                personal["UNIVERSITY"] = {
                    "value": llm_result["university"],
                    "source": "LLM-Gemini",
                }

        # ── Skill detection (whole-word matching, alias-aware, scoped) ─────
        # SKILL_MAP normalizes aliases/abbreviations (e.g. "ml", "deep
        # learning" -> "Machine Learning") to their canonical skill name,
        # so a resume mentioning any alias still surfaces the one entry.
        # Scoped to Skills+Projects (falls back to full document) so a skill
        # named in passing in Experience/Education doesn't get double-counted
        # or crowd out the dedicated Skills section's intent.
        text_lower = skills_scope.lower()
        for category, skill_list in SKILL_CATEGORIES.items():
            for skill in skill_list:
                search_terms = [skill.lower()] + SKILL_MAP.get(skill, [])
                for term in search_terms:
                    pattern = re.escape(term)
                    if re.search(rf'(?<!\w){pattern}(?!\w)', text_lower):
                        skills[category].append(skill)
                        break

        # ── _meta block ───────────────────────────────────────────────────
        all_fields    = {**personal, **contact}
        field_sources = {k: v["source"] for k, v in all_fields.items()}

        # Which section each field's extraction was scoped to. Only skill
        # matching is section-scoped (see note above); every PERSONAL/CONTACT
        # field runs against the full document.
        field_sections = {k: "Full Document" for k in all_fields}

        return {
            "PERSONAL": personal,
            "CONTACT":  contact,
            "SKILLS":   skills,
            "_meta": {
                "model_status":   "degraded" if self.model_error else "ok",
                "model_error":    self.model_error,
                "llm_used":       llm_calls > 0,
                "llm_calls_made": llm_calls,
                "field_sources":  field_sources,
                "field_sections": field_sections,
                "skills_section": skills_section_used,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FILE UTILITIES  (edge-case hardened)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_page_text_column_aware(page) -> str:
    """
    Reconstruct a page's text in column-then-row order instead of fitz's
    default (or sort=True's) reading order, both of which can interleave
    left/right column content character-by-character on genuinely
    multi-column layouts (verified on a real resume: "Chicago, IL" and
    "University of Illinois Chicago" from two different columns got glued
    into one garbled string). Block-level extraction keeps each block's text
    intact; only their ORDER changes.

    Column split detection: find the largest horizontal gap between block
    x0 positions. Only treat it as a real column boundary if it's a
    meaningful fraction of the page width AND no block's bounding box
    straddles it -- a single-column resume with centered/indented headers
    (e.g. a centered name above left-aligned body text) produces x0 gaps
    too, but its blocks span across any such "gap", which a genuine column
    split never does. Without this check, single-column resumes -- the
    common case -- would have their reading order scrambled instead of
    merely left unchanged.
    """
    blocks = [b for b in page.get_text("blocks") if b[4].strip()]
    if len(blocks) < 2:
        return "".join(b[4] for b in blocks)

    xs = sorted({round(b[0]) for b in blocks})
    page_width = page.rect.width
    best_gap, best_split = 0, None
    for prev, curr in zip(xs, xs[1:]):
        gap = curr - prev
        if gap > best_gap:
            best_gap, best_split = gap, (prev + curr) / 2

    is_real_column_split = (
        best_split is not None
        and best_gap > page_width * 0.12
        and page_width * 0.15 < best_split < page_width * 0.85
        and all(b[2] <= best_split or b[0] >= best_split for b in blocks)
    )

    if is_real_column_split:
        left = sorted((b for b in blocks if b[0] < best_split), key=lambda b: b[1])
        right = sorted((b for b in blocks if b[0] >= best_split), key=lambda b: b[1])
        ordered = left + right
    else:
        ordered = sorted(blocks, key=lambda b: (b[1], b[0]))

    return "".join(b[4] for b in ordered)


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """
    Convert raw uploaded bytes → plain text.

    Edge cases handled:
      Multi-column PDFs  — column-aware block reordering (see
                           _extract_page_text_column_aware) reads genuine
                           columns top-to-bottom, left-to-right without
                           scrambling single-column layouts
      Image-only PDFs    — empty text triggers a descriptive ValueError
      Oversized resumes  — hard cap at 50 000 chars to keep NLP pipeline fast
      Legacy TXT files   — UTF-8 with Latin-1 fallback
    """
    if filename.lower().endswith(".pdf"):
        try:
            doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
            full_text = "".join(_extract_page_text_column_aware(page) for page in doc).strip()

            if not full_text:
                raise ValueError(
                    "This PDF appears to be image-only (scanned document). "
                    "No machine-readable text was found. "
                    "Please provide a PDF with selectable text, or run OCR first."
                )

            return full_text[:50_000]

        except ValueError:
            raise   # re-raise our own clear error messages as-is
        except Exception as exc:           # noqa: BLE001
            raise ValueError(f"PDF read failed for '{filename}': {exc}") from exc

    # Plain-text fallback
    try:
        return file_bytes.decode("utf-8")[:50_000]
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")[:50_000]


def extract_text_from_path(file_path: str) -> str:
    """Kept for CLI / batch-processing backward compatibility."""
    with open(file_path, "rb") as fh:
        return extract_text_from_bytes(fh.read(), os.path.basename(file_path))
