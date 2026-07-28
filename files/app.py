"""
app.py  —  Week 8 FINAL | Hybrid AI Resume Parser  |  Streamlit Dashboard
==========================================================================
Run:   streamlit run app.py

New in Week 8 vs Week 7:
  - Gemini API key input in sidebar (stored in session_state, not on disk)
  - LLM-Gemini badge added to confidence legend and card renderer
  - "LLM Called" cost indicator appears when Gemini fires
  - Architecture diagram tab for Week 8 presentation mode
  - LinkedIn field displayed in Contact section
  - Graceful empty-state for every skill category
  - st.secrets integration: set GEMINI_API_KEY in .streamlit/secrets.toml
    for Streamlit Community Cloud deployment
"""

import json
import os
import pathlib
import time
import streamlit as st
import pandas as pd

from parser_engine import IndustrialParser, extract_text_from_bytes
from job_matcher import parse_job_description, score_match

APP_DIR = pathlib.Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid AI Resume Parser",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — stark industrial aesthetic: blank CD-R, one strip of red tape
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stSidebar"],
[data-testid="stSidebarContent"],
[data-testid="stHeader"] {
    font-family: 'Space Mono', monospace;
    background-color: #0a0a0a;
    color: #f5f5f0;
}
[data-testid="stApp"] * {
    color: #f5f5f0;
}
[data-testid="stSidebar"] hr,
[data-testid="stMain"] hr {
    border-color: #6b6b6b;
}

/* Streamlit's own default accent (#ff4b4b) leaks into tab indicators/focus
   rings via BaseWeb — override so red stays exclusive to the EntityRuler
   field marker, the single signature moment of the page. */
[data-baseweb="tab-highlight"] { background-color: #6b6b6b !important; }
[data-baseweb="tab"][aria-selected="true"],
[data-baseweb="tab"][aria-selected="true"] p { color: #f5f5f0 !important; }

/* No rounded corners anywhere */
button, [data-baseweb="tab"] { border-radius: 0 !important; }
.hero {
    background: #0a0a0a;
    border-bottom: 1px solid #6b6b6b;
    padding: 2.4rem 2rem 1.6rem;
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: "Helvetica Neue Condensed Bold", "Arial Narrow", sans-serif;
    font-weight: 700;
    font-size: 2.4rem;
    text-transform: uppercase;
    letter-spacing: 0.02em;
    color: #f5f5f0;
    margin: 0 0 0.3rem;
}
.hero p {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    color: #6b6b6b;
    margin: 0;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: #6b6b6b;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #6b6b6b;
}

/* Source tags — plain bracketed mono text, grey by default */
.tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #6b6b6b;
    letter-spacing: 0.03em;
}

/* Data rows — manifest lines, no card chrome */
.data-row {
    border-bottom: 1px solid #6b6b6b;
    padding: 0.7rem 0 0.7rem 0;
    margin-bottom: 0;
}
.data-row.is-verified {
    border-left: 3px solid #c41e2f;
    padding-left: 0.6rem;
}
.data-field-key {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #6b6b6b;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.data-field-val {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #f5f5f0;
    margin: 0.15rem 0 0.3rem;
    word-break: break-all;
}

/* Skills — plain slash-separated mono list, no pills */
.skill-row {
    border-bottom: 1px solid #6b6b6b;
    padding: 0.7rem 0;
}
.skill-list {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #f5f5f0;
    margin-top: 0.4rem;
    line-height: 1.6;
}

.llm-indicator {
    background: #0a0a0a;
    border: 1px solid #6b6b6b;
    border-radius: 0;
    padding: 0.6rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #f5f5f0;
    margin-bottom: 1rem;
}
.model-ok, .model-warn, .model-llm {
    font-family:'Space Mono',monospace; font-size:0.8rem; color:#f5f5f0;
}

[data-testid="stFileUploader"] {
    background: #0a0a0a;
    padding: 0;
}
[data-testid="stFileUploaderDropzone"] {
    background: #0a0a0a !important;
    border: 1px solid #6b6b6b !important;
    border-radius: 0 !important;
    transition: border-color 0.15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: #f5f5f0 !important; }
button[data-testid="stBaseButton-secondary"] {
    background: #0a0a0a !important;
    color: #f5f5f0 !important;
    border: 1px solid #6b6b6b !important;
    border-radius: 0 !important;
}

[data-testid="stTextArea"] textarea {
    background: #0a0a0a !important;
    color: #f5f5f0 !important;
    border: 1px solid #6b6b6b !important;
    border-radius: 0 !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #f5f5f0 !important;
    box-shadow: none !important;
}

.stProgress > div > div > div {
    background: #1a1a1a;
}
.stProgress > div > div > div > div {
    background: #f5f5f0;
}

.stExpander { background: #0a0a0a !important; border: 1px solid #6b6b6b !important; border-radius: 0 !important; }
hr { border-color: #6b6b6b; margin: 1.5rem 0; }

/* Architecture diagram tab — flattened, no per-tier colour coding */
.arch-tier {
    background: #0a0a0a;
    border-left: 1px solid #6b6b6b;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    border-radius: 0;
}
.arch-tier h4 {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #f5f5f0;
    margin: 0 0 0.25rem;
}
.arch-tier p { font-size: 0.82rem; color: #6b6b6b; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE → PLAIN BRACKETED TAG
# ─────────────────────────────────────────────────────────────────────────────
SOURCE_TAG = {
    "EntityRuler": "[ENTITYRULER]",
    "SpaCy-NER":   "[NER]",
    "Regex":       "[REGEX]",
    "Heuristic":   "[HEURISTIC]",
    "LLM-Gemini":  "[LLM-GEMINI]",
    "—":           "[—]",
}

def tag_html(source: str) -> str:
    label = SOURCE_TAG.get(source, SOURCE_TAG["—"])
    return f'<span class="tag">{label}</span>'

def render_data_card(key: str, value: str, source: str):
    row_class = "data-row is-verified" if source == "EntityRuler" else "data-row"
    st.markdown(f"""
    <div class="{row_class}">
        <div class="data-field-key">{key}</div>
        <div class="data-field-val">{value}</div>
        {tag_html(source)}
    </div>
    """, unsafe_allow_html=True)

def render_skills(skills_dict: dict):
    any_found = False
    for category, skill_list in skills_dict.items():
        if not skill_list:
            continue
        any_found = True
        joined = " / ".join(skill_list)
        st.markdown(f"""
        <div class="skill-row">
            <div class="data-field-key">{category}</div>
            <div class="skill-list">{joined}</div>
        </div>
        """, unsafe_allow_html=True)
    if not any_found:
        st.markdown("""
        <div class="skill-row" style="text-align:center; color:#6b6b6b; padding:1.5rem;">
            <div style="font-family:'Space Mono',monospace; font-size:0.75rem;">
                NO SKILLS MATCHED THE CATALOGUE
            </div>
        </div>
        """, unsafe_allow_html=True)

def build_flat_table(result: dict) -> pd.DataFrame:
    rows = []
    for section in ("PERSONAL", "CONTACT"):
        for field, obj in result[section].items():
            rows.append({
                "Section": section,
                "Field":   field,
                "Value":   obj["value"],
                "Source":  obj["source"],
            })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# PARSER  — cached so model loads once per session
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_parser(gemini_key: str) -> IndustrialParser:
    # model-best (full lg-vectors model, ~445MB) is gitignored -- too large
    # for GitHub and not present on a deployed instance. model-best-lite
    # (md-vectors, 74MB) is committed specifically so the deployed app still
    # has real NER instead of silently degrading to Heuristic-only. Paths are
    # resolved relative to this file (not the process cwd), since Streamlit
    # Community Cloud doesn't necessarily run with cwd set to files/ the way
    # local `streamlit run app.py` (launched from inside files/) does.
    full_model = APP_DIR / "model-best"
    lite_model = APP_DIR / "model-best-lite"
    model_path = str(full_model if full_model.is_dir() else lite_model)
    return IndustrialParser(model_path=model_path, gemini_api_key=gemini_key or None)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Configuration")

    # Gemini API key — try st.secrets first (for Cloud deployment), then input
    default_key = ""
    try:
        default_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    gemini_key = st.text_input(
        "Gemini API Key (optional)",
        value=default_key,
        type="password",
        help=(
            "Enables Tier 5 LLM fallback for Name/University when local "
            "extraction fails. Get a free key at aistudio.google.com. "
            "For Cloud deployment, set GEMINI_API_KEY in .streamlit/secrets.toml"
        ),
    )
    if gemini_key:
        st.markdown('<p class="model-llm">✦ LLM Tier Active</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 Engine Status")

    parser = load_parser(gemini_key)

    if parser.model_error:
        st.markdown(
            f'<p class="model-warn">⚠ Degraded — {parser.model_error}</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<p class="model-ok">● spaCy Model Online</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏷 Confidence Legend")
    legend_items = {
        "EntityRuler": "Verified pattern — highest confidence",
        "SpaCy-NER":   "Neural NER prediction",
        "Regex":       "Rule-based extraction",
        "Heuristic":   "Positional / keyword fallback",
        "LLM-Gemini":  "Gemini API — last resort",
    }
    for source, desc in legend_items.items():
        st.markdown(f"{tag_html(source)} &nbsp; {desc}", unsafe_allow_html=True)
        st.markdown("")

    st.markdown("---")
    st.markdown("### ℹ About")
    st.caption(
        "Week 8 Final — 8-week CSE project. "
        "Five-tier hybrid engine: EntityRuler → SpaCy → Regex → Heuristic → LLM."
    )

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Hybrid AI Resume Parser</h1>
    <p>ENTITYRULER / SPACY NER / REGEX / HEURISTICS / LLM FALLBACK &nbsp;|&nbsp; WEEK 8 FINAL</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS  — Parser | Job Match | Architecture (presentation mode)
# ─────────────────────────────────────────────────────────────────────────────
tab_parser, tab_job_match, tab_arch = st.tabs(["📄 Parser", "🎯 Job Match", "🏛 Architecture"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1  — Main parser UI
# ═════════════════════════════════════════════════════════════════════════════
with tab_parser:

    st.markdown('<div class="section-label">Upload Resume</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        label="",
        type=["pdf", "txt"],
        accept_multiple_files=False,
        help="PDF (text-based) or TXT. Max ~50 000 chars processed.",
    )

    if uploaded is not None:

        # ── Progress bar ──────────────────────────────────────────────────
        bar = st.progress(0, text="Reading file…")
        time.sleep(0.1)
        bar.progress(20, text="Extracting text from PDF…")

        try:
            raw_bytes = uploaded.read()
            text = extract_text_from_bytes(raw_bytes, uploaded.name)
        except ValueError as exc:
            st.error(f"❌ {exc}")
            st.stop()

        bar.progress(50, text="Running NER pipeline…")
        time.sleep(0.15)

        result = parser.parse(text)

        bar.progress(80, text="Detecting skills…")
        time.sleep(0.1)
        bar.progress(100, text="Done ✓")
        time.sleep(0.3)
        bar.empty()

        # ── Warnings ──────────────────────────────────────────────────────
        if result["_meta"]["model_status"] == "degraded":
            st.warning(
                f"⚠ spaCy running in heuristic-only mode — "
                f"{result['_meta']['model_error']}"
            )

        if result["_meta"]["llm_used"]:
            st.markdown(
                '<div class="llm-indicator">'
                "✦ Gemini API was called (1 request) because local extraction "
                "could not resolve Name or University. LLM tier is the last resort."
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Three-column results ───────────────────────────────────────────
        col_p, col_c, col_s = st.columns([1.1, 1.2, 1.4])

        with col_p:
            st.markdown('<div class="section-label">Personal</div>', unsafe_allow_html=True)
            for field, obj in result["PERSONAL"].items():
                render_data_card(field, obj["value"], obj["source"])

        with col_c:
            st.markdown('<div class="section-label">Contact & Academics</div>', unsafe_allow_html=True)
            for field, obj in result["CONTACT"].items():
                render_data_card(field, obj["value"], obj["source"])

        with col_s:
            st.markdown('<div class="section-label">Skills Detected</div>', unsafe_allow_html=True)
            render_skills(result["SKILLS"])

        st.markdown("---")

        # ── Export row ────────────────────────────────────────────────────
        flat_df   = build_flat_table(result)
        stem      = uploaded.name.rsplit(".", 1)[0]
        c1, c2, _ = st.columns([1, 1, 2])

        with c1:
            st.download_button(
                "⬇ Download CSV",
                data=flat_df.to_csv(index=False),
                file_name=f"{stem}_parsed.csv",
                mime="text/csv",
            )
        with c2:
            clean = {k: v for k, v in result.items() if k != "_meta"}
            st.download_button(
                "⬇ Download JSON",
                data=json.dumps(clean, indent=2),
                file_name=f"{stem}_parsed.json",
                mime="application/json",
            )

        # ── Detail expanders ──────────────────────────────────────────────
        with st.expander("📋 Full Data Table"):
            st.dataframe(flat_df, use_container_width=True, hide_index=True)

        with st.expander("🔍 Raw JSON (debug / Week 8 demo)"):
            st.json(result)

        with st.expander("📄 Extracted Text (sanity-check PDF parse)"):
            st.text_area(
                "",
                value=text[:4000] + ("…[truncated]" if len(text) > 4000 else ""),
                height=260,
            )

    else:
        st.markdown("""
        <div style="
            text-align:center; padding:4rem 2rem;
            border:1px solid #6b6b6b; border-radius:0; margin-top:1.5rem;
        ">
            <div style="font-family:'Space Mono',monospace;color:#f5f5f0;font-size:1rem;text-transform:uppercase;letter-spacing:0.05em;">
                Drop a PDF or TXT resume above to begin
            </div>
            <div style="color:#6b6b6b;font-size:0.82rem;margin-top:0.5rem;">
                Source tags show exactly which extraction tier caught each field
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2  — Job Match: score the parsed resume's skills against a JD
# ═════════════════════════════════════════════════════════════════════════════
with tab_job_match:

    if uploaded is None:
        st.markdown('<div class="section-label">Job Match</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="
            text-align:center; padding:4rem 2rem;
            border:1px solid #6b6b6b; border-radius:0; margin-top:0.5rem;
        ">
            <div style="font-family:'Space Mono',monospace;color:#f5f5f0;font-size:1rem;text-transform:uppercase;letter-spacing:0.05em;">
                Upload a resume in the Parser tab first
            </div>
            <div style="color:#6b6b6b;font-size:0.82rem;margin-top:0.5rem;">
                Job matching scores the resume's detected skills against a job description
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-label">Job Description</div>', unsafe_allow_html=True)
        jd_file = st.file_uploader(
            label="",
            type=["txt"],
            accept_multiple_files=False,
            key="jd_uploader",
            help="Plain-text job description (.txt).",
        )
        jd_pasted = st.text_area(
            "Or paste job description text",
            height=180,
            key="jd_text_area",
            placeholder="Requirements:\n- 2+ years of experience with Python\n...",
        )

        jd_text = ""
        if jd_file is not None:
            jd_text = jd_file.read().decode("utf-8", errors="replace")
        elif jd_pasted.strip():
            jd_text = jd_pasted

        if not jd_text.strip():
            st.markdown("""
            <div style="
                text-align:center; padding:3rem 2rem;
                border:1px solid #6b6b6b; border-radius:0; margin-top:1rem;
            ">
                <div style="color:#6b6b6b;font-size:0.82rem;">
                    Upload or paste a job description above to see the match score
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            resume_skills = {s for cat in result["SKILLS"].values() for s in cat}
            jd_tiers = parse_job_description(jd_text)
            outcome = score_match(resume_skills, jd_tiers)

            all_unrecognized = sorted({
                term
                for tier_data in jd_tiers.values()
                for term in tier_data["unrecognized_terms"]
            })

            st.markdown("---")

            st.markdown(f"""
            <div class="data-row">
                <div class="data-field-key">Match Score</div>
                <div class="data-field-val">{outcome['score']:.1%}</div>
                <span class="tag">[{outcome['earned_points']} / {outcome['max_points']} WEIGHTED POINTS]</span>
            </div>
            """, unsafe_allow_html=True)

            if all_unrecognized:
                st.markdown(f"""
                <div style="color:#6b6b6b; font-size:0.78rem; margin:0.6rem 0 1.2rem; font-style:italic;">
                    Not evaluated (outside skill catalog): {", ".join(all_unrecognized)}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            tier_labels = {
                "required":     "Required",
                "bonus":        "Bonus",
                "nice_to_have": "Nice to Have",
            }
            for tier_key, tier_label in tier_labels.items():
                matched = outcome["matched"][tier_key]
                missing = outcome["missing"][tier_key]
                unrecognized = jd_tiers[tier_key]["unrecognized_terms"]

                extra = ""
                if unrecognized:
                    extra = (
                        f'<div class="skill-list" style="color:#6b6b6b; font-size:0.75rem; '
                        f'font-style:italic; margin-top:0.3rem;">'
                        f'Not evaluated: {" / ".join(unrecognized)}</div>'
                    )

                st.markdown(f"""
                <div class="skill-row">
                    <div class="data-field-key">{tier_label.upper()}</div>
                    <div class="skill-list">Matched: {" / ".join(matched) or "—"}</div>
                    <div class="skill-list" style="color:#6b6b6b;">Missing: {" / ".join(missing) or "—"}</div>
                    {extra}
                </div>
                """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2  — Architecture (use during Week 8 presentation)
# ═════════════════════════════════════════════════════════════════════════════
with tab_arch:
    st.markdown("## Five-Tier Confidence Hierarchy")
    st.caption(
        "Each tier runs in order. A tier is skipped for a field once a "
        "higher-confidence tier has already filled it. This keeps API costs "
        "at zero for well-formatted resumes."
    )

    tiers = [
        ("Tier 1 — EntityRuler",
         "Verified patterns loaded from entity_patterns.json and added before the NER "
         "component in the spaCy pipeline — data-driven, so patterns can be edited or "
         "swapped per deployment without touching code. Guaranteed correct, never "
         "overridden. Tag: [ENTITYRULER]"),
        ("Tier 2 — SpaCy NER",
         "Custom-trained neural Named Entity Recogniser (./model-best). Predicts NAME and "
         "UNIVERSITY labels from context. Output is blocked by the Skill Shield to prevent "
         "tools like Docker being classified as names."),
        ("Tier 3 — Regex",
         "Deterministic rule-based extraction for structured fields: Email, Phone "
         "(10-digit guard against year ranges), GPA/CGPA/CPI, Graduation Batch (max year), "
         "GitHub URL, LinkedIn URL, Degree."),
        ("Tier 4 — Heuristic",
         "Positional logic: Name is almost always the first short line of a resume. "
         "University is found by scanning for keywords: University, Institute, IIT, NIT etc. "
         "Fast and free — no model required."),
        ("Tier 5 — LLM-Gemini",
         "Gemini 1.5 Flash API call. Fires ONLY when Name or University is still 'Not Found' "
         "after Tiers 1–4. Prompt is capped at 1 500 chars. One API call per resume maximum. "
         "Cost-efficiency principle: local is free, LLM costs money — use it last."),
    ]

    for title, desc in tiers:
        st.markdown(f"""
        <div class="arch-tier">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Edge-Case Hardening")

    edge_cases = {
        "Multi-column PDFs": (
            "`fitz page.get_text(sort=True)` re-orders text blocks spatially "
            "(top→bottom, left→right) so two-column résumés parse in reading "
            "order rather than interleaving both columns."
        ),
        "Image-only PDFs": (
            "After extraction, if `full_text.strip()` is empty, a descriptive "
            "`ValueError` is raised immediately — the user sees a clear message "
            "instead of a silent empty parse."
        ),
        "Oversized resumes": (
            "Text is hard-capped at 50 000 characters before entering the spaCy "
            "pipeline to prevent memory spikes on multi-page academic CVs."
        ),
        "Skill false positives": (
            "Whole-word regex matching (`(?<!\\w)skill(?!\\w)`) prevents 'C' "
            "matching inside 'CI/CD', and the Skill Shield blocks tool names "
            "from being promoted to person names."
        ),
        "Model missing": (
            "If `./model-best` is absent, the engine degrades gracefully to "
            "Heuristic + LLM tiers. The sidebar shows '⚠ Degraded' and the "
            "parse still returns useful data."
        ),
    }

    for case, detail in edge_cases.items():
        with st.expander(f"🛡 {case}"):
            st.markdown(detail)

    st.markdown("---")
    st.markdown("## Deployment Checklist")
    st.markdown("""
| Item | Status |
|------|--------|
| `model-best/` folder present | Check before presenting |
| `GEMINI_API_KEY` in `.streamlit/secrets.toml` | For Cloud deployment |
| `requirements.txt` committed to repo | Required by Streamlit Cloud |
| `setup.py` run to auto-download spaCy model | Run once on new machine |
| Tested on multi-column PDF | Torture-test before demo |
| Tested on image-only PDF | Should show clear error |
| Tested with no skills in resume | Should show empty-state card |
    """)
