"""
setup.py  —  Hybrid AI Resume Parser | One-time environment setup
=================================================================
Run once on any new machine (or on Streamlit Cloud via postinstall):

    python setup.py

What it does:
  1. Installs / verifies all pip dependencies
  2. Downloads en_core_web_sm if no spaCy model is present at ./model-best
  3. Prints a clear status report

This script is intentionally simple so you can walk through it during
your Week 8 presentation to show environment reproducibility.
"""

import subprocess
import sys
import os

REQUIRED_PACKAGES = [
    "streamlit>=1.35.0",
    "spacy>=3.7.0",
    "PyMuPDF>=1.24.0",
    "pandas>=2.0.0",
    "google-generativeai>=0.5.0",
]

SPACY_MODEL   = "en_core_web_sm"
LOCAL_MODEL   = "./model-best"


def run(cmd: list[str], label: str) -> bool:
    """Run a subprocess command, print result, return success bool."""
    print(f"\n▶ {label}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ Done")
        return True
    else:
        print(f"  ✗ Failed:\n{result.stderr.strip()}")
        return False


def main():
    print("=" * 58)
    print("  Hybrid AI Resume Parser — Environment Setup")
    print("=" * 58)

    # ── Step 1: pip install ───────────────────────────────────────────
    print("\n[1/3] Installing pip dependencies…")
    ok = run(
        [sys.executable, "-m", "pip", "install", "--quiet"] + REQUIRED_PACKAGES,
        "pip install",
    )
    if not ok:
        print("      Try running manually: pip install -r requirements.txt")

    # ── Step 2: spaCy model ───────────────────────────────────────────
    print("\n[2/3] Checking spaCy model…")

    if os.path.isdir(LOCAL_MODEL):
        print(f"  ✓ Custom model found at '{LOCAL_MODEL}' — skipping download.")
    else:
        print(
            f"  '{LOCAL_MODEL}' not found. "
            f"Downloading fallback model: {SPACY_MODEL}"
        )
        run(
            [sys.executable, "-m", "spacy", "download", SPACY_MODEL],
            f"spacy download {SPACY_MODEL}",
        )
        print(
            f"\n  ⚠  Note: '{SPACY_MODEL}' is a generic model.\n"
            f"     For best results, copy your trained model to '{LOCAL_MODEL}'.\n"
            f"     The parser will still run using heuristics + LLM fallback."
        )

    # ── Step 3: Status report ─────────────────────────────────────────
    print("\n[3/3] Status report")

    # Check spaCy import
    try:
        import spacy  # noqa: F401
        print("  ✓ spaCy importable")
    except ImportError:
        print("  ✗ spaCy not importable — check pip install")

    # Check fitz
    try:
        import fitz  # noqa: F401
        print("  ✓ PyMuPDF (fitz) importable")
    except ImportError:
        print("  ✗ PyMuPDF not importable")

    # Check Gemini SDK
    try:
        import google.generativeai  # noqa: F401
        print("  ✓ google-generativeai importable")
    except ImportError:
        print("  ⚠  google-generativeai not installed — LLM tier will be disabled")

    # Check model
    if os.path.isdir(LOCAL_MODEL):
        print(f"  ✓ Model at '{LOCAL_MODEL}'")
    else:
        try:
            import spacy
            spacy.load(SPACY_MODEL)
            print(f"  ✓ Fallback model '{SPACY_MODEL}' loadable")
        except Exception:
            print(f"  ✗ No model available — run: python -m spacy download {SPACY_MODEL}")

    # Check Gemini key
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        print("  ✓ GEMINI_API_KEY env variable set")
    else:
        print("  ⚠  GEMINI_API_KEY not set — LLM tier will need key via UI or secrets.toml")

    print("\n" + "=" * 58)
    print("  Setup complete. Launch the app with:")
    print("    streamlit run app.py")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    main()
