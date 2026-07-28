"""
Shared pytest fixtures. Adds files/ to sys.path (parser_engine.py and
job_matcher.py live there, not at the project root) and provides a
session-scoped parser instance so the spaCy model only loads once.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent
FILES_DIR = ROOT / "files"
sys.path.insert(0, str(FILES_DIR))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def resumes_dir():
    return ROOT / "resumes"


@pytest.fixture(scope="session")
def job_descriptions_dir():
    return ROOT / "job_descriptions"


@pytest.fixture(scope="session")
def parser():
    from parser_engine import IndustrialParser

    return IndustrialParser(
        model_path=str(FILES_DIR / "model-best"),
        entity_patterns_path=str(FILES_DIR / "entity_patterns.json"),
    )
