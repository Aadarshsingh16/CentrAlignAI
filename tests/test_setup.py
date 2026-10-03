import os
from pathlib import Path
from dotenv import load_dotenv

def test_project_structure():
    root = Path(__file__).parent.parent
    assert (root / "requirements.txt").exists(), "requirements.txt missing"
    assert (root / ".env.example").exists(), ".env.example missing"
    assert (root / ".gitignore").exists(), ".gitignore missing"
    assert (root / "PLAN.md").exists(), "PLAN.md missing"
    assert (root / "AGENTS.md").exists(), "AGENTS.md missing"
    assert (root / "EVALS.md").exists(), "EVALS.md missing"

def test_gitignore_ignores_env():
    root = Path(__file__).parent.parent
    gitignore_content = (root / ".gitignore").read_text(encoding="utf-8")
    lines = [line.strip() for line in gitignore_content.splitlines()]
    assert ".env" in lines, ".env must be ignored in .gitignore"

def test_dependencies_importable():
    import flask
    import playwright
    import pytest
    import yaml
    from dotenv import load_dotenv
    assert flask is not None
    assert playwright is not None
    assert pytest is not None
    assert yaml is not None
    assert load_dotenv is not None

def test_gemini_api_key_configured():
    root = Path(__file__).parent.parent
    load_dotenv(root / ".env")
    key = os.getenv("GEMINI_API_KEY", "").strip()
    assert len(key) > 0, "GEMINI_API_KEY is not set or empty in .env"
