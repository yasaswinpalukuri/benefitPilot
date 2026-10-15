"""Repository hygiene checks that are cheap and catch invisible mistakes."""

from __future__ import annotations

from pathlib import Path

from benefitpilot.config import REPO_ROOT

TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".toml", ".jsonl", ".txt"}
SKIP_DIRS = {".venv", ".git", "__pycache__", ".ruff_cache", ".pytest_cache", "data"}


def _text_files() -> list[Path]:
    files = []
    for p in REPO_ROOT.rglob("*"):
        if SKIP_DIRS.intersection(p.relative_to(REPO_ROOT).parts):
            continue
        if p.is_file() and (p.suffix in TEXT_SUFFIXES or p.name == ".env.example"):
            files.append(p)
    return files


def test_no_control_characters_in_source():
    """A regex once shipped with literal backspace bytes where `\\b` was meant
    (a shell heredoc collapsed the escape). It looked right in an editor and
    silently never matched."""
    bad = {}
    for p in _text_files():
        data = p.read_bytes()
        hits = [i for i, c in enumerate(data) if c < 32 and c not in (9, 10, 13)]
        if hits:
            bad[str(p.relative_to(REPO_ROOT))] = hits[:5]
    assert bad == {}


def test_no_private_ip_addresses():
    """Infrastructure addresses stay out of the repo (use hostnames, e.g. groot)."""
    import re

    private = re.compile(r"\b(?:10|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])|192\.168)\.\d+\.\d+")
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in _text_files()
        if private.search(p.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert offenders == []
