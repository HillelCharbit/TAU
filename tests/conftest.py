"""Pytest configuration helpers."""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """Ensure the in-repo ``src`` tree is importable before site packages."""
    src_dir = Path(__file__).resolve().parent.parent / "src"
    src_str = str(src_dir)
    if src_dir.exists() and src_str not in sys.path:
        sys.path.insert(0, src_str)
