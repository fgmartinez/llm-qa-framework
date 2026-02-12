"""Load test scenarios from JSON files."""

from __future__ import annotations

import json
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_scenarios(filename: str) -> list[dict]:
    """Load the 'scenarios' list from a JSON file in the data/ directory."""
    path = _DATA_DIR / filename
    data = json.loads(path.read_text())
    return data["scenarios"]


def load_knowledge_base(filename: str = "clinic_knowledge_base.md") -> str:
    """Load the full text of a knowledge-base document."""
    path = _DATA_DIR / filename
    return path.read_text()
