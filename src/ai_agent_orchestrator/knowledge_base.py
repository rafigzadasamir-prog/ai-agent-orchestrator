"""Loader for the bundled, offline research knowledge base.

Shared by the ``ResearchWorker`` (which answers lookups) and the
``RuleBasedLLM`` planner (which scans a free-form request for topics it can
actually answer). Keeping this in one place guarantees the two always agree
on which topics exist, so a rule-based plan never asks the research worker
to look up something it cannot find.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_DATA_PATH = Path(__file__).parent / "data" / "knowledge_base.json"


@lru_cache(maxsize=1)
def load_knowledge_base() -> dict[str, dict[str, Any]]:
    with _DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def find_topic(query: str) -> str | None:
    """Best-effort match of a free-text query to a knowledge-base key.

    Tries an exact (case-insensitive) match first, then falls back to
    substring containment in either direction so phrases like "the
    PostgreSQL database" or "Postgres" (via partial overlap) still resolve.
    """
    kb = load_knowledge_base()
    normalized = query.strip().lower()
    if normalized in kb:
        return normalized
    for key in kb:
        if key in normalized or normalized in key:
            return key
    return None


def topics_mentioned_in(text: str) -> list[str]:
    """Every knowledge-base topic whose key appears as a substring of text.

    Returns topics in the order their *first character* appears in the
    text, so results read in the same order the user mentioned them.
    """
    lowered = text.lower()
    hits = [(lowered.index(key), key) for key in load_knowledge_base() if key in lowered]
    hits.sort(key=lambda pair: pair[0])
    return [key for _, key in hits]
