"""Lightweight similarity functions that work without heavy ML dependencies."""

from __future__ import annotations

import math
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Bag-of-words cosine similarity between two texts. Returns 0.0–1.0."""
    tokens_a = Counter(_tokenize(text_a))
    tokens_b = Counter(_tokenize(text_b))

    common = set(tokens_a) & set(tokens_b)
    dot = sum(tokens_a[w] * tokens_b[w] for w in common)
    mag_a = math.sqrt(sum(v**2 for v in tokens_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in tokens_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity. Returns 0.0–1.0."""
    set_a = set(_tokenize(text_a))
    set_b = set(_tokenize(text_b))

    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)
