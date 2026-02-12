"""Simple, dependency-light evaluation metrics for LLM responses."""

from __future__ import annotations

import math
from collections import Counter


def contains_keywords(text: str, keywords: list[str]) -> bool:
    """Return True if *all* keywords appear in the text (case-insensitive)."""
    lower = text.lower()
    return all(kw.lower() in lower for kw in keywords)


def contains_any(text: str, keywords: list[str]) -> bool:
    """Return True if *any* keyword appears in the text (case-insensitive)."""
    lower = text.lower()
    return any(kw.lower() in lower for kw in keywords)


def response_length_in_range(text: str, min_words: int = 1, max_words: int = 10_000) -> bool:
    """Check that word count is within [min_words, max_words]."""
    count = len(text.split())
    return min_words <= count <= max_words


def bleu_score(reference: str, candidate: str, max_n: int = 4) -> float:
    """Simplified BLEU score (no brevity penalty smoothing). Returns 0.0–1.0."""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    if not cand_tokens or not ref_tokens:
        return 0.0

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(_ngrams(ref_tokens, n))
        cand_ngrams = Counter(_ngrams(cand_tokens, n))
        clipped = sum(min(cand_ngrams[ng], ref_ngrams[ng]) for ng in cand_ngrams)
        total = sum(cand_ngrams.values())
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped / total)

    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / len(precisions)

    brevity = min(1.0, len(cand_tokens) / len(ref_tokens))
    bp = math.exp(1 - 1 / brevity) if brevity < 1 else 1.0

    return bp * math.exp(log_avg)


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
