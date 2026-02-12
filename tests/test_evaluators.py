"""Unit tests for evaluator functions."""

from __future__ import annotations

import pytest

from llm_test_framework.evaluators import (
    bleu_score,
    contains_any,
    contains_keywords,
    cosine_similarity,
    jaccard_similarity,
    response_length_in_range,
)


class TestContainsKeywords:
    def test_all_present(self):
        assert contains_keywords("Python is great", ["python", "great"])

    def test_missing_keyword(self):
        assert not contains_keywords("Python is great", ["python", "terrible"])

    def test_case_insensitive(self):
        assert contains_keywords("Hello World", ["hello", "WORLD"])


class TestContainsAny:
    def test_one_present(self):
        assert contains_any("Hello world", ["goodbye", "hello"])

    def test_none_present(self):
        assert not contains_any("Hello world", ["foo", "bar"])


class TestResponseLength:
    def test_within_range(self):
        assert response_length_in_range("one two three", min_words=1, max_words=5)

    def test_below_range(self):
        assert not response_length_in_range("hi", min_words=3)

    def test_above_range(self):
        assert not response_length_in_range("a b c d e", max_words=3)


class TestCosineSimilarity:
    def test_identical_texts(self):
        assert cosine_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_completely_different(self):
        assert cosine_similarity("hello", "goodbye") == pytest.approx(0.0)

    def test_partial_overlap(self):
        score = cosine_similarity("the cat sat", "the dog sat")
        assert 0.3 < score < 0.9

    def test_empty_text(self):
        assert cosine_similarity("", "hello") == pytest.approx(0.0)


class TestJaccardSimilarity:
    def test_identical(self):
        assert jaccard_similarity("a b c", "a b c") == pytest.approx(1.0)

    def test_disjoint(self):
        assert jaccard_similarity("a b", "c d") == pytest.approx(0.0)

    def test_both_empty(self):
        assert jaccard_similarity("", "") == pytest.approx(1.0)


class TestBLEU:
    def test_identical(self):
        score = bleu_score("the cat sat on the mat", "the cat sat on the mat")
        assert score == pytest.approx(1.0)

    def test_no_overlap(self):
        assert bleu_score("hello world", "foo bar baz qux") == pytest.approx(0.0)

    def test_partial_match(self):
        score = bleu_score("the cat sat on the mat", "the cat sat on a mat")
        assert 0.3 < score < 1.0

    def test_empty_candidate(self):
        assert bleu_score("hello", "") == pytest.approx(0.0)
