"""Unit tests for evaluator functions (both high-level metrics and helpers)."""

from __future__ import annotations

import pytest

from llm_test_framework.evaluators import (
    MetricResult,
    bleu_score,
    contains_any,
    contains_keywords,
    correctness,
    cosine_similarity,
    faithfulness,
    fluency,
    jaccard_similarity,
    relevance,
    response_length_in_range,
    toxicity,
)

# ---- Correctness ----

class TestCorrectness:
    def test_identical_answers(self):
        r = correctness("The clinic opens at 8 AM.", "The clinic opens at 8 AM.")
        assert r.passed
        assert r.score == pytest.approx(1.0, abs=0.01)

    def test_partial_match(self):
        r = correctness(
            "The clinic opens at 8 AM on Monday.",
            "On Monday the clinic is open from 8 AM.",
            threshold=0.3,
        )
        assert r.passed
        assert r.score > 0.3

    def test_no_match(self):
        r = correctness("The clinic opens at 8 AM.", "Pizza is delicious.")
        assert not r.passed
        assert r.score < 0.2

    def test_empty_response(self):
        r = correctness("", "something")
        assert not r.passed
        assert r.score == 0.0


# ---- Relevance ----

class TestRelevance:
    def test_relevant_response(self):
        r = relevance("What are the clinic hours?", "The clinic hours are 8 AM to 6 PM.")
        assert r.passed
        assert r.score > 0.5

    def test_irrelevant_response(self):
        r = relevance("What are the clinic hours?", "The weather is sunny today.")
        assert r.score < 0.5

    def test_empty_response(self):
        r = relevance("question", "")
        assert not r.passed


# ---- Faithfulness ----

class TestFaithfulness:
    def test_grounded_response(self):
        contexts = ["Dr. Chen is available Tuesday, Thursday, and Saturday."]
        response = "Dr. Chen is available on Tuesday, Thursday, and Saturday."
        r = faithfulness(response, contexts, threshold=0.7)
        assert r.passed

    def test_hallucinated_response(self):
        contexts = ["The clinic is open Monday to Friday."]
        response = "The clinic is open on Sundays and offers free MRIs."
        r = faithfulness(response, contexts, threshold=0.7)
        assert not r.passed

    def test_no_context(self):
        r = faithfulness("any response", [], threshold=0.5)
        assert not r.passed
        assert r.score == 0.0

    def test_empty_response(self):
        r = faithfulness("", ["some context"])
        assert not r.passed


# ---- Toxicity ----

class TestToxicity:
    def test_clean_text(self):
        r = toxicity("The clinic is open on Monday.")
        assert r.passed
        assert r.score == 1.0

    def test_toxic_text(self):
        r = toxicity("You should steal the medicine.")
        assert not r.passed
        assert r.score < 1.0

    def test_custom_patterns(self):
        r = toxicity("This is spam spam spam.", extra_patterns=[r"\bspam\b"])
        assert not r.passed


# ---- Fluency ----

class TestFluency:
    def test_well_formed_sentence(self):
        r = fluency("The clinic is open from 8 AM to 6 PM every weekday.")
        assert r.passed
        assert r.score > 0.5

    def test_empty_response(self):
        r = fluency("")
        assert not r.passed
        assert r.score == 0.0

    def test_single_word(self):
        r = fluency("Yes", threshold=0.3)
        assert r.score > 0.0  # not zero, but low


# ---- MetricResult ----

class TestMetricResult:
    def test_immutable(self):
        r = MetricResult("test", 0.5, True, "ok")
        with pytest.raises(AttributeError):
            r.score = 0.9


# ---- Low-level helpers (unchanged from v1) ----

class TestContainsKeywords:
    def test_all_present(self):
        assert contains_keywords("Dr. Chen is available Tuesday.", ["chen", "tuesday"])

    def test_missing_keyword(self):
        assert not contains_keywords("Dr. Chen is available.", ["chen", "saturday"])

    def test_case_insensitive(self):
        assert contains_keywords("Hello World", ["hello", "WORLD"])


class TestContainsAny:
    def test_one_present(self):
        assert contains_any("Clinic open Monday", ["monday", "sunday"])

    def test_none_present(self):
        assert not contains_any("Clinic open Monday", ["sunday", "saturday"])


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
