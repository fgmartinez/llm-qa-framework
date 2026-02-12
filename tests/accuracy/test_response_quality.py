"""Accuracy tests — validate that LLM responses meet quality expectations."""

from __future__ import annotations

import pytest

from llm_test_framework.core.providers.mock import MockClient
from llm_test_framework.evaluators import (
    bleu_score,
    contains_keywords,
    cosine_similarity,
    response_length_in_range,
)


@pytest.mark.accuracy
class TestResponseQuality:
    def test_response_contains_expected_keywords(self, mock_client: MockClient):
        response = mock_client.complete("What is Python?")
        assert contains_keywords(response.text, ["python", "language"])

    def test_response_similarity_to_reference(self, mock_client: MockClient):
        response = mock_client.complete("What is Python?")
        reference = "Python is a popular high-level programming language."
        similarity = cosine_similarity(response.text, reference)
        assert similarity > 0.5, f"Similarity {similarity:.3f} below threshold"

    def test_response_length_within_bounds(self, mock_client: MockClient):
        response = mock_client.complete("Say hello")
        assert response_length_in_range(response.text, min_words=2, max_words=50)

    def test_bleu_against_reference(self, mock_client: MockClient):
        response = mock_client.complete("What is 2+2?")
        reference = "The answer is 4."
        score = bleu_score(reference, response.text)
        assert score > 0.5, f"BLEU {score:.3f} below threshold"

    def test_deterministic_mock_output(self, mock_client: MockClient):
        """Mock client should return the same output for the same prompt."""
        r1 = mock_client.complete("What is 2+2?")
        r2 = mock_client.complete("What is 2+2?")
        assert r1.text == r2.text


@pytest.mark.accuracy
@pytest.mark.parametrize(
    "prompt,expected_keywords",
    [
        ("What is Python?", ["python"]),
        ("Say hello", ["hello"]),
        ("What is 2+2?", ["4"]),
    ],
)
def test_keyword_presence_parametrized(
    mock_client: MockClient, prompt: str, expected_keywords: list[str]
):
    response = mock_client.complete(prompt)
    assert contains_keywords(response.text, expected_keywords)
