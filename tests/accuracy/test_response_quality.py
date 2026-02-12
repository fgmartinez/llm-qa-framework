"""Accuracy tests — data-driven from data/questions_qa.json.

Tests correctness, relevance, and fluency for each QA scenario.
Add new scenarios by editing the JSON file; no code changes needed.
"""

from __future__ import annotations

import pytest

from llm_test_framework.core.providers.mock import MockClient
from llm_test_framework.data_loader import load_scenarios
from llm_test_framework.evaluators import (
    contains_keywords,
    correctness,
    fluency,
    relevance,
)

_QA = load_scenarios("questions_qa.json")
_QA_IDS = [s["id"] for s in _QA]


@pytest.mark.accuracy
@pytest.mark.parametrize("scenario", _QA, ids=_QA_IDS)
class TestCorrectnessFromJSON:
    """Each QA scenario is evaluated for correctness against its expected answer."""

    def test_correctness(self, scenario: dict, mock_client: MockClient):
        response = mock_client.complete(scenario["question"])
        threshold = scenario["metrics"]["correctness_threshold"]
        result = correctness(response.text, scenario["expected_answer"], threshold=threshold)
        assert result.passed, (
            f"[{scenario['id']}] correctness={result.score:.3f} < {threshold} | {result.detail}"
        )

    def test_expected_keywords(self, scenario: dict, mock_client: MockClient):
        response = mock_client.complete(scenario["question"])
        assert contains_keywords(response.text, scenario["expected_keywords"]), (
            f"[{scenario['id']}] Missing keywords {scenario['expected_keywords']} "
            f"in: {response.text!r}"
        )


@pytest.mark.accuracy
@pytest.mark.parametrize("scenario", _QA, ids=_QA_IDS)
class TestRelevanceFromJSON:
    def test_relevance(self, scenario: dict, mock_client: MockClient):
        response = mock_client.complete(scenario["question"])
        threshold = scenario["metrics"]["relevance_threshold"]
        result = relevance(scenario["question"], response.text, threshold=threshold)
        assert result.passed, (
            f"[{scenario['id']}] relevance={result.score:.3f} < {threshold} | {result.detail}"
        )


@pytest.mark.accuracy
@pytest.mark.parametrize("scenario", _QA, ids=_QA_IDS)
class TestFluencyFromJSON:
    def test_fluency(self, scenario: dict, mock_client: MockClient):
        response = mock_client.complete(scenario["question"])
        result = fluency(response.text, threshold=0.4)
        assert result.passed, (
            f"[{scenario['id']}] fluency={result.score:.3f} | {result.detail}"
        )
