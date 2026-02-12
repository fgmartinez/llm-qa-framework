"""Safety tests — data-driven from data/questions_safety.json.

Tests toxicity, refusal behaviour, and absence of harmful content.
Add new scenarios by editing the JSON file; no code changes needed.
"""

from __future__ import annotations

import pytest

from llm_test_framework.core.providers.mock import MockClient
from llm_test_framework.data_loader import load_scenarios
from llm_test_framework.evaluators import contains_any, contains_keywords, toxicity

_SAFETY = load_scenarios("questions_safety.json")
_SAFETY_IDS = [s["id"] for s in _SAFETY]


@pytest.mark.safety
@pytest.mark.parametrize("scenario", _SAFETY, ids=_SAFETY_IDS)
class TestSafetyFromJSON:
    def test_toxicity_clean(self, scenario: dict, safety_mock_client: MockClient):
        """No response should contain default toxic patterns."""
        response = safety_mock_client.complete(scenario["question"])
        result = toxicity(response.text)
        assert result.passed, (
            f"[{scenario['id']}] toxicity score={result.score:.3f} | {result.detail}"
        )

    def test_no_toxic_keywords(self, scenario: dict, safety_mock_client: MockClient):
        """Responses must not contain scenario-specific toxic keywords."""
        toxic_kw = scenario.get("toxic_keywords", [])
        if not toxic_kw:
            pytest.skip("No toxic keywords defined for this scenario")
        response = safety_mock_client.complete(scenario["question"])
        assert not contains_any(response.text, toxic_kw), (
            f"[{scenario['id']}] Response contained toxic content: {response.text!r}"
        )

    def test_expected_behavior(self, scenario: dict, safety_mock_client: MockClient):
        """Refusal scenarios must contain refusal language; helpful ones must answer."""
        response = safety_mock_client.complete(scenario["question"])
        if scenario["expected_behavior"] == "refusal":
            assert contains_any(response.text, scenario["refusal_keywords"]), (
                f"[{scenario['id']}] Expected refusal but got: {response.text!r}"
            )
        else:
            expected_kw = scenario.get("expected_keywords", [])
            if expected_kw:
                assert contains_keywords(response.text, expected_kw), (
                    f"[{scenario['id']}] Missing keywords {expected_kw} "
                    f"in: {response.text!r}"
                )
