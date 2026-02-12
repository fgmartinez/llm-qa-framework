"""Performance tests — latency and throughput checks using clinic QA scenarios."""

from __future__ import annotations

import pytest

from llm_test_framework.core.providers.mock import MockClient
from llm_test_framework.data_loader import load_scenarios

_QA = load_scenarios("questions_qa.json")


@pytest.mark.performance
class TestLatency:
    def test_single_request_latency(self, mock_client: MockClient):
        response = mock_client.timed_complete(_QA[0]["question"])
        assert response.latency_ms < 100, (
            f"Latency {response.latency_ms:.1f}ms exceeds threshold"
        )

    def test_throughput_across_scenarios(self, mock_client: MockClient):
        """Run every QA scenario and check average latency."""
        total_ms = 0.0
        for scenario in _QA:
            resp = mock_client.timed_complete(scenario["question"])
            total_ms += resp.latency_ms
        avg = total_ms / len(_QA)
        assert avg < 50, f"Average latency {avg:.1f}ms exceeds threshold"

    def test_response_includes_token_counts(self, mock_client: MockClient):
        response = mock_client.complete(_QA[0]["question"])
        assert response.prompt_tokens > 0
        assert response.completion_tokens > 0
        assert response.total_tokens == response.prompt_tokens + response.completion_tokens
