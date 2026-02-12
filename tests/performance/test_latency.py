"""Performance tests — latency and throughput checks."""

from __future__ import annotations

import pytest

from llm_test_framework.core.providers.mock import MockClient


@pytest.mark.performance
class TestLatency:
    def test_single_request_latency(self, mock_client: MockClient):
        response = mock_client.timed_complete("Say hello")
        # Mock should be effectively instant (< 100ms)
        assert response.latency_ms < 100, f"Latency {response.latency_ms:.1f}ms exceeds threshold"

    def test_throughput_multiple_requests(self, mock_client: MockClient):
        """Run N requests and check average latency."""
        n = 20
        total_ms = 0.0
        for _ in range(n):
            resp = mock_client.timed_complete("What is 2+2?")
            total_ms += resp.latency_ms
        avg = total_ms / n
        assert avg < 50, f"Average latency {avg:.1f}ms exceeds threshold"

    def test_response_includes_token_counts(self, mock_client: MockClient):
        response = mock_client.complete("What is Python?")
        assert response.prompt_tokens > 0
        assert response.completion_tokens > 0
        assert response.total_tokens == response.prompt_tokens + response.completion_tokens
