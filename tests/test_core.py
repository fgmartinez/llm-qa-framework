"""Unit tests for core modules: config, client abstraction, factory."""

from __future__ import annotations

import pytest

from llm_test_framework.core.config import ProviderConfig
from llm_test_framework.core.llm_client import LLMResponse
from llm_test_framework.core.providers import create_client
from llm_test_framework.core.providers.mock import MockClient


class TestProviderConfig:
    def test_default_values(self):
        cfg = ProviderConfig(provider="mock", model="m")
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 1024
        assert cfg.timeout == 30.0

    def test_temperature_bounds(self):
        with pytest.raises(ValueError):
            ProviderConfig(provider="mock", model="m", temperature=3.0)

    def test_extra_params_forwarded(self):
        cfg = ProviderConfig(provider="mock", model="m", extra={"top_p": 0.9})
        assert cfg.extra["top_p"] == 0.9


class TestLLMResponse:
    def test_total_tokens(self):
        r = LLMResponse(
            text="hi", model="m", provider="mock", latency_ms=1.0,
            prompt_tokens=10, completion_tokens=5,
        )
        assert r.total_tokens == 15

    def test_immutable(self):
        r = LLMResponse(text="hi", model="m", provider="mock", latency_ms=1.0)
        with pytest.raises(AttributeError):
            r.text = "changed"


class TestFactory:
    def test_create_mock_client(self):
        cfg = ProviderConfig(provider="mock", model="mock-model")
        client = create_client(cfg)
        assert isinstance(client, MockClient)

    def test_unknown_provider_raises(self):
        cfg = ProviderConfig(provider="unknown", model="x")
        with pytest.raises(ValueError, match="Unknown provider"):
            create_client(cfg)


class TestMockClient:
    def test_records_calls(self):
        cfg = ProviderConfig(provider="mock", model="m")
        client = MockClient(cfg)
        client.complete("hello")
        client.complete("world")
        assert len(client.calls) == 2
        assert client.calls[0]["prompt"] == "hello"

    def test_custom_responses(self):
        cfg = ProviderConfig(provider="mock", model="m")
        client = MockClient(cfg, responses={"ping": "pong"})
        assert client.complete("ping").text == "pong"

    def test_default_response_fallback(self):
        cfg = ProviderConfig(provider="mock", model="m")
        client = MockClient(cfg)
        assert client.complete("anything").text == "This is a mock response."
