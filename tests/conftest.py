"""Shared pytest fixtures available to every test module."""

from __future__ import annotations

import pytest

from llm_test_framework.core.config import ProviderConfig, TestConfig
from llm_test_framework.core.providers import create_client
from llm_test_framework.core.providers.mock import MockClient
from llm_test_framework.core.rag.pipeline import RAGPipeline, StaticRetriever


@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    return TestConfig()


@pytest.fixture()
def provider_config(test_config: TestConfig) -> ProviderConfig:
    """Default provider config — uses mock by default, override via env vars."""
    return ProviderConfig(
        provider=test_config.default_provider,
        model=test_config.default_model,
        api_key="",
    )


@pytest.fixture()
def llm_client(provider_config: ProviderConfig):
    return create_client(provider_config)


@pytest.fixture()
def mock_client() -> MockClient:
    """A mock client with some canned responses for testing."""
    config = ProviderConfig(provider="mock", model="mock-model")
    responses = {
        "What is Python?": "Python is a high-level programming language.",
        "Say hello": "Hello! How can I help you today?",
        "What is 2+2?": "The answer is 4.",
    }
    return MockClient(config, responses=responses)


@pytest.fixture()
def rag_pipeline(mock_client: MockClient) -> RAGPipeline:
    docs = [
        "Python was created by Guido van Rossum and first released in 1991.",
        "Python supports multiple programming paradigms including OOP and functional.",
        "The Python Package Index (PyPI) hosts thousands of third-party packages.",
    ]
    retriever = StaticRetriever(docs)
    return RAGPipeline(client=mock_client, retriever=retriever)
