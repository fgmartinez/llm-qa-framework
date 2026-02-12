"""Shared pytest fixtures available to every test module."""

from __future__ import annotations

import pytest

from llm_test_framework.core.config import ProviderConfig, TestConfig
from llm_test_framework.core.providers import create_client
from llm_test_framework.core.providers.mock import MockClient
from llm_test_framework.core.rag.pipeline import RAGPipeline, StaticRetriever
from llm_test_framework.data_loader import load_knowledge_base, load_scenarios

# ---- Scenario data (loaded once per session) ----

@pytest.fixture(scope="session")
def qa_scenarios() -> list[dict]:
    return load_scenarios("questions_qa.json")


@pytest.fixture(scope="session")
def rag_scenarios() -> list[dict]:
    return load_scenarios("questions_rag.json")


@pytest.fixture(scope="session")
def safety_scenarios() -> list[dict]:
    return load_scenarios("questions_safety.json")


@pytest.fixture(scope="session")
def clinic_knowledge_base() -> str:
    return load_knowledge_base()


# ---- Config & clients ----

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
def mock_client(qa_scenarios: list[dict]) -> MockClient:
    """Mock client with canned responses from the QA dataset."""
    config = ProviderConfig(provider="mock", model="mock-model")
    responses = {s["question"]: s["expected_answer"] for s in qa_scenarios}
    return MockClient(config, responses=responses)


@pytest.fixture()
def safety_mock_client(safety_scenarios: list[dict]) -> MockClient:
    """Mock client with canned responses for safety scenarios."""
    config = ProviderConfig(provider="mock", model="mock-model")
    responses = {}
    for s in safety_scenarios:
        if s["expected_behavior"] == "refusal":
            responses[s["question"]] = "I cannot help with that request."
        elif "expected_keywords" in s:
            # Build a realistic mock response containing all expected keywords
            responses[s["question"]] = " ".join(s["expected_keywords"]) + "."
    return MockClient(config, responses=responses)


@pytest.fixture()
def rag_pipeline(mock_client: MockClient, clinic_knowledge_base: str) -> RAGPipeline:
    """RAG pipeline backed by the clinic knowledge base split into paragraphs."""
    paragraphs = [p.strip() for p in clinic_knowledge_base.split("\n\n") if p.strip()]
    retriever = StaticRetriever(paragraphs)
    return RAGPipeline(client=mock_client, retriever=retriever)
