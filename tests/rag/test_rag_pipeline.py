"""RAG pipeline tests — data-driven from data/questions_rag.json.

Tests faithfulness, correctness, and context retrieval for RAG scenarios.
Add new scenarios by editing the JSON file; no code changes needed.
"""

from __future__ import annotations

import pytest

from llm_test_framework.core.config import ProviderConfig
from llm_test_framework.core.providers.mock import MockClient
from llm_test_framework.core.rag.pipeline import RAGPipeline, RAGResult, StaticRetriever
from llm_test_framework.data_loader import load_scenarios
from llm_test_framework.evaluators import contains_keywords, correctness, faithfulness

_RAG = load_scenarios("questions_rag.json")
_RAG_IDS = [s["id"] for s in _RAG]


def _build_rag_pipeline(scenario: dict) -> tuple[MockClient, RAGPipeline]:
    """Build a mock-backed RAG pipeline from a single scenario's context."""
    config = ProviderConfig(provider="mock", model="mock-model")
    # Mock returns expected answer when the prompt template is sent
    client = MockClient(config, responses={})
    client._default_response = scenario["expected_answer"]
    retriever = StaticRetriever(scenario["contexts"])
    return client, RAGPipeline(client=client, retriever=retriever)


@pytest.mark.rag
@pytest.mark.parametrize("scenario", _RAG, ids=_RAG_IDS)
class TestRAGFromJSON:
    def test_faithfulness(self, scenario: dict):
        """Response should be grounded in the provided context."""
        _, pipeline = _build_rag_pipeline(scenario)
        result = pipeline.query(scenario["question"])
        threshold = scenario["metrics"]["faithfulness_threshold"]
        faith = faithfulness(result.response.text, scenario["contexts"], threshold=threshold)
        assert faith.passed, (
            f"[{scenario['id']}] faithfulness={faith.score:.3f} < {threshold} | {faith.detail}"
        )

    def test_correctness(self, scenario: dict):
        _, pipeline = _build_rag_pipeline(scenario)
        result = pipeline.query(scenario["question"])
        threshold = scenario["metrics"]["correctness_threshold"]
        corr = correctness(result.response.text, scenario["expected_answer"], threshold=threshold)
        assert corr.passed, (
            f"[{scenario['id']}] correctness={corr.score:.3f} < {threshold} | {corr.detail}"
        )

    def test_expected_keywords(self, scenario: dict):
        _, pipeline = _build_rag_pipeline(scenario)
        result = pipeline.query(scenario["question"])
        assert contains_keywords(result.response.text, scenario["expected_keywords"]), (
            f"[{scenario['id']}] Missing keywords {scenario['expected_keywords']} "
            f"in: {result.response.text!r}"
        )

    def test_contexts_injected_into_prompt(self, scenario: dict):
        """The context should be present in the prompt sent to the LLM."""
        client, pipeline = _build_rag_pipeline(scenario)
        pipeline.query(scenario["question"])
        last_prompt = client.calls[-1]["prompt"]
        for ctx in scenario["contexts"]:
            assert ctx in last_prompt, (
                f"[{scenario['id']}] Context not found in prompt: {ctx[:60]}..."
            )


# ---- Structural / unit tests for the pipeline itself ----

@pytest.mark.rag
class TestRAGPipelineStructure:
    def test_returns_rag_result(self, rag_pipeline: RAGPipeline):
        result = rag_pipeline.query("What are the clinic hours?")
        assert isinstance(result, RAGResult)

    def test_custom_prompt_template(self):
        config = ProviderConfig(provider="mock", model="m")
        client = MockClient(config)
        retriever = StaticRetriever(["Fact A.", "Fact B."])
        pipeline = RAGPipeline(
            client=client,
            retriever=retriever,
            prompt_template="Context: {context}\nQ: {query}\nA:",
        )
        pipeline.query("test question")
        assert client.calls[-1]["prompt"].startswith("Context:")

    def test_empty_retriever(self):
        config = ProviderConfig(provider="mock", model="m")
        client = MockClient(config)
        pipeline = RAGPipeline(client=client, retriever=StaticRetriever([]))
        result = pipeline.query("anything")
        assert result.retrieved_contexts == []
        assert result.response.text


@pytest.mark.rag
class TestStaticRetriever:
    def test_returns_up_to_top_k(self):
        retriever = StaticRetriever(["a", "b", "c", "d"])
        assert len(retriever.retrieve("q", top_k=2)) == 2

    def test_returns_all_if_fewer_than_top_k(self):
        retriever = StaticRetriever(["only one"])
        assert retriever.retrieve("q", top_k=5) == ["only one"]
