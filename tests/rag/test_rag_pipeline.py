"""RAG pipeline tests — validate retrieval + generation end-to-end."""

from __future__ import annotations

import pytest

from llm_test_framework.core.providers.mock import MockClient
from llm_test_framework.core.rag.pipeline import RAGPipeline, RAGResult, StaticRetriever


@pytest.mark.rag
class TestRAGPipeline:
    def test_pipeline_returns_rag_result(self, rag_pipeline: RAGPipeline):
        result = rag_pipeline.query("Who created Python?")
        assert isinstance(result, RAGResult)
        assert result.query == "Who created Python?"

    def test_retrieved_contexts_are_included(self, rag_pipeline: RAGPipeline):
        result = rag_pipeline.query("Tell me about Python", top_k=2)
        assert len(result.retrieved_contexts) == 2

    def test_prompt_template_is_filled(self, rag_pipeline: RAGPipeline):
        """The prompt sent to the LLM should contain the context and query."""
        rag_pipeline.query("Who created Python?")
        # Check the mock recorded a prompt with the context
        last_call = rag_pipeline.client.calls[-1]
        assert "Guido van Rossum" in last_call["prompt"]
        assert "Who created Python?" in last_call["prompt"]

    def test_custom_prompt_template(self, mock_client: MockClient):
        retriever = StaticRetriever(["Fact A.", "Fact B."])
        pipeline = RAGPipeline(
            client=mock_client,
            retriever=retriever,
            prompt_template="Context: {context}\nQ: {query}\nA:",
        )
        pipeline.query("test question")
        last_call = mock_client.calls[-1]
        assert last_call["prompt"].startswith("Context:")

    def test_empty_retriever(self, mock_client: MockClient):
        retriever = StaticRetriever([])
        pipeline = RAGPipeline(client=mock_client, retriever=retriever)
        result = pipeline.query("anything")
        assert result.retrieved_contexts == []
        assert result.response.text  # still gets a (mock) response


@pytest.mark.rag
class TestStaticRetriever:
    def test_returns_up_to_top_k(self):
        docs = ["a", "b", "c", "d"]
        retriever = StaticRetriever(docs)
        assert len(retriever.retrieve("q", top_k=2)) == 2

    def test_returns_all_if_fewer_than_top_k(self):
        retriever = StaticRetriever(["only one"])
        assert retriever.retrieve("q", top_k=5) == ["only one"]
