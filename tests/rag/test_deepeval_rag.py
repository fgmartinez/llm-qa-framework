"""Comprehensive RAG pipeline tests using DeepEval metrics.

These tests use DeepEval's RAG-specific metrics to evaluate retrieval quality,
contextual relevancy, faithfulness, and overall RAG pipeline performance.
"""

from __future__ import annotations

import pytest

from llm_test_framework.core.llm_client import LLMClient
from llm_test_framework.core.rag.pipeline import RAGPipeline, StaticRetriever
from llm_test_framework.evaluators import (
    DeepEvalMetrics,
    assert_metric,
    create_test_case_from_rag_result,
)


@pytest.fixture
def python_knowledge_retriever() -> StaticRetriever:
    """Retriever with Python programming knowledge."""
    return StaticRetriever(
        [
            "Python was created by Guido van Rossum and first released in 1991.",
            "Python is a high-level, interpreted programming language.",
            "Python emphasizes code readability with significant whitespace.",
            "Python supports multiple programming paradigms: OOP, functional, procedural.",
            "Python has a comprehensive standard library often called 'batteries included'.",
            "Popular Python frameworks include Django, Flask, FastAPI for web development.",
            "NumPy, Pandas, and Scikit-learn are popular for data science.",
        ]
    )


@pytest.fixture
def api_documentation_retriever() -> StaticRetriever:
    """Retriever with API documentation."""
    return StaticRetriever(
        [
            "Endpoint: GET /api/users - Returns list of all users",
            "Endpoint: POST /api/users - Creates a new user. Requires name and email.",
            "Endpoint: GET /api/users/{id} - Returns a specific user by ID",
            "Endpoint: PUT /api/users/{id} - Updates a user. Requires authentication.",
            "Endpoint: DELETE /api/users/{id} - Deletes a user. Requires admin role.",
            "Authentication: All endpoints require Bearer token in Authorization header.",
            "Rate limiting: 100 requests per minute per API key.",
        ]
    )


@pytest.mark.rag
@pytest.mark.slow
class TestRAGFaithfulness:
    """Test that RAG responses are faithful to retrieved context."""

    def test_factual_query_faithfulness(
        self,
        llm_client: LLMClient,
        python_knowledge_retriever: StaticRetriever,
    ):
        """Test RAG faithfulness for a factual query."""
        pipeline = RAGPipeline(client=llm_client, retriever=python_knowledge_retriever)
        result = pipeline.query("Who created Python and when was it released?")

        test_case = create_test_case_from_rag_result(result)
        metric = DeepEvalMetrics.faithfulness(threshold=0.7)

        assert_metric(test_case, metric)
        assert len(result.retrieved_contexts) > 0

    def test_multi_fact_query_faithfulness(
        self,
        llm_client: LLMClient,
        python_knowledge_retriever: StaticRetriever,
    ):
        """Test RAG faithfulness when multiple facts are involved."""
        pipeline = RAGPipeline(client=llm_client, retriever=python_knowledge_retriever)
        result = pipeline.query("What are the key characteristics of Python?")

        test_case = create_test_case_from_rag_result(result)
        metric = DeepEvalMetrics.faithfulness(threshold=0.7)

        assert_metric(test_case, metric)

    def test_technical_documentation_faithfulness(
        self,
        llm_client: LLMClient,
        api_documentation_retriever: StaticRetriever,
    ):
        """Test RAG faithfulness with technical documentation."""
        pipeline = RAGPipeline(client=llm_client, retriever=api_documentation_retriever)
        result = pipeline.query("How do I create a new user via the API?")

        test_case = create_test_case_from_rag_result(result)
        metric = DeepEvalMetrics.faithfulness(threshold=0.7)

        assert_metric(test_case, metric)


@pytest.mark.rag
@pytest.mark.slow
class TestRAGContextualRelevancy:
    """Test that retrieved context is relevant to the query."""

    def test_specific_query_context_relevancy(
        self,
        llm_client: LLMClient,
        python_knowledge_retriever: StaticRetriever,
    ):
        """Test that retrieved context is relevant for a specific query."""
        pipeline = RAGPipeline(client=llm_client, retriever=python_knowledge_retriever)
        result = pipeline.query("Who created Python?")

        test_case = create_test_case_from_rag_result(result)
        metric = DeepEvalMetrics.contextual_relevancy(threshold=0.7)

        assert_metric(test_case, metric)

    def test_broad_query_context_relevancy(
        self,
        llm_client: LLMClient,
        python_knowledge_retriever: StaticRetriever,
    ):
        """Test context relevancy for a broader query."""
        pipeline = RAGPipeline(client=llm_client, retriever=python_knowledge_retriever)
        result = pipeline.query("What is Python used for?")

        test_case = create_test_case_from_rag_result(result)
        metric = DeepEvalMetrics.contextual_relevancy(threshold=0.6)

        assert_metric(test_case, metric)

    def test_api_query_context_relevancy(
        self,
        llm_client: LLMClient,
        api_documentation_retriever: StaticRetriever,
    ):
        """Test context relevancy for API documentation queries."""
        pipeline = RAGPipeline(client=llm_client, retriever=api_documentation_retriever)
        result = pipeline.query("What authentication is required for the API?")

        test_case = create_test_case_from_rag_result(result)
        metric = DeepEvalMetrics.contextual_relevancy(threshold=0.7)

        assert_metric(test_case, metric)


@pytest.mark.rag
@pytest.mark.slow
class TestRAGContextualPrecision:
    """Test the precision of retrieved context."""

    def test_precision_with_expected_answer(
        self,
        llm_client: LLMClient,
        python_knowledge_retriever: StaticRetriever,
    ):
        """Test contextual precision with an expected answer."""
        pipeline = RAGPipeline(client=llm_client, retriever=python_knowledge_retriever)
        result = pipeline.query("When was Python first released?")

        expected_output = "Python was first released in 1991."

        from llm_test_framework.evaluators import create_test_case

        test_case = create_test_case(
            input=result.query,
            actual_output=result.response.text,
            expected_output=expected_output,
            retrieval_context=result.retrieved_contexts,
        )
        metric = DeepEvalMetrics.contextual_precision(threshold=0.7)

        assert_metric(test_case, metric)

    def test_precision_api_documentation(
        self,
        llm_client: LLMClient,
        api_documentation_retriever: StaticRetriever,
    ):
        """Test contextual precision for API documentation."""
        pipeline = RAGPipeline(client=llm_client, retriever=api_documentation_retriever)
        result = pipeline.query("How do I delete a user?")

        expected_output = (
            "Use DELETE /api/users/{id} endpoint. Requires admin role and authentication."
        )

        from llm_test_framework.evaluators import create_test_case

        test_case = create_test_case(
            input=result.query,
            actual_output=result.response.text,
            expected_output=expected_output,
            retrieval_context=result.retrieved_contexts,
        )
        metric = DeepEvalMetrics.contextual_precision(threshold=0.7)

        assert_metric(test_case, metric)


@pytest.mark.rag
@pytest.mark.slow
class TestRAGContextualRecall:
    """Test the recall of retrieved context."""

    def test_recall_with_expected_answer(
        self,
        llm_client: LLMClient,
        python_knowledge_retriever: StaticRetriever,
    ):
        """Test that all relevant context is retrieved."""
        pipeline = RAGPipeline(client=llm_client, retriever=python_knowledge_retriever)
        result = pipeline.query("What programming paradigms does Python support?")

        expected_output = (
            "Python supports object-oriented, functional, and procedural programming paradigms."
        )

        from llm_test_framework.evaluators import create_test_case

        test_case = create_test_case(
            input=result.query,
            actual_output=result.response.text,
            expected_output=expected_output,
            retrieval_context=result.retrieved_contexts,
        )
        metric = DeepEvalMetrics.contextual_recall(threshold=0.7)

        assert_metric(test_case, metric)


@pytest.mark.rag
@pytest.mark.slow
class TestRAGHallucination:
    """Test that RAG pipeline doesn't hallucinate beyond retrieved context."""

    def test_no_hallucination_constrained_query(
        self,
        llm_client: LLMClient,
        python_knowledge_retriever: StaticRetriever,
    ):
        """Test that RAG doesn't hallucinate facts not in context."""
        pipeline = RAGPipeline(client=llm_client, retriever=python_knowledge_retriever)
        result = pipeline.query("Who created Python?")

        test_case = create_test_case_from_rag_result(result)
        metric = DeepEvalMetrics.hallucination(threshold=0.5)

        assert_metric(test_case, metric)

    def test_no_hallucination_technical_docs(
        self,
        llm_client: LLMClient,
        api_documentation_retriever: StaticRetriever,
    ):
        """Test that RAG sticks to documentation facts."""
        pipeline = RAGPipeline(client=llm_client, retriever=api_documentation_retriever)
        result = pipeline.query("What is the rate limit for the API?")

        test_case = create_test_case_from_rag_result(result)
        metric = DeepEvalMetrics.hallucination(threshold=0.5)

        assert_metric(test_case, metric)


@pytest.mark.rag
@pytest.mark.slow
class TestRAGAnswerRelevancy:
    """Test that RAG answers are relevant to the query."""

    def test_answer_relevancy_factual(
        self,
        llm_client: LLMClient,
        python_knowledge_retriever: StaticRetriever,
    ):
        """Test answer relevancy for factual questions."""
        pipeline = RAGPipeline(client=llm_client, retriever=python_knowledge_retriever)
        result = pipeline.query("What is Python?")

        test_case = create_test_case_from_rag_result(result)
        metric = DeepEvalMetrics.answer_relevancy(threshold=0.7)

        assert_metric(test_case, metric)

    def test_answer_relevancy_how_to(
        self,
        llm_client: LLMClient,
        api_documentation_retriever: StaticRetriever,
    ):
        """Test answer relevancy for how-to questions."""
        pipeline = RAGPipeline(client=llm_client, retriever=api_documentation_retriever)
        result = pipeline.query("How do I authenticate with the API?")

        test_case = create_test_case_from_rag_result(result)
        metric = DeepEvalMetrics.answer_relevancy(threshold=0.7)

        assert_metric(test_case, metric)


@pytest.mark.rag
@pytest.mark.parametrize(
    "query,min_contexts",
    [
        ("What is Python?", 1),
        ("Who created Python?", 1),
        ("What are Python frameworks?", 1),
    ],
)
def test_rag_retrieval_parametrized(
    llm_client: LLMClient,
    python_knowledge_retriever: StaticRetriever,
    query: str,
    min_contexts: int,
):
    """Parametrized test ensuring RAG retrieves relevant contexts."""
    pipeline = RAGPipeline(client=llm_client, retriever=python_knowledge_retriever)
    result = pipeline.query(query)

    assert len(result.retrieved_contexts) >= min_contexts
    assert result.response.text

    test_case = create_test_case_from_rag_result(result)
    metric = DeepEvalMetrics.answer_relevancy(threshold=0.6)

    assert_metric(test_case, metric)
