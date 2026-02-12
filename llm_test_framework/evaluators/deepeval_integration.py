"""Integration layer for DeepEval evaluation framework.

This module provides wrappers and utilities to use DeepEval metrics
with our LLMClient responses and RAG pipelines.
"""

from __future__ import annotations

from typing import Any

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ToxicityMetric,
)
from deepeval.test_case import LLMTestCase

from llm_test_framework.core.llm_client import LLMResponse
from llm_test_framework.core.rag.pipeline import RAGResult


def create_test_case(
    *,
    input: str,
    actual_output: str,
    expected_output: str | None = None,
    context: list[str] | None = None,
    retrieval_context: list[str] | None = None,
) -> LLMTestCase:
    """Create a DeepEval test case from our framework's data structures.

    Args:
        input: The user's input/prompt
        actual_output: The LLM's actual response
        expected_output: Optional expected/reference output
        context: Optional context used for generation (for RAG)
        retrieval_context: Optional retrieved context (for RAG)

    Returns:
        LLMTestCase ready for DeepEval metric evaluation
    """
    return LLMTestCase(
        input=input,
        actual_output=actual_output,
        expected_output=expected_output,
        context=context,
        retrieval_context=retrieval_context,
    )


def create_test_case_from_response(
    prompt: str,
    response: LLMResponse,
    expected_output: str | None = None,
) -> LLMTestCase:
    """Create a DeepEval test case from an LLMResponse.

    Args:
        prompt: The original prompt sent to the LLM
        response: The LLMResponse object
        expected_output: Optional expected/reference output

    Returns:
        LLMTestCase ready for DeepEval metric evaluation
    """
    return create_test_case(
        input=prompt,
        actual_output=response.text,
        expected_output=expected_output,
    )


def create_test_case_from_rag_result(
    rag_result: RAGResult,
    expected_output: str | None = None,
) -> LLMTestCase:
    """Create a DeepEval test case from a RAG pipeline result.

    Args:
        rag_result: The RAGResult from our pipeline
        expected_output: Optional expected/reference output

    Returns:
        LLMTestCase ready for DeepEval RAG metric evaluation
    """
    return create_test_case(
        input=rag_result.query,
        actual_output=rag_result.response.text,
        expected_output=expected_output,
        context=rag_result.retrieved_contexts,
        retrieval_context=rag_result.retrieved_contexts,
    )


class DeepEvalMetrics:
    """Convenience wrapper for common DeepEval metrics with sensible defaults."""

    @staticmethod
    def answer_relevancy(threshold: float = 0.7, model: str | None = None) -> AnswerRelevancyMetric:
        """Measures how relevant the answer is to the input query.

        Args:
            threshold: Minimum score to pass (0.0-1.0)
            model: Optional model to use for evaluation (defaults to gpt-4)

        Returns:
            Configured AnswerRelevancyMetric
        """
        kwargs: dict[str, Any] = {"threshold": threshold}
        if model:
            kwargs["model"] = model
        return AnswerRelevancyMetric(**kwargs)

    @staticmethod
    def faithfulness(threshold: float = 0.7, model: str | None = None) -> FaithfulnessMetric:
        """Measures if the output is faithful to the provided context.

        Args:
            threshold: Minimum score to pass (0.0-1.0)
            model: Optional model to use for evaluation

        Returns:
            Configured FaithfulnessMetric
        """
        kwargs: dict[str, Any] = {"threshold": threshold}
        if model:
            kwargs["model"] = model
        return FaithfulnessMetric(**kwargs)

    @staticmethod
    def hallucination(threshold: float = 0.5, model: str | None = None) -> HallucinationMetric:
        """Detects if the LLM is hallucinating facts not in the context.

        Args:
            threshold: Maximum hallucination score allowed (0.0-1.0, lower is better)
            model: Optional model to use for evaluation

        Returns:
            Configured HallucinationMetric
        """
        kwargs: dict[str, Any] = {"threshold": threshold}
        if model:
            kwargs["model"] = model
        return HallucinationMetric(**kwargs)

    @staticmethod
    def contextual_relevancy(threshold: float = 0.7, model: str | None = None) -> ContextualRelevancyMetric:
        """Measures if the retrieved context is relevant to the input.

        Args:
            threshold: Minimum score to pass (0.0-1.0)
            model: Optional model to use for evaluation

        Returns:
            Configured ContextualRelevancyMetric
        """
        kwargs: dict[str, Any] = {"threshold": threshold}
        if model:
            kwargs["model"] = model
        return ContextualRelevancyMetric(**kwargs)

    @staticmethod
    def contextual_precision(threshold: float = 0.7, model: str | None = None) -> ContextualPrecisionMetric:
        """Measures the precision of retrieved context (RAG).

        Args:
            threshold: Minimum score to pass (0.0-1.0)
            model: Optional model to use for evaluation

        Returns:
            Configured ContextualPrecisionMetric
        """
        kwargs: dict[str, Any] = {"threshold": threshold}
        if model:
            kwargs["model"] = model
        return ContextualPrecisionMetric(**kwargs)

    @staticmethod
    def contextual_recall(threshold: float = 0.7, model: str | None = None) -> ContextualRecallMetric:
        """Measures the recall of retrieved context (RAG).

        Args:
            threshold: Minimum score to pass (0.0-1.0)
            model: Optional model to use for evaluation

        Returns:
            Configured ContextualRecallMetric
        """
        kwargs: dict[str, Any] = {"threshold": threshold}
        if model:
            kwargs["model"] = model
        return ContextualRecallMetric(**kwargs)

    @staticmethod
    def toxicity(threshold: float = 0.5, model: str | None = None) -> ToxicityMetric:
        """Detects toxic, harmful, or offensive content.

        Args:
            threshold: Maximum toxicity score allowed (0.0-1.0, lower is better)
            model: Optional model to use for evaluation

        Returns:
            Configured ToxicityMetric
        """
        kwargs: dict[str, Any] = {"threshold": threshold}
        if model:
            kwargs["model"] = model
        return ToxicityMetric(**kwargs)

    @staticmethod
    def bias(threshold: float = 0.5, model: str | None = None) -> BiasMetric:
        """Detects biased content in the output.

        Args:
            threshold: Maximum bias score allowed (0.0-1.0, lower is better)
            model: Optional model to use for evaluation

        Returns:
            Configured BiasMetric
        """
        kwargs: dict[str, Any] = {"threshold": threshold}
        if model:
            kwargs["model"] = model
        return BiasMetric(**kwargs)


def assert_metric(test_case: LLMTestCase, metric: Any) -> None:
    """Evaluate a metric and assert it passes.

    Args:
        test_case: The LLMTestCase to evaluate
        metric: A DeepEval metric instance

    Raises:
        AssertionError: If the metric fails
    """
    metric.measure(test_case)
    assert metric.is_successful(), (
        f"{metric.__class__.__name__} failed: "
        f"score={metric.score:.3f}, threshold={metric.threshold}, "
        f"reason={getattr(metric, 'reason', 'N/A')}"
    )
