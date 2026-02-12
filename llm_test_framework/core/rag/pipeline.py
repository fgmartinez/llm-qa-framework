"""Minimal RAG pipeline abstraction for testing retrieval-augmented generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from llm_test_framework.core.llm_client import LLMClient, LLMResponse


@dataclass(frozen=True)
class RAGResult:
    """Wraps the LLM response together with the retrieved context."""

    response: LLMResponse
    retrieved_contexts: list[str]
    query: str


class Retriever(ABC):
    """Interface for a document retriever."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Return top_k relevant text chunks for the given query."""


class StaticRetriever(Retriever):
    """Returns pre-defined documents regardless of query. Useful for testing."""

    def __init__(self, documents: list[str]):
        self._documents = documents

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        return self._documents[:top_k]


@dataclass
class RAGPipeline:
    """Connects a retriever to an LLM client for end-to-end RAG testing."""

    client: LLMClient
    retriever: Retriever
    prompt_template: str = field(
        default=(
            "Answer the question based on the context below.\n\n"
            "Context:\n{context}\n\n"
            "Question: {query}\n\n"
            "Answer:"
        )
    )

    def query(self, question: str, top_k: int = 3, **kwargs) -> RAGResult:
        contexts = self.retriever.retrieve(question, top_k=top_k)
        context_block = "\n---\n".join(contexts)
        prompt = self.prompt_template.format(context=context_block, query=question)
        response = self.client.complete(prompt, **kwargs)
        return RAGResult(response=response, retrieved_contexts=contexts, query=question)
