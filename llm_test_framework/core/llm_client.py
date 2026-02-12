from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMResponse:
    """Standardised response from any LLM provider."""

    text: str
    model: str
    provider: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(ABC):
    """Abstract base class every provider must implement."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Send a prompt and return a structured response."""

    def timed_complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Wrapper that measures wall-clock latency."""
        start = time.perf_counter()
        response = self.complete(prompt, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # Replace latency with measured wall-clock time
        return LLMResponse(
            text=response.text,
            model=response.model,
            provider=response.provider,
            latency_ms=elapsed_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            metadata=response.metadata,
        )
