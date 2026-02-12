from __future__ import annotations

from llm_test_framework.core.config import ProviderConfig
from llm_test_framework.core.llm_client import LLMClient, LLMResponse


class MockClient(LLMClient):
    """Deterministic mock client for unit testing without API calls."""

    def __init__(self, config: ProviderConfig, responses: dict[str, str] | None = None):
        self.config = config
        self._responses = responses or {}
        self._default_response = "This is a mock response."
        self._call_log: list[dict] = []

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        self._call_log.append({"prompt": prompt, **kwargs})
        text = self._responses.get(prompt, self._default_response)
        return LLMResponse(
            text=text,
            model=self.config.model,
            provider="mock",
            latency_ms=1.0,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
        )

    @property
    def calls(self) -> list[dict]:
        return list(self._call_log)
