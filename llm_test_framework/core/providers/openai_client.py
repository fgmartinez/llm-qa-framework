from __future__ import annotations

from llm_test_framework.core.config import ProviderConfig
from llm_test_framework.core.llm_client import LLMClient, LLMResponse


class OpenAIClient(LLMClient):
    """OpenAI-compatible provider (works with any OpenAI-API-compatible service)."""

    def __init__(self, config: ProviderConfig):
        import openai

        self.config = config
        api_key = config.api_key or None
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        params = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **self.config.extra,
            **kwargs,
        }
        resp = self._client.chat.completions.create(**params)
        choice = resp.choices[0]
        usage = resp.usage
        return LLMResponse(
            text=choice.message.content or "",
            model=resp.model,
            provider="openai",
            latency_ms=0.0,  # filled by timed_complete if needed
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            metadata={"finish_reason": choice.finish_reason},
        )
