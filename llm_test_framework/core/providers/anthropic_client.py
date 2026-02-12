from __future__ import annotations

from llm_test_framework.core.config import ProviderConfig
from llm_test_framework.core.llm_client import LLMClient, LLMResponse


class AnthropicClient(LLMClient):
    """Anthropic Claude provider."""

    def __init__(self, config: ProviderConfig):
        import anthropic

        self.config = config
        api_key = config.api_key or None
        self._client = anthropic.Anthropic(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        params = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
            **self.config.extra,
            **kwargs,
        }
        resp = self._client.messages.create(**params)
        text = resp.content[0].text if resp.content else ""
        return LLMResponse(
            text=text,
            model=resp.model,
            provider="anthropic",
            latency_ms=0.0,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            metadata={"stop_reason": resp.stop_reason},
        )
