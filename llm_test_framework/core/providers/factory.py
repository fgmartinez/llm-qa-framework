from __future__ import annotations

from llm_test_framework.core.config import ProviderConfig
from llm_test_framework.core.llm_client import LLMClient


def create_client(config: ProviderConfig) -> LLMClient:
    """Instantiate the right LLM client based on config.provider."""
    match config.provider:
        case "openai":
            from llm_test_framework.core.providers.openai_client import OpenAIClient

            return OpenAIClient(config)
        case "anthropic":
            from llm_test_framework.core.providers.anthropic_client import AnthropicClient

            return AnthropicClient(config)
        case "mock":
            from llm_test_framework.core.providers.mock import MockClient

            return MockClient(config)
        case _:
            raise ValueError(
                f"Unknown provider '{config.provider}'. "
                "Supported: 'openai', 'anthropic', 'mock'."
            )
