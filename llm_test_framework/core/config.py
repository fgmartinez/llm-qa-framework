from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    provider: str = Field(description="Provider name: 'openai', 'anthropic', or 'mock'")
    model: str = Field(description="Model identifier, e.g. 'gpt-4o', 'claude-sonnet-4-5-20250929'")
    api_key: str = Field(default="", description="API key (reads from env if empty)")
    base_url: str | None = Field(default=None, description="Override base URL for the API")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    timeout: float = Field(default=30.0, gt=0, description="Request timeout in seconds")
    extra: dict = Field(default_factory=dict, description="Provider-specific parameters")


class TestConfig(BaseSettings):
    """Global test configuration loaded from environment / .env files."""

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    default_provider: str = "mock"
    default_model: str = "mock-model"
    log_level: str = "INFO"

    model_config = {"env_prefix": "LLM_TEST_", "env_file": ".env", "extra": "ignore"}
