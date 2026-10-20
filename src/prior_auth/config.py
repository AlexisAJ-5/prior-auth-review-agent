"""Runtime configuration, read from environment variables or a local `.env` file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LlmProvider = Literal["gemini", "ollama", "openai", "openai_compatible"]

# OpenAI-compatible endpoints. Gemini normally goes through its native client (see llm.py).
DEFAULT_BASE_URLS: dict[str, str] = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "ollama": "http://localhost:11434/v1",
    "openai": "https://api.openai.com/v1",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: LlmProvider = "gemini"
    llm_model: str = "gemini-3.5-flash-lite"
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str | None = None
    # None = provider default. Google recommends the default for Gemini 3; use 0 elsewhere for repeatability.
    llm_temperature: float | None = None
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    # "required" forces a tool call on every turn (best ReAct discipline). Some OpenAI-compatible
    # servers ignore or reject it; "auto" plus the graph's nudge is the portable fallback.
    llm_tool_choice: Literal["required", "auto"] = "required"

    # Retrieval
    qdrant_url: str = ":memory:"  # Docker Compose sets http://qdrant:6333
    qdrant_collection: str = "policy_guidelines"
    embedding_backend: Literal["fastembed", "hashing"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: str | None = None
    record_search_k: int = 4
    guideline_search_k: int = 3

    # Data and storage
    policies_dir: Path = Path("data/policies")
    examples_dir: Path = Path("evals/cases")  # synthetic cases offered as examples in the UI
    database_path: Path = Path("storage/reviews.db")

    # Limits
    max_agent_turns: int = 12
    max_notes_chars: int = 20_000
    max_document_bytes: int = 10 * 1024 * 1024
    max_document_chars: int = 20_000

    @property
    def resolved_base_url(self) -> str:
        if self.llm_base_url:
            return self.llm_base_url
        if self.llm_provider == "openai_compatible":
            raise ValueError("LLM_BASE_URL is required when LLM_PROVIDER=openai_compatible")
        return DEFAULT_BASE_URLS[self.llm_provider]

    @property
    def llm_configured(self) -> bool:
        # Ollama runs locally and needs no key.
        return bool(self.llm_model) and (
            self.llm_provider == "ollama" or bool(self.llm_api_key.get_secret_value())
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
