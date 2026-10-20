"""Chat model factory.

Gemini uses its native LangChain integration, which round-trips Gemini 3 "thought signatures"
across tool-calling turns. Ollama, OpenAI and any other server use the OpenAI-compatible client.
"""

import openai
from google.genai import errors as genai_errors
from langchain_core.exceptions import ModelError
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_openai import ChatOpenAI

from .config import Settings


def build_chat_model(settings: Settings) -> BaseChatModel:
    api_key = settings.llm_api_key.get_secret_value()
    if settings.llm_provider == "gemini" and not settings.llm_base_url:
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            api_key=api_key,
            temperature=settings.llm_temperature,  # None keeps Google's recommended default for Gemini 3
            request_timeout=settings.llm_timeout_s,
            retries=settings.llm_max_retries,
        )
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.resolved_base_url,
        api_key=api_key or "not-needed",  # Ollama ignores the key
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


# Errors raised by the model provider (network, quota, overload), as opposed to bugs in this code.
PROVIDER_ERRORS: tuple[type[Exception], ...] = (
    openai.APIError,
    genai_errors.APIError,
    ChatGoogleGenerativeAIError,
    ModelError,
)


def is_transient(exc: BaseException) -> bool:
    """True for errors worth retrying later: rate limits, overload, timeouts, 5xx."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and (code == 429 or code >= 500):
        return True
    return any(
        k in type(exc).__name__ for k in ("RateLimit", "Timeout", "Server", "Connection", "Unavailable")
    )


def is_quota_exhausted(exc: BaseException) -> bool:
    """True when a daily (or longer) quota is used up: retrying within minutes cannot succeed."""
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text and ("PerDay" in text or "per day" in text.lower())
