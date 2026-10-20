import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from prior_auth.config import Settings
from prior_auth.llm import build_chat_model


def test_gemini_uses_the_native_client() -> None:
    model = build_chat_model(Settings(llm_provider="gemini", llm_model="gemini-x", llm_api_key="k"))  # type: ignore[arg-type]
    assert isinstance(model, ChatGoogleGenerativeAI)


def test_ollama_uses_the_openai_compatible_client_without_a_key() -> None:
    settings = Settings(llm_provider="ollama", llm_model="qwen3:8b", llm_api_key="")  # type: ignore[arg-type]
    model = build_chat_model(settings)
    assert isinstance(model, ChatOpenAI)
    assert str(model.openai_api_base).startswith("http://localhost:11434")
    assert settings.llm_configured


def test_custom_endpoint_requires_a_base_url() -> None:
    settings = Settings(llm_provider="openai_compatible", llm_model="m", llm_api_key="k")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        _ = settings.resolved_base_url


def test_transient_errors_are_recognised() -> None:
    from google.genai import errors as genai_errors

    from prior_auth.llm import is_transient

    busy = genai_errors.ServerError(
        503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}
    )
    bad_request = genai_errors.ClientError(
        400, {"error": {"code": 400, "message": "bad", "status": "INVALID"}}
    )
    assert is_transient(busy)
    assert not is_transient(bad_request)


def test_daily_quota_is_not_treated_as_a_short_wait() -> None:
    from google.genai import errors as genai_errors

    from prior_auth.llm import is_quota_exhausted, is_transient

    body = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "message": "Quota exceeded, quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier",
        }
    }
    daily = genai_errors.ClientError(429, body)
    assert is_transient(daily)
    assert is_quota_exhausted(daily)
