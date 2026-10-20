# ADR 0005: Switch LLM provider by configuration

**Status:** accepted · **Date:** 2026-10

## Context

The project must run without a paid API key and should not depend on one vendor. Gemini has a free
tier, Ollama runs models locally, and OpenAI and most other servers expose OpenAI-compatible chat
endpoints with tool calling.

Gemini also has an OpenAI-compatible endpoint, but Gemini 3 models attach "thought signatures" to
their tool calls and expect them back on the next turn. The generic OpenAI client does not carry
that metadata, so a multi-turn ReAct loop is fragile on that path.

## Decision

`llm.build_chat_model` picks the client from `LLM_PROVIDER`:

- `gemini`: `ChatGoogleGenerativeAI` (langchain-google-genai), which round-trips thought signatures.
- `ollama`, `openai`, `openai_compatible` (with `LLM_BASE_URL`): `ChatOpenAI` with the matching base URL.

`LLM_TOOL_CHOICE=auto` is available for servers that reject `required`; the graph then nudges the
model when it answers without a tool call. `LLM_TEMPERATURE` defaults to the provider default,
because Google recommends the default temperature for Gemini 3.

## Consequences

- Switching provider is a configuration change, and tests use a scripted model with the same interface.
- Two client libraries to keep up to date.
- Free-tier rate limits are handled by the SDK retries and the `--delay` option of the evaluation.
