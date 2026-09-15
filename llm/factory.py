"""
LLM provider factory — settings.llm_provider ("ollama" | "anthropic")
picks the backing chat model. Centralizes two provider-specific quirks
found empirically (not guessed from docs):

- ChatOllama.with_structured_output()'s default method
  ("function_calling") silently fails against qwen2.5:14b-16k -- the
  model narrates about the missing function instead of calling it,
  giving parsed=None with no exception. method="json_schema" works.
- ChatAnthropic has no native format= mode; its structured output goes
  through tool-calling ("function_calling"), which is what actually
  works for Claude models.

Never guess which method a new provider needs -- verify with a real
call before adding it here.
"""
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from config import get_settings

log = logging.getLogger("saipa_engine.llm.factory")

_STRUCTURED_OUTPUT_METHOD = {
    "ollama": "json_schema",
    "anthropic": "function_calling",
}


def get_chat_model(*, temperature: float, model_purpose: str = "chat") -> BaseChatModel:
    """
    model_purpose: "chat" (default) or "grading". Only affects which
    Anthropic model is used (Haiku for chat/RAG/alerts, Sonnet+ for
    EVAL-IA grading -- see config.py). Ollama uses a single local model
    for both purposes; there is no split for a single self-hosted model.
    """
    settings = get_settings()

    if settings.llm_provider == "anthropic":
        model = (
            settings.anthropic_model_grading
            if model_purpose == "grading"
            else settings.anthropic_model_chat
        )
        return ChatAnthropic(model=model, api_key=settings.anthropic_api_key, temperature=temperature)

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        request_timeout=float(settings.ollama_timeout),
    )


def get_structured_llm(schema: type[BaseModel], *, temperature: float, model_purpose: str = "chat"):
    """
    Returns a Runnable bound to structured output for `schema`, using
    the provider-appropriate method. Caller awaits invoke_structured()
    on the result rather than .ainvoke() directly, to get a loud
    failure instead of a silent None on a parse miss.
    """
    settings = get_settings()
    method = _STRUCTURED_OUTPUT_METHOD[settings.llm_provider]
    return get_chat_model(temperature=temperature, model_purpose=model_purpose).with_structured_output(
        schema, method=method, include_raw=True
    )


async def invoke_structured(llm, messages) -> BaseModel:
    """
    Awaits a structured-output Runnable and raises explicitly if
    parsing failed, instead of letting a silent `None` propagate into
    an AttributeError several frames away at the caller.
    """
    result = await llm.ainvoke(messages)
    return _unwrap_structured_result(result)


def invoke_structured_sync(llm, messages) -> BaseModel:
    """Sync counterpart of invoke_structured() -- for call sites already
    running inside a worker thread (asyncio.to_thread), not an event loop."""
    result = llm.invoke(messages)
    return _unwrap_structured_result(result)


def _unwrap_structured_result(result: dict) -> BaseModel:
    if result["parsed"] is None:
        raw = result["raw"].content if result.get("raw") is not None else ""
        raise RuntimeError(
            f"structured output parsing failed: {result.get('parsing_error')}; raw={raw[:300]!r}"
        )
    return result["parsed"]
