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
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from config import get_settings

log = logging.getLogger("saipa_engine.llm.factory")

# Single source of truth for valid providers -- main.py's startup check
# imports this same dict's keys rather than duplicating the set, so
# adding a provider here can't silently drift out of sync with what
# main.py accepts.
_STRUCTURED_OUTPUT_METHOD = {
    "ollama": "json_schema",
    "anthropic": "function_calling",
}

# Anthropic's SDK default is 1024 -- too low for a full rubric (up to 40
# structured items) or a ~600-word feedback message (FEEDBACK_SYSTEM's
# own instruction gets close to 1024 tokens by itself). Ollama has no
# equivalent low default (num_predict defers to the server). Verified
# empirically, not assumed from docs.
_ANTHROPIC_MAX_TOKENS = 8192

ModelPurpose = Literal["chat", "grading"]


def get_chat_model(*, temperature: float, model_purpose: ModelPurpose = "chat") -> BaseChatModel:
    """
    model_purpose: "chat" (default) or "grading". Only affects which
    Anthropic model is used (Haiku for chat/RAG/alerts, Sonnet+ for
    EVAL-IA grading -- see config.py). Ollama uses a single local model
    for both purposes; there is no split for a single self-hosted model.
    """
    settings = get_settings()

    if settings.llm_provider not in _STRUCTURED_OUTPUT_METHOD:
        raise RuntimeError(
            f"Unknown llm_provider {settings.llm_provider!r} -- this should be unreachable, "
            "main.py's startup check is supposed to guarantee a valid value."
        )

    if settings.llm_provider == "anthropic":
        model = (
            settings.anthropic_model_grading
            if model_purpose == "grading"
            else settings.anthropic_model_chat
        )
        # `temperature` is deliberately NOT passed here. Verified empirically:
        # the recommended grading model (claude-sonnet-5, Claude 5 family)
        # rejects any explicit temperature != 1 outright ("`temperature` is
        # deprecated for this model", HTTP 400) -- every grading call would
        # fail on the very first request with the recommended default.
        # Older models (e.g. claude-haiku-4-5-20251001) accept a custom
        # temperature fine, but there's no reliable way to tell from the
        # model string alone which regime a given model ID falls into, so
        # `temperature` (still used for Ollama below) is not forwarded to
        # Anthropic at all -- letting the API use its own default/required
        # behavior is the one choice that works for both regimes.
        return ChatAnthropic(
            model=model,
            api_key=settings.anthropic_api_key,
            max_tokens=_ANTHROPIC_MAX_TOKENS,
        )

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        request_timeout=float(settings.ollama_timeout),
    )


def get_structured_llm(schema: type[BaseModel], *, temperature: float, model_purpose: ModelPurpose = "chat"):
    """
    Returns a Runnable bound to structured output for `schema`, using
    the provider-appropriate method. Caller awaits invoke_structured()
    on the result rather than .ainvoke() directly, to get a loud
    failure instead of a silent None on a parse miss.
    """
    settings = get_settings()
    chat_model = get_chat_model(temperature=temperature, model_purpose=model_purpose)  # validates llm_provider
    method = _STRUCTURED_OUTPUT_METHOD[settings.llm_provider]
    return chat_model.with_structured_output(schema, method=method, include_raw=True)


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
