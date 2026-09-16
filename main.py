#!/usr/bin/env python3
"""
saipa-engine — SAIPA AI Backend
Port 8052

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8052 --reload

Or in production:
    uvicorn main:app --host 0.0.0.0 --port 8052 --workers 4
"""
import asyncio
import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from license_validator import validate_license
from llm.factory import _STRUCTURED_OUTPUT_METHOD
from routers.health import router as health_router
from routers.chat import router as chat_router
from routers.index import router as index_router
from routers.analytics import router as analytics_router
from routers.eval import router as eval_router
from telegram.webhook import router as telegram_router

# === Logging ===
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("saipa_engine")

# === License check at startup ===
try:
    validate_license()
except RuntimeError as e:
    log.critical("LICENSE VALIDATION FAILED: %s", e)
    sys.exit(1)

# === API auth check at startup ===
# An empty SAIPA_API_TOKEN makes the engine accept unauthenticated requests.
# Refuse to start in that state unless SAIPA_DEV_MODE is explicitly enabled.
if not settings.saipa_api_token and not settings.saipa_dev_mode:
    log.critical(
        "SAIPA_API_TOKEN is not set. The engine would accept unauthenticated "
        "requests. Set SAIPA_API_TOKEN (e.g. `openssl rand -hex 32`), or set "
        "SAIPA_DEV_MODE=true for local development only."
    )
    sys.exit(1)
if settings.saipa_dev_mode and not settings.saipa_api_token:
    log.warning("SAIPA_DEV_MODE is on and no API token is set — Bearer auth is DISABLED.")

# === LLM provider check at startup ===
# Real deployments must choose explicitly (the university admin's call,
# not ours) -- same fail-closed pattern as SAIPA_API_TOKEN above.
_VALID_LLM_PROVIDERS = tuple(_STRUCTURED_OUTPUT_METHOD.keys())  # single source of truth: llm/factory.py

if not settings.llm_provider:
    if settings.saipa_dev_mode:
        settings.llm_provider = "ollama"
        log.warning("LLM_PROVIDER not set — SAIPA_DEV_MODE is on, defaulting to 'ollama' for local dev.")
    else:
        log.critical(
            "LLM_PROVIDER is not set. Set it to one of %s "
            "(e.g. in saipa-engine.env), or set SAIPA_DEV_MODE=true for local development only.",
            _VALID_LLM_PROVIDERS,
        )
        sys.exit(1)
elif settings.llm_provider not in _VALID_LLM_PROVIDERS:
    log.critical("LLM_PROVIDER must be one of %s, got %r", _VALID_LLM_PROVIDERS, settings.llm_provider)
    sys.exit(1)

# An empty ANTHROPIC_API_KEY with LLM_PROVIDER=anthropic would otherwise
# start fine and only fail per-request with an opaque auth error from the
# Anthropic SDK -- same fail-closed treatment as SAIPA_API_TOKEN above.
if settings.llm_provider == "anthropic" and not settings.anthropic_api_key:
    if settings.saipa_dev_mode:
        log.warning("SAIPA_DEV_MODE is on and ANTHROPIC_API_KEY is not set — every LLM call will fail.")
    else:
        log.critical(
            "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
            "Set it, or set SAIPA_DEV_MODE=true for local development only."
        )
        sys.exit(1)

# An empty OPENAI_COMPATIBLE_BASE_URL (or no model set) with
# LLM_PROVIDER=openai_compatible would otherwise start fine and only fail
# per-request -- same fail-closed treatment as the Anthropic check above.
# api_key is deliberately NOT required here: many self-hosted servers
# (vLLM, LM Studio) don't check it at all.
if settings.llm_provider == "openai_compatible" and (
    not settings.openai_compatible_base_url or not settings.openai_compatible_model_chat
):
    if settings.saipa_dev_mode:
        log.warning(
            "SAIPA_DEV_MODE is on and OPENAI_COMPATIBLE_BASE_URL/MODEL_CHAT "
            "are not fully set — every LLM call will fail."
        )
    else:
        log.critical(
            "LLM_PROVIDER=openai_compatible but OPENAI_COMPATIBLE_BASE_URL and/or "
            "OPENAI_COMPATIBLE_MODEL_CHAT are not set. Set both (OPENAI_COMPATIBLE_MODEL_GRADING "
            "is optional and falls back to the chat model), or set SAIPA_DEV_MODE=true for "
            "local development only."
        )
        sys.exit(1)

# === Telegram webhook secret check at startup ===
# An empty telegram_webhook_secret with the bot enabled makes /telegram/webhook
# accept updates from anyone who finds the URL, not just Telegram's servers --
# same fail-closed pattern as SAIPA_API_TOKEN above.
if settings.telegram_bot_token and not settings.telegram_webhook_secret:
    if settings.saipa_dev_mode:
        log.warning(
            "SAIPA_DEV_MODE is on and TELEGRAM_WEBHOOK_SECRET is not set — "
            "/telegram/webhook will accept unverified requests."
        )
    else:
        log.critical(
            "TELEGRAM_BOT_TOKEN is set but TELEGRAM_WEBHOOK_SECRET is not. "
            "Set it (e.g. `openssl rand -hex 32`, then register the same value "
            "as secret_token in setWebhook), or set SAIPA_DEV_MODE=true for "
            "local development only."
        )
        sys.exit(1)

# === FastAPI app ===
# /docs, /redoc and the raw /openapi.json enumerate every route for anyone
# who can reach this port -- fine in dev, an unnecessary map handed to a
# scanner in production. Off unless SAIPA_DEV_MODE is explicitly enabled.
app = FastAPI(
    title="SAIPA Engine",
    description="AI backend for the SAIPA Moodle plugin system",
    version="0.1.0",
    docs_url="/docs" if settings.saipa_dev_mode else None,
    redoc_url="/redoc" if settings.saipa_dev_mode else None,
    openapi_url="/openapi.json" if settings.saipa_dev_mode else None,
)

# "*" (any origin) is a dev-only convenience -- production must list the
# real caller origins explicitly. This mainly matters if the engine is ever
# reachable from browser JS; the current nginx.conf.example blocks it from
# the internet entirely, but CORS should not depend on that holding true.
_cors_origins = [
    "http://localhost:8080",
    "http://host.docker.internal:8080",
    "http://saipa-moodle:8080",
]
if settings.saipa_dev_mode:
    _cors_origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# === Routers ===
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(index_router)
app.include_router(analytics_router)  # /analytics/risk, /analytics/risk/batch
app.include_router(eval_router)       # /eval/* (EVAL-IA: rubric, questions, exam, grade, portfolio, feedback)
app.include_router(telegram_router)   # /telegram/webhook, /telegram/notify, /telegram/status


@app.on_event("startup")
async def on_startup() -> None:
    log.info("saipa-engine v0.1.0 started on port 8052")
    log.info("LLM provider: %s", settings.llm_provider)
    if settings.llm_provider == "ollama":
        log.info("Ollama model: %s at %s", settings.ollama_model, settings.ollama_base_url)
    elif settings.llm_provider == "openai_compatible":
        log.info(
            "OpenAI-compatible model: %s at %s",
            settings.openai_compatible_model_chat, settings.openai_compatible_base_url,
        )
    log.info("ChromaDB persist dir: %s", settings.chroma_persist_dir)

    # Warm up the embeddings model now -- it downloads from HuggingFace
    # Hub on first construction (hundreds of MB), which must not happen
    # inside a real request's timeout budget. Fails closed with a clear
    # message (e.g. a firewalled/offline install can't reach HF Hub)
    # instead of a raw stack trace from an uncaught startup exception.
    from rag.embeddings import get_embedder
    log.info("Warming up embeddings model (%s)...", settings.sentence_transformers_model)
    try:
        await asyncio.to_thread(lambda: get_embedder().embed_query("warmup"))
    except Exception as exc:
        log.critical(
            "Failed to load the embeddings model (%s): %s. Check network access to "
            "HuggingFace Hub, or that SENTENCE_TRANSFORMERS_MODEL is a valid model name.",
            settings.sentence_transformers_model, exc,
        )
        sys.exit(1)
    log.info("Embeddings model ready.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8052, reload=True)
