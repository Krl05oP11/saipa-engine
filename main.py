#!/usr/bin/env python3
"""
saipa-engine — SAIPA AI Backend
Port 8052

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8052 --reload

Or in production:
    uvicorn main:app --host 0.0.0.0 --port 8052 --workers 4
"""
import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from license_validator import validate_license
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

# === FastAPI app ===
app = FastAPI(
    title="SAIPA Engine",
    description="AI backend for the SAIPA Moodle plugin system",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://host.docker.internal:8080",
        "http://saipa-moodle:8080",
        "*",  # Tighten this in production
    ],
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
    log.info("Ollama model: %s at %s", settings.ollama_model, settings.ollama_base_url)
    log.info("ChromaDB persist dir: %s", settings.chroma_persist_dir)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8052, reload=True)
