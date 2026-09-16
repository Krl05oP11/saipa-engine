"""
Centralised settings for saipa-engine using pydantic-settings.
All values come from environment variables / .env file.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Security
    saipa_api_token: str = ""
    # When true, an empty saipa_api_token is allowed and Bearer auth is skipped.
    # Intended for local development only — never set it in a deployment.
    saipa_dev_mode: bool = False

    # Moodle
    moodle_url: str = "http://localhost:8080"
    moodle_token: str = ""
    # Public hostname Moodle expects in the Host header (used when the engine
    # reaches Moodle through a name that differs from moodle_url, e.g. behind a
    # reverse proxy). Empty = send the host from moodle_url.
    moodle_public_host: str = ""

    # LLM provider — "ollama", "anthropic" or "openai_compatible". Empty is
    # refused at startup unless saipa_dev_mode is on (see main.py), same
    # fail-closed pattern as saipa_api_token: real deployments must choose
    # explicitly, this is the university admin's call, not a default we
    # impose.
    llm_provider: str = ""

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    ollama_timeout: int = 60

    # Anthropic — recommended models only (the admin can override):
    # Haiku for SAIPA chat/RAG/alerts, Sonnet+ for EVAL-IA grading (a
    # weak model gives defensible-looking wrong scores on free-text
    # answers). See local_evalia/README.md "AI Model".
    anthropic_api_key: str = ""
    anthropic_model_chat: str = "claude-haiku-4-5-20251001"
    anthropic_model_grading: str = "claude-sonnet-5"

    # OpenAI-compatible — one generic adapter for any backend that speaks
    # the OpenAI chat-completions API: self-hosted (vLLM, LM Studio, TGI,
    # Ollama's own /v1 shim) or hosted (OpenRouter, Together, etc.). No
    # safe default for base_url -- unlike Ollama this always varies per
    # deployment. api_key may legitimately be empty for a local server
    # that doesn't check it.
    openai_compatible_base_url: str = ""
    openai_compatible_api_key: str = ""
    openai_compatible_model_chat: str = ""
    openai_compatible_model_grading: str = ""

    # Embeddings — always sentence-transformers (CPU, no GPU/API key
    # needed). Not provider-switchable: unlike the LLM, there's no
    # admin decision here, it just runs. Multilingual since course
    # content is in Spanish.
    sentence_transformers_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_data"

    # Telegram — leave telegram_bot_token empty to disable the Telegram channel.
    telegram_bot_token: str = ""
    telegram_bot_username: str = ""
    telegram_webhook_secret: str = ""
    telegram_webhook_url: str = ""
    # Telegram chat id used to deliver EVAL-IA demo feedback (optional).
    evalia_demo_telegram_id: int = 0

    # WhatsApp (E9, 2026-09-16): whatsapp/client.py (Evolution API) is the
    # only real implementation -- Twilio config and the sender.py/webhook.py
    # stubs were dead (zero callers, zero real code) and removed. The
    # channel still isn't mounted as a router in main.py, so client.py has
    # no callers either right now -- kept as the reference implementation
    # for whenever the channel is actually wired up, deliberately not now.
    evolution_api_url: str = ""
    evolution_api_key: str = ""
    evolution_instance_name: str = "saipa"
    whatsapp_webhook_url: str = ""

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
