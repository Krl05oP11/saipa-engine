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

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout: int = 60

    # ChromaDB
    chroma_persist_dir: str = "./chroma_data"

    # Telegram — leave telegram_bot_token empty to disable the Telegram channel.
    telegram_bot_token: str = ""
    telegram_bot_username: str = ""
    telegram_webhook_secret: str = ""
    telegram_webhook_url: str = ""
    # Telegram chat id used to deliver EVAL-IA demo feedback (optional).
    evalia_demo_telegram_id: str = ""

    # WhatsApp
    # NOTE: provider wiring is inconsistent across the codebase (Twilio here,
    # Evolution API in whatsapp/client.py, AWS in the installer). Pick one in E9.
    whatsapp_provider: str = "twilio"
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from: str = "whatsapp:+14155238886"
    # Evolution API (used by whatsapp/client.py)
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
