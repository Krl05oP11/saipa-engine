"""
Telegram Bot API client for SAIPA.
Thin async wrapper over the official Telegram Bot API (plain HTTPS — no extra library needed).

Singleton exposed via get_telegram_client().
"""
import logging

import httpx

from config import get_settings

log = logging.getLogger("saipa_engine.telegram")


class TelegramClient:

    def __init__(self) -> None:
        settings = get_settings()
        self._token   = settings.telegram_bot_token
        self._secret  = settings.telegram_webhook_secret
        self._botname = settings.telegram_bot_username
        self._base    = f"https://api.telegram.org/bot{self._token}"

    def enabled(self) -> bool:
        return bool(self._token)

    async def start(self) -> None:
        """Register webhook with Telegram on engine startup."""
        if not self.enabled():
            log.warning("Telegram bot token not configured — Telegram channel disabled")
            return
        settings = get_settings()
        if not settings.telegram_webhook_url:
            log.warning("TELEGRAM_WEBHOOK_URL not set — skipping setWebhook (manual registration required)")
            return
        try:
            await self._set_webhook(settings.telegram_webhook_url)
            log.info("Telegram webhook registered for @%s", self._botname)
        except Exception as exc:
            log.warning("Telegram webhook registration failed (non-fatal): %s", exc)

    async def _set_webhook(self, engine_public_url: str) -> None:
        url = engine_public_url.rstrip("/") + "/telegram/webhook"
        payload: dict = {
            "url": url,
            "allowed_updates": ["message"],
        }
        if self._secret:
            payload["secret_token"] = self._secret
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(f"{self._base}/setWebhook", json=payload)
        if not r.is_success:
            raise RuntimeError(f"setWebhook failed {r.status_code}: {r.text}")
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"setWebhook error: {data.get('description')}")

    async def send_message(self, chat_id: int, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a text message to a Telegram chat.
        Returns True on success, False on any error (non-fatal).
        """
        if not self.enabled():
            return False
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(
                    f"{self._base}/sendMessage",
                    json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode},
                )
            if r.is_success and r.json().get("ok"):
                log.debug("Telegram message sent to chat_id=%s", chat_id)
                return True
            log.error("Telegram sendMessage error for chat_id=%s: %s", chat_id, r.text)
            return False
        except Exception as exc:
            log.error("Exception sending Telegram message to chat_id=%s: %s", chat_id, exc)
            return False

    async def get_me(self) -> dict:
        """Returns bot info — used by /health/telegram."""
        if not self.enabled():
            return {"ok": False, "error": "Token not configured"}
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self._base}/getMe")
            return r.json()
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


# ── Singleton (lazy) ──────────────────────────────────────────────────────────
# Built on first use, not at import time, so importing this module never touches
# settings — a router that imports it can be mounted even if Telegram is unset.
_client: TelegramClient | None = None


def get_telegram_client() -> TelegramClient:
    global _client
    if _client is None:
        _client = TelegramClient()
    return _client
