"""
Telegram long-polling loop for SAIPA (development / no-public-URL mode).

Used when TELEGRAM_WEBHOOK_URL is not set.
Runs as an asyncio background task started in main.py on_startup.

The loop calls getUpdates with long-polling (timeout=25s), dispatches each
update to the same handlers used by the webhook endpoint, and manages the
offset so no update is processed twice.
"""
import asyncio
import logging

import httpx

from config import get_settings

log = logging.getLogger("saipa_engine.telegram.polling")


async def run_polling() -> None:
    """
    Long-polling loop. Runs indefinitely until the event loop is stopped.
    Dispatches messages to the same internal handlers in telegram.webhook.
    """
    # Import here to avoid circular dependency at module load time
    from telegram.webhook import _dispatch_update  # noqa: PLC0415

    settings = get_settings()
    token    = settings.telegram_bot_token
    base     = f"https://api.telegram.org/bot{token}"
    offset   = 0

    log.info("Telegram polling started for @%s", settings.telegram_bot_username)

    async with httpx.AsyncClient(timeout=35) as client:
        while True:
            try:
                r = await client.get(
                    f"{base}/getUpdates",
                    params={"offset": offset, "timeout": 25, "allowed_updates": ["message"]},
                )
                if not r.is_success:
                    log.warning("getUpdates HTTP %s — retrying in 5s", r.status_code)
                    await asyncio.sleep(5)
                    continue

                data = r.json()
                if not data.get("ok"):
                    log.warning("getUpdates not ok: %s — retrying in 5s", data)
                    await asyncio.sleep(5)
                    continue

                for update in data.get("result", []):
                    offset = update["update_id"] + 1
                    asyncio.create_task(_dispatch_update(update))

            except asyncio.CancelledError:
                log.info("Telegram polling cancelled — shutting down")
                return
            except Exception as exc:
                log.error("Telegram polling error: %s — retrying in 5s", exc)
                await asyncio.sleep(5)
