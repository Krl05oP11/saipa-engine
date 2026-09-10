"""
Helpers for calling Moodle web services from saipa-engine in the context of
Telegram account linking and session resolution.

All functions use the engine's MOODLE_TOKEN (the SAIPA service user token)
to call Moodle's REST web service endpoint.
"""
import logging

import httpx

from config import get_settings

log = logging.getLogger("saipa_engine.telegram.moodle")


async def _call_ws(wsfunction: str, params: dict) -> dict:
    """POST to Moodle REST API with the engine's service token."""
    settings = get_settings()
    url  = settings.moodle_url.rstrip("/") + "/webservice/rest/server.php"
    data = {
        "wstoken":            settings.moodle_token,
        "wsfunction":         wsfunction,
        "moodlewsrestformat": "json",
        **{str(k): str(v) for k, v in params.items()},
    }
    headers = {"Host": settings.moodle_public_host, "X-Forwarded-Proto": "https"}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, data=data, headers=headers)
    r.raise_for_status()
    result = r.json()
    # Moodle returns {"exception": ..., "message": ...} on WS-level errors.
    if isinstance(result, dict) and "exception" in result:
        raise RuntimeError(f"Moodle WS error ({wsfunction}): {result.get('message', result)}")
    return result


async def confirm_link(token: str, telegram_id: int, telegram_username: str) -> dict:
    """
    Validates an HMAC link token and binds the Telegram chat_id to the Moodle user.
    Returns {"user_id": int, "moodle_username": str, "fullname": str} on success.
    Raises RuntimeError on invalid/expired token.
    """
    return await _call_ws("local_saipa_telegram_confirm_link", {
        "token":            token,
        "telegram_id":      telegram_id,
        "telegram_username": telegram_username,
    })


async def get_session(telegram_id: int) -> dict:
    """
    Resolves a Telegram chat_id to a Moodle user + course context.
    Returns {"linked": bool, "user_id": int, "course_id": int, "fullname": str}.
    """
    return await _call_ws("local_saipa_telegram_get_session", {
        "telegram_id": telegram_id,
    })


async def unlink_by_telegram_id(telegram_id: int) -> None:
    """Removes the Telegram link for the given chat_id."""
    await _call_ws("local_saipa_telegram_unlink_by_id", {
        "telegram_id": telegram_id,
    })


async def mark_alert_responded(telegram_id: int, timereceived: int) -> dict:
    """
    Marks the most recent pending teacher_alert as responded.
    Called silently when a student sends any message to the bot.
    Returns {"found": bool, "delay_minutes": int, "moodle_accessed": bool}.
    """
    try:
        return await _call_ws("local_saipa_mark_alert_responded", {
            "telegram_id":  telegram_id,
            "timereceived": timereceived,
        })
    except Exception as exc:
        log.debug("mark_alert_responded chat_id=%s: %s", telegram_id, exc)
        return {"found": False, "delay_minutes": 0, "moodle_accessed": False}
