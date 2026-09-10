"""
WhatsApp client for SAIPA using Evolution API (Baileys / WhatsApp Web).
Adapted from ChatVida's WhatsAppClient — stripped to essentials (no app DB).
"""
import logging

import httpx

from config import get_settings

log = logging.getLogger("saipa_engine.whatsapp")


class WhatsAppClient:
    def __init__(self):
        settings = get_settings()
        self._api_url        = settings.evolution_api_url
        self._api_key        = settings.evolution_api_key
        self._instance_name  = settings.evolution_instance_name
        self._webhook_url    = settings.whatsapp_webhook_url
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._api_url,
            headers={"apikey": self._api_key, "Content-Type": "application/json"},
            timeout=30.0,
        )
        try:
            await self._ensure_instance()
            log.info("WhatsApp client ready (instance=%s)", self._instance_name)
        except Exception as e:
            log.warning("WhatsApp client init warning (non-fatal): %s", e)

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()

    async def _ensure_instance(self) -> None:
        resp = await self._client.get("/instance/fetchInstances")
        if resp.status_code != 200:
            raise RuntimeError(f"Evolution API error: {resp.status_code}")

        instances = resp.json()
        for inst in instances:
            if inst.get("instance", {}).get("instanceName") == self._instance_name:
                state = inst.get("instance", {}).get("state", "unknown")
                log.info("Instance '%s' found, state=%s", self._instance_name, state)
                if state != "open":
                    log.info("Scan QR at http://localhost:8055/manager to connect WhatsApp")
                return

        # Instance doesn't exist — create it.
        payload = {
            "instanceName": self._instance_name,
            "qrcode": True,
            "integration": "WHATSAPP-BAILEYS",
            "webhookUrl": self._webhook_url,
            "webhookByEvents": False,
            "webhookBase64": False,
            "rejectCall": False,
            "groupsIgnore": True,
            "alwaysOnline": True,
            "readMessages": True,
        }
        resp = await self._client.post("/instance/create", json=payload)
        if resp.status_code in (200, 201):
            log.info("Instance '%s' created. Open http://localhost:8055/manager and scan the QR code.", self._instance_name)
        else:
            raise RuntimeError(f"Failed to create instance: {resp.status_code} {resp.text}")

    async def send_message(self, phone: str, text: str) -> bool:
        """Send a text message. phone is in format '5491112345678' (no @s.whatsapp.net)."""
        if not self._client:
            log.error("WhatsApp client not started")
            return False
        if not phone.endswith("@s.whatsapp.net"):
            phone = f"{phone}@s.whatsapp.net"
        try:
            resp = await self._client.post(
                f"/message/sendText/{self._instance_name}",
                json={"number": phone, "text": text},
            )
            if resp.status_code == 200:
                log.info("WhatsApp message sent to %s", phone)
                return True
            log.error("Evolution API send error: %s %s", resp.status_code, resp.text)
            return False
        except Exception as e:
            log.error("Exception sending WhatsApp message: %s", e)
            return False

    async def get_qr(self) -> str | None:
        """Returns base64 QR code if the instance is not yet connected."""
        if not self._client:
            return None
        try:
            resp = await self._client.get(f"/instance/qrcode/{self._instance_name}")
            if resp.status_code == 200:
                return resp.json().get("qrcode", {}).get("base64")
        except Exception:
            pass
        return None

    async def get_status(self) -> dict:
        if not self._client:
            return {"state": "not_initialized"}
        try:
            resp = await self._client.get(f"/instance/connectionState/{self._instance_name}")
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            return {"state": "error", "error": str(e)}
        return {"state": "unknown"}


# Singleton (lazy) — built on first use, not at import time.
_client: WhatsAppClient | None = None


def get_whatsapp_client() -> WhatsAppClient:
    global _client
    if _client is None:
        _client = WhatsAppClient()
    return _client
