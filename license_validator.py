"""
License validation for saipa-engine.
Checks license key against api.schaller-ponce.com.ar at startup.
Falls back to offline mode for up to SAIPA_OFFLINE_GRACE_DAYS days.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import httpx
from pydantic_settings import BaseSettings

log = logging.getLogger(__name__)

_CACHE_FILE = Path(".license_cache")
_CACHE_MAX_AGE = 3600  # Re-check every hour if online.


class LicenseSettings(BaseSettings):
    saipa_license_key: str = ""
    saipa_license_validation_url: str = "https://api.schaller-ponce.com.ar/license/validate"
    saipa_offline_grace_days: int = 30

    class Config:
        env_file = ".env"
        extra = "ignore"


def _load_cache() -> dict:
    if _CACHE_FILE.exists():
        try:
            return json.loads(_CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_cache(data: dict) -> None:
    try:
        _CACHE_FILE.write_text(json.dumps(data))
    except Exception:
        pass


def validate_license() -> bool:
    """
    Returns True if the license is valid (online check or within grace period).
    Raises RuntimeError if the license is invalid or grace period has expired.
    """
    cfg = LicenseSettings()

    if not cfg.saipa_license_key:
        log.warning("SAIPA_LICENSE_KEY not set — running in development mode (no validation).")
        return True

    cache = _load_cache()
    now = time.time()

    # Use cache if fresh enough.
    if cache.get("checked_at", 0) + _CACHE_MAX_AGE > now:
        if cache.get("valid"):
            return True
        raise RuntimeError(f"License invalid (cached): {cache.get('reason', 'unknown')}")

    # Try online validation.
    try:
        resp = httpx.post(
            cfg.saipa_license_validation_url,
            json={"key": cfg.saipa_license_key},
            timeout=10.0,
        )
        data = resp.json()
        if resp.status_code == 200 and data.get("valid"):
            _save_cache({"valid": True, "checked_at": now, "expires_at": data.get("expires_at")})
            log.info("License validated successfully.")
            return True
        else:
            reason = data.get("reason", "rejected by server")
            _save_cache({"valid": False, "checked_at": now, "reason": reason})
            raise RuntimeError(f"License invalid: {reason}")

    except httpx.RequestError as exc:
        # Network failure — check grace period from last valid check.
        last_valid = cache.get("checked_at", 0) if cache.get("valid") else 0
        grace_seconds = cfg.saipa_offline_grace_days * 86400
        if last_valid and (now - last_valid) < grace_seconds:
            days_left = int((grace_seconds - (now - last_valid)) / 86400)
            log.warning(
                "Cannot reach license server (%s). Offline grace period: %d days remaining.",
                exc, days_left,
            )
            return True
        raise RuntimeError(
            f"License server unreachable and offline grace period expired. "
            f"Please check your network connection. Error: {exc}"
        )
