"""
Health check router for saipa-engine.
Called by local_saipa PHP (local_saipa_engine_request('/health')) to verify connectivity.
"""
import time

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import get_settings

router = APIRouter()
_bearer = HTTPBearer(auto_error=False)
_startup_time = time.time()


def _verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Dependency: validates the Bearer token sent by Moodle.

    Auth is skipped only when SAIPA_DEV_MODE is on and no token is configured
    (startup in main.py refuses that combination otherwise). A configured token
    is always enforced.
    """
    settings = get_settings()
    if not settings.saipa_api_token:
        if settings.saipa_dev_mode:
            return
        # Should be unreachable (main.py exits at startup), but fail closed.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine API token is not configured",
        )
    if credentials is None or credentials.credentials != settings.saipa_api_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token",
        )


@router.get("/health", tags=["system"])
async def health(token: None = Depends(_verify_token)) -> dict:
    """
    Returns engine status. Called by Moodle's health_check web service.
    Response fields: status, message, version, uptime_seconds.
    """
    return {
        "status": "ok",
        "message": "saipa-engine is running",
        "version": "0.1.0",
        "uptime_seconds": round(time.time() - _startup_time, 1),
    }
