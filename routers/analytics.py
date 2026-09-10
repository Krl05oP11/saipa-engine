"""
Analytics router — Fase 3: dropout risk prediction.

Endpoints:
  POST /analytics/risk        — single student
  POST /analytics/risk/batch  — all enrolled students in a course
"""
import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from analytics.features import extract_features
from analytics.risk_model import predict
from routers.health import _verify_token

log = logging.getLogger("saipa_engine.routers.analytics")

router = APIRouter()


# ── Pydantic models ────────────────────────────────────────────────────────────

class RiskRequest(BaseModel):
    user_id: int
    course_id: int
    features: dict | None = None  # pre-computed features; if None, fetch from Moodle
    # Optional low/medium/high cut-offs. None -> engine defaults (0.40 / 0.75).
    # local_saipa forwards the site's admin-configured thresholds.
    threshold_medium: float | None = None
    threshold_high: float | None = None


class RiskResult(BaseModel):
    user_id: int
    score: float
    risk_level: str          # "low" | "medium" | "high"
    factors: dict[str, float]


class BatchRiskRequest(BaseModel):
    course_id: int
    user_ids: list[int]
    # Optional: pre-computed features per user_id (avoids engine calling Moodle).
    # Keys are stringified user IDs.
    features_map: dict[str, dict] = {}
    # Optional low/medium/high cut-offs, applied to every student in the batch.
    threshold_medium: float | None = None
    threshold_high: float | None = None


class BatchRiskResponse(BaseModel):
    results: list[RiskResult]
    errors: list[dict]       # [{user_id, error}] for failed lookups


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _assess(
    user_id: int,
    course_id: int,
    precomputed: dict | None = None,
    *,
    threshold_medium: float | None = None,
    threshold_high: float | None = None,
) -> RiskResult:
    features = precomputed if precomputed is not None else await extract_features(user_id, course_id)
    assessment = predict(
        features,
        threshold_medium=threshold_medium,
        threshold_high=threshold_high,
    )
    return RiskResult(
        user_id=user_id,
        score=assessment["score"],
        risk_level=assessment["risk_level"],
        factors=assessment["factors"],
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/analytics/risk", response_model=RiskResult, tags=["analytics"])
async def risk_single(
    req: RiskRequest,
    token: None = Depends(_verify_token),
) -> RiskResult:
    """
    Predicts dropout risk for a single student.
    If `features` is supplied in the request body, Moodle is not queried.
    """
    try:
        return await _assess(
            req.user_id,
            req.course_id,
            req.features,
            threshold_medium=req.threshold_medium,
            threshold_high=req.threshold_high,
        )
    except Exception as exc:
        log.error("risk_single user=%d course=%d error=%s", req.user_id, req.course_id, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/analytics/risk/batch", response_model=BatchRiskResponse, tags=["analytics"])
async def risk_batch(
    req: BatchRiskRequest,
    token: None = Depends(_verify_token),
) -> BatchRiskResponse:
    """
    Predicts dropout risk for a list of students in a course.
    Runs feature extraction concurrently; errors are reported per-student.
    """
    tasks = [
        _assess(
            uid,
            req.course_id,
            req.features_map.get(str(uid)),
            threshold_medium=req.threshold_medium,
            threshold_high=req.threshold_high,
        )
        for uid in req.user_ids
    ]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    results, errors = [], []
    for uid, outcome in zip(req.user_ids, raw):
        if isinstance(outcome, Exception):
            errors.append({"user_id": uid, "error": str(outcome)})
            log.warning("risk_batch user=%d course=%d error=%s", uid, req.course_id, outcome)
        else:
            results.append(outcome)

    log.info(
        "risk_batch course=%d users=%d ok=%d errors=%d",
        req.course_id, len(req.user_ids), len(results), len(errors),
    )
    return BatchRiskResponse(results=results, errors=errors)
