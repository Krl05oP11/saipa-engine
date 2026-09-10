"""
Feature extraction for dropout-risk scoring.

In the live flow the Moodle plugin computes the 11 features itself
(`local_saipa\\external\\get_student_features`) and sends them to the engine in
`features_map`, so this function is only reached if `/analytics/risk` is called
without a `features` body. Extracting features engine-side would require a Moodle
web-services client with a course-scoped token; that path is intentionally not
built — callers must supply the feature vector.
"""
from __future__ import annotations

from analytics.risk_model import FEATURE_NAMES

__all__ = ["FEATURE_NAMES", "extract_features"]


async def extract_features(user_id: int, course_id: int) -> dict:
    """Not implemented on purpose — see module docstring. Supply `features` instead."""
    raise RuntimeError(
        "Engine-side feature extraction is not available. "
        "Send the pre-computed feature vector in the request "
        "(RiskRequest.features / BatchRiskRequest.features_map)."
    )
