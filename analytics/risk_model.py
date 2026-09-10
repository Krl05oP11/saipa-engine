"""
Dropout-risk scoring.

This is a **deterministic rule-based heuristic**, not a trained ML model. It maps
11 engagement signals (see FEATURE_NAMES) to a risk score in [0, 1] by turning
each signal into a bounded "risk component" and taking a weighted average.

`_build_synthetic_dataset` produces pseudo-labelled rows (label = heuristic score
>= 0.45) for anyone who later wants to fit a real classifier on top; the live
`predict()` path does not use it.

Feature semantics (all values non-negative; the caller — `get_student_features`
in local_saipa — supplies Moodle-neutral defaults for absent data):

    last_access_days            days since last course access      (higher = worse)
    submission_rate             assignment submission rate 0-1     (lower  = worse)
    login_count_7d              Moodle logins, last 7 days          (lower  = worse)
    login_count_30d             Moodle logins, last 30 days         (lower  = worse)
    saipa_days_since_last_chat  days since last SAIPA message       (higher = worse)
    saipa_message_count         total SAIPA messages by student     (lower  = worse)
    quiz_avg_score              average quiz score 0-100            (lower  = worse)
    quiz_attempt_count          number of quiz attempts             (lower  = worse)
    forum_post_count            forum posts in this course          (lower  = worse)
    positive_feedback_ratio     ratio of positive SAIPA feedback    (lower  = worse)
    completion_rate             activity completion rate 0-1        (lower  = worse)
"""
from __future__ import annotations

FEATURE_NAMES: list[str] = [
    "last_access_days",
    "submission_rate",
    "login_count_7d",
    "login_count_30d",
    "saipa_days_since_last_chat",
    "saipa_message_count",
    "quiz_avg_score",
    "quiz_attempt_count",
    "forum_post_count",
    "positive_feedback_ratio",
    "completion_rate",
]

# Pseudo-label threshold: a heuristic score at or above this counts as "dropout".
LABEL_THRESHOLD = 0.45

# Default risk_level cut-offs applied to `score`. Callers may override them per
# request (RiskRequest / BatchRiskRequest carry `threshold_medium` /
# `threshold_high`): local_saipa forwards the site's admin-configured values so
# the label always matches the plugin's own settings. These defaults match the
# plugin's out-of-the-box configuration (medium 0.40, high 0.75).
_MEDIUM_CUTOFF = 0.40
_HIGH_CUTOFF = 0.75

# Per-feature weight in the average, and how to turn the raw value into a risk
# component in [0, 1]. `scale` is the value at which the component saturates.
#   direction "low"  -> component = 1 - clamp(value / scale)   (more is better)
#   direction "high" -> component =     clamp(value / scale)   (more is worse)
_FEATURES: dict[str, tuple[float, str, float]] = {
    # name:                        (weight, direction, scale)
    "last_access_days":            (2.5, "high", 30.0),
    "submission_rate":             (2.0, "low", 1.0),
    "completion_rate":             (2.0, "low", 1.0),
    "login_count_7d":              (1.5, "low", 10.0),
    "quiz_avg_score":              (1.5, "low", 100.0),
    "login_count_30d":             (1.0, "low", 30.0),
    "quiz_attempt_count":          (1.0, "low", 5.0),
    "forum_post_count":            (0.5, "low", 5.0),
    "saipa_days_since_last_chat":  (0.5, "high", 30.0),
    "saipa_message_count":         (0.5, "low", 20.0),
    "positive_feedback_ratio":     (0.5, "low", 1.0),
}

_TOTAL_WEIGHT = sum(w for w, _, _ in _FEATURES.values())


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _risk_component(value: float, direction: str, scale: float) -> float:
    """Raw feature value -> risk contribution in [0, 1]."""
    ratio = _clamp01(value / scale) if scale else 0.0
    return ratio if direction == "high" else 1.0 - ratio


def _coerce(features: dict, name: str) -> float:
    raw = features.get(name)
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _raw_components(features: dict) -> dict[str, float]:
    """{feature: weight * risk_component / total_weight} — full precision.

    One non-negative entry per feature; the values sum to the (pre-clamp) score.
    """
    out: dict[str, float] = {}
    for name, (weight, direction, scale) in _FEATURES.items():
        comp = _risk_component(_coerce(features, name), direction, scale)
        out[name] = weight * comp / _TOTAL_WEIGHT
    return out


def _heuristic_risk(features: dict) -> float:
    """Weighted-average risk score in [0, 1]. Deterministic."""
    return _clamp01(sum(_raw_components(features).values()))


def _risk_level(score: float, medium: float = _MEDIUM_CUTOFF, high: float = _HIGH_CUTOFF) -> str:
    if score >= high:
        return "high"
    if score >= medium:
        return "medium"
    return "low"


def predict(
    features: dict,
    *,
    threshold_medium: float | None = None,
    threshold_high: float | None = None,
) -> dict:
    """
    Risk assessment for one student.

    Args:
        features: dict keyed by FEATURE_NAMES (extra keys ignored, missing
                  keys treated as 0.0). Never raises on a partial/empty dict.
        threshold_medium / threshold_high: optional overrides for the
                  low/medium/high cut-offs. `None` uses the module defaults
                  (0.40 / 0.75). local_saipa passes its admin-configured values.

    Returns:
        {
          "score":      float in [0, 1], rounded to 4 dp,
          "risk_level": "low" | "medium" | "high",
          "factors":    {feature_name: contribution}  (non-negative; the rounded
                        values sum to approximately `score`),
        }
    """
    medium = _MEDIUM_CUTOFF if threshold_medium is None else float(threshold_medium)
    high = _HIGH_CUTOFF if threshold_high is None else float(threshold_high)

    raw = _raw_components(features)
    score = round(_clamp01(sum(raw.values())), 4)
    return {
        "score": float(score),
        "risk_level": _risk_level(score, medium, high),
        "factors": {name: round(value, 4) for name, value in raw.items()},
    }


def _build_synthetic_dataset(n: int = 500, seed: int = 42):
    """
    Pseudo-labelled feature matrix for optionally fitting a real classifier later.

    Returns (X, y) as numpy arrays: X is (n, len(FEATURE_NAMES)) of non-negative
    feature values sampled across plausible ranges; y is (n,) with
    y_i = 1 iff _heuristic_risk(row) >= LABEL_THRESHOLD.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    # Upper bound per feature, in FEATURE_NAMES order. Lower bound is 0.
    highs = np.array(
        [60.0, 1.0, 14.0, 60.0, 100.0, 50.0, 100.0, 12.0, 20.0, 1.0, 1.0]
    )
    X = rng.uniform(0.0, highs, size=(n, len(FEATURE_NAMES)))
    y = np.array(
        [
            1 if _heuristic_risk(dict(zip(FEATURE_NAMES, row))) >= LABEL_THRESHOLD else 0
            for row in X
        ],
        dtype=int,
    )
    return X, y
