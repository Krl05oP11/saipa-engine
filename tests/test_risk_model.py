"""
pytest tests for analytics/risk_model.py

Run from saipa-engine/:
    pip install pytest
    pytest tests/test_risk_model.py -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analytics.risk_model import (
    predict,
    _heuristic_risk,
    _build_synthetic_dataset,
    FEATURE_NAMES,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def low_risk_features() -> dict:
    return {
        "last_access_days": 1,
        "submission_rate": 0.95,
        "login_count_7d": 9,
        "login_count_30d": 28,
        "saipa_days_since_last_chat": 1,
        "saipa_message_count": 20,
        "quiz_avg_score": 82,
        "quiz_attempt_count": 6,
        "forum_post_count": 5,
        "positive_feedback_ratio": 0.90,
        "completion_rate": 0.92,
    }


def high_risk_features() -> dict:
    return {
        "last_access_days": 45,
        "submission_rate": 0.0,
        "login_count_7d": 0,
        "login_count_30d": 1,
        "saipa_days_since_last_chat": 99,
        "saipa_message_count": 0,
        "quiz_avg_score": 0,
        "quiz_attempt_count": 0,
        "forum_post_count": 0,
        "positive_feedback_ratio": 0.0,
        "completion_rate": 0.0,
    }


# ── predict() return structure ────────────────────────────────────────────────

class TestPredictStructure:

    def test_returns_dict_with_required_keys(self):
        result = predict(low_risk_features())
        assert "score" in result
        assert "risk_level" in result
        assert "factors" in result

    def test_score_is_float_between_0_and_1(self):
        result = predict(low_risk_features())
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0

    def test_risk_level_is_valid_string(self):
        for feats in [low_risk_features(), high_risk_features()]:
            r = predict(feats)
            assert r["risk_level"] in ("low", "medium", "high"), \
                f"Unexpected risk_level: {r['risk_level']}"

    def test_factors_is_non_empty_dict(self):
        result = predict(low_risk_features())
        assert isinstance(result["factors"], dict)
        assert len(result["factors"]) > 0

    def test_factor_keys_are_valid_feature_names(self):
        result = predict(low_risk_features())
        for key in result["factors"]:
            assert key in FEATURE_NAMES, f"Unknown factor key: {key}"

    def test_factor_values_are_non_negative(self):
        result = predict(high_risk_features())
        for k, v in result["factors"].items():
            assert v >= 0.0, f"Negative factor weight for {k}: {v}"


# ── Prediction direction ──────────────────────────────────────────────────────

class TestPredictionDirection:

    def test_low_risk_student_scores_lower_than_high_risk(self):
        low  = predict(low_risk_features())["score"]
        high = predict(high_risk_features())["score"]
        assert low < high, f"Low risk ({low:.3f}) should score below high risk ({high:.3f})"

    def test_low_risk_classified_correctly(self):
        result = predict(low_risk_features())
        assert result["risk_level"] == "low", \
            f"Expected 'low', got '{result['risk_level']}' (score={result['score']:.3f})"

    def test_high_risk_classified_correctly(self):
        result = predict(high_risk_features())
        assert result["risk_level"] == "high", \
            f"Expected 'high', got '{result['risk_level']}' (score={result['score']:.3f})"

    def test_predict_is_deterministic(self):
        """Same input must always produce same output."""
        r1 = predict(low_risk_features())
        r2 = predict(low_risk_features())
        assert r1["score"] == r2["score"]
        assert r1["risk_level"] == r2["risk_level"]


# ── Missing / edge-case features ──────────────────────────────────────────────

class TestEdgeCases:

    def test_missing_features_use_zero_default(self):
        """predict() must not raise on empty dict."""
        result = predict({})
        assert "score" in result

    def test_zero_features_is_high_risk(self):
        """All-zero features = absent student = should be high or medium risk."""
        result = predict({k: 0.0 for k in FEATURE_NAMES})
        assert result["risk_level"] in ("high", "medium"), \
            f"All-zero features should be high/medium risk, got {result['risk_level']}"

    def test_perfect_features_is_low_risk(self):
        perfect = {
            "last_access_days": 0,
            "submission_rate": 1.0,
            "login_count_7d": 14,
            "login_count_30d": 60,
            "saipa_days_since_last_chat": 0,
            "saipa_message_count": 50,
            "quiz_avg_score": 100,
            "quiz_attempt_count": 10,
            "forum_post_count": 20,
            "positive_feedback_ratio": 1.0,
            "completion_rate": 1.0,
        }
        result = predict(perfect)
        assert result["risk_level"] == "low"


# ── Configurable thresholds ───────────────────────────────────────────────────

class TestConfigurableThresholds:

    def test_default_thresholds_match_plugin_defaults(self):
        """A borderline score sits 'medium' with defaults (0.40 / 0.75)."""
        feats = {k: 0.0 for k in FEATURE_NAMES}
        feats["submission_rate"] = 0.5
        feats["completion_rate"] = 0.5
        r = predict(feats)
        assert 0.40 <= r["score"] < 0.75
        assert r["risk_level"] == "medium"

    def test_lower_high_cutoff_promotes_to_high(self):
        feats = {k: 0.0 for k in FEATURE_NAMES}
        feats["submission_rate"] = 0.5
        feats["completion_rate"] = 0.5
        base = predict(feats)
        tightened = predict(feats, threshold_high=base["score"] - 0.01)
        assert tightened["risk_level"] == "high"
        assert tightened["score"] == base["score"]  # score is threshold-independent

    def test_raise_medium_cutoff_demotes_to_low(self):
        f = low_risk_features()
        f["submission_rate"] = 0.4
        r_default = predict(f)
        r_raised = predict(f, threshold_medium=r_default["score"] + 0.05)
        assert r_raised["risk_level"] == "low"

    def test_none_thresholds_use_defaults(self):
        f = high_risk_features()
        assert predict(f) == predict(f, threshold_medium=None, threshold_high=None)


# ── Factor accounting ─────────────────────────────────────────────────────────

class TestFactorAccounting:

    def test_factors_sum_approximates_score(self):
        for feats in [low_risk_features(), high_risk_features(),
                      {k: 0.0 for k in FEATURE_NAMES}, {}]:
            r = predict(feats)
            assert abs(sum(r["factors"].values()) - r["score"]) < 0.01

    def test_score_matches_heuristic(self):
        for feats in [low_risk_features(), high_risk_features()]:
            r = predict(feats)
            assert abs(r["score"] - round(_heuristic_risk(feats), 4)) < 1e-9


# ── Heuristic ─────────────────────────────────────────────────────────────────

class TestHeuristicRisk:

    def test_heuristic_bounds(self):
        for feats in [low_risk_features(), high_risk_features(), {k: 0.0 for k in FEATURE_NAMES}]:
            score = _heuristic_risk(feats)
            assert 0.0 <= score <= 1.0, f"Heuristic out of bounds: {score}"

    def test_high_risk_heuristic_above_low(self):
        lo = _heuristic_risk(low_risk_features())
        hi = _heuristic_risk(high_risk_features())
        assert lo < hi

    def test_heuristic_high_risk_above_threshold(self):
        """High-risk profile must surpass the 0.45 pseudo-label threshold."""
        assert _heuristic_risk(high_risk_features()) >= 0.45


# ── Synthetic dataset ─────────────────────────────────────────────────────────

class TestSyntheticDataset:

    def test_dataset_shape(self):
        X, y = _build_synthetic_dataset(n=300)
        assert X.shape == (300, len(FEATURE_NAMES))
        assert y.shape == (300,)

    def test_dataset_has_both_classes(self):
        X, y = _build_synthetic_dataset(n=300, seed=0)
        assert 0 in y, "Dataset must contain class 0 (non-dropout)"
        assert 1 in y, "Dataset must contain class 1 (dropout)"

    def test_feature_values_are_non_negative(self):
        X, _ = _build_synthetic_dataset(n=200)
        assert (X >= 0).all(), "Feature values must be non-negative"
