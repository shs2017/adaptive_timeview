import numpy as np
import pytest
import torch

from scores import (
    compute_all_metrics,
    compute_calibration,
    crps_gaussian,
    interval_score,
)


class TestScoringFunctions:
    def test_crps_gaussian_shape(self):
        y = torch.randn(10, 20)
        mu = torch.randn(10, 20)
        sigma = torch.ones(10, 20) * 0.5
        crps = crps_gaussian(y, mu, sigma)
        assert crps.shape == (10, 20)

    def test_crps_gaussian_non_negative(self):
        y = torch.randn(10, 20)
        mu = torch.randn(10, 20)
        sigma = torch.ones(10, 20) * 0.5
        crps = crps_gaussian(y, mu, sigma)
        assert (crps >= -1e-6).all(), f"CRPS should be non-negative, min={crps.min()}"

    def test_crps_zero_when_perfect(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        mu = y.clone()
        sigma = torch.ones_like(y) * 0.001
        crps = crps_gaussian(y, mu, sigma)
        assert crps.mean() < 0.01, f"CRPS too high for perfect predictions: {crps.mean()}"

    def test_crps_increases_with_error(self):
        y = torch.zeros(100)
        sigma = torch.ones(100)
        crps_close = crps_gaussian(y, torch.zeros(100), sigma).mean()
        crps_far = crps_gaussian(y, torch.ones(100) * 3.0, sigma).mean()
        assert crps_far > crps_close, "CRPS should increase with error"

    def test_interval_score_shape(self):
        y = torch.randn(10, 20)
        lower = y - 1.0
        upper = y + 1.0
        score = interval_score(y, lower, upper)
        assert score.shape == (10, 20)

    def test_interval_score_non_negative(self):
        y = torch.randn(10, 20)
        lower = y - 1.0
        upper = y + 1.0
        score = interval_score(y, lower, upper)
        assert (score >= -1e-6).all(), "Interval score should be non-negative"

    def test_interval_score_penalizes_miscoverage(self):
        y = torch.tensor([0.0])
        s_covered = interval_score(y, torch.tensor([-1.0]), torch.tensor([1.0])).item()
        s_missed = interval_score(y, torch.tensor([1.0]), torch.tensor([2.0])).item()
        assert s_missed > s_covered, "Should penalize miscoverage"

    def test_compute_calibration_shape(self):
        y = torch.randn(100)
        mu = torch.randn(100)
        sigma = torch.ones(100) * 0.5
        cal = compute_calibration(y, mu, sigma, n_bins=10)
        assert len(cal["expected"]) == 10
        assert len(cal["observed"]) == 10
        assert "calibration_error" in cal

    def test_compute_calibration_perfect(self):
        torch.manual_seed(42)
        n = 10000
        y = torch.randn(n)
        mu = torch.zeros(n)
        sigma = torch.ones(n)
        cal = compute_calibration(y, mu, sigma, n_bins=10)
        assert cal["calibration_error"] < 0.1, \
            f"Well-calibrated predictions have high error: {cal['calibration_error']}"


class TestComputeAllMetrics:
    def test_returns_expected_keys(self, sample_data):
        x, t, y = sample_data
        mu = torch.randn_like(y)
        var = torch.ones_like(y) * 0.5
        metrics = compute_all_metrics(y, mu, var)
        expected_keys = [
            "mse", "mae", "crps", "nll", "interval_score_90", "interval_score_95",
            "coverage_90", "coverage_95", "calibration_error", "avg_std",
            "sharpness_90", "sharpness_95", "winkler_95_norm", "predictive_efficiency",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_prefix_applied(self, sample_data):
        x, t, y = sample_data
        mu = torch.randn_like(y)
        var = torch.ones_like(y) * 0.5
        metrics = compute_all_metrics(y, mu, var, prefix="test_")
        assert "test_mse" in metrics
        assert "test_coverage_95" in metrics

    def test_mse_correct(self):
        y = torch.tensor([[1.0, 2.0, 3.0]])
        mu = torch.tensor([[1.1, 2.2, 2.8]])
        var = torch.ones_like(y)
        metrics = compute_all_metrics(y, mu, var)
        expected_mse = ((y - mu) ** 2).mean().item()
        assert abs(metrics["mse"] - expected_mse) < 1e-6

    def test_predictive_efficiency_positive(self, sample_data):
        x, t, y = sample_data
        mu = y + torch.randn_like(y) * 0.1
        var = torch.ones_like(y) * 0.1
        metrics = compute_all_metrics(y, mu, var)
        assert metrics["predictive_efficiency"] > 0, "PE should be positive"

    def test_coverage_in_range(self, sample_data):
        x, t, y = sample_data
        mu = torch.randn_like(y)
        var = torch.ones_like(y)
        metrics = compute_all_metrics(y, mu, var)
        assert 0 <= metrics["coverage_90"] <= 1
        assert 0 <= metrics["coverage_95"] <= 1
        assert metrics["coverage_95"] >= metrics["coverage_90"], \
            "95% coverage should be >= 90% coverage"
