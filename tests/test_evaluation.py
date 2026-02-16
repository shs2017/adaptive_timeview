import numpy as np
import torch

from evaluation import eval_model_full, streaming_evaluation
from timeview_adaptive import TimeviewAdaptive, TimeviewAdaptiveGated, train_model


class TestStreamingEvaluation:
    def test_returns_expected_keys(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        results = streaming_evaluation(model, x[:2], t, y[:2])
        expected_keys = ["n_obs", "mse_future", "crps_future", "avg_uncertainty", "coverage_95"]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

    def test_starts_with_zero_obs(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        results = streaming_evaluation(model, x[:2], t, y[:2])
        assert results["n_obs"][0] == 0, "Should start with 0 observations"

    def test_mse_list_populated(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        train_model(model, x, t, y, n_epochs=30, lr=1e-3, n_obs=10, verbose=False)
        model.eval()

        results = streaming_evaluation(model, x[:4], t, y[:4])
        mse_values = results["mse_future"]
        assert len(mse_values) > 10, "Should have MSE values for many observation counts"
        assert all(np.isfinite(m) for m in mse_values), "All MSE values should be finite"
        assert all(m >= 0 for m in mse_values), "All MSE values should be non-negative"


class TestStreamingEdgeCases:
    def test_streaming_with_single_sample(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        results = streaming_evaluation(model, x[:1], t, y[:1])
        assert len(results["n_obs"]) > 0
        assert results["n_obs"][0] == 0

    def test_streaming_coverage_bounded(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        results = streaming_evaluation(model, x[:2], t, y[:2])
        for cov in results["coverage_95"]:
            assert 0 <= cov <= 1, f"Coverage out of range: {cov}"


class TestEvalModelFull:
    def test_eval_model_full_returns_metrics(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        model.eval()

        result = eval_model_full(model, x, y, t, n_obs=5)
        assert isinstance(result, dict)
        assert "future_mse" in result
        assert "future_mae" in result
        assert "future_crps" in result

    def test_eval_model_full_gated(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptiveGated(**basic_dims, learn_noise=True)
        model.eval()

        result = eval_model_full(model, x, y, t, n_obs=5)
        assert isinstance(result, dict)
        assert "future_mse" in result
