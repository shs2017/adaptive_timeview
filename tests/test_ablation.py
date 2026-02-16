import numpy as np
import torch

from ablation import AblationConfig, _evaluate_fix, run_n_obs_ablation
from timeview_adaptive import TimeviewAdaptive


class TestAblationConfig:
    def test_defaults(self):
        config = AblationConfig()
        assert config.n_basis_values == [5, 7, 9, 11]
        assert config.n_obs_values == [2, 5, 10, 15, 20, 30]
        assert config.noise_values == [0.05, 0.1, 0.2, 0.5]
        assert config.covariance_types == ["diagonal", "low_rank", "full_cholesky"]
        assert config.kl_weight_values == [0.0, 0.001, 0.01, 0.1]
        assert config.dropout_values == [0.0, 0.1, 0.2, 0.3]

    def test_custom_values(self):
        config = AblationConfig(n_basis_values=[3, 5], noise_values=[0.1])
        assert config.n_basis_values == [3, 5]
        assert config.noise_values == [0.1]
        assert config.covariance_types == ["diagonal", "low_rank", "full_cholesky"]

    def test_ablation_config_custom_extended(self):
        config = AblationConfig(
            n_basis_values=[3, 5],
            n_obs_values=[1, 2],
            kl_weight_values=[0.0, 1.0],
        )
        assert config.n_basis_values == [3, 5]
        assert config.n_obs_values == [1, 2]
        assert config.kl_weight_values == [0.0, 1.0]
        assert config.noise_values == [0.05, 0.1, 0.2, 0.5]


class TestRunNObsAblation:
    def test_n_obs_ablation_structure(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        model.eval()

        config = AblationConfig(n_obs_values=[2, 5, 10])
        results = run_n_obs_ablation(model, x, y, t, config)

        assert isinstance(results, dict)
        for n in [2, 5, 10]:
            assert n in results
            assert "future_mse" in results[n]
            assert "observed_mse" in results[n]

    def test_n_obs_ablation_skips_large_values(self, basic_dims):
        x = torch.randn(4, basic_dims["input_dim"])
        t = torch.linspace(0, 1, 10)
        y = torch.randn(4, 10)
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        model.eval()

        config = AblationConfig(n_obs_values=[2, 5, 10, 15, 20])
        results = run_n_obs_ablation(model, x, y, t, config)

        assert 2 in results
        assert 5 in results
        assert 10 not in results
        assert 15 not in results

    def test_n_obs_ablation_future_and_observed_metrics(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        model.eval()

        config = AblationConfig(n_obs_values=[5])
        results = run_n_obs_ablation(model, x, y, t, config)

        metrics = results[5]
        future_keys = [k for k in metrics if k.startswith("future_")]
        observed_keys = [k for k in metrics if k.startswith("observed_")]
        assert len(future_keys) > 0
        assert len(observed_keys) > 0


class TestEvaluateFix:
    def test_evaluate_fix_returns_metrics(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        model.eval()

        result = _evaluate_fix(model, x, y, t, n_obs=5)
        assert isinstance(result, dict)
        assert "future_mse" in result
        assert "prior_avg_var" in result
        assert "learned_sigma" in result
        assert result["prior_avg_var"] > 0
        assert result["learned_sigma"] > 0

    def test_evaluate_fix_future_metrics_only(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        model.eval()

        result = _evaluate_fix(model, x, y, t, n_obs=10)
        for key, val in result.items():
            assert np.isfinite(val), f"Non-finite value for {key}: {val}"
