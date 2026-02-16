import numpy as np
import torch

from timeview_adaptive import (
    TimeviewAdaptive,
    TimeviewAdaptiveGated,
    TimeviewAdaptiveGatedHeteroscedastic,
    TimeviewAdaptiveHeteroscedastic,
    train_model,
)
from timeview_adaptive_experiments import (
    compute_adaptation_efficiency_curve,
    compute_uncertainty_decomposition,
    run_active_scheduling_experiment,
    run_gp_baseline_comparison,
    run_heteroscedastic_comparison,
    run_temperature_scaling_experiment,
    run_uncertainty_decomposition_experiment,
)


class TestBugFix_HeteroscedasticUncertaintyDecomp:
    def test_heteroscedastic_decomposition_uses_noise_model(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptiveHeteroscedastic(**basic_dims)
        model.eval()

        with torch.no_grad():
            decomp = compute_uncertainty_decomposition(model, x, t[:5], y[:, :5], t)

        noise = decomp["noise"]
        assert noise.shape == (16, 30), f"Wrong noise shape: {noise.shape}"
        assert (noise > 0).all(), "Noise variance should be positive"

    def test_homoscedastic_decomposition_has_constant_noise(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        with torch.no_grad():
            decomp = compute_uncertainty_decomposition(model, x, t[:5], y[:, :5], t)

        noise = decomp["noise"]
        noise_std_across_time = noise.std(dim=1)
        assert (noise_std_across_time < 1e-6).all(), \
            "Homoscedastic noise should be constant across time"

    def test_decomposition_sums_correctly(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        with torch.no_grad():
            decomp = compute_uncertainty_decomposition(model, x, t[:5], y[:, :5], t)

        total = decomp["total_var"]
        expected = decomp["posterior_epistemic"] + decomp["noise"]
        assert torch.allclose(total, expected, atol=1e-5), \
            f"Decomposition doesn't sum: max diff={torch.abs(total - expected).max()}"

    def test_info_gained_equals_prior_minus_posterior(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        with torch.no_grad():
            decomp = compute_uncertainty_decomposition(model, x, t[:5], y[:, :5], t)

        info_gained = decomp["info_gained"]
        expected = decomp["prior_epistemic"] - decomp["posterior_epistemic"]
        assert torch.allclose(info_gained, expected, atol=1e-5), \
            "info_gained != prior - posterior epistemic"

    def test_info_gained_non_negative(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        with torch.no_grad():
            decomp = compute_uncertainty_decomposition(model, x, t[:5], y[:, :5], t)

        assert (decomp["info_gained"] >= -1e-5).all(), \
            f"Negative info gained: min={decomp['info_gained'].min()}"


class TestAEC:
    def test_returns_expected_keys(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        aec = compute_adaptation_efficiency_curve(model, x, t, y, max_obs=5)
        expected_keys = ["n_obs", "mse", "aec", "cumulative_reduction", "mse_prior", "aaec"]
        for key in expected_keys:
            assert key in aec, f"Missing key: {key}"

    def test_n_obs_matches_max(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        aec = compute_adaptation_efficiency_curve(model, x, t, y, max_obs=10)
        assert len(aec["n_obs"]) == 10
        assert aec["n_obs"] == list(range(1, 11))

    def test_aec_formula_correct(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        aec = compute_adaptation_efficiency_curve(model, x, t, y, max_obs=5)
        for i, k in enumerate(aec["n_obs"]):
            expected = (aec["mse_prior"] - aec["mse"][i]) / k
            assert abs(aec["aec"][i] - expected) < 1e-6, \
                f"AEC formula wrong at k={k}: {aec['aec'][i]} != {expected}"

    def test_aaec_is_finite(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        aec = compute_adaptation_efficiency_curve(model, x, t, y, max_obs=5)
        assert np.isfinite(aec["aaec"]), f"AAEC not finite: {aec['aaec']}"


class TestAECTrained:
    def test_aec_positive_after_training(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        train_model(model, x, t, y, n_epochs=50, lr=1e-3, n_obs=10, verbose=False)
        model.eval()

        aec = compute_adaptation_efficiency_curve(model, x, t, y, max_obs=10)
        max_aec = max(aec["aec"])
        assert max_aec > -1.0, f"AEC should have some positive values, max={max_aec}"

    def test_aec_cumulative_reduction_monotonic(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        aec = compute_adaptation_efficiency_curve(model, x, t, y, max_obs=10)
        reductions = aec["cumulative_reduction"]
        assert len(reductions) == 10
        assert all(np.isfinite(r) for r in reductions)


class TestUncertaintyDecompHeteroscedastic:
    def test_gated_heteroscedastic_decomposition(self, sample_data):
        x, t, y = sample_data
        model = TimeviewAdaptiveGatedHeteroscedastic(input_dim=4, n_basis=7)
        model.eval()

        with torch.no_grad():
            decomp = compute_uncertainty_decomposition(model, x, t[:5], y[:, :5], t)

        noise = decomp["noise"]
        assert noise.shape == (16, 30)
        assert (noise > 0).all()

        total = decomp["total_var"]
        expected = decomp["posterior_epistemic"] + decomp["noise"]
        assert torch.allclose(total, expected, atol=1e-5)

    def test_decomposition_with_many_observations(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        with torch.no_grad():
            decomp_few = compute_uncertainty_decomposition(model, x, t[:2], y[:, :2], t)
            decomp_many = compute_uncertainty_decomposition(model, x, t[:15], y[:, :15], t)

        avg_ig_few = decomp_few["info_gained"].mean().item()
        avg_ig_many = decomp_many["info_gained"].mean().item()
        assert avg_ig_many >= avg_ig_few - 1e-5, \
            f"More observations should yield more info gained: {avg_ig_few} vs {avg_ig_many}"


class TestRunTemperatureScalingExperiment:
    def test_temperature_scaling_experiment(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        train_model(model, x, t, y, n_epochs=20, n_obs=5, verbose=False)
        model.eval()

        x_cal, x_test = x[:8], x[8:]
        y_cal, y_test = y[:8], y[8:]

        result = run_temperature_scaling_experiment(
            model, x_cal, y_cal, x_test, y_test, t, n_obs=5
        )

        assert "temperature" in result
        assert result["temperature"] > 0
        assert "before_mse" in result
        assert "before_coverage_95" in result
        assert "after_mse" in result
        assert "after_coverage_95" in result
        assert abs(result["before_mse"] - result["after_mse"]) < 1e-4


class TestRunUncertaintyDecompositionExperiment:
    def test_decomposition_experiment_structure(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        model.eval()

        results = run_uncertainty_decomposition_experiment(
            model, x, y, t, n_obs_values=[2, 5, 10]
        )

        assert isinstance(results, dict)
        for n in [2, 5, 10]:
            assert n in results
            entry = results[n]
            assert "avg_prior_epistemic" in entry
            assert "avg_info_gained" in entry
            assert "avg_posterior_epistemic" in entry
            assert "avg_noise" in entry
            assert "avg_total" in entry
            assert "info_fraction" in entry
            assert "decomp" in entry

    def test_decomposition_defaults(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        model.eval()

        results = run_uncertainty_decomposition_experiment(model, x, y, t)
        assert 2 in results
        assert 5 in results
        assert 10 in results
        assert 20 in results

    def test_info_fraction_increases(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        train_model(model, x, t, y, n_epochs=30, n_obs=5, verbose=False)
        model.eval()

        results = run_uncertainty_decomposition_experiment(
            model, x, y, t, n_obs_values=[2, 10, 20]
        )
        frac_2 = results[2]["info_fraction"]
        frac_20 = results[20]["info_fraction"]
        assert frac_2 >= -0.1, f"Info fraction at n=2 is very negative: {frac_2}"
        assert frac_20 >= -0.1, f"Info fraction at n=20 is very negative: {frac_20}"

    def test_decomposition_skips_large_n_obs(self, basic_dims):
        x = torch.randn(4, basic_dims["input_dim"])
        t = torch.linspace(0, 1, 8)
        y = torch.randn(4, 8)
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        model.eval()

        results = run_uncertainty_decomposition_experiment(
            model, x, y, t, n_obs_values=[2, 5, 8, 15]
        )
        assert 2 in results
        assert 5 in results
        assert 8 not in results
        assert 15 not in results


class TestRunHeteroscedasticComparison:
    def test_heteroscedastic_comparison_structure(self):
        torch.manual_seed(42)
        x_train = torch.randn(12, 4)
        x_test = torch.randn(6, 4)
        t = torch.linspace(0, 1, 15)
        y_train = torch.randn(12, 15)
        y_test = torch.randn(6, 15)

        results = run_heteroscedastic_comparison(
            x_train, y_train, x_test, y_test, t, n_obs=5, n_epochs=20
        )

        assert "homoscedastic" in results
        assert "heteroscedastic" in results
        assert "gated" in results
        for variant in ["homoscedastic", "heteroscedastic", "gated"]:
            assert "future_mse" in results[variant]
            assert "future_coverage_95" in results[variant]


class TestRunGPBaselineComparison:
    def test_gp_comparison_structure(self):
        torch.manual_seed(42)
        x_train = torch.randn(12, 4)
        x_test = torch.randn(6, 4)
        t = torch.linspace(0, 1, 15)
        y_train = torch.randn(12, 15)
        y_test = torch.randn(6, 15)

        results = run_gp_baseline_comparison(
            x_train, y_train, x_test, y_test, t, n_obs=5, n_epochs=20
        )

        assert "gp_baseline" in results
        assert "timeview_adaptive" in results
        assert "static_baseline" in results
        for key in results:
            assert "future_mse" in results[key]


class TestRunActiveSchedulingExperiment:
    def test_active_scheduling_structure(self, basic_dims):
        torch.manual_seed(42)
        x_test = torch.randn(4, basic_dims["input_dim"])
        t = torch.linspace(0, 1, 20)
        y_test = torch.randn(4, 20)
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        model.eval()

        results = run_active_scheduling_experiment(
            model, x_test, y_test, t, n_initial_obs=3, n_additional=5
        )

        assert "active_mse" in results
        assert "uniform_mse" in results
        assert "active_coverage" in results
        assert "uniform_coverage" in results
        assert "active_crps" in results
        assert "uniform_crps" in results
        assert "selected_times" in results
        assert "n_steps" in results

        assert len(results["active_mse"]) == 5
        assert len(results["uniform_mse"]) == 5
        assert len(results["selected_times"]) == 5

    def test_active_scheduling_selected_times_valid(self, basic_dims):
        torch.manual_seed(42)
        x_test = torch.randn(4, basic_dims["input_dim"])
        t = torch.linspace(0, 1, 20)
        y_test = torch.randn(4, 20)
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        model.eval()

        results = run_active_scheduling_experiment(
            model, x_test, y_test, t, n_initial_obs=3, n_additional=3
        )

        for time_val in results["selected_times"]:
            assert 0 <= time_val <= 1, f"Selected time {time_val} out of range [0, 1]"


class TestComputeUncertaintyDecomposition:
    def test_decomposition_components(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        model.eval()

        decomp = compute_uncertainty_decomposition(model, x, t[:5], y[:, :5], t)

        assert "t" in decomp
        assert "total_var" in decomp
        assert "prior_epistemic" in decomp
        assert "info_gained" in decomp
        assert "noise" in decomp
        assert "posterior_epistemic" in decomp

        assert decomp["total_var"].shape == (x.shape[0], len(t))
        assert decomp["prior_epistemic"].shape == (x.shape[0], len(t))

    def test_decomposition_identity(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        model.eval()

        decomp = compute_uncertainty_decomposition(model, x, t[:5], y[:, :5], t)

        recomputed = decomp["posterior_epistemic"] + decomp["noise"]
        assert torch.allclose(decomp["total_var"], recomputed, atol=1e-5)

    def test_decomposition_identity_alt(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        model.eval()

        decomp = compute_uncertainty_decomposition(model, x, t[:5], y[:, :5], t)

        recomputed = decomp["prior_epistemic"] - decomp["info_gained"] + decomp["noise"]
        assert torch.allclose(decomp["total_var"], recomputed, atol=1e-5)

    def test_decomposition_heteroscedastic(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptiveHeteroscedastic(**basic_dims)
        model.eval()

        decomp = compute_uncertainty_decomposition(model, x, t[:5], y[:, :5], t)

        assert decomp["total_var"].shape == (x.shape[0], len(t))
        noise = decomp["noise"]
        assert noise.shape == (x.shape[0], len(t))

    def test_info_gained_non_negative(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        train_model(model, x, t, y, n_epochs=30, n_obs=5, verbose=False)
        model.eval()

        decomp = compute_uncertainty_decomposition(model, x, t[:5], y[:, :5], t)

        assert (decomp["info_gained"] >= -1e-5).all(), \
            f"Info gained has negative values: min={decomp['info_gained'].min()}"
