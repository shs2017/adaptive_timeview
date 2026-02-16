import json

import numpy as np
import pytest
import torch

from timeview_adaptive import (
    ActiveObservationScheduler,
    AdaptiveGate,
    BayesianUpdater,
    GPBaseline,
    HeteroscedasticNoise,
    ProbabilisticEncoder,
    TemperatureScaling,
    TimeviewAdaptive,
    TimeviewAdaptiveGated,
    TimeviewAdaptiveGatedHeteroscedastic,
    TimeviewAdaptiveHeteroscedastic,
    bspline_basis,
    create_knots,
    evaluate_model,
    train_model,
)


# ========================== B-Spline Tests ==========================


class TestBSplineBasis:
    def test_output_shape(self):
        t = torch.linspace(0, 1, 50)
        knots = create_knots(7)
        Phi = bspline_basis(t, knots)
        assert Phi.shape == (50, 7)

    def test_output_shape_different_basis(self):
        t = torch.linspace(0, 1, 30)
        for n_basis in [5, 7, 9, 11]:
            knots = create_knots(n_basis)
            Phi = bspline_basis(t, knots)
            assert Phi.shape == (30, n_basis), f"Failed for n_basis={n_basis}"

    def test_partition_of_unity(self):
        t = torch.linspace(0.01, 0.99, 100)
        knots = create_knots(7)
        Phi = bspline_basis(t, knots)
        row_sums = Phi.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            f"B-spline partition of unity violated: max deviation {(row_sums - 1).abs().max()}"

    def test_non_negativity(self):
        t = torch.linspace(0, 1, 200)
        knots = create_knots(7)
        Phi = bspline_basis(t, knots)
        assert (Phi >= -1e-7).all(), "B-splines have negative values"

    def test_endpoints(self):
        t = torch.tensor([0.0, 1.0])
        knots = create_knots(7)
        Phi = bspline_basis(t, knots)
        assert Phi.shape == (2, 7)
        assert Phi[0].sum() > 0, "No basis function active at t=0"
        t_near_end = torch.tensor([0.999])
        Phi_near = bspline_basis(t_near_end, knots)
        assert Phi_near[0].sum() > 0, "No basis function active near t=1"


class TestCreateKnots:
    def test_knot_count(self):
        for n_basis in [5, 7, 9]:
            knots = create_knots(n_basis, degree=3)
            assert len(knots) == n_basis + 4, f"Wrong knot count for n_basis={n_basis}"

    def test_boundary_knots(self):
        knots = create_knots(7, t_min=0.0, t_max=1.0)
        assert knots[0] == 0.0
        assert knots[-1] == 1.0
        assert (knots[:3] == 0.0).all()
        assert (knots[-3:] == 1.0).all()

    def test_monotonicity(self):
        knots = create_knots(7)
        diffs = knots[1:] - knots[:-1]
        assert (diffs >= 0).all(), "Knots not monotonically non-decreasing"


class TestBSplineEdgeCases:
    def test_right_endpoint_handled(self):
        t = torch.tensor([1.0])
        knots = create_knots(7)
        Phi = bspline_basis(t, knots)
        assert Phi.shape == (1, 7)
        assert Phi.sum() > 0.5, "Right endpoint should have non-zero basis"
        assert torch.allclose(Phi.sum(), torch.tensor(1.0), atol=1e-5), \
            "Right endpoint should satisfy partition of unity"

    def test_single_point(self):
        t = torch.tensor([0.5])
        knots = create_knots(7)
        Phi = bspline_basis(t, knots)
        assert Phi.shape == (1, 7)

    def test_many_basis_functions(self):
        t = torch.linspace(0, 1, 100)
        for n_basis in [5, 7, 9, 11, 15]:
            knots = create_knots(n_basis)
            Phi = bspline_basis(t, knots)
            assert Phi.shape == (100, n_basis)
            interior = Phi[10:-10]
            sums = interior.sum(dim=1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
                f"Partition of unity violated for n_basis={n_basis}"


# ========================== Encoder Tests ==========================


class TestProbabilisticEncoder:
    def test_diagonal_output_shape(self, seed, basic_dims):
        enc = ProbabilisticEncoder(**basic_dims, covariance_type="diagonal")
        x = torch.randn(8, basic_dims["input_dim"])
        mu, Sigma, bias = enc(x)
        assert mu.shape == (8, 7)
        assert Sigma.shape == (8, 7, 7)

    def test_low_rank_output_shape(self, seed, basic_dims):
        enc = ProbabilisticEncoder(**basic_dims, covariance_type="low_rank")
        x = torch.randn(8, basic_dims["input_dim"])
        mu, Sigma, bias = enc(x)
        assert mu.shape == (8, 7)
        assert Sigma.shape == (8, 7, 7)

    def test_covariance_positive_definite(self, seed, basic_dims):
        for cov_type in ["diagonal", "low_rank"]:
            enc = ProbabilisticEncoder(**basic_dims, covariance_type=cov_type)
            x = torch.randn(8, basic_dims["input_dim"])
            _, Sigma, _ = enc(x)
            eigenvalues = torch.linalg.eigvalsh(Sigma)
            assert (eigenvalues > -1e-6).all(), \
                f"Non-PD covariance for {cov_type}: min eigenvalue={eigenvalues.min()}"

    def test_covariance_symmetric(self, seed, basic_dims):
        enc = ProbabilisticEncoder(**basic_dims, covariance_type="low_rank")
        x = torch.randn(4, basic_dims["input_dim"])
        _, Sigma, _ = enc(x)
        diff = (Sigma - Sigma.transpose(1, 2)).abs().max()
        assert diff < 1e-5, f"Covariance not symmetric: max asymmetry = {diff}"

    def test_diagonal_covariance_is_diagonal(self, seed, basic_dims):
        enc = ProbabilisticEncoder(**basic_dims, covariance_type="diagonal")
        x = torch.randn(4, basic_dims["input_dim"])
        _, Sigma, _ = enc(x)
        mask = ~torch.eye(7, dtype=torch.bool).unsqueeze(0).expand(4, -1, -1)
        off_diag = Sigma[mask].abs().max()
        assert off_diag < 1e-6, f"Diagonal covariance has off-diag elements: {off_diag}"

    def test_invalid_covariance_type(self, basic_dims):
        with pytest.raises(ValueError, match="Unknown covariance type"):
            ProbabilisticEncoder(**basic_dims, covariance_type="invalid")


# ========================== Bayesian Updater Tests ==========================


class TestBayesianUpdater:
    def test_output_shape(self, seed):
        updater = BayesianUpdater(observation_noise=0.1)
        batch, n_basis, n_obs = 4, 7, 10
        mu_0 = torch.randn(batch, n_basis)
        Sigma_0 = torch.eye(n_basis).unsqueeze(0).expand(batch, -1, -1)
        Phi = torch.randn(batch, n_obs, n_basis)
        y_obs = torch.randn(batch, n_obs)

        mu_n, Sigma_n = updater.update(mu_0, Sigma_0, Phi, y_obs)
        assert mu_n.shape == (batch, n_basis)
        assert Sigma_n.shape == (batch, n_basis, n_basis)

    def test_posterior_precision_increases(self, seed):
        updater = BayesianUpdater(observation_noise=0.1)
        batch, n_basis = 2, 5
        mu_0 = torch.zeros(batch, n_basis)
        Sigma_0 = torch.eye(n_basis).unsqueeze(0).expand(batch, -1, -1).clone()
        Phi = torch.randn(batch, 3, n_basis)
        y_obs = torch.randn(batch, 3)

        _, Sigma_n = updater.update(mu_0, Sigma_0, Phi, y_obs)

        prior_diag = torch.diagonal(Sigma_0, dim1=1, dim2=2)
        post_diag = torch.diagonal(Sigma_n, dim1=1, dim2=2)
        assert (post_diag <= prior_diag + 1e-5).all(), "Posterior variance exceeds prior"

    def test_monotonic_variance_reduction(self, seed):
        updater = BayesianUpdater(observation_noise=0.1)
        n_basis = 5
        mu_0 = torch.zeros(1, n_basis)
        Sigma_0 = torch.eye(n_basis).unsqueeze(0)

        t = torch.linspace(0, 1, 20)
        knots = create_knots(n_basis)
        Phi_all = bspline_basis(t, knots).unsqueeze(0)

        prev_trace = float("inf")
        for n_obs in [1, 3, 5, 10, 15]:
            y_obs = torch.randn(1, n_obs)
            _, Sigma_n = updater.update(mu_0, Sigma_0, Phi_all[:, :n_obs], y_obs)
            trace = torch.diagonal(Sigma_n, dim1=1, dim2=2).sum().item()
            assert trace <= prev_trace + 1e-5, \
                f"Variance increased at n_obs={n_obs}: {trace} > {prev_trace}"
            prev_trace = trace

    def test_heteroscedastic_update(self, seed):
        updater = BayesianUpdater(observation_noise=0.1)
        batch, n_basis, n_obs = 2, 5, 8
        mu_0 = torch.zeros(batch, n_basis)
        Sigma_0 = torch.eye(n_basis).unsqueeze(0).expand(batch, -1, -1).clone()
        Phi = torch.randn(batch, n_obs, n_basis)
        y_obs = torch.randn(batch, n_obs)
        noise_var = torch.ones(n_obs) * 0.01

        mu_n, Sigma_n = updater.update(mu_0, Sigma_0, Phi, y_obs, noise_var=noise_var)
        assert mu_n.shape == (batch, n_basis)
        assert Sigma_n.shape == (batch, n_basis, n_basis)

    def test_high_noise_preserves_prior(self, seed):
        updater = BayesianUpdater(observation_noise=100.0)
        n_basis = 5
        mu_0 = torch.randn(1, n_basis)
        Sigma_0 = torch.eye(n_basis).unsqueeze(0)
        Phi = torch.randn(1, 3, n_basis)
        y_obs = torch.randn(1, 3)

        mu_n, Sigma_n = updater.update(mu_0, Sigma_0, Phi, y_obs)
        assert torch.allclose(mu_n, mu_0, atol=0.1), "High noise should preserve prior mean"
        assert torch.allclose(Sigma_n, Sigma_0, atol=0.1), "High noise should preserve prior cov"


class TestBayesianUpdateEdgeCases:
    def test_single_observation(self, seed):
        updater = BayesianUpdater(observation_noise=0.1)
        n_basis = 5
        mu_0 = torch.zeros(1, n_basis)
        Sigma_0 = torch.eye(n_basis).unsqueeze(0)
        Phi = torch.randn(1, 1, n_basis)
        y_obs = torch.randn(1, 1)

        mu_n, Sigma_n = updater.update(mu_0, Sigma_0, Phi, y_obs)
        assert mu_n.shape == (1, n_basis)
        assert Sigma_n.shape == (1, n_basis, n_basis)

    def test_many_observations(self, seed):
        updater = BayesianUpdater(observation_noise=0.01)
        n_basis = 5
        mu_0 = torch.zeros(1, n_basis)
        Sigma_0 = torch.eye(n_basis).unsqueeze(0)

        n_obs = 100
        Phi = torch.randn(1, n_obs, n_basis)
        y_obs = torch.randn(1, n_obs)

        mu_n, Sigma_n = updater.update(mu_0, Sigma_0, Phi, y_obs)
        post_var = torch.diagonal(Sigma_n, dim1=1, dim2=2)
        prior_var = torch.diagonal(Sigma_0, dim1=1, dim2=2)
        assert (post_var < prior_var * 0.1).all(), \
            "100 observations should dramatically reduce posterior variance"

    def test_heteroscedastic_varied_noise(self, seed):
        updater = BayesianUpdater(observation_noise=0.1)
        n_basis = 5
        mu_0 = torch.zeros(2, n_basis)
        Sigma_0 = torch.eye(n_basis).unsqueeze(0).expand(2, -1, -1).clone()
        Phi = torch.randn(2, 5, n_basis)
        y_obs = torch.randn(2, 5)

        noise_var = torch.tensor([0.001, 0.001, 100.0, 100.0, 100.0])
        mu_n, Sigma_n = updater.update(mu_0, Sigma_0, Phi, y_obs, noise_var=noise_var)

        assert mu_n.shape == (2, n_basis)
        assert torch.isfinite(mu_n).all()
        assert torch.isfinite(Sigma_n).all()


# ========================== Noise Model Tests ==========================


class TestHeteroscedasticNoise:
    def test_output_shape(self):
        noise = HeteroscedasticNoise()
        t = torch.linspace(0, 1, 50)
        sigma2 = noise(t)
        assert sigma2.shape == (50,)

    def test_positive_variance(self):
        noise = HeteroscedasticNoise()
        t = torch.linspace(0, 1, 100)
        sigma2 = noise(t)
        assert (sigma2 > 0).all(), "Noise variance should be positive"

    def test_initial_value(self):
        base = 0.2
        noise = HeteroscedasticNoise(base_noise=base)
        t = torch.linspace(0, 1, 50)
        sigma2 = noise(t)
        expected = base**2
        assert torch.allclose(sigma2, torch.full_like(sigma2, expected), atol=1e-3), \
            f"Initial noise variance {sigma2.mean():.4f} far from expected {expected}"


class TestHeteroscedasticNoiseEdgeCases:
    def test_extreme_time_values(self):
        noise = HeteroscedasticNoise()
        t = torch.tensor([-100.0, 0.0, 1.0, 100.0])
        sigma2 = noise(t)
        assert torch.isfinite(sigma2).all(), f"Non-finite noise at extreme t: {sigma2}"
        assert (sigma2 > 0).all(), "Noise should be positive at all times"

    def test_output_clamped(self):
        noise = HeteroscedasticNoise()
        t = torch.linspace(-10, 10, 100)
        sigma2 = noise(t)
        assert (sigma2 >= np.exp(-10) - 1e-6).all(), "Sigma2 below clamp lower bound"
        assert (sigma2 <= np.exp(4) + 1e-6).all(), "Sigma2 above clamp upper bound"


# ========================== Model Tests ==========================


class TestTimeviewAdaptive:
    def test_construction(self, basic_dims):
        model = TimeviewAdaptive(**basic_dims)
        assert model.n_basis == 7

    def test_forward_shapes(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        with torch.no_grad():
            n_obs = 10
            y_mean, y_var, mu_post, Sigma_post = model.update_and_predict(
                x, t[:n_obs], y[:, :n_obs], t
            )
        assert y_mean.shape == (16, 30)
        assert y_var.shape == (16, 30)
        assert mu_post.shape == (16, 7)
        assert Sigma_post.shape == (16, 7, 7)

    def test_variance_positive(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        with torch.no_grad():
            y_mean, y_var, _, _ = model.update_and_predict(x, t[:5], y[:, :5], t)
        assert (y_var > 0).all(), "Predictive variance should be positive"

    def test_loss_finite(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        loss = model.loss(x, t, y, n_obs=5)
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    def test_loss_components(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        loss, components = model.loss(x, t, y, n_obs=5, return_components=True)
        assert "nll" in components
        assert "kl" in components
        assert np.isfinite(components["nll"])
        assert np.isfinite(components["kl"])
        assert components["kl"] >= 0, "KL divergence should be non-negative"

    def test_kl_divergence_zero_for_standard_normal(self, basic_dims):
        model = TimeviewAdaptive(**basic_dims)
        mu = torch.zeros(4, 7)
        Sigma = torch.eye(7).unsqueeze(0).expand(4, -1, -1)
        kl = model.kl_divergence(mu, Sigma)
        assert kl.abs() < 0.1, f"KL(N(0,I) || N(0,I)) should be ~0, got {kl}"

    def test_encode_prior(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        mu_0, Sigma_0, bias = model.encode(x)
        assert mu_0.shape == (16, 7)
        assert Sigma_0.shape == (16, 7, 7)

    def test_gradient_flow(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        loss = model.loss(x, t, y, n_obs=5)
        loss.backward()
        for name, param in model.encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


class TestTimeviewAdaptiveHeteroscedastic:
    def test_construction(self, basic_dims):
        model = TimeviewAdaptiveHeteroscedastic(**basic_dims)
        assert hasattr(model, "noise_model")

    def test_forward_shapes(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptiveHeteroscedastic(**basic_dims)
        model.eval()

        with torch.no_grad():
            y_mean, y_var, _, _ = model.update_and_predict(x, t[:5], y[:, :5], t)
        assert y_mean.shape == (16, 30)
        assert y_var.shape == (16, 30)

    def test_loss_finite(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptiveHeteroscedastic(**basic_dims)
        loss = model.loss(x, t, y, n_obs=5)
        assert torch.isfinite(loss), f"Heteroscedastic loss not finite: {loss}"


class TestTimeviewAdaptiveGated:
    def test_construction(self, basic_dims):
        model = TimeviewAdaptiveGated(**basic_dims)
        assert hasattr(model, "gate")

    def test_forward_shapes(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptiveGated(**basic_dims)
        model.eval()

        with torch.no_grad():
            y_mean, y_var, _, _ = model.update_and_predict(x, t[:5], y[:, :5], t)
        assert y_mean.shape == (16, 30)
        assert y_var.shape == (16, 30)

    def test_gate_output_range(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptiveGated(**basic_dims)
        model.eval()

        with torch.no_grad():
            mu_0, Sigma_0, bias = model.encode(x)
            y_prior, y_prior_var = model.predict(mu_0, Sigma_0, t[:5], bias=bias)
            prior_mse = ((y[:, :5] - y_prior) ** 2).mean(dim=1, keepdim=True)
            obs_unc = y_prior_var.mean(dim=1, keepdim=True)
            gate_val = model.gate(x, prior_mse, obs_unc)

        assert (gate_val >= 0).all() and (gate_val <= 1).all(), \
            f"Gate values outside [0,1]: min={gate_val.min()}, max={gate_val.max()}"


class TestTimeviewAdaptiveGatedHeteroscedastic:
    def test_construction(self):
        model = TimeviewAdaptiveGatedHeteroscedastic(input_dim=4)
        assert hasattr(model, "gate")
        assert hasattr(model, "noise_model")
        assert model.n_basis == 9

    def test_forward_shapes(self, sample_data):
        x, t, y = sample_data
        model = TimeviewAdaptiveGatedHeteroscedastic(input_dim=4, n_basis=7)
        model.eval()

        with torch.no_grad():
            y_mean, y_var, _, _ = model.update_and_predict(x, t[:5], y[:, :5], t)
        assert y_mean.shape == (16, 30)
        assert y_var.shape == (16, 30)

    def test_loss_finite(self, sample_data):
        x, t, y = sample_data
        model = TimeviewAdaptiveGatedHeteroscedastic(input_dim=4, n_basis=7)
        loss = model.loss(x, t, y, n_obs=5)
        assert torch.isfinite(loss), f"GatedHeteroscedastic loss not finite: {loss}"


class TestHeteroscedasticModelBehavior:
    def test_get_noise_var_positive(self, basic_dims):
        model = TimeviewAdaptiveHeteroscedastic(**basic_dims)
        t = torch.linspace(0, 1, 50)
        noise_var = model.get_noise_var(t)
        assert (noise_var > 0).all(), "Noise variance should be positive"

    def test_noise_variance_varies_after_training(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptiveHeteroscedastic(**basic_dims)
        train_model(model, x, t, y, n_epochs=50, lr=1e-3, n_obs=5, verbose=False)
        model.eval()

        with torch.no_grad():
            noise_var = model.get_noise_var(t)
        assert noise_var.shape == (30,)
        assert torch.isfinite(noise_var).all()

    def test_heteroscedastic_update_uses_per_obs_noise(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptiveHeteroscedastic(**basic_dims)
        model.eval()

        with torch.no_grad():
            mu_0, Sigma_0, bias = model.encode(x)
            noise_var_obs = model.get_noise_var(t[:5])
            assert noise_var_obs.shape == (5,)

            y_mean, y_var, mu_post, Sigma_post = model.update_and_predict(
                x, t[:5], y[:, :5], t
            )
            assert y_mean.shape == (16, 30)


class TestHeteroscedasticLossComponents:
    def test_heteroscedastic_loss_components(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptiveHeteroscedastic(**basic_dims)
        loss, components = model.loss(x, t, y, n_obs=5, return_components=True)
        assert torch.isfinite(loss)
        assert "nll" in components
        assert "kl" in components
        assert np.isfinite(components["nll"])
        assert components["kl"] >= 0

    def test_gated_heteroscedastic_loss_components(self, sample_data):
        x, t, y = sample_data
        model = TimeviewAdaptiveGatedHeteroscedastic(input_dim=4, n_basis=7)
        loss, components = model.loss(x, t, y, n_obs=5, return_components=True)
        assert torch.isfinite(loss)
        assert "nll" in components
        assert "kl" in components


# ========================== Active Observation Tests ==========================


class TestActiveObservationScheduler:
    def test_information_gain_shape(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()
        scheduler = ActiveObservationScheduler(model)

        with torch.no_grad():
            _, _, mu_post, Sigma_post = model.update_and_predict(x, t[:5], y[:, :5], t)
            ig = scheduler.compute_information_gain(mu_post, Sigma_post, t)
        assert ig.shape == (16, 30)

    def test_information_gain_non_negative(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()
        scheduler = ActiveObservationScheduler(model)

        with torch.no_grad():
            _, _, mu_post, Sigma_post = model.update_and_predict(x, t[:5], y[:, :5], t)
            ig = scheduler.compute_information_gain(mu_post, Sigma_post, t)
        assert (ig >= -1e-5).all(), f"Negative IG found: min={ig.min()}"

    def test_suggest_next_observation(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()
        scheduler = ActiveObservationScheduler(model)

        best_times, ig = scheduler.suggest_next_observation(
            x, t[:5], y[:, :5], t[5:]
        )
        assert best_times.shape == (16,)
        assert ig.shape == (16, 25)

    def test_suggest_picks_highest_ig(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()
        scheduler = ActiveObservationScheduler(model)

        best_times, ig = scheduler.suggest_next_observation(
            x[:1], t[:5], y[:1, :5], t[5:]
        )
        expected_idx = ig[0].argmax().item()
        expected_time = t[5:][expected_idx]
        assert torch.allclose(best_times[0], expected_time), \
            f"Suggested time {best_times[0]} != max IG time {expected_time}"


# ========================== GP Baseline Tests ==========================


class TestGPBaseline:
    def test_prediction_shape(self):
        gp = GPBaseline()
        t_obs = torch.linspace(0, 1, 5)
        y_obs = torch.randn(4, 5)
        t_pred = torch.linspace(0, 1, 20)

        y_mean, y_var = gp.predict(t_obs, y_obs, t_pred)
        assert y_mean.shape == (4, 20)
        assert y_var.shape == (4, 20)

    def test_variance_positive(self):
        gp = GPBaseline()
        t_obs = torch.linspace(0, 1, 5)
        y_obs = torch.randn(2, 5)
        t_pred = torch.linspace(0, 1, 20)

        _, y_var = gp.predict(t_obs, y_obs, t_pred)
        assert (y_var > 0).all(), "GP variance should be positive"

    def test_interpolation(self):
        gp = GPBaseline(noise_var=1e-6)
        t_obs = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        y_obs = torch.tensor([[0.0, 1.0, 0.0, -1.0, 0.0]])

        y_mean, _ = gp.predict(t_obs, y_obs, t_obs)
        assert torch.allclose(y_mean, y_obs, atol=0.1), \
            "GP should interpolate observations with low noise"


class TestGPBaselineRBFKernel:
    def test_kernel_shape(self):
        gp = GPBaseline()
        t1 = torch.linspace(0, 1, 10)
        t2 = torch.linspace(0, 1, 15)
        K = gp.rbf_kernel(t1, t2)
        assert K.shape == (10, 15)

    def test_kernel_symmetric(self):
        gp = GPBaseline()
        t = torch.linspace(0, 1, 10)
        K = gp.rbf_kernel(t, t)
        assert torch.allclose(K, K.T, atol=1e-6), "Kernel should be symmetric"

    def test_kernel_positive_definite(self):
        gp = GPBaseline()
        t = torch.linspace(0, 1, 10)
        K = gp.rbf_kernel(t, t)
        eigenvalues = torch.linalg.eigvalsh(K)
        assert (eigenvalues >= -1e-6).all(), \
            f"Kernel not PD: min eigenvalue={eigenvalues.min()}"

    def test_kernel_diagonal_is_signal_var(self):
        gp = GPBaseline(signal_var=2.0)
        t = torch.linspace(0, 1, 10)
        K = gp.rbf_kernel(t, t)
        diag = torch.diag(K)
        assert torch.allclose(diag, torch.ones(10) * 2.0, atol=1e-5), \
            "Diagonal should equal signal_var"

    def test_kernel_decays_with_distance(self):
        gp = GPBaseline(length_scale=0.1)
        t = torch.tensor([0.0, 0.01, 0.5])
        K = gp.rbf_kernel(t, t)
        assert K[0, 1] > K[0, 2], "Kernel should decay with distance"


# ========================== Integration Tests ==========================


class TestIntegration:
    def test_training_reduces_loss(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)

        initial_loss = model.loss(x, t, y, n_obs=5).item()
        train_model(model, x, t, y, n_epochs=20, lr=1e-3, n_obs=5, verbose=False)
        final_loss = model.loss(x, t, y, n_obs=5).item()

        assert final_loss < initial_loss, \
            f"Training did not reduce loss: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_posterior_variance_less_than_prior(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        train_model(model, x, t, y, n_epochs=30, lr=1e-3, n_obs=10, verbose=False)
        model.eval()

        with torch.no_grad():
            mu_0, Sigma_0, bias = model.encode(x)
            _, y_prior_var = model.predict(mu_0, Sigma_0, t)

            n_obs = 10
            _, y_post_var, _, _ = model.update_and_predict(x, t[:n_obs], y[:, :n_obs], t)

        assert (y_post_var <= y_prior_var + 1e-5).all(), \
            "Posterior variance exceeds prior variance"

    def test_all_model_variants_trainable(self, sample_data):
        x, t, y = sample_data
        input_dim = x.shape[1]

        model_classes = [
            (TimeviewAdaptive, {"n_basis": 7}),
            (TimeviewAdaptiveHeteroscedastic, {"n_basis": 7}),
            (TimeviewAdaptiveGated, {"n_basis": 7}),
            (TimeviewAdaptiveGatedHeteroscedastic, {"n_basis": 7}),
        ]

        for cls, extra_kwargs in model_classes:
            model = cls(input_dim=input_dim, **extra_kwargs)
            train_model(model, x, t, y, n_epochs=5, lr=1e-3, n_obs=5, verbose=False)
            loss = model.loss(x, t, y, n_obs=5)
            assert torch.isfinite(loss), f"{cls.__name__} produced non-finite loss"


# ========================== Bug Fix Tests ==========================


class TestBugFix_DoubleEncode:
    def test_loss_uses_precomputed_prior(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, dropout_p=0.3)
        model.train()

        call_count = [0]
        original_encode = model.encode

        def counting_encode(x):
            call_count[0] += 1
            return original_encode(x)

        model.encode = counting_encode
        model.loss(x, t, y, n_obs=5)

        assert call_count[0] == 1, f"encode called {call_count[0]} times, expected 1"

    def test_loss_passes_prior_to_update_and_predict(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, dropout_p=0.3)
        model.train()

        received_kwargs = {}
        original_uap = model.update_and_predict

        def capturing_uap(x, t_obs, y_obs, t_pred, mu_0=None, Sigma_0=None, bias=None):
            received_kwargs["mu_0"] = mu_0
            received_kwargs["Sigma_0"] = Sigma_0
            return original_uap(x, t_obs, y_obs, t_pred, mu_0=mu_0, Sigma_0=Sigma_0, bias=bias)

        model.update_and_predict = capturing_uap
        model.loss(x, t, y, n_obs=5)

        assert received_kwargs["mu_0"] is not None, "mu_0 not passed to update_and_predict"
        assert received_kwargs["Sigma_0"] is not None, "Sigma_0 not passed to update_and_predict"

    def test_heteroscedastic_loss_encodes_once(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptiveHeteroscedastic(**basic_dims, dropout_p=0.3)
        model.train()

        call_count = [0]
        original_encode = model.encode

        def counting_encode(x):
            call_count[0] += 1
            return original_encode(x)

        model.encode = counting_encode
        model.loss(x, t, y, n_obs=5)
        assert call_count[0] == 1, f"Heteroscedastic encode called {call_count[0]} times"

    def test_gated_heteroscedastic_loss_encodes_once(self, sample_data):
        x, t, y = sample_data
        model = TimeviewAdaptiveGatedHeteroscedastic(input_dim=4, n_basis=7, dropout_p=0.3)
        model.train()

        call_count = [0]
        original_encode = model.encode

        def counting_encode(x):
            call_count[0] += 1
            return original_encode(x)

        model.encode = counting_encode
        model.loss(x, t, y, n_obs=5)
        assert call_count[0] == 1, f"GatedHeteroscedastic encode called {call_count[0]} times"


class TestBugFix_GatedVarianceCrossTerm:
    def test_gated_variance_exceeds_component_average(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptiveGated(**basic_dims)
        model.eval()

        with torch.no_grad():
            mu_0, Sigma_0, bias = model.encode(x)
            y_prior_mean, y_prior_var = model.predict(mu_0, Sigma_0, t, bias=bias)

            Phi_obs = model.get_basis(t[:5]).unsqueeze(0).expand(x.shape[0], -1, -1)
            model.updater.sigma2 = model.sigma ** 2
            mu_post, Sigma_post = model.updater.update(mu_0, Sigma_0, Phi_obs, y[:, :5])
            y_post_mean, y_post_var = model.predict(mu_post, Sigma_post, t, bias=bias)

            y_gated_mean, y_gated_var, _, _ = model.update_and_predict(x, t[:5], y[:, :5], t)

            y_prior_at_obs, y_prior_var_at_obs = model.predict(mu_0, Sigma_0, t[:5], bias=bias)
            prior_mse = ((y[:, :5] - y_prior_at_obs) ** 2).mean(dim=1, keepdim=True)
            obs_unc = y_prior_var_at_obs.mean(dim=1, keepdim=True)
            gate_w = model.gate(x, prior_mse, obs_unc)

            weighted_avg = gate_w * y_post_var + (1 - gate_w) * y_prior_var

            assert (y_gated_var >= weighted_avg - 1e-5).all(), \
                "Gated variance missing cross-term: should be >= weighted average"

    def test_cross_term_is_positive(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptiveGated(**basic_dims)
        model.eval()

        with torch.no_grad():
            mu_0, Sigma_0, bias = model.encode(x)
            y_prior_mean, y_prior_var = model.predict(mu_0, Sigma_0, t, bias=bias)

            Phi_obs = model.get_basis(t[:5]).unsqueeze(0).expand(x.shape[0], -1, -1)
            model.updater.sigma2 = model.sigma ** 2
            mu_post, Sigma_post = model.updater.update(mu_0, Sigma_0, Phi_obs, y[:, :5])
            y_post_mean, y_post_var = model.predict(mu_post, Sigma_post, t, bias=bias)

            y_prior_at_obs, y_prior_var_at_obs = model.predict(mu_0, Sigma_0, t[:5], bias=bias)
            prior_mse = ((y[:, :5] - y_prior_at_obs) ** 2).mean(dim=1, keepdim=True)
            obs_unc = y_prior_var_at_obs.mean(dim=1, keepdim=True)
            gate_w = model.gate(x, prior_mse, obs_unc)

            cross_term = gate_w * (1 - gate_w) * (y_post_mean - y_prior_mean) ** 2
            assert (cross_term >= -1e-7).all(), "Cross-term should be non-negative"

    def test_gated_heteroscedastic_variance_cross_term(self, sample_data):
        x, t, y = sample_data
        model = TimeviewAdaptiveGatedHeteroscedastic(input_dim=4, n_basis=7)
        model.eval()

        with torch.no_grad():
            mu_0, Sigma_0, bias = model.encode(x)
            y_prior_mean, y_prior_var = model.predict_with_noise(mu_0, Sigma_0, t, bias=bias)

            Phi_obs = model.get_basis(t[:5]).unsqueeze(0).expand(x.shape[0], -1, -1)
            noise_var_obs = model.get_noise_var(t[:5])
            mu_post, Sigma_post = model.updater.update(
                mu_0, Sigma_0, Phi_obs, y[:, :5], noise_var=noise_var_obs
            )
            y_post_mean, y_post_var = model.predict_with_noise(mu_post, Sigma_post, t, bias=bias)

            y_gated_mean, y_gated_var, _, _ = model.update_and_predict(x, t[:5], y[:, :5], t)

            y_prior_at_obs, y_prior_var_at_obs = model.predict_with_noise(mu_0, Sigma_0, t[:5], bias=bias)
            prior_mse = ((y[:, :5] - y_prior_at_obs) ** 2).mean(dim=1, keepdim=True)
            obs_unc = y_prior_var_at_obs.mean(dim=1, keepdim=True)
            gate_w = model.gate(x, prior_mse, obs_unc)

            weighted_avg = gate_w * y_post_var + (1 - gate_w) * y_prior_var
            assert (y_gated_var >= weighted_avg - 1e-5).all(), \
                "GatedHeteroscedastic variance missing cross-term"


# ========================== Precomputed Prior Tests ==========================


class TestUpdateAndPredictPrecomputed:
    def test_precomputed_matches_internal(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        with torch.no_grad():
            mu_0, Sigma_0, bias = model.encode(x)
            y1, v1, mp1, sp1 = model.update_and_predict(
                x, t[:5], y[:, :5], t, mu_0=mu_0, Sigma_0=Sigma_0, bias=bias
            )
            y2, v2, mp2, sp2 = model.update_and_predict(x, t[:5], y[:, :5], t)

        assert torch.allclose(y1, y2, atol=1e-5), "Pre-computed should match internal encode"
        assert torch.allclose(v1, v2, atol=1e-5), "Pre-computed variance should match"

    def test_heteroscedastic_precomputed(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptiveHeteroscedastic(**basic_dims)
        model.eval()

        with torch.no_grad():
            mu_0, Sigma_0, bias = model.encode(x)
            y1, v1, _, _ = model.update_and_predict(
                x, t[:5], y[:, :5], t, mu_0=mu_0, Sigma_0=Sigma_0, bias=bias
            )
            y2, v2, _, _ = model.update_and_predict(x, t[:5], y[:, :5], t)

        assert torch.allclose(y1, y2, atol=1e-5)

    def test_gated_precomputed(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptiveGated(**basic_dims)
        model.eval()

        with torch.no_grad():
            mu_0, Sigma_0, bias = model.encode(x)
            y1, v1, _, _ = model.update_and_predict(
                x, t[:5], y[:, :5], t, mu_0=mu_0, Sigma_0=Sigma_0, bias=bias
            )
            y2, v2, _, _ = model.update_and_predict(x, t[:5], y[:, :5], t)

        assert torch.allclose(y1, y2, atol=1e-5)


# ========================== Gate Tests ==========================


class TestAdaptiveGateDetailed:
    def test_gate_shape(self, sample_data, basic_dims):
        x, t, y = sample_data
        gate = AdaptiveGate(input_dim=basic_dims["input_dim"])
        prior_mse = torch.randn(16, 1).abs()
        obs_unc = torch.randn(16, 1).abs()
        output = gate(x, prior_mse, obs_unc)
        assert output.shape == (16, 1)

    def test_gate_bounded(self, sample_data, basic_dims):
        x, t, y = sample_data
        gate = AdaptiveGate(input_dim=basic_dims["input_dim"])
        for scale in [0.001, 1.0, 100.0]:
            prior_mse = torch.randn(16, 1).abs() * scale
            obs_unc = torch.randn(16, 1).abs() * scale
            output = gate(x * scale, prior_mse, obs_unc)
            assert (output >= 0).all() and (output <= 1).all(), \
                f"Gate outside [0,1] at scale {scale}"


class TestGatedModelBehavior:
    def test_gated_loss_finite(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptiveGated(**basic_dims)
        loss = model.loss(x, t, y, n_obs=5)
        assert torch.isfinite(loss), f"Gated loss not finite: {loss}"
        loss.backward()
        for name, param in model.gate.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for gate.{name}"

    def test_gated_loss_components(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptiveGated(**basic_dims)
        loss, components = model.loss(x, t, y, n_obs=5, return_components=True)
        assert "nll" in components
        assert "kl" in components


# ========================== KL Divergence Tests ==========================


class TestKLDivergence:
    def test_kl_zero_for_standard_normal(self, basic_dims):
        model = TimeviewAdaptive(**basic_dims)
        mu = torch.zeros(4, 7)
        Sigma = torch.eye(7).unsqueeze(0).expand(4, -1, -1)
        kl = model.kl_divergence(mu, Sigma)
        assert abs(kl.item()) < 0.1, f"KL(standard normal) should be ~0, got {kl.item()}"

    def test_kl_positive_for_shifted_mean(self, basic_dims):
        model = TimeviewAdaptive(**basic_dims)
        mu = torch.ones(4, 7) * 2.0
        Sigma = torch.eye(7).unsqueeze(0).expand(4, -1, -1)
        kl = model.kl_divergence(mu, Sigma)
        assert kl.item() > 0, f"KL should be positive for shifted mean, got {kl.item()}"

    def test_kl_positive_for_scaled_covariance(self, basic_dims):
        model = TimeviewAdaptive(**basic_dims)
        mu = torch.zeros(4, 7)
        Sigma = (torch.eye(7) * 2.0).unsqueeze(0).expand(4, -1, -1)
        kl = model.kl_divergence(mu, Sigma)
        assert kl.item() > 0, f"KL should be positive for scaled covariance, got {kl.item()}"

    def test_kl_symmetric_in_mean(self, basic_dims):
        model = TimeviewAdaptive(**basic_dims)
        Sigma = torch.eye(7).unsqueeze(0).expand(4, -1, -1)
        mu_pos = torch.ones(4, 7)
        mu_neg = -torch.ones(4, 7)
        kl_pos = model.kl_divergence(mu_pos, Sigma)
        kl_neg = model.kl_divergence(mu_neg, Sigma)
        assert abs(kl_pos.item() - kl_neg.item()) < 0.01, \
            f"KL should be symmetric in mean: {kl_pos.item()} vs {kl_neg.item()}"


# ========================== Train & Evaluate Tests ==========================


class TestTrainModel:
    def test_early_stopping(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        losses = train_model(
            model, x, t, y, n_epochs=500, lr=1e-2, n_obs=5,
            patience=5, verbose=False
        )
        assert len(losses) <= 500, f"Ran {len(losses)} epochs"
        assert all(np.isfinite(l) for l in losses), "All losses should be finite"

    def test_returns_loss_history(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        losses = train_model(model, x, t, y, n_epochs=10, lr=1e-3, n_obs=5, verbose=False)
        assert len(losses) == 10
        assert all(np.isfinite(l) for l in losses)

    def test_validation_set(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        x_val = x[:4]
        y_val = y[:4]
        losses = train_model(
            model, x, t, y, n_epochs=10, lr=1e-3, n_obs=5,
            x_val=x_val, y_val=y_val, verbose=False
        )
        assert len(losses) == 10

    def test_gradient_clipping(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        optimizer.zero_grad()
        loss = model.loss(x, t, y, n_obs=5)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        assert torch.isfinite(total_norm), f"Gradient norm not finite: {total_norm}"


class TestEvaluateModel:
    def test_returns_expected_keys(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        metrics = evaluate_model(model, x, t, y, n_obs=5)
        assert "mse_future" in metrics
        assert "coverage_2std" in metrics
        assert "avg_uncertainty" in metrics

    def test_mse_non_negative(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        metrics = evaluate_model(model, x, t, y, n_obs=5)
        assert metrics["mse_future"] >= 0

    def test_coverage_in_range(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        metrics = evaluate_model(model, x, t, y, n_obs=5)
        assert 0 <= metrics["coverage_2std"] <= 1


# ========================== Predict Tests ==========================


class TestPredictDirect:
    def test_predict_shape(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        with torch.no_grad():
            mu = torch.randn(16, 7)
            Sigma = torch.eye(7).unsqueeze(0).expand(16, -1, -1)
            y_mean, y_var = model.predict(mu, Sigma, t)
        assert y_mean.shape == (16, 30)
        assert y_var.shape == (16, 30)

    def test_predict_mean_is_linear(self, basic_dims):
        model = TimeviewAdaptive(**basic_dims)
        model.eval()
        t = torch.linspace(0, 1, 20)

        with torch.no_grad():
            mu = torch.randn(2, 7)
            Sigma = torch.eye(7).unsqueeze(0).expand(2, -1, -1)
            y_mean_1, _ = model.predict(mu, Sigma, t)
            y_mean_2, _ = model.predict(2 * mu, Sigma, t)
        assert torch.allclose(y_mean_2, 2 * y_mean_1, atol=1e-5), \
            "Mean prediction should be linear in mu"

    def test_predict_variance_independent_of_mean(self, basic_dims):
        model = TimeviewAdaptive(**basic_dims)
        model.eval()
        t = torch.linspace(0, 1, 20)

        with torch.no_grad():
            Sigma = torch.eye(7).unsqueeze(0).expand(2, -1, -1)
            _, y_var_1 = model.predict(torch.zeros(2, 7), Sigma, t)
            _, y_var_2 = model.predict(torch.ones(2, 7) * 5.0, Sigma, t)
        assert torch.allclose(y_var_1, y_var_2, atol=1e-5), \
            "Variance should not depend on mu"

    def test_get_basis_shape(self, basic_dims):
        model = TimeviewAdaptive(**basic_dims)
        t = torch.linspace(0, 1, 25)
        Phi = model.get_basis(t)
        assert Phi.shape == (25, 7)


class TestPredictWithNoise:
    def test_shape(self, sample_data):
        x, t, y = sample_data
        model = TimeviewAdaptiveGatedHeteroscedastic(input_dim=4, n_basis=7)
        model.eval()

        with torch.no_grad():
            mu = torch.randn(16, 7)
            Sigma = torch.eye(7).unsqueeze(0).expand(16, -1, -1)
            y_mean, y_var = model.predict_with_noise(mu, Sigma, t)
        assert y_mean.shape == (16, 30)
        assert y_var.shape == (16, 30)

    def test_variance_includes_noise(self, sample_data):
        x, t, y = sample_data
        model = TimeviewAdaptiveGatedHeteroscedastic(input_dim=4, n_basis=7)
        model.eval()

        with torch.no_grad():
            mu = torch.randn(4, 7)
            Sigma = torch.eye(7).unsqueeze(0).expand(4, -1, -1)
            y_mean, y_var = model.predict_with_noise(mu, Sigma, t)

            Phi = model.get_basis(t)
            PhiSigma = torch.matmul(Phi.unsqueeze(0), Sigma)
            epistemic_only = torch.sum(PhiSigma * Phi.unsqueeze(0), dim=-1)

        assert (y_var >= epistemic_only - 1e-6).all(), \
            "Total variance should be >= epistemic variance"


# ========================== Fixed Noise Tests ==========================


class TestFixedNoise:
    def test_learn_noise_false(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, learn_noise=False)
        initial_sigma = model.sigma.item()

        train_model(model, x, t, y, n_epochs=20, lr=1e-3, n_obs=5, verbose=False)
        final_sigma = model.sigma.item()

        assert abs(initial_sigma - final_sigma) < 1e-6, \
            f"Fixed noise changed: {initial_sigma} -> {final_sigma}"

    def test_learn_noise_true(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)

        assert isinstance(model.log_sigma, torch.nn.Parameter), \
            "learn_noise=True should make log_sigma a Parameter"

    def test_sigma_property(self, basic_dims):
        model = TimeviewAdaptive(**basic_dims, observation_noise=0.2)
        expected = torch.exp(torch.log(torch.tensor(0.2)))
        assert abs(model.sigma.item() - expected.item()) < 1e-5


# ========================== Temperature Scaling Tests ==========================


class TestTemperatureScaling:
    def test_initial_temperature_is_one(self):
        scaler = TemperatureScaling()
        assert abs(scaler.temperature.item() - 1.0) < 1e-5

    def test_apply_scales_variance(self):
        scaler = TemperatureScaling()
        scaler.log_temperature.data = torch.log(torch.tensor(4.0))
        var = torch.ones(5, 10)
        scaled = scaler.apply(var)
        assert torch.allclose(scaled, var * 4.0, atol=1e-5)

    def test_calibrate_finds_temperature(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        train_model(model, x, t, y, n_epochs=20, lr=1e-3, n_obs=5, verbose=False)

        scaler = TemperatureScaling()
        temp = scaler.calibrate(model, x, t, y, n_obs=5, target_coverage=0.95)
        assert temp > 0, f"Temperature should be positive: {temp}"
        assert 0.1 <= temp <= 100, f"Temperature out of search range: {temp}"

    def test_calibrate_improves_coverage(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        train_model(model, x, t, y, n_epochs=30, lr=1e-3, n_obs=5, verbose=False)
        model.eval()

        with torch.no_grad():
            y_mean, y_var, _, _ = model.update_and_predict(x, t[:5], y[:, :5], t)

        scaler = TemperatureScaling()
        scaler.calibrate(model, x, t, y, n_obs=5, target_coverage=0.95)

        with torch.no_grad():
            var_scaled = scaler.apply(y_var)

        z = 1.96
        cov_before = (torch.abs(y - y_mean) < z * torch.sqrt(y_var)).float().mean().item()
        cov_after = (torch.abs(y - y_mean) < z * torch.sqrt(var_scaled)).float().mean().item()
        assert abs(cov_after - 0.95) <= abs(cov_before - 0.95) + 0.05, \
            f"Scaling should improve coverage: before={cov_before:.3f}, after={cov_after:.3f}"


# ========================== Serialization Tests ==========================


class TestConvertToSerializable:
    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = arr.tolist()
        assert result == [1.0, 2.0, 3.0]
        json.dumps(result)

    def test_torch_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = t.tolist()
        assert result == [1.0, 2.0, 3.0]
        json.dumps(result)

    def test_numpy_scalar_types(self):
        val32 = np.float32(1.5)
        val64 = np.float64(2.5)
        int32 = np.int32(10)
        int64 = np.int64(20)

        assert isinstance(float(val32), float)
        assert isinstance(float(val64), float)
        assert isinstance(int(int32), int)
        assert isinstance(int(int64), int)

        json.dumps({"a": float(val32), "b": float(val64), "c": int(int32), "d": int(int64)})

    def test_nested_dict_conversion(self):
        nested = {
            "array": np.array([1, 2, 3]).tolist(),
            "tensor": torch.tensor([4.0, 5.0]).tolist(),
            "scalar": float(np.float64(1.5)),
            "nested": {
                "inner": np.array([6, 7]).tolist(),
            },
            "list": [float(np.float32(x)) for x in [1.0, 2.0]],
        }
        json_str = json.dumps(nested)
        recovered = json.loads(json_str)
        assert recovered["array"] == [1, 2, 3]
        assert recovered["scalar"] == 1.5
