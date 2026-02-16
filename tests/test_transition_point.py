import torch

from timeview_adaptive import TimeviewAdaptive, train_model
from transition_point import TransitionPointAnalyzer


class TestTransitionPointAnalysis:
    def test_sample_trajectories_shape(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        analyzer = TransitionPointAnalyzer(model, n_samples=50)
        with torch.no_grad():
            mu_0, Sigma_0, bias = model.encode(x[:2])
            y_samples = analyzer.sample_trajectories(mu_0, Sigma_0, t)
        assert y_samples.shape == (2, 50, 30)

    def test_compute_derivative_shape(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        analyzer = TransitionPointAnalyzer(model)

        y_input = torch.randn(2, 50, 30)
        dy = analyzer.compute_derivative(y_input, t)
        assert dy.shape == (2, 50, 29), f"Wrong derivative shape: {dy.shape}"

    def test_full_analysis_returns_expected_keys(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        analyzer = TransitionPointAnalyzer(model, n_samples=20)
        result = analyzer.analyze(x[:2], t[:5], y[:2, :5], t)

        expected_keys = ["y_mean", "y_var", "y_samples", "transitions",
                         "coefficient_mean", "coefficient_std"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_transitions_has_batch_entries(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims)
        model.eval()

        analyzer = TransitionPointAnalyzer(model, n_samples=20)
        result = analyzer.analyze(x[:3], t[:5], y[:3, :5], t)
        assert len(result["transitions"]) == 3, "Should have one entry per batch sample"


class TestTransitionPointAnalyzerExtended:
    def test_analyzer_creation(self, basic_dims):
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        analyzer = TransitionPointAnalyzer(model, n_samples=50)
        assert analyzer.model is model
        assert analyzer.n_samples == 50

    def test_sample_trajectories(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        model.eval()

        analyzer = TransitionPointAnalyzer(model, n_samples=20)

        with torch.no_grad():
            _, _, mu_post, Sigma_post = model.update_and_predict(x[:2], t[:5], y[:2, :5], t)
            samples = analyzer.sample_trajectories(mu_post, Sigma_post, t)

        assert samples.shape == (2, 20, len(t))

    def test_compute_derivative(self, basic_dims):
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        analyzer = TransitionPointAnalyzer(model)

        t = torch.linspace(0, 1, 10)
        y = t.unsqueeze(0) * 2  # y = 2t, dy/dt = 2
        dy = analyzer.compute_derivative(y, t)
        assert dy.shape == (1, 9)
        assert torch.allclose(dy, torch.full_like(dy, 2.0), atol=1e-4)

    def test_analyze_full_pipeline(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptive(**basic_dims, learn_noise=True)
        train_model(model, x, t, y, n_epochs=30, n_obs=5, verbose=False)
        model.eval()

        analyzer = TransitionPointAnalyzer(model, n_samples=20)
        results = analyzer.analyze(x[:2], t[:5], y[:2, :5], t)

        assert "y_mean" in results
        assert "y_var" in results
        assert "y_samples" in results
        assert "transitions" in results
        assert "coefficient_mean" in results
        assert "coefficient_std" in results
        assert results["y_samples"].shape[0] == 2
        assert results["y_samples"].shape[1] == 20
        assert len(results["transitions"]) == 2
