import numpy as np
import pytest
import torch

from models import ProbabilisticEncoderFullCholesky, TimeviewAdaptiveFullCholesky, TimeviewStatic
from train import train_static_model


class TestTimeviewStatic:
    def test_construction(self, basic_dims):
        model = TimeviewStatic(input_dim=basic_dims["input_dim"], n_basis=basic_dims["n_basis"])
        assert model.n_basis == 7

    def test_forward_shapes(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewStatic(input_dim=basic_dims["input_dim"], n_basis=basic_dims["n_basis"])
        model.eval()
        with torch.no_grad():
            y_mean, y_var = model(x, t)
        assert y_mean.shape == (16, 30)
        assert y_var.shape == (16, 30)

    def test_variance_positive(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewStatic(input_dim=basic_dims["input_dim"])
        model.eval()
        with torch.no_grad():
            _, y_var = model(x, t)
        assert (y_var > 0).all(), "Static model variance should be positive"

    def test_variance_is_homoscedastic(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewStatic(input_dim=basic_dims["input_dim"])
        model.eval()
        with torch.no_grad():
            _, y_var = model(x, t)
        var_std = y_var.std(dim=1)
        assert (var_std < 1e-6).all(), "Static model variance should be constant across time"

    def test_loss_finite(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewStatic(input_dim=basic_dims["input_dim"])
        loss = model.loss(x, t, y)
        assert torch.isfinite(loss), f"Static loss not finite: {loss}"

    def test_trainable(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewStatic(input_dim=basic_dims["input_dim"])
        initial_loss = model.loss(x, t, y).item()
        train_static_model(model, x, t, y, n_epochs=30, lr=1e-3)
        final_loss = model.loss(x, t, y).item()
        assert final_loss < initial_loss, \
            f"Static training did not reduce loss: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_no_adaptation(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewStatic(input_dim=basic_dims["input_dim"])
        model.eval()
        with torch.no_grad():
            y_mean_1, _ = model(x, t)
            y_mean_2, _ = model(x, t)
        assert torch.allclose(y_mean_1, y_mean_2), "Static predictions should be deterministic"

    def test_static_forward_shape_extended(self):
        model = TimeviewStatic(input_dim=4, n_basis=7)
        x = torch.randn(8, 4)
        t = torch.linspace(0, 1, 20)
        y_mean, y_var = model(x, t)
        assert y_mean.shape == (8, 20)
        assert y_var.shape == (8, 20)

    def test_static_variance_homoscedastic_extended(self):
        model = TimeviewStatic(input_dim=4, n_basis=7)
        model.eval()
        x = torch.randn(4, 4)
        t = torch.linspace(0, 1, 20)
        with torch.no_grad():
            _, y_var = model(x, t)
        for i in range(4):
            assert torch.allclose(y_var[i], y_var[i, 0].expand_as(y_var[i]))


class TestFullCholeskyEncoder:
    def test_output_shape(self, seed, basic_dims):
        enc = ProbabilisticEncoderFullCholesky(
            input_dim=basic_dims["input_dim"],
            n_basis=basic_dims["n_basis"],
        )
        x = torch.randn(8, basic_dims["input_dim"])
        mu, Sigma, bias = enc(x)
        assert mu.shape == (8, 7)
        assert Sigma.shape == (8, 7, 7)

    def test_positive_definite(self, seed, basic_dims):
        enc = ProbabilisticEncoderFullCholesky(
            input_dim=basic_dims["input_dim"],
            n_basis=basic_dims["n_basis"],
        )
        x = torch.randn(4, basic_dims["input_dim"])
        _, Sigma, _ = enc(x)
        eigenvalues = torch.linalg.eigvalsh(Sigma)
        assert (eigenvalues > -1e-6).all(), \
            f"Cholesky covariance not PD: min eigenvalue={eigenvalues.min()}"

    def test_symmetric(self, seed, basic_dims):
        enc = ProbabilisticEncoderFullCholesky(
            input_dim=basic_dims["input_dim"],
            n_basis=basic_dims["n_basis"],
        )
        x = torch.randn(4, basic_dims["input_dim"])
        _, Sigma, _ = enc(x)
        diff = (Sigma - Sigma.transpose(1, 2)).abs().max()
        assert diff < 1e-5, f"Not symmetric: max diff={diff}"

    def test_full_cholesky_model_trainable(self, sample_data, basic_dims):
        x, t, y = sample_data
        model = TimeviewAdaptiveFullCholesky(
            input_dim=basic_dims["input_dim"], n_basis=basic_dims["n_basis"]
        )
        loss = model.loss(x, t, y, n_obs=5)
        assert torch.isfinite(loss), f"Full Cholesky loss not finite: {loss}"
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestFullCholeskyModel:
    def test_full_cholesky_forward(self):
        model = TimeviewAdaptiveFullCholesky(input_dim=4, n_basis=5)
        x = torch.randn(4, 4)
        t = torch.linspace(0, 1, 15)
        t_obs = t[:5]
        y_obs = torch.randn(4, 5)
        model.eval()
        with torch.no_grad():
            y_mean, y_var, _, _ = model.update_and_predict(x, t_obs, y_obs, t)
        assert y_mean.shape == (4, 15)
        assert y_var.shape == (4, 15)
        assert (y_var > 0).all()

    def test_full_cholesky_covariance_symmetric(self):
        encoder = ProbabilisticEncoderFullCholesky(input_dim=4, n_basis=5)
        x = torch.randn(4, 4)
        mu, Sigma, bias = encoder(x)
        assert mu.shape == (4, 5)
        assert Sigma.shape == (4, 5, 5)
        assert torch.allclose(Sigma, Sigma.transpose(1, 2), atol=1e-5)
        eigvals = torch.linalg.eigvalsh(Sigma)
        assert (eigvals > 0).all(), f"Not positive definite: min eigval={eigvals.min()}"

    def test_full_cholesky_loss(self):
        model = TimeviewAdaptiveFullCholesky(input_dim=4, n_basis=5)
        x = torch.randn(8, 4)
        t = torch.linspace(0, 1, 15)
        y = torch.randn(8, 15)
        loss = model.loss(x, t, y, n_obs=5)
        assert torch.isfinite(loss)
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None
