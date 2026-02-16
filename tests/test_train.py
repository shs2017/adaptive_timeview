import numpy as np
import torch

from models import TimeviewStatic
from train import train_static_model


class TestTrainStaticModel:
    def test_train_static_model_loss_decreases(self, capsys):
        torch.manual_seed(42)
        model = TimeviewStatic(input_dim=4, n_basis=7)
        x = torch.randn(16, 4)
        t = torch.linspace(0, 1, 20)
        y = torch.randn(16, 20)

        losses = train_static_model(model, x, t, y, n_epochs=100, lr=1e-3)
        assert len(losses) == 100
        assert losses[-1] < losses[0], "Loss should decrease during training"
        captured = capsys.readouterr()
        assert "50/100" in captured.out

    def test_train_static_model_returns_list(self):
        model = TimeviewStatic(input_dim=4, n_basis=7)
        x = torch.randn(8, 4)
        t = torch.linspace(0, 1, 20)
        y = torch.randn(8, 20)

        losses = train_static_model(model, x, t, y, n_epochs=10)
        assert isinstance(losses, list)
        assert len(losses) == 10
        assert all(np.isfinite(l) for l in losses)
