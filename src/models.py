import torch
import torch.nn as nn
import torch.nn.functional as F

from timeview_adaptive import (
    BayesianUpdater,
    TimeviewAdaptive,
    bspline_basis,
    create_knots,
)


class ProbabilisticEncoderFullCholesky(nn.Module):

    def __init__(self, input_dim: int, hidden_sizes: list[int] | None = None, n_basis: int = 9):
        super().__init__()
        self.n_basis = n_basis
        if hidden_sizes is None:
            hidden_sizes = [32, 64, 32]

        layers = []
        layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        self.feature_net = nn.Sequential(*layers)
        last_hidden = hidden_sizes[-1]

        self.mean_head = nn.Linear(last_hidden, n_basis + 1)

        n_cholesky_params = n_basis * (n_basis + 1) // 2
        self.cholesky_head = nn.Linear(last_hidden, n_cholesky_params)

        self.register_buffer("tril_indices", torch.tril_indices(n_basis, n_basis))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        features = self.feature_net(x)
        output = self.mean_head(features)
        mu_0 = output[:, :self.n_basis]
        bias = output[:, self.n_basis:]

        chol_params = self.cholesky_head(features)

        L = torch.zeros(batch_size, self.n_basis, self.n_basis, device=x.device, dtype=x.dtype)
        L[:, self.tril_indices[0], self.tril_indices[1]] = chol_params

        diag_indices = torch.arange(self.n_basis, device=x.device)
        L[:, diag_indices, diag_indices] = F.softplus(L[:, diag_indices, diag_indices]) + 1e-4

        Sigma_0 = torch.bmm(L, L.transpose(1, 2))

        return mu_0, Sigma_0, bias


class TimeviewAdaptiveFullCholesky(TimeviewAdaptive):

    def __init__(
        self,
        input_dim: int,
        n_basis: int = 9,
        hidden_sizes: list[int] | None = None,
        observation_noise: float = 0.1,
        learn_noise: bool = True,
        kl_weight: float = 0.01,
        knots: torch.Tensor | None = None,
    ):
        nn.Module.__init__(self)

        self.n_basis = n_basis
        self.kl_weight = kl_weight
        self.learn_noise = learn_noise
        self.encoder = ProbabilisticEncoderFullCholesky(input_dim, hidden_sizes, n_basis)
        self.updater = BayesianUpdater(observation_noise)

        if knots is not None:
            self.register_buffer("knots", knots)
        else:
            self.register_buffer("knots", create_knots(n_basis))

        if learn_noise:
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(observation_noise)))
        else:
            self.register_buffer("log_sigma", torch.log(torch.tensor(observation_noise)))



class TimeviewStatic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_basis: int = 9,
        hidden_sizes: list[int] | None = None,
        dropout_p: float = 0.2,
        knots: torch.Tensor | None = None,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 64, 32]

        self.n_basis = n_basis

        layers = []
        layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_p))

        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))

        layers.append(nn.Linear(hidden_sizes[-1], n_basis + 1))
        self.encoder = nn.Sequential(*layers)

        if knots is not None:
            self.register_buffer("knots", knots)
        else:
            self.register_buffer("knots", create_knots(n_basis))

    def get_basis(self, t: torch.Tensor) -> torch.Tensor:
        return bspline_basis(t, self.knots)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.encoder(x)
        coeffs = output[:, :self.n_basis]
        bias = output[:, self.n_basis:]
        Phi = self.get_basis(t)

        y_mean = torch.matmul(coeffs, Phi.T) + bias
        y_var = 0.01 * torch.ones_like(y_mean)

        return y_mean, y_var

    def loss(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        output = self.encoder(x)

        coeffs = output[:, :self.n_basis]
        bias = output[:, self.n_basis:]

        Phi = self.get_basis(t)
        y_mean = torch.matmul(coeffs, Phi.T) + bias

        return nn.functional.mse_loss(y_mean, y)

    def loss_per_sample(self, x: torch.Tensor, Phis: list[torch.Tensor], ys: list[torch.Tensor]) -> torch.Tensor:
        output = self.encoder(x)
        coeffs = output[:, :self.n_basis]
        bias = output[:, self.n_basis:]
        losses = []
        for d in range(len(ys)):
            pred = Phis[d] @ coeffs[d] + bias[d, 0]
            losses.append(nn.functional.mse_loss(pred, ys[d]))
        return torch.mean(torch.stack(losses))

    def predict_per_sample(self, x: torch.Tensor, Phis: list[torch.Tensor]) -> list[torch.Tensor]:
        output = self.encoder(x)
        coeffs = output[:, :self.n_basis]
        bias = output[:, self.n_basis:]

        preds = []
        for d in range(x.shape[0]):
            pred = Phis[d] @ coeffs[d] + bias[d, 0]
            preds.append(pred)

        return preds

