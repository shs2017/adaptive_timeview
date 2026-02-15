from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from constant import (
    PROJECT_ROOT,
    FIGURES_DIR,
    TABLES_DIR
)

def bspline_basis(t: torch.Tensor, knots: torch.Tensor, degree: int = 3) -> torch.Tensor:
    n_knots = len(knots)
    n_basis = n_knots - degree - 1

    t_flat = t.reshape(-1)
    T = len(t_flat)

    B = torch.zeros(T, n_knots - 1, device=t.device, dtype=t.dtype)
    for i in range(n_knots - 1):
        B[:, i] = ((t_flat >= knots[i]) & (t_flat < knots[i + 1])).float()

    for d in range(1, degree + 1):
        B_new = torch.zeros(T, n_knots - d - 1, device=t.device, dtype=t.dtype)
        for i in range(n_knots - d - 1):
            denom1 = knots[i + d] - knots[i]
            if denom1 > 0:
                B_new[:, i] += (t_flat - knots[i]) / denom1 * B[:, i]

            denom2 = knots[i + d + 1] - knots[i + 1]
            if denom2 > 0:
                B_new[:, i] += (knots[i + d + 1] - t_flat) / denom2 * B[:, i + 1]

        B = B_new

    right_mask = t_flat == knots[-1]
    if right_mask.any():
        B[right_mask, :] = 0.0
        B[right_mask, -1] = 1.0

    output_shape = list(t.shape) + [n_basis]
    return B.reshape(output_shape)


def create_knots(
    n_basis: int, degree: int = 3, t_min: float = 0.0, t_max: float = 1.0,
    internal_knots: np.ndarray | None = None,
) -> torch.Tensor:
    if internal_knots is not None:
        internal = torch.as_tensor(internal_knots, dtype=torch.float32)
        knots = torch.cat([
            torch.full((degree,), t_min),
            internal,
            torch.full((degree,), t_max),
        ])
    else:
        n_internal = n_basis - degree + 1
        internal = torch.linspace(t_min, t_max, n_internal)
        knots = torch.cat([torch.full((degree,), t_min), internal, torch.full((degree,), t_max)])
    return knots


class ProbabilisticEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int] | None = None,
        n_basis: int = 9,
        covariance_type: str = "diagonal",
        dropout_p: float = 0.2,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 64, 32]
        self.n_basis = n_basis
        self.covariance_type = covariance_type
        self.use_batchnorm = use_batchnorm

        layers = []
        layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_p))

        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))

        self.feature_net = nn.Sequential(*layers)

        last_hidden = hidden_sizes[-1]

        self.mean_head = nn.Linear(last_hidden, n_basis + 1)

        if covariance_type == "diagonal":
            self.logvar_head = nn.Linear(last_hidden, n_basis)
        elif covariance_type == "low_rank":
            self.rank = min(3, n_basis)
            self.logvar_head = nn.Linear(last_hidden, n_basis)
            self.lowrank_head = nn.Linear(last_hidden, n_basis * self.rank)
        else:
            raise ValueError(f"Unknown covariance type: {covariance_type}")

        nn.init.constant_(self.logvar_head.bias, 0.0)
        nn.init.zeros_(self.logvar_head.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_net(x)
        output = self.mean_head(features)  # (batch, n_basis + 1)
        mu_0 = output[:, :self.n_basis]  # (batch, n_basis)
        bias = output[:, self.n_basis:]  # (batch, 1)

        if self.covariance_type == "diagonal":
            logvar = self.logvar_head(features)

            # Clamp for numerical stability
            logvar = torch.clamp(logvar, min=-10, max=10)
            var = torch.exp(logvar)
            Sigma_0 = torch.diag_embed(var)

        elif self.covariance_type == "low_rank":
            logvar = self.logvar_head(features)
            logvar = torch.clamp(logvar, min=-10, max=10)
            var = torch.exp(logvar)

            U = self.lowrank_head(features).reshape(-1, self.n_basis, self.rank)
            Sigma_0 = torch.diag_embed(var) + torch.bmm(U, U.transpose(1, 2))

        return mu_0, Sigma_0, bias


class HeteroscedasticNoise(nn.Module):
    def __init__(self, hidden_dim: int = 16, base_noise: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[2].weight)
        nn.init.constant_(self.net[2].bias, np.log(base_noise**2))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_input = t.reshape(-1, 1)
        log_sigma2 = self.net(t_input)
        log_sigma2 = torch.clamp(log_sigma2, min=-10, max=4)
        sigma2 = torch.exp(log_sigma2).reshape(t.shape)
        return sigma2


class BayesianUpdater:
    def __init__(self, observation_noise: float = 0.1):
        self.sigma2 = observation_noise**2

    def update(
        self, mu_0: torch.Tensor, Sigma_0: torch.Tensor, Phi: torch.Tensor, y_obs: torch.Tensor,
        noise_var: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_basis = mu_0.shape[1]

        Sigma_0_inv = torch.linalg.inv(Sigma_0 + 1e-6 * torch.eye(n_basis, device=Sigma_0.device))

        if noise_var is not None:
            weights = 1.0 / (noise_var + 1e-8)  # (n_obs,)
            Phi_weighted = Phi * weights.unsqueeze(0).unsqueeze(-1)  # (batch, n_obs, n_basis)
            PhiTPhi = torch.bmm(Phi_weighted.transpose(1, 2), Phi)  # (batch, n_basis, n_basis)
            PhiTy = torch.bmm(Phi_weighted.transpose(1, 2), y_obs.unsqueeze(-1)).squeeze(-1)

            precision_n = Sigma_0_inv + PhiTPhi
            Sigma_n = torch.linalg.inv(
                precision_n + 1e-6 * torch.eye(n_basis, device=precision_n.device)
            )

            prior_term = torch.bmm(Sigma_0_inv, mu_0.unsqueeze(-1)).squeeze(-1)
            mu_n = torch.bmm(Sigma_n, (prior_term + PhiTy).unsqueeze(-1)).squeeze(-1)
        else:
            PhiTPhi = torch.bmm(Phi.transpose(1, 2), Phi)  # (batch, n_basis, n_basis)
            precision_n = Sigma_0_inv + PhiTPhi / self.sigma2
            Sigma_n = torch.linalg.inv(
                precision_n + 1e-6 * torch.eye(n_basis, device=precision_n.device)
            )
            PhiTy = torch.bmm(Phi.transpose(1, 2), y_obs.unsqueeze(-1)).squeeze(-1)
            prior_term = torch.bmm(Sigma_0_inv, mu_0.unsqueeze(-1)).squeeze(-1)
            mu_n = torch.bmm(Sigma_n, (prior_term + PhiTy / self.sigma2).unsqueeze(-1)).squeeze(-1)

        return mu_n, Sigma_n


class TimeviewAdaptive(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_basis: int = 9,
        hidden_sizes: list[int] | None = None,
        observation_noise: float = 0.1,
        covariance_type: str = "diagonal",
        dropout_p: float = 0.2,
        use_batchnorm: bool = True,
        learn_noise: bool = True,
        kl_weight: float = 0.01,
        knots: torch.Tensor | None = None,
    ):
        super().__init__()

        self.n_basis = n_basis
        self.kl_weight = kl_weight
        self.learn_noise = learn_noise
        self.encoder = ProbabilisticEncoder(
            input_dim, hidden_sizes, n_basis, covariance_type, dropout_p, use_batchnorm
        )
        self.updater = BayesianUpdater(observation_noise)

        if knots is not None:
            self.register_buffer("knots", knots)
        else:
            self.register_buffer("knots", create_knots(n_basis))

        if learn_noise:
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(observation_noise)))
        else:
            self.register_buffer("log_sigma", torch.log(torch.tensor(observation_noise)))

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    def get_basis(self, t: torch.Tensor) -> torch.Tensor:
        return bspline_basis(t, self.knots)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def predict(
        self, mu: torch.Tensor, Sigma: torch.Tensor, t: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Phi = self.get_basis(t)  # (T, n_basis)

        y_mean = torch.matmul(mu, Phi.T)  # (batch, T)
        if bias is not None:
            y_mean = y_mean + bias

        PhiSigma = torch.matmul(Phi.unsqueeze(0), Sigma)
        y_var = torch.sum(PhiSigma * Phi.unsqueeze(0), dim=-1) + self.sigma * 2

        return y_mean, y_var

    def update_and_predict(
        self, x: torch.Tensor, t_obs: torch.Tensor, y_obs: torch.Tensor, t_pred: torch.Tensor,
        mu_0: torch.Tensor | None = None, Sigma_0: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mu_0 is None or Sigma_0 is None:
            mu_0, Sigma_0, bias = self.encode(x)

        Phi_obs = self.get_basis(t_obs)  # (n_obs, n_basis)
        Phi_obs = Phi_obs.unsqueeze(0).expand(x.shape[0], -1, -1)  # (batch, n_obs, n_basis)

        y_obs_centered = y_obs
        if bias is not None:
            y_obs_centered = y_obs - bias

        self.updater.sigma2 = self.sigma**2
        mu_post, Sigma_post = self.updater.update(mu_0, Sigma_0, Phi_obs, y_obs_centered)

        y_mean, y_var = self.predict(mu_post, Sigma_post, t_pred, bias)

        return y_mean, y_var, mu_post, Sigma_post

    def kl_divergence(self, mu: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
        n_basis = mu.shape[1]

        trace_Sigma = torch.diagonal(Sigma, dim1=1, dim2=2).sum(dim=1)  # (batch,)
        mu_sq = (mu**2).sum(dim=1)  # (batch,)

        sign, logdet = torch.linalg.slogdet(Sigma + 1e-6 * torch.eye(n_basis, device=Sigma.device))
        logdet = torch.where(sign > 0, logdet, torch.zeros_like(logdet))

        kl = 0.5 * (trace_Sigma + mu_sq - n_basis - logdet)
        return kl.mean()

    def loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        n_obs: int = 5,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        t_obs = t[:n_obs]
        y_obs = y[:, :n_obs]

        mu_0, Sigma_0, bias = self.encode(x)

        y_mean, y_var, _, _ = self.update_and_predict(
            x, t_obs, y_obs, t, mu_0=mu_0, Sigma_0=Sigma_0, bias=bias
        )

        nll = 0.5 * torch.log(2 * np.pi * y_var) + 0.5 * (y - y_mean) ** 2 / y_var
        nll_loss = nll.mean()

        kl_loss = self.kl_divergence(mu_0, Sigma_0)

        total_loss = nll_loss + self.kl_weight * kl_loss

        if return_components:
            return total_loss, {"nll": nll_loss.item(), "kl": kl_loss.item()}
        return total_loss


class TimeviewAdaptiveHeteroscedastic(TimeviewAdaptive):
    def __init__(
        self,
        input_dim: int,
        n_basis: int = 9,
        hidden_sizes: list[int] | None = None,
        observation_noise: float = 0.1,
        covariance_type: str = "diagonal",
        dropout_p: float = 0.2,
        use_batchnorm: bool = True,
        kl_weight: float = 0.01,
        noise_hidden_dim: int = 16,
        knots: torch.Tensor | None = None,
    ):
        super().__init__(
            input_dim=input_dim,
            n_basis=n_basis,
            hidden_sizes=hidden_sizes,
            observation_noise=observation_noise,
            covariance_type=covariance_type,
            dropout_p=dropout_p,
            use_batchnorm=use_batchnorm,
            learn_noise=False,
            kl_weight=kl_weight,
            knots=knots,
        )
        self.noise_model = HeteroscedasticNoise(noise_hidden_dim, observation_noise)

    def get_noise_var(self, t: torch.Tensor) -> torch.Tensor:
        return self.noise_model(t)

    def predict(
        self, mu: torch.Tensor, Sigma: torch.Tensor, t: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Phi = self.get_basis(t)
        y_mean = torch.matmul(mu, Phi.T)
        if bias is not None:
            y_mean = y_mean + bias

        PhiSigma = torch.matmul(Phi.unsqueeze(0), Sigma)
        epistemic_var = torch.sum(PhiSigma * Phi.unsqueeze(0), dim=-1)

        noise_var = self.get_noise_var(t)
        y_var = epistemic_var + noise_var.unsqueeze(0)

        return y_mean, y_var

    def update_and_predict(
        self, x: torch.Tensor, t_obs: torch.Tensor, y_obs: torch.Tensor, t_pred: torch.Tensor,
        mu_0: torch.Tensor | None = None, Sigma_0: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mu_0 is None or Sigma_0 is None:
            mu_0, Sigma_0, bias = self.encode(x)
        Phi_obs = self.get_basis(t_obs).unsqueeze(0).expand(x.shape[0], -1, -1)

        y_obs_centered = y_obs
        if bias is not None:
            y_obs_centered = y_obs - bias

        noise_var_obs = self.get_noise_var(t_obs)
        mu_post, Sigma_post = self.updater.update(
            mu_0, Sigma_0, Phi_obs, y_obs_centered, noise_var=noise_var_obs
        )

        y_mean, y_var = self.predict(mu_post, Sigma_post, t_pred, bias)
        return y_mean, y_var, mu_post, Sigma_post

    def loss(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor,
        n_obs: int = 5, return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        t_obs = t[:n_obs]
        y_obs = y[:, :n_obs]

        mu_0, Sigma_0, bias = self.encode(x)

        y_mean, y_var, _, _ = self.update_and_predict(
            x, t_obs, y_obs, t, mu_0=mu_0, Sigma_0=Sigma_0, bias=bias
        )

        nll = 0.5 * torch.log(2 * np.pi * y_var) + 0.5 * (y - y_mean) ** 2 / y_var
        nll_loss = nll.mean()

        kl_loss = self.kl_divergence(mu_0, Sigma_0)
        total_loss = nll_loss + self.kl_weight * kl_loss

        if return_components:
            return total_loss, {"nll": nll_loss.item(), "kl": kl_loss.item()}
        return total_loss


class AdaptiveGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()

        self.gate_net = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, prior_mse: torch.Tensor, obs_uncertainty: torch.Tensor
    ) -> torch.Tensor:
        gate_input = torch.cat([x, prior_mse, obs_uncertainty], dim=-1)
        return self.gate_net(gate_input)


class TimeviewAdaptiveGated(TimeviewAdaptive):
    def __init__(
        self,
        input_dim: int,
        n_basis: int = 9,
        hidden_sizes: list[int] | None = None,
        observation_noise: float = 0.1,
        covariance_type: str = "diagonal",
        dropout_p: float = 0.2,
        use_batchnorm: bool = True,
        learn_noise: bool = True,
        kl_weight: float = 0.01,
        knots: torch.Tensor | None = None,
    ):
        super().__init__(
            input_dim=input_dim,
            n_basis=n_basis,
            hidden_sizes=hidden_sizes,
            observation_noise=observation_noise,
            covariance_type=covariance_type,
            dropout_p=dropout_p,
            use_batchnorm=use_batchnorm,
            learn_noise=learn_noise,
            kl_weight=kl_weight,
            knots=knots,
        )
        self.gate = AdaptiveGate(input_dim, hidden_dim=32)

    def update_and_predict(
        self, x: torch.Tensor, t_obs: torch.Tensor, y_obs: torch.Tensor, t_pred: torch.Tensor,
        mu_0: torch.Tensor | None = None, Sigma_0: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if mu_0 is None or Sigma_0 is None:
            mu_0, Sigma_0, bias = self.encode(x)

        y_prior_mean, y_prior_var = super().predict(mu_0, Sigma_0, t_pred, bias)

        y_prior_at_obs, y_prior_var_at_obs = super().predict(mu_0, Sigma_0, t_obs, bias)
        prior_mse = ((y_obs - y_prior_at_obs) ** 2).mean(dim=1, keepdim=True)
        obs_uncertainty = y_prior_var_at_obs.mean(dim=1, keepdim=True)

        Phi_obs = self.get_basis(t_obs).unsqueeze(0).expand(x.shape[0], -1, -1)
        y_obs_centered = y_obs
        if bias is not None:
            y_obs_centered = y_obs - bias
        self.updater.sigma2 = self.sigma**2
        mu_post, Sigma_post = self.updater.update(mu_0, Sigma_0, Phi_obs, y_obs_centered)

        y_post_mean, y_post_var = super().predict(mu_post, Sigma_post, t_pred, bias)

        gate_weight = self.gate(x, prior_mse, obs_uncertainty)

        y_mean = gate_weight * y_post_mean + (1 - gate_weight) * y_prior_mean
        y_var = (gate_weight * y_post_var + (1 - gate_weight) * y_prior_var
                 + gate_weight * (1 - gate_weight) * (y_post_mean - y_prior_mean) ** 2)

        return y_mean, y_var, mu_post, Sigma_post


class TimeviewAdaptiveGatedHeteroscedastic(TimeviewAdaptive):
    def __init__(
        self,
        input_dim: int,
        n_basis: int = 9,
        hidden_sizes: list[int] | None = None,
        observation_noise: float = 0.1,
        covariance_type: str = "low_rank",
        dropout_p: float = 0.2,
        use_batchnorm: bool = True,
        kl_weight: float = 0.01,
        noise_hidden_dim: int = 16,
        knots: torch.Tensor | None = None,
    ):
        super().__init__(
            input_dim=input_dim,
            n_basis=n_basis,
            hidden_sizes=hidden_sizes,
            observation_noise=observation_noise,
            covariance_type=covariance_type,
            dropout_p=dropout_p,
            use_batchnorm=use_batchnorm,
            learn_noise=False,
            kl_weight=kl_weight,
            knots=knots,
        )
        self.noise_model = HeteroscedasticNoise(noise_hidden_dim, observation_noise)
        self.gate = AdaptiveGate(input_dim, hidden_dim=32)

    def get_noise_var(self, t: torch.Tensor) -> torch.Tensor:
        return self.noise_model(t)

    def predict_with_noise(
        self, mu: torch.Tensor, Sigma: torch.Tensor, t: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Phi = self.get_basis(t)
        y_mean = torch.matmul(mu, Phi.T)
        if bias is not None:
            y_mean = y_mean + bias
        PhiSigma = torch.matmul(Phi.unsqueeze(0), Sigma)
        epistemic_var = torch.sum(PhiSigma * Phi.unsqueeze(0), dim=-1)
        noise_var = self.get_noise_var(t)
        y_var = epistemic_var + noise_var.unsqueeze(0)
        return y_mean, y_var

    def update_and_predict(
        self, x: torch.Tensor, t_obs: torch.Tensor, y_obs: torch.Tensor, t_pred: torch.Tensor,
        mu_0: torch.Tensor | None = None, Sigma_0: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mu_0 is None or Sigma_0 is None:
            mu_0, Sigma_0, bias = self.encode(x)

        y_prior_mean, y_prior_var = self.predict_with_noise(mu_0, Sigma_0, t_pred, bias)

        y_prior_at_obs, y_prior_var_at_obs = self.predict_with_noise(mu_0, Sigma_0, t_obs, bias)
        prior_mse = ((y_obs - y_prior_at_obs) ** 2).mean(dim=1, keepdim=True)
        obs_uncertainty = y_prior_var_at_obs.mean(dim=1, keepdim=True)

        Phi_obs = self.get_basis(t_obs).unsqueeze(0).expand(x.shape[0], -1, -1)
        y_obs_centered = y_obs
        if bias is not None:
            y_obs_centered = y_obs - bias
        noise_var_obs = self.get_noise_var(t_obs)
        mu_post, Sigma_post = self.updater.update(
            mu_0, Sigma_0, Phi_obs, y_obs_centered, noise_var=noise_var_obs
        )

        y_post_mean, y_post_var = self.predict_with_noise(mu_post, Sigma_post, t_pred, bias)

        gate_weight = self.gate(x, prior_mse, obs_uncertainty)
        y_mean = gate_weight * y_post_mean + (1 - gate_weight) * y_prior_mean
        y_var = (gate_weight * y_post_var + (1 - gate_weight) * y_prior_var
                 + gate_weight * (1 - gate_weight) * (y_post_mean - y_prior_mean) ** 2)

        return y_mean, y_var, mu_post, Sigma_post

    def loss(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor,
        n_obs: int = 5, return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        t_obs = t[:n_obs]
        y_obs = y[:, :n_obs]

        mu_0, Sigma_0, bias = self.encode(x)
        y_mean, y_var, _, _ = self.update_and_predict(
            x, t_obs, y_obs, t, mu_0=mu_0, Sigma_0=Sigma_0, bias=bias
        )

        nll = 0.5 * torch.log(2 * np.pi * y_var) + 0.5 * (y - y_mean) ** 2 / y_var
        nll_loss = nll.mean()
        kl_loss = self.kl_divergence(mu_0, Sigma_0)
        total_loss = nll_loss + self.kl_weight * kl_loss

        if return_components:
            return total_loss, {"nll": nll_loss.item(), "kl": kl_loss.item()}
        return total_loss


class ActiveObservationScheduler:

    def __init__(self, model: TimeviewAdaptive):
        self.model = model

    def compute_information_gain(
        self,
        mu_post: torch.Tensor,
        Sigma_post: torch.Tensor,
        t_candidates: torch.Tensor,
        sigma2: float | None = None,
    ) -> torch.Tensor:
        if sigma2 is None:
            sigma2 = self.model.sigma.item() ** 2

        batch_size = mu_post.shape[0]
        n_basis = mu_post.shape[1]
        n_candidates = len(t_candidates)

        _, current_logdet = torch.linalg.slogdet(
            Sigma_post + 1e-6 * torch.eye(n_basis, device=Sigma_post.device)
        )

        Phi_cand = self.model.get_basis(t_candidates)

        ig = torch.zeros(batch_size, n_candidates, device=mu_post.device)

        for j in range(n_candidates):
            phi_j = Phi_cand[j]

            Sigma_phi = torch.matmul(Sigma_post, phi_j.unsqueeze(-1))
            phi_Sigma_phi = torch.matmul(phi_j.unsqueeze(0), Sigma_phi).squeeze(-1)

            Sigma_new = Sigma_post - torch.bmm(Sigma_phi, Sigma_phi.transpose(1, 2)) / (
                sigma2 + phi_Sigma_phi.unsqueeze(-1)
            )

            _, new_logdet = torch.linalg.slogdet(
                Sigma_new + 1e-6 * torch.eye(n_basis, device=Sigma_new.device)
            )

            ig[:, j] = 0.5 * (current_logdet - new_logdet)

        return ig

    def suggest_next_observation(
        self,
        x: torch.Tensor,
        t_obs: torch.Tensor,
        y_obs: torch.Tensor,
        t_candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            _, _, mu_post, Sigma_post = self.model.update_and_predict(
                x, t_obs, y_obs, t_obs
            )

            ig = self.compute_information_gain(mu_post, Sigma_post, t_candidates)

            best_idx = ig.argmax(dim=1)
            best_times = t_candidates[best_idx]

        return best_times, ig


class GPBaseline:
    def __init__(self, length_scale: float = 0.2, signal_var: float = 1.0, noise_var: float = 0.01):
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.noise_var = noise_var

    def rbf_kernel(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:

        dist_sq = (t1.unsqueeze(-1) - t2.unsqueeze(-2)) ** 2
        return self.signal_var * torch.exp(-dist_sq / (2 * self.length_scale**2))

    def predict(
        self,
        t_obs: torch.Tensor,
        y_obs: torch.Tensor,
        t_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_obs = len(t_obs)

        K_obs = self.rbf_kernel(t_obs, t_obs) + self.noise_var * torch.eye(
            n_obs, device=t_obs.device
        )
        K_pred_obs = self.rbf_kernel(t_pred, t_obs)
        K_pred = self.rbf_kernel(t_pred, t_pred)

        L = torch.linalg.cholesky(K_obs + 1e-6 * torch.eye(n_obs, device=K_obs.device))

        alpha = torch.cholesky_solve(y_obs.T, L)

        y_mean = (K_pred_obs @ alpha).T

        V = torch.linalg.solve_triangular(L, K_pred_obs.T, upper=False)
        y_var = (torch.diag(K_pred) - (V * V).sum(dim=0) + self.noise_var).unsqueeze(
            0
        ).expand(y_obs.shape[0], -1)

        return y_mean, y_var


def train_model(
    model: TimeviewAdaptive,
    x_train: torch.Tensor,
    t: torch.Tensor,
    y_train: torch.Tensor,
    n_epochs: int = 1000,
    lr: float = 1e-3,
    n_obs: int = 10,
    weight_decay: float = 1e-5,
    patience: int = 10,
    batch_size: int = 32,
    x_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    verbose: bool = True,
) -> list:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []

    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    x_eval = x_val if x_val is not None else x_train
    y_eval = y_val if y_val is not None else y_train

    n_samples = x_train.shape[0]

    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start : start + batch_size]
            if len(batch_idx) < 2:
                continue
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            optimizer.zero_grad()
            loss = model.loss(x_batch, t, y_batch, n_obs=n_obs)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            val_loss = model.loss(x_eval, t, y_eval, n_obs=n_obs)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

        if patience > 0 and epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return losses


def evaluate_model(
    model: TimeviewAdaptive, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, n_obs: int = 10
) -> dict:
    model.eval()
    with torch.no_grad():
        t_obs = t[:n_obs]
        y_obs = y[:, :n_obs]

        y_mean, y_var, mu_post, Sigma_post = model.update_and_predict(x, t_obs, y_obs, t)

        mse_future = ((y[:, n_obs:] - y_mean[:, n_obs:]) ** 2).mean().item()

        y_std = torch.sqrt(y_var)
        in_interval = torch.abs(y - y_mean) < 2 * y_std
        coverage = in_interval.float().mean().item()

        avg_uncertainty = y_std.mean().item()

    return {"mse_future": mse_future, "coverage_2std": coverage, "avg_uncertainty": avg_uncertainty}


class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def calibrate(
        self,
        model: TimeviewAdaptive,
        x_cal: torch.Tensor,
        t: torch.Tensor,
        y_cal: torch.Tensor,
        n_obs: int = 10,
        n_steps: int = 100,
        target_coverage: float = 0.95,
    ):
        model.eval()
        with torch.no_grad():
            t_obs = t[:n_obs]
            y_obs = y_cal[:, :n_obs]
            y_mean, y_var, _, _ = model.update_and_predict(x_cal, t_obs, y_obs, t)

        z = 1.96

        best_temp = 1.0
        best_loss = float("inf")

        for temp in np.logspace(-1, 2, 50):
            var_scaled = y_var * temp
            y_std = torch.sqrt(var_scaled)
            in_interval = torch.abs(y_cal - y_mean) < z * y_std
            coverage = in_interval.float().mean().item()
            loss = (coverage - target_coverage) ** 2

            if loss < best_loss:
                best_loss = loss
                best_temp = temp

        self.log_temperature.data = torch.log(torch.tensor(best_temp))

        return best_temp

    def apply(self, y_var: torch.Tensor) -> torch.Tensor:
        return y_var * self.temperature


def visualize_results(
    model: TimeviewAdaptive,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    n_obs: int = 10,
    sample_idx: int = 0,
):
    model.eval()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    with torch.no_grad():
        t_obs = t[:n_obs]
        y_obs = y[:, :n_obs]

        mu_0, Sigma_0, bias = model.encode(x)
        y_prior_mean, y_prior_var = model.predict(mu_0, Sigma_0, t, bias)

        y_post_mean, y_post_var, _, _ = model.update_and_predict(x, t_obs, y_obs, t)

        t_np = t.numpy()

        ax = axes[0]
        y_prior_std = torch.sqrt(y_prior_var[sample_idx]).numpy()
        ax.fill_between(
            t_np,
            y_prior_mean[sample_idx].numpy() - 2 * y_prior_std,
            y_prior_mean[sample_idx].numpy() + 2 * y_prior_std,
            alpha=0.3,
            label="Prior +/-2sig",
        )
        ax.plot(t_np, y_prior_mean[sample_idx].numpy(), "b-", label="Prior mean")
        ax.plot(t_np, y[sample_idx].numpy(), "k.", alpha=0.5, label="True trajectory")
        ax.axvline(
            x=t_obs[-1].item(), color="r", linestyle="--", alpha=0.5, label="Observation cutoff"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title("Prior Prediction (from static features only)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        y_post_std = torch.sqrt(y_post_var[sample_idx]).numpy()
        ax.fill_between(
            t_np,
            y_post_mean[sample_idx].numpy() - 2 * y_post_std,
            y_post_mean[sample_idx].numpy() + 2 * y_post_std,
            alpha=0.3,
            color="green",
            label="Posterior +/-2sig",
        )
        ax.plot(t_np, y_post_mean[sample_idx].numpy(), "g-", label="Posterior mean")
        ax.plot(t_np, y[sample_idx].numpy(), "k.", alpha=0.5, label="True trajectory")
        ax.scatter(
            t_obs.numpy(), y_obs[sample_idx].numpy(), c="red", s=50, zorder=5, label="Observations"
        )
        ax.axvline(
            x=t_obs[-1].item(), color="r", linestyle="--", alpha=0.5, label="Observation cutoff"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title("Posterior Prediction (after Bayesian update)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = FIGURES_DIR / "timeview_adaptive_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved to {output_path}")


def demonstrate_sequential_updates(
    model: TimeviewAdaptive, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, sample_idx: int = 0
):
    model.eval()

    observation_counts = [2, 5, 10, 20, 30]
    fig, axes = plt.subplots(1, len(observation_counts), figsize=(4 * len(observation_counts), 4))

    with torch.no_grad():
        for i, n_obs in enumerate(observation_counts):
            ax = axes[i]

            t_obs = t[:n_obs]
            y_obs = y[:, :n_obs]

            y_mean, y_var, _, _ = model.update_and_predict(x, t_obs, y_obs, t)

            t_np = t.numpy()
            y_std = torch.sqrt(y_var[sample_idx]).numpy()

            ax.fill_between(
                t_np,
                y_mean[sample_idx].numpy() - 2 * y_std,
                y_mean[sample_idx].numpy() + 2 * y_std,
                alpha=0.3,
                color="blue",
            )
            ax.plot(t_np, y_mean[sample_idx].numpy(), "b-", linewidth=2)
            ax.plot(t_np, y[sample_idx].numpy(), "k.", alpha=0.3)
            ax.scatter(t_obs.numpy(), y_obs[sample_idx].numpy(), c="red", s=30, zorder=5)
            ax.axvline(x=t_obs[-1].item(), color="r", linestyle="--", alpha=0.5)
            ax.set_xlabel("Time")
            ax.set_title(f"{n_obs} observations")
            ax.set_ylim([y[sample_idx].min().item() - 1, y[sample_idx].max().item() + 1])
            ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Value")
    plt.suptitle("Uncertainty Reduction with More Observations", fontsize=14)
    plt.tight_layout()
    output_path = FIGURES_DIR / "timeview_adaptive_sequential.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    from timeview_adaptive_experiments import load_dataset

    print("=" * 60)
    print("TIMEVIEW-Adaptive: Proof of Concept")
    print("=" * 60)

    n_basis = 9
    noise_std = 0.1
    n_obs_train = 10

    print("\n1. Loading AIRFOIL dataset...")
    x, t, y = load_dataset("airfoil")
    n_samples = x.shape[0]
    n_timepoints = len(t)
    input_dim = x.shape[1]

    n_train = int(0.8 * n_samples)
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print(f"   Loaded {n_samples} samples, {n_timepoints} time points, {input_dim} features")
    print(f"   Training samples: {n_train}")
    print(f"   Test samples: {n_samples - n_train}")
    print(f"   Observations for update: {n_obs_train}")

    print("\n2. Creating TIMEVIEW-Adaptive model...")
    model = TimeviewAdaptive(
        input_dim=input_dim,
        n_basis=n_basis,
        observation_noise=noise_std,
        covariance_type="diagonal",
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Number of basis functions: {n_basis}")
    print(f"   Total parameters: {total_params}")

    print("\n3. Training model...")
    losses = train_model(model, x_train, t, y_train, n_epochs=200, lr=1e-3, n_obs=n_obs_train)

    print("\n4. Evaluating model...")
    train_metrics = evaluate_model(model, x_train, t, y_train, n_obs=n_obs_train)
    test_metrics = evaluate_model(model, x_test, t, y_test, n_obs=n_obs_train)

    print("\n   Train Metrics:")
    print(f"     MSE (future points): {train_metrics['mse_future']:.4f}")
    print(f"     Coverage (2sig): {train_metrics['coverage_2std']:.2%}")
    print(f"     Avg Uncertainty: {train_metrics['avg_uncertainty']:.4f}")

    print("\n   Test Metrics:")
    print(f"     MSE (future points): {test_metrics['mse_future']:.4f}")
    print(f"     Coverage (2sig): {test_metrics['coverage_2std']:.2%}")
    print(f"     Avg Uncertainty: {test_metrics['avg_uncertainty']:.4f}")

    print("\n5. Generating visualizations...")
    visualize_results(model, x_test, t, y_test, n_obs=n_obs_train, sample_idx=0)
    demonstrate_sequential_updates(model, x_test, t, y_test, sample_idx=0)

    print("\n" + "=" * 60)
    print("Done! Check the generated PNG files for visualizations.")
    print("=" * 60)
