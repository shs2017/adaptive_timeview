import numpy as np
import torch

from timeview_adaptive import (
    TimeviewAdaptive,
)


class TransitionPointAnalyzer:

    def __init__(self, model: TimeviewAdaptive, n_samples: int = 1000):
        self.model = model
        self.n_samples = n_samples

    def sample_trajectories(
        self, mu: torch.Tensor, Sigma: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        batch_size = mu.shape[0]
        n_basis = mu.shape[1]

        L = torch.linalg.cholesky(Sigma + 1e-6 * torch.eye(n_basis, device=Sigma.device))
        eps = torch.randn(batch_size, self.n_samples, n_basis, device=mu.device)
        c_samples = mu.unsqueeze(1) + torch.einsum("bij,bsj->bsi", L, eps)

        Phi = self.model.get_basis(t)
        y_samples = torch.einsum("bsc,tc->bst", c_samples, Phi)

        return y_samples

    def compute_derivative(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dt = t[1] - t[0]
        dy = (y[..., 1:] - y[..., :-1]) / dt
        return dy

    def find_transition_points(
        self, y_samples: torch.Tensor, t: torch.Tensor, threshold: float = 0.5
    ) -> dict:
        dy = self.compute_derivative(y_samples, t)

        signs = torch.sign(dy)
        sign_changes = (signs[..., 1:] != signs[..., :-1]).float()

        t_mid = (t[:-1] + t[1:]) / 2
        t_transitions = t_mid[:-1]

        batch_size = y_samples.shape[0]
        results = []

        for b in range(batch_size):
            sample_transitions = []
            for s in range(self.n_samples):
                change_idx = torch.where(sign_changes[b, s] > 0.5)[0]
                if len(change_idx) > 0:
                    sample_transitions.extend(t_transitions[change_idx].tolist())

            if sample_transitions:
                transitions = np.array(sample_transitions)
                results.append(
                    {
                        "mean": np.mean(transitions),
                        "std": np.std(transitions),
                        "ci_lower": np.percentile(transitions, 2.5),
                        "ci_upper": np.percentile(transitions, 97.5),
                        "count": len(transitions) / self.n_samples,
                    }
                )
            else:
                results.append(
                    {"mean": None, "std": None, "ci_lower": None, "ci_upper": None, "count": 0}
                )

        return results

    def analyze(
        self, x: torch.Tensor, t_obs: torch.Tensor, y_obs: torch.Tensor, t: torch.Tensor
    ) -> dict:
        self.model.eval()

        with torch.no_grad():
            y_mean, y_var, mu_post, Sigma_post = self.model.update_and_predict(x, t_obs, y_obs, t)

            y_samples = self.sample_trajectories(mu_post, Sigma_post, t)
            transitions = self.find_transition_points(y_samples, t)
            coef_std = torch.sqrt(torch.diagonal(Sigma_post, dim1=-2, dim2=-1))

        return {
            "y_mean": y_mean,
            "y_var": y_var,
            "y_samples": y_samples,
            "transitions": transitions,
            "coefficient_mean": mu_post,
            "coefficient_std": coef_std,
        }
