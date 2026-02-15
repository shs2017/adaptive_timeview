import torch

from scores import (
    compute_all_metrics,
    crps_gaussian,
)
from timeview_adaptive import (
    TimeviewAdaptive,
)


def eval_model_full(
    model: TimeviewAdaptive,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    t: torch.Tensor,
    n_obs: int,
) -> dict:
    model.eval()
    with torch.no_grad():
        t_obs = t[:n_obs]
        y_obs = y_test[:, :n_obs]
        y_mean, y_var, _, _ = model.update_and_predict(x_test, t_obs, y_obs, t)
        metrics = compute_all_metrics(
            y_test[:, n_obs:], y_mean[:, n_obs:], y_var[:, n_obs:], prefix="future_"
        )
    return metrics


def streaming_evaluation(
    model: TimeviewAdaptive, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, sample_idx: int = 0
) -> dict[str, list]:
    model.eval()
    n_timepoints = len(t)

    results = {
        "n_obs": [],
        "mse_future": [],
        "crps_future": [],
        "avg_uncertainty": [],
        "coverage_95": [],
    }

    with torch.no_grad():
        mu_0, Sigma_0, bias = model.encode(x)
        y_prior_mean, y_prior_var = model.predict(mu_0, Sigma_0, t, bias=bias)

        results["n_obs"].append(0)
        results["mse_future"].append(((y - y_prior_mean) ** 2).mean().item())
        results["crps_future"].append(
            crps_gaussian(y, y_prior_mean, torch.sqrt(y_prior_var)).mean().item()
        )
        results["avg_uncertainty"].append(torch.sqrt(y_prior_var).mean().item())
        z_95 = 1.96
        coverage = (
            (torch.abs(y - y_prior_mean) < z_95 * torch.sqrt(y_prior_var)).float().mean().item()
        )
        results["coverage_95"].append(coverage)

        for n_obs in range(1, n_timepoints):
            t_obs = t[:n_obs]
            y_obs = y[:, :n_obs]

            y_mean, y_var, _, _ = model.update_and_predict(x, t_obs, y_obs, t)

            y_future = y[:, n_obs:]
            y_mean_future = y_mean[:, n_obs:]
            y_var_future = y_var[:, n_obs:]

            if y_future.shape[1] > 0:
                results["n_obs"].append(n_obs)
                results["mse_future"].append(((y_future - y_mean_future) ** 2).mean().item())
                results["crps_future"].append(
                    crps_gaussian(y_future, y_mean_future, torch.sqrt(y_var_future)).mean().item()
                )
                results["avg_uncertainty"].append(torch.sqrt(y_var_future).mean().item())
                coverage = (
                    (torch.abs(y_future - y_mean_future) < z_95 * torch.sqrt(y_var_future))
                    .float()
                    .mean()
                    .item()
                )
                results["coverage_95"].append(coverage)

    return results
