import numpy as np
import torch
from scipy import stats


def crps_gaussian(y_true: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    CRPS(F, y) = sigma * [z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
    """
    z = (y_true - mu) / sigma

    Phi_z = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
    phi_z = torch.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)

    crps = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1 / np.sqrt(np.pi))
    return crps


def interval_score(
    y_true: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, alpha: float = 0.05
) -> torch.Tensor:
    """
    interval score = (upper - lower) + (2/alpha) * (lower - y) * I(y < lower) + (2/alpha) * (y - upper) * I(y > upper)
    """
    width = upper - lower

    below = torch.clamp(lower - y_true, min=0)
    above = torch.clamp(y_true - upper, min=0)

    score = width + (2 / alpha) * below + (2 / alpha) * above
    return score


def compute_calibration(
    y_true: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, n_bins: int = 10
) -> dict[str, np.ndarray]:
    confidence_levels = np.linspace(0.1, 0.99, n_bins)
    observed_coverage = []

    y_true_np = y_true.detach().cpu().numpy().flatten()
    mu_np = mu.detach().cpu().numpy().flatten()
    sigma_np = sigma.detach().cpu().numpy().flatten()

    for conf in confidence_levels:
        alpha = 1 - conf
        z = stats.norm.ppf(1 - alpha / 2)

        lower = mu_np - z * sigma_np
        upper = mu_np + z * sigma_np

        in_interval = (y_true_np >= lower) & (y_true_np <= upper)
        observed_coverage.append(in_interval.mean())

    return {
        "expected": confidence_levels,
        "observed": np.array(observed_coverage),
        "calibration_error": np.mean(np.abs(confidence_levels - np.array(observed_coverage))),
    }


def compute_all_metrics(
    y_true: torch.Tensor, y_mean: torch.Tensor, y_var: torch.Tensor, prefix: str = ""
) -> dict[str, float]:
    y_std = torch.sqrt(y_var)

    mse = ((y_true - y_mean) ** 2).mean().item()
    mae = torch.abs(y_true - y_mean).mean().item()
    crps = crps_gaussian(y_true, y_mean, y_std).mean().item()
    nll = (0.5 * torch.log(2 * np.pi * y_var) + 0.5 * (y_true - y_mean) ** 2 / y_var).mean().item()

    z_90 = 1.645
    z_95 = 1.96

    is_90 = (
        interval_score(y_true, y_mean - z_90 * y_std, y_mean + z_90 * y_std, alpha=0.10)
        .mean()
        .item()
    )
    is_95 = (
        interval_score(y_true, y_mean - z_95 * y_std, y_mean + z_95 * y_std, alpha=0.05)
        .mean()
        .item()
    )

    coverage_90 = (torch.abs(y_true - y_mean) < z_90 * y_std).float().mean().item()
    coverage_95 = (torch.abs(y_true - y_mean) < z_95 * y_std).float().mean().item()

    calibration = compute_calibration(y_true, y_mean, y_std)
    sharpness_90 = (2 * z_90 * y_std).mean().item()
    sharpness_95 = (2 * z_95 * y_std).mean().item()
    winkler_95_normalized = is_95 / (y_true.std().item() + 1e-8)
    pe = 1.0 / (mse * sharpness_95 + 1e-8)

    metrics = {
        f"{prefix}mse": mse,
        f"{prefix}mae": mae,
        f"{prefix}crps": crps,
        f"{prefix}nll": nll,
        f"{prefix}interval_score_90": is_90,
        f"{prefix}interval_score_95": is_95,
        f"{prefix}coverage_90": coverage_90,
        f"{prefix}coverage_95": coverage_95,
        f"{prefix}calibration_error": calibration["calibration_error"],
        f"{prefix}avg_std": y_std.mean().item(),
        f"{prefix}sharpness_90": sharpness_90,
        f"{prefix}sharpness_95": sharpness_95,
        f"{prefix}winkler_95_norm": winkler_95_normalized,
        f"{prefix}predictive_efficiency": pe,
    }

    return metrics
