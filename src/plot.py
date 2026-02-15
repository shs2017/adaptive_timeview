
import matplotlib.pyplot as plt
import numpy as np
import torch

from constant import FIGURES_DIR
from scores import (
    compute_calibration,
)


def plot_transition_analysis(
    analysis: dict, t: torch.Tensor, y_true: torch.Tensor, t_obs: torch.Tensor, sample_idx: int = 0
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t_np = t.numpy()

    ax = axes[0, 0]
    y_samples = analysis["y_samples"][sample_idx].numpy()
    for i in range(min(50, y_samples.shape[0])):
        ax.plot(t_np, y_samples[i], "b-", alpha=0.1)
    ax.plot(t_np, analysis["y_mean"][sample_idx].numpy(), "r-", linewidth=2, label="Posterior mean")
    ax.plot(t_np, y_true[sample_idx].numpy(), "k.", alpha=0.5, label="True")
    ax.axvline(x=t_obs[-1].item(), color="g", linestyle="--", alpha=0.7, label="Obs cutoff")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Trajectory Samples from Posterior")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    coef_mean = analysis["coefficient_mean"][sample_idx].numpy()
    coef_std = analysis["coefficient_std"][sample_idx].numpy()
    x_pos = np.arange(len(coef_mean))
    ax.bar(x_pos, coef_mean, yerr=2 * coef_std, capsize=3, alpha=0.7)
    ax.set_xlabel("Basis function index")
    ax.set_ylabel("Coefficient value")
    ax.set_title("Posterior Coefficient Distribution (±2σ)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    dy_samples = np.diff(y_samples, axis=1) / (t_np[1] - t_np[0])
    t_deriv = (t_np[:-1] + t_np[1:]) / 2

    dy_mean = dy_samples.mean(axis=0)
    dy_std = dy_samples.std(axis=0)
    ax.fill_between(t_deriv, dy_mean - 2 * dy_std, dy_mean + 2 * dy_std, alpha=0.3)
    ax.plot(t_deriv, dy_mean, "b-", linewidth=2)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.axvline(x=t_obs[-1].item(), color="g", linestyle="--", alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Derivative")
    ax.set_title("Trajectory Derivative with Uncertainty")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    trans = analysis["transitions"][sample_idx]
    if trans["mean"] is not None:
        dy = np.diff(y_samples, axis=1)
        signs = np.sign(dy)
        sign_changes = signs[:, 1:] != signs[:, :-1]

        all_transitions = []
        for s in range(y_samples.shape[0]):
            idx = np.where(sign_changes[s])[0]
            all_transitions.extend(t_deriv[idx])

        if all_transitions:
            ax.hist(all_transitions, bins=30, density=True, alpha=0.7)
            ax.axvline(x=trans["mean"], color="r", linewidth=2, label=f"Mean: {trans['mean']:.3f}")
            ax.axvline(x=trans["ci_lower"], color="r", linestyle="--", label="95% CI")
            ax.axvline(x=trans["ci_upper"], color="r", linestyle="--")

    ax.set_xlabel("Time")
    ax.set_ylabel("Density")
    ax.set_title("Transition Point Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = FIGURES_DIR / "transition_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved to {output_path}")


def plot_streaming_results(results: dict[str, list], title: str = "Streaming Evaluation"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    n_obs = results["n_obs"]

    ax = axes[0, 0]
    ax.plot(n_obs, results["mse_future"], "b-o", markersize=3)
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("MSE (future points)")
    ax.set_title("Prediction Error vs Observations")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    ax = axes[0, 1]
    ax.plot(n_obs, results["crps_future"], "g-o", markersize=3)
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("CRPS (future points)")
    ax.set_title("CRPS vs Observations")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(n_obs, results["avg_uncertainty"], "r-o", markersize=3)
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("Average std")
    ax.set_title("Uncertainty vs Observations")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(n_obs, results["coverage_95"], "m-o", markersize=3)
    ax.axhline(y=0.95, color="k", linestyle="--", label="Target (95%)")
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("95% Coverage")
    ax.set_title("Calibration vs Observations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    output_path = FIGURES_DIR / "streaming_evaluation.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved to {output_path}")


def plot_ablation_with_error_bars(
    results: dict,
    x_key: str,
    y_metric: str,
    ax: plt.Axes,
    color: str = "blue",
    label: str = None,
):
    x_vals = list(results.keys())
    y_means = [results[k][y_metric]["mean"] for k in x_vals]
    y_stds = [results[k][y_metric]["std"] for k in x_vals]

    ax.errorbar(
        x_vals,
        y_means,
        yerr=y_stds,
        fmt="o-",
        color=color,
        capsize=4,
        capthick=1.5,
        label=label,
        markersize=6,
    )
    ax.fill_between(
        x_vals,
        np.array(y_means) - np.array(y_stds),
        np.array(y_means) + np.array(y_stds),
        alpha=0.2,
        color=color,
    )


def plot_bar_with_error_bars(
    results: dict,
    y_metric: str,
    ax: plt.Axes,
    colors: list[str] = None,
):
    x_vals = list(results.keys())
    y_means = [results[k][y_metric]["mean"] for k in x_vals]
    y_stds = [results[k][y_metric]["std"] for k in x_vals]

    if colors is None:
        colors = ["blue", "green", "red", "orange", "purple"][: len(x_vals)]

    bars = ax.bar(x_vals, y_means, yerr=y_stds, capsize=4, color=colors, alpha=0.8)
    return bars


def plot_active_scheduling(results: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    steps = results["n_steps"]

    ax = axes[0]
    ax.plot(steps, results["active_mse"], "b-o", markersize=4, label="Active (IG)")
    ax.plot(steps, results["uniform_mse"], "r--s", markersize=4, label="Uniform")
    ax.set_xlabel("Additional observations")
    ax.set_ylabel("MSE (remaining points)")
    ax.set_title("Prediction Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, results["active_coverage"], "b-o", markersize=4, label="Active (IG)")
    ax.plot(steps, results["uniform_coverage"], "r--s", markersize=4, label="Uniform")
    ax.axhline(y=0.95, color="k", linestyle="--", alpha=0.5, label="Target")
    ax.set_xlabel("Additional observations")
    ax.set_ylabel("95% Coverage")
    ax.set_title("Calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    ax = axes[2]
    ax.plot(steps, results["active_crps"], "b-o", markersize=4, label="Active (IG)")
    ax.plot(steps, results["uniform_crps"], "r--s", markersize=4, label="Uniform")
    ax.set_xlabel("Additional observations")
    ax.set_ylabel("CRPS")
    ax.set_title("Probabilistic Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Active vs Uniform Observation Scheduling", fontsize=13)
    plt.tight_layout()
    output_path = FIGURES_DIR / "active_scheduling.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def plot_gp_comparison(results: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    methods = list(results.keys())
    method_labels = {"gp_baseline": "GP", "timeview_adaptive": "TIMEVIEW-Adaptive", "static_baseline": "Static"}
    colors = {"gp_baseline": "orange", "timeview_adaptive": "blue", "static_baseline": "gray"}

    ax = axes[0]
    ax.set_ylabel("MSE")
    ax.set_title("Prediction Error")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    cov_vals = [results[m]["future_coverage_95"] for m in methods]
    ax.bar(
        [method_labels[m] for m in methods], cov_vals,
        color=[colors[m] for m in methods], alpha=0.8,
    )
    ax.axhline(y=0.95, color="r", linestyle="--", label="Target")
    ax.set_ylabel("95% Coverage")
    ax.set_title("Calibration")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    sharp_vals = [results[m]["future_sharpness_95"] for m in methods]
    ax.bar(
        [method_labels[m] for m in methods], sharp_vals,
        color=[colors[m] for m in methods], alpha=0.8,
    )
    ax.set_ylabel("Interval Width (95%)")
    ax.set_title("Sharpness (lower is better)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Method Comparison: GP vs TIMEVIEW-Adaptive vs Static", fontsize=13)
    plt.tight_layout()
    output_path = FIGURES_DIR / "gp_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def plot_heteroscedastic_comparison(results: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    methods = list(results.keys())
    colors_map = {"homoscedastic": "steelblue", "heteroscedastic": "green", "gated": "purple"}

    ax = axes[0]
    mse_vals = [results[m]["future_mse"] for m in methods]
    ax.bar(methods, mse_vals, color=[colors_map[m] for m in methods], alpha=0.8)
    ax.set_ylabel("MSE")
    ax.set_title("Prediction Error")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    cov_vals = [results[m]["future_coverage_95"] for m in methods]
    ax.bar(methods, cov_vals, color=[colors_map[m] for m in methods], alpha=0.8)
    ax.axhline(y=0.95, color="r", linestyle="--", label="Target")
    ax.set_ylabel("95% Coverage")
    ax.set_title("Calibration")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    cal_vals = [results[m]["future_calibration_error"] for m in methods]
    ax.bar(methods, cal_vals, color=[colors_map[m] for m in methods], alpha=0.8)
    ax.set_ylabel("Calibration Error")
    ax.set_title("Calibration Error (lower is better)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Noise Model Comparison", fontsize=13)
    plt.tight_layout()
    output_path = FIGURES_DIR / "heteroscedastic_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def plot_aec(aec_results: dict, label: str = "Model", ax: plt.Axes | None = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(aec_results["n_obs"], aec_results["aec"], "o-", markersize=4, label=label)
    ax.set_xlabel("Number of observations (k)")
    ax.set_ylabel("AEC(k) = (MSE_prior - MSE_k) / k")
    ax.set_title("Adaptation Efficiency Curve")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_uncertainty_decomposition(
    decomp: dict,
    sample_idx: int = 0,
    t_obs: torch.Tensor | None = None,
    ax: plt.Axes | None = None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    t_np = decomp["t"].numpy()
    noise = decomp["noise"][sample_idx].numpy()
    post_ep = decomp["posterior_epistemic"][sample_idx].numpy()
    info_g = decomp["info_gained"][sample_idx].numpy()

    ax.fill_between(t_np, 0, noise, alpha=0.4, color="gray", label="Irreducible noise (σ²)")
    ax.fill_between(t_np, noise, noise + post_ep, alpha=0.4, color="steelblue",
                    label="Remaining epistemic uncertainty")
    ax.fill_between(t_np, noise + post_ep, noise + post_ep + info_g, alpha=0.3,
                    color="green", label="Information gained from observations")

    ax.plot(t_np, decomp["total_var"][sample_idx].numpy(), "k-", linewidth=2,
            label="Total posterior variance")
    ax.plot(t_np, decomp["prior_epistemic"][sample_idx].numpy() + noise, "r--",
            linewidth=1.5, alpha=0.7, label="Prior total variance")

    if t_obs is not None:
        for to in t_obs.numpy():
            ax.axvline(x=to, color="orange", alpha=0.2, linewidth=1)
        ax.axvline(x=t_obs[-1].item(), color="orange", alpha=0.5, linewidth=2,
                   label=f"Observations (n={len(t_obs)})")

    ax.set_xlabel("Time")
    ax.set_ylabel("Variance")
    ax.set_title("Structured Uncertainty Decomposition")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    return ax


def plot_best_config_comparison(results: dict):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    methods = list(results.keys())
    colors_map = {
        "standard": "steelblue", "gated": "purple",
        "heteroscedastic": "green", "best_config": "orange",
    }

    ax = axes[0]
    vals = [results[m]["future_mse"] for m in methods]
    ax.bar(methods, vals, color=[colors_map.get(m, "gray") for m in methods], alpha=0.8)
    ax.set_ylabel("MSE")
    ax.set_title("Prediction Error")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticklabels(methods, rotation=30, ha="right")

    ax = axes[1]
    vals = [results[m]["future_calibration_error"] for m in methods]
    ax.bar(methods, vals, color=[colors_map.get(m, "gray") for m in methods], alpha=0.8)
    ax.set_ylabel("Calibration Error")
    ax.set_title("Calibration (lower is better)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticklabels(methods, rotation=30, ha="right")

    ax = axes[2]
    vals = [results[m]["future_predictive_efficiency"] for m in methods]
    ax.bar(methods, vals, color=[colors_map.get(m, "gray") for m in methods], alpha=0.8)
    ax.set_ylabel("PE = 1 / (MSE × Sharpness)")
    ax.set_title("Predictive Efficiency (higher is better)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticklabels(methods, rotation=30, ha="right")

    ax = axes[3]
    vals = [results[m]["future_crps"] for m in methods]
    ax.bar(methods, vals, color=[colors_map.get(m, "gray") for m in methods], alpha=0.8)
    ax.set_ylabel("CRPS")
    ax.set_title("CRPS (lower is better)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticklabels(methods, rotation=30, ha="right")

    plt.suptitle("Best Configuration vs Model Variants", fontsize=13)
    plt.tight_layout()
    output_path = FIGURES_DIR / "best_config_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def plot_aec_comparison(aec_results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors_map = {
        "standard": "steelblue", "gated": "purple", "best_config": "orange",
    }

    ax = axes[0]
    for name, aec in aec_results.items():
        ax.plot(aec["n_obs"], aec["aec"], "o-", markersize=3,
                color=colors_map.get(name, "gray"),
                label=f"{name} (AAEC={aec['aaec']:.4f})")
    ax.set_xlabel("Number of observations (k)")
    ax.set_ylabel("AEC(k) = (MSE_prior - MSE_k) / k")
    ax.set_title("Adaptation Efficiency Curve")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, aec in aec_results.items():
        ax.plot(aec["n_obs"], aec["mse"], "o-", markersize=3,
                color=colors_map.get(name, "gray"), label=name)
        ax.axhline(y=aec["mse_prior"], color=colors_map.get(name, "gray"),
                   linestyle="--", alpha=0.3)
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("MSE (all points)")
    ax.set_title("MSE Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Adaptation Efficiency: How Fast Does Each Model Learn?", fontsize=13)
    plt.tight_layout()
    output_path = FIGURES_DIR / "aec_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def plot_uncertainty_decomposition_grid(decomp_results: dict, t: torch.Tensor):
    n_panels = len(decomp_results)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    for i, (n_obs, res) in enumerate(decomp_results.items()):
        ax = axes[i]
        t_obs = t[:n_obs]
        plot_uncertainty_decomposition(res["decomp"], sample_idx=0, t_obs=t_obs, ax=ax)
        pct = res["info_fraction"] * 100
        ax.set_title(f"n_obs={n_obs}\nInfo gained: {pct:.0f}% of prior")

    plt.suptitle("Uncertainty Decomposition vs Number of Observations", fontsize=13)
    plt.tight_layout()
    output_path = FIGURES_DIR / "uncertainty_decomposition.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def plot_calibration(
    y_true: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    ax: plt.Axes | None = None,
    label: str = "Model",
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    calibration = compute_calibration(y_true, y_mean, y_std)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(calibration["expected"], calibration["observed"], "o-", label=label)
    ax.set_xlabel("Expected coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title(f"Calibration (Error: {calibration['calibration_error']:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return ax
