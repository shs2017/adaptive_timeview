import sys
from pathlib import Path

### UNCOMMENT when running in a terminal multiplexer ###
# sys.stdout.reconfigure(line_buffering=True)

# import builtins
# _orig_print = builtins.print
# def print(*args, **kwargs):
#     kwargs.setdefault("flush", True)
#     _orig_print(*args, **kwargs)

# sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import load_and_normalize_dataset
from models import TimeviewStatic
from scores import compute_all_metrics
from timeview_adaptive import TimeviewAdaptive, train_model
from train import train_static_model


def eval_static(model: TimeviewStatic, x_test: torch.Tensor, y_test: torch.Tensor, t: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        y_mean, y_var = model(x_test, t)
    return compute_all_metrics(y_test, y_mean, y_var, prefix="")


def eval_prior(model: TimeviewAdaptive, x_test: torch.Tensor, y_test: torch.Tensor, t: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        mu_0, Sigma_0, bias = model.encode(x_test)
        y_mean, y_var = model.predict(mu_0, Sigma_0, t, bias=bias)
    return compute_all_metrics(y_test, y_mean, y_var, prefix="")


def eval_posterior(
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
        full_metrics = compute_all_metrics(y_test, y_mean, y_var, prefix="full_") # evaluation on all points

        if n_obs < y_test.shape[1]:
            future_metrics = compute_all_metrics(
                y_test[:, n_obs:], y_mean[:, n_obs:], y_var[:, n_obs:], prefix="future_"
            ) # evaluation on unobserved points
        else:
            future_metrics = {}
    return {**full_metrics, **future_metrics}


def run_prior_vs_static(
    dataset_name: str,
    seed: int = 0,
    n_epochs: int = 1000,
    training_n_obs_values: list[int] | None = None,
    eval_n_obs_values: list[int] | None = None,
    kl_weight: float = 0.01,
) -> dict:
    if training_n_obs_values is None:
        training_n_obs_values = [2, 5, 10, 15, 20, 30]
    if eval_n_obs_values is None:
        eval_n_obs_values = [1, 2, 5, 10, 15, 20, 30]

    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'=' * 60}")

    (
        x_tr, x_va, x_te, t_ds, y_tr, y_va, y_te, y_norm, knots_ds, n_basis_ds,
        ts_tr, ys_tr, ts_va, ys_va, ts_te, ys_te,
    ) = load_and_normalize_dataset(dataset_name, seed=seed)

    n_time = len(t_ds)
    print(f"  Samples: train={len(x_tr)}, val={len(x_va)}, test={len(x_te)}")
    print(f"  Time points: {n_time}, n_basis: {n_basis_ds}")

    eval_n_obs_values = [n for n in eval_n_obs_values if n < n_time]
    training_n_obs_values = [n for n in training_n_obs_values if n < n_time]

    print("\n  Training static model...")
    torch.manual_seed(seed)
    static_model = TimeviewStatic(
        input_dim=x_tr.shape[1], n_basis=n_basis_ds, knots=knots_ds
    )
    train_static_model(
        static_model, x_tr, t_ds, y_tr,
        x_val=x_va, y_val=y_va,
        ts_train=ts_tr, ys_train=ys_tr,
        ts_val=ts_va, ys_val=ys_va,
    )
    static_metrics = eval_static(static_model, x_te, y_te, t_ds)
    print(f"  Static model MSE: {static_metrics['mse']:.5f}, CRPS: {static_metrics['crps']:.5f}")

    results = {
        "static": static_metrics,
        "adaptive_models": {},
    }

    for train_n_obs in training_n_obs_values:
        print(f"\n  Training adaptive model (train_n_obs={train_n_obs}, kl_weight={kl_weight})...")
        torch.manual_seed(seed)
        adaptive_model = TimeviewAdaptive(
            input_dim=x_tr.shape[1],
            n_basis=n_basis_ds,
            observation_noise=0.1,
            covariance_type="diagonal",
            use_batchnorm=True,
            learn_noise=True,
            kl_weight=kl_weight,
            knots=knots_ds,
        )
        train_model(
            adaptive_model, x_tr, t_ds, y_tr,
            n_epochs=n_epochs, n_obs=train_n_obs, verbose=False,
        )

        prior_metrics = eval_prior(adaptive_model, x_te, y_te, t_ds)
        print(f"    Prior MSE: {prior_metrics['mse']:.5f} "
              f"(static: {static_metrics['mse']:.5f}, "
              f"ratio: {prior_metrics['mse'] / static_metrics['mse']:.3f}x)")

        posterior_results = {}
        for eval_n_obs in eval_n_obs_values:
            post_metrics = eval_posterior(adaptive_model, x_te, y_te, t_ds, eval_n_obs)
            posterior_results[eval_n_obs] = post_metrics

        results["adaptive_models"][train_n_obs] = {
            "prior": prior_metrics,
            "posterior": posterior_results,
        }

    return results


def print_summary_table(results: dict, dataset_name: str):
    static_mse = results["static"]["mse"]
    static_crps = results["static"]["crps"]

    print(f"\n{'=' * 80}")
    print(f"SUMMARY: {dataset_name.upper()}")
    print(f"{'=' * 80}")
    print(f"Static model: MSE={static_mse:.5f}, CRPS={static_crps:.5f}")
    print()

    print("Prior (0 observations) vs Static:")
    print(f"  {'train_n_obs':>12} {'Prior MSE':>12} {'vs Static':>12} {'Prior CRPS':>12} {'vs Static':>12}")
    print(f"  {'-' * 64}")
    for train_n_obs, model_res in results["adaptive_models"].items():
        prior = model_res["prior"]
        ratio_mse = prior["mse"] / static_mse
        ratio_crps = prior["crps"] / static_crps
        print(f"  {train_n_obs:>12} {prior['mse']:>12.5f} {ratio_mse:>11.3f}x {prior['crps']:>12.5f} {ratio_crps:>11.3f}x")

    print()

    first_model = next(iter(results["adaptive_models"].values()))
    available_eval_nobs = sorted(first_model["posterior"].keys())
    sample_nobs = [n for n in [5, 10, 20] if n in available_eval_nobs]
    if not sample_nobs:
        sample_nobs = available_eval_nobs[:3]

    for eval_n in sample_nobs:
        print(f"Posterior (eval_n_obs={eval_n}) future MSE vs Static:")
        print(f"  {'train_n_obs':>12} {'Post MSE':>12} {'vs Static':>12} {'Post CRPS':>12}")
        print(f"  {'-' * 52}")
        for train_n_obs, model_res in results["adaptive_models"].items():
            if eval_n in model_res["posterior"]:
                post = model_res["posterior"][eval_n]
                future_mse = post.get("future_mse", post.get("full_mse", float("nan")))
                future_crps = post.get("future_crps", post.get("full_crps", float("nan")))
                ratio = future_mse / static_mse
                print(f"  {train_n_obs:>12} {future_mse:>12.5f} {ratio:>11.3f}x {future_crps:>12.5f}")
        print()


def plot_prior_vs_static(all_dataset_results: dict, output_path: Path | None = None):
    datasets = list(all_dataset_results.keys())
    n_datasets = len(datasets)

    fig, axes = plt.subplots(2, n_datasets, figsize=(5 * n_datasets, 10))
    if n_datasets == 1:
        axes = axes.reshape(-1, 1)

    for col, ds_name in enumerate(datasets):
        res = all_dataset_results[ds_name]
        static_mse = res["static"]["mse"]
        adaptive = res["adaptive_models"]

        train_nobs = sorted(adaptive.keys())

        ax0 = axes[0, col]
        prior_mses = [adaptive[n]["prior"]["mse"] for n in train_nobs]
        prior_crps = [adaptive[n]["prior"]["crps"] for n in train_nobs]

        ax0.plot(train_nobs, prior_mses, "o-", color="#2196F3", linewidth=2, markersize=7, label="Prior MSE")
        ax0.axhline(y=static_mse, color="#E53935", linestyle="--", linewidth=2, label=f"Static MSE ({static_mse:.4f})")
        ax0.set_xlabel("Training n_obs")
        ax0.set_ylabel("MSE")
        ax0.set_title(f"{ds_name.replace('_', '-').title()}\nPrior vs Static (full trajectory)")
        ax0.legend(fontsize=8)
        ax0.grid(True, alpha=0.3)
        ax0.set_xticks(train_nobs)
        ax0.set_ylim(bottom=0)

        ax1 = axes[1, col]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(train_nobs)))

        for i, train_n in enumerate(train_nobs):
            post_res = adaptive[train_n]["posterior"]
            eval_ns = sorted(post_res.keys())
            future_mses = [post_res[n].get("future_mse", float("nan")) for n in eval_ns]
            ax1.plot(eval_ns, future_mses, "o-", color=colors[i], linewidth=1.5,
                     markersize=5, label=f"train_n={train_n}")

        ax1.axhline(y=static_mse, color="#E53935", linestyle="--", linewidth=2,
                    label=f"Static ({static_mse:.4f})")
        ax1.set_xlabel("Eval n_obs")
        ax1.set_ylabel("Future MSE")
        ax1.set_title(f"{ds_name.replace('_', '-').title()}\nPosterior Future MSE vs n_obs")
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)

    plt.suptitle("Prior/Posterior vs Static: Effect of Training n_obs", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if output_path is None:
        output_path = Path(__file__).parent.parent / "figures" / "prior_vs_static_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot to: {output_path}")


def plot_prior_ratio(all_dataset_results: dict, output_path: Path | None = None):
    datasets = list(all_dataset_results.keys())
    n_datasets = len(datasets)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    markers = ["o", "s", "^", "D"]

    for i, ds_name in enumerate(datasets):
        res = all_dataset_results[ds_name]
        static_mse = res["static"]["mse"]
        adaptive = res["adaptive_models"]

        train_nobs = sorted(adaptive.keys())
        ratios = [adaptive[n]["prior"]["mse"] / static_mse for n in train_nobs]

        label = ds_name.replace("_", "-").title()
        ax.plot(train_nobs, ratios, f"{markers[i % len(markers)]}-",
                color=colors[i % len(colors)], linewidth=2, markersize=7, label=label)

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, label="Static baseline (ratio=1)")
    ax.set_xlabel("Training n_obs", fontsize=12)
    ax.set_ylabel("Prior MSE / Static MSE", fontsize=12)
    ax.set_title("Prior Model Quality Relative to Static\n(ratio < 1 = prior is better)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is None:
        output_path = Path(__file__).parent.parent / "figures" / "prior_ratio_vs_static.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {output_path}")


def print_kl_comparison(results_with_kl: dict, results_no_kl: dict):
    """Side-by-side comparison of prior MSE ratios with and without KL."""
    all_datasets = list(results_with_kl.keys())
    all_train_nobs = sorted({
        n for res in results_with_kl.values()
        for n in res["adaptive_models"]
    })

    for ds_name in all_datasets:
        static_mse = results_with_kl[ds_name]["static"]["mse"]
        print(f"\n{ds_name.upper()} — Prior MSE / Static MSE (static MSE={static_mse:.5f})")
        print(f"  {'train_n_obs':>12} {'kl=0.01':>12} {'kl=0.00':>12} {'diff':>10}")
        print(f"  {'-' * 50}")
        for train_n in all_train_nobs:
            if train_n not in results_with_kl[ds_name]["adaptive_models"]:
                continue
            r_kl = results_with_kl[ds_name]["adaptive_models"][train_n]["prior"]["mse"] / static_mse
            r_no = results_no_kl[ds_name]["adaptive_models"][train_n]["prior"]["mse"] / static_mse
            diff = r_no - r_kl
            print(f"  {train_n:>12} {r_kl:>11.3f}x {r_no:>11.3f}x {diff:>+10.3f}")


def main():
    seed = 0
    n_epochs = 1000
    datasets = ["airfoil", "flchain", "stress_strain"]
    training_n_obs_values = [2, 5, 10, 15, 20, 30]
    eval_n_obs_values = [1, 2, 5, 10, 15, 20, 30]

    # Run with kl_weight=0.01 (default)
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: kl_weight=0.01 (default)")
    print("=" * 80)
    results_with_kl = {}
    for dataset_name in datasets:
        try:
            results = run_prior_vs_static(
                dataset_name, seed=seed, n_epochs=n_epochs,
                training_n_obs_values=training_n_obs_values,
                eval_n_obs_values=eval_n_obs_values, kl_weight=0.01,
            )
            results_with_kl[dataset_name] = results
            print_summary_table(results, dataset_name)
        except FileNotFoundError as e:
            print(f"\nSkipping {dataset_name}: {e}")

    # Run with kl_weight=0.0
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: kl_weight=0.0 (no KL regularization)")
    print("=" * 80)
    results_no_kl = {}
    for dataset_name in datasets:
        try:
            results = run_prior_vs_static(
                dataset_name, seed=seed, n_epochs=n_epochs,
                training_n_obs_values=training_n_obs_values,
                eval_n_obs_values=eval_n_obs_values, kl_weight=0.0,
            )
            results_no_kl[dataset_name] = results
            print_summary_table(results, dataset_name)
        except FileNotFoundError as e:
            print(f"\nSkipping {dataset_name}: {e}")

    # Side-by-side comparison
    if results_with_kl and results_no_kl:
        print("\n" + "=" * 80)
        print("KL ABLATION: Prior MSE ratio with vs without KL regularization")
        print("(Does removing KL reduce prior degradation with large n_obs?)")
        print("=" * 80)
        print_kl_comparison(results_with_kl, results_no_kl)

        plot_prior_vs_static(results_with_kl,
            output_path=Path(__file__).parent.parent / "figures" / "prior_vs_static_kl001.png")
        plot_prior_vs_static(results_no_kl,
            output_path=Path(__file__).parent.parent / "figures" / "prior_vs_static_kl000.png")
        plot_prior_ratio(results_with_kl,
            output_path=Path(__file__).parent.parent / "figures" / "prior_ratio_kl001.png")
        plot_prior_ratio(results_no_kl,
            output_path=Path(__file__).parent.parent / "figures" / "prior_ratio_kl000.png")


if __name__ == "__main__":
    main()
