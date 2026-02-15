from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from constant import FIGURES_DIR
from data import load_and_normalize_dataset
from models import TimeviewAdaptiveFullCholesky, TimeviewStatic
from scores import compute_all_metrics
from timeview_adaptive import (
    TimeviewAdaptive,
    TimeviewAdaptiveGated,
    create_knots,
    train_model,
)
from train import train_static_model


@dataclass
class AblationConfig:
    n_basis_values: list[int] = None
    n_obs_values: list[int] = None
    noise_values: list[float] = None
    covariance_types: list[str] = None
    kl_weight_values: list[float] = None
    dropout_values: list[float] = None

    def __post_init__(self):
        self.n_basis_values = self.n_basis_values or [5, 7, 9, 11]
        self.n_obs_values = self.n_obs_values or [2, 5, 10, 15, 20, 30]
        self.noise_values = self.noise_values or [0.05, 0.1, 0.2, 0.5]
        self.covariance_types = self.covariance_types or ["diagonal", "low_rank", "full_cholesky"]
        self.kl_weight_values = self.kl_weight_values or [0.0, 0.001, 0.01, 0.1]
        self.dropout_values = self.dropout_values or [0.0, 0.1, 0.2, 0.3]


def run_ablation_with_seeds(
    ablation_fn,
    seeds: list[int],
    *args,
    **kwargs,
) -> dict[str, dict[str, float]]:
    all_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        results = ablation_fn(*args, **kwargs)
        all_results.append(results)

    aggregated = {}
    keys = list(all_results[0].keys())

    for key in keys:
        metrics = {}
        for metric_name in all_results[0][key].keys():
            values = [r[key][metric_name] for r in all_results]
            metrics[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }
        aggregated[key] = metrics

    return aggregated


def run_n_basis_ablation(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    t: torch.Tensor,
    config: AblationConfig,
    n_obs: int = 10,
    n_epochs: int = 200,
) -> dict[int, dict]:
    results = {}
    t_min, t_max = float(t[0]), float(t[-1])

    for n_basis in config.n_basis_values:
        print(f"\n  n_basis = {n_basis}")
        knots_nb = create_knots(n_basis, t_min=t_min, t_max=t_max)

        model = TimeviewAdaptive(
            input_dim=x_train.shape[1],
            n_basis=n_basis,
            observation_noise=0.1,
            covariance_type="diagonal",
            knots=knots_nb,
        )

        train_model(model, x_train, t, y_train, n_epochs=n_epochs, lr=1e-3, n_obs=n_obs)

        model.eval()
        with torch.no_grad():
            t_obs = t[:n_obs]
            y_obs = y_test[:, :n_obs]
            y_mean, y_var, _, _ = model.update_and_predict(x_test, t_obs, y_obs, t)

            metrics = compute_all_metrics(
                y_test[:, n_obs:], y_mean[:, n_obs:], y_var[:, n_obs:], prefix="future_"
            )
            metrics["n_params"] = sum(p.numel() for p in model.parameters())

        results[n_basis] = metrics

    return results


def run_n_obs_ablation(
    model: TimeviewAdaptive,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    t: torch.Tensor,
    config: AblationConfig,
) -> dict[int, dict]:
    results = {}
    model.eval()

    with torch.no_grad():
        for n_obs in config.n_obs_values:
            if n_obs >= len(t):
                continue

            t_obs = t[:n_obs]
            y_obs = y_test[:, :n_obs]
            y_mean, y_var, _, _ = model.update_and_predict(x_test, t_obs, y_obs, t)

            metrics = compute_all_metrics(
                y_test[:, n_obs:], y_mean[:, n_obs:], y_var[:, n_obs:], prefix="future_"
            )

            metrics_obs = compute_all_metrics(
                y_test[:, :n_obs], y_mean[:, :n_obs], y_var[:, :n_obs], prefix="observed_"
            )
            metrics.update(metrics_obs)

            results[n_obs] = metrics

    return results


def run_covariance_ablation(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    t: torch.Tensor,
    config: AblationConfig,
    n_obs: int = 10,
    n_epochs: int = 200,
    knots: torch.Tensor | None = None,
) -> dict[str, dict]:
    results = {}

    for cov_type in config.covariance_types:
        print(f"\n  Covariance type: {cov_type}")

        if cov_type == "full_cholesky":
            model = TimeviewAdaptiveFullCholesky(
                input_dim=x_train.shape[1], n_basis=9, observation_noise=0.1, knots=knots,
            )
        else:
            model = TimeviewAdaptive(
                input_dim=x_train.shape[1],
                n_basis=9,
                observation_noise=0.1,
                covariance_type=cov_type,
                knots=knots,
            )

        train_model(model, x_train, t, y_train, n_epochs=n_epochs, lr=1e-3, n_obs=n_obs)

        model.eval()
        with torch.no_grad():
            t_obs = t[:n_obs]
            y_obs = y_test[:, :n_obs]
            y_mean, y_var, _, _ = model.update_and_predict(x_test, t_obs, y_obs, t)

            metrics = compute_all_metrics(
                y_test[:, n_obs:], y_mean[:, n_obs:], y_var[:, n_obs:], prefix="future_"
            )
            metrics["n_params"] = sum(p.numel() for p in model.parameters())

        results[cov_type] = metrics

    return results


def run_fix_ablation(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    t: torch.Tensor,
    config: AblationConfig,
    n_obs: int = 10,
    n_epochs: int = 200,
    knots: torch.Tensor | None = None,
) -> dict[str, dict]:
    results = {}
    input_dim = x_train.shape[1]

    print("\n  Fix: baseline (no fixes)")
    model_baseline = TimeviewAdaptive(
        input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="diagonal",
        dropout_p=0.0, use_batchnorm=False, learn_noise=False, kl_weight=0.0, knots=knots,
    )
    train_model(model_baseline, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                patience=0, weight_decay=0.0, verbose=False)
    results["baseline"] = _evaluate_fix(model_baseline, x_test, y_test, t, n_obs)

    print("\n  Fix: batchnorm")
    model_bn = TimeviewAdaptive(
        input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="diagonal",
        dropout_p=0.0, use_batchnorm=True, learn_noise=False, kl_weight=0.0, knots=knots,
    )
    train_model(model_bn, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                patience=0, weight_decay=0.0, verbose=False)
    results["batchnorm"] = _evaluate_fix(model_bn, x_test, y_test, t, n_obs)

    print("\n  Fix: dropout")
    model_dropout = TimeviewAdaptive(
        input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="diagonal",
        dropout_p=0.2, use_batchnorm=False, learn_noise=False, kl_weight=0.0, knots=knots,
    )
    train_model(model_dropout, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                patience=0, weight_decay=0.0, verbose=False)
    results["dropout"] = _evaluate_fix(model_dropout, x_test, y_test, t, n_obs)

    print("\n  Fix: kl_reg")
    model_kl = TimeviewAdaptive(
        input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="diagonal",
        dropout_p=0.0, use_batchnorm=False, learn_noise=False, kl_weight=0.01, knots=knots,
    )
    train_model(model_kl, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                patience=0, weight_decay=0.0, verbose=False)
    results["kl_reg"] = _evaluate_fix(model_kl, x_test, y_test, t, n_obs)

    print("\n  Fix: learn_noise")
    model_noise = TimeviewAdaptive(
        input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="diagonal",
        dropout_p=0.0, use_batchnorm=False, learn_noise=True, kl_weight=0.0, knots=knots,
    )
    train_model(model_noise, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                patience=0, weight_decay=0.0, verbose=False)
    results["learn_noise"] = _evaluate_fix(model_noise, x_test, y_test, t, n_obs)

    print("\n  Fix: early_stop")
    model_es = TimeviewAdaptive(
        input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="diagonal",
        dropout_p=0.0, use_batchnorm=False, learn_noise=False, kl_weight=0.0, knots=knots,
    )
    train_model(model_es, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                patience=20, weight_decay=0.0, verbose=False)
    results["early_stop"] = _evaluate_fix(model_es, x_test, y_test, t, n_obs)

    print("\n  Fix: weight_decay")
    model_wd = TimeviewAdaptive(
        input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="diagonal",
        dropout_p=0.0, use_batchnorm=False, learn_noise=False, kl_weight=0.0, knots=knots,
    )
    train_model(model_wd, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                patience=0, weight_decay=1e-5, verbose=False)
    results["weight_decay"] = _evaluate_fix(model_wd, x_test, y_test, t, n_obs)

    print("\n  Fix: all_fixes")
    model_all = TimeviewAdaptive(
        input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="diagonal",
        dropout_p=0.2, use_batchnorm=True, learn_noise=True, kl_weight=0.01, knots=knots,
    )
    train_model(model_all, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                verbose=False)
    results["all_fixes"] = _evaluate_fix(model_all, x_test, y_test, t, n_obs)

    return results


def _evaluate_fix(
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
        y_mean, y_var, mu_post, Sigma_post = model.update_and_predict(x_test, t_obs, y_obs, t)

        metrics = compute_all_metrics(
            y_test[:, n_obs:], y_mean[:, n_obs:], y_var[:, n_obs:], prefix="future_"
        )

        mu_0, Sigma_0, _bias = model.encode(x_test)
        prior_var = torch.diagonal(Sigma_0, dim1=1, dim2=2).mean().item()
        metrics["prior_avg_var"] = prior_var

        metrics["learned_sigma"] = model.sigma.item()

    return metrics


def run_kl_weight_ablation(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    t: torch.Tensor,
    config: AblationConfig,
    n_obs: int = 10,
    n_epochs: int = 200,
    knots: torch.Tensor | None = None,
) -> dict[float, dict]:
    results = {}

    for kl_weight in config.kl_weight_values:
        print(f"\n  kl_weight = {kl_weight}")

        model = TimeviewAdaptive(
            input_dim=x_train.shape[1],
            n_basis=9,
            observation_noise=0.1,
            covariance_type="diagonal",
            dropout_p=0.2,
            use_batchnorm=True,
            learn_noise=True,
            kl_weight=kl_weight,
            knots=knots,
        )

        train_model(model, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                    verbose=False)

        results[kl_weight] = _evaluate_fix(model, x_test, y_test, t, n_obs)

    return results


def run_nobs_ablation_all_datasets(seed: int = 0, n_epochs: int = 1000):
    datasets = ["airfoil", "flchain", "stress_strain"]
    n_obs_values = [5, 10, 15, 20, 25, 30, 35, 40, 45]

    print("=" * 70)
    print("N_OBS ABLATION ACROSS DATASETS")
    print("=" * 70)

    all_dataset_results = {}

    for dataset_name in datasets:
        print(f"\n--- {dataset_name.upper()} ---")
        try:
            (
                x_tr, x_va, x_te, t_ds, y_tr, y_va, y_te, y_norm, knots_ds, n_basis_ds,
                ts_tr, ys_tr, ts_va, ys_va, ts_te, ys_te,
            ) = load_and_normalize_dataset(dataset_name, seed=seed)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        n_time = len(t_ds)
        print(f"  t range: [{t_ds[0]:.4f}, {t_ds[-1]:.4f}], {n_time} points, n_basis={n_basis_ds}")

        torch.manual_seed(seed)
        static_model = TimeviewStatic(input_dim=x_tr.shape[1], n_basis=n_basis_ds, knots=knots_ds)
        train_static_model(static_model, x_tr, t_ds, y_tr, x_val=x_va, y_val=y_va,
                           ts_train=ts_tr, ys_train=ys_tr, ts_val=ts_va, ys_val=ys_va)
        static_model.eval()
        with torch.no_grad():
            static_mse = ((y_te - static_model(x_te, t_ds)[0]) ** 2).mean().item()

        print("  Training adaptive (standard)...")
        torch.manual_seed(seed)
        adaptive = TimeviewAdaptive(
            input_dim=x_tr.shape[1], n_basis=n_basis_ds,
            observation_noise=0.1, covariance_type="diagonal", knots=knots_ds,
        )
        train_model(adaptive, x_tr, t_ds, y_tr, n_epochs=n_epochs, n_obs=10)

        print("  Training adaptive (gated)...")
        torch.manual_seed(seed)
        gated = TimeviewAdaptiveGated(
            input_dim=x_tr.shape[1], n_basis=n_basis_ds,
            observation_noise=0.1, covariance_type="diagonal", knots=knots_ds,
        )
        train_model(gated, x_tr, t_ds, y_tr, n_epochs=n_epochs, n_obs=10)

        adaptive.eval()
        gated.eval()
        valid_nobs = [n for n in n_obs_values if n < n_time]
        std_mses, gated_mses, t_maxes = [], [], []

        for n_obs in valid_nobs:
            t_obs = t_ds[:n_obs]
            y_obs = y_te[:, :n_obs]
            t_maxes.append(t_obs[-1].item())
            with torch.no_grad():
                y_std, _, _, _ = adaptive.update_and_predict(x_te, t_obs, y_obs, t_ds)
                std_mses.append(((y_te[:, n_obs:] - y_std[:, n_obs:]) ** 2).mean().item())
                y_g, _, _, _ = gated.update_and_predict(x_te, t_obs, y_obs, t_ds)
                gated_mses.append(((y_te[:, n_obs:] - y_g[:, n_obs:]) ** 2).mean().item())

        all_dataset_results[dataset_name] = {
            "n_obs_values": valid_nobs,
            "std_mses": std_mses,
            "gated_mses": gated_mses,
            "static_mse": static_mse,
            "t_maxes": t_maxes,
            "n_time": n_time,
            "t_max": t_ds[-1].item(),
            "knot_max": float(knots_ds[-1]),
        }

        fractions = [n / n_time * 100 for n in valid_nobs]
        std_improv = [(1 - m / static_mse) * 100 for m in std_mses]
        gated_improv = [(1 - m / static_mse) * 100 for m in gated_mses]

        print(f"\n  {'n_obs':>6} {'t_max':>8} {'%Traj':>6} {'Std MSE':>10} {'Gated MSE':>11} {'Std Improv':>12} {'Gated Improv':>14}")
        print(f"  {'-' * 72}")
        for i, n_obs in enumerate(valid_nobs):
            print(
                f"  {n_obs:>6} {t_maxes[i]:>8.3f} {fractions[i]:>5.0f}% "
                f"{std_mses[i]:>10.4f} {gated_mses[i]:>11.4f} "
                f"{std_improv[i]:>11.1f}% {gated_improv[i]:>13.1f}%"
            )
        print(f"  Static MSE: {static_mse:.4f}")

    n_datasets = len(all_dataset_results)
    fig, axes = plt.subplots(2, n_datasets, figsize=(5 * n_datasets, 9))
    if n_datasets == 1:
        axes = axes.reshape(-1, 1)

    colors_std = "#2196F3"
    colors_gated = "#FF9800"

    for col, (ds_name, ds_res) in enumerate(all_dataset_results.items()):
        nobs = ds_res["n_obs_values"]
        n_time = ds_res["n_time"]
        static_mse = ds_res["static_mse"]
        s_improv = [(1 - m / static_mse) * 100 for m in ds_res["std_mses"]]
        g_improv = [(1 - m / static_mse) * 100 for m in ds_res["gated_mses"]]

        ax = axes[0, col]
        ax.plot(nobs, ds_res["std_mses"], "o-", color=colors_std, linewidth=2, markersize=6, label="Standard")
        ax.plot(nobs, ds_res["gated_mses"], "s-", color=colors_gated, linewidth=2, markersize=6, label="Gated")
        ax.axhline(y=static_mse, color="#666", linestyle="--", linewidth=2, label=f"Static ({static_mse:.3f})")
        ax.set_xlabel("n_obs")
        ax.set_ylabel("Future MSE")
        title = ds_name.replace("_", "-").title()
        ax.set_title(f"{title}\ndata [0, {ds_res['t_max']:.2f}], knots [0, {ds_res['knot_max']:.1f}]")
        ax.legend(fontsize=8)
        ax.set_xticks(nobs)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        ax2 = axes[1, col]
        width = 1.2
        ax2.bar([n - width / 2 for n in nobs], s_improv, width=width, color=colors_std, alpha=0.8, label="Standard")
        ax2.bar([n + width / 2 for n in nobs], g_improv, width=width, color=colors_gated, alpha=0.8, label="Gated")
        ax2.axhline(y=0, color="black", linewidth=1)
        ax2.set_xlabel("n_obs (% trajectory)")
        ax2.set_ylabel("MSE Improvement (%)")
        ax2.set_xticks(nobs)
        ax2.set_xticklabels([f"{n}\n({n / n_time * 100:.0f}%)" for n in nobs], fontsize=7)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Adaptation Benefit vs Number of Observations", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = FIGURES_DIR / "nobs_ablation_all_datasets.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")

    return all_dataset_results
