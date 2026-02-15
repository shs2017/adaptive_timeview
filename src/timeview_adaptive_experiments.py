import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ablation import (
    AblationConfig,
    run_covariance_ablation,
    run_fix_ablation,
    run_kl_weight_ablation,
    run_n_basis_ablation,
    run_n_obs_ablation,
    run_nobs_ablation_all_datasets,
)
from constant import FIGURES_DIR, TABLES_DIR
from data import load_and_normalize_dataset
from evaluation import eval_model_full, streaming_evaluation
from models import TimeviewStatic
from plot import (
    plot_active_scheduling,
    plot_aec_comparison,
    plot_best_config_comparison,
    plot_calibration,
    plot_gp_comparison,
    plot_heteroscedastic_comparison,
    plot_streaming_results,
    plot_transition_analysis,
    plot_uncertainty_decomposition_grid,
)
from scores import (
    compute_all_metrics,
    crps_gaussian,
)
from timeview_adaptive import (
    ActiveObservationScheduler,
    GPBaseline,
    TemperatureScaling,
    TimeviewAdaptive,
    TimeviewAdaptiveGated,
    TimeviewAdaptiveGatedHeteroscedastic,
    TimeviewAdaptiveHeteroscedastic,
    bspline_basis,
    train_model,
)
from train import train_static_model, train_static_with_params, tune_static_hyperparams
from transition_point import TransitionPointAnalyzer


def run_temperature_scaling_experiment(
    model: TimeviewAdaptive,
    x_cal: torch.Tensor,
    y_cal: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    t: torch.Tensor,
    n_obs: int = 10,
) -> dict:
    model.eval()

    with torch.no_grad():
        t_obs = t[:n_obs]
        y_obs_test = y_test[:, :n_obs]
        y_mean, y_var, _, _ = model.update_and_predict(x_test, t_obs, y_obs_test, t)

        metrics_before = compute_all_metrics(
            y_test[:, n_obs:], y_mean[:, n_obs:], y_var[:, n_obs:], prefix="before_"
        )

    temp_scaler = TemperatureScaling()
    temperature = temp_scaler.calibrate(model, x_cal, t, y_cal, n_obs=n_obs)

    with torch.no_grad():
        y_var_scaled = temp_scaler.apply(y_var)
        metrics_after = compute_all_metrics(
            y_test[:, n_obs:], y_mean[:, n_obs:], y_var_scaled[:, n_obs:], prefix="after_"
        )

    return {
        "temperature": temperature,
        **metrics_before,
        **metrics_after,
    }


def run_heteroscedastic_comparison(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    t: torch.Tensor,
    n_obs: int = 10,
    n_epochs: int = 200,
    knots: torch.Tensor | None = None,
) -> dict[str, dict]:
    results = {}
    input_dim = x_train.shape[1]

    print("  Training homoscedastic model...")
    model_homo = TimeviewAdaptive(
        input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="diagonal",
        use_batchnorm=True, learn_noise=True, kl_weight=0.01, knots=knots,
    )
    train_model(model_homo, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                verbose=False)
    results["homoscedastic"] = eval_model_full(model_homo, x_test, y_test, t, n_obs)

    print("  Training heteroscedastic model...")
    model_hetero = TimeviewAdaptiveHeteroscedastic(
        input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="diagonal",
        use_batchnorm=True, kl_weight=0.01, knots=knots,
    )
    train_model(model_hetero, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                verbose=False)
    results["heteroscedastic"] = eval_model_full(model_hetero, x_test, y_test, t, n_obs)

    print("  Training gated model...")
    model_gated = TimeviewAdaptiveGated(
        input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="diagonal",
        use_batchnorm=True, learn_noise=True, kl_weight=0.01, knots=knots,
    )
    train_model(model_gated, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                verbose=False)
    results["gated"] = eval_model_full(model_gated, x_test, y_test, t, n_obs)

    return results



def run_gp_baseline_comparison(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    t: torch.Tensor,
    n_obs: int = 10,
    n_epochs: int = 200,
    knots: torch.Tensor | None = None,
    ts_train: list[np.ndarray] | None = None,
    ys_train: list[np.ndarray] | None = None,
    ts_val: list[np.ndarray] | None = None,
    ys_val: list[np.ndarray] | None = None,
) -> dict[str, dict]:
    results = {}
    input_dim = x_train.shape[1]

    print("  Evaluating GP baseline...")
    gp = GPBaseline(length_scale=0.2, signal_var=1.0, noise_var=0.01)

    best_ll = -float("inf")
    best_params = (0.2, 1.0, 0.01)
    for ls in [0.05, 0.1, 0.2, 0.3, 0.5]:
        for sv in [0.5, 1.0, 2.0]:
            for nv in [0.001, 0.01, 0.05, 0.1]:
                gp_trial = GPBaseline(length_scale=ls, signal_var=sv, noise_var=nv)
                try:
                    t_obs = t[:n_obs]
                    y_obs = y_train[:, :n_obs]
                    y_mean, y_var = gp_trial.predict(t_obs, y_obs, t)

                    ll = -0.5 * ((y_train - y_mean) ** 2 / y_var + torch.log(y_var)).mean().item()
                    if ll > best_ll:
                        best_ll = ll
                        best_params = (ls, sv, nv)
                except Exception:
                    continue

    gp = GPBaseline(*best_params)
    print(f"  Best GP params: length_scale={best_params[0]}, signal_var={best_params[1]}, noise_var={best_params[2]}")

    with torch.no_grad():
        t_obs = t[:n_obs]
        y_obs = y_test[:, :n_obs]
        y_gp_mean, y_gp_var = gp.predict(t_obs, y_obs, t)
        gp_metrics = compute_all_metrics(
            y_test[:, n_obs:], y_gp_mean[:, n_obs:], y_gp_var[:, n_obs:], prefix="future_"
        )
    results["gp_baseline"] = gp_metrics

    print("  Training TIMEVIEW-Adaptive for comparison...")
    model = TimeviewAdaptive(
        input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="diagonal",
        use_batchnorm=True, learn_noise=True, kl_weight=0.01, knots=knots,
    )
    train_model(model, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                verbose=False)
    results["timeview_adaptive"] = eval_model_full(model, x_test, y_test, t, n_obs)

    print("  Training static baseline for comparison...")
    static_model = TimeviewStatic(input_dim=input_dim, n_basis=9, knots=knots)
    train_static_model(static_model, x_train, t, y_train,
                       ts_train=ts_train, ys_train=ys_train,
                       ts_val=ts_val, ys_val=ys_val)
    static_model.eval()
    with torch.no_grad():
        y_static_mean, y_static_var = static_model(x_test, t)
        static_metrics = compute_all_metrics(
            y_test[:, n_obs:], y_static_mean[:, n_obs:], y_static_var[:, n_obs:], prefix="future_"
        )
    results["static_baseline"] = static_metrics

    return results


def run_active_scheduling_experiment(
    model: TimeviewAdaptive,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    t: torch.Tensor,
    n_initial_obs: int = 5,
    n_additional: int = 10,
) -> dict:
    model.eval()
    scheduler = ActiveObservationScheduler(model)

    n_timepoints = len(t)

    initial_idx = list(range(n_initial_obs))

    active_mse = []
    uniform_mse = []
    active_coverage = []
    uniform_coverage = []
    active_crps = []
    uniform_crps = []
    selected_times = []

    with torch.no_grad():
        active_obs_idx = list(initial_idx)
        for _ in range(n_additional):
            t_obs = t[active_obs_idx]
            y_obs = y_test[:, active_obs_idx]

            unobserved_idx = [i for i in range(n_timepoints) if i not in active_obs_idx]
            if not unobserved_idx:
                break
            t_candidates = t[unobserved_idx]

            best_times, ig = scheduler.suggest_next_observation(
                x_test, t_obs, y_obs, t_candidates
            )

            mean_ig = ig.mean(dim=0)
            best_cand_idx = mean_ig.argmax().item()
            new_obs_idx = unobserved_idx[best_cand_idx]
            active_obs_idx.append(new_obs_idx)
            selected_times.append(t[new_obs_idx].item())

            t_obs_new = t[active_obs_idx]
            y_obs_new = y_test[:, active_obs_idx]
            y_mean, y_var, _, _ = model.update_and_predict(
                x_test, t_obs_new, y_obs_new, t
            )

            remaining_idx = [i for i in range(n_timepoints) if i not in active_obs_idx]
            if remaining_idx:
                y_future = y_test[:, remaining_idx]
                y_mean_f = y_mean[:, remaining_idx]
                y_var_f = y_var[:, remaining_idx]
                active_mse.append(((y_future - y_mean_f) ** 2).mean().item())
                y_std_f = torch.sqrt(y_var_f)
                active_coverage.append(
                    (torch.abs(y_future - y_mean_f) < 1.96 * y_std_f).float().mean().item()
                )
                active_crps.append(
                    crps_gaussian(y_future, y_mean_f, y_std_f).mean().item()
                )

        remaining_after_initial = [i for i in range(n_timepoints) if i not in initial_idx]
        uniform_additional = np.linspace(
            0, len(remaining_after_initial) - 1, n_additional, dtype=int
        )
        uniform_obs_idx = list(initial_idx)

        for step in range(n_additional):
            if step < len(uniform_additional):
                new_idx = remaining_after_initial[uniform_additional[step]]
                uniform_obs_idx.append(new_idx)

            t_obs = t[uniform_obs_idx]
            y_obs = y_test[:, uniform_obs_idx]
            y_mean, y_var, _, _ = model.update_and_predict(x_test, t_obs, y_obs, t)

            remaining_idx = [i for i in range(n_timepoints) if i not in uniform_obs_idx]
            if remaining_idx:
                y_future = y_test[:, remaining_idx]
                y_mean_f = y_mean[:, remaining_idx]
                y_var_f = y_var[:, remaining_idx]
                uniform_mse.append(((y_future - y_mean_f) ** 2).mean().item())
                y_std_f = torch.sqrt(y_var_f)
                uniform_coverage.append(
                    (torch.abs(y_future - y_mean_f) < 1.96 * y_std_f).float().mean().item()
                )
                uniform_crps.append(
                    crps_gaussian(y_future, y_mean_f, y_std_f).mean().item()
                )

    return {
        "active_mse": active_mse,
        "uniform_mse": uniform_mse,
        "active_coverage": active_coverage,
        "uniform_coverage": uniform_coverage,
        "active_crps": active_crps,
        "uniform_crps": uniform_crps,
        "selected_times": selected_times,
        "n_steps": list(range(1, len(active_mse) + 1)),
    }


def compute_adaptation_efficiency_curve(
    model: TimeviewAdaptive,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    max_obs: int | None = None,
) -> dict:
    model.eval()
    n_timepoints = len(t)
    if max_obs is None:
        max_obs = min(n_timepoints - 2, 30)

    with torch.no_grad():
        mu_0, Sigma_0, bias = model.encode(x)
        y_prior_mean, _ = model.predict(mu_0, Sigma_0, t, bias=bias)
        mse_prior = ((y - y_prior_mean) ** 2).mean().item()

        n_obs_list = []
        mse_list = []
        aec_list = []
        cumulative_reduction = []

        for k in range(1, max_obs + 1):
            t_obs = t[:k]
            y_obs = y[:, :k]
            y_mean, y_var, _, _ = model.update_and_predict(x, t_obs, y_obs, t)

            mse_k = ((y - y_mean) ** 2).mean().item()
            reduction = mse_prior - mse_k
            efficiency = reduction / k

            n_obs_list.append(k)
            mse_list.append(mse_k)
            aec_list.append(efficiency)
            cumulative_reduction.append(reduction)

    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aaec = float(trapz_fn(aec_list, n_obs_list)) / max_obs if aec_list else 0.0

    return {
        "n_obs": n_obs_list,
        "mse": mse_list,
        "aec": aec_list,
        "cumulative_reduction": cumulative_reduction,
        "mse_prior": mse_prior,
        "aaec": aaec,
    }


def run_aec_experiment(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    t: torch.Tensor,
    n_obs: int = 10,
    n_epochs: int = 200,
    knots: torch.Tensor | None = None,
) -> dict:
    """Compare AEC across model variants."""
    results = {}
    input_dim = x_train.shape[1]

    configs = {
        "standard": dict(
            cls=TimeviewAdaptive,
            kwargs=dict(input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="diagonal",
                        dropout_p=0.2, use_batchnorm=True, learn_noise=True, kl_weight=0.01, knots=knots),
        ),
        "gated": dict(
            cls=TimeviewAdaptiveGated,
            kwargs=dict(input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="diagonal",
                        dropout_p=0.2, use_batchnorm=True, learn_noise=True, kl_weight=0.01, knots=knots),
        ),
        "best_config": dict(
            cls=TimeviewAdaptiveGatedHeteroscedastic,
            kwargs=dict(input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="low_rank",
                        dropout_p=0.2, use_batchnorm=True, kl_weight=0.01, knots=knots),
        ),
    }

    for name, cfg in configs.items():
        print(f"    Training {name}...")
        model = cfg["cls"](**cfg["kwargs"])
        train_model(model, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                    verbose=False)
        aec = compute_adaptation_efficiency_curve(model, x_test, t, y_test, max_obs=25)
        results[name] = aec

    return results


def compute_uncertainty_decomposition(
    model: TimeviewAdaptive,
    x: torch.Tensor,
    t_obs: torch.Tensor,
    y_obs: torch.Tensor,
    t_pred: torch.Tensor,
) -> dict:
    """
    Decompose predictive variance into three components:
        Var[y(t)] = prior_epistemic(t) - info_gained(t) + noise(t)
    """
    model.eval()
    with torch.no_grad():
        mu_0, Sigma_0, _bias = model.encode(x)

        Phi_obs = model.get_basis(t_obs).unsqueeze(0).expand(x.shape[0], -1, -1)
        has_noise_model = hasattr(model, 'noise_model') and model.noise_model is not None
        if has_noise_model:
            noise_var_obs = model.get_noise_var(t_obs)
            mu_post, Sigma_post = model.updater.update(
                mu_0, Sigma_0, Phi_obs, y_obs, noise_var=noise_var_obs
            )
        else:
            model.updater.sigma2 = model.sigma ** 2
            mu_post, Sigma_post = model.updater.update(mu_0, Sigma_0, Phi_obs, y_obs)

        Phi_pred = model.get_basis(t_pred)

        PhiSigma0 = torch.matmul(Phi_pred.unsqueeze(0), Sigma_0)
        prior_epistemic = torch.sum(PhiSigma0 * Phi_pred.unsqueeze(0), dim=-1)

        PhiSigmaN = torch.matmul(Phi_pred.unsqueeze(0), Sigma_post)
        posterior_epistemic = torch.sum(PhiSigmaN * Phi_pred.unsqueeze(0), dim=-1)

        info_gained = prior_epistemic - posterior_epistemic

        if has_noise_model:
            noise = model.get_noise_var(t_pred).unsqueeze(0).expand_as(prior_epistemic)
        else:
            noise_var = (model.sigma ** 2).item()
            noise = torch.full_like(prior_epistemic, noise_var)

        total_var = posterior_epistemic + noise

    return {
        "t": t_pred,
        "total_var": total_var,
        "prior_epistemic": prior_epistemic,
        "info_gained": info_gained,
        "noise": noise,
        "posterior_epistemic": posterior_epistemic,
    }


def run_uncertainty_decomposition_experiment(
    model: TimeviewAdaptive,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    t: torch.Tensor,
    n_obs_values: list[int] | None = None,
) -> dict:
    """
    Run uncertainty decomposition at different observation counts.
    """
    if n_obs_values is None:
        n_obs_values = [2, 5, 10, 20]

    results = {}
    for n_obs in n_obs_values:
        if n_obs >= len(t):
            continue
        t_obs = t[:n_obs]
        y_obs = y_test[:, :n_obs]
        decomp = compute_uncertainty_decomposition(model, x_test, t_obs, y_obs, t)

        results[n_obs] = {
            "decomp": decomp,
            "avg_prior_epistemic": decomp["prior_epistemic"].mean().item(),
            "avg_info_gained": decomp["info_gained"].mean().item(),
            "avg_posterior_epistemic": decomp["posterior_epistemic"].mean().item(),
            "avg_noise": decomp["noise"].mean().item(),
            "avg_total": decomp["total_var"].mean().item(),
            "info_fraction": (decomp["info_gained"].mean() / (decomp["prior_epistemic"].mean() + 1e-8)).item(),
        }

    return results


def run_best_config_experiment(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    t: torch.Tensor,
    n_obs: int = 10,
    n_epochs: int = 200,
    knots: torch.Tensor | None = None,
) -> dict:
    results = {}
    input_dim = x_train.shape[1]

    print("    Training standard model (n_basis=9, diagonal)...")
    model_std = TimeviewAdaptive(
        input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="diagonal",
        use_batchnorm=True, learn_noise=True, kl_weight=0.01, knots=knots,
    )
    train_model(model_std, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                verbose=False)
    results["standard"] = eval_model_full(model_std, x_test, y_test, t, n_obs)

    print("    Training gated model (n_basis=9)...")
    model_gated = TimeviewAdaptiveGated(
        input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="diagonal",
        use_batchnorm=True, learn_noise=True, kl_weight=0.01, knots=knots,
    )
    train_model(model_gated, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                verbose=False)
    results["gated"] = eval_model_full(model_gated, x_test, y_test, t, n_obs)

    print("    Training heteroscedastic model (n_basis=9)...")
    model_hetero = TimeviewAdaptiveHeteroscedastic(
        input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="diagonal",
        use_batchnorm=True, kl_weight=0.01, knots=knots,
    )
    train_model(model_hetero, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                verbose=False)
    results["heteroscedastic"] = eval_model_full(model_hetero, x_test, y_test, t, n_obs)

    print("    Training best config (gated+hetero, n_basis=9, low_rank)...")
    model_best = TimeviewAdaptiveGatedHeteroscedastic(
        input_dim=input_dim, n_basis=9, observation_noise=0.1, covariance_type="low_rank",
        use_batchnorm=True, kl_weight=0.01, knots=knots,
    )
    train_model(model_best, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                verbose=False)
    results["best_config"] = eval_model_full(model_best, x_test, y_test, t, n_obs)

    return results


def run_all_experiments(seed: int = 0, n_epochs: int = 1000, n_seeds: int = 3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    ablation_seeds = [seed + i * 100 for i in range(n_seeds)]

    print("=" * 70)
    print("TIMEVIEW-Adaptive: Extended Experiments")
    print(f"  Using {n_seeds} seeds for ablation studies: {ablation_seeds}")
    print("=" * 70)

    n_obs = 10

    print("\n[1/10] Loading AIRFOIL dataset...")
    (
        x_train, x_val, x_test, t, y_train, y_val, y_test, y_normalizer, knots, n_basis_ds,
        ts_train_ps, ys_train_ps, ts_val_ps, ys_val_ps, ts_test_ps, ys_test_ps,
    ) = load_and_normalize_dataset("airfoil", seed=seed)
    n_train = x_train.shape[0]
    n_samples = n_train + x_val.shape[0] + x_test.shape[0]
    n_timepoints = len(t)
    input_dim = x_train.shape[1]

    print(f"  Loaded {n_samples} samples, {n_timepoints} time points, {input_dim} features")
    print(f"  Train: {n_train}, Val: {x_val.shape[0]}, Test: {x_test.shape[0]}")
    if knots is not None:
        print(f"  Knots: {knots.shape[0]} ({knots.shape[0] - 2*3} internal)")

    all_results = {}

    print("\n[2/10] Running covariance type ablation with multiple seeds...")
    config = AblationConfig()

    cov_results_all_seeds = []
    for s in ablation_seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        cov_res = run_covariance_ablation(
            x_train, y_train, x_test, y_test, t, config, n_obs, n_epochs, knots=knots
        )
        cov_results_all_seeds.append(cov_res)

    cov_results_aggregated = {}
    for cov_type in config.covariance_types:
        metrics_agg = {}
        for metric_name in cov_results_all_seeds[0][cov_type].keys():
            values = [r[cov_type][metric_name] for r in cov_results_all_seeds]
            metrics_agg[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }
        cov_results_aggregated[cov_type] = metrics_agg

    all_results["covariance_ablation"] = cov_results_aggregated

    print("\n  Results (mean ± std across seeds):")
    for cov_type, metrics in cov_results_aggregated.items():
        print(
            f"    {cov_type}: MSE={metrics['future_mse']['mean']:.4f}±{metrics['future_mse']['std']:.4f}, "
            f"CRPS={metrics['future_crps']['mean']:.4f}±{metrics['future_crps']['std']:.4f}, "
            f"Coverage95={metrics['future_coverage_95']['mean']:.2%}"
        )

    print("\n[3/10] Training static TIMEVIEW baseline...")

    static_model = TimeviewStatic(input_dim=input_dim, n_basis=9, knots=knots)
    train_static_model(static_model, x_train, t, y_train,
                       x_val=x_val, y_val=y_val,
                       ts_train=ts_train_ps, ys_train=ys_train_ps,
                       ts_val=ts_val_ps, ys_val=ys_val_ps)


    adaptive_model = TimeviewAdaptive(
        input_dim=input_dim,
        n_basis=9,
        observation_noise=0.1,
        covariance_type="diagonal",
        knots=knots,
    )
    print("  Training adaptive model...")
    train_model(adaptive_model, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs)

    static_model.eval()
    adaptive_model.eval()

    with torch.no_grad():
        Phis_test = [
            bspline_basis(torch.tensor(ts, dtype=torch.float32), static_model.knots)
            for ts in ts_test_ps
        ]
        ys_test_t = [torch.tensor(ys, dtype=torch.float32) for ys in ys_test_ps]
        static_preds = static_model.predict_per_sample(x_test, Phis_test)
        static_mse_persample = float(np.mean([
            nn.functional.mse_loss(pred, y_true).item()
            for pred, y_true in zip(static_preds, ys_test_t, strict=True)
        ]))

        y_static_mean, y_static_var = static_model(x_test, t)
        static_metrics = compute_all_metrics(
            y_test, y_static_mean, y_static_var, prefix=""
        )

        static_metrics["mse"] = static_mse_persample

        t_obs = t[:n_obs]
        y_obs = y_test[:, :n_obs]
        y_adapt_mean, y_adapt_var, _, _ = adaptive_model.update_and_predict(x_test, t_obs, y_obs, t)
        adaptive_metrics = compute_all_metrics(
            y_test, y_adapt_mean, y_adapt_var, prefix=""
        )

    all_results["baseline_comparison"] = {"static": static_metrics, "adaptive": adaptive_metrics}

    print("\n  Static vs Adaptive comparison (all points, static per-sample):")
    print(f"    Static:   MSE={static_metrics['mse']:.4f}, CRPS={static_metrics['crps']:.4f}")
    print(f"    Adaptive: MSE={adaptive_metrics['mse']:.4f}, CRPS={adaptive_metrics['crps']:.4f}")
    print(
        f"    Improvement: {(1 - adaptive_metrics['mse'] / static_metrics['mse']) * 100:.1f}% MSE reduction"
    )

    print("\n[4/10] Running n_basis ablation with multiple seeds...")

    n_basis_results_all_seeds = []
    for s in ablation_seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        nb_res = run_n_basis_ablation(x_train, y_train, x_test, y_test, t, config, n_obs, n_epochs)
        n_basis_results_all_seeds.append(nb_res)

    n_basis_results_aggregated = {}
    for n_basis in config.n_basis_values:
        metrics_agg = {}
        for metric_name in n_basis_results_all_seeds[0][n_basis].keys():
            values = [r[n_basis][metric_name] for r in n_basis_results_all_seeds]
            metrics_agg[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }
        n_basis_results_aggregated[n_basis] = metrics_agg

    all_results["n_basis_ablation"] = n_basis_results_aggregated

    print("\n  Results (mean ± std):")
    for n_basis, metrics in n_basis_results_aggregated.items():
        print(
            f"    n_basis={n_basis}: MSE={metrics['future_mse']['mean']:.4f}±{metrics['future_mse']['std']:.4f}, "
            f"params={int(metrics['n_params']['mean'])}"
        )

    print("\n[5/10] Running n_obs ablation...")

    n_obs_results_all_seeds = []
    for s in ablation_seeds:
        torch.manual_seed(s)
        perm = torch.randperm(x_test.shape[0])
        x_test_perm = x_test[perm]
        y_test_perm = y_test[perm]
        nobs_res = run_n_obs_ablation(adaptive_model, x_test_perm, y_test_perm, t, config)
        n_obs_results_all_seeds.append(nobs_res)

    n_obs_results_aggregated = {}
    n_obs_keys = [k for k in config.n_obs_values if k < len(t)]
    for n_obs_val in n_obs_keys:
        metrics_agg = {}
        for metric_name in n_obs_results_all_seeds[0][n_obs_val].keys():
            values = [r[n_obs_val][metric_name] for r in n_obs_results_all_seeds]
            metrics_agg[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }
        n_obs_results_aggregated[n_obs_val] = metrics_agg

    all_results["n_obs_ablation"] = n_obs_results_aggregated

    print("\n  Results (mean ± std):")
    for n_obs_val, metrics in list(n_obs_results_aggregated.items())[:5]:
        print(
            f"    n_obs={n_obs_val}: Future MSE={metrics['future_mse']['mean']:.4f}±{metrics['future_mse']['std']:.4f}, "
            f"Uncertainty={metrics['future_avg_std']['mean']:.4f}"
        )

    print("\n[6/10] Running transition point analysis...")
    analyzer = TransitionPointAnalyzer(adaptive_model, n_samples=500)
    analysis = analyzer.analyze(x_test[:5], t[:n_obs], y_test[:5, :n_obs], t)

    all_results["transition_analysis"] = {"sample_transitions": analysis["transitions"]}

    print("\n  Transition points for first 5 test samples:")
    for i, trans in enumerate(analysis["transitions"]):
        if trans["mean"] is not None:
            print(
                f"    Sample {i}: mean={trans['mean']:.3f}, CI=[{trans['ci_lower']:.3f}, {trans['ci_upper']:.3f}]"
            )
        else:
            print(f"    Sample {i}: No clear transitions detected")

    print("\n[7/10] Running fix ablation study (comparing individual fixes)...")

    fix_results_all_seeds = []
    for s in ablation_seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        fix_res = run_fix_ablation(
            x_train, y_train, x_test, y_test, t, config, n_obs, n_epochs, knots=knots
        )
        fix_results_all_seeds.append(fix_res)

    fix_results_aggregated = {}
    fix_names = list(fix_results_all_seeds[0].keys())
    for fix_name in fix_names:
        metrics_agg = {}
        for metric_name in fix_results_all_seeds[0][fix_name].keys():
            values = [r[fix_name][metric_name] for r in fix_results_all_seeds]
            metrics_agg[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }
        fix_results_aggregated[fix_name] = metrics_agg

    all_results["fix_ablation"] = fix_results_aggregated

    print("\n  Fix ablation results (mean ± std):")
    for fix_name, metrics in fix_results_aggregated.items():
        cov95 = metrics["future_coverage_95"]["mean"]
        print(
            f"    {fix_name:<15}: MSE={metrics['future_mse']['mean']:.4f}±{metrics['future_mse']['std']:.4f}, "
            f"CRPS={metrics['future_crps']['mean']:.4f}, Coverage95={cov95:.2%}, "
            f"PriorVar={metrics['prior_avg_var']['mean']:.4f}"
        )

    print("\n[8/10] Running KL weight ablation study...")

    kl_results_all_seeds = []
    for s in ablation_seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        kl_res = run_kl_weight_ablation(
            x_train, y_train, x_test, y_test, t, config, n_obs, n_epochs, knots=knots
        )
        kl_results_all_seeds.append(kl_res)

    kl_results_aggregated = {}
    for kl_weight in config.kl_weight_values:
        metrics_agg = {}
        for metric_name in kl_results_all_seeds[0][kl_weight].keys():
            values = [r[kl_weight][metric_name] for r in kl_results_all_seeds]
            metrics_agg[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }
        kl_results_aggregated[kl_weight] = metrics_agg

    all_results["kl_weight_ablation"] = kl_results_aggregated

    print("\n  KL weight ablation results (mean ± std):")
    for kl_weight, metrics in kl_results_aggregated.items():
        print(
            f"    kl_weight={kl_weight}: MSE={metrics['future_mse']['mean']:.4f}, "
            f"Coverage95={metrics['future_coverage_95']['mean']:.2%}, "
            f"PriorVar={metrics['prior_avg_var']['mean']:.4f}"
        )

    print("\n[9/10] Running temperature scaling experiment...")

    model_for_temp = TimeviewAdaptive(
        input_dim=input_dim,
        n_basis=9,
        observation_noise=0.1,
        covariance_type="diagonal",
        dropout_p=0.2,
        use_batchnorm=True,
        learn_noise=True,
        kl_weight=0.01,
        knots=knots,
    )
    train_model(model_for_temp, x_train, t, y_train, n_epochs=n_epochs, n_obs=n_obs,
                verbose=False)

    temp_results = run_temperature_scaling_experiment(
        model_for_temp, x_val, y_val, x_test, y_test, t, n_obs
    )
    all_results["temperature_scaling"] = temp_results

    print("\n  Temperature scaling results:")
    print(f"    Learned temperature: {temp_results['temperature']:.4f}")
    print(f"    Before: Coverage95={temp_results['before_coverage_95']:.2%}, CalibError={temp_results['before_calibration_error']:.4f}")
    print(f"    After:  Coverage95={temp_results['after_coverage_95']:.2%}, CalibError={temp_results['after_calibration_error']:.4f}")

    print("\n[10/10] Testing on real datasets with Static vs Adaptive comparison...")

    dataset_results = {}
    real_datasets = ["airfoil", "flchain", "stress_strain"]

    for dataset_name in real_datasets:
        print(f"\n  === {dataset_name.upper()} Dataset ===")
        try:
            (
                x_tr, x_va, x_te, t_ds, y_tr, y_va, y_te, y_norm_ds, knots_ds, n_basis_ds,
                ts_tr, ys_tr, ts_va, ys_va, ts_te, ys_te,
            ) = load_and_normalize_dataset(dataset_name, seed=seed)
            print(
                f"    Loaded {len(x_tr) + len(x_va) + len(x_te)} samples, {len(t_ds)} time points, {x_tr.shape[1]} features"
            )
        except FileNotFoundError as e:
            print(f"    Skipping {dataset_name}: {e}")
            continue

        print("    Training static baseline...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        static_ds = TimeviewStatic(
            input_dim=x_tr.shape[1],
            n_basis=n_basis_ds,
            knots=knots_ds,
        )
        train_static_model(static_ds, x_tr, t_ds, y_tr,
                           x_val=x_va, y_val=y_va,
                           ts_train=ts_tr, ys_train=ys_tr,
                           ts_val=ts_va, ys_val=ys_va)

        print("    Training adaptive model (standard)...")
        adaptive_ds = TimeviewAdaptive(
            input_dim=x_tr.shape[1],
            n_basis=n_basis_ds,
            observation_noise=0.1,
            covariance_type="diagonal",
            knots=knots_ds,
        )
        train_model(adaptive_ds, x_tr, t_ds, y_tr, n_epochs=n_epochs, n_obs=n_obs)

        print("    Training adaptive model (gated)...")
        gated_ds = TimeviewAdaptiveGated(
            input_dim=x_tr.shape[1],
            n_basis=n_basis_ds,
            observation_noise=0.1,
            covariance_type="diagonal",
            knots=knots_ds,
        )
        train_model(gated_ds, x_tr, t_ds, y_tr, n_epochs=n_epochs, n_obs=n_obs)

        static_ds.eval()
        adaptive_ds.eval()
        gated_ds.eval()

        with torch.no_grad():
            t_obs_ds = t_ds[:n_obs]
            y_obs_ds = y_te[:, :n_obs]

            Phis_test = [
                bspline_basis(torch.tensor(ts, dtype=torch.float32), static_ds.knots)
                for ts in ts_te
            ]
            ys_test_t = [torch.tensor(ys, dtype=torch.float32) for ys in ys_te]
            static_preds = static_ds.predict_per_sample(x_te, Phis_test)
            static_mse_persample = float(np.mean([
                nn.functional.mse_loss(pred, y_true).item()
                for pred, y_true in zip(static_preds, ys_test_t, strict=True)
            ]))

            y_static_mean, y_static_var = static_ds(x_te, t_ds)
            static_metrics = compute_all_metrics(
                y_te, y_static_mean, y_static_var, prefix=""
            )
            static_metrics["mse"] = static_mse_persample

            y_adapt_mean, y_adapt_var, _, _ = adaptive_ds.update_and_predict(
                x_te, t_obs_ds, y_obs_ds, t_ds
            )
            adaptive_metrics = compute_all_metrics(
                y_te, y_adapt_mean, y_adapt_var, prefix=""
            )

            y_gated_mean, y_gated_var, _, _ = gated_ds.update_and_predict(
                x_te, t_obs_ds, y_obs_ds, t_ds
            )
            gated_metrics = compute_all_metrics(
                y_te, y_gated_mean, y_gated_var, prefix=""
            )

        if gated_metrics["mse"] < adaptive_metrics["mse"]:
            best_adaptive_metrics = gated_metrics
            best_variant = "gated"
        else:
            best_adaptive_metrics = adaptive_metrics
            best_variant = "standard"

        mse_improvement = (1 - best_adaptive_metrics["mse"] / static_metrics["mse"]) * 100
        crps_improvement = (1 - best_adaptive_metrics["crps"] / static_metrics["crps"]) * 100

        dataset_results[dataset_name] = {
            "static": static_metrics,
            "adaptive": adaptive_metrics,
            "gated": gated_metrics,
            "best_adaptive": best_adaptive_metrics,
            "best_variant": best_variant,
            "mse_improvement": mse_improvement,
            "crps_improvement": crps_improvement,
            "n_samples": len(x_tr) + len(x_va) + len(x_te),
            "n_features": x_tr.shape[1],
        }

        print(
            f"    Static:   MSE={static_metrics['mse']:.4f}, CRPS={static_metrics['crps']:.4f}, "
            f"Coverage95={static_metrics['coverage_95']:.2%}"
        )
        print(
            f"    Standard: MSE={adaptive_metrics['mse']:.4f}, CRPS={adaptive_metrics['crps']:.4f}, "
            f"Coverage95={adaptive_metrics['coverage_95']:.2%}"
        )
        print(
            f"    Gated:    MSE={gated_metrics['mse']:.4f}, CRPS={gated_metrics['crps']:.4f}, "
            f"Coverage95={gated_metrics['coverage_95']:.2%}"
        )
        print(f"    Best: {best_variant} ({mse_improvement:.1f}% MSE, {crps_improvement:.1f}% CRPS)")

    all_results["dataset_comparison"] = dataset_results

    print("\n  --- Summary Table: Static vs Best Adaptive on Real Datasets ---")
    print(f"  {'Dataset':<15} {'Static MSE':>12} {'Best Adap MSE':>14} {'Variant':>10} {'Improvement':>12}")
    print("  " + "-" * 67)
    for ds_name, ds_res in dataset_results.items():
        print(
            f"  {ds_name:<15} {ds_res['static']['mse']:>12.4f} {ds_res['best_adaptive']['mse']:>14.4f} "
            f"{ds_res['best_variant']:>10} {ds_res['mse_improvement']:>11.1f}%"
        )

    print("\n[Bonus] Running streaming evaluation...")
    streaming_results = streaming_evaluation(adaptive_model, x_test[:10], t, y_test[:10])
    all_results["streaming"] = streaming_results

    print("\n[NEW 1/6] Heteroscedastic vs Homoscedastic vs Gated comparison...")
    hetero_results = run_heteroscedastic_comparison(
        x_train, y_train, x_test, y_test, t, n_obs, n_epochs, knots=knots
    )
    all_results["noise_model_comparison"] = hetero_results

    print("\n  Noise model comparison:")
    for method, metrics in hetero_results.items():
        print(
            f"    {method:<20}: MSE={metrics['future_mse']:.4f}, "
            f"Coverage95={metrics['future_coverage_95']:.2%}, "
            f"CalibError={metrics['future_calibration_error']:.4f}, "
            f"Sharpness95={metrics['future_sharpness_95']:.4f}"
        )

    print("\n[NEW 2/6] GP Baseline comparison...")
    gp_results = run_gp_baseline_comparison(
        x_train, y_train, x_test, y_test, t, n_obs, knots=knots,
        ts_train=ts_train_ps, ys_train=ys_train_ps,
        ts_val=ts_val_ps, ys_val=ys_val_ps,
    )
    all_results["gp_comparison"] = gp_results

    print("\n  GP vs TIMEVIEW-Adaptive vs Static:")
    for method, metrics in gp_results.items():
        print(
            f"    {method:<20}: MSE={metrics['future_mse']:.4f}, "
            f"Coverage95={metrics['future_coverage_95']:.2%}, "
            f"CRPS={metrics['future_crps']:.4f}, "
            f"Sharpness95={metrics['future_sharpness_95']:.4f}"
        )

    print("\n[NEW 3/6] Active observation scheduling experiment...")
    active_results = run_active_scheduling_experiment(
        adaptive_model, x_test[:20], y_test[:20], t,
        n_initial_obs=5, n_additional=15,
    )
    all_results["active_scheduling"] = active_results

    if active_results["active_mse"] and active_results["uniform_mse"]:
        final_active_mse = active_results["active_mse"][-1]
        final_uniform_mse = active_results["uniform_mse"][-1]
        improvement = (1 - final_active_mse / final_uniform_mse) * 100
        print("\n  Active scheduling results:")
        print(f"    Final Active MSE:  {final_active_mse:.4f}")
        print(f"    Final Uniform MSE: {final_uniform_mse:.4f}")
        print(f"    Improvement: {improvement:.1f}%")
        print(f"    Selected times: {[f'{t:.3f}' for t in active_results['selected_times'][:5]]}...")

    print("\n[NEW 4/6] Best configuration comparison...")
    best_config_results = run_best_config_experiment(
        x_train, y_train, x_test, y_test, t, n_obs, n_epochs, knots=knots
    )
    all_results["best_config_comparison"] = best_config_results

    print("\n  Best config comparison:")
    for method, metrics in best_config_results.items():
        print(
            f"    {method:<20}: MSE={metrics['future_mse']:.4f}, "
            f"CRPS={metrics['future_crps']:.4f}, "
            f"PE={metrics['future_predictive_efficiency']:.4f}, "
            f"CalibError={metrics['future_calibration_error']:.4f}"
        )

    print("\n[NEW 5/6] Computing Adaptation Efficiency Curves...")
    aec_results = run_aec_experiment(
        x_train, y_train, x_test, y_test, t, n_obs, n_epochs, knots=knots
    )
    all_results["aec_summary"] = {
        name: {"aaec": aec["aaec"], "mse_prior": aec["mse_prior"],
               "final_mse": aec["mse"][-1] if aec["mse"] else None}
        for name, aec in aec_results.items()
    }

    print("\n  AEC results (Area under Adaptation Efficiency Curve):")
    for name, aec in aec_results.items():
        print(f"    {name:<20}: AAEC={aec['aaec']:.4f}, Prior MSE={aec['mse_prior']:.4f}")

    print("\n[NEW 6/6] Computing uncertainty decomposition...")
    decomp_results = run_uncertainty_decomposition_experiment(
        adaptive_model, x_test[:20], y_test[:20], t, n_obs_values=[2, 5, 10, 20]
    )
    all_results["uncertainty_decomposition"] = {
        str(k): {key: val for key, val in v.items() if key != "decomp"}
        for k, v in decomp_results.items()
    }

    print("\n  Uncertainty decomposition:")
    for n_obs_val, res in decomp_results.items():
        pct = res["info_fraction"] * 100
        print(
            f"    n_obs={n_obs_val}: Prior={res['avg_prior_epistemic']:.4f}, "
            f"InfoGained={res['avg_info_gained']:.4f} ({pct:.0f}%), "
            f"Noise={res['avg_noise']:.4f}"
        )

    print("\n" + "=" * 70)
    print("Generating visualizations...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    with torch.no_grad():
        y_static_mean, y_static_var = static_model(x_test, t)
        plot_calibration(y_test, y_static_mean, torch.sqrt(y_static_var), axes[0], "Static")
        axes[0].set_title("Static TIMEVIEW Calibration")

        t_obs_synth = t[:n_obs]
        y_obs_synth = y_test[:, :n_obs]
        y_adapt_mean_synth, y_adapt_var_synth, _, _ = adaptive_model.update_and_predict(
            x_test, t_obs_synth, y_obs_synth, t
        )
        plot_calibration(
            y_test, y_adapt_mean_synth, torch.sqrt(y_adapt_var_synth), axes[1], "Adaptive"
        )
        axes[1].set_title("TIMEVIEW-Adaptive Calibration")

    plt.tight_layout()
    output_path = FIGURES_DIR / "calibration_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")

    plot_streaming_results(streaming_results, "Online Adaptation: Metrics vs Observations")
    plot_transition_analysis(analysis, t, y_test[:5], t[:n_obs], sample_idx=0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    n_basis_vals = list(n_basis_results_aggregated.keys())
    mse_means = [n_basis_results_aggregated[k]["future_mse"]["mean"] for k in n_basis_vals]
    mse_stds = [n_basis_results_aggregated[k]["future_mse"]["std"] for k in n_basis_vals]
    ax.errorbar(
        n_basis_vals, mse_means, yerr=mse_stds, fmt="bo-", capsize=4, capthick=1.5, markersize=6
    )
    ax.fill_between(
        n_basis_vals,
        np.array(mse_means) - np.array(mse_stds),
        np.array(mse_means) + np.array(mse_stds),
        alpha=0.2,
        color="blue",
    )
    ax.set_xlabel("Number of basis functions")
    ax.set_ylabel("MSE (future)")
    ax.set_title("Effect of n_basis")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    n_obs_vals = list(n_obs_results_aggregated.keys())
    mse_means = [n_obs_results_aggregated[k]["future_mse"]["mean"] for k in n_obs_vals]
    mse_stds = [n_obs_results_aggregated[k]["future_mse"]["std"] for k in n_obs_vals]
    ax.errorbar(
        n_obs_vals, mse_means, yerr=mse_stds, fmt="go-", capsize=4, capthick=1.5, markersize=6
    )
    ax.fill_between(
        n_obs_vals,
        np.array(mse_means) - np.array(mse_stds),
        np.array(mse_means) + np.array(mse_stds),
        alpha=0.2,
        color="green",
    )
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("MSE (future)")
    ax.set_title("Effect of n_obs")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    cov_types = list(cov_results_aggregated.keys())
    mse_means = [cov_results_aggregated[k]["future_mse"]["mean"] for k in cov_types]
    mse_stds = [cov_results_aggregated[k]["future_mse"]["std"] for k in cov_types]
    colors = ["blue", "green", "red"]
    bars = ax.bar(cov_types, mse_means, yerr=mse_stds, capsize=4, color=colors, alpha=0.8)
    ax.set_xlabel("Covariance type")
    ax.set_ylabel("MSE (future)")
    ax.set_title("Covariance Parameterization")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = FIGURES_DIR / "ablation_studies.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    fix_names_plot = list(fix_results_aggregated.keys())
    mse_means = [fix_results_aggregated[k]["future_mse"]["mean"] for k in fix_names_plot]
    mse_stds = [fix_results_aggregated[k]["future_mse"]["std"] for k in fix_names_plot]
    x_pos = np.arange(len(fix_names_plot))
    bars = ax.bar(x_pos, mse_means, yerr=mse_stds, capsize=3, alpha=0.8)

    for i, bar in enumerate(bars):
        if fix_names_plot[i] == "baseline":
            bar.set_color("gray")
        elif fix_names_plot[i] == "all_fixes":
            bar.set_color("green")
        else:
            bar.set_color("steelblue")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fix_names_plot, rotation=45, ha="right")
    ax.set_ylabel("MSE (future)")
    ax.set_title("Fix Ablation: Prediction Error")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    cov_means = [fix_results_aggregated[k]["future_coverage_95"]["mean"] for k in fix_names_plot]
    cov_stds = [fix_results_aggregated[k]["future_coverage_95"]["std"] for k in fix_names_plot]
    bars = ax.bar(x_pos, cov_means, yerr=cov_stds, capsize=3, alpha=0.8)
    for i, bar in enumerate(bars):
        if fix_names_plot[i] == "baseline":
            bar.set_color("gray")
        elif fix_names_plot[i] == "all_fixes":
            bar.set_color("green")
        else:
            bar.set_color("steelblue")
    ax.axhline(y=0.95, color="r", linestyle="--", label="Target (95%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fix_names_plot, rotation=45, ha="right")
    ax.set_ylabel("Coverage (95% CI)")
    ax.set_title("Fix Ablation: Calibration")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    kl_weights = list(kl_results_aggregated.keys())
    kl_cov = [kl_results_aggregated[k]["future_coverage_95"]["mean"] for k in kl_weights]
    kl_prior_var = [kl_results_aggregated[k]["prior_avg_var"]["mean"] for k in kl_weights]

    ax2 = ax.twinx()
    line1 = ax.plot(range(len(kl_weights)), kl_cov, "bo-", label="Coverage95", markersize=6)
    line2 = ax2.plot(range(len(kl_weights)), kl_prior_var, "rs-", label="Prior Var", markersize=6)
    ax.axhline(y=0.95, color="b", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(kl_weights)))
    ax.set_xticklabels([str(k) for k in kl_weights])
    ax.set_xlabel("KL Weight")
    ax.set_ylabel("Coverage (95% CI)", color="b")
    ax2.set_ylabel("Prior Variance", color="r")
    ax.set_title("KL Regularization Effect")
    lines = line1 + line2
    labels = [current_line.get_label() for current_line in lines]
    ax.legend(lines, labels, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = FIGURES_DIR / "fix_ablation_studies.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.bar(["Coverage 90%", "Coverage 95%"],
           [temp_results["before_coverage_90"], temp_results["before_coverage_95"]],
           color=["steelblue", "steelblue"], alpha=0.8)
    ax.axhline(y=0.90, color="r", linestyle="--", alpha=0.5, label="Target")
    ax.axhline(y=0.95, color="r", linestyle="--", alpha=0.5)
    ax.set_ylabel("Actual Coverage")
    ax.set_title(f"Before Temperature Scaling\n(CalibError: {temp_results['before_calibration_error']:.4f})")
    ax.set_ylim([0, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(["Coverage 90%", "Coverage 95%"],
           [temp_results["after_coverage_90"], temp_results["after_coverage_95"]],
           color=["green", "green"], alpha=0.8)
    ax.axhline(y=0.90, color="r", linestyle="--", alpha=0.5, label="Target")
    ax.axhline(y=0.95, color="r", linestyle="--", alpha=0.5)
    ax.set_ylabel("Actual Coverage")
    ax.set_title(f"After Temperature Scaling (T={temp_results['temperature']:.3f})\n(CalibError: {temp_results['after_calibration_error']:.4f})")
    ax.set_ylim([0, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = FIGURES_DIR / "temperature_scaling.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")

    if dataset_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ds_names = list(dataset_results.keys())
        x_pos = np.arange(len(ds_names))
        width = 0.35


        ax = axes[0]
        static_mse = [dataset_results[ds]["static"]["mse"] for ds in ds_names]
        adaptive_mse = [dataset_results[ds]["adaptive"]["mse"] for ds in ds_names]
        ax.bar(x_pos - width / 2, static_mse, width, label="Static", color="gray", alpha=0.8)
        ax.bar(x_pos + width / 2, adaptive_mse, width, label="Adaptive", color="blue", alpha=0.8)
        ax.set_ylabel("MSE (future)")
        ax.set_title("Prediction Error: Static vs Adaptive")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([ds.replace("_", "\n") for ds in ds_names])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[1]
        improvements = [dataset_results[ds]["mse_improvement"] for ds in ds_names]
        colors = ["green" if imp > 0 else "red" for imp in improvements]
        bars = ax.bar(ds_names, improvements, color=colors, alpha=0.8)
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        ax.set_ylabel("MSE Improvement (%)")
        ax.set_title("Improvement from Bayesian Adaptation")
        ax.set_xticklabels([ds.replace("_", "\n") for ds in ds_names])
        ax.grid(True, alpha=0.3, axis="y")

        for bar, imp in zip(bars, improvements, strict=False):
            height = bar.get_height()
            ax.annotate(
                f"{imp:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        output_path = FIGURES_DIR / "dataset_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {output_path}")

    if "noise_model_comparison" in all_results:
        plot_heteroscedastic_comparison(all_results["noise_model_comparison"])

    if "gp_comparison" in all_results:
        plot_gp_comparison(all_results["gp_comparison"])

    if "active_scheduling" in all_results and all_results["active_scheduling"]["active_mse"]:
        plot_active_scheduling(all_results["active_scheduling"])

    if "best_config_comparison" in all_results:
        plot_best_config_comparison(all_results["best_config_comparison"])

    if aec_results:
        plot_aec_comparison(aec_results)

    if decomp_results:
        plot_uncertainty_decomposition_grid(decomp_results, t)

    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist() if hasattr(obj, "tolist") else list(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    results_path = TABLES_DIR / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"  Saved results to {results_path}")

    if dataset_results:
        md_table = "# TIMEVIEW-Adaptive: Real Dataset Results\n\n"
        md_table += "## Static vs Adaptive Comparison\n\n"
        md_table += "| Dataset | Method | MSE | CRPS | Coverage (95%) | Calibration Error |\n"
        md_table += "|---------|--------|-----|------|----------------|------------------|\n"
        for ds_name, ds_res in dataset_results.items():
            for method in ["static", "adaptive"]:
                m = ds_res[method]
                md_table += f"| {ds_name} | {method.capitalize()} | "
                md_table += f"{m['mse']:.4f} | {m['crps']:.4f} | {m['coverage_95']:.1%} | {m['calibration_error']:.4f} |\n"
        md_table += "\n## Improvement Summary\n\n"
        md_table += "| Dataset | MSE Improvement | CRPS Improvement |\n"
        md_table += "|---------|-----------------|------------------|\n"
        for ds_name, ds_res in dataset_results.items():
            md_table += f"| {ds_name} | {ds_res['mse_improvement']:.1f}% | {ds_res['crps_improvement']:.1f}% |\n"

        md_path = TABLES_DIR / "dataset_comparison.md"
        with open(md_path, "w") as f:
            f.write(md_table)
        print(f"  Saved Markdown table to {md_path}")

    print("\n" + "=" * 70)
    print("All experiments completed!")
    print("=" * 70)

    return all_results


def run_baseline_comparison(seed: int = 0, n_epochs: int = 1000, n_tune: int = 100, n_trials: int = 3):
    datasets = ["airfoil", "flchain", "stress_strain"]

    print("=" * 70)
    print("TIMEVIEW Baseline Comparison (per-sample pipeline)")
    print("Validating matched pipeline against TIMEVIEW paper")
    print("=" * 70)

    results = {}
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

        n_total = len(x_tr) + len(x_va) + len(x_te)
        print(f"  Samples: {n_total} (train={len(x_tr)}, val={len(x_va)}, test={len(x_te)})")
        print(f"  Features: {x_tr.shape[1]}, Time points: {len(t_ds)}")
        print(f"  T range: [{t_ds[0].item():.3f}, {t_ds[-1].item():.3f}]")
        if knots_ds is not None:
            print(f"  Knots: {knots_ds.shape[0]} total")

        pts_per_sample = [len(ts) for ts in ts_tr]
        print(f"  Points per sample: min={min(pts_per_sample)}, max={max(pts_per_sample)}, "
              f"mean={np.mean(pts_per_sample):.1f}")

        print(f"  Tuning hyperparameters (n_basis={n_basis_ds}, {n_tune} trials, {n_epochs} epochs)...")
        best_params = tune_static_hyperparams(
            input_dim=x_tr.shape[1], n_basis=n_basis_ds, knots=knots_ds,
            x_train=x_tr, t=t_ds, y_train=y_tr,
            x_val=x_va, y_val=y_va,
            ts_train=ts_tr, ys_train=ys_tr,
            ts_val=ts_va, ys_val=ys_va,
            n_tune=n_tune, n_epochs=n_epochs, seed=seed,
        )

        print(f"  Training {n_trials} models with different seeds...")
        trial_mses_persample = []
        trial_mses_grid = []
        best_mse = float("inf")
        for trial_i in range(n_trials):
            trial_seed = seed + trial_i
            model_i = train_static_with_params(
                best_params,
                input_dim=x_tr.shape[1], n_basis=n_basis_ds, knots=knots_ds,
                x_train=x_tr, t=t_ds, y_train=y_tr,
                x_val=x_va, y_val=y_va,
                ts_train=ts_tr, ys_train=ys_tr,
                ts_val=ts_va, ys_val=ys_va,
                n_epochs=n_epochs, seed=trial_seed,
            )

            model_i.eval()
            with torch.no_grad():
                Phis_test = [
                    bspline_basis(torch.tensor(ts, dtype=torch.float32), model_i.knots)
                    for ts in ts_te
                ]
                ys_test_t = [torch.tensor(ys, dtype=torch.float32) for ys in ys_te]
                preds = model_i.predict_per_sample(x_te, Phis_test)
                per_sample_mse = [
                    nn.functional.mse_loss(pred, y_true).item()
                    for pred, y_true in zip(preds, ys_test_t, strict=True)
                ]
                mse_i = float(np.mean(per_sample_mse))

            with torch.no_grad():
                y_pred_grid, _ = model_i(x_te, t_ds)
                mse_grid_i = ((y_te - y_pred_grid) ** 2).mean().item()

            trial_mses_persample.append(mse_i)
            trial_mses_grid.append(mse_grid_i)
            print(f"    Trial {trial_i+1}/{n_trials} (seed={trial_seed}): "
                  f"per-sample MSE={mse_i:.6f}, grid MSE={mse_grid_i:.6f}")

            if mse_i < best_mse:
                best_mse = mse_i

        mse_norm = float(np.mean(trial_mses_persample))
        mse_std = float(np.std(trial_mses_persample))
        mse_grid = float(np.mean(trial_mses_grid))
        print(f"  Average per-sample MSE: {mse_norm:.6f} ± {mse_std:.6f}")

        if y_norm is not None:
            mse_orig = mse_norm * (y_norm.y_std ** 2)
        else:
            mse_orig = mse_norm

        results[dataset_name] = {
            "mse_normalized": mse_norm,
            "mse_std": mse_std,
            "mse_grid": mse_grid,
            "mse_original": mse_orig,
            "n_train": len(x_tr),
            "n_val": len(x_va),
            "n_test": len(x_te),
            "n_features": x_tr.shape[1],
        }

        print(f"  MSE per-sample (normalized y): {mse_norm:.6f} ± {mse_std:.6f} ({n_trials} seeds)")
        print(f"  MSE grid      (normalized y): {mse_grid:.6f}")
        print(f"  MSE per-sample (original y):  {mse_orig:.6f}")

        print(f"  Training TimeviewAdaptive (n_basis={n_basis_ds}, {n_epochs} epochs)...")
        torch.manual_seed(seed)
        adaptive_model = TimeviewAdaptive(
            input_dim=x_tr.shape[1], n_basis=n_basis_ds,
            observation_noise=0.1, covariance_type="diagonal",
            knots=knots_ds,
        )
        train_model(adaptive_model, x_tr, t_ds, y_tr, n_epochs=n_epochs, n_obs=10)

        adaptive_model.eval()
        with torch.no_grad():
            n_obs = 10
            t_obs = t_ds[:n_obs]
            y_obs = y_te[:, :n_obs]
            y_adapt_mean, y_adapt_var, _, _ = adaptive_model.update_and_predict(
                x_te, t_obs, y_obs, t_ds
            )
            mse_adapt = ((y_te[:, n_obs:] - y_adapt_mean[:, n_obs:]) ** 2).mean().item()

        improvement = (1 - mse_adapt / mse_grid) * 100
        results[dataset_name]["mse_adaptive"] = mse_adapt
        results[dataset_name]["improvement_pct"] = improvement

        print(f"  MSE adaptive (10 obs, future): {mse_adapt:.6f}")
        print(f"  Improvement over static grid: {improvement:.1f}%")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Dataset':<15} {'Per-sample MSE':>20} {'Grid MSE':>10} {'Adaptive MSE':>14} {'Improvement':>12}")
    print("-" * 74)
    for ds_name, ds_res in results.items():
        mse_str = f"{ds_res['mse_normalized']:.4f}±{ds_res['mse_std']:.4f}"
        print(
            f"{ds_name:<15} {mse_str:>20} "
            f"{ds_res['mse_grid']:>10.6f} "
            f"{ds_res.get('mse_adaptive', float('nan')):>14.6f} "
            f"{ds_res.get('improvement_pct', 0):>11.1f}%"
        )

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "baseline":
        run_baseline_comparison()
    elif len(sys.argv) > 1 and sys.argv[1] == "nobs":
        run_nobs_ablation_all_datasets()
    else:
        results = run_all_experiments(seed=0, n_epochs=1000)
