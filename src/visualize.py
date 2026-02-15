
import matplotlib.pyplot as plt
import numpy as np
import torch

from constant import FIGURES_DIR, PAPER_FIGURES
from timeview_adaptive import (
    ActiveObservationScheduler,
    TimeviewAdaptive,
    TimeviewAdaptiveGated,
    TimeviewAdaptiveGatedHeteroscedastic,
    TimeviewAdaptiveHeteroscedastic,
    bspline_basis,
    create_knots,
    train_model,
)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

SEED = 42


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def generate_synthetic_data(n_samples=100, n_timepoints=50, input_dim=4, seed=SEED):
    set_seed(seed)
    t = torch.linspace(0, 1, n_timepoints)

    x = torch.randn(n_samples, input_dim)
    y = torch.zeros(n_samples, n_timepoints)

    for i in range(n_samples):
        amp1 = 0.5 + 0.3 * x[i, 0].item()
        amp2 = 0.3 + 0.2 * x[i, 1].item()
        freq = 2.0 + 0.5 * x[i, 2].item()
        trend = 0.5 * x[i, 3].item()
        noise = 0.05 * torch.randn(n_timepoints)
        y[i] = amp1 * torch.sin(2 * np.pi * freq * t) + amp2 * t + trend * t**2 + noise

    return x, t, y


def train_demo_model(x, t, y, model_class=TimeviewAdaptive, n_basis=7, **kwargs):
    input_dim = x.shape[1]
    model = model_class(input_dim=input_dim, n_basis=n_basis, **kwargs)
    train_model(model, x, t, y, n_epochs=100, lr=1e-3, n_obs=10, verbose=False)
    return model


def plot_bspline_basis(save=True):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    t = torch.linspace(0, 1, 200)

    for ax, nb in zip(axes, [5, 7, 9], strict=True):
        knots = create_knots(nb)
        Phi = bspline_basis(t, knots)

        for j in range(nb):
            ax.plot(t.numpy(), Phi[:, j].numpy(), linewidth=1.5, label=f"$\\phi_{{{j+1}}}$")
        ax.set_title(f"$B = {nb}$ basis functions")
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("$\\phi_j(t)$")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7, ncol=2, loc="upper right")

    fig.suptitle("Cubic B-Spline Basis Functions", fontsize=13, y=1.02)
    plt.tight_layout()
    if save:
        for d in [PAPER_FIGURES, FIGURES_DIR]:
            fig.savefig(d / "bspline_basis.png")
    plt.close()
    print("  [ok] bspline_basis.png")


def plot_adaptation_demo(save=True):
    set_seed()
    x, t, y = generate_synthetic_data(n_samples=50, input_dim=4)

    model = train_demo_model(x, t, y, n_basis=7, covariance_type="low_rank")
    model.eval()

    idx = 5
    x_i = x[idx : idx + 1]
    y_i = y[idx]

    with torch.no_grad():
        mu_0, Sigma_0 = model.encode(x_i)

        y_prior_mean, y_prior_var = model.predict(mu_0, Sigma_0, t)

        obs_counts = [0, 3, 10, 25]
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)

        for ax, n_obs in zip(axes, obs_counts, strict=True):
            if n_obs == 0:
                mean = y_prior_mean[0].numpy()
                var = y_prior_var[0].numpy()
                title = "Prior only (0 obs)"
            else:
                t_obs = t[:n_obs]
                y_obs = y_i[:n_obs].unsqueeze(0)
                y_mean, y_var, _, _ = model.update_and_predict(x_i, t_obs, y_obs, t)
                mean = y_mean[0].numpy()
                var = y_var[0].numpy()
                title = f"Posterior ({n_obs} obs)"

            std = np.sqrt(var)
            t_np = t.numpy()

            ax.plot(t_np, y_i.numpy(), "k-", alpha=0.3, linewidth=1, label="Ground truth")
            ax.plot(t_np, mean, "b-", linewidth=1.5, label="Prediction")
            ax.fill_between(t_np, mean - 1.96 * std, mean + 1.96 * std,
                            alpha=0.2, color="blue", label="95% CI")

            if n_obs > 0:
                ax.scatter(t[:n_obs].numpy(), y_i[:n_obs].numpy(),
                           c="red", s=30, zorder=5, label="Observations")

            ax.set_title(title)
            ax.set_xlabel("Time $t$")
            if ax == axes[0]:
                ax.set_ylabel("$y(t)$")
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Bayesian Adaptation: Prior → Posterior as Observations Arrive", fontsize=13, y=1.02)
    plt.tight_layout()
    if save:
        for d in [PAPER_FIGURES, FIGURES_DIR]:
            fig.savefig(d / "adaptation_demo.png")
    plt.close()
    print("  [ok] adaptation_demo.png")


def plot_coefficient_evolution(save=True):
    set_seed()
    x, t, y = generate_synthetic_data(n_samples=50, input_dim=4)
    model = train_demo_model(x, t, y, n_basis=7, covariance_type="low_rank")
    model.eval()

    idx = 5
    x_i = x[idx : idx + 1]
    y_i = y[idx]

    with torch.no_grad():
        mu_0, Sigma_0 = model.encode(x_i)

        obs_counts = [0, 2, 5, 15]
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)

        for ax, n_obs in zip(axes, obs_counts, strict=True):
            if n_obs == 0:
                mu = mu_0[0].numpy()
                std = np.sqrt(np.diag(Sigma_0[0].numpy()))
            else:
                t_obs = t[:n_obs]
                y_obs = y_i[:n_obs].unsqueeze(0)
                _, _, mu_post, Sigma_post = model.update_and_predict(x_i, t_obs, y_obs, t)
                mu = mu_post[0].numpy()
                std = np.sqrt(np.diag(Sigma_post[0].numpy()))

            n_basis = len(mu)
            x_pos = np.arange(n_basis)
            ax.bar(x_pos, mu, yerr=1.96 * std, capsize=4, color="steelblue",
                   edgecolor="navy", alpha=0.7)
            ax.set_title(f"{n_obs} observations")
            ax.set_xlabel("Basis index $j$")
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"$c_{{{j+1}}}$" for j in range(n_basis)])
            if ax == axes[0]:
                ax.set_ylabel("Coefficient value")

    fig.suptitle("Coefficient Distributions: Uncertainty Narrows with More Observations", fontsize=13, y=1.02)
    plt.tight_layout()
    if save:
        for d in [PAPER_FIGURES, FIGURES_DIR]:
            fig.savefig(d / "coefficient_evolution.png")
    plt.close()
    print("  [ok] coefficient_evolution.png")


def plot_uncertainty_decomposition(save=True):
    set_seed()
    x, t, y = generate_synthetic_data(n_samples=50, input_dim=4)
    model = train_demo_model(x, t, y, n_basis=7, covariance_type="low_rank")
    model.eval()

    idx = 5
    x_i = x[idx : idx + 1]
    y_i = y[idx]

    obs_counts = [2, 5, 10, 20]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    with torch.no_grad():
        mu_0, Sigma_0 = model.encode(x_i)
        Phi = model.get_basis(t)

        PhiSigma0 = torch.matmul(Phi.unsqueeze(0), Sigma_0)
        prior_var = torch.sum(PhiSigma0 * Phi.unsqueeze(0), dim=-1)[0].numpy()

        sigma2 = model.sigma.item() ** 2
        noise_var = np.full_like(prior_var, sigma2)

        for ax, n_obs in zip(axes, obs_counts, strict=True):
            t_obs = t[:n_obs]
            y_obs = y_i[:n_obs].unsqueeze(0)
            _, _, mu_post, Sigma_post = model.update_and_predict(x_i, t_obs, y_obs, t)

            PhiSigmaN = torch.matmul(Phi.unsqueeze(0), Sigma_post)
            post_var = torch.sum(PhiSigmaN * Phi.unsqueeze(0), dim=-1)[0].numpy()

            info_gained = prior_var - post_var
            remaining_epistemic = post_var

            t_np = t.numpy()
            pct_resolved = info_gained.sum() / prior_var.sum() * 100

            ax.fill_between(t_np, 0, noise_var, alpha=0.4, color="gray", label="Noise (aleatoric)")
            ax.fill_between(t_np, noise_var, noise_var + remaining_epistemic,
                            alpha=0.4, color="steelblue", label="Remaining epistemic")
            ax.fill_between(t_np, noise_var + remaining_epistemic,
                            noise_var + remaining_epistemic + info_gained,
                            alpha=0.4, color="green", label="Info gained")

            for ti in t_obs.numpy():
                ax.axvline(ti, color="red", alpha=0.3, linewidth=0.5)

            ax.set_title(f"$n_{{obs}}={n_obs}$\nInfo gained: {pct_resolved:.0f}% of prior")
            ax.set_xlabel("Time $t$")
            if ax == axes[0]:
                ax.set_ylabel("Variance")
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Uncertainty Decomposition vs Number of Observations", fontsize=13, y=1.02)
    plt.tight_layout()
    if save:
        for d in [PAPER_FIGURES, FIGURES_DIR]:
            fig.savefig(d / "uncertainty_decomposition_viz.png")
    plt.close()
    print("  [ok] uncertainty_decomposition_viz.png")


def plot_active_scheduling(save=True):
    set_seed()
    x, t, y = generate_synthetic_data(n_samples=50, input_dim=4)
    model = train_demo_model(x, t, y, n_basis=7, covariance_type="low_rank")
    model.eval()

    scheduler = ActiveObservationScheduler(model)

    n_test = 20
    x_test = x[:n_test]
    y_test = y[:n_test]

    n_initial = 3
    n_steps = 12

    active_mses = []
    uniform_mses = []

    with torch.no_grad():
        for step in range(n_steps + 1):
            n_obs = n_initial + step
            t_obs_uniform = t[:n_obs]
            y_obs_uniform = y_test[:, :n_obs]

            y_pred_u, y_var_u, _, _ = model.update_and_predict(x_test, t_obs_uniform, y_obs_uniform, t)
            remaining_mask = torch.ones(len(t), dtype=torch.bool)
            remaining_mask[:n_obs] = False
            mse_uniform = ((y_pred_u[:, remaining_mask] - y_test[:, remaining_mask]) ** 2).mean().item()
            uniform_mses.append(mse_uniform)

            if step == 0:
                active_obs_idx = list(range(n_initial))
            else:
                t_active_obs = t[active_obs_idx]
                y_active_obs = y_test[:, active_obs_idx]

                all_idx = set(range(len(t)))
                cand_idx = sorted(all_idx - set(active_obs_idx))
                t_candidates = t[cand_idx]

                _, ig = scheduler.suggest_next_observation(x_test, t_active_obs, y_active_obs, t_candidates)
                best_cand = ig.mean(dim=0).argmax().item()
                active_obs_idx.append(cand_idx[best_cand])

            t_active_obs = t[active_obs_idx]
            y_active_obs = y_test[:, active_obs_idx]
            y_pred_a, _, _, _ = model.update_and_predict(x_test, t_active_obs, y_active_obs, t)
            remaining_active = torch.ones(len(t), dtype=torch.bool)
            remaining_active[active_obs_idx] = False
            mse_active = ((y_pred_a[:, remaining_active] - y_test[:, remaining_active]) ** 2).mean().item()
            active_mses.append(mse_active)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    steps = range(n_initial, n_initial + n_steps + 1)
    ax.plot(steps, active_mses, "o-", color="green", label="Active (IG)", markersize=5)
    ax.plot(steps, uniform_mses, "s-", color="orange", label="Uniform", markersize=5)
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("MSE (on remaining points)")
    ax.set_title("Active vs Uniform Observation Scheduling")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        for d in [PAPER_FIGURES, FIGURES_DIR]:
            fig.savefig(d / "active_vs_uniform.png")
    plt.close()
    print("  [ok] active_vs_uniform.png")


def plot_gating_demo(save=True):
    set_seed()
    x, t, y = generate_synthetic_data(n_samples=80, input_dim=4)
    model = train_demo_model(x, t, y, model_class=TimeviewAdaptiveGated,
                             n_basis=7, covariance_type="low_rank")
    model.eval()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    with torch.no_grad():
        samples = [0, 10, 20]
        for ax, idx in zip(axes, samples, strict=True):
            x_i = x[idx : idx + 1]
            y_i = y[idx]

            mu_0, Sigma_0 = model.encode(x_i)
            y_prior, _ = model.predict(mu_0, Sigma_0, t)

            n_obs = 10
            t_obs = t[:n_obs]
            y_obs = y_i[:n_obs].unsqueeze(0)
            y_gated, y_var, _, _ = model.update_and_predict(x_i, t_obs, y_obs, t)

            y_pr_obs, y_pr_var_obs = model.predict(mu_0, Sigma_0, t_obs)
            prior_mse = ((y_obs - y_pr_obs) ** 2).mean(dim=1, keepdim=True)
            obs_unc = y_pr_var_obs.mean(dim=1, keepdim=True)
            gate_val = model.gate(x_i, prior_mse, obs_unc).item()

            t_np = t.numpy()
            std = np.sqrt(y_var[0].numpy())
            ax.plot(t_np, y_i.numpy(), "k-", alpha=0.3, linewidth=1, label="Ground truth")
            ax.plot(t_np, y_prior[0].numpy(), "--", color="orange", linewidth=1, label="Prior")
            ax.plot(t_np, y_gated[0].numpy(), "-", color="blue", linewidth=1.5, label="Gated")
            ax.fill_between(t_np, y_gated[0].numpy() - 1.96 * std,
                            y_gated[0].numpy() + 1.96 * std,
                            alpha=0.15, color="blue")
            ax.scatter(t[:n_obs].numpy(), y_i[:n_obs].numpy(),
                       c="red", s=25, zorder=5, label="Observations")
            ax.set_title(f"Sample {idx+1}: gate $\\alpha$ = {gate_val:.2f}")
            ax.set_xlabel("Time $t$")
            if ax == axes[0]:
                ax.set_ylabel("$y(t)$")
                ax.legend(fontsize=7)

    fig.suptitle("Adaptive Gating: Blending Prior and Posterior Predictions", fontsize=13, y=1.02)
    plt.tight_layout()
    if save:
        for d in [PAPER_FIGURES, FIGURES_DIR]:
            fig.savefig(d / "gating_demo.png")
    plt.close()
    print("  [ok] gating_demo.png")


def plot_model_comparison(save=True):
    set_seed()
    x, t, y = generate_synthetic_data(n_samples=80, input_dim=4)

    models = {
        "Standard": train_demo_model(x, t, y, TimeviewAdaptive, n_basis=7),
        "Heteroscedastic": train_demo_model(x, t, y, TimeviewAdaptiveHeteroscedastic, n_basis=7),
        "Gated": train_demo_model(x, t, y, TimeviewAdaptiveGated, n_basis=7, covariance_type="low_rank"),
        "Best Config": train_demo_model(x, t, y, TimeviewAdaptiveGatedHeteroscedastic,
                                        n_basis=9, covariance_type="low_rank"),
    }

    idx = 7
    x_i = x[idx : idx + 1]
    y_i = y[idx]
    n_obs = 10

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for ax, (name, model), color in zip(axes, models.items(), colors, strict=True):
        model.eval()
        with torch.no_grad():
            y_pred, y_var, _, _ = model.update_and_predict(
                x_i, t[:n_obs], y_i[:n_obs].unsqueeze(0), t
            )
            mean = y_pred[0].numpy()
            std = np.sqrt(y_var[0].numpy())
            mse = ((y_pred[0] - y_i) ** 2).mean().item()

        t_np = t.numpy()
        ax.plot(t_np, y_i.numpy(), "k-", alpha=0.3, linewidth=1, label="Truth")
        ax.plot(t_np, mean, "-", color=color, linewidth=1.5, label="Prediction")
        ax.fill_between(t_np, mean - 1.96 * std, mean + 1.96 * std,
                        alpha=0.2, color=color, label="95% CI")
        ax.scatter(t[:n_obs].numpy(), y_i[:n_obs].numpy(),
                   c="red", s=25, zorder=5)
        ax.set_title(f"{name}\nMSE = {mse:.4f}")
        ax.set_xlabel("Time $t$")
        if ax == axes[0]:
            ax.set_ylabel("$y(t)$")
            ax.legend(fontsize=7)

    fig.suptitle("Model Variant Comparison (10 observations)", fontsize=13, y=1.02)
    plt.tight_layout()
    if save:
        for d in [PAPER_FIGURES, FIGURES_DIR]:
            fig.savefig(d / "model_comparison.png")
    plt.close()
    print("  [ok] model_comparison.png")


def plot_information_gain_landscape(save=True):
    set_seed()
    x, t, y = generate_synthetic_data(n_samples=50, input_dim=4)
    model = train_demo_model(x, t, y, n_basis=7, covariance_type="low_rank")
    model.eval()

    scheduler = ActiveObservationScheduler(model)

    idx = 5
    x_i = x[idx : idx + 1]
    y_i = y[idx]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    obs_counts = [1, 3, 5, 8, 12, 20]

    with torch.no_grad():
        for ax, n_obs in zip(axes.flat, obs_counts, strict=True):
            t_obs = t[:n_obs]
            y_obs = y_i[:n_obs].unsqueeze(0)
            _, _, mu_post, Sigma_post = model.update_and_predict(x_i, t_obs, y_obs, t)

            ig = scheduler.compute_information_gain(mu_post, Sigma_post, t)
            ig_np = ig[0].numpy()

            _, y_var = model.predict(mu_post, Sigma_post, t)
            std = np.sqrt(y_var[0].numpy())

            t_np = t.numpy()
            ax2 = ax.twinx()
            ax.bar(t_np, ig_np, width=0.018, alpha=0.6, color="green", label="Info gain")
            ax2.plot(t_np, std, "b-", alpha=0.7, linewidth=1, label="Pred. std")

            for ti in t_obs.numpy():
                ax.axvline(ti, color="red", alpha=0.4, linewidth=0.8, linestyle="--")

            ax.set_title(f"{n_obs} observations")
            ax.set_xlabel("Time $t$")
            ax.set_ylabel("Info gain", color="green")
            ax2.set_ylabel("Pred. std", color="blue")

            if n_obs == 1:
                ax.legend(fontsize=7, loc="upper left")
                ax2.legend(fontsize=7, loc="upper right")

    fig.suptitle("Information Gain Landscape Evolution", fontsize=13, y=1.02)
    plt.tight_layout()
    if save:
        for d in [PAPER_FIGURES, FIGURES_DIR]:
            fig.savefig(d / "ig_landscape.png")
    plt.close()
    print("  [ok] ig_landscape.png")


def main():
    print("TIMEVIEW-Adaptive Visualization Program")
    print("=" * 50)
    print()

    print("[1/8] B-spline basis functions...")
    plot_bspline_basis()

    print("[2/8] Adaptation demo (prior → posterior)...")
    plot_adaptation_demo()

    print("[3/8] Coefficient distribution evolution...")
    plot_coefficient_evolution()

    print("[4/8] Uncertainty decomposition...")
    plot_uncertainty_decomposition()

    print("[5/8] Active vs uniform scheduling...")
    plot_active_scheduling()

    print("[6/8] Gating mechanism demo...")
    plot_gating_demo()

    print("[7/8] Model variant comparison...")
    plot_model_comparison()

    print("[8/8] Information gain landscape...")
    plot_information_gain_landscape()

    print()
    print("All figures saved to:")
    print(f"  {PAPER_FIGURES}")
    print(f"  {FIGURES_DIR}")


if __name__ == "__main__":
    main()
