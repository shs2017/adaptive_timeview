"""
TIMEVIEW-Adaptive GIF Animation Generator

Generates 4 supplemental animations for the paper:
1. adaptation_progression.gif - Bayesian adaptation as observations arrive
2. active_vs_uniform.gif - Side-by-side active vs uniform scheduling
3. coefficient_evolution.gif - Coefficient posterior concentration
4. ig_landscape_evolution.gif - Information gain landscape shifting

Run from src/ directory:
    python generate_animations.py

Output: ../figures/animations/
"""

from io import BytesIO
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from timeview_adaptive import (
    ActiveObservationScheduler,
    TimeviewAdaptive,
    train_model,
)

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
ANIMATIONS_DIR = PROJECT_ROOT / "figures" / "animations"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
ANIMATIONS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Plotting style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 100,
    "savefig.dpi": 150,
})

SEED = 42


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def generate_synthetic_data(n_samples=100, n_timepoints=50, input_dim=4, seed=SEED):
    """Generate synthetic data for animations."""
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


def get_model(x, t, y):
    """Load or train model, caching to checkpoint."""
    ckpt_path = CHECKPOINT_DIR / "animation_model.pt"
    input_dim = x.shape[1]
    model = TimeviewAdaptive(input_dim=input_dim, n_basis=7, covariance_type="low_rank")

    if ckpt_path.exists():
        print("  Loading cached model...")
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    else:
        print("  Training model...")
        train_model(model, x, t, y, n_epochs=100, lr=1e-3, n_obs=10, verbose=False)
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint to {ckpt_path}")

    model.eval()
    return model


def render_frame(fig):
    """Convert matplotlib figure to PIL Image with fixed canvas size."""
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def save_gif(frames, path, fps=2):
    """Save list of PIL Images as GIF, resizing to consistent dimensions."""
    # Resize all frames to match the first frame
    target_size = frames[0].size
    resized = []
    for f in frames:
        if f.size != target_size:
            f = f.resize(target_size, Image.LANCZOS)
        resized.append(np.array(f))
    duration = int(1000 / fps)  # ms per frame
    iio.imwrite(path, resized, duration=duration, loop=0)
    print(f"  Saved {path} ({len(frames)} frames, {fps} fps)")


# =============================================================================
# Animation 1: Adaptation Progression
# =============================================================================

def generate_adaptation_progression(model, x, t, y):
    """25 frames showing observation-by-observation adaptation."""
    print("\n[1/4] Generating adaptation_progression.gif...")

    idx = 5
    x_i = x[idx : idx + 1]
    y_i = y[idx]
    n_frames = 25
    t_np = t.numpy()
    y_true = y_i.numpy()

    frames = []
    with torch.no_grad():
        mu_0, Sigma_0 = model.encode(x_i)

        for frame in range(n_frames):
            n_obs = frame  # 0 to 24

            fig, ax = plt.subplots(figsize=(8, 5))

            if n_obs == 0:
                y_mean, y_var = model.predict(mu_0, Sigma_0, t)
                mean = y_mean[0].numpy()
                var = y_var[0].numpy()
            else:
                t_obs = t[:n_obs]
                y_obs = y_i[:n_obs].unsqueeze(0)
                y_mean, y_var, _, _ = model.update_and_predict(x_i, t_obs, y_obs, t)
                mean = y_mean[0].numpy()
                var = y_var[0].numpy()

            std = np.sqrt(var)

            # Ground truth
            ax.plot(t_np, y_true, "k--", alpha=0.4, linewidth=1.5, label="Ground truth")
            # Prediction
            ax.plot(t_np, mean, "b-", linewidth=2, label="Prediction")
            # CI
            ax.fill_between(t_np, mean - 1.96 * std, mean + 1.96 * std,
                            alpha=0.2, color="blue", label="95% CI")
            # Observations
            if n_obs > 0:
                ax.scatter(t[:n_obs].numpy(), y_i[:n_obs].numpy(),
                           c="red", s=50, zorder=5, edgecolors="darkred", label="Observations")

            ax.set_xlabel("Time $t$")
            ax.set_ylabel("$y(t)$")
            ax.set_title(f"Bayesian Adaptation — {n_obs} observation{'s' if n_obs != 1 else ''}")
            ax.legend(loc="upper right", fontsize=9)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.2)

            frames.append(render_frame(fig))
            plt.close(fig)

    # Hold last frame for 2 extra seconds
    for _ in range(4):
        frames.append(frames[-1])

    save_gif(frames, ANIMATIONS_DIR / "adaptation_progression.gif", fps=2)


# =============================================================================
# Animation 2: Active vs Uniform Scheduling
# =============================================================================

def generate_active_vs_uniform(model, x, t, y):
    """15 frames comparing active vs uniform scheduling side by side."""
    print("\n[2/4] Generating active_vs_uniform.gif...")

    idx = 5
    x_i = x[idx : idx + 1]
    y_i = y[idx]
    t_np = t.numpy()
    y_true = y_i.numpy()

    scheduler = ActiveObservationScheduler(model)
    n_initial = 3
    n_steps = 12

    # Pre-compute active observation sequence
    active_obs_idx = list(range(n_initial))
    active_sequence = [list(active_obs_idx)]

    with torch.no_grad():
        for _ in range(n_steps):
            t_active_obs = t[active_obs_idx]
            y_active_obs = y_i[active_obs_idx].unsqueeze(0)

            all_idx = set(range(len(t)))
            cand_idx = sorted(all_idx - set(active_obs_idx))
            t_candidates = t[cand_idx]

            _, ig = scheduler.suggest_next_observation(x_i, t_active_obs, y_active_obs, t_candidates)
            best_cand = ig[0].argmax().item()
            active_obs_idx.append(cand_idx[best_cand])
            active_sequence.append(list(active_obs_idx))

    # Generate frames
    frames = []
    with torch.no_grad():
        mu_0, Sigma_0 = model.encode(x_i)

        for frame_idx in range(n_steps + 1):
            n_obs = n_initial + frame_idx

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

            # Active scheduling (left)
            a_idx = active_sequence[frame_idx]
            t_active = t[a_idx]
            y_active = y_i[a_idx].unsqueeze(0)
            y_mean_a, y_var_a, _, _ = model.update_and_predict(x_i, t_active, y_active, t)
            mean_a = y_mean_a[0].numpy()
            std_a = np.sqrt(y_var_a[0].numpy())

            ax1.plot(t_np, y_true, "k--", alpha=0.4, linewidth=1.5, label="Ground truth")
            ax1.plot(t_np, mean_a, "g-", linewidth=2, label="Prediction")
            ax1.fill_between(t_np, mean_a - 1.96 * std_a, mean_a + 1.96 * std_a,
                             alpha=0.2, color="green")
            ax1.scatter(t_active.numpy(), y_i[a_idx].numpy(),
                        c="red", s=50, zorder=5, edgecolors="darkred", label="Observations")

            remaining_a = torch.ones(len(t), dtype=torch.bool)
            remaining_a[a_idx] = False
            mse_a = ((y_mean_a[0, remaining_a] - y_i[remaining_a]) ** 2).mean().item()
            ax1.set_title(f"Active Scheduling ({n_obs} obs)\nMSE = {mse_a:.4f}")
            ax1.set_xlabel("Time $t$")
            ax1.set_ylabel("$y(t)$")
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.2)

            # Uniform scheduling (right)
            t_uniform = t[:n_obs]
            y_uniform = y_i[:n_obs].unsqueeze(0)
            y_mean_u, y_var_u, _, _ = model.update_and_predict(x_i, t_uniform, y_uniform, t)
            mean_u = y_mean_u[0].numpy()
            std_u = np.sqrt(y_var_u[0].numpy())

            ax2.plot(t_np, y_true, "k--", alpha=0.4, linewidth=1.5, label="Ground truth")
            ax2.plot(t_np, mean_u, color="orange", linewidth=2, label="Prediction")
            ax2.fill_between(t_np, mean_u - 1.96 * std_u, mean_u + 1.96 * std_u,
                             alpha=0.2, color="orange")
            ax2.scatter(t_uniform.numpy(), y_i[:n_obs].numpy(),
                        c="red", s=50, zorder=5, edgecolors="darkred", label="Observations")

            remaining_u = torch.ones(len(t), dtype=torch.bool)
            remaining_u[:n_obs] = False
            mse_u = ((y_mean_u[0, remaining_u] - y_i[remaining_u]) ** 2).mean().item()
            ax2.set_title(f"Uniform Scheduling ({n_obs} obs)\nMSE = {mse_u:.4f}")
            ax2.set_xlabel("Time $t$")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.2)

            fig.suptitle("Active vs Uniform Observation Scheduling", fontsize=14, y=1.02)
            plt.tight_layout()

            frames.append(render_frame(fig))
            plt.close(fig)

    # Hold last frame
    for _ in range(4):
        frames.append(frames[-1])

    save_gif(frames, ANIMATIONS_DIR / "active_vs_uniform.gif", fps=2)


# =============================================================================
# Animation 3: Coefficient Evolution
# =============================================================================

def generate_coefficient_evolution(model, x, t, y):
    """20 frames showing coefficient posterior concentration."""
    print("\n[3/4] Generating coefficient_evolution.gif...")

    idx = 5
    x_i = x[idx : idx + 1]
    y_i = y[idx]
    n_frames = 20

    frames = []
    with torch.no_grad():
        mu_0, Sigma_0 = model.encode(x_i)
        n_basis = mu_0.shape[1]
        x_pos = np.arange(n_basis)

        # Get consistent y-limits from prior
        mu_prior = mu_0[0].numpy()
        std_prior = np.sqrt(np.diag(Sigma_0[0].numpy()))
        y_min = (mu_prior - 2.5 * std_prior).min()
        y_max = (mu_prior + 2.5 * std_prior).max()

        for frame in range(n_frames):
            n_obs = frame  # 0 to 19

            fig, ax = plt.subplots(figsize=(8, 5))

            if n_obs == 0:
                mu = mu_0[0].numpy()
                std = np.sqrt(np.diag(Sigma_0[0].numpy()))
            else:
                t_obs = t[:n_obs]
                y_obs = y_i[:n_obs].unsqueeze(0)
                _, _, mu_post, Sigma_post = model.update_and_predict(x_i, t_obs, y_obs, t)
                mu = mu_post[0].numpy()
                std = np.sqrt(np.diag(Sigma_post[0].numpy()))

            colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_basis))
            ax.bar(x_pos, mu, yerr=1.96 * std, capsize=6, color=colors,
                   edgecolor="black", alpha=0.8, linewidth=0.5)
            ax.set_xlabel("Basis index")
            ax.set_ylabel("Coefficient value")
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"$c_{{{j+1}}}$" for j in range(n_basis)])
            ax.set_title(f"Coefficient Distribution — {n_obs} observation{'s' if n_obs != 1 else ''}")
            ax.set_ylim(y_min - 0.3, y_max + 0.3)
            ax.grid(True, alpha=0.2, axis="y")

            # Add text showing total uncertainty
            total_unc = std.sum()
            ax.text(0.98, 0.95, f"Total std: {total_unc:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

            frames.append(render_frame(fig))
            plt.close(fig)

    # Hold last frame
    for _ in range(4):
        frames.append(frames[-1])

    save_gif(frames, ANIMATIONS_DIR / "coefficient_evolution.gif", fps=2)


# =============================================================================
# Animation 4: Information Gain Landscape Evolution
# =============================================================================

def generate_ig_landscape_evolution(model, x, t, y):
    """15 frames showing IG landscape shifting as observations are added."""
    print("\n[4/4] Generating ig_landscape_evolution.gif...")

    idx = 5
    x_i = x[idx : idx + 1]
    y_i = y[idx]
    t_np = t.numpy()

    scheduler = ActiveObservationScheduler(model)
    n_frames = 15

    frames = []
    obs_indices = []

    with torch.no_grad():
        mu_0, Sigma_0 = model.encode(x_i)

        for frame in range(n_frames):
            n_obs = frame + 1  # 1 to 15

            if frame == 0:
                obs_indices = [0]
            else:
                # Use the observations accumulated so far
                t_obs = t[obs_indices]
                y_obs = y_i[obs_indices].unsqueeze(0)
                _, _, mu_post, Sigma_post = model.update_and_predict(x_i, t_obs, y_obs, t)

                # Find next best observation
                all_idx = set(range(len(t)))
                cand_idx = sorted(all_idx - set(obs_indices))
                t_candidates = t[cand_idx]
                ig_cand = scheduler.compute_information_gain(mu_post, Sigma_post, t_candidates)
                best_cand = ig_cand[0].argmax().item()
                obs_indices.append(cand_idx[best_cand])

            # Compute IG landscape with current observations
            t_obs = t[obs_indices]
            y_obs = y_i[obs_indices].unsqueeze(0)
            _, _, mu_post, Sigma_post = model.update_and_predict(x_i, t_obs, y_obs, t)
            ig = scheduler.compute_information_gain(mu_post, Sigma_post, t)
            ig_np = ig[0].numpy()

            fig, ax = plt.subplots(figsize=(8, 5))

            # IG bars
            ax.bar(t_np, ig_np, width=0.018, alpha=0.7, color="green", label="Info gain")

            # Mark observed times
            for oi in obs_indices:
                ax.axvline(t[oi].item(), color="red", alpha=0.6, linewidth=1.5, linestyle="--")

            # Mark observation positions on x-axis
            ax.scatter(t[obs_indices].numpy(), np.zeros(len(obs_indices)),
                       c="red", s=80, zorder=5, marker="^", edgecolors="darkred",
                       label=f"Observed ({len(obs_indices)})")

            ax.set_xlabel("Time $t$")
            ax.set_ylabel("Information Gain")
            ax.set_title(f"Information Gain Landscape — {n_obs} observation{'s' if n_obs != 1 else ''}")
            ax.legend(loc="upper right", fontsize=9)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.2)

            frames.append(render_frame(fig))
            plt.close(fig)

    # Hold last frame
    for _ in range(4):
        frames.append(frames[-1])

    save_gif(frames, ANIMATIONS_DIR / "ig_landscape_evolution.gif", fps=2)


# =============================================================================
# Main
# =============================================================================

def main():
    print("TIMEVIEW-Adaptive Animation Generator")
    print("=" * 50)

    # Generate data and train/load model
    x, t, y = generate_synthetic_data(n_samples=100, input_dim=4)
    model = get_model(x, t, y)

    generate_adaptation_progression(model, x, t, y)
    generate_active_vs_uniform(model, x, t, y)
    generate_coefficient_evolution(model, x, t, y)
    generate_ig_landscape_evolution(model, x, t, y)

    print(f"\nAll animations saved to: {ANIMATIONS_DIR}")


if __name__ == "__main__":
    main()
