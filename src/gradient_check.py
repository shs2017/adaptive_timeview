"""
Check whether gradient to prior parameters (mu_0) attenuates as n_obs increases.

Hypothesis: d_loss/d_mu_0 = d_loss/d_mu_n * Sigma_n * Sigma_0_inv
As n_obs grows, Sigma_n -> 0, so the Jacobian shrinks and prior gets no gradient.
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/shs2017/Code/adaptive_timeview/src")
from timeview_adaptive import TimeviewAdaptive, bspline_basis, create_knots

torch.manual_seed(42)

# --- Setup ---
input_dim = 10
n_basis = 9
n_time = 50
batch_size = 32
n_obs_values = [0, 1, 2, 3, 5, 8, 12, 20, 30, 40, 49]

model = TimeviewAdaptive(
    input_dim=input_dim,
    n_basis=n_basis,
    hidden_sizes=[32, 64, 32],
    observation_noise=0.1,
    use_batchnorm=False,
    kl_weight=0.01,
)
model.train()

t_all = torch.linspace(0, 1, n_time)
knots = create_knots(n_basis)
Phi_all = bspline_basis(t_all, knots)  # (n_time, n_basis)

# Synthetic data
x = torch.randn(batch_size, input_dim)
y = torch.randn(batch_size, n_time)

grad_mu0_norms = []
sigma_n_traces = []
jacobian_norms = []

for n_obs in n_obs_values:
    model.zero_grad()

    mu_0, Sigma_0, bias = model.encode(x)
    mu_0.retain_grad()

    if n_obs == 0:
        # Prior prediction only
        mu_pred, _ = model.predict(mu_0, Sigma_0, t_all, bias)
        loss = ((mu_pred - y) ** 2).mean()
        sigma_n_trace = float("nan")
        jac_norm = float("nan")
    else:
        y_obs = y[:, :n_obs]
        Phi_obs = Phi_all[:n_obs].unsqueeze(0).expand(batch_size, -1, -1)

        mu_n, Sigma_n = model.updater.update(mu_0, Sigma_0, Phi_obs, y_obs)
        mu_pred, _ = model.predict(mu_n, Sigma_n, t_all, bias)
        loss = ((mu_pred - y) ** 2).mean()

        with torch.no_grad():
            Sigma_0_inv = torch.linalg.inv(Sigma_0 + 1e-6 * torch.eye(n_basis))
            J = torch.bmm(Sigma_n, Sigma_0_inv)
            jac_norm = J.norm(dim=(1, 2)).mean().item()
            sigma_n_trace = Sigma_n.diagonal(dim1=-2, dim2=-1).sum(dim=-1).mean().item()

    loss.backward()

    grad_norm = mu_0.grad.norm(dim=-1).mean().item()
    grad_mu0_norms.append(grad_norm)
    sigma_n_traces.append(sigma_n_trace)
    jacobian_norms.append(jac_norm)

    print(f"n_obs={n_obs:3d}  |  grad_mu0={grad_norm:.5f}  |  "
          f"tr(Sigma_n)={sigma_n_trace:.5f}  |  ||J||={jac_norm:.5f}")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

ax = axes[0]
ax.plot(n_obs_values, grad_mu0_norms, "o-", color="steelblue")
ax.set_xlabel("n_obs (context points)")
ax.set_ylabel("||d_loss / d_mu_0||")
ax.set_title("Gradient norm to prior mean (mu_0)")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

obs_vals_no0 = [v for v in n_obs_values if v > 0]
traces_no0 = [t for t in sigma_n_traces if not np.isnan(t)]
jacs_no0 = [j for j in jacobian_norms if not np.isnan(j)]

ax = axes[1]
ax.plot(obs_vals_no0, traces_no0, "o-", color="darkorange")
ax.set_xlabel("n_obs")
ax.set_ylabel("tr(Sigma_n)")
ax.set_title("Posterior variance (tr Sigma_n)")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(obs_vals_no0, jacs_no0, "o-", color="firebrick")
ax.set_xlabel("n_obs")
ax.set_ylabel("||Sigma_n @ Sigma_0_inv||")
ax.set_title("Jacobian magnitude (Sigma_n @ Sigma_0_inv)")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "/home/shs2017/Code/adaptive_timeview/figures/gradient_check.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved to {out_path}")
