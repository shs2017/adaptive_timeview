import argparse

import dash
import numpy as np
import plotly.graph_objects as go
import torch
from dash import Input, Output, State, callback_context, dcc, html
from plotly.subplots import make_subplots

from constant import CHECKPOINT_DIR
from timeview_adaptive import (
    ActiveObservationScheduler,
    TimeviewAdaptive,
    TimeviewAdaptiveGated,
    TimeviewAdaptiveGatedHeteroscedastic,
    TimeviewAdaptiveHeteroscedastic,
    train_model,
)
from timeview_adaptive_experiments import load_dataset

SEED = 42

MODEL_CONFIGS = {
    "Standard": {
        "class": TimeviewAdaptive,
        "kwargs": {"n_basis": 7, "covariance_type": "diagonal"},
    },
    "Heteroscedastic": {
        "class": TimeviewAdaptiveHeteroscedastic,
        "kwargs": {"n_basis": 7, "covariance_type": "diagonal"},
    },
    "Gated": {
        "class": TimeviewAdaptiveGated,
        "kwargs": {"n_basis": 7, "covariance_type": "low_rank"},
    },
    "Best Config": {
        "class": TimeviewAdaptiveGatedHeteroscedastic,
        "kwargs": {"n_basis": 9, "covariance_type": "low_rank"},
    },
}

DATASETS = ["airfoil", "flchain", "stress_strain"]

def load_data(dataset_name):
    """Load dataset and return x, t, y tensors."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    x, t, y = load_dataset(dataset_name)
    return x, t, y


def get_checkpoint_path(dataset, variant):
    safe_name = variant.lower().replace(" ", "_")
    return CHECKPOINT_DIR / f"{dataset}_{safe_name}.pt"


def train_or_load_model(dataset, variant, x, t, y, retrain=False):
    config = MODEL_CONFIGS[variant]
    input_dim = x.shape[1]
    model = config["class"](input_dim=input_dim, **config["kwargs"])

    ckpt_path = get_checkpoint_path(dataset, variant)

    if ckpt_path.exists() and not retrain:
        print(f"  Loading {variant} from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    else:
        print(f"  Training {variant}...")
        n_train = int(0.8 * x.shape[0])
        train_model(
            model, x[:n_train], t, y[:n_train],
            n_epochs=100, lr=1e-3, n_obs=10, verbose=False,
        )
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved to {ckpt_path}")

    model.eval()
    return model


def create_app(dataset_name, retrain=False):
    print(f"Loading dataset: {dataset_name}")
    x, t, y = load_data(dataset_name)
    n_samples = x.shape[0]
    input_dim = x.shape[1]
    n_test_start = int(0.8 * n_samples)
    n_test = n_samples - n_test_start

    print(f"  {n_samples} samples, {len(t)} time points, {input_dim} features")

    print("Loading models...")
    models = {}
    for variant in MODEL_CONFIGS:
        models[variant] = train_or_load_model(dataset_name, variant, x, t, y, retrain)

    t_np = t.numpy().tolist()
    y_np = y.numpy().tolist()
    x_np = x.numpy().tolist()

    app = dash.Dash(
        __name__,
        title="TIMEVIEW-Adaptive Explorer",
        suppress_callback_exceptions=True,
    )

    sidebar = html.Div([
        html.H3("Controls", style={"marginBottom": "15px"}),

        html.Label("Dataset"),
        dcc.Dropdown(
            id="dataset-dropdown",
            options=[{"label": d.replace("_", " ").title(), "value": d} for d in DATASETS],
            value=dataset_name,
            clearable=False,
        ),

        html.Label("Test Sample", style={"marginTop": "15px"}),
        dcc.Slider(
            id="sample-slider",
            min=n_test_start,
            max=n_samples - 1,
            value=n_test_start,
            step=1,
            marks={n_test_start: "0", n_samples - 1: str(n_test - 1)},
            tooltip={"placement": "bottom"},
        ),

        html.Label("Model Variant", style={"marginTop": "15px"}),
        dcc.Dropdown(
            id="model-dropdown",
            options=[{"label": v, "value": v} for v in MODEL_CONFIGS],
            value="Standard",
            clearable=False,
        ),

        html.Hr(),
        html.Label("Auto Observations"),
        dcc.Slider(
            id="n-obs-slider",
            min=0,
            max=len(t) - 1,
            value=0,
            step=1,
            marks={0: "0", len(t) // 2: str(len(t) // 2), len(t) - 1: str(len(t) - 1)},
            tooltip={"placement": "bottom"},
        ),

        html.Div([
            html.Button("Clear All", id="clear-btn", n_clicks=0,
                        style={"marginRight": "10px"}),
            html.Button("Random Sample", id="random-btn", n_clicks=0),
        ], style={"marginTop": "10px", "display": "flex"}),

    ], style={
        "width": "250px",
        "padding": "20px",
        "backgroundColor": "#f8f9fa",
        "borderRight": "1px solid #dee2e6",
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "overflowY": "auto",
    })

    main_content = html.Div([
        html.H2("TIMEVIEW-Adaptive Explorer", style={"textAlign": "center", "padding": "10px"}),

        dcc.Tabs(id="main-tabs", value="tab-adaptation", children=[
            dcc.Tab(label="Bayesian Adaptation", value="tab-adaptation"),
            dcc.Tab(label="Uncertainty Decomposition", value="tab-uncertainty"),
            dcc.Tab(label="Active Scheduling", value="tab-active"),
            dcc.Tab(label="Gating Mechanism", value="tab-gating"),
            dcc.Tab(label="Model Comparison", value="tab-comparison"),
        ]),

        html.Div(id="tab-content", style={"padding": "15px"}),

        # Hidden stores
        dcc.Store(id="observation-store", data=[]),
        dcc.Store(id="dataset-cache", data={
            "t": t_np,
            "y": y_np,
            "x": x_np,
            "n_test_start": n_test_start,
        }),
    ], style={
        "marginLeft": "290px",
        "padding": "10px",
    })

    app.layout = html.Div([sidebar, main_content])

    @app.callback(
        Output("observation-store", "data"),
        [
            Input("n-obs-slider", "value"),
            Input("clear-btn", "n_clicks"),
        ],
        [State("dataset-cache", "data")],
    )
    def update_observations(n_obs, clear_clicks, cache):
        """Update observation list from slider or clear button."""
        ctx = callback_context
        if ctx.triggered and ctx.triggered[0]["prop_id"] == "clear-btn.n_clicks":
            return []

        if n_obs == 0:
            return []

        t_vals = cache["t"]
        obs = list(range(min(n_obs, len(t_vals))))
        return obs

    @app.callback(
        Output("sample-slider", "value"),
        Input("random-btn", "n_clicks"),
        State("dataset-cache", "data"),
        prevent_initial_call=True,
    )
    def random_sample(n_clicks, cache):
        """Pick a random test sample."""
        n_start = cache["n_test_start"]
        n_total = len(cache["y"])
        return int(np.random.randint(n_start, n_total))

    @app.callback(
        Output("tab-content", "children"),
        [
            Input("main-tabs", "value"),
            Input("sample-slider", "value"),
            Input("model-dropdown", "value"),
            Input("observation-store", "data"),
        ],
        [State("dataset-cache", "data")],
    )
    def render_tab(tab, sample_idx, variant, obs_indices, cache):
        """Render the active tab content."""
        t_tensor = torch.tensor(cache["t"], dtype=torch.float32)
        y_all = torch.tensor(cache["y"], dtype=torch.float32)
        x_all = torch.tensor(cache["x"], dtype=torch.float32)

        x_i = x_all[sample_idx : sample_idx + 1]
        y_i = y_all[sample_idx]
        t_np = np.array(cache["t"])
        y_true = y_i.numpy()
        model = models[variant]

        if tab == "tab-adaptation":
            return render_adaptation_tab(model, x_i, y_i, t_tensor, t_np, y_true, obs_indices)
        elif tab == "tab-uncertainty":
            return render_uncertainty_tab(model, x_i, y_i, t_tensor, t_np, obs_indices)
        elif tab == "tab-active":
            return render_active_tab(model, x_i, y_i, t_tensor, t_np, y_true, obs_indices)
        elif tab == "tab-gating":
            return render_gating_tab(models, x_i, y_i, t_tensor, t_np, y_true, obs_indices)
        elif tab == "tab-comparison":
            return render_comparison_tab(models, x_i, y_i, t_tensor, t_np, y_true, obs_indices)
        return html.Div("Select a tab")

    def render_adaptation_tab(model, x_i, y_i, t_tensor, t_np, y_true, obs_indices):
        n_obs = len(obs_indices)

        with torch.no_grad():
            mu_0, Sigma_0 = model.encode(x_i)

            # Prior predictions
            y_prior_mean, y_prior_var = model.predict(mu_0, Sigma_0, t_tensor)
            prior_mean = y_prior_mean[0].numpy()
            prior_std = np.sqrt(y_prior_var[0].numpy())

            if n_obs > 0:
                t_obs = t_tensor[obs_indices]
                y_obs = y_i[obs_indices].unsqueeze(0)
                y_mean, y_var, mu_post, Sigma_post = model.update_and_predict(
                    x_i, t_obs, y_obs, t_tensor
                )
                post_mean = y_mean[0].numpy()
                post_std = np.sqrt(y_var[0].numpy())
                mu_coeffs = mu_post[0].numpy()
                std_coeffs = np.sqrt(np.diag(Sigma_post[0].numpy()))
            else:
                post_mean = prior_mean
                post_std = prior_std
                mu_coeffs = mu_0[0].numpy()
                std_coeffs = np.sqrt(np.diag(Sigma_0[0].numpy()))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=t_np, y=y_true, mode="lines",
            name="Ground truth", line=dict(color="black", dash="dash", width=1.5),
        ))

        fig.add_trace(go.Scatter(
            x=np.concatenate([t_np, t_np[::-1]]),
            y=np.concatenate([prior_mean + 1.96 * prior_std,
                              (prior_mean - 1.96 * prior_std)[::-1]]),
            fill="toself", fillcolor="rgba(255,165,0,0.1)",
            line=dict(color="rgba(255,165,0,0)"),
            name="Prior 95% CI",
        ))
        fig.add_trace(go.Scatter(
            x=t_np, y=prior_mean, mode="lines",
            name="Prior mean", line=dict(color="orange", dash="dot", width=1.5),
        ))

        if n_obs > 0:
            fig.add_trace(go.Scatter(
                x=np.concatenate([t_np, t_np[::-1]]),
                y=np.concatenate([post_mean + 1.96 * post_std,
                                  (post_mean - 1.96 * post_std)[::-1]]),
                fill="toself", fillcolor="rgba(0,100,255,0.15)",
                line=dict(color="rgba(0,100,255,0)"),
                name="Posterior 95% CI",
            ))
            fig.add_trace(go.Scatter(
                x=t_np, y=post_mean, mode="lines",
                name="Posterior mean", line=dict(color="blue", width=2),
            ))

            # Observations
            obs_t = t_np[obs_indices]
            obs_y = y_true[obs_indices]
            fig.add_trace(go.Scatter(
                x=obs_t, y=obs_y, mode="markers",
                name=f"Observations ({n_obs})",
                marker=dict(color="red", size=10, line=dict(color="darkred", width=1)),
            ))

        mse = float(np.mean((post_mean - y_true) ** 2))
        fig.update_layout(
            title=f"Bayesian Adaptation ({n_obs} observations, MSE={mse:.4f})",
            xaxis_title="Time t",
            yaxis_title="y(t)",
            height=500,
            hovermode="x unified",
        )

        n_basis = len(mu_coeffs)
        coeff_fig = go.Figure()
        coeff_fig.add_trace(go.Bar(
            x=[f"c{j+1}" for j in range(n_basis)],
            y=mu_coeffs,
            error_y=dict(type="data", array=1.96 * std_coeffs, visible=True),
            marker_color="steelblue",
            name="Coefficients",
        ))
        coeff_fig.update_layout(
            title="Coefficient Distribution (95% CI)",
            xaxis_title="Basis index",
            yaxis_title="Value",
            height=300,
        )

        return html.Div([
            dcc.Graph(figure=fig, id="trajectory-plot"),
            html.P("Use the 'Auto Observations' slider to add observations sequentially.",
                   style={"color": "gray", "fontSize": "12px", "textAlign": "center"}),
            dcc.Graph(figure=coeff_fig),
        ])

    def render_uncertainty_tab(model, x_i, y_i, t_tensor, t_np, obs_indices):
        n_obs = max(len(obs_indices), 2)
        obs_to_use = list(range(n_obs)) if len(obs_indices) < 2 else obs_indices

        with torch.no_grad():
            mu_0, Sigma_0 = model.encode(x_i)
            Phi = model.get_basis(t_tensor)

            PhiSigma0 = torch.matmul(Phi.unsqueeze(0), Sigma_0)
            prior_var = torch.sum(PhiSigma0 * Phi.unsqueeze(0), dim=-1)[0].numpy()

            has_noise_model = hasattr(model, 'noise_model') and model.noise_model is not None
            if has_noise_model:
                noise_var = model.get_noise_var(t_tensor).numpy()
            else:
                sigma2 = model.sigma.item() ** 2
                noise_var = np.full_like(prior_var, sigma2)

            t_obs = t_tensor[obs_to_use]
            y_obs = y_i[obs_to_use].unsqueeze(0)
            _, _, mu_post, Sigma_post = model.update_and_predict(x_i, t_obs, y_obs, t_tensor)

            PhiSigmaN = torch.matmul(Phi.unsqueeze(0), Sigma_post)
            post_var = torch.sum(PhiSigmaN * Phi.unsqueeze(0), dim=-1)[0].numpy()

        info_gained = prior_var - post_var
        remaining_epistemic = post_var
        pct_resolved = info_gained.sum() / prior_var.sum() * 100

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=t_np, y=noise_var, fill="tozeroy",
            name="Noise (aleatoric)", fillcolor="rgba(128,128,128,0.4)",
            line=dict(color="gray"),
        ))
        fig.add_trace(go.Scatter(
            x=t_np, y=noise_var + remaining_epistemic, fill="tonexty",
            name="Remaining epistemic", fillcolor="rgba(70,130,180,0.4)",
            line=dict(color="steelblue"),
        ))
        fig.add_trace(go.Scatter(
            x=t_np, y=noise_var + remaining_epistemic + info_gained, fill="tonexty",
            name="Information gained", fillcolor="rgba(0,128,0,0.4)",
            line=dict(color="green"),
        ))

        for idx in obs_to_use:
            fig.add_vline(x=t_np[idx], line_dash="dash", line_color="red", opacity=0.3)

        fig.update_layout(
            title=f"Uncertainty Decomposition ({n_obs} obs, {pct_resolved:.1f}% epistemic resolved)",
            xaxis_title="Time t",
            yaxis_title="Variance",
            height=500,
        )

        stats_div = html.Div([
            html.H4("Summary Statistics"),
            html.Table([
                html.Tr([html.Td("Observations:"), html.Td(f"{n_obs}")]),
                html.Tr([html.Td("Epistemic resolved:"), html.Td(f"{pct_resolved:.1f}%")]),
                html.Tr([html.Td("Noise floor:"), html.Td(f"{sigma2:.4f}")]),
                html.Tr([html.Td("Remaining epistemic (avg):"),
                         html.Td(f"{remaining_epistemic.mean():.4f}")]),
            ], style={"margin": "10px auto", "borderCollapse": "collapse"}),
        ], style={"textAlign": "center"})

        return html.Div([dcc.Graph(figure=fig), stats_div])

    def render_active_tab(model, x_i, y_i, t_tensor, t_np, y_true, obs_indices):
        n_obs = max(len(obs_indices), 3)
        obs_to_use = list(range(n_obs)) if len(obs_indices) < 3 else obs_indices

        scheduler = ActiveObservationScheduler(model)

        with torch.no_grad():
            t_obs = t_tensor[obs_to_use]
            y_obs = y_i[obs_to_use].unsqueeze(0)
            _, _, mu_post, Sigma_post = model.update_and_predict(x_i, t_obs, y_obs, t_tensor)

            all_idx = set(range(len(t_tensor)))
            cand_idx = sorted(all_idx - set(obs_to_use))
            t_candidates = t_tensor[cand_idx]

            ig = scheduler.compute_information_gain(mu_post, Sigma_post, t_candidates)
            ig_np = ig[0].numpy()

            best_cand = ig_np.argmax()
            best_time = t_candidates[best_cand].item()

        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            "Information Gain Landscape", "Suggested Next Observation"
        ])

        fig.add_trace(go.Bar(
            x=[t_np[i] for i in cand_idx],
            y=ig_np,
            marker_color=["red" if i == best_cand else "green" for i in range(len(cand_idx))],
            name="Info gain",
        ), row=1, col=1)

        for idx in obs_to_use:
            fig.add_vline(x=t_np[idx], line_dash="dash", line_color="red", opacity=0.4,
                          row=1, col=1)

        with torch.no_grad():
            y_mean, y_var, _, _ = model.update_and_predict(x_i, t_obs, y_obs, t_tensor)
            mean = y_mean[0].numpy()
            std = np.sqrt(y_var[0].numpy())

        fig.add_trace(go.Scatter(
            x=t_np, y=y_true, mode="lines",
            name="Ground truth", line=dict(color="black", dash="dash"),
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=t_np, y=mean, mode="lines",
            name="Prediction", line=dict(color="blue", width=2),
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=np.concatenate([t_np, t_np[::-1]]),
            y=np.concatenate([mean + 1.96 * std, (mean - 1.96 * std)[::-1]]),
            fill="toself", fillcolor="rgba(0,100,255,0.15)",
            line=dict(color="rgba(0,100,255,0)"), name="95% CI",
            showlegend=False,
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=[t_np[i] for i in obs_to_use],
            y=[y_true[i] for i in obs_to_use],
            mode="markers", name="Current obs",
            marker=dict(color="red", size=10),
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=[best_time], y=[y_true[cand_idx[best_cand]]],
            mode="markers", name=f"Suggested (t={best_time:.3f})",
            marker=dict(color="gold", size=15, symbol="star",
                        line=dict(color="orange", width=2)),
        ), row=1, col=2)

        fig.update_layout(height=500, title_text=f"Active Observation Scheduling ({n_obs} obs)")

        return html.Div([dcc.Graph(figure=fig)])

    def render_gating_tab(all_models, x_i, y_i, t_tensor, t_np, y_true, obs_indices):
        n_obs = max(len(obs_indices), 5)
        obs_to_use = list(range(n_obs)) if len(obs_indices) < 5 else obs_indices

        gated_model = all_models.get("Gated")
        if gated_model is None:
            return html.Div("Gated model not available")

        with torch.no_grad():
            mu_0, Sigma_0 = gated_model.encode(x_i)

            y_prior, y_prior_var = gated_model.predict(mu_0, Sigma_0, t_tensor)
            prior_mean = y_prior[0].numpy()

            t_obs = t_tensor[obs_to_use]
            y_obs = y_i[obs_to_use].unsqueeze(0)
            y_gated, y_var, _, _ = gated_model.update_and_predict(x_i, t_obs, y_obs, t_tensor)
            gated_mean = y_gated[0].numpy()
            gated_std = np.sqrt(y_var[0].numpy())

            y_pr_obs, y_pr_var_obs = gated_model.predict(mu_0, Sigma_0, t_obs)
            prior_mse = ((y_obs - y_pr_obs) ** 2).mean(dim=1, keepdim=True)
            obs_unc = y_pr_var_obs.mean(dim=1, keepdim=True)
            gate_val = gated_model.gate(x_i, prior_mse, obs_unc).item()

            Phi_obs = gated_model.get_basis(t_obs).unsqueeze(0).expand(1, -1, -1)
            gated_model.updater.sigma2 = gated_model.sigma ** 2
            mu_post, Sigma_post = gated_model.updater.update(mu_0, Sigma_0, Phi_obs, y_obs)
            y_post, _ = gated_model.predict(mu_post, Sigma_post, t_tensor)
            post_mean = y_post[0].numpy()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=t_np, y=y_true, mode="lines",
            name="Ground truth", line=dict(color="black", dash="dash", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=t_np, y=prior_mean, mode="lines",
            name="Prior", line=dict(color="orange", dash="dot", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=t_np, y=post_mean, mode="lines",
            name="Pure posterior", line=dict(color="lightblue", dash="dash", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=t_np, y=gated_mean, mode="lines",
            name="Gated blend", line=dict(color="purple", width=2.5),
        ))

        fig.add_trace(go.Scatter(
            x=np.concatenate([t_np, t_np[::-1]]),
            y=np.concatenate([gated_mean + 1.96 * gated_std,
                              (gated_mean - 1.96 * gated_std)[::-1]]),
            fill="toself", fillcolor="rgba(128,0,128,0.1)",
            line=dict(color="rgba(128,0,128,0)"), name="Gated 95% CI",
        ))

        fig.add_trace(go.Scatter(
            x=[t_np[i] for i in obs_to_use],
            y=[y_true[i] for i in obs_to_use],
            mode="markers", name="Observations",
            marker=dict(color="red", size=10),
        ))

        fig.update_layout(
            title=f"Gating Mechanism (gate alpha = {gate_val:.3f}, {n_obs} obs)",
            xaxis_title="Time t",
            yaxis_title="y(t)",
            height=500,
        )

        gate_bar = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gate_val,
            title={"text": "Gate Weight (alpha)"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "purple"},
                "steps": [
                    {"range": [0, 0.3], "color": "rgba(255,165,0,0.3)"},
                    {"range": [0.3, 0.7], "color": "rgba(200,200,200,0.3)"},
                    {"range": [0.7, 1.0], "color": "rgba(0,100,255,0.3)"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": gate_val,
                },
            },
        ))
        gate_bar.update_layout(height=250)

        return html.Div([
            dcc.Graph(figure=fig),
            html.Div([
                html.Div([dcc.Graph(figure=gate_bar)], style={"width": "50%", "display": "inline-block"}),
                html.Div([
                    html.H4("Gate Interpretation"),
                    html.P(f"alpha = {gate_val:.3f}"),
                    html.P("alpha near 0: trust prior (static prediction)"),
                    html.P("alpha near 1: trust posterior (Bayesian update)"),
                    html.P(f"Prior MSE on obs: {prior_mse.item():.4f}"),
                    html.P(f"Prior uncertainty at obs: {obs_unc.item():.4f}"),
                ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top",
                          "padding": "20px"}),
            ]),
        ])

    def render_comparison_tab(all_models, x_i, y_i, t_tensor, t_np, y_true, obs_indices):
        n_obs = max(len(obs_indices), 5)
        obs_to_use = list(range(n_obs)) if len(obs_indices) < 5 else obs_indices

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(MODEL_CONFIGS.keys()),
        )

        colors = ["blue", "orange", "green", "red"]
        metrics = []

        for i, (variant, color) in enumerate(zip(MODEL_CONFIGS, colors, strict=True)):
            row = i // 2 + 1
            col = i % 2 + 1
            model = all_models[variant]

            with torch.no_grad():
                t_obs = t_tensor[obs_to_use]
                y_obs = y_i[obs_to_use].unsqueeze(0)
                y_mean, y_var, _, _ = model.update_and_predict(x_i, t_obs, y_obs, t_tensor)
                mean = y_mean[0].numpy()
                std = np.sqrt(y_var[0].numpy())
                mse = float(np.mean((mean - y_true) ** 2))

                in_interval = np.abs(y_true - mean) < 1.96 * std
                coverage = float(in_interval.mean())
                sharpness = float(2 * 1.96 * std.mean())

            metrics.append({"Model": variant, "MSE": mse, "Coverage": coverage,
                            "Sharpness": sharpness})

            fig.add_trace(go.Scatter(
                x=t_np, y=y_true, mode="lines",
                line=dict(color="black", dash="dash", width=1),
                showlegend=(i == 0), name="Truth",
            ), row=row, col=col)

            fig.add_trace(go.Scatter(
                x=np.concatenate([t_np, t_np[::-1]]),
                y=np.concatenate([mean + 1.96 * std, (mean - 1.96 * std)[::-1]]),
                fill="toself", fillcolor="rgba(100,100,200,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, name="CI",
            ), row=row, col=col)

            fig.add_trace(go.Scatter(
                x=t_np, y=mean, mode="lines",
                line=dict(color=color, width=2),
                showlegend=(i == 0), name="Prediction",
            ), row=row, col=col)

            fig.add_trace(go.Scatter(
                x=[t_np[j] for j in obs_to_use],
                y=[y_true[j] for j in obs_to_use],
                mode="markers",
                marker=dict(color="red", size=6),
                showlegend=(i == 0), name="Obs",
            ), row=row, col=col)

        fig.update_layout(
            height=700,
            title_text=f"Model Comparison ({n_obs} observations)",
        )

        table = html.Table(
            [html.Tr([html.Th(h) for h in ["Model", "MSE", "Coverage", "Sharpness"]])] +
            [html.Tr([
                html.Td(m["Model"]),
                html.Td(f"{m['MSE']:.4f}"),
                html.Td(f"{m['Coverage']:.1%}"),
                html.Td(f"{m['Sharpness']:.3f}"),
            ]) for m in metrics],
            style={
                "margin": "20px auto",
                "borderCollapse": "collapse",
                "border": "1px solid #ccc",
            },
        )

        return html.Div([dcc.Graph(figure=fig), table])

    return app


def main():
    parser = argparse.ArgumentParser(description="TIMEVIEW-Adaptive Interactive Dashboard")
    parser.add_argument("--dataset", default="airfoil", choices=DATASETS,
                        help="Dataset to load (default: airfoil)")
    parser.add_argument("--retrain", action="store_true",
                        help="Force re-training of all models")
    parser.add_argument("--port", type=int, default=8050,
                        help="Port to run the server on (default: 8050)")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode")
    args = parser.parse_args()

    app = create_app(args.dataset, retrain=args.retrain)

    print(f"\nStarting dashboard at http://localhost:{args.port}")
    print("Press Ctrl+C to stop.\n")
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
