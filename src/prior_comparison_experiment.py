import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

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


def compute_loss(
    model: TimeviewAdaptive,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    t: torch.Tensor,
    n_obs: int,
    prior_weight: float = 0.0,
    future_only: bool = False,
) -> torch.Tensor:
    t_obs = t[:n_obs]
    y_obs = y_batch[:, :n_obs]

    mu_0, Sigma_0, bias = model.encode(x_batch)
    y_mean, y_var, _, _ = model.update_and_predict(
        x_batch, t_obs, y_obs, t, mu_0=mu_0, Sigma_0=Sigma_0, bias=bias,
    )

    if future_only and n_obs < y_batch.shape[1]:
        y_future = y_batch[:, n_obs:]
        y_mean_future = y_mean[:, n_obs:]
        y_var_future = y_var[:, n_obs:]
        post_nll = 0.5 * torch.log(2 * np.pi * y_var_future) + 0.5 * (y_future - y_mean_future) ** 2 / y_var_future
    else:
        post_nll = 0.5 * torch.log(2 * np.pi * y_var) + 0.5 * (y_batch - y_mean) ** 2 / y_var

    loss = post_nll.mean() + model.kl_weight * model.kl_divergence(mu_0, Sigma_0)

    if prior_weight > 0.0:
        y_prior_mean, y_prior_var = model.predict(mu_0, Sigma_0, t, bias=bias)
        prior_nll = 0.5 * torch.log(2 * np.pi * y_prior_var) + 0.5 * (y_batch - y_prior_mean) ** 2 / y_prior_var
        loss = loss + prior_weight * prior_nll.mean()

    return loss


def train_model_custom(
    model: TimeviewAdaptive,
    x_train: torch.Tensor,
    t: torch.Tensor,
    y_train: torch.Tensor,
    n_obs: int,
    n_epochs: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    patience: int = 10,
    check_val_every_n_epoch: int = 1,
    prior_weight: float = 0.0,
    future_only: bool = False,
    x_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_eval = x_val if x_val is not None else x_train
    y_eval = y_val if y_val is not None else y_train
    n_samples = x_train.shape[0]
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(n_samples)

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            if len(batch_idx) < 2:
                continue
            loss = compute_loss(
                model, x_train[batch_idx], y_train[batch_idx], t, n_obs,
                prior_weight=prior_weight, future_only=future_only,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if (epoch + 1) % check_val_every_n_epoch == 0:
            model.eval()
            with torch.no_grad():
                val_loss = compute_loss(
                    model, x_eval, y_eval, t, n_obs,
                    prior_weight=prior_weight, future_only=future_only,
                )

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience > 0 and epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def train_model_random_nobs(
    model: TimeviewAdaptive,
    x_train: torch.Tensor,
    t: torch.Tensor,
    y_train: torch.Tensor,
    n_epochs: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    patience: int = 10,
    check_val_every_n_epoch: int = 1,
    future_only: bool = False,
    x_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_eval = x_val if x_val is not None else x_train
    y_eval = y_val if y_val is not None else y_train
    n_time = len(t)
    n_samples = x_train.shape[0]
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(n_samples)

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            if len(batch_idx) < 2:
                continue
            n_obs = torch.randint(0, n_time, (1,)).item()
            loss = compute_loss(
                model, x_train[batch_idx], y_train[batch_idx], t, n_obs,
                future_only=future_only,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if (epoch + 1) % check_val_every_n_epoch == 0:
            model.eval()
            with torch.no_grad():
                # Validate at n_obs=0 and mid-range, average both
                mid = n_time // 2
                val_loss = 0.5 * compute_loss(model, x_eval, y_eval, t, 0, future_only=future_only)
                val_loss = val_loss + 0.5 * compute_loss(model, x_eval, y_eval, t, mid, future_only=future_only)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience > 0 and epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def train_model_two_phase(
    model: TimeviewAdaptive,
    x_train: torch.Tensor,
    t: torch.Tensor,
    y_train: torch.Tensor,
    n_obs: int,
    n_epochs: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    patience: int = 10,
    check_val_every_n_epoch: int = 1,
    phase1_fraction: float = 0.5,
    x_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
) -> None:
    """Two-phase training.
    Phase 1: train the full model on prior NLL
    Phase 2: freeze the encoder, train only log_sigma on posterior NLL
    """
    x_eval = x_val if x_val is not None else x_train
    y_eval = y_val if y_val is not None else y_train
    n_samples = x_train.shape[0]
    n_epochs_1 = int(n_epochs * phase1_fraction)
    n_epochs_2 = n_epochs - n_epochs_1

    # Phase 1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss, best_state, epochs_no_imp = float("inf"), None, 0

    for epoch in range(n_epochs_1):
        model.train()
        indices = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = indices[start:start + batch_size]
            if len(idx) < 2:
                continue
            loss = compute_loss(model, x_train[idx], y_train[idx], t, n_obs=0)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if (epoch + 1) % check_val_every_n_epoch == 0:
            model.eval()
            with torch.no_grad():
                val_loss = compute_loss(model, x_eval, y_eval, t, n_obs=0)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_imp = 0
            else:
                epochs_no_imp += 1
            if patience > 0 and epochs_no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Phase 2
    for p in model.encoder.parameters():
        p.requires_grad = False

    posterior_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(posterior_params, lr=lr, weight_decay=weight_decay)
    best_loss, best_state, epochs_no_imp = float("inf"), None, 0

    for epoch in range(n_epochs_2):
        model.train()
        indices = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = indices[start:start + batch_size]
            if len(idx) < 2:
                continue
            loss = compute_loss(model, x_train[idx], y_train[idx], t, n_obs=n_obs)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(posterior_params, max_norm=1.0)
            optimizer.step()

        if (epoch + 1) % check_val_every_n_epoch == 0:
            model.eval()
            with torch.no_grad():
                val_loss = compute_loss(model, x_eval, y_eval, t, n_obs=n_obs)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_imp = 0
            else:
                epochs_no_imp += 1
            if patience > 0 and epochs_no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    for p in model.encoder.parameters():
        p.requires_grad = True


def train_model_weighted_random_nobs(
    model: TimeviewAdaptive,
    x_train: torch.Tensor,
    t: torch.Tensor,
    y_train: torch.Tensor,
    prior_weight: float = 0.5,
    n_epochs: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    patience: int = 10,
    check_val_every_n_epoch: int = 1,
    x_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
) -> None:
    """Weighted sum of prior and posterior NLL with random n_obs each batch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_eval = x_val if x_val is not None else x_train
    y_eval = y_val if y_val is not None else y_train
    n_time = len(t)
    n_samples = x_train.shape[0]
    best_loss, best_state, epochs_no_imp = float("inf"), None, 0

    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(n_samples)

        for start in range(0, n_samples, batch_size):
            idx = indices[start:start + batch_size]
            if len(idx) < 2:
                continue
            xb, yb = x_train[idx], y_train[idx]
            n_obs_post = torch.randint(1, n_time, (1,)).item()

            try:
                prior_loss = compute_loss(model, xb, yb, t, n_obs=0)
                post_loss  = compute_loss(model, xb, yb, t, n_obs=n_obs_post)
                loss = prior_weight * prior_loss + (1.0 - prior_weight) * post_loss
            except torch.linalg.LinAlgError:
                continue

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if (epoch + 1) % check_val_every_n_epoch == 0:
            model.eval()
            with torch.no_grad():
                mid = n_time // 2
                try:
                    val_loss = (prior_weight * compute_loss(model, x_eval, y_eval, t, 0)
                                + (1 - prior_weight) * compute_loss(model, x_eval, y_eval, t, mid))
                except torch.linalg.LinAlgError:
                    val_loss = torch.tensor(float("inf"))
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_imp = 0
            else:
                epochs_no_imp += 1
            if patience > 0 and epochs_no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def train_model_random_obs(
    model: TimeviewAdaptive,
    x_train: torch.Tensor,
    t: torch.Tensor,
    y_train: torch.Tensor,
    n_obs: int,
    n_epochs: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    patience: int = 10,
    x_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
) -> None:
    """Like train_model but samples n_obs random (sorted) time points each batch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_eval = x_val if x_val is not None else x_train
    y_eval = y_val if y_val is not None else y_train
    n_time = len(t)
    n_samples = x_train.shape[0]
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            if len(batch_idx) < 2:
                continue
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            obs_idx = torch.randperm(n_time)[:n_obs].sort().values
            t_obs = t[obs_idx]
            y_obs = y_batch[:, obs_idx]

            mu_0, Sigma_0, bias = model.encode(x_batch)
            y_mean, y_var, _, _ = model.update_and_predict(
                x_batch, t_obs, y_obs, t, mu_0=mu_0, Sigma_0=Sigma_0, bias=bias,
            )
            nll = 0.5 * torch.log(2 * np.pi * y_var) + 0.5 * (y_batch - y_mean) ** 2 / y_var
            loss = nll.mean() + model.kl_weight * model.kl_divergence(mu_0, Sigma_0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            obs_idx = torch.randperm(n_time)[:n_obs].sort().values
            t_obs = t[obs_idx]
            y_obs = y_eval[:, obs_idx]
            mu_0, Sigma_0, bias = model.encode(x_eval)
            y_mean, y_var, _, _ = model.update_and_predict(
                x_eval, t_obs, y_obs, t, mu_0=mu_0, Sigma_0=Sigma_0, bias=bias,
            )
            nll = 0.5 * torch.log(2 * np.pi * y_var) + 0.5 * (y_eval - y_mean) ** 2 / y_var
            val_loss = nll.mean() + model.kl_weight * model.kl_divergence(mu_0, Sigma_0)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)


def tune_adaptive_hyperparams(
    variant_kwargs: dict,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_va: torch.Tensor,
    y_va: torch.Tensor,
    t_ds: torch.Tensor,
    n_basis_ds: int,
    knots_ds: torch.Tensor,
    train_n_obs: int,
    n_tune: int = 30,
    n_epochs: int = 1000,
    seed: int = 0,
    seed_trials: list[dict] | None = None,
    lr_min: float = 1e-4,
    lr_max: float = 1e-1,
    storage: str | None = None,
    study_name: str | None = None,
) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    random_nobs = variant_kwargs.get("random_nobs", False)
    prior_weight_fixed = variant_kwargs.get("prior_weight", 0.0)
    future_only = variant_kwargs.get("future_only", False)
    kl_weight = variant_kwargs.get("kl_weight", 0.01)
    two_phase = variant_kwargs.get("two_phase", False)
    weighted_random_nobs = variant_kwargs.get("weighted_random_nobs", False)
    n_time = len(t_ds)

    def objective(trial: optuna.Trial) -> float:
        hidden_sizes = [trial.suggest_int(f"h{i}", 16, 128) for i in range(3)]
        lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        dropout_p = trial.suggest_float("dropout_p", 0.0, 0.4)
        prior_weight = (
            trial.suggest_float("prior_weight", 0.1, 2.0, log=True)
            if (prior_weight_fixed > 0.0 or weighted_random_nobs) else 0.0
        )

        torch.manual_seed(seed)
        model = TimeviewAdaptive(
            input_dim=x_tr.shape[1],
            n_basis=n_basis_ds,
            observation_noise=0.1,
            covariance_type="diagonal",
            use_batchnorm=True,
            learn_noise=True,
            kl_weight=kl_weight,
            knots=knots_ds,
            hidden_sizes=hidden_sizes,
            dropout_p=dropout_p,
        )

        if two_phase:
            phase1_fraction = trial.suggest_float("phase1_fraction", 0.2, 0.8)
            train_model_two_phase(
                model, x_tr, t_ds, y_tr,
                n_obs=train_n_obs, n_epochs=n_epochs, lr=lr,
                weight_decay=weight_decay, batch_size=batch_size,
                phase1_fraction=phase1_fraction,
                x_val=x_va, y_val=y_va,
            )
        elif weighted_random_nobs:
            train_model_weighted_random_nobs(
                model, x_tr, t_ds, y_tr,
                prior_weight=prior_weight, n_epochs=n_epochs, lr=lr,
                weight_decay=weight_decay, batch_size=batch_size,
                x_val=x_va, y_val=y_va,
            )
        elif random_nobs:
            train_model_random_nobs(
                model, x_tr, t_ds, y_tr,
                n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
                batch_size=batch_size, future_only=future_only,
                x_val=x_va, y_val=y_va,
            )
        elif prior_weight > 0.0 or future_only:
            train_model_custom(
                model, x_tr, t_ds, y_tr,
                n_obs=train_n_obs, n_epochs=n_epochs, lr=lr,
                weight_decay=weight_decay, batch_size=batch_size,
                prior_weight=prior_weight, future_only=future_only,
                x_val=x_va, y_val=y_va,
            )
        else:
            train_model(
                model, x_tr, t_ds, y_tr,
                n_epochs=n_epochs, n_obs=train_n_obs, lr=lr,
                weight_decay=weight_decay, batch_size=batch_size,
                verbose=False, x_val=x_va, y_val=y_va,
            )

        # Val metric: average of prior MSE and posterior MSE at train_n_obs
        model.eval()
        with torch.no_grad():
            prior_m = eval_prior(model, x_va, y_va, t_ds)
            post_m = eval_posterior(model, x_va, y_va, t_ds, min(train_n_obs, n_time - 1))
        return 0.5 * prior_m["mse"] + 0.5 * post_m["full_mse"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        sampler=sampler, direction="minimize",
        storage=storage, study_name=study_name, load_if_exists=True,
    )
    if seed_trials:
        for params in seed_trials:
            study.enqueue_trial(params)
    study.optimize(objective, n_trials=n_tune)

    best = study.best_trial.params
    best["hidden_sizes"] = [best.pop(f"h{i}") for i in range(3)]
    if prior_weight_fixed == 0.0:
        best["prior_weight"] = 0.0
    print(f"    Best hparams (val={study.best_value:.5f}): {best}")
    return best


def run_prior_vs_static(
    dataset_name: str,
    seed: int = 0,
    n_epochs: int = 1000,
    training_n_obs_values: list[int] | None = None,
    eval_n_obs_values: list[int] | None = None,
    kl_weight: float = 0.01,
    hidden_sizes: list[int] | None = None,
    random_obs: bool = False,
    prior_weight: float = 0.0,
    future_only: bool = False,
    random_nobs: bool = False,
    two_phase: bool = False,
    weighted_random_nobs: bool = False,
    hparams: dict | None = None,
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
        hp = hparams or {}
        _hidden = hp.get("hidden_sizes", hidden_sizes)
        _lr = hp.get("lr", 1e-3)
        _wd = hp.get("weight_decay", 1e-5)
        _bs = hp.get("batch_size", 32)
        _dp = hp.get("dropout_p", 0.2)
        _pw = hp.get("prior_weight", prior_weight)

        print(f"\n  Training adaptive model (train_n_obs={train_n_obs}, lr={_lr:.5f}, "
              f"wd={_wd:.2e}, bs={_bs}, hidden={_hidden}, prior_weight={_pw:.3f}, "
              f"random_nobs={random_nobs})...")
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
            hidden_sizes=_hidden,
            dropout_p=_dp,
        )
        _phase1_frac = hp.get("phase1_fraction", 0.5)

        if two_phase:
            train_model_two_phase(
                adaptive_model, x_tr, t_ds, y_tr,
                n_obs=train_n_obs, n_epochs=n_epochs, lr=_lr,
                weight_decay=_wd, batch_size=_bs,
                phase1_fraction=_phase1_frac,
                x_val=x_va, y_val=y_va,
            )
        elif weighted_random_nobs:
            train_model_weighted_random_nobs(
                adaptive_model, x_tr, t_ds, y_tr,
                prior_weight=_pw, n_epochs=n_epochs, lr=_lr,
                weight_decay=_wd, batch_size=_bs,
                x_val=x_va, y_val=y_va,
            )
        elif random_nobs:
            train_model_random_nobs(
                adaptive_model, x_tr, t_ds, y_tr,
                n_epochs=n_epochs, lr=_lr, weight_decay=_wd, batch_size=_bs,
                future_only=future_only, x_val=x_va, y_val=y_va,
            )
        elif random_obs:
            train_model_random_obs(
                adaptive_model, x_tr, t_ds, y_tr,
                n_obs=train_n_obs, n_epochs=n_epochs,
                x_val=x_va, y_val=y_va,
            )
        elif _pw > 0.0 or future_only:
            train_model_custom(
                adaptive_model, x_tr, t_ds, y_tr,
                n_obs=train_n_obs, n_epochs=n_epochs, lr=_lr,
                weight_decay=_wd, batch_size=_bs,
                prior_weight=_pw, future_only=future_only,
                x_val=x_va, y_val=y_va,
            )
        else:
            train_model(
                adaptive_model, x_tr, t_ds, y_tr,
                n_epochs=n_epochs, n_obs=train_n_obs, lr=_lr,
                weight_decay=_wd, batch_size=_bs,
                verbose=False, x_val=x_va, y_val=y_va,
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
        print(f"Posterior (eval_n_obs={eval_n}) full MSE vs Static:")
        print(f"  {'train_n_obs':>12} {'Post MSE':>12} {'vs Static':>12} {'Post CRPS':>12}")
        print(f"  {'-' * 52}")
        for train_n_obs, model_res in results["adaptive_models"].items():
            if eval_n in model_res["posterior"]:
                post = model_res["posterior"][eval_n]
                full_mse = post.get("full_mse", float("nan"))
                full_crps = post.get("full_crps", float("nan"))
                ratio = full_mse / static_mse
                print(f"  {train_n_obs:>12} {full_mse:>12.5f} {ratio:>11.3f}x {full_crps:>12.5f}")
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


def print_variant_comparison(all_variant_results: dict[str, dict]):
    """Print prior MSE ratios for all variants side by side, per dataset."""
    variant_names = list(all_variant_results.keys())
    # Collect datasets from first variant
    datasets = list(next(iter(all_variant_results.values())).keys())
    all_train_nobs = sorted({
        n
        for res in next(iter(all_variant_results.values())).values()
        for n in res["adaptive_models"]
    })

    for ds_name in datasets:
        static_mse = next(iter(all_variant_results.values()))[ds_name]["static"]["mse"]
        print(f"\n{ds_name.upper()} — Prior MSE / Static MSE (static={static_mse:.5f})")
        header = f"  {'train_n_obs':>12}" + "".join(f"  {v:>14}" for v in variant_names)
        print(header)
        print(f"  {'-' * (14 + 16 * len(variant_names))}")
        for train_n in all_train_nobs:
            row = f"  {train_n:>12}"
            for v_name, v_results in all_variant_results.items():
                if ds_name in v_results and train_n in v_results[ds_name]["adaptive_models"]:
                    ratio = v_results[ds_name]["adaptive_models"][train_n]["prior"]["mse"] / static_mse
                    row += f"  {ratio:>13.3f}x"
                else:
                    row += f"  {'N/A':>14}"
            print(row)


def plot_nobs_curve(
    variant_results: dict[str, dict],
    dataset_name: str,
    eval_n_obs_values: list[int],
    output_path: Path | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {
        "default": "#E53935", "prior_nll": "#2196F3", "random_nobs": "#4CAF50",
        "future_only": "#FF9800", "random_nobs_fo": "#9C27B0",
        "two_phase": "#00BCD4", "weighted_random": "#795548",
    }
    markers = {
        "default": "o", "prior_nll": "s", "random_nobs": "^",
        "future_only": "D", "random_nobs_fo": "v",
        "two_phase": "P", "weighted_random": "X",
    }

    for ax_idx, metric in enumerate(["mse", "crps"]):
        ax = axes[ax_idx]

        for v_name, res in variant_results.items():
            static_val = res["static"][metric]

            model_res = next(iter(res["adaptive_models"].values()))

            # n_obs=0 is the prior
            nobs_vals = [0]
            metric_vals = [model_res["prior"][metric]]

            for n in sorted(eval_n_obs_values):
                if n in model_res["posterior"]:
                    full_key = f"full_{metric}"
                    val = model_res["posterior"][n].get(full_key, float("nan"))
                    nobs_vals.append(n)
                    metric_vals.append(val)

            color = colors.get(v_name, "grey")
            marker = markers.get(v_name, "o")
            ax.plot(nobs_vals, metric_vals, f"{marker}-", color=color,
                    linewidth=2, markersize=6, label=v_name)

        # Static baseline from first variant
        first_res = next(iter(variant_results.values()))
        static_val = first_res["static"][metric]
        ax.axhline(y=static_val, color="black", linestyle="--", linewidth=1.5, label="static")

        ax.set_xlabel("n_obs at inference", fontsize=11)
        ax.set_ylabel(f"{metric.upper()} (full trajectory)", fontsize=11)
        ax.set_title(f"{dataset_name} — {metric.upper()} vs observations", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.suptitle(f"How performance scales with observations: {dataset_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path is None:
        output_path = Path(__file__).parent.parent / "figures" / f"nobs_curve_{dataset_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {output_path}")


def main():
    seed = 0
    datasets = ["airfoil", "flchain", "stress_strain"]

    # Single representative train_n_obs — large enough to show the degradation problem
    train_n_obs = 20

    # Dense eval sweep to show full n_obs curve at inference time
    eval_n_obs_values = [1, 2, 5, 10, 15, 20, 25, 30, 40]

    variants = {
        "default":         dict(n_epochs=1000, kl_weight=0.01, random_obs=False, random_nobs=False, prior_weight=0.0, future_only=False),
        "prior_nll":       dict(n_epochs=1000, kl_weight=0.01, random_obs=False, random_nobs=False, prior_weight=0.5, future_only=False),
        "random_nobs":     dict(n_epochs=1000, kl_weight=0.01, random_obs=False, random_nobs=True,  prior_weight=0.0, future_only=False),
        "two_phase":       dict(n_epochs=1000, kl_weight=0.01, random_obs=False, random_nobs=False, prior_weight=0.0, future_only=False, two_phase=True),
        "weighted_random": dict(n_epochs=1000, kl_weight=0.01, random_obs=False, random_nobs=False, prior_weight=0.5, future_only=False, weighted_random_nobs=True),
    }

    figures_dir = Path(__file__).parent.parent / "figures"
    db_path = f"sqlite:///{figures_dir.parent}/hpo.db"

    # Best hyperparameters from the previous run (lr in [1e-4, 1e-2])
    # These are enqueued so Optuna evaluates them and they compete with new trials
    old_best_hparams = {
        "default": {
            "airfoil":       {"lr": 0.00195, "weight_decay": 1.02e-6,  "batch_size": 64, "dropout_p": 0.337, "h0": 54,  "h1": 33,  "h2": 16},
            "flchain":       {"lr": 0.00874, "weight_decay": 8.86e-5,  "batch_size": 16, "dropout_p": 0.004, "h0": 28,  "h1": 20,  "h2": 82},
            "stress_strain": {"lr": 0.00231, "weight_decay": 1.80e-6,  "batch_size": 32, "dropout_p": 0.118, "h0": 98,  "h1": 74,  "h2": 61},
        },
        "prior_nll": {
            "airfoil":       {"lr": 0.00078, "weight_decay": 8.56e-5,  "batch_size": 16, "dropout_p": 0.077, "h0": 114, "h1": 17,  "h2": 102, "prior_weight": 1.303},
            "flchain":       {"lr": 0.00364, "weight_decay": 2.26e-6,  "batch_size": 64, "dropout_p": 0.209, "h0": 126, "h1": 106, "h2": 68,  "prior_weight": 0.346},
            "stress_strain": {"lr": 0.00710, "weight_decay": 1.63e-6,  "batch_size": 64, "dropout_p": 0.311, "h0": 105, "h1": 75,  "h2": 80,  "prior_weight": 1.355},
        },
        "random_nobs": {
            "airfoil":       {"lr": 0.00231, "weight_decay": 1.20e-5,  "batch_size": 32, "dropout_p": 0.267, "h0": 85,  "h1": 85,  "h2": 122},
            "flchain":       {"lr": 0.00077, "weight_decay": 5.58e-5,  "batch_size": 32, "dropout_p": 0.010, "h0": 111, "h1": 113, "h2": 18},
            "stress_strain": {"lr": 0.00081, "weight_decay": 1.67e-5,  "batch_size": 32, "dropout_p": 0.192, "h0": 115, "h1": 110, "h2": 115},
        },
        # New variants — seed with default hparams as starting point
        "future_only": {
            "airfoil":       {"lr": 0.00195, "weight_decay": 1.02e-6,  "batch_size": 64, "dropout_p": 0.337, "h0": 54,  "h1": 33,  "h2": 16},
            "flchain":       {"lr": 0.00874, "weight_decay": 8.86e-5,  "batch_size": 16, "dropout_p": 0.004, "h0": 28,  "h1": 20,  "h2": 82},
            "stress_strain": {"lr": 0.00231, "weight_decay": 1.80e-6,  "batch_size": 32, "dropout_p": 0.118, "h0": 98,  "h1": 74,  "h2": 61},
        },
        "random_nobs_fo": {
            "airfoil":       {"lr": 0.00231, "weight_decay": 1.20e-5,  "batch_size": 32, "dropout_p": 0.267, "h0": 85,  "h1": 85,  "h2": 122},
            "flchain":       {"lr": 0.00077, "weight_decay": 5.58e-5,  "batch_size": 32, "dropout_p": 0.010, "h0": 111, "h1": 113, "h2": 18},
            "stress_strain": {"lr": 0.00081, "weight_decay": 1.67e-5,  "batch_size": 32, "dropout_p": 0.192, "h0": 115, "h1": 110, "h2": 115},
        },
        # New variants — seed with default hparams + phase1_fraction=0.5 / prior_weight=0.5
        "two_phase": {
            "airfoil":       {"lr": 0.00195, "weight_decay": 1.02e-6,  "batch_size": 64, "dropout_p": 0.337, "h0": 54,  "h1": 33,  "h2": 16,  "phase1_fraction": 0.5},
            "flchain":       {"lr": 0.00874, "weight_decay": 8.86e-5,  "batch_size": 16, "dropout_p": 0.004, "h0": 28,  "h1": 20,  "h2": 82,  "phase1_fraction": 0.5},
            "stress_strain": {"lr": 0.00231, "weight_decay": 1.80e-6,  "batch_size": 32, "dropout_p": 0.118, "h0": 98,  "h1": 74,  "h2": 61,  "phase1_fraction": 0.5},
        },
        "weighted_random": {
            "airfoil":       {"lr": 0.00078, "weight_decay": 8.56e-5,  "batch_size": 16, "dropout_p": 0.077, "h0": 114, "h1": 17,  "h2": 102, "prior_weight": 1.303},
            "flchain":       {"lr": 0.00364, "weight_decay": 2.26e-6,  "batch_size": 64, "dropout_p": 0.209, "h0": 126, "h1": 106, "h2": 68,  "prior_weight": 0.346},
            "stress_strain": {"lr": 0.00710, "weight_decay": 1.63e-6,  "batch_size": 64, "dropout_p": 0.311, "h0": 105, "h1": 75,  "h2": 80,  "prior_weight": 1.355},
        },
    }

    n_tune = 5  # additional trials on top of the enqueued old best

    # Previous results with lr in [1e-4, 1e-2]
    old_results = {
        "default": {
            "airfoil":       {"prior_mse_ratio": 3.979},
            "flchain":       {"prior_mse_ratio": 2.658},
            "stress_strain": {"prior_mse_ratio": 1.273},
        },
        "prior_nll": {
            "airfoil":       {"prior_mse_ratio": 0.648},
            "flchain":       {"prior_mse_ratio": 1.147},
            "stress_strain": {"prior_mse_ratio": 0.838},
        },
        "random_nobs": {
            "airfoil":       {"prior_mse_ratio": 1.108},
            "flchain":       {"prior_mse_ratio": 1.928},
            "stress_strain": {"prior_mse_ratio": 1.084},
        },
    }

    # {variant_name: {dataset_name: results}}
    all_variant_results: dict[str, dict] = {v: {} for v in variants}

    for v_name, v_kwargs in variants.items():
        print("\n" + "=" * 80)
        print(f"VARIANT: {v_name}")
        print("=" * 80)
        for dataset_name in datasets:
            try:
                (
                    x_tr, x_va, x_te, t_ds, y_tr, y_va, y_te, y_norm,
                    knots_ds, n_basis_ds, ts_tr, ys_tr, ts_va, ys_va, ts_te, ys_te,
                ) = load_and_normalize_dataset(dataset_name, seed=seed)

                print(f"\n  Tuning {v_name} on {dataset_name} "
                      f"(1 seed + {n_tune} new trials, lr in [1e-4, 1e-1])...")
                seed_trial = old_best_hparams[v_name][dataset_name]
                sname = f"{v_name}_{dataset_name}"
                best_hparams = tune_adaptive_hyperparams(
                    v_kwargs, x_tr, y_tr, x_va, y_va, t_ds,
                    n_basis_ds, knots_ds, train_n_obs,
                    n_tune=n_tune, n_epochs=1000, seed=seed,
                    seed_trials=[seed_trial],
                    storage=db_path, study_name=sname,
                )

                results = run_prior_vs_static(
                    dataset_name, seed=seed,
                    training_n_obs_values=[train_n_obs],
                    eval_n_obs_values=eval_n_obs_values,
                    hidden_sizes=None,
                    hparams=best_hparams,
                    **v_kwargs,
                )
                all_variant_results[v_name][dataset_name] = results
                print_summary_table(results, dataset_name)
            except FileNotFoundError as e:
                print(f"\nSkipping {dataset_name}: {e}")

    print("\n" + "=" * 80)
    print("VARIANT COMPARISON (lr in [1e-4, 1e-1]): Prior MSE / Static MSE")
    print("=" * 80)
    print_variant_comparison(all_variant_results)

    # Compare old vs new for variants that have prior results
    print("\n" + "=" * 80)
    print("OLD (lr≤1e-2) vs NEW: Prior MSE / Static MSE ratio")
    print("=" * 80)
    for ds in datasets:
        print(f"\n{ds.upper()}")
        print(f"  {'variant':>16} {'old':>8} {'new':>8} {'diff':>8}")
        print(f"  {'-'*44}")
        for v_name in variants:
            if ds in all_variant_results[v_name] and v_name in old_results and ds in old_results[v_name]:
                new_ratio = (
                    next(iter(all_variant_results[v_name][ds]["adaptive_models"].values()))["prior"]["mse"]
                    / all_variant_results[v_name][ds]["static"]["mse"]
                )
                old_ratio = old_results[v_name][ds]["prior_mse_ratio"]
                diff = new_ratio - old_ratio
                print(f"  {v_name:>16} {old_ratio:>7.3f}x {new_ratio:>7.3f}x {diff:>+7.3f}")
            elif ds in all_variant_results[v_name]:
                new_ratio = (
                    next(iter(all_variant_results[v_name][ds]["adaptive_models"].values()))["prior"]["mse"]
                    / all_variant_results[v_name][ds]["static"]["mse"]
                )
                print(f"  {v_name:>16} {'N/A':>8} {new_ratio:>7.3f}x {'':>8}")

    # Plot n_obs curves per dataset
    for dataset_name in datasets:
        variant_ds = {
            v_name: all_variant_results[v_name][dataset_name]
            for v_name in variants
            if dataset_name in all_variant_results[v_name]
        }
        if variant_ds:
            plot_nobs_curve(
                variant_ds, dataset_name, eval_n_obs_values,
                output_path=figures_dir / f"nobs_curve_{dataset_name}_widerlr.png",
            )


if __name__ == "__main__":
    main()
