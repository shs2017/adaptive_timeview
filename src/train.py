import numpy as np
import torch
import torch.nn as nn

from models import TimeviewStatic
from timeview_adaptive import (
    bspline_basis,
)


def train_static_model(
    model: TimeviewStatic,
    x_train: torch.Tensor,
    t: torch.Tensor,
    y_train: torch.Tensor,
    n_epochs: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    patience: int = 10,
    check_val_every_n_epoch: int = 10,
    x_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    ts_train: list[np.ndarray] | None = None,
    ys_train: list[np.ndarray] | None = None,
    ts_val: list[np.ndarray] | None = None,
    ys_val: list[np.ndarray] | None = None,
) -> list[float]:
    """
    Train the static baseline model matching TIMEVIEW's training procedure.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    n_samples = x_train.shape[0]

    use_per_sample = ts_train is not None and ys_train is not None
    if use_per_sample:
        Phis_train = [
            bspline_basis(torch.tensor(ts, dtype=torch.float32), model.knots)
            for ts in ts_train
        ]
        ys_train_t = [torch.tensor(ys, dtype=torch.float32) for ys in ys_train]

        if ts_val is not None and ys_val is not None:
            Phis_val = [
                bspline_basis(torch.tensor(ts, dtype=torch.float32), model.knots)
                for ts in ts_val
            ]
            ys_val_t = [torch.tensor(ys, dtype=torch.float32) for ys in ys_val]
        else:
            Phis_val, ys_val_t = Phis_train, ys_train_t
        x_eval = x_val if x_val is not None else x_train
    else:
        x_eval = x_val if x_val is not None else x_train
        y_eval = y_val if y_val is not None else y_train

    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start : start + batch_size]
            if len(batch_idx) < 2:
                continue
            x_batch = x_train[batch_idx]

            optimizer.zero_grad()
            if use_per_sample:
                Phis_batch = [Phis_train[i] for i in batch_idx]
                ys_batch = [ys_train_t[i] for i in batch_idx]
                loss = model.loss_per_sample(x_batch, Phis_batch, ys_batch)
            else:
                y_batch = y_train[batch_idx]
                loss = model.loss(x_batch, t, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % check_val_every_n_epoch == 0:
            model.eval()
            with torch.no_grad():
                if use_per_sample:
                    val_loss = model.loss_per_sample(x_eval, Phis_val, ys_val_t)
                else:
                    val_loss = model.loss(x_eval, t, y_eval)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience > 0 and epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return losses


def tune_static_hyperparams(
    input_dim: int,
    n_basis: int,
    knots: torch.Tensor,
    x_train: torch.Tensor,
    t: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    ts_train: list[np.ndarray],
    ys_train: list[np.ndarray],
    ts_val: list[np.ndarray],
    ys_val: list[np.ndarray],
    n_tune: int = 100,
    n_epochs: int = 1000,
    seed: int = 0,
) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        hidden_sizes = [
            trial.suggest_int(f"hidden_size_{i}", 16, 128) for i in range(3)
        ]
        dropout_p = trial.suggest_float("dropout_p", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)

        torch.manual_seed(seed)
        model = TimeviewStatic(
            input_dim=input_dim,
            n_basis=n_basis,
            hidden_sizes=hidden_sizes,
            dropout_p=dropout_p,
            knots=knots,
        )
        train_static_model(
            model, x_train, t, y_train,
            n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size,
            x_val=x_val, y_val=y_val,
            ts_train=ts_train, ys_train=ys_train,
            ts_val=ts_val, ys_val=ys_val,
        )

        model.eval()
        with torch.no_grad():
            Phis_val = [
                bspline_basis(torch.tensor(ts, dtype=torch.float32), model.knots)
                for ts in ts_val
            ]
            ys_val_t = [torch.tensor(ys, dtype=torch.float32) for ys in ys_val]
            preds = model.predict_per_sample(x_val, Phis_val)
            val_mse = float(np.mean([
                nn.functional.mse_loss(pred, y_true).item()
                for pred, y_true in zip(preds, ys_val_t, strict=True)
            ]))
        return val_mse

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=n_tune)

    best = study.best_trial.params
    print(f"    Best hyperparameters (val MSE={study.best_value:.6f}):")
    print(f"      hidden_sizes=[{best['hidden_size_0']}, {best['hidden_size_1']}, {best['hidden_size_2']}]")
    print(f"      dropout_p={best['dropout_p']:.3f}, lr={best['lr']:.6f}")
    print(f"      batch_size={best['batch_size']}, weight_decay={best['weight_decay']:.6f}")
    return best


def train_static_with_params(
    params: dict,
    input_dim: int,
    n_basis: int,
    knots: torch.Tensor,
    x_train: torch.Tensor,
    t: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    ts_train: list[np.ndarray],
    ys_train: list[np.ndarray],
    ts_val: list[np.ndarray],
    ys_val: list[np.ndarray],
    n_epochs: int = 1000,
    seed: int = 0,
) -> TimeviewStatic:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TimeviewStatic(
        input_dim=input_dim,
        n_basis=n_basis,
        hidden_sizes=[params[f"hidden_size_{i}"] for i in range(3)],
        dropout_p=params["dropout_p"],
        knots=knots,
    )
    train_static_model(
        model, x_train, t, y_train,
        n_epochs=n_epochs, lr=params["lr"],
        weight_decay=params["weight_decay"],
        batch_size=params["batch_size"],
        x_val=x_val, y_val=y_val,
        ts_train=ts_train, ys_train=ys_train,
        ts_val=ts_val, ys_val=ys_val,
    )
    return model



def tune_and_train_static_model(
    input_dim: int,
    n_basis: int,
    knots: torch.Tensor,
    x_train: torch.Tensor,
    t: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    ts_train: list[np.ndarray],
    ys_train: list[np.ndarray],
    ts_val: list[np.ndarray],
    ys_val: list[np.ndarray],
    n_tune: int = 100,
    n_epochs: int = 1000,
    seed: int = 0,
) -> TimeviewStatic:
    best = tune_static_hyperparams(
        input_dim, n_basis, knots, x_train, t, y_train,
        x_val, y_val, ts_train, ys_train, ts_val, ys_val,
        n_tune=n_tune, n_epochs=n_epochs, seed=seed,
    )
    return train_static_with_params(
        best, input_dim, n_basis, knots, x_train, t, y_train,
        x_val, y_val, ts_train, ys_train, ts_val, ys_val,
        n_epochs=n_epochs, seed=seed,
    )
