from pathlib import Path

import numpy as np
import torch

from constant import DATA_DIR
from knot_selection import calculate_knot_placement
from timeview_adaptive import create_knots


class DatasetLoader:

    def load(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[np.ndarray], list[np.ndarray]]:
        raise NotImplementedError


class AirfoilLoader(DatasetLoader):

    def __init__(self, data_path: Path | None = None, log_t: bool = True):
        self.data_path = data_path or DATA_DIR / "airfoil" / "airfoil_self_noise.dat"
        self.log_t = log_t

    def load(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[np.ndarray], list[np.ndarray]]:
        import pandas as pd

        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Airfoil data not found at {self.data_path}")

        df = pd.read_csv(path, sep="\t", header=None)
        df.columns = ["t", "angle", "chord", "velocity", "thickness", "y"]

        df["id"] = 0
        prev = float("inf")
        curr_id = 0
        for idx, row in df.iterrows():
            if row["t"] < prev:
                curr_id += 1
            prev = row["t"]
            df.at[idx, "id"] = curr_id

        df = df[["id", "angle", "chord", "velocity", "thickness", "t", "y"]]

        df["t"] = df["t"] / 200
        if self.log_t:
            df["t"] = np.log(df["t"])
        else:
            df["t"] = df["t"] / 100

        X_list, t_list, y_list = [], [], []
        for sample_id in df["id"].unique():
            sample_df = df[df["id"] == sample_id].copy()
            static = sample_df[["angle", "chord", "velocity", "thickness"]].iloc[0].values
            X_list.append(static)
            sample_df.sort_values(by="t", inplace=True)
            t_list.append(sample_df["t"].values.astype(np.float64))
            y_list.append(sample_df["y"].values.astype(np.float64))

        n_timepoints = 50
        t_max = max(t[-1] for t in t_list)
        t_min = min(t[0] for t in t_list)
        t_common = np.linspace(t_min, t_max, n_timepoints)
        y_interp = []
        for t_sample, y_sample in zip(t_list, y_list, strict=True):
            y_interp.append(np.interp(t_common, t_sample, y_sample))

        if self.log_t:
            t_out = t_common
            ts_out = t_list
        else:
            t_out = (t_common - t_min) / (t_max - t_min)
            ts_out = [(ts - t_min) / (t_max - t_min) for ts in t_list]

        X = np.array(X_list, dtype=np.float32)
        y_array = np.array(y_interp, dtype=np.float32)

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(t_out, dtype=torch.float32),
            torch.tensor(y_array, dtype=torch.float32),
            ts_out,
            y_list,
        )


class FLChainLoader(DatasetLoader):
    FLC_GRP_CATEGORIES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def __init__(self, data_path: Path | None = None, subset: int | str = 1000):
        self.data_path = data_path or DATA_DIR / "flchain" / "flchain.csv"
        self.subset = subset

    def load(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[np.ndarray], list[np.ndarray]]:
        import pandas as pd

        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"FLChain data not found at {self.data_path}")

        df = pd.read_csv(path)

        feature_cols = ["age", "sex", "creatinine", "kappa", "lambda", "flc.grp", "mgus"]

        df = df[["id"] + feature_cols + ["t", "y"]]

        if not pd.api.types.is_numeric_dtype(df["sex"]):
            df["sex"] = df["sex"].map({"M": 1, "F": 0}).astype(float)
        if not pd.api.types.is_numeric_dtype(df["mgus"]):
            df["mgus"] = df["mgus"].map({"yes": 1, "no": 0}).astype(float)

        df["t"] = df["t"] / 5000

        if self.subset != "all":
            all_ids = df["id"].unique()
            gen = np.random.default_rng(0)
            subset_size = min(int(self.subset), len(all_ids))
            selected_ids = gen.choice(all_ids, size=subset_size, replace=False)
            df = df[df["id"].isin(selected_ids)]

        X_list, t_list, y_list = [], [], []
        for sample_id in df["id"].unique():
            sample_df = df[df["id"] == sample_id].copy()
            row = sample_df.iloc[0]
            age = [float(row["age"])]
            sex = [float(row["sex"])]
            creatinine = [float(row["creatinine"])]
            kappa = [float(row["kappa"])]
            lam = [float(row["lambda"])]
            flc_grp_val = int(row["flc.grp"])
            flc_one_hot = [1.0 if flc_grp_val == cat else 0.0 for cat in self.FLC_GRP_CATEGORIES]
            mgus = [float(row["mgus"])]
            X_list.append(age + sex + creatinine + kappa + lam + flc_one_hot + mgus)
            sample_df.sort_values(by="t", inplace=True)
            t_list.append(sample_df["t"].values.astype(np.float64))
            y_list.append(sample_df["y"].values.astype(np.float64))

        n_timepoints = 50
        t_max = max(t[-1] for t in t_list)
        t_min = min(t[0] for t in t_list)
        t_common = np.linspace(t_min, t_max, n_timepoints)
        y_interp = []
        for t_sample, y_sample in zip(t_list, y_list, strict=True):
            y_interp.append(np.interp(t_common, t_sample, y_sample))

        t_normalized = (t_common - t_min) / (t_max - t_min + 1e-8)

        ts_normalized = [(ts - t_min) / (t_max - t_min + 1e-8) for ts in t_list]

        X = np.array(X_list, dtype=np.float32)

        y_array = np.array(y_interp, dtype=np.float32)

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(t_normalized, dtype=torch.float32),
            torch.tensor(y_array, dtype=torch.float32),
            ts_normalized,
            y_list,
        )


class StressStrainLoader(DatasetLoader):
    ALL_LOTS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

    def __init__(
        self,
        data_path: Path | None = None,
        lot: str = "all",
        max_strain: float = 0.2,
        downsample: bool = True,
        include_lot_as_feature: bool = True,
    ):
        self.data_path = data_path or DATA_DIR / "stress-strain-curves"
        self.lot = lot
        self.max_strain = max_strain
        self.downsample = downsample
        self.include_lot_as_feature = include_lot_as_feature

    def load(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[np.ndarray], list[np.ndarray]]:
        import glob

        import pandas as pd

        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Stress-strain data not found at {self.data_path}")

        if self.lot == "all":
            pattern = str(path / "T_*.csv")
        else:
            pattern = str(path / f"T_*{self.lot}*.csv")

        filenames = glob.glob(pattern)
        if not filenames:
            raise FileNotFoundError(f"No stress-strain CSV files found matching {pattern}")

        X_list, t_list, y_list = [], [], []

        for filename in filenames:
            df = pd.read_csv(filename)
            df.columns = ["t", "y"]

            df.drop(df.tail(1).index, inplace=True)

            if self.downsample:
                df = df.iloc[::3, :]

            basename = Path(filename).stem.split("T_")[1]
            parts = basename.split("_")
            temp = float(parts[0])
            lot = parts[1]

            df = df[df["t"] >= 0]
            df = df[df["y"] >= 0]
            df = df[df["t"] <= self.max_strain]

            if len(df) < 5:
                continue

            df["y"] = df["y"] / 300

            if self.include_lot_as_feature:
                one_hot = [1.0 if lot == current_lot else 0.0 for current_lot in self.ALL_LOTS]
                X_list.append([temp] + one_hot)
            else:
                X_list.append([temp])

            t_raw = df["t"].values.astype(np.float64)
            y_raw = df["y"].values.astype(np.float64)
            sort_idx = np.argsort(t_raw)
            t_sorted = t_raw[sort_idx]
            y_sorted = y_raw[sort_idx]
            _, unique_idx = np.unique(t_sorted, return_index=True)
            t_list.append(t_sorted[unique_idx])
            y_list.append(y_sorted[unique_idx])

        n_timepoints = 50
        t_common = np.linspace(0, self.max_strain, n_timepoints)
        y_interp = []
        for t_sample, y_sample in zip(t_list, y_list, strict=True):
            t_sorted = t_sample
            y_sorted = y_sample

            t_sorted = np.clip(t_sorted, 0, self.max_strain)
            y_interp.append(np.interp(t_common, t_sorted, y_sorted))

        X = np.array(X_list, dtype=np.float32)
        y_array = np.array(y_interp, dtype=np.float32)

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(t_common, dtype=torch.float32),
            torch.tensor(y_array, dtype=torch.float32),
            t_list,
            y_list,
        )


# TODO: Implement this
# class TacrolimusLoader(DatasetLoader):
#     def __init__(self, data_path: Path | None = None):
#         self.data_path = data_path or DATA_DIR / "tacrolimus"

#     def load(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         path = Path(self.data_path)

#         if not path.exists() or not (path / "tac_pccp_mr4_250423.csv").exists():
#             raise FileNotFoundError(
#                 f"Tacrolimus data not found at {self.data_path}. "
#                 "This dataset is not publicly available. "
#                 "Contact TIMEVIEW paper authors to obtain the dataset."
#             )
#         raise NotImplementedError("Real Tacrolimus data loading not implemented")


class PhysioNetLoader(DatasetLoader):
    def __init__(self, data_path: Path | None = None, vital: str = "HR"):
        self.data_path = (
            data_path or DATA_DIR / "physionet.org" / "files" / "challenge-2012" / "1.0.0"
        )
        self.vital = vital

    def load(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[np.ndarray], list[np.ndarray]]:
        import glob

        import pandas as pd

        path = Path(self.data_path)

        set_b_path = path / "phase1" / "set-b"
        if not set_b_path.exists():
            raise FileNotFoundError(
                f"PhysioNet data not found at {set_b_path}. "
                "Download from: https://physionet.org/content/challenge-2012/1.0.0/"
            )

        patient_files = glob.glob(str(set_b_path / "*.txt"))
        if not patient_files:
            raise FileNotFoundError(
                f"No patient files found in {set_b_path}. "
                "Ensure PhysioNet Challenge 2012 data is properly downloaded."
            )

        X_list, t_list, y_list = [], [], []
        static_features = ["Age", "Gender", "Height", "Weight"]

        for pfile in patient_files[:300]:
            try:
                df = pd.read_csv(pfile)

                static_df = df[df["Time"] == "00:00"]
                static_vals = []
                for feat in static_features:
                    val = static_df[static_df["Parameter"] == feat]["Value"].values
                    static_vals.append(float(val[0]) if len(val) > 0 else np.nan)

                if any(np.isnan(static_vals)):
                    continue

                vital_df = df[df["Parameter"] == self.vital].copy()
                if len(vital_df) < 5:
                    continue

                def time_to_hours(t_str):
                    parts = t_str.split(":")
                    return float(parts[0]) + float(parts[1]) / 60

                vital_df["t_hours"] = vital_df["Time"].apply(time_to_hours)
                vital_df = vital_df.sort_values("t_hours")

                X_list.append(static_vals)
                t_list.append(vital_df["t_hours"].values.astype(np.float64))
                y_list.append(vital_df["Value"].values.astype(np.float64))

            except Exception:
                continue

        if len(X_list) < 10:
            raise ValueError(
                f"Too few valid PhysioNet samples ({len(X_list)}). "
                "Check that the data is correctly formatted."
            )

        n_timepoints = 50
        t_max = max(t[-1] for t in t_list)
        t_common = np.linspace(0, t_max, n_timepoints)
        y_interp = []
        for t_sample, y_sample in zip(t_list, y_list, strict=True):
            y_interp.append(np.interp(t_common, t_sample, y_sample))

        t_normalized = t_common / t_max
        X = np.array(X_list, dtype=np.float32)
        y_array = np.array(y_interp, dtype=np.float32)

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(t_normalized, dtype=torch.float32),
            torch.tensor(y_array, dtype=torch.float32),
            t_list,
            y_list,
        )


def load_dataset(name: str, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[np.ndarray], list[np.ndarray]]:
    loaders = {
        "airfoil": lambda: AirfoilLoader(
            data_path=kwargs.get("data_path"),
            log_t=kwargs.get("log_t", True),
        ).load(),
        "flchain": lambda: FLChainLoader(
            data_path=kwargs.get("data_path"),
            subset=kwargs.get("subset", 1000),
        ).load(),
        "stress_strain": lambda: StressStrainLoader(
            data_path=kwargs.get("data_path"),
            lot=kwargs.get("lot", "all"),
            max_strain=kwargs.get("max_strain", 0.2),
            downsample=kwargs.get("downsample", True),
            include_lot_as_feature=kwargs.get("include_lot_as_feature", True),
        ).load(),
        # "tacrolimus": lambda: TacrolimusLoader(kwargs.get("data_path")).load(),
        "physionet": lambda: PhysioNetLoader(
            data_path=kwargs.get("data_path"),
            vital=kwargs.get("vital", "HR"),
        ).load(),
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")

    return loaders[name]()


DATASET_CONTINUOUS_INDICES = {
    "airfoil": [0, 1, 2, 3],  # angle, chord, velocity, thickness
    "flchain": [0, 2, 3, 4],  # age, creatinine, kappa, lambda
    "stress_strain": [0],  # temperature only (lot is one-hot)
    "physionet": [0, 1, 2, 3],  # Age, Gender, Height, Weight
}

DATASET_N_BASIS = {
    "airfoil": 9,
    "flchain": 9,
    "stress_strain": 9,
}

DATASET_USE_DATA_DRIVEN_KNOTS = {
    "airfoil": True,
    "flchain": True,
    "stress_strain": True,
}

# Per-dataset T override for basis function range.
# TIMEVIEW uses T=1.0 for stress-strain even though data lives in [0, 0.2].
# If not specified, T defaults to the last time point in the data.
DATASET_T = {
    "stress_strain": 1.0,
}


def normalize_features(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    continuous_indices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_train = x_train.clone()
    x_test = x_test.clone()

    if continuous_indices is None:
        continuous_indices = list(range(x_train.shape[1]))

    if len(continuous_indices) > 0:
        idx = continuous_indices
        mean = x_train[:, idx].mean(dim=0)
        std = x_train[:, idx].std(dim=0).clamp(min=1e-8)
        x_train[:, idx] = (x_train[:, idx] - mean) / std
        x_test[:, idx] = (x_test[:, idx] - mean) / std

    return x_train, x_test


class YNormalizer:
    def __init__(self):
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def fit(self, y_train: torch.Tensor | list[np.ndarray]) -> "YNormalizer":
        if isinstance(y_train, list):
            Y = np.concatenate(y_train)
            self.y_mean = float(np.mean(Y))
            self.y_std = float(np.std(Y))
        else:
            self.y_mean = y_train.mean().item()
            self.y_std = y_train.std().item()
        if self.y_std < 1e-8:
            self.y_std = 1.0
        return self

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean) / self.y_std

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std + self.y_mean


def load_and_normalize_dataset(
    name: str, seed: int = 0, compute_knots: bool = True, **kwargs
) -> tuple:
    x, t, y, ts_raw, ys_raw = load_dataset(name, **kwargs)

    n_samples = x.shape[0]
    gen = np.random.default_rng(seed)
    train_indices = gen.choice(n_samples, int(n_samples * 0.7), replace=False)
    train_indices = [i.item() for i in train_indices]
    val_indices = gen.choice(
        list(set(range(n_samples)) - set(train_indices)),
        int(n_samples * 0.15),
        replace=False,
    )
    val_indices = [i.item() for i in val_indices]
    test_indices = list(set(range(n_samples)) - set(train_indices) - set(val_indices))

    x_train, x_val, x_test = x[train_indices], x[val_indices], x[test_indices]
    y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]

    ts_train_raw = [ts_raw[i] for i in train_indices]
    ys_train_raw = [ys_raw[i] for i in train_indices]
    ts_val_raw = [ts_raw[i] for i in val_indices]
    ys_val_raw = [ys_raw[i] for i in val_indices]
    ts_test_raw = [ts_raw[i] for i in test_indices]
    ys_test_raw = [ys_raw[i] for i in test_indices]

    continuous_idx = DATASET_CONTINUOUS_INDICES.get(name, list(range(x.shape[1])))
    x_train_norm = x_train.clone()
    x_val_norm = x_val.clone()
    x_test_norm = x_test.clone()
    if len(continuous_idx) > 0:
        idx = continuous_idx
        mean_x = x_train[:, idx].mean(dim=0)
        std_x = x_train[:, idx].std(dim=0).clamp(min=1e-8)
        x_train_norm[:, idx] = (x_train[:, idx] - mean_x) / std_x
        x_val_norm[:, idx] = (x_val[:, idx] - mean_x) / std_x
        x_test_norm[:, idx] = (x_test[:, idx] - mean_x) / std_x

    y_normalizer = YNormalizer().fit(ys_train_raw)
    y_train_norm = y_normalizer.transform(y_train)
    y_val_norm = y_normalizer.transform(y_val)
    y_test_norm = y_normalizer.transform(y_test)

    y_mean, y_std = y_normalizer.y_mean, y_normalizer.y_std
    ts_train = ts_train_raw
    ys_train = [(ys - y_mean) / y_std for ys in ys_train_raw]
    ts_val = ts_val_raw
    ys_val = [(ys - y_mean) / y_std for ys in ys_val_raw]
    ts_test = ts_test_raw
    ys_test = [(ys - y_mean) / y_std for ys in ys_test_raw]

    knots_tensor = None
    n_basis = DATASET_N_BASIS.get(name, 5)
    use_data_driven = DATASET_USE_DATA_DRIVEN_KNOTS.get(name, False)
    if compute_knots:
        t_np = t.numpy()
        n_internal_knots = n_basis - 3 + 1

        T = DATASET_T.get(name, float(t_np[-1]))
        t_min = float(t_np[0])
        if use_data_driven:
            try:
                internal_knots = calculate_knot_placement(ts_train, ys_train, n_internal_knots, T, seed=seed)
                knots_tensor = create_knots(n_basis, t_min=t_min, t_max=T, internal_knots=internal_knots)
            except Exception as e:
                print(f"  Warning: data-driven knot placement failed ({e}), using uniform knots")
                knots_tensor = create_knots(n_basis, t_min=t_min, t_max=T)
        else:
            knots_tensor = create_knots(n_basis, t_min=t_min, t_max=T)

    return (
        x_train_norm, x_val_norm, x_test_norm, t, y_train_norm, y_val_norm, y_test_norm,
        y_normalizer, knots_tensor, n_basis,
        ts_train, ys_train, ts_val, ys_val, ts_test, ys_test,
    )
