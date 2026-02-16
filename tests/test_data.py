import pytest
import torch

from data import (
    AirfoilLoader,
    DatasetLoader,
    FLChainLoader,
    StressStrainLoader,
    load_dataset,
)


class TestDataLoaders:
    def test_load_dataset_invalid_name(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_dataset")

    def test_airfoil_loader_shapes(self):
        try:
            loader = AirfoilLoader()
            x, t, y, ts_raw, ys_raw = loader.load()
        except FileNotFoundError:
            pytest.skip("Airfoil data not available")

        assert x.ndim == 2, f"x should be 2D, got {x.ndim}D"
        assert t.ndim == 1, f"t should be 1D, got {t.ndim}D"
        assert y.ndim == 2, f"y should be 2D, got {y.ndim}D"
        assert x.shape[0] == y.shape[0], "x and y batch sizes should match"
        assert y.shape[1] == len(t), "y time dim should match t"
        assert x.shape[1] == 4, f"Airfoil should have 4 features, got {x.shape[1]}"
        assert t.min() >= 0, "Time should be non-negative"

    def test_flchain_loader_shapes(self):
        try:
            loader = FLChainLoader(subset=50)
            x, t, y, _, _ = loader.load()
        except FileNotFoundError:
            pytest.skip("FLChain data not available")

        assert x.ndim == 2
        assert t.ndim == 1
        assert y.ndim == 2
        assert x.shape[1] == 16, f"FLChain should have 16 features, got {x.shape[1]}"

    def test_flchain_one_hot_encoding(self):
        try:
            loader = FLChainLoader(subset=50)
            x, _, _, _, _ = loader.load()
        except FileNotFoundError:
            pytest.skip("FLChain data not available")

        one_hot = x[:, 5:15]
        row_sums = one_hot.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums)), \
            f"One-hot rows should sum to 1, got: {row_sums[:5]}"
        assert ((one_hot == 0) | (one_hot == 1)).all(), \
            "One-hot values should be 0 or 1"

    def test_stress_strain_loader_shapes(self):
        try:
            loader = StressStrainLoader(include_lot_as_feature=True)
            x, t, y, _, _ = loader.load()
        except FileNotFoundError:
            pytest.skip("Stress-strain data not available")

        assert x.ndim == 2
        assert t.ndim == 1
        assert y.ndim == 2
        assert x.shape[1] == 10, f"Stress-strain should have 10 features, got {x.shape[1]}"

    def test_stress_strain_lot_one_hot(self):
        try:
            loader = StressStrainLoader(include_lot_as_feature=True)
            x, _, _, _, _ = loader.load()
        except FileNotFoundError:
            pytest.skip("Stress-strain data not available")

        one_hot = x[:, 1:10]
        row_sums = one_hot.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums)), \
            f"Lot one-hot rows should sum to 1, got: {row_sums[:5]}"

    def test_stress_strain_without_lot(self):
        try:
            loader = StressStrainLoader(include_lot_as_feature=False)
            x, _, _, _, _ = loader.load()
        except FileNotFoundError:
            pytest.skip("Stress-strain data not available")

        assert x.shape[1] == 1, f"Without lot, should have 1 feature, got {x.shape[1]}"


class TestLoadDatasetDispatch:
    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_dataset")

    def test_airfoil_dispatch(self):
        try:
            x, t, y, _, _ = load_dataset("airfoil")
        except FileNotFoundError:
            pytest.skip("Airfoil data not available")
        assert x.ndim == 2 and t.ndim == 1 and y.ndim == 2

    def test_flchain_dispatch(self):
        try:
            x, t, y, _, _ = load_dataset("flchain", subset=30)
        except FileNotFoundError:
            pytest.skip("FLChain data not available")
        assert x.shape[1] == 16

    def test_stress_strain_dispatch(self):
        try:
            x, t, y, _, _ = load_dataset("stress_strain")
        except FileNotFoundError:
            pytest.skip("Stress-strain data not available")
        assert x.shape[1] == 10

    def test_tacrolimus_not_implemented(self):
        with pytest.raises((FileNotFoundError, NotImplementedError, ValueError)):
            load_dataset("tacrolimus")

    def test_available_dataset_names(self):
        for name in ["airfoil", "flchain", "stress_strain", "physionet"]:
            try:
                load_dataset(name)
            except (FileNotFoundError, NotImplementedError, ValueError) as e:
                if "Unknown dataset" in str(e):
                    raise


class TestDatasetLoaderBase:
    def test_base_class_raises_not_implemented(self):
        loader = DatasetLoader()
        with pytest.raises(NotImplementedError):
            loader.load()
