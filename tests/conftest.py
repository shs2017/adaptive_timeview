"""Add src/ to the import path so tests can import project modules."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def seed():
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def basic_dims():
    return {"input_dim": 4, "n_basis": 7}


@pytest.fixture
def sample_data(seed):
    """Generate a small synthetic dataset."""
    batch_size = 16
    input_dim = 4
    n_timepoints = 30

    x = torch.randn(batch_size, input_dim)
    t = torch.linspace(0, 1, n_timepoints)
    y = torch.randn(batch_size, n_timepoints)
    return x, t, y
