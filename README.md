# TIMEVIEW-Adaptive

Online Bayesian Adaptation for Interpretable Time Series Forecasting.

## Overview

This project extends TIMEVIEW ("Towards Transparent Time Series Forecasting") with online Bayesian adaptation. The key innovation is replacing the deterministic encoder with a probabilistic one that:

1. Outputs prior distribution parameters from baseline features
2. Performs closed-form Bayesian updates as observations arrive
3. Provides uncertainty over trajectory compositions

## Installation

```bash
uv venv
uv pip install -e ".[dev]"
```

## Development

```bash
# Lint
uv run ruff check src/

# Format
uv run ruff format src/
```
