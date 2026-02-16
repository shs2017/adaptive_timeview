# TIMEVIEW-Adaptive

**Online Bayesian Adaptation for Interpretable Time Series Forecasting**

TIMEVIEW-Adaptive extends [TIMEVIEW](https://github.com/sjblim/timeview) ("Towards Transparent Time Series Forecasting") with online Bayesian adaptation. Instead of a fixed deterministic encoder, we use a probabilistic one that updates its beliefs as new observations arrive — producing calibrated uncertainty estimates and interpretable trajectory decompositions in a streaming setting.

---

## Key Results

<table>
<tr>
<td width="50%">
<img src="figures/transition_analysis.png" alt="Transition Analysis"/>
<p align="center"><b>Posterior Trajectory Analysis</b><br/><sub>Posterior samples, coefficient distributions, derivative uncertainty, and transition point detection from a single trajectory.</sub></p>
</td>
<td width="50%">
<img src="figures/uncertainty_decomposition.png" alt="Uncertainty Decomposition"/>
<p align="center"><b>Uncertainty Decomposition</b><br/><sub>As observations increase, epistemic uncertainty shrinks while the model gains information — 62% of prior uncertainty resolved with 20 observations.</sub></p>
</td>
</tr>
<tr>
<td width="50%">
<img src="figures/calibration_comparison.png" alt="Calibration Comparison"/>
<p align="center"><b>Calibration: Static vs Adaptive</b><br/><sub>Static TIMEVIEW is severely miscalibrated (left). Bayesian adaptation achieves near-perfect calibration (right), closely tracking the diagonal.</sub></p>
</td>
<td width="50%">
<img src="figures/active_scheduling.png" alt="Active Scheduling"/>
<p align="center"><b>Active Observation Scheduling</b><br/><sub>Information-gain-based active scheduling converges significantly faster than uniform spacing across all metrics.</sub></p>
</td>
</tr>
</table>

---

## Method Overview

<p align="center">
<img src="figures/bspline_basis.png" alt="B-Spline Basis Functions" width="80%"/>
</p>
<p align="center"><sub>Cubic B-spline basis functions used as the trajectory representation. Trajectories are expressed as weighted sums of these smooth, local basis functions.</sub></p>

The approach works in three stages:

1. **Prior from encoder** — A neural encoder maps baseline features to prior distribution parameters (mean and covariance) over B-spline coefficients
2. **Bayesian update** — As streaming observations arrive, closed-form conjugate updates refine the posterior over coefficients
3. **Uncertainty propagation** — The posterior over coefficients induces calibrated predictive distributions over future trajectory values

---

## Streaming Adaptation

<p align="center">
<img src="figures/streaming_evaluation.png" alt="Streaming Evaluation" width="40%"/>
</p>
<p align="center"><sub>All metrics improve monotonically as observations stream in: prediction error and CRPS decrease, uncertainty tightens, and calibration converges toward the 95% target.</sub></p>

---

## Comparisons

<table>
<tr>
<td width="50%">
<img src="figures/gp_comparison.png" alt="GP Comparison"/>
<p align="center"><b>GP vs TIMEVIEW-Adaptive vs Static</b><br/><sub>TIMEVIEW-Adaptive matches GP calibration while producing much sharper (tighter) intervals.</sub></p>
</td>
<td width="50%">
<img src="figures/dataset_comparison.png" alt="Dataset Comparison"/>
<p align="center"><b>Cross-Dataset Evaluation</b><br/><sub>Prediction error comparison across airfoil, flchain, and stress-strain datasets, with MSE improvement breakdown.</sub></p>
</td>
</tr>
</table>

---

## Ablation Studies

<table>
<tr>
<td width="50%">
<img src="figures/ablation_studies.png" alt="Ablation Studies"/>
<p align="center"><b>Hyperparameter Sensitivity</b><br/><sub>Effect of number of basis functions, observation count, and covariance parameterization on prediction error.</sub></p>
</td>
<td width="50%">
<img src="figures/best_config_comparison.png" alt="Best Config Comparison"/>
<p align="center"><b>Model Variant Comparison</b><br/><sub>Standard, gated, heteroscedastic, and best-config variants compared on MSE, calibration, predictive efficiency, and CRPS.</sub></p>
</td>
</tr>
<tr>
<td width="50%">
<img src="figures/fix_ablation_studies.png" alt="Fix Ablation Studies"/>
<p align="center"><b>Training Fix Ablation</b><br/><sub>Cumulative impact of batchnorm, dropout, KL regularization, learned noise, early stopping, and weight decay on prediction error and calibration.</sub></p>
</td>
<td width="50%">
<img src="figures/nobs_ablation_all_datasets.png" alt="Nobs Ablation"/>
<p align="center"><b>Adaptation Benefit vs Observation Count</b><br/><sub>MSE improvement over static baseline as a function of the number of observations, across all three datasets.</sub></p>
</td>
</tr>
</table>

---

## Additional Analysis

<table>
<tr>
<td width="50%">
<img src="figures/heteroscedastic_comparison.png" alt="Noise Model Comparison"/>
<p align="center"><b>Noise Model Comparison</b><br/><sub>Homoscedastic, heteroscedastic, and gated noise models compared on prediction error, calibration, and calibration error.</sub></p>
</td>
<td width="50%">
<img src="figures/temperature_scaling.png" alt="Temperature Scaling"/>
<p align="center"><b>Temperature Scaling</b><br/><sub>Post-hoc temperature scaling reduces calibration error from 0.090 to 0.034.</sub></p>
</td>
</tr>
<tr>
<td colspan="2" align="center">
<img src="figures/aec_comparison.png" alt="Adaptation Efficiency" width="60%"/>
<p align="center"><b>Adaptation Efficiency Curves</b><br/><sub>How fast each model variant learns from new observations. The best configuration achieves the lowest AAEC (0.0118).</sub></p>
</td>
</tr>
</table>

---

## Installation

```bash
chmod +x setup.sh
./setup.sh
```

## Reproducing Results

```bash
uv run src/timeview_adaptive_experiments.py
```

## Development

```bash
# Lint
uv run ruff check src/

# Format
uv run ruff format src/

# Test
uv run pytest tests/
```

## Project Structure

```
src/
  timeview_adaptive.py      # Core adaptive model
  timeview_extended.py      # Extended TIMEVIEW base
  models.py                 # Neural network architectures
  train.py                  # Training loop
  evaluation.py             # Evaluation metrics
  scores.py                 # Scoring functions (CRPS, calibration)
  data.py                   # Dataset loading
  ablation.py               # Ablation study experiments
  plot.py                   # Plotting utilities
  visualize.py              # Visualization helpers
  transition_point.py       # Transition point detection
  knot_selection.py         # Knot placement for B-splines
  active_scheduling.py      # Active observation scheduling (if applicable)
  dash_app.py               # Interactive dashboard
  generate_animations.py    # Animation generation
figures/                    # All experimental result figures
tests/                      # Unit tests
```
