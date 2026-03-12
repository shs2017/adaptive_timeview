# TIMEVIEW-Adaptive

**Online Bayesian Adaptation for Interpretable Time Series Forecasting**

TIMEVIEW-Adaptive extends [TIMEVIEW](https://github.com/krzysztof-kacprzyk/TIMEVIEW/) ("Towards Transparent Time Series Forecasting") with online Bayesian adaptation. We replace the deterministic encoder with a probabilistic one that outputs prior distributions $p(\mathbf{c}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)$ over basis function coefficients, then performs *closed-form* Bayesian updates as new observations arrive. This preserves TIMEVIEW's interpretable B-spline structure while adding: (i) uncertainty quantification over trajectory compositions, (ii) monotonic uncertainty reduction with each observation, and (iii) active observation scheduling via Bayesian experimental design. We further introduce *heteroscedastic noise modelling* for time-varying measurement uncertainty and an *adaptive gating mechanism* that learns when to trust Bayesian updates.

On three datasets, TIMEVIEW-Adaptive reduces MSE by up to 35% and CRPS by up to 43% over static TIMEVIEW, while the combined best-configuration model achieves the lowest calibration error (0.035) with near-perfect 95% coverage.

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
<p align="center"><b>Uncertainty Decomposition</b><br/><sub>Epistemic-aleatoric decomposition at increasing observation counts. Just 2 observations resolve 25% of prior epistemic uncertainty, rising to 62% with 20 observations.</sub></p>
</td>
</tr>
<tr>
<td width="50%">
<img src="figures/calibration_comparison.png" alt="Calibration Comparison"/>
<p align="center"><b>Calibration: Static vs Adaptive</b><br/><sub>Static TIMEVIEW is severely miscalibrated with only 26.0% coverage at the 95% level (left). Bayesian adaptation achieves near-perfect calibration with 96.8% coverage (right), closely tracking the diagonal.</sub></p>
</td>
<td width="50%">
<img src="figures/active_scheduling.png" alt="Active Scheduling"/>
<p align="center"><b>Active Observation Scheduling</b><br/><sub>Information-gain-based active scheduling achieves 2&ndash;3&times; faster convergence than uniform spacing by selecting time points where posterior uncertainty is highest.</sub></p>
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

1. **Prior from encoder** — A probabilistic neural encoder maps baseline features to prior distribution parameters (mean and covariance) over B-spline coefficients
2. **Closed-form Bayesian update** — As streaming observations arrive, conjugate Gaussian updates refine the posterior over coefficients in $\mathcal{O}(B^3)$ time (~0.3ms per update), enabling real-time clinical deployment
3. **Uncertainty propagation** — The posterior over coefficients induces calibrated predictive distributions, with variance guaranteed to decrease monotonically with each observation

---

## Streaming Adaptation

<p align="center">
<img src="figures/streaming_evaluation.png" alt="Streaming Evaluation" width="40%"/>
</p>
<p align="center"><sub>Online adaptation metrics as observations stream in. Prediction error (MSE) and CRPS decrease, average uncertainty tightens monotonically (guaranteed by Proposition 1), and calibration coverage converges toward the 95% target.</sub></p>

---

## Comparisons

<table>
<tr>
<td width="50%">
<img src="figures/gp_comparison.png" alt="GP Comparison"/>
<p align="center"><b>GP vs TIMEVIEW-Adaptive vs Static</b><br/><sub>TIMEVIEW-Adaptive dominates the GP baseline: 6.5&times; lower MSE, better coverage (96.8% vs 89.6%), and 2&times; sharper intervals. The static baseline under-covers catastrophically (26.0%).</sub></p>
</td>
<td width="50%">
<img src="figures/dataset_comparison.png" alt="Dataset Comparison"/>
<p align="center"><b>Cross-Dataset Evaluation</b><br/><sub>Static vs Adaptive on future time points (n<sub>obs</sub>=10). FLChain benefits most (35.4% MSE reduction) due to high inter-patient variability. Stress-Strain's degradation is due to data-knot range mismatch, not a method limitation.</sub></p>
</td>
</tr>
</table>

---

## (WIP) Prior vs Static Comparison

We compare the adaptive model's **prior** against the static model to measure how posterior performance changes with the number of observations used during training.

<table>
<tr>
<td width="40%">
<img src="figures/prior_ratio_vs_static.png" alt="Prior Ratio vs Static"/>
<p align="center"><b>Prior MSE / Static MSE vs Training n_obs</b><br/><sub>The adaptive model's prior is always worse than the static model, and degrades as training n_obs increases. Ratios at n_obs=30: Airfoil 10.5×, FLChain 4.5×, Stress-Strain 8.4×.</sub></p>
</td>
<td width="60%">
<img src="figures/prior_vs_static_comparison.png" alt="Prior/Posterior vs Static"/>
<p align="center"><b>Prior and Posterior MSE vs Static</b><br/><sub>Top: prior MSE as a function of training n_obs (dashed = static baseline). Bottom: posterior future MSE vs eval n_obs for each training configuration. Models perform best when eval n_obs matches training n_obs.</sub></p>
</td>
</tr>
</table>

**Prior MSE / Static MSE ratios across datasets:**

| train\_n\_obs | Airfoil | FLChain | Stress-Strain |
|:---:|:---:|:---:|:---:|
| 2  | 1.26× | 1.67× | 1.66× |
| 5  | 1.93× | 2.09× | 1.43× |
| 10 | 2.92× | 2.39× | 1.86× |
| 15 | 5.02× | 2.32× | 1.55× |
| 20 | 6.12× | 3.30× | 1.38× |
| 30 | 10.5× | 4.54× | 8.37× |

The adaptive prior is consistently worse than static, and degrades as `train_n_obs` increases. The training objective optimises posterior NLL (after observing `train_n_obs` points), so the prior is shaped to produce a good posterior at that specific observation count rather than to be a good standalone predictor.

Posterior performance (when eval n_obs matches training n_obs) consistently beats static: e.g. on Airfoil with train/eval n_obs=10, posterior future MSE is 0.716× static. However, large mismatches between training and evaluation n_obs lead to degraded performance.

---

## Ablation Studies

<table>
<tr>
<td width="50%">
<img src="figures/ablation_studies.png" alt="Ablation Studies"/>
<p align="center"><b>Hyperparameter Sensitivity</b><br/><sub>B=9 basis functions is optimal; diminishing returns beyond ~15 observations. Low-rank covariance provides the best MSE-coverage trade-off.</sub></p>
</td>
<td width="50%">
<img src="figures/best_config_comparison.png" alt="Best Config Comparison"/>
<p align="center"><b>Model Variant Comparison</b><br/><sub>Best configuration (gating + heteroscedastic noise + low-rank covariance) achieves the lowest calibration error (0.035), less than half the standard model's. Standard achieves the lowest raw MSE and CRPS.</sub></p>
</td>
</tr>
<tr>
<td width="50%">
<img src="figures/fix_ablation_studies.png" alt="Fix Ablation Studies"/>
<p align="center"><b>Training Fix Ablation</b><br/><sub>Individual contribution of each training improvement on Airfoil. BatchNorm has the largest single impact (29% MSE reduction). All fixes combined achieve 96.3% coverage vs 65.5% baseline.</sub></p>
</td>
<td width="50%">
<img src="figures/nobs_ablation_all_datasets.png" alt="Nobs Ablation"/>
<p align="center"><b>Adaptation Benefit vs Observation Count</b><br/><sub>Adaptation benefit depends on observation coverage relative to knot support. Airfoil improves consistently; FLChain scales monotonically to 51% at n<sub>obs</sub>=45. Stress-Strain requires sufficient coverage (&ge;70% trajectory) but then yields up to 95% MSE improvement.</sub></p>
</td>
</tr>
</table>

---

## Additional Analysis

<table>
<tr>
<td width="50%">
<img src="figures/heteroscedastic_comparison.png" alt="Noise Model Comparison"/>
<p align="center"><b>Noise Model Comparison</b><br/><sub>All variants achieve near-perfect 95% coverage on Airfoil. Gating achieves the best calibration error (0.035, 43% reduction) and sharpest intervals; heteroscedastic noise achieves the highest coverage (97.4%).</sub></p>
</td>
<td width="50%">
<img src="figures/temperature_scaling.png" alt="Temperature Scaling"/>
<p align="center"><b>Temperature Scaling</b><br/><sub>Post-hoc temperature scaling reduces calibration error from 0.090 to 0.034.</sub></p>
</td>
</tr>
<tr>
<td colspan="2" align="center">
<img src="figures/aec_comparison.png" alt="Adaptation Efficiency" width="60%"/>
<p align="center"><b>Adaptation Efficiency Curves</b><br/><sub>Per-observation MSE reduction efficiency. The standard model has the highest AAEC (0.049) due to more room for improvement; the best-config model achieves the lowest (0.012).</sub></p>
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
