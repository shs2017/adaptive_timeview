# TIMEVIEW-Adaptive: Real Dataset Results

## Static vs Adaptive Comparison

| Dataset | Method | MSE | CRPS | Coverage (95%) | Calibration Error |
|---------|--------|-----|------|----------------|------------------|
| airfoil | Static | 0.1286 | 0.2966 | 37.2% | 0.3581 |
| airfoil | Adaptive | 0.1336 | 0.1781 | 96.4% | 0.1008 |
| flchain | Static | 0.2090 | 0.2530 | 53.9% | 0.2823 |
| flchain | Adaptive | 0.1351 | 0.1437 | 96.3% | 0.0501 |
| stress_strain | Static | 0.1096 | 0.3001 | 50.5% | 0.2866 |
| stress_strain | Adaptive | 0.2870 | 0.2515 | 90.3% | 0.0259 |

## Improvement Summary

| Dataset | MSE Improvement | CRPS Improvement |
|---------|-----------------|------------------|
| airfoil | -1.9% | 39.5% |
| flchain | 35.4% | 43.2% |
| stress_strain | -145.6% | 25.8% |
