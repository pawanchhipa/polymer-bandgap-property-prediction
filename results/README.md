# Polymer Model Results

This directory contains the results and performance metrics for the polymer property prediction model.

## Directory Structure

```
results/
├── model_performance/
│   └── rf_model_metrics.json    # Model performance metrics
└── figures/
    └── parity_plot.png         # Parity plot visualization
```

## Metrics Description

- `mean_rmse_val`: Mean Root Mean Square Error for validation sets
- `mean_r2_val`: Mean R² score for validation sets
- `fold_results`: Detailed results for each cross-validation fold

## Usage

To view the latest results:
```bash
cat results/model_performance/rf_model_metrics.json
```
