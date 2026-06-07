# Flipkart Grid Traffic Demand Forecasting

This repository contains the solution for predicting spatial-temporal traffic demand across geohashes, built specifically to tackle the **Autoregressive Forecasting Paradox**.

## Problem Statement
The objective is to forecast traffic demand (a bounded continuous variable between 0 and 1) for various Geohashes over an 11-hour forecast horizon (Day 49, Hours 2 through 13). The training dataset contains historical demand, categorical context (RoadType, Weather, etc.), and spatial attributes (Geohashes).

## The Challenge: The Autoregressive Paradox
In multi-step time-series forecasting, engineers typically rely on lag features (e.g., `demand_15min_ago`). However, this dataset presents two mathematically conflicting constraints:
1. **The Short Lag Paradox**: The test set is an 11-hour block. Using short lags (`lag_15m`) requires predicting step 1 to feed into step 2. Without a slow, recursive autoregressive loop, short lags instantly cascade into `NaN` values across the test set, destroying predictions.
2. **The Long Lag Paradox**: To avoid the `NaN` cascade, one could use a lag equal to the forecast horizon (`lag_12h`). However, the training dataset only contains 24 hours of history. Using a 12-hour lag forces you to discard half of the training data (the "Warm-up Penalty"), starving the model of real historical data.

## The Solution: Pure Spatial-Temporal Profiler
We abandoned autoregressive lag features entirely. Instead, this solution treats the problem as a **non-autoregressive multivariate regression** by profiling the spatial and temporal signatures independently:

### 1. The Spatial Anchor (Hierarchical Target Encoding)
We extract the pure spatial signature of every geohash (`spatial_prior`) using strict K-Fold Out-Of-Fold (OOF) target encoding to prevent leakage. 
- **The Fallback System**: If a geohash is entirely unseen in the training set (a map "dead zone"), the encoder gracefully falls back through hierarchical resolutions: `Geohash` ➡️ `Geohash_5` ➡️ `Geohash_4` ➡️ `Global Minimum (0.0001)`. This guarantees robust generalization on unseen locations without falling into the imputation trap.

### 2. The Temporal Anchor
We extract the shape of the daily traffic curve using purely cyclical features (`hour_sin`, `hour_cos`, `minute_sin`, `minute_cos`). The model learns the *shape* of the daily traffic wave rather than memorizing the previous 15 minutes.

### 3. Logit-Bounded Targets
To respect the physical constraints of the demand metric (bounded between 0 and 1), the target is transformed into Logit space $y = \log(\frac{demand}{1 - demand})$ before training. Predictions are passed through a Sigmoid inverse, strictly guaranteeing valid outputs and preventing negative demand predictions.

## Repository Structure
```text
flipkart-grid/
├── README.md                  # Comprehensive problem & solution description
├── requirements.txt           # Python dependencies
├── notebooks/
│   └── EDA.ipynb              # Jupyter notebook with spatial/temporal visualizations
├── dataset/                   # (Ignored in git) The raw csv files
│   ├── train.csv
│   └── test.csv
└── src/
    ├── data_loader.py         # Functions to load dataset and parse cyclical dates
    ├── features.py            # The Pure Spatial-Temporal Profiler (OOF Encodings)
    └── pipeline.py            # Pipeline execution, cross-validation, and inference
```

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place `train.csv` and `test.csv` inside the `dataset/` directory.

3. Run the training and inference pipeline:
```bash
python src/pipeline.py
```

The script will:
- Parse all time features.
- Build the Hierarchical Spatial Encodings using 5-Fold OOF.
- Train a `CatBoostRegressor` on the Logit-transformed targets.
- Output Holdout Validation Metrics (R², MAE, RMSE).
- Generate the final `dataset/submission.csv`.
