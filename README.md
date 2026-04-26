# Time Series Forecasting with Chronos-2

A Jupyter notebook that demonstrates how to use **Amazon Chronos-2** — a state-of-the-art foundation model for time series forecasting — covering everything from quick-start inference to fine-tuning and benchmarking.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Notebook Contents](#notebook-contents)
  - [1. Load the Pipeline](#1-load-the-pipeline)
  - [2. Univariate Forecasting](#2-univariate-forecasting)
  - [3. Forecasting with Covariates](#3-forecasting-with-covariates)
  - [4. Low-Level Tensor API](#4-low-level-tensor-api)
  - [5. Cross-Learning](#5-cross-learning)
  - [6. Fine-Tuning](#6-fine-tuning)
  - [7. Evaluation & Benchmarking](#7-evaluation--benchmarking)
- [Model Comparison](#model-comparison)
- [Key Metrics](#key-metrics)

---

## Overview

[Chronos-2](https://huggingface.co/amazon/chronos-2) is a transformer-based foundation model for time series forecasting released by Amazon. It extends the original Chronos family with support for:

- **Covariates** (past and future known features)
- **Cross-learning** (joint attention across multiple time series in a batch)
- **LoRA fine-tuning** for efficient domain adaptation
- **Probabilistic forecasting** via quantile outputs

This notebook (`chronos2_testing_&_benchmarks.ipynb`) tests and benchmarks Chronos-2 against Chronos-Bolt-base and the original Chronos-T5-small on standard in-domain and zero-shot evaluation suites.

---

## Requirements

| Dependency | Notes |
|---|---|
| Python ≥ 3.9 | |
| `chronos-forecasting[extras] >= 2.2` | Core library + LoRA / extra features |
| `chronos-forecasting[dev] >= 2.2` | Adds evaluation tools (`gluonts`, `fev`, etc.) |
| `matplotlib` | Plotting |
| `seaborn` | Benchmark visualisations |
| `scipy` | Statistical utilities |
| GPU (T4 or better) | Recommended for evaluation runs |

---

## Installation

**Option A – quick install (inference only):**

```bash
pip install 'chronos-forecasting[extras]>=2.2' matplotlib
```

**Option B – full development install (includes evaluation scripts):**

```bash
pip install 'chronos-forecasting[dev]>=2.2' matplotlib scipy
git clone https://github.com/MiaYangM/chronos-forecasting.git
pip install -e './chronos-forecasting[extras]'
cd chronos-forecasting
```

---

## Notebook Contents

### 1. Load the Pipeline

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from chronos import BaseChronosPipeline, Chronos2Pipeline

pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda"   # or "cpu" if no GPU
)
```

`BaseChronosPipeline.from_pretrained` auto-detects the model type (Chronos, Chronos-Bolt, or Chronos-2) from the HuggingFace config.

---

### 2. Univariate Forecasting

Demonstrates the simplest use-case: loading a long-format DataFrame and generating 24-step-ahead quantile forecasts on the **M4 Hourly** dataset.

```python
import pandas as pd

context_df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv")

pred_df = pipeline.predict_df(
    context_df,
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9],
)
```

---

### 3. Forecasting with Covariates

#### Energy Price Forecasting

Forecast hourly electricity prices for multiple countries using past covariate columns plus known future covariates.

```python
energy_pred_df = pipeline.predict_df(
    energy_context_df,
    future_df=energy_future_df,          # known future covariate values
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="target",
)
```

The notebook includes a `plot_forecast()` helper that overlays historical data, ground-truth future values, the median forecast, and a 10–90 % prediction interval.

#### Retail Demand Forecasting (Rossmann Sales)

Forecast 13 weeks of store sales using historical sales, customer footfall (`Customers`), and known future indicators (`Open`, `Promo`, `SchoolHoliday`, `StateHoliday`). Side-by-side plots compare forecasts **with** and **without** covariates.

---

### 4. Low-Level Tensor API

For workflows that bypass pandas (e.g., when inputs are already NumPy / PyTorch arrays):

```python
import numpy as np

# Univariate: batch of 32 series, each length 100
quantiles, mean = pipeline.predict_quantiles(
    np.random.randn(32, 1, 100),
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9],
)

# With past + future covariates (dict API)
inputs = [
    {
        "target": np.random.randn(200),
        "past_covariates":   {"temp": np.random.randn(200)},
        "future_covariates": {"temp": np.random.randn(64)},
    }
    for _ in range(8)
]
quantiles, mean = pipeline.predict_quantiles(inputs, prediction_length=64, quantile_levels=[0.1, 0.5, 0.9])
```

---

### 5. Cross-Learning

A unique Chronos-2 capability: all time series in the batch attend to each other during inference, enabling information sharing across items.

```python
pred_df_cross = pipeline.predict_df(
    context_df,
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9],
    cross_learning=True,   # enable cross-item attention
    batch_size=100,
)
```

---

### 6. Fine-Tuning

#### Full Fine-Tuning

```python
finetuned_pipeline = pipeline.fit(
    inputs=train_inputs,        # list of dicts with target + covariate arrays
    prediction_length=13,
    num_steps=1000,
    learning_rate=1e-5,
    batch_size=32,
    logging_steps=100,
)
```

#### LoRA Fine-Tuning (parameter-efficient)

```python
lora_pipeline = pipeline.fit(
    inputs=train_inputs,
    prediction_length=13,
    num_steps=1000,
    learning_rate=1e-4,       # higher LR recommended for LoRA
    batch_size=32,
    finetune_mode="lora",     # default: r=8, alpha=16
)
```

Both modes adapt the model to the **Rossmann retail sales** dataset.

---

### 7. Evaluation & Benchmarking

All evaluation commands are run from inside the cloned `chronos-forecasting` repository.

#### Smoke Test (7 datasets, ~CI suite)

```bash
python scripts/evaluation/evaluate.py chronos-2 \
    ci/evaluate/backtest_config.yaml \
    /tmp/chronos2-smoke-test.csv \
    --model-id "amazon/chronos-2" \
    --device cuda --torch-dtype float32 --batch-size 16
```

Datasets: `taxi_30min`, `ETTh`, `monash_covid_deaths`, `monash_nn5_weekly`, `monash_fred_md`, `monash_m3_quarterly`, `monash_tourism_yearly`.

#### Full In-Domain Evaluation (15 datasets, ~20 min on T4)

```bash
python scripts/evaluation/evaluate.py chronos-2 \
    scripts/evaluation/configs/in-domain.yaml \
    scripts/evaluation/results/chronos-2-in-domain.csv \
    --model-id "amazon/chronos-2" \
    --device cuda --torch-dtype float32 --batch-size 32
```

#### Full Zero-Shot Evaluation (27 datasets)

```bash
python scripts/evaluation/evaluate.py chronos-2 \
    scripts/evaluation/configs/zero-shot.yaml \
    scripts/evaluation/results/chronos-2-zero-shot.csv \
    --model-id "amazon/chronos-2" \
    --device cuda --torch-dtype float32 --batch-size 32
```

#### Aggregated Relative Scores (vs Seasonal Naïve baseline)

```bash
python scripts/evaluation/agg-relative-score.py chronos-2 \
    --baseline-name seasonal-naive \
    --results-dir scripts/evaluation/results/
```

Results are saved to `scripts/evaluation/results/chronos-2-agg-rel-scores.csv`.

The notebook visualises all results with **seaborn** bar charts comparing MASE and WQL scores per dataset and per evaluation type (in-domain vs. zero-shot).

---

## Model Comparison

The final section of the notebook benchmarks three Amazon Chronos family models side-by-side on the in-domain suite:

| Model | Variant | dtype | Batch size | Sampling |
|---|---|---|---|---|
| **Chronos-2** | `amazon/chronos-2` | float32 | 32 | deterministic |
| **Chronos-Bolt-base** | `amazon/chronos-bolt-base` | bfloat16 | 64 | deterministic |
| **Chronos-T5-small** | `amazon/chronos-t5-small` | bfloat16 | 32 | 20 trajectories |

Faceted bar charts (one facet per metric) compare all three models across every dataset.

---

## Key Metrics

| Metric | Description |
|---|---|
| **MASE** | Mean Absolute Scaled Error — lower is better |
| **WQL** | Weighted Quantile Loss — lower is better |

A lower score on either metric indicates better forecasting performance. The notebook reports both per-dataset scores and macro-averaged scores across the full benchmark suite.
