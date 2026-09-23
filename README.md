# Multilayer Perceptron vs Linear Baseline on Handwritten Digits

[![CI](https://github.com/devissaputra/mlp_neural_network/actions/workflows/ci.yml/badge.svg)](https://github.com/devissaputra/mlp_neural_network/actions/workflows/ci.yml)

![Project overview](assets/01_cover.svg)

A compact neural-network experiment with one rule: **the MLP has to earn its complexity by beating a strong linear baseline**.

## Data

The scikit-learn handwritten-digits dataset contains:

- 1,797 grayscale images
- 8 × 8 pixels
- 64 numerical inputs
- 10 classes
- stratified 75/25 train/test split
- seed 42

## Models

![Training pipeline](assets/02_data_pipeline.svg)

Both models use standardized inputs.

### Linear baseline
Multinomial logistic regression.

### MLP
```text
64 inputs
  ↓
128 hidden units
  ↓
64 hidden units
  ↓
10 output classes
```

The MLP uses early stopping and a maximum of 450 optimization iterations.

## Recorded results

![Network and optimization](assets/03_data_or_model.svg)

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Logistic regression | **0.9778** | **0.9776** |
| MLP | 0.9578 | 0.9575 |

The MLP stopped after 29 optimization iterations.

![Held-out evaluation](assets/04_evaluation_or_results.svg)

The important result is that the neural network does **not** win here. The dataset is small, low-resolution, and already close to linearly separable after scaling. That makes this a better engineering lesson than a cherry-picked neural-network victory: extra model complexity should be justified by evidence.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

## Test

```bash
pip install pytest
pytest
```

Tests verify deterministic data splitting, baseline/MLP metric output, and import-safe execution.

## Why this repo matters

This project demonstrates:

- a neural classifier implemented with scikit-learn;
- a meaningful baseline;
- class-balanced evaluation through Macro-F1;
- early stopping;
- reproducible experiment code;
- behavioral tests and CI;
- willingness to keep a negative result when the simpler model is better.

## Limitations

One split cannot establish general model superiority. A stronger extension would repeat the comparison across seeds, tune both models with nested validation, add confidence intervals, and compare against CNN features or larger image datasets.
