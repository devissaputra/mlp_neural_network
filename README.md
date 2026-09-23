# Multilayer Perceptron for Handwritten Digits

![Project overview](assets/01_cover.svg)

I built this project as my first neural-network classifier in this series. The aim is to keep the model small enough to understand while still learning nonlinear patterns from real image data.

The model is a two-hidden-layer multilayer perceptron trained on scikit-learn's handwritten digits dataset.

## Data

The dataset contains:

- 1,797 digit images;
- 8 × 8 grayscale pixels;
- 64 numerical input features;
- 10 classes, digits 0 through 9.

I use a stratified 75/25 train/test split.

## How the experiment works

![Training pipeline](assets/02_data_pipeline.svg)

The pipeline standardizes the 64 pixel features and then fits an `MLPClassifier`.

The network is:

```text
64 input features
      ↓
128 hidden units
      ↓
64 hidden units
      ↓
10 output classes
```

Early stopping is enabled, with `random_state=42` and a maximum of 450 iterations.

## Training behaviour

![Network and optimization](assets/03_data_or_model.svg)

The recorded run stopped after 29 optimization iterations, well before the maximum. That is expected when early stopping decides that further training is no longer improving the internal validation result.

## Results

![Held-out evaluation](assets/04_evaluation_or_results.svg)

The recorded run produced:

| Metric | Result |
|---|---:|
| Accuracy | 0.9578 |
| Macro-F1 | 0.9575 |
| Optimization iterations | 29 |

Macro-F1 is almost the same as accuracy, which suggests that the model is performing fairly consistently across the ten digit classes rather than relying on only a few easy classes.

This is still one split on a small benchmark dataset. A stronger comparison would repeat the experiment across several seeds and compare it directly with a linear classifier and the CNN in the next project.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

On Windows, use `.venv\Scripts\activate`.

## Repository notes

- [DATA.md](DATA.md) explains the dataset.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) records the settings needed to repeat the run.
- [paper/paper.md](paper/paper.md) contains the longer technical write-up.
