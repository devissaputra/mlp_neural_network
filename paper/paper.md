# Multilayer Perceptron for Handwritten Digits

## Question

How well can a small fully connected neural network classify handwritten digits when the model is kept simple and the training process is easy to inspect?

## Data

I use scikit-learn's digits dataset. It contains 1,797 images, each represented by 64 pixel values.

I create a stratified 75/25 train/test split.

## Method

The 64 inputs are standardized and passed to an MLP with two hidden layers:

```text
64 → 128 → 64 → 10
```

The classifier uses early stopping, `random_state=42`, and a maximum of 450 iterations.

## Results

| Metric | Result |
|---|---:|
| Accuracy | 0.9578 |
| Macro-F1 | 0.9575 |
| Optimization iterations | 29 |

## Interpretation

Accuracy and macro-F1 are almost identical, which suggests that performance is fairly balanced across the ten classes.

Early stopping ended the recorded run after 29 iterations, so the model did not need the full optimization budget.

## Limitations

This is one benchmark dataset and one train/test split. The model also ignores the two-dimensional structure of the images because all pixels are treated as a flat vector.

That limitation leads naturally to the CNN project, where the same kind of data is processed spatially.

## Reproduce

```bash
pip install -r requirements.txt
python src/run_experiment.py
```
