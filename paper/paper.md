# Multilayer Perceptron vs Linear Baseline on Handwritten Digits

## Abstract

This experiment compares a two-hidden-layer MLP with standardized logistic regression on the scikit-learn handwritten-digits dataset. The comparison is included to test whether nonlinear neural-network complexity improves held-out performance on a small 8×8 image benchmark.

## Results

Logistic regression reaches 0.9778 accuracy and 0.9776 Macro-F1. The MLP reaches 0.9578 accuracy and 0.9575 Macro-F1 and stops after 29 optimization iterations.

## Interpretation

The linear baseline outperforms the MLP in this recorded split. This negative result is useful: the dataset is small and low-dimensional enough that a strong linear model remains highly competitive. Model complexity should therefore be justified empirically rather than assumed to be superior.

## Limitations

A stronger comparison would repeat the experiment across seeds, tune both models under nested validation, and include uncertainty intervals.


## Calculation definitions and evidence audit

Accuracy = correct / n; macro-F1 = mean(F1 for each digit class).

Macro-F1 gives each class equal weight. A single seeded holdout does not establish a universal ranking of architectures; training and preprocessing choices remain part of the comparison.

This experiment compares a multilayer perceptron with multinomial logistic regression on the same standardized handwritten-digit split. The recorded linear baseline reaches 0.9778 accuracy, compared with 0.9578 for the MLP, with macro-F1 showing the same ordering. Keeping this negative result makes the engineering question clear: additional nonlinear capacity must justify its complexity under an explicit evaluation protocol.

The [calculation guide](../CALCULATIONS.md) provides exact evidence paths and a function-level implementation map.

![Study design](../assets/review_overview.svg)

![Calculation and selected evidence](../assets/review_calculations.svg)

### Selected evidence and interpretation

| Quantity | Value | Unit / meaning | JSON path |
|---|---:|---|---|
| logistic | 0.9777777777777777 | accuracy ↑ | `logistic.accuracy` |
| mlp | 0.9577777777777777 | accuracy ↑ | `mlp.accuracy` |

These values are read from `results/metrics.json`. They must be interpreted with the split, data status and limitations above. The complete data/model experiment was not rerun in this review.

### Reproduction and claim boundaries

The existing suite requires unavailable dependencies; no full-suite pass is claimed. The figure generator can be checked with `python scripts/build_review_figures.py --check`. This verifies the displayed calculation evidence, not an independent replication of the complete scientific experiment. The manuscript is a working report, not a peer-reviewed publication.
