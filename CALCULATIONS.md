# Calculation guide

## Question and evidence

Does a nonlinear network beat a linear digit classifier?

scikit-learn handwritten digits; 1,347 training and 450 test observations.

**Status:** RECORDED SMALL-SAMPLE BENCHMARK | see validation scope.

## Design

Shared stratified split; training-fitted scaling; multinomial logistic regression versus MLP.

## Calculation and interpretation

`Accuracy = correct / n; macro-F1 = mean(F1 for each digit class).`

Macro-F1 gives each class equal weight. A single seeded holdout does not establish a universal ranking of architectures; training and preprocessing choices remain part of the comparison.

## Evidence table

Selected recorded values (units and context shown). Full precision below is for traceability, not a claim of measurement precision.

| Quantity | Value | Unit / meaning | JSON path |
|---|---:|---|---|
| logistic | 0.9777777777777777 | accuracy ↑ | `logistic.accuracy` |
| mlp | 0.9577777777777777 | accuracy ↑ | `mlp.accuracy` |

Source: [results/metrics.json](results/metrics.json). Values resolve directly from this file when figures are regenerated.

This experiment compares a multilayer perceptron with multinomial logistic regression on the same standardized handwritten-digit split. The recorded linear baseline reaches 0.9778 accuracy, compared with 0.9578 for the MLP, with macro-F1 showing the same ordering. Keeping this negative result makes the engineering question clear: additional nonlinear capacity must justify its complexity under an explicit evaluation protocol.

## Verification performed in this review

The existing suite requires unavailable dependencies; no full-suite pass is claimed. The complete data/model experiment was not rerun in this review.

The figure-generation check verifies agreement between the selected source values and SVGs. It does not validate the raw dataset, fitted model, identification assumptions, or external generalization.

```bash
python scripts/build_review_figures.py
python scripts/build_review_figures.py --check
```

## Implementation map

Follow these functions to inspect each transformation. Validation helpers and private functions remain visible in the linked modules.

| Function | Purpose / documented behavior |
|---|---|
| [`load_split`](src/run_experiment.py#L23) | Inspect the explicit implementation and its callers. |
| [`build_models`](src/run_experiment.py#L34) | Inspect the explicit implementation and its callers. |
| [`metrics`](src/run_experiment.py#L59) | Inspect the explicit implementation and its callers. |
| [`run_experiment`](src/run_experiment.py#L66) | Inspect the explicit implementation and its callers. |
| [`main`](src/run_experiment.py#L126) | Inspect the explicit implementation and its callers. |

## What remains before a stronger research claim

Macro-F1 gives each class equal weight. A single seeded holdout does not establish a universal ranking of architectures; training and preprocessing choices remain part of the comparison. A successful software test is not validation of a scientific construct. New experiments should state their split unit, comparator, outcome, uncertainty procedure and failure criteria before examining final test results.
