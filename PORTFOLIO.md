# Portfolio Summary

## Multilayer Perceptron for Handwritten Digits

I train a compact two-hidden-layer MLP on 8 × 8 handwritten digit images. The inputs are standardized and the model uses early stopping.

### Images

![Project overview](assets/01_cover.svg)

![Training pipeline](assets/02_data_pipeline.svg)

![Network and optimization](assets/03_data_or_model.svg)

![Held-out evaluation](assets/04_evaluation_or_results.svg)

**Architecture:** 64 inputs → 128 hidden units → 64 hidden units → 10 classes.

**Key result:** 0.9578 accuracy and 0.9575 macro-F1 in the recorded run.
