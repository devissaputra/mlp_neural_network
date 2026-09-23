# Multilayer Perceptron vs Linear Baseline

**Focus:** testing whether neural-network complexity earns its place.

A two-hidden-layer MLP (128 → 64) and standardized logistic regression are evaluated on the same handwritten-digits split. Logistic regression reaches 0.9778 accuracy and 0.9776 Macro-F1, while the MLP reaches 0.9578 and 0.9575.

The project deliberately keeps the negative result: on this small 8×8 dataset, the simpler linear model is better. That makes the repository a model-selection exercise rather than a neural-network showcase.

The experiment is import-safe, reproducible, behaviorally tested, and checked by GitHub Actions.
