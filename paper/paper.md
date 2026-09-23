# Multilayer Perceptron vs Linear Baseline on Handwritten Digits

## Abstract

This experiment compares a two-hidden-layer MLP with standardized logistic regression on the scikit-learn handwritten-digits dataset. The comparison is included to test whether nonlinear neural-network complexity improves held-out performance on a small 8×8 image benchmark.

## Results

Logistic regression reaches 0.9778 accuracy and 0.9776 Macro-F1. The MLP reaches 0.9578 accuracy and 0.9575 Macro-F1 and stops after 29 optimization iterations.

## Interpretation

The linear baseline outperforms the MLP in this recorded split. This negative result is useful: the dataset is small and low-dimensional enough that a strong linear model remains highly competitive. Model complexity should therefore be justified empirically rather than assumed to be superior.

## Limitations

A stronger comparison would repeat the experiment across seeds, tune both models under nested validation, and include uncertainty intervals.
