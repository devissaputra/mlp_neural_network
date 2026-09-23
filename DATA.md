# Data

This project uses scikit-learn's Optical Recognition of Handwritten Digits dataset.

- 1,797 images
- image size: 8 × 8
- 64 numerical pixel features
- 10 classes

Source documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

The experiment loads the data directly with `load_digits` and uses a stratified 75/25 train/test split.
