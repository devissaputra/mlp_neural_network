# Reproducing the experiment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

The train/test split is stratified and uses seed 42. Logistic regression and the MLP also use seed 42 where applicable. Both models receive standardized inputs.

The MLP uses hidden layers (128, 64), early stopping, and at most 450 optimization iterations.

Outputs are written to `results/metrics.json` and `results/figures/`.

Tests:

```bash
pip install pytest
pytest
```

GitHub Actions runs the tests automatically. Exact early-stopping behaviour may vary slightly across scikit-learn versions.
