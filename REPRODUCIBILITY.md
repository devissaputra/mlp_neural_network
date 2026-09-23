# Reproducing the Experiment

Run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

The data split is stratified with `random_state=42`. The MLP also uses `random_state=42`, hidden layers of 128 and 64 units, and early stopping.

The script writes accuracy, macro-F1, and the number of optimization iterations to `results/metrics.json`.

Exact stopping behaviour can change slightly across scikit-learn versions, so record the package version when comparing runs.
