from pathlib import Path

from src.run_experiment import load_split, run_experiment


def test_split_is_deterministic():
    first = load_split()
    second = load_split()
    for left, right in zip(first, second):
        assert (left == right).all()


def test_both_models_are_evaluated(tmp_path):
    result = run_experiment(tmp_path, make_plots=False)
    assert "logistic" in result
    assert "mlp" in result
    assert 0.0 <= result["logistic"]["accuracy"] <= 1.0
    assert 0.0 <= result["mlp"]["accuracy"] <= 1.0
    assert result["mlp"]["iterations"] > 0


def test_repository_structure():
    root = Path(__file__).resolve().parents[1]
    assert (root / ".github/workflows/ci.yml").exists()
    assert (root / "src/run_experiment.py").exists()
    assert (root / "paper/paper.md").exists()
