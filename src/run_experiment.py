# Calculation reading guide: ../CALCULATIONS.md (repository root).
# Accuracy = correct / n; macro-F1 = mean(F1 for each digit class).
# Macro-F1 gives each class equal weight. A single seeded holdout does not establish a universal ranking of architectures; training and preprocessing choices remain part of the comparison.

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42


def load_split(seed: int = SEED):
    X, y = load_digits(return_X_y=True)
    return train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )


def build_models(seed: int = SEED):
    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=3000, random_state=seed),
            ),
        ]
    )
    mlp_estimator = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=450,
        early_stopping=True,
        random_state=seed,
    )
    mlp = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", mlp_estimator),
        ]
    )
    return logistic, mlp, mlp_estimator


def metrics(y_true, prediction):
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
    }


def run_experiment(
    results_dir: str | Path = "results",
    seed: int = SEED,
    make_plots: bool = True,
):
    X_train, X_test, y_train, y_test = load_split(seed)
    logistic, mlp, mlp_estimator = build_models(seed)

    logistic.fit(X_train, y_train)
    logistic_prediction = logistic.predict(X_test)

    mlp.fit(X_train, y_train)
    mlp_prediction = mlp.predict(X_test)

    results = {
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "logistic": metrics(y_test, logistic_prediction),
        "mlp": {
            **metrics(y_test, mlp_prediction),
            "iterations": int(len(mlp_estimator.loss_curve_)),
        },
    }

    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    if make_plots:
        figures = output / "figures"
        figures.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(7, 5))
        plt.plot(mlp_estimator.loss_curve_)
        plt.xlabel("Optimization iteration")
        plt.ylabel("Training loss")
        plt.title("MLP optimization curve")
        plt.tight_layout()
        plt.savefig(figures / "mlp_loss.png", dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            mlp_prediction,
            ax=ax,
            colorbar=False,
        )
        ax.set_title("MLP held-out confusion matrix")
        fig.tight_layout()
        fig.savefig(figures / "mlp_confusion_matrix.png", dpi=150)
        plt.close(fig)

    return results


def main() -> None:
    print(json.dumps(run_experiment(), indent=2))


if __name__ == "__main__":
    main()
