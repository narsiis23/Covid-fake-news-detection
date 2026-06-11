"""Evaluation metrics and reporting utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import get_label_names, save_json


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute standard classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def build_classification_report(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    config: dict,
) -> str:
    """Return a sklearn classification report string."""
    label_names = get_label_names(config)
    target_names = [label_names[0], label_names[1]]
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )


def get_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Return confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def get_roc_curve(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ROC curve coordinates."""
    return roc_curve(y_true, y_prob)


def summarize_training_results(
    training_results: dict[str, Any],
    config: dict,
) -> dict[str, Any]:
    """Summarize metrics for all trained models."""
    test_y = training_results["labels"]["test_y"]
    summary: dict[str, Any] = {"models": {}}

    for name, model_result in training_results["models"].items():
        test_metrics = compute_metrics(
            test_y,
            model_result["test_predictions"],
            model_result["test_probabilities"],
        )
        summary["models"][name] = {
            "cv_mean_f1": model_result["cv_mean"],
            "cv_std_f1": model_result["cv_std"],
            "test_metrics": test_metrics,
            "classification_report": build_classification_report(
                test_y,
                model_result["test_predictions"],
                config,
            ),
            "confusion_matrix": get_confusion_matrix(
                test_y,
                model_result["test_predictions"],
            ).tolist(),
        }

    return summary


def save_evaluation_report(summary: dict[str, Any], path: str) -> None:
    """Save evaluation summary without numpy objects."""
    serializable = {
        "models": {
            name: {
                "cv_mean_f1": result["cv_mean_f1"],
                "cv_std_f1": result["cv_std_f1"],
                "test_metrics": result["test_metrics"],
                "classification_report": result["classification_report"],
                "confusion_matrix": result["confusion_matrix"],
            }
            for name, result in summary["models"].items()
        }
    }
    save_json(serializable, path)


def metrics_table(summary: dict[str, Any]) -> pd.DataFrame:
    """Build a comparison table across models."""
    rows = []
    for name, result in summary["models"].items():
        row = {"model": name}
        row.update(result["test_metrics"])
        row["cv_mean_f1"] = result["cv_mean_f1"]
        row["cv_std_f1"] = result["cv_std_f1"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1", ascending=False)
