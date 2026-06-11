"""Publication-quality visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from src.evaluation import get_confusion_matrix, get_roc_curve
from src.utils import ensure_dir, get_label_names


def _save_figure(fig: plt.Figure, path: str | Path) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(
    labels: pd.Series,
    config: dict,
    output_path: str | Path,
) -> None:
    """Plot label distribution."""
    label_names = get_label_names(config)
    mapped = labels.map(label_names)
    counts = mapped.value_counts().reindex(["Fake", "Real"])

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, palette="Set2", ax=ax)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    for index, value in enumerate(counts.values):
        ax.text(index, value, str(value), ha="center", va="bottom")
    _save_figure(fig, output_path)


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    config: dict,
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot a confusion matrix heatmap."""
    matrix = get_confusion_matrix(y_true, y_pred)
    label_names = get_label_names(config)
    labels = [label_names[0], label_names[1]]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    _save_figure(fig, output_path)


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    metric: str = "f1",
) -> None:
    """Plot model comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=metrics_df, x="model", y=metric, palette="viridis", ax=ax)
    ax.set_title(f"Model Comparison ({metric.upper()})")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    ax.tick_params(axis="x", rotation=20)
    _save_figure(fig, output_path)


def plot_roc_curve(
    y_true: pd.Series,
    y_prob: np.ndarray,
    output_path: str | Path,
    title: str = "ROC Curve",
) -> None:
    """Plot ROC curve for binary classification."""
    fpr, tpr, _ = get_roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    _save_figure(fig, output_path)


def plot_embedding_pca(
    embeddings: np.ndarray,
    labels: pd.Series,
    config: dict,
    output_path: str | Path,
    n_components: int = 2,
) -> None:
    """Visualize embeddings with PCA."""
    reducer = PCA(n_components=n_components, random_state=config["project"]["random_seed"])
    reduced = reducer.fit_transform(embeddings)
    label_names = get_label_names(config)

    fig, ax = plt.subplots(figsize=(7, 5))
    for label_value in sorted(labels.unique()):
        mask = labels == label_value
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            alpha=0.6,
            label=label_names[int(label_value)],
            s=20,
        )
    ax.set_title("LaBSE Embeddings (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    _save_figure(fig, output_path)


def generate_all_figures(
    training_results: dict,
    summary: dict,
    metrics_df: pd.DataFrame,
    config: dict,
    results_dir: str | Path,
) -> None:
    """Generate all standard figures for a training run."""
    results_dir = Path(results_dir)
    ensure_dir(results_dir)

    labels = training_results["labels"]["full_y"]
    plot_class_distribution(labels, config, results_dir / "class_distribution.png")

    features = training_results["features"]["full_matrix"]
    plot_embedding_pca(features, labels, config, results_dir / "embedding_pca.png")

    plot_model_comparison(metrics_df, results_dir / "model_comparison_f1.png", metric="f1")
    plot_model_comparison(
        metrics_df,
        results_dir / "model_comparison_accuracy.png",
        metric="accuracy",
    )

    for name, model_result in training_results["models"].items():
        test_y = training_results["labels"]["test_y"]
        plot_confusion_matrix(
            test_y,
            model_result["test_predictions"],
            config,
            results_dir / f"confusion_matrix_{name}.png",
            title=f"Confusion Matrix — {name}",
        )
        if model_result["test_probabilities"] is not None:
            plot_roc_curve(
                test_y,
                model_result["test_probabilities"],
                results_dir / f"roc_curve_{name}.png",
                title=f"ROC Curve — {name}",
            )
