"""Model training, cross-validation, and persistence."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.embedding import build_embedder
from src.utils import ensure_dir, set_random_seed


@dataclass
class TrainArtifacts:
    """Container for fitted preprocessing and model objects."""

    model: ClassifierMixin
    embedder_transformer: Any
    scaler: StandardScaler | None
    feature_names: list[str]


def split_data(
    dataframe: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible stratified train/test split."""
    data_cfg = config["data"]
    seed = config["project"]["random_seed"]
    set_random_seed(seed)

    features = dataframe.drop(columns=[data_cfg["label_column"]])
    labels = dataframe[data_cfg["label_column"]].astype(int)

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=data_cfg["test_size"],
        random_state=seed,
        stratify=labels if data_cfg.get("stratify", True) else None,
    )
    return (
        train_x.reset_index(drop=True),
        test_x.reset_index(drop=True),
        train_y.reset_index(drop=True),
        test_y.reset_index(drop=True),
    )


def transform_features(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, Any, StandardScaler | None]:
    """Embed text and optionally scale features."""
    data_cfg = config["data"]
    embedder = build_embedder(config)
    preprocessor = embedder.build_sklearn_transformer(data_cfg["text_column"])

    train_matrix = preprocessor.fit_transform(train_x)
    test_matrix = preprocessor.transform(test_x)

    scaler: StandardScaler | None = None
    if config["training"].get("scale_features", True):
        scaler = StandardScaler()
        train_matrix = scaler.fit_transform(train_matrix)
        test_matrix = scaler.transform(test_matrix)

    return train_matrix, test_matrix, preprocessor, scaler


def embed_dataframe(
    dataframe: pd.DataFrame,
    config: dict,
    preprocessor: Any | None = None,
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[np.ndarray, Any, StandardScaler | None]:
    """Embed an entire dataframe using an optional fitted preprocessor."""
    data_cfg = config["data"]
    features = dataframe[[data_cfg["text_column"]]]

    if preprocessor is None:
        embedder = build_embedder(config)
        preprocessor = embedder.build_sklearn_transformer(data_cfg["text_column"])

    matrix = preprocessor.fit_transform(features) if fit else preprocessor.transform(features)

    if config["training"].get("scale_features", True):
        if scaler is None:
            scaler = StandardScaler()
            matrix = scaler.fit_transform(matrix)
        elif fit:
            matrix = scaler.fit_transform(matrix)
        else:
            matrix = scaler.transform(matrix)

    return matrix, preprocessor, scaler


def build_model(name: str, config: dict) -> ClassifierMixin:
    """Instantiate a classifier by name."""
    train_cfg = config["training"]
    seed = config["project"]["random_seed"]

    if name == "logistic_regression":
        lr_cfg = train_cfg["logistic_regression"]
        return LogisticRegression(
            solver=lr_cfg.get("solver", "liblinear"),
            max_iter=lr_cfg.get("max_iter", 1000),
            random_state=seed,
        )
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=train_cfg["knn"]["n_neighbors"])
    if name == "svm":
        svm_cfg = train_cfg["svm"]
        return SVC(
            C=svm_cfg.get("C", 100),
            probability=svm_cfg.get("probability", True),
            random_state=seed,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        xgb_cfg = train_cfg["xgboost"]
        return XGBClassifier(
            eval_metric=xgb_cfg.get("eval_metric", "logloss"),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            learning_rate=xgb_cfg.get("learning_rate", 0.4),
            max_depth=xgb_cfg.get("max_depth", 4),
            n_estimators=xgb_cfg.get("n_estimators", 100),
            random_state=xgb_cfg.get("random_state", seed),
        )
    if name == "stacking":
        return build_stacking_classifier(config)
    raise ValueError(f"Unknown model name: {name}")


def build_stacking_classifier(config: dict) -> StackingClassifier:
    """Build a stacking ensemble over base classifiers."""
    train_cfg = config["training"]
    stack_cfg = train_cfg["stacking"]

    base_estimators = [
        ("lr", build_model("logistic_regression", config)),
        ("knn", build_model("knn", config)),
        ("svm", build_model("svm", config)),
        ("xgb", build_model("xgboost", config)),
    ]

    meta_name = stack_cfg.get("meta_learner", "svm")
    meta_learner = build_model(meta_name, config)

    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=stack_cfg.get("cv", 5),
        n_jobs=-1,
    )


def get_model_registry(config: dict) -> dict[str, ClassifierMixin]:
    """Return all models evaluated in the paper pipeline."""
    return {
        "logistic_regression": build_model("logistic_regression", config),
        "knn": build_model("knn", config),
        "svm": build_model("svm", config),
        "xgboost": build_model("xgboost", config),
        "stacking": build_model("stacking", config),
    }


def fit_model(
    model: ClassifierMixin,
    train_x: np.ndarray,
    train_y: pd.Series,
) -> ClassifierMixin:
    """Fit a classifier on training data."""
    model.fit(train_x, train_y)
    return model


def cross_validate_model(
    model: ClassifierMixin,
    features: np.ndarray,
    labels: pd.Series,
    config: dict,
    scoring: str = "f1",
) -> dict[str, Any]:
    """Run repeated stratified cross-validation."""
    train_cfg = config["training"]
    seed = config["project"]["random_seed"]

    cv = RepeatedStratifiedKFold(
        n_splits=train_cfg.get("cv_folds", 10),
        n_repeats=train_cfg.get("cv_repeats", 3),
        random_state=seed,
    )
    scores = cross_val_score(model, features, labels, scoring=scoring, cv=cv, n_jobs=-1)
    predictions = cross_val_predict(model, features, labels, cv=cv, n_jobs=-1)

    return {
        "scores": scores,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "predictions": predictions,
    }


def train_all_models(
    dataframe: pd.DataFrame,
    config: dict,
) -> dict[str, Any]:
    """End-to-end training for all registered models."""
    train_x, test_x, train_y, test_y = split_data(dataframe, config)
    train_matrix, test_matrix, preprocessor, scaler = transform_features(train_x, test_x, config)

    full_x = pd.concat([train_x, test_x], ignore_index=True)
    full_labels = pd.concat([train_y, test_y], ignore_index=True)
    full_matrix, _, _ = embed_dataframe(
        full_x,
        config,
        preprocessor=preprocessor,
        scaler=scaler,
        fit=False,
    )

    results: dict[str, Any] = {
        "splits": {
            "train_size": len(train_y),
            "test_size": len(test_y),
        },
        "models": {},
    }

    for name, model in get_model_registry(config).items():
        model_template = build_model(name, config)
        fitted = fit_model(build_model(name, config), train_matrix, train_y)
        cv_result = cross_validate_model(model_template, full_matrix, full_labels, config)

        test_predictions = fitted.predict(test_matrix)
        test_probabilities = (
            fitted.predict_proba(test_matrix)[:, 1]
            if hasattr(fitted, "predict_proba")
            else None
        )

        results["models"][name] = {
            "cv_mean": cv_result["mean"],
            "cv_std": cv_result["std"],
            "cv_predictions": cv_result["predictions"],
            "test_predictions": test_predictions,
            "test_probabilities": test_probabilities,
            "artifact": TrainArtifacts(
                model=fitted,
                embedder_transformer=preprocessor,
                scaler=scaler,
                feature_names=[config["data"]["text_column"]],
            ),
        }

    results["labels"] = {
        "train_y": train_y,
        "test_y": test_y,
        "full_y": full_labels,
    }
    results["features"] = {
        "train_matrix": train_matrix,
        "test_matrix": test_matrix,
        "full_matrix": full_matrix,
    }
    return results


def save_model(artifact: TrainArtifacts, path: str | Path) -> None:
    """Persist a trained model artifact."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("wb") as handle:
        pickle.dump(artifact, handle)


def load_model(path: str | Path) -> TrainArtifacts:
    """Load a persisted model artifact."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
