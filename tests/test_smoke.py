"""Smoke tests for the Persian COVID-19 fake news detection package."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import compute_metrics
from src.preprocessing import clean_persian_text, load_dataset
from src.utils import load_config


def test_clean_persian_text_removes_urls() -> None:
    text = "این یک خبر است https://example.com/page"
    cleaned = clean_persian_text(text, normalize_persian=False)
    assert "https" not in cleaned
    assert "خبر" in cleaned


def test_load_dataset_shape() -> None:
    config = load_config()
    dataset_path = PROJECT_ROOT / config["paths"]["raw_dataset"]
    if not dataset_path.exists():
        return
    dataframe = load_dataset(str(dataset_path), config)
    assert len(dataframe) > 0
    assert config["data"]["label_column"] in dataframe.columns


def test_compute_metrics_perfect_prediction() -> None:
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = [0, 1, 0, 1]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
