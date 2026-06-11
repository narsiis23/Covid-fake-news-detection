"""Text preprocessing and dataset loading for Persian fake news detection."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
from sklearn.utils import shuffle

from src.utils import resolve_path, set_random_seed

try:
    from hazm import Normalizer as HazmNormalizer
except ImportError:  # pragma: no cover - optional dependency
    HazmNormalizer = None

URL_PATTERN = re.compile(
    r"https?://[^\s]+|www\.[^\s]+|t\.co/[^\s]+",
    flags=re.IGNORECASE,
)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_persian_text(
    text: str,
    *,
    remove_urls: bool = True,
    remove_emojis: bool = True,
    normalize_persian: bool = True,
    remove_extra_whitespace: bool = True,
) -> str:
    """Clean a single Persian text sample."""
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()

    if remove_urls:
        cleaned = URL_PATTERN.sub(" ", cleaned)

    if remove_emojis:
        cleaned = EMOJI_PATTERN.sub(" ", cleaned)

    if normalize_persian and HazmNormalizer is not None:
        cleaned = HazmNormalizer().normalize(cleaned)

    if remove_extra_whitespace:
        cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()

    return cleaned


def preprocess_texts(
    texts: Iterable[str],
    config: dict,
) -> list[str]:
    """Apply preprocessing to an iterable of texts."""
    prep_cfg = config.get("preprocessing", {})
    return [
        clean_persian_text(
            text,
            remove_urls=prep_cfg.get("remove_urls", True),
            remove_emojis=prep_cfg.get("remove_emojis", True),
            normalize_persian=prep_cfg.get("normalize_persian", True),
            remove_extra_whitespace=prep_cfg.get("remove_extra_whitespace", True),
        )
        for text in texts
    ]


def load_dataset(path: str, config: dict) -> pd.DataFrame:
    """Load and clean a labeled dataset CSV."""
    data_cfg = config["data"]
    seed = config["project"]["random_seed"]

    dataframe = pd.read_csv(resolve_path(path, config))
    dataframe = dataframe.loc[:, ~dataframe.columns.str.match(r"^Unnamed")]

    text_col = data_cfg["text_column"]
    label_col = data_cfg["label_column"]

    if text_col not in dataframe.columns:
        raise ValueError(f"Missing text column '{text_col}' in {path}")

    if label_col not in dataframe.columns:
        raise ValueError(f"Missing label column '{label_col}' in {path}")

    dataframe = dataframe.dropna(subset=[text_col, label_col]).copy()
    dataframe[text_col] = preprocess_texts(dataframe[text_col].astype(str), config)

    min_len = config.get("preprocessing", {}).get("min_text_length", 1)
    dataframe = dataframe[dataframe[text_col].str.len() >= min_len]

    if data_cfg.get("drop_duplicates", True):
        dataframe = dataframe.drop_duplicates(subset=text_col, keep="first")

    if data_cfg.get("shuffle", True):
        set_random_seed(seed)
        dataframe = shuffle(dataframe, random_state=seed)

    dataframe = dataframe.reset_index(drop=True)
    return dataframe


def normalize_raw_twitter_labels(dataframe: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Map Twitter raw labels (Fake/Real strings) to binary integers."""
    data_cfg = config["data"]
    label_col = data_cfg["label_column"]
    mapping = {
        "Fake": data_cfg["fake_label"],
        "Real": data_cfg["real_label"],
        0: data_cfg["fake_label"],
        1: data_cfg["real_label"],
    }
    dataframe = dataframe.copy()
    dataframe[label_col] = dataframe[label_col].map(mapping)
    return dataframe.dropna(subset=[label_col])
