# Dataset Card

## Overview

This dataset contains Persian-language social media posts collected during the COVID-19 pandemic. The original repository includes raw Twitter and Telegram CSVs under `data/raw/` and processed/balanced tables under `data/processed/`. For GitHub distribution we provide privacy-preserving public artifacts under `data/public/` where free-text columns have been replaced by one-way SHA-256 hashes.

## Motivation

The dataset was collected to support research on detecting misinformation in Persian-language social media posts, with a focus on COVID-19-related content.

## Composition

- Sources: Twitter (primary) and Telegram (secondary). Raw CSVs include full metadata (user, timestamps, engagement, media).
- Public artifacts: `data/public/cleaned_public_dataset.csv` (labels + hashed text + minimal metadata), `data/public/tweet_ids.csv` (tweet IDs and labels), and hashed copies of original CSVs `data/public/hashed_*.csv`.

## Labels

- `fake` — content labeled as false rumor or misinformation.
- `real` — content labeled as true/verified.

Labeling was done via weak supervision based on source reliability; consequently labels may contain noise and should be treated as approximate.

## Collection Process

- Tweets were collected via public crawls during 2020–2021. The raw collection contains tweet ids, user metadata, timestamps, counts (retweets/likes), and media links.
- No private or deleted content was intentionally included; the dataset only contains posts that were publicly accessible at collection time.

## Preprocessing

- Deduplication and normalization applied in `src/preprocessing.py`.
- For public release, PII and direct text were removed or replaced: free-text columns were hashed (SHA-256); user identifiers and profile fields were not included in public artifacts.
- Numeric label encodings (`0`/`1`) were normalized to `fake`/`real` in the public CSVs.

## Recommended Uses

- Training and evaluating text-classification models on metadata and label distributions (when text is rehydrated with IDs).
- Research on label noise and weak supervision methods in low-resource languages.

## Limitations

- Labels are weakly supervised and may contain systematic biases. Use with caution for high-stakes decisions.
- Hashed text in public artifacts prevents direct linguistic analysis without rehydration via the platform API.

## Ethics & Privacy

- Public artifacts intentionally avoid redistributing copyrighted text and reduce PII exposure.
- Researchers should follow platform terms of service and local laws when rehydrating tweet text using `tweet_ids.csv`.

## Contact

For questions about dataset provenance or to request additional detail, open an issue in this repository.
