# Persian COVID-19 Fake News Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Detect fake Persian COVID-19 news on **Twitter** and **Telegram** using **LaBSE sentence embeddings** and classical machine learning classifiers (SVM, Logistic Regression, KNN, XGBoost, Stacking Ensemble).

This repository is a reproducible, open-source refactor of the research project by **Narges Sedrehneshin**. The accompanying paper is available in [`docs/paper/paper.pdf`](docs/paper/paper.pdf).

---

## Overview

During the COVID-19 pandemic, Persian health-related misinformation spread rapidly on social media. Manual fact-checking does not scale for low-resource languages such as Persian. This project:

1. Collects Persian COVID-19 tweets and Telegram messages.
2. Applies **weak supervision** by labeling posts based on source trustworthiness.
3. Converts text to dense vectors with **LaBSE** (Language-agnostic BERT Sentence Embedding).
4. Trains and compares multiple ML classifiers to distinguish fake from real news.

Reported best performance in the original study: **~92% F1-score** with SVM on the Twitter dataset and comparable results on the Telegram (Sepehr-RumTel01) dataset.

---

## Dataset

| File | Location | Rows | Description |
|------|----------|------|-------------|
| `labeled_dataset_v3.csv` | `data/raw/` | 1,869 | **Primary labeled dataset** for model training. Columns: `tweet`, `Label` (0 = fake/false rumor, 1 = real/true rumor). Weakly supervised labels. |
| `twitter_covid_raw.csv` | `data/raw/` | 9,043 | **Raw Twitter crawl** with full metadata (user, timestamps, engagement, media flags). Labels: `Fake` / `Real`. |
| `preprocessed_balanced.csv` | `data/processed/` | 7,170 | **Balanced, preprocessed** dataset with engineered features (`urls`, `photos`, `video`, `text_length`). Used for exploratory analysis and class balancing. |

### Label semantics

| Value | Meaning |
|-------|---------|
| `0` | Fake / false rumor |
| `1` | Real / true rumor |

### Weak supervision

Labels were assigned based on **source trustworthiness** (e.g., verified outlets vs. rumor channels) rather than exhaustive manual annotation. This reduces labeling cost but introduces label noise — see [Limitations](#scientific-limitations-and-future-work).

---

## **Dataset (Public release notes)**

- **Original collection:** Data was crawled from public social platforms (Twitter and Telegram) and assembled into raw CSV files under `data/raw/`.
- **Cleaning process:** PII (user IDs, usernames, profile URLs, geo coordinates, media links) was identified and removed for the public release. Preprocessing steps (tokenization, normalization, URL removal) were applied in `src/preprocessing.py`.
- **Public artifacts:** The repository includes a curated `data/public/` folder with a cleaned CSV (`cleaned_public_dataset.csv`) containing only `text`, `label`, `source` and a `tweet_ids.csv` for the Twitter dump. Raw tweet metadata remain in `data/raw/` but are not redistributed in the public release.
- **Redistribution:** Due to Twitter/X content ownership and redistribution policies, the repository does not include the complete raw tweet corpus. Use `data/public/tweet_ids.csv` with the Twitter API to hydrate tweet text where permitted.

**Note on partial uploads & hashing:** Some public artifacts are intentionally partial: where raw tweet text or message content would raise copyright or privacy concerns, we replace the original text with a one-way SHA-256 hash. This preserves provenance and enables researchers to match records without redistributing copyrighted content. Hashed CSVs appear under `data/public/` with filenames prefixed `hashed_` and any original free-text columns are replaced by corresponding `*_hash` columns. If you need the full text for research, use the provided `tweet_ids.csv` to rehydrate tweets via the platform API in accordance with the platform's terms of service.

For full details see `DATASET_AUDIT.md` and `DATASET_CARD.md`.

## Methodology

### 1. Data collection

Persian COVID-19 posts were collected from Twitter and Telegram during the pandemic. The raw Twitter file retains social metadata; the labeled subset focuses on tweet text for fair, content-only classification.

### 2. Weak supervision labeling

Posts were labeled according to the reliability of their publishing source. This weakly supervised approach enables larger datasets without full manual fact-checking, at the cost of noisier ground truth.

### 3. Text preprocessing

The refactored pipeline (`src/preprocessing.py`) applies:

- URL removal
- Emoji removal
- Persian normalization via [Hazm](https://github.com/sobhe/hazm) (optional)
- Duplicate removal
- Minimum length filtering
- Reproducible shuffling (`random_state=42`)

### 4. Sentence Transformers (LaBSE)

Text is embedded with [`sentence-transformers/LaBSE`](https://huggingface.co/sentence-transformers/LaBSE), pretrained on 109 languages including Persian. LaBSE captures cross-lingual semantic similarity, which supports fake-news detection based on textual content.

### 5. Feature extraction

Each tweet is mapped to a 768-dimensional vector. Features are optionally **standardized** (`StandardScaler`) before classifier training.

### 6. Machine learning classifiers

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear baseline (`solver=liblinear`) |
| **KNN** | k-Nearest Neighbors (`k=4`) |
| **SVM** | RBF Support Vector Machine (`C=100`, `probability=True`) |
| **XGBoost** | Gradient boosted trees |
| **Stacking Ensemble** | Meta-learner (SVM) over LR + KNN + SVM + XGBoost |

Evaluation uses **stratified 70/30 train-test split** and **10-fold × 3-repeat stratified cross-validation**.

---

## Project Structure

```
persian-covid-fake-news/
├── config/
│   └── default.yaml          # Reproducible experiment configuration
├── data/
│   ├── raw/                  # Original CSV datasets
│   └── processed/            # Balanced / preprocessed data
## Dataset

This repository contains the original raw data (kept under `data/raw/` for reproducibility) and a set of public, privacy-preserving artifacts under `data/public/` suitable for GitHub distribution.

### Core files (public-ready)

- `data/public/cleaned_public_dataset.csv` — curated public table with the research labels and hashed text. Columns: `tweet_id`, `tweet_text_hash`, `label`, `date`, `language` (when available). Raw free-text is not included.
- `data/public/tweet_ids.csv` — Twitter IDs and labels (use this with the Twitter API to rehydrate tweets where permitted by the platform terms).
- `data/public/hashed_<orig>.csv` — hashed copies of CSVs that originally contained free text (column names ending with `_hash` replace the original text columns).

### Label semantics

- `fake` / `real` — labels are provided in a consistent lowercase string form. Original numeric encodings (`0`/`1`) were normalized to `fake`/`real`.

### Public release notes

- **Original collection:** Data were crawled from public social platforms (Twitter and Telegram); full raw CSVs are retained under `data/raw/` for provenance but are not suitable for redistribution on GitHub.
- **Privacy & copyright:** To prepare a GitHub-safe release we removed or replaced PII and copyrighted text. Free-text columns were replaced by one-way SHA-256 hashes so researchers may match records without redistributing content. See `data/public/` for the generated hashed files.
- **Rehydration:** If you require full tweet text for research, use `data/public/tweet_ids.csv` with the official platform API and follow that platform's terms of service.

For full details see `DATASET_AUDIT.md` and `DATASET_CARD.md`.

- Python 3.10+
- ~2 GB disk space (LaBSE model cache)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/persian-covid-fake-news.git
cd persian-covid-fake-news

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
# Or editable install:
pip install -e ".[notebooks,dev]"
```

---

## Usage

### Preprocess / inspect data

```bash
python -m src.cli preprocess
```

### Train all models (Twitter)

```bash
python -m src.cli train --platform twitter
```

### Train on Telegram-labeled data

```bash
python -m src.cli train --platform telegram
```

### Custom dataset path

```bash
python -m src.cli train --dataset data/processed/preprocessed_balanced.csv --platform balanced
```

---

## Training

The training pipeline (`src/training.py`):

1. Loads and preprocesses the dataset
2. Splits data (stratified, seed=42)
3. Embeds text with LaBSE
4. Trains LR, KNN, SVM, XGBoost, and Stacking
5. Runs repeated stratified cross-validation
6. Saves the best model to `models/<platform>/`
7. Writes metrics and figures to `results/<platform>/`

Configure hyperparameters in [`config/default.yaml`](config/default.yaml).

---

## Evaluation

Metrics implemented in `src/evaluation.py`:

| Metric | Function |
|--------|----------|
| Accuracy | `accuracy_score` |
| Precision | `precision_score` (binary) |
| Recall | `recall_score` (binary) |
| F1-score | `f1_score` (binary) |
| ROC-AUC | `roc_auc_score` (when probabilities available) |
| Confusion Matrix | `confusion_matrix` |

### Generated figures (`results/<platform>/`)

- `class_distribution.png` — label balance
- `embedding_pca.png` — 2D PCA of LaBSE vectors
- `model_comparison_f1.png` / `model_comparison_accuracy.png`
- `confusion_matrix_<model>.png` — per-model confusion matrices
- `roc_curve_<model>.png` — ROC curves (probability-based models)
- `evaluation_report.json` — full metrics JSON
- `model_comparison.csv` — tabular comparison

---

## Results

Original paper results (approximate):

| Platform | Best Model | F1-Score |
|----------|------------|----------|
| Twitter | SVM | ~92% |
| Telegram (Sepehr-RumTel01) | SVM | ~92% |

Re-run the pipeline to reproduce results on your hardware:

```bash
python -m src.cli train --platform twitter
cat results/twitter/model_comparison.csv
```

> **Note:** Exact numbers may vary slightly due to library updates (PyTorch, scikit-learn, XGBoost). All random seeds are fixed in `config/default.yaml`.

---

## Reproducibility

| Control | Location |
|---------|----------|
| Random seed | `config/default.yaml` → `project.random_seed: 42` |
| Train/test split | `sklearn.model_selection.train_test_split(..., random_state=42, stratify=y)` |
| Cross-validation | `RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)` |
| Model hyperparameters | `config/default.yaml` → `training.*` |
| Embedding model | `sentence-transformers/LaBSE` (pinned in config) |

To change any setting, edit `config/default.yaml` or pass `--config path/to/custom.yaml`.

---

## Citation

If you use this dataset or code, please cite:

```bibtex
@misc{sedrehneshin2022persian,
  author       = {Sedrehneshin, Narges},
  title        = {Fake News Detection on Twitter using Semantic Similarity and Ensemble Learning},
  year         = {2022},
  note         = {Persian COVID-19 fake news detection on social media},
  howpublished = {\url{https://github.com/YOUR_USERNAME/persian-covid-fake-news}}
}
```

Paper (Persian): [`docs/paper/paper.pdf`](docs/paper/paper.pdf)

**Keywords:** fake news, semantic similarity, machine learning, social media, Persian NLP, COVID-19

---

## Scientific Limitations

### Limitations

1. **Weak supervision noise** — Labels derived from source trustworthiness may not reflect ground truth for individual posts.
2. **Text-only features** — User metadata, propagation graphs, and engagement signals were excluded to avoid bias, but they could improve detection.
3. **Temporal drift** — Data from 2020 may not generalize to current misinformation campaigns.
4. **Class imbalance** — The primary labeled set has ~65% real / 35% fake; results depend on stratification and metric choice.
5. **Telegram notebook parity** — Original Telegram notebook duplicated Twitter code and used the same CSV; platform-specific Telegram raw data was not bundled separately.
6. **Computational cost** — LaBSE embedding of thousands of tweets requires GPU or patience on CPU.




## Author

**Narges Sedrehneshin**

---

## Acknowledgments

- [LaBSE](https://arxiv.org/abs/2007.01852) — Language-agnostic BERT Sentence Embedding
- [Hazm](https://github.com/sobhe/hazm) — Persian NLP toolkit
- [Sentence Transformers](https://www.sbert.net/) — embedding framework
