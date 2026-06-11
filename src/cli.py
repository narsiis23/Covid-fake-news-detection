"""Command-line interface for the Persian COVID-19 fake news detection pipeline."""

from __future__ import annotations

import argparse
import logging

from src.evaluation import metrics_table, save_evaluation_report, summarize_training_results
from src.preprocessing import load_dataset
from src.training import save_model, train_all_models
from src.utils import ensure_dir, load_config, resolve_path, set_random_seed
from src.visualization import generate_all_figures

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Persian COVID-19 fake news detection pipeline",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML configuration file",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate all models")
    train_parser.add_argument(
        "--dataset",
        default=None,
        help="Override dataset path from config",
    )
    train_parser.add_argument(
        "--platform",
        choices=["twitter", "telegram"],
        default="twitter",
        help="Platform tag used for output naming",
    )

    preprocess_parser = subparsers.add_parser("preprocess", help="Load and preview dataset")
    preprocess_parser.add_argument("--dataset", default=None)

    return parser


def run_preprocess(args: argparse.Namespace) -> None:
    """Load and preview the dataset according to the provided args."""
    config = load_config(args.config)
    dataset_path = args.dataset or config["paths"]["raw_dataset"]
    dataframe = load_dataset(dataset_path, config)
    logger.info("Loaded %d samples from %s", len(dataframe), dataset_path)
    logger.info("Label distribution:\n%s", dataframe[config["data"]["label_column"]].value_counts())


def run_train(args: argparse.Namespace) -> None:
    """Train models and save evaluation artifacts for the selected platform."""
    config = load_config(args.config)
    set_random_seed(config["project"]["random_seed"])

    dataset_path = args.dataset or config["paths"]["raw_dataset"]
    dataframe = load_dataset(dataset_path, config)

    results_dir = resolve_path(config["paths"]["results_dir"], config) / args.platform
    models_dir = resolve_path(config["paths"]["models_dir"], config) / args.platform
    ensure_dir(results_dir)
    ensure_dir(models_dir)

    logger.info("Training on %d samples (%s)", len(dataframe), args.platform)
    training_results = train_all_models(dataframe, config)
    summary = summarize_training_results(training_results, config)
    metrics_df = metrics_table(summary)

    save_evaluation_report(summary, results_dir / "evaluation_report.json")
    metrics_df.to_csv(results_dir / "model_comparison.csv", index=False)
    generate_all_figures(training_results, summary, metrics_df, config, results_dir)

    best_model = metrics_df.iloc[0]["model"]
    artifact = training_results["models"][best_model]["artifact"]
    save_model(artifact, models_dir / f"{best_model}.pkl")

    logger.info("Model comparison (test set):\n%s", metrics_df.to_string(index=False))
    logger.info("Best model: %s", best_model)
    logger.info("Results saved to: %s", results_dir)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        run_preprocess(args)
    elif args.command == "train":
        run_train(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
