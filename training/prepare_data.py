"""Command-line helper for building processed training data."""
from __future__ import annotations

import argparse
from pathlib import Path

from training.data_pipeline import main as build_processed_data


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare multilingual government chatbot data")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Directory for processed artifacts")
    return parser.parse_args()


def main() -> None:
    """Delegate to the shared data-pipeline entry point."""
    args = parse_args()
    import sys

    sys.argv = [sys.argv[0], "--data-dir", args.data_dir, "--processed-dir", args.processed_dir]
    build_processed_data()


if __name__ == "__main__":
    main()
