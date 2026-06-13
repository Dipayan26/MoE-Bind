"""Train a protein language model (pre-training or fine-tuning) from a YAML config.

Usage:
    python -m scripts.train --config configs/pretrain/demo/gpt2.yaml
"""

import argparse

from src.training.trainer import train_model


def main():
    parser = argparse.ArgumentParser(description="Train protein language models")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    train_model(args.config)


if __name__ == "__main__":
    main()
