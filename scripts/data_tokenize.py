"""Tokenize raw protein sequences (FASTA) into memory-mapped .bin files for pre-training.

Usage:
    python -m scripts.data_tokenize --config configs/data_tokenize/demo_pretrain_tokenize.yaml
"""

import argparse

from src.data.tokenization.data_tokenizer import tokenize_data


def main():
    parser = argparse.ArgumentParser(description="Tokenize protein sequences for training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    tokenize_data(args.config)


if __name__ == "__main__":
    main()
