"""
Usage:
    python -m scripts.preprocess_finetune --config configs/finetune_preprocess/demo_finetune_preprocess.yaml
"""

import argparse
from pathlib import Path

import yaml
from datasets import load_dataset

from src.data.tokenization.protein_character_tokenizer import ProteinTokenizerHF


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def tokenize_example(example, tokenizer, block_size, conditional_masking,
                     protein_b_id, closing_tag_id, special_ids):
    text = (
        "<PROTEIN_A>" + example["sequence1"] + "</PROTEIN_A>"
        "<PROTEIN_B>" + example["sequence2"] + "</PROTEIN_B>"
    )

    enc = tokenizer(
        text,
        truncation=True,
        padding=False,
        max_length=block_size,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"]

    if conditional_masking:
        targets = [-100] * len(input_ids)
        try:
            b_pos = input_ids.index(protein_b_id)
        except ValueError:
            b_pos = len(input_ids)

        for i in range(b_pos + 1, len(input_ids)):
            token_id = input_ids[i]
            is_closing_tag = (token_id == closing_tag_id)
            is_eos = (token_id == tokenizer.eos_token_id)
            if (token_id not in special_ids) or is_closing_tag or is_eos:
                targets[i] = token_id
    else:
        targets = list(input_ids)

    return {"input_ids": input_ids, "labels": targets}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to string_preprocess.yaml")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    cfg_dir = Path(args.config).parent

    csv_path          = cfg_dir / cfg["data"]["csv_path"]
    min_len           = cfg["filter"]["min_total_len"]
    max_len           = cfg["filter"]["max_total_len"]
    block_size        = cfg["tokenization"]["block_size"]
    conditional_masking = cfg["tokenization"]["conditional_masking"]
    test_size         = cfg["split"]["test_size"]
    seed              = cfg["split"]["seed"]
    train_out         = cfg_dir / cfg["output"]["train"]
    val_out           = cfg_dir / cfg["output"]["val"]

    # Build tokenizer 
    tokenizer = ProteinTokenizerHF()
    vocab = tokenizer.vocab

    protein_b_id  = vocab["<PROTEIN_B>"]
    closing_tag_id = vocab["</PROTEIN_B>"]
    special_ids   = {
        vocab["<PROTEIN_A>"],
        vocab["</PROTEIN_A>"],
        vocab["<PROTEIN_B>"],
        vocab["</PROTEIN_B>"],
        vocab["<pad>"],
    }

    # --- Step 1: Load CSV ---
    max_samples = cfg["data"].get("max_samples")  # None = use all data
    print(f"Loading CSV: {csv_path}")
    full_dataset = load_dataset("csv", data_files=str(csv_path))["train"]
    print("Shuffling rows...")
    full_dataset = full_dataset.shuffle(seed=seed)
    
    if max_samples:
        full_dataset = full_dataset.select(range(min(max_samples, len(full_dataset))))
        print(f"Using {len(full_dataset):,} rows (max_samples={max_samples})")
    else:
        print(f"Total rows: {len(full_dataset):,}")

    # --- Step 2: Filter by combined sequence length ---
    print("Filtering by sequence length...")
    full_dataset = full_dataset.filter(
        lambda ex: min_len <= len(ex["sequence1"]) + len(ex["sequence2"]) < max_len,
        num_proc=4,
        desc="Length filter",
    )
    print(f"After filter: {len(full_dataset):,}")

    # --- Step 3: Train / val split ---
    split    = full_dataset.train_test_split(test_size=test_size, seed=seed)
    train_hf = split["train"]
    val_hf   = split["test"]
    print(f"Train: {len(train_hf):,}  |  Val: {len(val_hf):,}")

    # --- Step 4: Tokenize in parallel ---
    def _tok(ex):
        return tokenize_example(
            ex, tokenizer, block_size, conditional_masking,
            protein_b_id, closing_tag_id, special_ids,
        )

    print("Tokenizing train split...")
    train_tok = train_hf.map(
        _tok, num_proc=4, desc="Tokenize train",
        remove_columns=train_hf.column_names,
    )

    print("Tokenizing val split...")
    val_tok = val_hf.map(
        _tok, num_proc=4, desc="Tokenize val",
        remove_columns=val_hf.column_names,
    )

    # --- Step 5: Save Arrow datasets to disk ---
    train_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving train → {train_out}")
    train_tok.save_to_disk(str(train_out))

    print(f"Saving val   → {val_out}")
    val_tok.save_to_disk(str(val_out))

    print("\nDone.")
    print(f"  preprocessed_train: {train_out}")
    print(f"  preprocessed_val:   {val_out}")


if __name__ == "__main__":
    main()
