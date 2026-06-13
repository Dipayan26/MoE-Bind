# Demo data

These are intentionally small dataset, they are for smoke-testing the
code path.

| File | Rows | Format | Used by |
|---|---|---|---|
| `demo_pretrain.fasta` | 1,500 sequences | FASTA (raw amino acids) | `scripts/data_tokenize.py` → `.bin` |
| `demo_finetune.csv` | 600 pairs | CSV: `sequence1,sequence2` | `scripts/preprocess_finetune.py` → Arrow |
| `demo_benchmark.csv` | 5 targets | CSV: `complex_id,ligand_id,receptor_id,ligand_seq,receptor_seq` | `scripts/generate_batch.py` |

The data is **not** pre-tokenized, tokenization is the first step.

## Sources

- `demo_pretrain.fasta` — a length-filtered sample (40–250 aa) of plant protein
  sequences from UniProt.
- `demo_finetune.csv` — a length-filtered sample (combined length 30–240 aa) of
  physical protein–protein interaction pairs derived from STRING DB v12.0.
- `demo_benchmark.csv` — a small subset of the Docking Benchmark 5 (DB5)
  receptor/ligand complexes used as generation targets.

For the full datasets and exact preprocessing used in the paper, see the repo
README.
