# MoE-Bind

De novo protein binder design has been dominated by structure-based pipelines
that require known three-dimensional target conformations and consume
substantial compute and generation time per design, limiting their throughput
and accessibility for routine large-scale binder exploration.
Sequence-only generative models promise a faster and lighter alternative, yet
existing systems remain uniformly dense and frequently reintroduce structural
computation at inference, undermining the core advantages they were intended
to deliver.

We present MoE-Bind, an autoregressive protein binder generator that,
for the first time in this domain, combines Multi-head Latent Attention with
a sparse Mixture-of-Experts feed-forward network
Despite activating less than half the per-token parameters of compute-matched
dense baselines, MoE-Bind matches or exceeds them on full-length
receptor-conditioned binder generation
Routing analysis on generated binders reveals interpretable expert
specialization at both the individual amino acid and biochemical group level,
a structured expert-token alignment not previously reported for
natural-language MoE models. These results show that sparse architectural design, rather than scale, can deliver fast, structure-free, and interpretable protein binder generation.

<img src="moebind-diag.png" alt="Project Diagram" width="800">


| Architecture | Attention | FFN 
|---|---|---|
| **MHA** (GPT-2 style) | Multi-Head Attention (dense) | MLP (GELU)
| **GQA** (LLaMA2 style) | Grouped-Query Attention | MLP (SwiGLU) 
| **MLA + MoE** (DeepSeekV3 style) | Multi-head Latent Attention | sparse MoE (top-2/8 + shared)


> This repository contains the model training code. A small demo dataset is
> included so the entire pipeline can be run end to end in a few minutes on
> CPU or GPU. The 100M paper-scale configs are also provided.

## Installation

```bash
# Option A — conda (recommended)
conda env create -f environment.yml
conda activate moebind

# Option B — pip / venv
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- `requirements.txt` pins the **CUDA 12.1** PyTorch wheels (`torch==2.5.1+cu121`).
- For CPU-only or a different CUDA version, install torch separately from the
  default index, e.g. `pip install torch==2.5.1`, then `pip install -r requirements.txt`.
- Training logs to **Weights & Biases**. To run without logging in, set
  `export WANDB_MODE=offline` (or run `wandb login`).
- All commands are run as modules (`python -m scripts.<name>`) from the repo
  root.

## Quick demo (~ Fast, CPU-friendly)

The demo uses small models so the whole loop runs fast.

```bash
export WANDB_MODE=offline

# 1. Tokenize the raw pre-training FASTA -> .bin
python -m scripts.data_tokenize --config configs/data_tokenize/demo_pretrain_tokenize.yaml

# 2. Pre-train (run any / all of the three architectures)
python -m scripts.train --config configs/pretrain/demo/gpt2.yaml       # MHA
python -m scripts.train --config configs/pretrain/demo/llama2.yaml     # GQA
python -m scripts.train --config configs/pretrain/demo/deepseek.yaml   # MLA + MoE

# 3. Tokenize the raw fine-tuning CSV -> Arrow datasets
python -m scripts.preprocess_finetune --config configs/finetune_preprocess/demo_finetune_preprocess.yaml

# 4. Fine-tune on PPI pairs
python -m scripts.train --config configs/finetune/demo/gpt2.yaml
python -m scripts.train --config configs/finetune/demo/llama2.yaml
python -m scripts.train --config configs/finetune/demo/deepseek.yaml

# 5. Generate receptor-conditioned binders (filtered: run-length + repetitiveness + perplexity)
python -m scripts.generate_batch --config configs/generate/gpt2.yaml
python -m scripts.generate_batch --config configs/generate/llama2.yaml
python -m scripts.generate_batch --config configs/generate/deepseek.yaml
```

Generation writes a CSV to `outputs/generated/<arch>/` with columns
`complex_id, generated_seq, seq_len, ppl`. Tune the `generation:` and
`filtering:` blocks in `configs/generate/*.yaml` to control sampling and
candidate selection.

## Reproducing at scale (100M)

The 100M paper-scale configs live in `configs/pretrain/*_100M.yaml` and
`configs/finetune/*_100M.yaml`. You supply your own corpora:

```bash
# Tokenize your full pre-training corpus (edit the FASTA path in the config first)
python -m scripts.data_tokenize --config configs/data_tokenize/pretrain_tokenize.yaml

# Pre-train, then fine-tune (example: DeepSeek)
python -m scripts.train --config configs/pretrain/deepseek_100M.yaml
python -m scripts.preprocess_finetune --config configs/finetune_preprocess/finetune_preprocess.yaml
python -m scripts.train --config configs/finetune/deepseek_100M.yaml
```

Approximate sizes: GPT2-100M ≈ 99.85M, LLaMA2-100M ≈ 100.7M, DeepSeek-100M ≈
102.7M total (~38.9M active per token; top-2 of 8 experts + shared expert).

## License

MIT — see [LICENSE](LICENSE).
<!-- 
## Citation -->