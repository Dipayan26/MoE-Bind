"""
Batch binder generation with quality filtering.
Reads a generation YAML, loads the `active_model` (or every model with --all-models)

Usage:
    python -m scripts.generate_batch --config configs/generate/deepseek.yaml
    python -m scripts.generate_batch --config configs/generate/deepseek.yaml --all-models
"""

import argparse
import math
import os
from collections import Counter

import pandas as pd
import torch
import torch.nn.functional as F

from scripts.generate import generate
from src.data.tokenization.protein_character_tokenizer import ProteinTokenizerHF
from src.models.MOE_latent_bind import DeepSeekV3
from src.models.config import DeepSeekConfig, GPT2Config, LLaMA2Config
from src.models.gpt2_style import GPT2
from src.models.Llama2_style import Llama2
from src.utils.config import load_yaml


def _repetitiveness_score(seq: str) -> float:
    """Fraction of sequence occupied by the most common amino acid (0–1)."""
    if not seq:
        return 0.0
    return max(Counter(seq).values()) / len(seq)


@torch.no_grad()
def _compute_binder_ppl(model, prompt_ids: torch.Tensor, binder_ids: torch.Tensor) -> float:
    """
    Teacher-forced NLL over binder tokens.

    GPT2/LLaMA2 return full-sequence logits [1, seq_len, vocab] — handled in one
    forward pass.  DeepSeek returns only the last-position logit [1, 1, vocab], so
    we fall back to the token-by-token approach (feed growing prefix, read last logit).
    """
    full_ids = torch.cat([prompt_ids, binder_ids], dim=1)
    block_size = model.config.block_size
    if full_ids.shape[1] > block_size:
        full_ids = full_ids[:, -block_size:]
    binder_len = min(binder_ids.shape[1], full_ids.shape[1] - 1)
    if binder_len == 0:
        return float("inf")

    outputs = model(full_ids)
    logits = outputs["logits"]   # [1, L, vocab]  or  [1, 1, vocab]

    if logits.shape[1] >= full_ids.shape[1]:
        # Full-sequence logits — one-pass efficient computation
        pred_logits = logits[0, -(binder_len + 1):-1, :].contiguous()
        target = full_ids[0, -binder_len:].contiguous()
        loss = F.cross_entropy(pred_logits, target)
        return math.exp(loss.item())
    else:
        # Last-position logits only — token-by-token (e.g. DeepSeek)
        prompt_end = full_ids.shape[1] - binder_len
        nlls = []
        for i in range(binder_len):
            pos = prompt_end + i
            ctx = full_ids[:, :pos]
            if ctx.shape[1] > block_size:
                ctx = ctx[:, -block_size:]
            out = model(ctx)
            log_prob = F.log_softmax(out["logits"][0, -1, :], dim=-1)
            nlls.append(-log_prob[full_ids[0, pos]].item())
        return math.exp(sum(nlls) / len(nlls))


def _longest_run(seq: str) -> int:
    """Length of the longest consecutive run of a single amino acid."""
    if not seq:
        return 0
    run = max_run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        if run > max_run:
            max_run = run
    return max_run


def _apply_filters(model, prompt_ids, candidates, filter_cfg, device, tokenizer):
    """
    Three-stage filter:
      0. Hard cutoff  — discard any sequence with a consecutive run > max_run_len.
      1. Repetitiveness — drop top rep_top_pct% by max single-AA fraction.
      2. PPL filter   — drop top ppl_top_pct% highest-PPL from survivors.
    min_survivors guarantees at least one candidate survives each stage.
    Returns (best_seq, best_ppl).
    """
    max_run_len = filter_cfg.get("max_run_len", None)
    rep_pct     = filter_cfg.get("repetitiveness_top_pct", 0.20)
    ppl_pct     = filter_cfg.get("ppl_top_pct", 0.30)
    min_surv    = max(1, filter_cfg.get("min_survivors", 1))


    if max_run_len is not None:
        passed = [s for s in candidates if _longest_run(s) <= max_run_len]
        if len(passed) >= min_surv:
            candidates = passed
        else:

            candidates = sorted(candidates, key=_longest_run)[:min_surv]


    scored = sorted(candidates, key=_repetitiveness_score)
    n_drop = min(int(len(scored) * rep_pct), len(scored) - min_surv)
    survivors = scored[:len(scored) - n_drop] if n_drop > 0 else scored


    ppl_scored = []
    for seq in survivors:
        b_ids = tokenizer.encode(seq, return_tensors="pt", add_special_tokens=False).to(device)
        ppl_scored.append((seq, _compute_binder_ppl(model, prompt_ids, b_ids)))
    ppl_scored.sort(key=lambda x: x[1])

    n_drop2 = min(int(len(ppl_scored) * ppl_pct), len(ppl_scored) - min_surv)
    final_pool = ppl_scored[:len(ppl_scored) - n_drop2] if n_drop2 > 0 else ppl_scored

    best_seq, best_ppl = final_pool[0]
    return best_seq, best_ppl


# ── Model loader ───────────────────────────────────────────────────────────────

def _load_model(model_cfg_dict: dict, device: str):
    run_type = model_cfg_dict["run_type"]
    ckpt_path = model_cfg_dict["checkpoint"]
    arch = model_cfg_dict["arch"]

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)

    if "GPT2" in run_type:
        cfg = GPT2Config(**arch)
        model = GPT2(cfg)
    elif "LLAMA2" in run_type:
        cfg = LLaMA2Config(**arch)
        model = Llama2(cfg)
    elif "MOE_bind" in run_type:
        cfg = DeepSeekConfig(**arch)
        model = DeepSeekV3(cfg)
    else:
        raise ValueError(f"Unknown run_type: {run_type}")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


# ── Main ───────────────────────────────────────────────────────────────────────

def _run_one_model(cfg: dict, active_model: str, device: str) -> None:
    model_cfg_dict = cfg["models"][active_model]
    gen_cfg = cfg["generation"]

    out_dir     = cfg["output"]["dir"]
    out_csv     = os.path.join(out_dir, f"{active_model}.csv")
    partial_csv = os.path.join(out_dir, f"{active_model}.partial.csv")

    if os.path.exists(out_csv):
        print(f"[SKIP] {active_model} — output already exists: {out_csv}")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Resume from partial checkpoint if it exists
    done_ids: set[str] = set()
    if os.path.exists(partial_csv):
        partial_df = pd.read_csv(partial_csv)
        done_ids = set(partial_df["complex_id"].astype(str))
        print(f"[RESUME] {active_model} — {len(done_ids)} rows already done, continuing from checkpoint.")

    # Load cached results from a prior run (e.g. 22-pair results reused in 121-pair run)
    cache_lookup: dict[str, dict] = {}
    cache_dir = cfg["output"].get("cache_dir", None)
    if cache_dir:
        cache_csv = os.path.join(cache_dir, f"{active_model}.csv")
        if os.path.exists(cache_csv):
            cache_df = pd.read_csv(cache_csv)
            for _, crow in cache_df.iterrows():
                cache_lookup[str(crow["complex_id"])] = crow.to_dict()
            print(f"[CACHE] Loaded {len(cache_lookup)} cached rows from {cache_csv}")

    print(f"\n{'='*60}")
    print(f"Active model : {active_model}")
    print(f"Device       : {device}")
    print(f"Checkpoint   : {model_cfg_dict['checkpoint']}")
    print(f"{'='*60}")

    tokenizer = ProteinTokenizerHF()
    vocab = tokenizer.vocab
    closing_b_id   = vocab["</PROTEIN_B>"]
    stop_token_ids = torch.tensor([closing_b_id], dtype=torch.long)

    print("Loading model...")
    model = _load_model(model_cfg_dict, device)
    print("Model loaded.\n")

    df = pd.read_csv(cfg["data"]["benchmark_csv"])
    max_pairs = cfg["data"].get("max_pairs", None)
    if max_pairs is not None:
        df = df.head(int(max_pairs))
        print(f"[DATA] max_pairs={max_pairs} — using first {len(df)} rows.")
    target_col     = cfg["data"]["target_col"]
    length_ref_col = cfg["data"]["length_ref_col"]

    num_samples = gen_cfg.get("num_samples", 1)
    filter_cfg  = cfg.get("filtering", None)


    write_header = not os.path.exists(partial_csv)
    partial_fh = open(partial_csv, "a", buffering=1)  # line-buffered

    n_done_before = len(done_ids)
    n_newly_done  = 0

    for _, row in df.iterrows():
        cid          = str(row["complex_id"])
        if cid in done_ids:
            continue  


        if cid in cache_lookup:
            cached = cache_lookup[cid]
            result_row = {
                "complex_id":    cid,
                "generated_seq": cached["generated_seq"],
                "seq_len":       cached["seq_len"],
                "ppl":           cached["ppl"],
            }
            pd.DataFrame([result_row]).to_csv(partial_fh, header=write_header, index=False)
            write_header = False
            partial_fh.flush()
            done_ids.add(cid)
            n_newly_done += 1
            print(f"  {cid:8s}  [CACHED]  seq_len={int(cached['seq_len'])}  ppl={cached['ppl']}")
            continue

        receptor_seq = str(row[target_col]).strip()
        ligand_seq   = str(row[length_ref_col]).strip()
        target_len   = len(ligand_seq)

        prompt = f"<PROTEIN_A>{receptor_seq}</PROTEIN_A><PROTEIN_B>"
        input_ids = tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=False,
        ).to(device)

        max_new = target_len + max(10, int(target_len * 0.2))

        candidates = []
        for _ in range(num_samples):
            with torch.no_grad():
                output_ids = generate(
                    model, input_ids=input_ids, max_new_tokens=max_new,
                    temperature=gen_cfg["temperature"],
                    top_k=gen_cfg.get("top_k", None),
                    top_p=gen_cfg.get("top_p", None),
                    repetition_penalty=gen_cfg.get("repetition_penalty", 1.0),
                    MOE_route=False, stop_token_ids=stop_token_ids,
                )
            gen_ids = output_ids[0, input_ids.shape[1]:]
            raw = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
            aa_seq = raw.replace(" ", "")
            if len(aa_seq) >= target_len:
                aa_seq = aa_seq[:target_len]
            else:
                aa_seq = aa_seq.ljust(target_len, "A")
            candidates.append(aa_seq)

        if filter_cfg is not None and len(candidates) > 1:
            aa_seq, best_ppl = _apply_filters(
                model, input_ids, candidates, filter_cfg, device, tokenizer
            )
            print(f"  {cid:8s}  target_len={target_len:4d}  gen_len={len(aa_seq):4d}"
                  f"  ppl={best_ppl:.2f}  run={_longest_run(aa_seq)}  seq[:20]={aa_seq[:20]}")
        else:
            aa_seq = candidates[0]
            best_ppl = None
            print(f"  {cid:8s}  target_len={target_len:4d}  gen_len={len(aa_seq):4d}  seq[:20]={aa_seq[:20]}")

        result_row = {
            "complex_id":    cid,
            "generated_seq": aa_seq,
            "seq_len":       len(aa_seq),
            "ppl":           round(best_ppl, 4) if best_ppl is not None else None,
        }

        pd.DataFrame([result_row]).to_csv(
            partial_fh, header=write_header, index=False
        )
        write_header = False  
        partial_fh.flush()
        done_ids.add(cid)
        n_newly_done += 1

    partial_fh.close()


    final_df = pd.read_csv(partial_csv)
    final_df.to_csv(out_csv, index=False)
    os.remove(partial_csv)
    print(f"Saved {len(final_df)} sequences ({n_done_before} resumed + {n_newly_done} new) → {out_csv}")


    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(config_path: str, all_models: bool = False):
    cfg = load_yaml(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if all_models:
        model_list = list(cfg["models"].keys())
        print(f"Running all {len(model_list)} models sequentially:")
        for i, name in enumerate(model_list, 1):
            print(f"  [{i}/{len(model_list)}] {name}")
        for name in model_list:
            _run_one_model(cfg, name, device)
        print("\nAll models done.")
        return

    _run_one_model(cfg, cfg["active_model"], device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to generate_binders_all_models.yaml")
    parser.add_argument("--all-models", action="store_true",
                        help="Run all models listed in the config sequentially")
    args = parser.parse_args()
    main(args.config, all_models=args.all_models)
