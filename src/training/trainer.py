from datasets import load_dataset
import torch
import math
import os
import wandb
from tqdm.auto import tqdm
from contextlib import nullcontext
from src.models.MOE_latent_bind import DeepSeekV3
from src.data.data_loader.data_loader import get_pretrain_batch, estimate_loss_pretrain
from pathlib import Path
from torch.amp.autocast_mode import autocast
from torch.cuda.amp import GradScaler
import time
import yaml
from src.utils.config import load_yaml
from src.models.config import DeepSeekConfig, GPT2Config, LLaMA2Config
from src.data.Datasets.PPI_finetune_dataset import PPIFinetuneDataset, FinetuneCollator, PreTokenizedPPIDataset
from src.data.tokenization.protein_character_tokenizer import ProteinTokenizerHF
from torch.utils.data import DataLoader, RandomSampler
from src.models.gpt2_style import GPT2
from src.models.Llama2_style import Llama2

PRETRAIN_TYPES = ["pretrain_MOE_bind", "pretrain_gpt2", "pretrain_llama2"]
FINETUNE_TYPES = ["Full_finetune_MOE_bind", "Full_finetune_GPT2", "Full_finetune_LLAMA2"]


def build_model(run_type, model_cfg_dict):
    """Instantiate the model + config dataclass for a given run type."""
    if run_type in ("pretrain_MOE_bind", "Full_finetune_MOE_bind"):
        model_cfg = DeepSeekConfig(**model_cfg_dict)
        return model_cfg, DeepSeekV3(model_cfg)
    if run_type in ("pretrain_gpt2", "Full_finetune_GPT2"):
        model_cfg = GPT2Config(**model_cfg_dict)
        return model_cfg, GPT2(model_cfg)
    if run_type in ("pretrain_llama2", "Full_finetune_LLAMA2"):
        model_cfg = LLaMA2Config(**model_cfg_dict)
        return model_cfg, Llama2(model_cfg)
    raise ValueError(f"Unknown run type: {run_type}")


def train_model(config_path: str):

    cfg = load_yaml(config_path)
    run_type = cfg["run"]["type"]

    if run_type not in PRETRAIN_TYPES + FINETUNE_TYPES:
        raise ValueError(
            f"Unsupported run type '{run_type}'. "
            f"Use one of {PRETRAIN_TYPES + FINETUNE_TYPES}."
        )

    model_cfg, model = build_model(run_type, cfg["model"])

    train_cfg = cfg["training"]
    log_cfg = cfg["logging"]
    data_cfg = cfg["data"]

    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{train_cfg['out_mod_name']}.pt"

    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)
    print(f"Configuration saved to {ckpt_dir / 'config.yaml'}")

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = (nullcontext() if device_type == "cpu" else autocast(device_type="cuda", dtype=ptdtype))

    # Initialize wandb
    wandb.init(
        project=log_cfg["project"],
        id=log_cfg["id"],
        config={**cfg["model"], **cfg["training"], **cfg["logging"]}
        )

    torch.cuda.manual_seed_all(log_cfg["seed"])

    # ── Pre-training: random init, no checkpoint to load ──────────────────────
    if run_type in PRETRAIN_TYPES:
        print("Running pre-training...")
        model = model.to(device)

    # ── Full fine-tuning: load pre-trained weights into the model ─────────────
    else:
        print("Running full fine-tuning...")
        pretrained_path = cfg["run"]["pretrained_ckpt"]
        print(f"Loading weights from: {pretrained_path}")
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Checkpoint not found at {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # strict=False ignores unexpected keys (e.g. MTP heads from pre-training)
        if cfg["run"]["mod_pretrained_with_MTP"]:
            keys = model.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded weights (strict=False). Ignored {len(keys.unexpected_keys)} extra keys (MTP heads).")
        else:
            model.load_state_dict(state_dict)
            print("✓ Loaded weights (strict=True).")

        model = model.to(device)

    # ── Efficiency / FLOPs tracking ───────────────────────────────────────────
    fpt = model.flops_per_token()
    train_fpt = 6 * fpt   # fwd + 2x bwd
    tokens_per_iter = train_cfg["batch_size"] * model_cfg.block_size
    tokens_consumed = 0
    t0 = 0.0

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    eff_batch_tokens = train_cfg["batch_size"] * train_cfg["gradient_accumulation_steps"] * model_cfg.block_size
    print(f"{run_type} model with {total_params:,} parameters ({trainable_params:,} trainable)")
    print(f"FLOPs/token  fwd: {fpt/1e9:.3f} GFLOPs | train (×6): {train_fpt/1e9:.3f} GFLOPs")
    print(f"Efficiency | GPU: {gpu_name} | effective_batch: {eff_batch_tokens:,} tokens/step")
    wandb.config.update({
        "efficiency/total_params": total_params,
        "efficiency/trainable_params": trainable_params,
        "efficiency/effective_batch_tokens": eff_batch_tokens,
        "efficiency/gpu_name": gpu_name,
        "flops_per_token_fwd": fpt,
        "flops_per_token_train": train_fpt,
    }, allow_val_change=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    weight_decay = 0.01 if run_type in FINETUNE_TYPES else 0.1
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        eps=1e-9
    )

    # ── Fine-tuning data preparation ──────────────────────────────────────────
    if run_type in FINETUNE_TYPES:

        # Fast path: pre-tokenized Arrow datasets saved by scripts/preprocess_finetune.py
        if data_cfg.get("preprocessed_train") and data_cfg.get("preprocessed_val"):
            print("Loading pre-tokenized datasets from disk (fast path)...")
            train_ds = PreTokenizedPPIDataset(data_cfg["preprocessed_train"])
            val_ds   = PreTokenizedPPIDataset(data_cfg["preprocessed_val"])

            tokenizer = ProteinTokenizerHF()   # still needed for pad_token_id
            pad_id = tokenizer.pad_token_id or 0

        # Slow path: parse the raw CSV at runtime
        else:
            print("No pre-tokenized data found — loading from CSV (slow, high RAM)...")
            tokenizer = ProteinTokenizerHF()

            full_dataset = load_dataset("csv", data_files=data_cfg["csv_path"])["train"]

            def filter_sequence_length(example):
                total_len = len(example["sequence1"]) + len(example["sequence2"])
                return 30 <= total_len < model_cfg.block_size

            full_dataset = full_dataset.filter(filter_sequence_length)

            dataset_split = full_dataset.train_test_split(
                test_size=0.2,
                seed=log_cfg["seed"]
            )

            train_ds = PPIFinetuneDataset(
                dataset_split["train"],
                tokenizer,
                model_cfg.block_size,
                conditional_masking=train_cfg["conditional_masking"]
            )

            val_ds = PPIFinetuneDataset(
                dataset_split["test"],
                tokenizer,
                model_cfg.block_size,
                conditional_masking=train_cfg["conditional_masking"]
            )

            pad_id = tokenizer.pad_token_id or 0

        collator = FinetuneCollator(pad_token_id=pad_id)

        train_loader = DataLoader(
            train_ds,
            batch_size=train_cfg["batch_size"],
            sampler=RandomSampler(train_ds),
            collate_fn=collator,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            collate_fn=collator,
            pin_memory=True,
        )

        train_iter = iter(train_loader)

        print(f"Fine-tuning dataset loaded: {len(train_ds)} samples")

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    best_val_loss = float('inf')
    scaler = GradScaler(enabled=(dtype == 'float16'))

    lr = train_cfg["learning_rate"]  # default value

    for step in tqdm(range(train_cfg["max_iters"])):
        # Evaluation step
        if step % train_cfg["eval_iters"] == 0 and step != 0:

            if run_type in PRETRAIN_TYPES:
                losses = estimate_loss_pretrain(
                    model, data_cfg, model_cfg, train_cfg["eval_iters"], train_cfg["batch_size"], device_type, device, ctx
                )
                val_loss = losses["val"]
                wandb.log({
                    "step": step,
                    "train_loss": losses["train"],
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss
                })

            else:
                model.eval()
                val_losses = []
                with torch.inference_mode():
                    for i, (vX, vy) in enumerate(val_loader):
                        if i >= 20:
                            break
                        vX = vX.to(device)
                        vy = vy.to(device)
                        with ctx:
                            outputs = model(input_ids=vX[:, :-1], labels=vy[:, 1:])
                            v_loss = outputs["loss"]
                        val_losses.append(v_loss.item())
                val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
                model.train()
                wandb.log({
                    "step": step,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss
                })
            print(f"step {step}: Val Loss {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Saving full model...")
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "step": step,
                    "best_val_loss": best_val_loss
                }
                torch.save(checkpoint, ckpt_path)
                print(f"Best model saved with val loss: {best_val_loss:.4f}")
                wandb.log({"best_val_loss_updated": best_val_loss})

        # Training step
        if run_type in PRETRAIN_TYPES:
            X, y = get_pretrain_batch("train", data_cfg, model_cfg, train_cfg["batch_size"], device_type, device)
        else:
            # Fine-tuning: get the next batch from the DataLoader
            try:
                batch_X, batch_y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch_X, batch_y = next(train_iter)

            batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)


            X = batch_X[:, :-1]
            y = batch_y[:, 1:]

        if step % train_cfg["gradient_accumulation_steps"] == 0:
            t0 = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        with ctx:
            outputs = model(input_ids=X, labels=y)
            total_loss = outputs["loss"]
            loss = total_loss / train_cfg["gradient_accumulation_steps"]
            scaler.scale(loss).backward()

        if ((step + 1) % train_cfg["gradient_accumulation_steps"] == 0) or (step == train_cfg["max_iters"] - 1):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            dt = time.perf_counter() - t0
            effective_tokens = tokens_per_iter * train_cfg["gradient_accumulation_steps"]
            tokens_consumed += effective_tokens
            gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            tflops_per_sec = train_fpt * effective_tokens / dt / 1e12 if train_fpt > 0 else 0.0
            cumulative_tflops = train_fpt * tokens_consumed / 1e12 if train_fpt > 0 else 0.0

            wandb.log({
                "efficiency/step_time_sec": dt,
                "efficiency/tokens_per_sec": effective_tokens / dt,
                "efficiency/tokens_consumed_M": tokens_consumed / 1e6,
                "efficiency/gpu_mem_peak_GB": gpu_mem_gb,
                "efficiency/tflops_per_sec": tflops_per_sec,
                "efficiency/cumulative_tflops": cumulative_tflops,
            })

            if run_type in PRETRAIN_TYPES:
                print(f"step {step} | {effective_tokens/dt:,.0f} tok/s | {tflops_per_sec:.3f} TFLOPs/s | {tokens_consumed/1e6:.2f}M tokens | {gpu_mem_gb:.2f}GB VRAM")
                wandb.log({
                    "tflops_per_sec": tflops_per_sec,
                    "tokens_per_sec": effective_tokens / dt,
                    "tokens_consumed": tokens_consumed,
                    "total_tflops": cumulative_tflops,
                })

            current_step = (step + 1) // train_cfg["gradient_accumulation_steps"]
            total_steps = train_cfg["max_iters"] // train_cfg["gradient_accumulation_steps"]

            if current_step < train_cfg["warmup_steps"]:
                lr = train_cfg["learning_rate"] * (current_step + 1) / train_cfg["warmup_steps"]
            else:
                progress = (current_step - train_cfg["warmup_steps"]) / (total_steps - train_cfg["warmup_steps"])
                progress = max(0.0, min(1.0, progress))   # clamp to [0, 1]
                lr = train_cfg["min_lr"] + (train_cfg["learning_rate"] - train_cfg["min_lr"]) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if step % 10 == 0:
            total_loss_log = outputs["loss"]
            main_loss_log = outputs.get("main_loss", torch.tensor(0.0, device=total_loss_log.device))
            mtp_loss_log = outputs.get("mtp_loss", torch.tensor(0.0, device=total_loss_log.device))
            aux_loss_log = outputs.get("aux_loss", torch.tensor(0.0, device=total_loss_log.device))

            wandb.log({
                "step": step,
                "total_loss": total_loss_log.item(),
                "main_loss": main_loss_log.item(),
                "mtp_loss": mtp_loss_log.item(),
                "aux_loss": aux_loss_log.item(),
                "learning_rate": lr,
                "scaled_loss": loss.item()
            })

    print("Training completed!")
    wandb.finish()
    return model, cfg
