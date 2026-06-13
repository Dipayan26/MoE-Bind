
import os
import gzip
import hashlib
import random
import numpy as np
from tqdm import tqdm
from typing import Iterator

from src.data.tokenization.protein_character_tokenizer import ProteinTokenizerHF
from src.data.tokenization.data_tokenize_config import TokenizerConfig
from src.utils.config import load_yaml


# ── Sequence Streamers ──────────────────────────────────────────


def stream_fasta(path: str) -> Iterator[str]:
    """Yield sequences from FASTA or gzipped FASTA."""
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    with opener(path, mode) as f:
        seq_parts = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_parts:
                    yield "".join(seq_parts)
                    seq_parts = []
            else:
                seq_parts.append(line)
        if seq_parts:
            yield "".join(seq_parts)


def stream_tsv(path: str) -> Iterator[str]:
    """Yield sequences from TSV (sequence in column 2)."""
    with open(path, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1].strip():
                yield parts[1]


def stream_multi(paths: list[str]) -> Iterator[str]:
    """Stream from multiple files sequentially."""
    for path in paths:
        if path.endswith(".tsv"):
            yield from stream_tsv(path)
        else:
            yield from stream_fasta(path)


# ── Shuffle Buffer ──────────────────────────────────────────────


class ShuffleBuffer:


    def __init__(self, buffer_size: int, seed: int = 42):
        self.buffer_size = buffer_size
        self.buffer = []
        self.rng = random.Random(seed)

    def stream(self, iterator: Iterator[str]) -> Iterator[str]:
        for item in iterator:
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(item)
            else:
                idx = self.rng.randint(0, self.buffer_size - 1)
                yield self.buffer[idx]
                self.buffer[idx] = item

        # drain remaining buffer in random order
        self.rng.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()


# ── Train/Val Split ─────────────────────────────────────────────


def is_train(seq: str, train_ratio: float) -> bool:
    """Deterministic hash-based split."""
    h = int(hashlib.md5(seq.encode()).hexdigest(), 16)
    return (h % 10000) < int(train_ratio * 10000)


# ── Streaming Memmap Writer ─────────────────────────────────────


class MemmapWriter:

    def __init__(self, path: str, dtype=np.uint16, budget: int = None,
                 flush_every: int = 5_000_000):
        self.path = path
        self.dtype = dtype
        self.budget = budget
        self.flush_every = flush_every
        self.buffer = []
        self.total_written = 0
        self._file_created = False

    @property
    def is_full(self) -> bool:
        if self.budget is None:
            return False
        return (self.total_written + len(self.buffer)) >= self.budget

    def add(self, tokens: list[int]):
        """Add tokens to buffer. Respects budget, flushes when buffer is large."""
        if self.is_full:
            return

        if self.budget is not None:
            space = self.budget - self.total_written - len(self.buffer)
            if space <= 0:
                return
            tokens = tokens[:space]

        self.buffer.extend(tokens)

        if len(self.buffer) >= self.flush_every:
            self._flush()

    def _flush(self):
        if not self.buffer:
            return

        chunk = np.array(self.buffer, dtype=self.dtype)
        n = len(chunk)

        if self.budget is not None:

            if not self._file_created:
                mm = np.memmap(self.path, dtype=self.dtype, mode="w+", shape=(self.budget,))
                mm[:n] = chunk
                mm.flush()
                self._file_created = True
            else:
                old_size = self.total_written
                mm = np.memmap(self.path, dtype=self.dtype, mode="r+", shape=(self.budget,))
                mm[old_size:old_size + n] = chunk
                mm.flush()
        else:
            file_mode = "ab" if self._file_created else "wb"
            with open(self.path, file_mode) as f:
                chunk.tofile(f)
            self._file_created = True

        self.total_written += n
        self.buffer.clear()

    def finalize(self) -> int:
        """Flush remaining buffer and return total tokens written."""
        self._flush()
        return self.total_written


# ── Core Tokenization ──────────────────────────────────────────


def get_stream(cfg: dict, seed: int = 42) -> Iterator[str]:
    """Build sequence stream from config, with optional shuffle."""
    run_type = cfg["run"]["type"]
    data_cfg = cfg["data"]

    # raw stream
    if "paths" in data_cfg:
        raw = stream_multi(data_cfg["paths"])
    elif run_type == "tsv_tokenize_protein":
        raw = stream_tsv(data_cfg["tsv_path"])
    else:
        path = data_cfg.get("fasta_path") or data_cfg.get("zip_path")
        raw = stream_fasta(path)

    # optional shuffle
    shuffle_size = data_cfg.get("shuffle_buffer_size", 0)
    if shuffle_size > 0:
        print(f"Shuffle buffer: {shuffle_size:,} sequences")
        return ShuffleBuffer(buffer_size=shuffle_size, seed=seed).stream(raw)

    return raw


def _process_batch(tokenizer, batch_seqs, train_writer, val_writer, train_ratio):
    """Tokenize a batch and route each sequence to train or val."""
    encoded = tokenizer(
        batch_seqs, padding=False, truncation=False
    )["input_ids"]

    for seq, ids in zip(batch_seqs, encoded):
        if is_train(seq, train_ratio):
            train_writer.add(ids)
        else:
            val_writer.add(ids)


def tokenize_dataset(
    tokenizer,
    seq_stream: Iterator[str],
    train_writer: MemmapWriter,
    val_writer: MemmapWriter,
    train_ratio: float,
    batch_size: int = 512,
    log_every: int = 100_000,
) -> tuple[int, int]:

    batch_seqs = []
    seen = 0
    skipped = 0

    for seq in tqdm(seq_stream, desc="Tokenizing", unit=" seqs"):
        if len(seq) < 2:
            skipped += 1
            continue

        batch_seqs.append(seq.upper())
        seen += 1

        if seen % log_every == 0:
            tqdm.write(
                f"  Seen {seen:>12,} | "
                f"Train {train_writer.total_written:>14,} | "
                f"Val {val_writer.total_written:>12,} | "
                f"Skipped {skipped:,}"
            )

        if len(batch_seqs) < batch_size:
            continue

        _process_batch(tokenizer, batch_seqs, train_writer, val_writer, train_ratio)
        batch_seqs = []

        if train_writer.is_full and val_writer.is_full:
            break

    # flush leftover batch
    if batch_seqs:
        _process_batch(tokenizer, batch_seqs, train_writer, val_writer, train_ratio)

    return seen, skipped


# ── Entry Point ─────────────────────────────────────────────────


def tokenize_data(config_path: str):
    cfg = load_yaml(config_path)
    tok_cfg = TokenizerConfig(**cfg["tokenizer"])
    out_cfg = cfg["output"]

    dtype = getattr(np, tok_cfg.dtype) if isinstance(tok_cfg.dtype, str) else tok_cfg.dtype
    seed = tok_cfg.random_seed

    # ── tokenizer ──
    tokenizer = ProteinTokenizerHF()
    print(f"Vocab size: {tokenizer.vocab_size}")

    # ── budgets ──
    val_budget = int((1 - tok_cfg.train_ratio) * tok_cfg.total_tokens)
    train_budget = tok_cfg.total_tokens - val_budget
    print(f"Train budget: {train_budget:,} tokens")
    print(f"Val budget:   {val_budget:,} tokens")


    flush_every = tok_cfg.write_block

    os.makedirs(os.path.dirname(out_cfg["train_bin"]) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_cfg["val_bin"]) or ".", exist_ok=True)

    train_writer = MemmapWriter(
        out_cfg["train_bin"], dtype=dtype,
        budget=train_budget, flush_every=flush_every,
    )
    val_writer = MemmapWriter(
        out_cfg["val_bin"], dtype=dtype,
        budget=val_budget, flush_every=flush_every,
    )

    # ── run ──
    seq_stream = get_stream(cfg, seed=seed)

    seen, skipped = tokenize_dataset(
        tokenizer=tokenizer,
        seq_stream=seq_stream,
        train_writer=train_writer,
        val_writer=val_writer,
        train_ratio=tok_cfg.train_ratio,
        batch_size=tok_cfg.batch_size,
    )

    # ── finalize ──
    train_total = train_writer.finalize()
    val_total = val_writer.finalize()

    print(f"\n{'='*50}")
    print(f"DONE")
    print(f"Sequences seen:    {seen:,}")
    print(f"Sequences skipped: {skipped:,}")
    print(f"Train tokens:      {train_total:,} / {train_budget:,}")
    print(f"Val tokens:        {val_total:,} / {val_budget:,}")
    print(f"Train file:        {out_cfg['train_bin']}")
    print(f"Val file:          {out_cfg['val_bin']}")
    print(f"{'='*50}")