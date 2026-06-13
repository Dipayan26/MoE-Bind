import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.self_attention import CausalSelfAttention


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')    # approximate='tanh' reproduces the GPT-2 paper
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1.0

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """ Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            h = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
            ln_f = nn.LayerNorm(self.config.n_embd)
        ))
        # language modeling head
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        # weight tying between input embeddings and output head
        self.transformer.wte.weight = self.lm_head.weight
        # init params (iterates over all submodules and applies _init_weights)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                std /= (2 * self.config.n_layer)**0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)    # per the OpenAI GPT-2 source
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def flops_per_token(self, seq_len: int | None = None) -> int:
        """
        Analytical forward-pass FLOPs per token (amortized over seq_len for attention).
        Multiply by 6 for total training FLOPs (fwd + 2x bwd).
        Convention: linear (m->n) on T tokens = 2*T*m*n FLOPs.
        """
        cfg = self.config
        T = seq_len if seq_len is not None else cfg.block_size
        C = cfg.n_embd

        # Attention: c_attn (C->3C fused Q,K,V), QK+AV (amortized), c_proj (C->C)
        attn = (
            2 * C * 3 * C   # c_attn: fused Q,K,V projection
          + 2 * C * T       # QK scores  (amortized)
          + 2 * C * T       # AV context (amortized)
          + 2 * C * C       # c_proj
        )

        # MLP: c_fc (C->4C), c_proj (4C->C)
        mlp = (
            2 * C * 4 * C   # c_fc
          + 2 * 4 * C * C   # c_proj
        )

        lm_head = 2 * C * cfg.vocab_size

        return cfg.n_layer * (attn + mlp) + lm_head

    def forward(self, input_ids, labels=None, **kwargs):
        B, T = input_ids.shape
        assert T <= self.config.block_size, f'sequence length {T} should be <= {self.config.block_size}'
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)    # (T,)
        pos_embd = self.transformer.wpe(pos)    # (T, n_embd)
        tok_embd = self.transformer.wte(input_ids)    # (B, T, n_embd)
        x = pos_embd + tok_embd    # (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)    # (B, T, n_embd)
        logits = self.lm_head(x)    # (B, T, vocab_size)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return {
            "logits": logits,
            "loss": loss,
        }
