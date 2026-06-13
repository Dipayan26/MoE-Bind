import torch
import torch.nn as nn
from src.modules.RMSnorm import RMSNorm
from src.modules.MLA_block import MultiHeadLatentAttention
from src.modules.moe_block import MoELayer


class MultiTokenPredictionHead(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        self.depth = depth
        self.n_embd = config.n_embd

        self.combine_proj = nn.Linear(2 * config.n_embd, config.n_embd, bias=config.bias)

        self.norm1 = RMSNorm(config.n_embd)
        self.norm2 = RMSNorm(config.n_embd)

        self.attn = MultiHeadLatentAttention(config)
        self.mlp = MoELayer(config)
        self.attn_norm = RMSNorm(config.n_embd)
        self.mlp_norm = RMSNorm(config.n_embd)

    def forward(self, prev_hidden, future_token_embed):
        prev_norm = self.norm1(prev_hidden)
        future_norm = self.norm2(future_token_embed)

        combined = torch.cat([prev_norm, future_norm], dim=-1)
        hidden = self.combine_proj(combined)

        hidden = hidden + self.attn(self.attn_norm(hidden))
        hidden = hidden + self.mlp(self.mlp_norm(hidden))

        return hidden

