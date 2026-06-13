import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, f"embedding dim should be divisible by number of heads"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1.0

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)  
        q, k, v = qkv.split(self.n_embd, dim=-1)   
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)    
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)    
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)   
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)   
        out = out.transpose(1, 2).contiguous().view(B, T, C)   
        out = self.c_proj(out)  
        return out
