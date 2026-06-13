# src/models/config.py
from dataclasses import dataclass

@dataclass
class DeepSeekConfig:
    # Model architecture
    vocab_size: int
    block_size: int
    n_layer: int
    n_embd: int
    n_head: int

    # MLA / LoRA
    kv_lora_rank: int = 0
    q_lora_rank: int = 0
    rope_dim: int = 0
    bias: bool = True

    # MoE
    n_experts: int = 0
    n_experts_per_token: int = 0
    expert_intermediate_size: int | None = None
    shared_expert_intermediate_size: int | None = None
    use_shared_expert: bool = False
    MOE_route_out : bool = False # if True, return the routing decisions (top-k indices) from the MoE layer for analysis. If False, only return the combined output.

    # MTP
    mtp_num_heads: int = 0

    # Regularization
    dropout: float = 0.0

    # Loss weights
    aux_loss_weight: float = 0.0
    mtp_loss_weight: float = 0.0

    def __post_init__(self):
        if self.n_experts > 0:
            assert self.expert_intermediate_size is not None, (
                "expert_intermediate_size must be set when using MoE"
            )
            


@dataclass
class GPT2Config:
    vocab_size: int
    block_size: int
    n_layer: int
    n_embd: int
    n_head: int
    dropout: float = 0.1
    bias: bool = True

    


from typing import Optional


@dataclass
class LLaMA2Config:
    # Model architecture (common with DeepSeek / GPT2)
    vocab_size: int
    block_size: int
    n_layer: int
    n_embd: int
    n_head: int

    # LLaMA-specific: Grouped-Query Attention
    n_kv_heads: Optional[int] = None  # None = same as n_head (MHA)

    # LLaMA-specific: SwiGLU FFN sizing
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None

    # Normalization
    norm_eps: float = 1e-5

    # Regularization
    dropout: float = 0.0
 

    def __post_init__(self):
        if self.n_kv_heads is not None:
            assert self.n_head % self.n_kv_heads == 0, (
                f"n_head ({self.n_head}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
