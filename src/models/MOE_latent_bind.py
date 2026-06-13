import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.modules.RMSnorm import RMSNorm
from src.modules.MLA_block import MultiHeadLatentAttention
from src.modules.moe_block import MoELayer
from src.modules.MTP_block import MultiTokenPredictionHead
from src.data.tokenization.protein_character_tokenizer import ProteinTokenizerHF

tokenizer = ProteinTokenizerHF()


class DeepSeekBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = MultiHeadLatentAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MoELayer(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))

        if self.config.MOE_route_out:
            moe_out, top_k_indices, top_k_logits = self.mlp(self.ln_2(x))
            x = x + moe_out
            return x, top_k_indices, top_k_logits
        else:
            x = x + self.mlp(self.ln_2(x))
            return x


class DeepSeekV3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.h = nn.ModuleList([DeepSeekBlock(config) for _ in range(config.n_layer)])

        self.ln_f = RMSNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.wte.weight = self.lm_head.weight

        if config.mtp_num_heads > 0:
            self.mtp_heads = nn.ModuleList([
                MultiTokenPredictionHead(config, depth)
                for depth in range(1, config.mtp_num_heads + 1)
            ])
        else:
            self.mtp_heads = None

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('down_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def flops_per_token(self, seq_len: int | None = None) -> int:
        """
        Analytical forward-pass FLOPs per token (amortized over seq_len for attention).
        Multiply by 6 for total training FLOPs (fwd + 2x bwd).
        Convention: linear (m->n) on T tokens = 2*T*m*n FLOPs.
        """
        cfg = self.config
        T = seq_len if seq_len is not None else cfg.block_size
        C = cfg.n_embd
        H = cfg.n_head
        d = C // H
        kv_r = cfg.kv_lora_rank
        q_r  = cfg.q_lora_rank
        r    = cfg.rope_dim

        mla = (
            2 * C * kv_r          
          + 2 * kv_r * C          
          + 2 * kv_r * C          
          + 2 * C * q_r           
          + 2 * q_r * C           
          + 2 * C * H * r         
          + 2 * q_r * H * r       
          + 2 * H * T * (d + r)   
          + 2 * H * T * d         
          + 2 * C * C             
        )

        E   = cfg.expert_intermediate_size or 0 
        K   = cfg.n_experts_per_token
        n_e = cfg.n_experts

        moe = (
            2 * C * n_e      
          + K * 6 * C * E     
        )
        if cfg.use_shared_expert and cfg.shared_expert_intermediate_size:
            moe += 6 * C * cfg.shared_expert_intermediate_size

        lm_head = 2 * C * cfg.vocab_size

        return cfg.n_layer * (mla + moe) + lm_head

    def forward(self, input_ids, labels=None, **kwargs):

        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size

        tok_emb = self.wte(input_ids)                
        x = self.drop(tok_emb)                       

        if self.config.MOE_route_out:
            top_k_indices_per_layer = []
            top_k_logits_per_layer = []

            for block in self.h:
                x, top_k_indices, top_k_logits = block(x)
                top_k_indices_per_layer.append(top_k_indices)
                top_k_logits_per_layer.append(top_k_logits)

            top_k_indices_per_layer = torch.stack(top_k_indices_per_layer)
            top_k_logits_per_layer = torch.stack(top_k_logits_per_layer)
        else:
            for block in self.h:
                x = block(x)

        x = self.ln_f(x)                            

        main_logits = self.lm_head(x)                

        main_loss = None
        if labels is not None:
            main_loss = F.cross_entropy(
                main_logits.reshape(-1, main_logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100
            )

        mtp_loss = None

        if (
            labels is not None
            and self.mtp_heads is not None
            and self.config.mtp_loss_weight > 0.0
        ):
            mtp_losses = []
            current_hidden = x

            for depth, mtp_head in enumerate(self.mtp_heads, 1):

                if t <= depth:
                    break

                future_indices = input_ids[:, depth:]         
                future_embeds = self.wte(future_indices)      

                if future_embeds.size(1) < current_hidden.size(1):
                    pad_size = current_hidden.size(1) - future_embeds.size(1)
                    padding = torch.zeros(
                        b, pad_size, self.config.n_embd,
                        device=device,
                        dtype=future_embeds.dtype
                    )
                    future_embeds = torch.cat([future_embeds, padding], dim=1)
                else:
                    future_embeds = future_embeds[:, :current_hidden.size(1)]

                current_hidden = mtp_head(current_hidden, future_embeds)
                mtp_logits = self.lm_head(current_hidden)

                if t > depth + 1:
                    shift_logits = mtp_logits[..., :-(depth+1), :]
                    shift_labels = labels[..., depth+1:]

                    if shift_labels.numel() > 0:
                        mtp_loss_single = F.cross_entropy(
                            shift_logits.reshape(-1, shift_logits.size(-1)),
                            shift_labels.reshape(-1),
                            ignore_index=-100
                        )
                        mtp_losses.append(mtp_loss_single)

            if mtp_losses:
                mtp_loss = torch.stack(mtp_losses).mean()

        if labels is not None:

            total_loss = main_loss

            if mtp_loss is not None:
                total_loss = total_loss + self.config.mtp_loss_weight * mtp_loss

            aux_loss = None
            if self.config.aux_loss_weight > 0:
                aux_losses = [block.mlp.last_aux_loss for block in self.h
                              if block.mlp.last_aux_loss is not None]
                if aux_losses:
                    aux_loss = torch.stack(aux_losses).mean()
                    total_loss = total_loss + self.config.aux_loss_weight * aux_loss

            return {
                "loss": total_loss,
                "logits": main_logits,
                "main_loss": main_loss,
                "mtp_loss": mtp_loss if mtp_loss is not None else torch.tensor(0.0, device=device),
                "aux_loss": aux_loss if aux_loss is not None else torch.tensor(0.0, device=device)
            }

        if self.config.MOE_route_out:
            return {
                "logits": main_logits[:, [-1], :],
                "moe_routing": top_k_indices_per_layer,
                "moe_routing_logits": top_k_logits_per_layer
            }
        else:
            return {
                "logits": main_logits[:, [-1], :]
            }
