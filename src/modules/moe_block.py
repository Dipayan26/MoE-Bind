import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.SwiGLU import SwiGLU


class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts            # total number of experts
        self.top_k = config.n_experts_per_token      # experts each token routes to
        self.n_embd = config.n_embd                  # hidden size

        # Router
        self.router = nn.Linear(config.n_embd, config.n_experts, bias=False)

        # Expert MLPs
        self.experts = nn.ModuleList([
            SwiGLU(
                config.n_embd,
                config.expert_intermediate_size,
                config.n_embd,
                config.bias
            ) for _ in range(config.n_experts)
        ])

        if config.use_shared_expert:
            self.shared_expert = SwiGLU(
                config.n_embd,
                config.shared_expert_intermediate_size,
                config.n_embd,
                config.bias
            )
            self.shared_expert_gate = nn.Linear(config.n_embd, 1, bias=False)
        else:
            self.shared_expert = None

        self.register_buffer('expert_bias', torch.zeros(config.n_experts))
        self.bias_update_rate = 0.001
        self.last_aux_loss = None

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)

        raw_router_logits = self.router(x_flat)
        router_logits = raw_router_logits + self.expert_bias

        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = torch.zeros_like(router_logits)
        routing_weights.scatter_(-1, top_k_indices, F.softmax(top_k_logits, dim=-1))
        top_k_logits_out = F.softmax(top_k_logits, dim=-1)

        output = torch.zeros_like(x_flat)
        expert_usage = torch.zeros(self.n_experts, device=x.device)

        for expert_idx in range(self.n_experts):
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            expert_usage[expert_idx] = expert_mask.sum().float()

            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = self.experts[expert_idx](expert_input)

                weights = routing_weights[expert_mask, expert_idx].unsqueeze(-1)
                output[expert_mask] += expert_output * weights

        if self.shared_expert is not None:
            shared_output = self.shared_expert(x_flat)
            gate = torch.sigmoid(self.shared_expert_gate(x_flat))  
            output += gate * shared_output

        if self.training:
            total_tokens = x_flat.size(0)
            f = expert_usage / total_tokens                         
            P = F.softmax(raw_router_logits, dim=-1).mean(dim=0)    
            self.last_aux_loss = self.n_experts * (f * P).sum()

            with torch.no_grad():
                avg_usage = expert_usage.mean()
                for i in range(self.n_experts):
                    if expert_usage[i] > avg_usage:
                        self.expert_bias[i] -= self.bias_update_rate
                    else:
                        self.expert_bias[i] += self.bias_update_rate

        if self.config.MOE_route_out:
            return (
                output.view(batch_size, seq_len, hidden_size),
                top_k_indices.view(batch_size, seq_len, self.top_k),
                top_k_logits_out.view(batch_size, seq_len, self.top_k)
            )
        else:
            return output.view(batch_size, seq_len, hidden_size)
