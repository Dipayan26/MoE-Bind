
import torch

from src.data.tokenization.protein_character_tokenizer import ProteinTokenizerHF

tokenizer = ProteinTokenizerHF()


@torch.no_grad()
def generate(model, input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
             repetition_penalty=1.0, MOE_route=False, valid_token_ids=None, stop_token_ids=None):
    model.eval()
    _valid_ids = valid_token_ids if valid_token_ids is not None else tokenizer.get_valid_aa_token_ids()
    _stop_ids = set(stop_token_ids.tolist()) if stop_token_ids is not None else set()

    _allow_ids = torch.cat([
        _valid_ids,
        stop_token_ids if stop_token_ids is not None else torch.tensor([], dtype=torch.long),
    ]).long()
    _logit_mask = None 

    for _ in range(max_new_tokens):
        input_ids_cond = (
            input_ids
            if input_ids.size(1) <= model.config.block_size
            else input_ids[:, -model.config.block_size:]
        )

        outputs = model(input_ids_cond)

        logits = outputs["logits"][:, -1, :] / temperature


        if _logit_mask is None:
            _logit_mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
            _logit_mask[_allow_ids.to(logits.device)] = False
        logits = logits.masked_fill(_logit_mask.unsqueeze(0), -float("inf"))


        if repetition_penalty != 1.0:
            for token_id in set(input_ids[0].tolist()):
                if token_id < logits.shape[-1]:
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty


        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_indices_to_remove] = -float("inf")
            logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)
        elif top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        # Early stop 
        if _stop_ids and next_token.item() in _stop_ids:
            break

        input_ids = torch.cat([input_ids, next_token], dim=1)

    if MOE_route:
        return input_ids, outputs.get("moe_routing", None), outputs.get("moe_routing_logits", None)
    return input_ids
