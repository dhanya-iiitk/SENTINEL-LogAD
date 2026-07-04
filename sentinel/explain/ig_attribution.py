import torch
import torch.nn.functional as F
import numpy as np


def integrated_gradients(model, seq, seq_len, device, n_steps=50):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    seq       = seq.unsqueeze(0).to(device)
    seq_len_t = seq_len.unsqueeze(0).to(device)
    embed_weight    = model.base.embedding.weight.detach()
    input_embeds    = embed_weight[seq[0]].detach()
    baseline_embeds = torch.zeros_like(input_embeds)
    T   = input_embeds.shape[0]
    pe  = model.base.pe[:, :T].detach().clone()
    causal_mask = torch.triu(
        torch.ones(T, T, device=device), diagonal=1).bool()
    pad_mask = torch.zeros(1, T, dtype=torch.bool, device=device)
    actual = seq_len_t[0].item()
    if actual < T:
        pad_mask[0, actual:] = True
    target = seq[:, 1:]
    accumulated_grads = torch.zeros_like(input_embeds)

    def forward_from_embeds(embeds):
        x = embeds.unsqueeze(0) + pe
        for layer in model.base.transformer.layers:
            x = layer(x, src_mask=causal_mask,
                       src_key_padding_mask=pad_mask)
        h      = model.base.norm(x)
        logits = model.task_head(h[:, :-1, :])
        B, T1, V = logits.shape
        per_token = F.cross_entropy(
            logits.reshape(-1, V), target.reshape(-1),
            ignore_index=0, reduction='none'
        ).reshape(B, T1)
        mask  = (target != 0).float()
        score = (per_token * mask).sum(dim=1) / \
                (mask.sum(dim=1) + 1e-9)
        return score.sum()

    for step in range(1, n_steps + 1):
        alpha  = step / n_steps
        interp = (baseline_embeds +
                  alpha * (input_embeds - baseline_embeds)).clone()
        interp.requires_grad_(True)
        score  = forward_from_embeds(interp)
        grad,  = torch.autograd.grad(score, interp)
        accumulated_grads += grad.detach()

    for p in model.parameters():
        p.requires_grad_(True)
    ig           = (input_embeds - baseline_embeds) * \
                   (accumulated_grads / n_steps)
    attributions = ig.norm(dim=-1).cpu().numpy()
    return attributions


def get_top_events(event_ids, attributions, id2template, top_k=5):
    valid_len = len(event_ids)
    idx       = np.argsort(attributions[:valid_len])[::-1][:top_k]
    results   = []
    for rank, i in enumerate(idx, 1):
        eid = event_ids[i]
        results.append({
            'rank':        rank,
            'position':    int(i),
            'event_id':    int(eid),
            'template':    id2template.get(eid, f'Event_{eid}'),
            'attribution': float(attributions[i]),
        })
    return results
