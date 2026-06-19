# ============================================================
# sentinel/explain/llm_explain.py
# Integrated Gradient attribution + Groq LLM explanation
# ============================================================

import torch
import numpy as np
from typing import List, Dict, Tuple
from groq import Groq


# ─────────────────────────────────────────────
# 1.  Integrated Gradients Attribution
# ─────────────────────────────────────────────

def integrated_gradients_log(
    model,
    seq:      torch.Tensor,   # [1, T]
    seq_len:  torch.Tensor,   # [1]
    graph,
    device:   str = 'cuda',
    n_steps:  int = 50,
) -> np.ndarray:
    """
    Compute Integrated Gradients attribution for each
    log event token in the sequence.

    Attribution measures how much each event contributed
    to the anomaly score.

    Returns: attribution scores [T] numpy array
    """
    model.eval()
    model.to(device)
    seq     = seq.to(device)
    seq_len = seq_len.to(device)
    graph   = graph.to(device)

    # Get embedding weights
    embed_weight = model.embedding.embedding.weight   # [V, D]

    # Baseline: all PAD tokens (zero embedding)
    baseline_ids = torch.zeros_like(seq)   # [1, T]

    # Interpolate between baseline and input in embedding space
    input_embeds    = embed_weight[seq[0]]        # [T, D]
    baseline_embeds = embed_weight[baseline_ids[0]]  # [T, D]

    accumulated_grads = torch.zeros_like(input_embeds)   # [T, D]

    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        interp_embeds = (baseline_embeds +
                         alpha * (input_embeds - baseline_embeds)
                         ).unsqueeze(0).requires_grad_(True)   # [1, T, D]

        # Forward through transformer with interpolated embeddings
        # (bypass embedding lookup — inject directly)
        B, T, D = interp_embeds.shape

        # Positional encoding
        pos_enc_out = model.transformer.pos_enc(interp_embeds)

        # CLS token
        cls = model.transformer.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, pos_enc_out], dim=1)   # [1, T+1, D]

        # Padding mask
        pad_mask = torch.zeros(B, T + 1, dtype=torch.bool, device=device)
        actual = seq_len[0].item()
        if actual < T:
            pad_mask[0, actual + 1:] = True

        x = model.transformer.transformer(x, src_key_padding_mask=pad_mask)
        x = model.transformer.norm(x)
        z_t = x[:, 0, :]   # [1, D] — CLS

        # GNN path (fixed — not interpolated)
        with torch.no_grad():
            z_g = model.gnn(graph)

        # Fusion + flow score
        z     = model.fusion(z_t, z_g)
        score = model.flow.anomaly_score(z)   # [1]
        score.backward()

        accumulated_grads += interp_embeds.grad[0].detach()   # [T, D]
        interp_embeds.grad.zero_()

    # IG = (input - baseline) * avg_gradient
    ig_embeds = (input_embeds - baseline_embeds) * (accumulated_grads / n_steps)
    # Aggregate over embedding dimension → scalar per token
    attributions = ig_embeds.norm(dim=-1).cpu().numpy()   # [T]

    return attributions


# ─────────────────────────────────────────────
# 2.  Extract Top-K Events
# ─────────────────────────────────────────────

def get_top_events(
    event_ids:    List[int],
    attributions: np.ndarray,
    id2template:  Dict[int, str],
    top_k:        int = 5,
) -> List[Dict]:
    """
    Return top-k most attributed log events.

    Returns list of dicts:
      {'rank': 1, 'event_id': 23, 'template': '...', 'attribution': 0.82}
    """
    k = min(top_k, len(event_ids))
    top_indices = np.argsort(attributions[:len(event_ids)])[::-1][:k]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        eid = event_ids[idx]
        results.append({
            'rank':        rank,
            'event_id':    eid,
            'template':    id2template.get(eid, f'Event_{eid}'),
            'attribution': float(attributions[idx]),
        })
    return results


# ─────────────────────────────────────────────
# 3.  Groq LLM Explanation
# ─────────────────────────────────────────────

def build_explanation_prompt(
    top_events:    List[Dict],
    anomaly_score: float,
    threshold:     float,
    dataset_name:  str = 'HDFS',
) -> str:
    events_str = '\n'.join([
        f"  {e['rank']}. [{e['template']}]  "
        f"(attribution weight: {e['attribution']:.3f})"
        for e in top_events
    ])

    prompt = f"""You are an expert cybersecurity analyst specializing in system log analysis.

A log session from the {dataset_name} system was flagged as ANOMALOUS by the SENTINEL framework.

Anomaly Score : {anomaly_score:.4f}
Decision Threshold : {threshold:.4f}
Score exceeds threshold by : {anomaly_score - threshold:.4f}

Top contributing log events (ranked by attribution score):
{events_str}

Based on these log events and their attribution weights:
1. Explain in 2-3 sentences WHY this session is anomalous.
2. Identify the most likely cyber threat or system fault this pattern indicates.
3. Suggest one immediate remediation step.

Be concise and technical. Use cybersecurity terminology."""

    return prompt


def generate_explanation(
    top_events:    List[Dict],
    anomaly_score: float,
    threshold:     float,
    groq_api_key:  str,
    dataset_name:  str = 'HDFS',
    model_name:    str = 'llama-3.1-70b-versatile',
) -> str:
    """
    Generate LLM-powered explanation via Groq API (free tier).
    """
    client = Groq(api_key=groq_api_key)
    prompt = build_explanation_prompt(top_events, anomaly_score,
                                       threshold, dataset_name)

    response = client.chat.completions.create(
        model    = model_name,
        messages = [{"role": "user", "content": prompt}],
        temperature = 0.2,
        max_tokens  = 300,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# 4.  Full Explanation Pipeline
# ─────────────────────────────────────────────

def explain_anomaly(
    model,
    session_events: List[int],
    anomaly_score:  float,
    threshold:      float,
    id2template:    Dict[int, str],
    groq_api_key:   str,
    dataset_name:   str   = 'HDFS',
    top_k:          int   = 5,
    ig_steps:       int   = 50,
    device:         str   = 'cuda',
    max_seq_len:    int   = 50,
    pad_idx:        int   = 0,
) -> Dict:
    """
    Full explanation pipeline for one anomalous session.

    Returns dict with:
      top_events   : list of top-k attributed events
      attributions : full attribution array
      explanation  : LLM-generated text
    """
    from torch_geometric.data import Batch
    from sentinel.data.dataset import build_session_graph

    # Prepare inputs
    eids    = session_events[:max_seq_len]
    seq_len = len(eids)
    padded  = eids + [pad_idx] * (max_seq_len - seq_len)
    seq     = torch.tensor(padded, dtype=torch.long).unsqueeze(0)
    seq_len_t = torch.tensor([seq_len], dtype=torch.long)
    graph   = Batch.from_data_list([build_session_graph(eids)]).to(device)

    # Integrated Gradients
    attributions = integrated_gradients_log(
        model, seq, seq_len_t, graph, device, ig_steps
    )

    # Top events
    top_events = get_top_events(eids, attributions, id2template, top_k)

    # LLM Explanation
    explanation = generate_explanation(
        top_events, anomaly_score, threshold,
        groq_api_key, dataset_name,
    )

    return {
        'session_events': session_events,
        'anomaly_score':  anomaly_score,
        'threshold':      threshold,
        'top_events':     top_events,
        'attributions':   attributions.tolist(),
        'explanation':    explanation,
    }


def print_explanation(result: Dict):
    print(f"\n{'='*60}")
    print(f"  SENTINEL Anomaly Explanation")
    print(f"{'='*60}")
    print(f"  Anomaly Score : {result['anomaly_score']:.4f}  "
          f"(threshold={result['threshold']:.4f})")
    print(f"\n  Top Contributing Log Events:")
    for e in result['top_events']:
        print(f"    {e['rank']}. [{e['template']}]  "
              f"(attr={e['attribution']:.3f})")
    print(f"\n  LLM Explanation:")
    print(f"  {result['explanation']}")
    print(f"{'='*60}")
