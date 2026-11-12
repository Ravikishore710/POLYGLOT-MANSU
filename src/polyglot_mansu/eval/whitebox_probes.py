"""
src/polyglot_mansu/eval/whitebox_probes.py
White-Box Probing Suite:
- Logit-Lens across all 36 layers of Tiny Aya
- Layerwise Linear Probes to prove knowledge is structurally gone from residual stream
"""

from typing import Dict, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

def run_logit_lens(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_token_id: int
) -> List[Dict[str, Any]]:
    """
    Decodes residual stream at every transformer layer using model's lm_head (unembedding matrix).
    Returns target token probability and rank across all layers.
    """
    model.eval()
    results = []

    # Capture hidden states across all layers
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of (layer_0, ..., layer_L)
        lm_head = getattr(model, "lm_head", None)

        if lm_head is None:
            # Fallback if tied embeddings or module layout differs
            lm_head = getattr(model.model, "embed_tokens", None)

        for layer_idx, h in enumerate(hidden_states):
            last_token_h = h[:, -1, :]  # Shape: (batch, hidden_dim)
            if lm_head is not None:
                logits = lm_head(last_token_h)
            else:
                logits = last_token_h

            probs = F.softmax(logits, dim=-1)
            target_prob = probs[0, target_token_id].item() if target_token_id < probs.shape[-1] else 0.0

            # Compute rank of target token
            sorted_indices = torch.argsort(logits[0], descending=True)
            target_rank = int((sorted_indices == target_token_id).nonzero(as_tuple=True)[0].item()) + 1 if target_token_id < probs.shape[-1] else probs.shape[-1]

            results.append({
                "layer": layer_idx,
                "target_prob": target_prob,
                "target_rank": target_rank
            })

    return results

def train_linear_probe(
    hidden_dim: int,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_steps: int = 100,
    lr: float = 0.01
) -> float:
    """
    Trains a linear classifier on residual stream representations.
    Returns test accuracy (near random chance ~25% for 4-class probes proves erasure).
    """
    probe = nn.Linear(hidden_dim, 4)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(num_steps):
        optimizer.zero_grad()
        out = probe(train_features)
        loss = loss_fn(out, train_labels)
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        preds = torch.argmax(probe(test_features), dim=-1)
        acc = (preds == test_labels).float().mean().item()

    return acc
