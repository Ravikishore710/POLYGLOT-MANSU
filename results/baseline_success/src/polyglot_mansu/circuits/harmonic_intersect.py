"""
src/polyglot_mansu/circuits/harmonic_intersect.py
Cross-Lingual Harmonic Circuit Attribution & Mask Generation:
Computes H_v(theta_0) across language subsets to reject monolingual surface heads
and isolate language-agnostic conceptual knowledge circuits (C*).
"""

import torch
from typing import Dict, List, Tuple

def compute_harmonic_score(
    attribution_scores_per_lang: Dict[str, Dict[str, torch.Tensor]],
    eps: float = 1e-6
) -> Dict[str, torch.Tensor]:
    """
    Computes H_v(theta_0) = (1 / |L| sum_{l in L} 1 / (s_tilde_v^(l) + eps))^(-1)
    across language attributions.
    """
    langs = list(attribution_scores_per_lang.keys())
    if not langs:
        raise ValueError("Must provide at least one language for harmonic attribution")

    num_langs = len(langs)
    # Find all node keys present across languages
    sample_lang = langs[0]
    node_names = list(attribution_scores_per_lang[sample_lang].keys())

    # Step 1: Normalize scores per language to [0, 1]
    norm_scores = {l: {} for l in langs}
    for l in langs:
        lang_dict = attribution_scores_per_lang[l]
        # Find global max across all nodes in this language
        max_val = max(torch.max(torch.abs(t)).item() for t in lang_dict.values()) + eps
        for name in node_names:
            norm_scores[l][name] = torch.abs(lang_dict[name]) / max_val

    # Step 2: Compute harmonic mean across languages
    harmonic_scores = {}
    for name in node_names:
        inv_sum = torch.zeros_like(attribution_scores_per_lang[sample_lang][name], dtype=torch.float32)
        for l in langs:
            s_tilde = norm_scores[l][name].to(torch.float32)
            inv_sum += 1.0 / (s_tilde + eps)
        
        harmonic_scores[name] = num_langs / (inv_sum + eps)

    return harmonic_scores

def extract_circuit_mask(
    harmonic_scores: Dict[str, torch.Tensor],
    sparsity_ratio: float = 0.10
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Extracts binary parameter mask M_{C*} for the top-k% highest harmonic attribution nodes.
    sparsity_ratio: e.g. 0.05, 0.10, 0.15, 0.25 (default 10%)
    """
    # Flatten all scores to find top-k threshold
    all_scores = torch.cat([t.reshape(-1) for t in harmonic_scores.values()])
    total_elements = all_scores.numel()
    k = max(1, int(sparsity_ratio * total_elements))
    
    threshold = torch.kthvalue(all_scores, total_elements - k + 1).values.item()
    
    masks = {}
    for name, t in harmonic_scores.items():
        masks[name] = (t >= threshold).float()

    return masks, threshold
