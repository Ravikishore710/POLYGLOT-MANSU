"""
src/polyglot_mansu/circuits/cad_metrics.py
Mechanistic Erasure Metrics from Lexsi Labs:
- CAD (Circuit Attribution Divergence)
- AS-C (In-Circuit Activation Shift)
- AS-NC (Non-Circuit Drift)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple

def compute_cad(
    scores_base: Dict[str, torch.Tensor],
    scores_unlearned: Dict[str, torch.Tensor],
    eps: float = 1e-8
) -> float:
    """
    Computes Circuit Attribution Divergence (CAD):
    CAD = 1 - CosineSimilarity(flatten(s_base), flatten(s_unlearned))
    CAD in [0, 2]. CAD ~ 0.0 means knowledge circuit was untouched (behavioral camouflage);
    CAD > 0.30 indicates structural circuit dismantling.
    """
    flat_base = torch.cat([t.reshape(-1).to(torch.float32) for t in scores_base.values()])
    flat_unlearn = torch.cat([scores_unlearned[k].reshape(-1).to(torch.float32) for k in scores_base.keys()])

    cos_sim = F.cosine_similarity(flat_base.unsqueeze(0), flat_unlearn.unsqueeze(0), eps=eps).item()
    cad = max(0.0, 1.0 - cos_sim)
    return cad

def compute_activation_shifts(
    activations_base: Dict[str, torch.Tensor],
    activations_unlearned: Dict[str, torch.Tensor],
    circuit_mask: Dict[str, torch.Tensor],
    eps: float = 1e-6
) -> Tuple[float, float]:
    """
    Computes (AS-C, AS-NC):
    - AS-C: Normalized activation shift on parameters/nodes inside C* (should be HIGH)
    - AS-NC: Normalized activation shift on parameters/nodes outside C* (should be LOW)
    """
    shift_c_list = []
    shift_nc_list = []

    for name, act_base in activations_base.items():
        act_unlearn = activations_unlearned[name]
        mask = circuit_mask.get(name)
        if mask is None:
            mask = torch.zeros_like(act_base)

        diff_norm = torch.norm(act_unlearn - act_base, dim=-1)
        base_norm = torch.norm(act_base, dim=-1) + eps
        rel_shift = diff_norm / base_norm

        mask_bool = (mask > 0)
        if mask_bool.any():
            shift_c_list.append(rel_shift[mask_bool].mean().item())
        if (~mask_bool).any():
            shift_nc_list.append(rel_shift[~mask_bool].mean().item())

    as_c = sum(shift_c_list) / max(1, len(shift_c_list)) if shift_c_list else 0.0
    as_nc = sum(shift_nc_list) / max(1, len(shift_nc_list)) if shift_nc_list else 0.0

    return as_c, as_nc
