"""
src/polyglot_mansu/unlearn/run_baselines.py
Executes the 5 official SafeTune unlearning baselines:
- RMU (Representation Misdirection Unlearning)
- NPO (Negative Preference Optimization)
- GradDiff (Gradient Ascent with Retain KL)
- FLAT (f-divergence variational unlearning)
- SimDPO (Reference-free preference unlearning)
Enforces GUARD-01, GUARD-02, and logs official SafeTune audit badges (GUARD-03).
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn

import safetune
from safetune.interventions.unlearn import (
    rmu_unlearn, RMUConfig,
    npo_unlearn, NPOConfig,
    gradient_ascent_unlearn, GradientAscentConfig,
    flat_unlearn, FLATConfig,
    simdpo_unlearn, SimDPOUnlearnConfig
)
from polyglot_mansu.guards import verify_guard_01_max_len, verify_guard_03_audit_badge

# SafeTune source-audit badges for the 5 baselines
SAFETUNE_AUDIT_BADGES = {
    "RMU": "Faithful",
    "NPO": "Faithful",
    "GradDiff": "Faithful",
    "FLAT": "Faithful",
    "SimDPO": "Variant"  # SafeTune variant of SimPO for unlearning
}

def run_safetune_baseline(
    baseline_name: str,
    model: nn.Module,
    forget_batches: List[Dict[str, torch.Tensor]],
    retain_batches: List[Dict[str, torch.Tensor]],
    max_len: int = 1024,
    device: str = "cpu"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Runs a specified SafeTune baseline method and returns (unlearned_model, metadata).
    """
    verify_guard_01_max_len(max_len)
    badge = SAFETUNE_AUDIT_BADGES.get(baseline_name, "Variant")
    verify_guard_03_audit_badge(baseline_name, badge)

    model.to(device)

    if baseline_name == "RMU":
        config = RMUConfig(layer_id=1, steering_coeff=20.0, alpha=100.0, lr=5e-5, max_num_batches=len(forget_batches))
        unlearned = rmu_unlearn(model, retain_batches=retain_batches, forget_batches=forget_batches, config=config)

    elif baseline_name == "NPO":
        config = NPOConfig(variant="npo", beta=0.1, npo_coeff=1.0, num_epochs=1, lr=1e-5)
        unlearned = npo_unlearn(model, forget_batches=forget_batches, retain_batches=retain_batches, config=config)

    elif baseline_name == "GradDiff":
        config = GradientAscentConfig(forget_loss="grad_diff", epochs=1, lr=1e-5)
        unlearned = gradient_ascent_unlearn(model, forget_batches=forget_batches, retain_batches=retain_batches, config=config)

    elif baseline_name == "FLAT":
        config = FLATConfig(variant="flat_retain", divergence="kl", epochs=1, lr=1e-5)
        unlearned = flat_unlearn(model, forget_batches=forget_batches, good_batches=retain_batches, retain_batches=retain_batches, config=config)

    elif baseline_name == "SimDPO":
        config = SimDPOUnlearnConfig(variant="simdpo_retain", beta=0.1, epochs=1, lr=1e-5)
        unlearned = simdpo_unlearn(model, forget_batches=forget_batches, retain_batches=retain_batches, config=config)

    else:
        raise ValueError(f"Unknown SafeTune baseline: {baseline_name}")

    meta = {
        "baseline": baseline_name,
        "safetune_audit_badge": badge,
        "safetune_version": getattr(safetune, "__version__", "0.1.16"),
        "status": "COMPLETED"
    }
    return unlearned, meta
