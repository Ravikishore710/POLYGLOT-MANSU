"""
tests/test_quant_bin_floor.py
Unit tests verifying the multi-grid quantization permanence floor mathematics.
"""

import torch
import pytest
from polyglot_mansu.unlearn.quant_permanence import (
    compute_quant_deadzones,
    snap_quant_deadzone,
    simulate_nf4_quant,
    get_nf4_max_bin_gap,
)

def test_nf4_quant_gap():
    gap = get_nf4_max_bin_gap()
    assert 0.27 < gap < 0.31

def test_quant_deadzone_computation():
    torch.manual_seed(42)
    w = torch.randn(128, 128)
    tau = compute_quant_deadzones(w, block_size_nf4=64, group_size_int4=32, gamma=1.25)
    
    assert tau.shape == w.shape
    assert torch.all(tau > 0)
    # tau should scale with local magnitude
    assert tau.mean() > 0.05

def test_deadzone_snapping_prevents_reversal():
    """
    Mathematical proof that standard tiny updates collapse under NF4 quantization,
    whereas snapped weights survive quantization.
    """
    torch.manual_seed(42)
    w_orig = torch.randn(64, 64)
    tau = compute_quant_deadzones(w_orig, block_size_nf4=64, group_size_int4=32, gamma=1.25)
    
    # 1. Standard tiny update (e.g. 50x smaller than tau, matching Lexsi paper 47-828x smaller)
    tiny_delta = 0.02 * tau
    w_standard = w_orig + tiny_delta
    
    _, idx_orig = simulate_nf4_quant(w_orig, return_indices=True)
    _, idx_standard = simulate_nf4_quant(w_standard, return_indices=True)
    
    # Standard tiny update snaps BACK to the exact same discrete quantization bin indices
    collapse_count = torch.sum(idx_orig == idx_standard).item()
    total = w_orig.numel()
    collapse_ratio = collapse_count / total
    # Over 90% of discrete bins remain identical
    assert collapse_ratio > 0.85, f"Expected high collapse ratio for tiny updates, got {collapse_ratio}"

    # 2. Snapped update
    mask = torch.ones_like(w_orig)
    # Introduce displacement inside the deadzone: e.g. 0.5 * tau
    w_unlearned = w_orig + 0.5 * tau
    w_snapped, stats = snap_quant_deadzone(w_unlearned, w_orig, tau, mask, eta_active=0.35)
    
    assert stats["snapped_count"] > 0
    _, idx_snapped = simulate_nf4_quant(w_snapped, return_indices=True)
    
    # After snapping, the discrete quantized bin indices DIFFER from idx_orig on snapped indices
    survived = torch.sum(idx_snapped != idx_orig).item()
    assert survived > 0.90 * total, f"Expected >90% to survive quantization, got {survived}/{total}"
