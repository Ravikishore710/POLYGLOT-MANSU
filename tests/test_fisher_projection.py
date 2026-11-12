"""
tests/test_fisher_projection.py
Unit tests verifying Fisher null-space preconditioning and gradient orthogonalization.
"""

import torch
import pytest
from polyglot_mansu.unlearn.fisher_nullspace import (
    compute_fisher_preconditioner,
    orthogonalize_gradients,
)

def test_fisher_preconditioner_masks_top_percentile():
    torch.manual_seed(42)
    # Synthetic parameters
    fisher = {"weight": torch.rand(100, 100)}
    circuit_mask = {"weight": torch.ones(100, 100)}
    
    # 85th percentile cutoff means top 15% should be strictly zeroed out
    p = compute_fisher_preconditioner(fisher, circuit_mask, q=0.85, alpha_F=1.0)
    
    assert "weight" in p
    p_weight = p["weight"]
    zero_count = torch.sum(p_weight == 0).item()
    total = p_weight.numel()
    
    # Approx 15% zeroed out
    assert 0.14 <= (zero_count / total) <= 0.16
    
    # Non-zero entries must be smoothly damped
    non_zeros = p_weight[p_weight > 0]
    assert torch.all(non_zeros <= 1.0)
    assert torch.all(non_zeros > 0.0)

def test_gradient_orthogonalization_removes_negative_dot():
    torch.manual_seed(42)
    # Construct conflicting gradients
    g_r = torch.tensor([1.0, 0.0, 0.0])
    g_f = torch.tensor([-1.0, 2.0, 0.0])
    
    dot_initial = torch.sum(g_f * g_r).item()
    assert dot_initial < 0, "Initial dot product must be negative"
    
    g_total, dot_logged = orthogonalize_gradients(g_f, g_r, lambda_ret=1.0)
    
    # Check that the modified forget component is orthogonal to g_r
    # g_total = g_f_orth + g_r, so <g_f_orth, g_r> = 0
    g_f_orth = g_total - g_r
    dot_after = torch.sum(g_f_orth * g_r).item()
    
    assert abs(dot_after) < 1e-5, f"Expected orthogonal forget gradient, got dot={dot_after}"
    # Verify overall projection preserves positive retain contribution
    assert torch.sum(g_total * g_r).item() > 0
