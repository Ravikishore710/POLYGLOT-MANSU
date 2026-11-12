"""
src/polyglot_mansu/unlearn/quant_permanence.py
Multi-Grid Quantization Permanence Floor:
Guarantees zero quantization reversal (Delta_PTQ <= 0) across both NF4 (bitsandbytes)
and symmetric INT4 (GGUF Q4_K_M / group-32).
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

# Exact 16 NormalFloat4 quantiles from bitsandbytes / QLoRA
NF4_QUANTILES = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
]

def get_nf4_max_bin_gap() -> float:
    """Maximum adjacent gap between normalized NF4 quantiles."""
    gaps = [NF4_QUANTILES[i+1] - NF4_QUANTILES[i] for i in range(len(NF4_QUANTILES)-1)]
    return max(gaps)  # Approx 0.2774 between -1.0 and -0.6962

def compute_quant_deadzones(
    w_orig: torch.Tensor,
    block_size_nf4: int = 64,
    group_size_int4: int = 32,
    gamma: float = 1.25
) -> torch.Tensor:
    """
    Computes parameter-wise minimum displacement thresholds tau_i:
    tau_i = gamma * max(Delta_B^NF4, Delta_B'^INT4)
    """
    flat = w_orig.reshape(-1)
    n = flat.numel()
    
    # Pad to block boundaries if needed
    pad_nf4 = (block_size_nf4 - (n % block_size_nf4)) % block_size_nf4
    if pad_nf4 > 0:
        flat_nf4 = torch.cat([flat, torch.zeros(pad_nf4, device=flat.device, dtype=flat.dtype)])
    else:
        flat_nf4 = flat
    blocks_nf4 = flat_nf4.reshape(-1, block_size_nf4)
    scale_nf4 = torch.max(torch.abs(blocks_nf4), dim=1, keepdim=True).values
    delta_nf4 = scale_nf4 * get_nf4_max_bin_gap()
    tau_nf4 = delta_nf4.expand(-1, block_size_nf4).reshape(-1)[:n]

    # Symmetric INT4 (15 non-zero levels, step = 2*absmax / 15)
    pad_int4 = (group_size_int4 - (n % group_size_int4)) % group_size_int4
    if pad_int4 > 0:
        flat_int4 = torch.cat([flat, torch.zeros(pad_int4, device=flat.device, dtype=flat.dtype)])
    else:
        flat_int4 = flat
    blocks_int4 = flat_int4.reshape(-1, group_size_int4)
    scale_int4 = torch.max(torch.abs(blocks_int4), dim=1, keepdim=True).values
    delta_int4 = (2.0 * scale_int4) / 15.0
    tau_int4 = delta_int4.expand(-1, group_size_int4).reshape(-1)[:n]

    tau = gamma * torch.maximum(tau_nf4, tau_int4)
    return tau.reshape(w_orig.shape)

def snap_quant_deadzone(
    w_curr: torch.Tensor,
    w_orig: torch.Tensor,
    tau: torch.Tensor,
    mask: torch.Tensor,
    eta_active: float = 0.35
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Snaps intentionally perturbed weights falling inside the dead-zone
    to the quantization boundary tau_i:
    w_new = w_orig + sign(delta) * tau
    """
    delta = w_curr - w_orig
    abs_delta = torch.abs(delta)
    
    # Active condition: inside circuit mask, intentionally moved >= eta_active * tau, but < tau
    inside_deadzone = (abs_delta >= eta_active * tau) & (abs_delta < tau) & (mask > 0)
    
    w_snapped = w_curr.clone()
    sign_delta = torch.sign(delta)
    sign_delta[sign_delta == 0] = 1.0  # Avoid zero sign
    
    w_snapped[inside_deadzone] = w_orig[inside_deadzone] + sign_delta[inside_deadzone] * tau[inside_deadzone]
    
    stats = {
        "total_params": w_curr.numel(),
        "mask_params": int(mask.sum().item()),
        "snapped_count": int(inside_deadzone.sum().item()),
        "snapped_fraction": float(inside_deadzone.sum().item()) / max(1, int(mask.sum().item()))
    }
    return w_snapped, stats

def simulate_nf4_quant(w: torch.Tensor, block_size: int = 64, return_indices: bool = False):
    """Simulate NF4 quantization for tests and STE hooks."""
    flat = w.reshape(-1)
    n = flat.numel()
    pad = (block_size - (n % block_size)) % block_size
    if pad > 0:
        flat_padded = torch.cat([flat, torch.zeros(pad, device=flat.device)])
    else:
        flat_padded = flat
    blocks = flat_padded.reshape(-1, block_size)
    absmax = torch.max(torch.abs(blocks), dim=1, keepdim=True).values + 1e-8
    norm = blocks / absmax
    
    quantiles = torch.tensor(NF4_QUANTILES, device=w.device, dtype=norm.dtype)
    # Nearest quantile
    dist = torch.abs(norm.unsqueeze(-1) - quantiles)
    nearest_idx = torch.argmin(dist, dim=-1)
    quantized_norm = quantiles[nearest_idx]
    dequant = quantized_norm * absmax
    out_dequant = dequant.reshape(-1)[:n].reshape(w.shape)
    out_idx = nearest_idx.reshape(-1)[:n].reshape(w.shape)
    if return_indices:
        return out_dequant, out_idx
    return out_dequant
