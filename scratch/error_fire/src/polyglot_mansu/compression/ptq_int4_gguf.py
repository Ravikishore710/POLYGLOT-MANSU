"""
src/polyglot_mansu/compression/ptq_int4_gguf.py
Symmetric 4-bit Group Quantization (INT4 / GGUF Q4_K_M Style) & Reversal Gap Evaluator:
Calculates Delta_PTQ_INT4 = Acc_f(INT4) - Acc_f(BF16).
"""

from typing import Dict, Any
import torch
import torch.nn as nn

def quantize_tensor_int4(w: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Simulates symmetric 4-bit quantization with group size 32."""
    flat = w.reshape(-1)
    n = flat.numel()
    pad = (group_size - (n % group_size)) % group_size
    if pad > 0:
        flat_padded = torch.cat([flat, torch.zeros(pad, device=flat.device)])
    else:
        flat_padded = flat

    blocks = flat_padded.reshape(-1, group_size)
    absmax = torch.max(torch.abs(blocks), dim=1, keepdim=True).values + 1e-8
    scale = absmax / 7.0  # 4-bit signed ranges from -7 to 7

    # Quantize to discrete integers
    q = torch.clamp(torch.round(blocks / scale), -7, 7)
    dequant = q * scale
    return dequant.reshape(-1)[:n].reshape(w.shape)

def quantize_model_int4(model: nn.Module, group_size: int = 32) -> nn.Module:
    """Quantizes model weights to symmetric INT4."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                param.data.copy_(quantize_tensor_int4(param.data, group_size=group_size))
    return model

def evaluate_int4_reversal_gap(
    acc_bf16: float,
    acc_int4: float
) -> Dict[str, Any]:
    """Delta_PTQ_INT4 = Acc_f(INT4) - Acc_f(BF16)."""
    delta_ptq = acc_int4 - acc_bf16
    return {
        "acc_bf16": acc_bf16,
        "acc_int4": acc_int4,
        "delta_ptq_int4": delta_ptq,
        "resurrection_detected": bool(delta_ptq > 0.01),
        "quant_permanent": bool(delta_ptq <= 0.0)
    }
