"""
src/polyglot_mansu/unlearn/fisher_nullspace.py
Fisher Null-Space Preconditioning and Adaptive Conflict Orthogonalization (AGTAO):
Protects high-sensitivity retain + hard-boundary knowledge while unlearning.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

def compute_diagonal_fisher(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    circuit_mask: Dict[str, torch.Tensor],
    device: torch.device,
    max_batches: int = 10,
    max_seq_len: int = 256
) -> Dict[str, torch.Tensor]:
    """
    Computes empirical diagonal Fisher sensitivities F_i on retain/boundary data
    restricted to the circuit mask M_C*.
    Truncates sequence length to max_seq_len to prevent activation VRAM OOM.
    """
    model.eval()
    fisher = {name: torch.zeros_like(p, device=device) for name, p in model.named_parameters() if name in circuit_mask}
    total_samples = 0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        model.zero_grad()
        input_ids = batch["input_ids"].to(device)
        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, :max_seq_len]
        labels = batch.get("labels", input_ids).to(device)
        if labels.shape[1] > max_seq_len:
            labels = labels[:, :max_seq_len]
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        batch_size = input_ids.shape[0]
        total_samples += batch_size

        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in circuit_mask and p.grad is not None:
                    fisher[name] += (p.grad ** 2) * batch_size

        model.zero_grad()
        del outputs, loss, input_ids, labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with torch.no_grad():
        for name in fisher:
            if total_samples > 0:
                mask = circuit_mask[name].to(device)
                fisher[name] = (fisher[name] / total_samples) * mask

    return fisher

def compute_fisher_preconditioner(
    fisher: Dict[str, torch.Tensor],
    circuit_mask: Dict[str, torch.Tensor],
    q: float = 0.85,
    alpha_F: float = 1.0,
    eps: float = 1e-6
) -> Dict[str, torch.Tensor]:
    """
    P_i = M_{C*, i} * I[F_i <= Q_q(F_{C*})] * (1 + F_i / (median(F_{C*}) + eps))^(-alpha_F)
    Zeroes out the top (1 - q) fraction (e.g. 15%) highest-sensitivity retain parameters.
    """
    preconditioner = {}
    
    # Collect all non-zero Fisher values across circuit mask
    all_vals = []
    for name, f in fisher.items():
        m = circuit_mask.get(name)
        if m is not None:
            active_f = f[m.to(f.device) > 0]
            if active_f.numel() > 0:
                all_vals.append(active_f.detach().cpu())

    if all_vals:
        concat_f = torch.cat(all_vals)
        if concat_f.numel() > 1_000_000:
            step = concat_f.numel() // 1_000_000
            sample_f = concat_f[::step][:1_000_000]
        else:
            sample_f = concat_f
        cutoff_val = torch.quantile(sample_f, q).item()
        med_val = torch.median(sample_f).item()
    else:
        cutoff_val = float("inf")
        med_val = 1.0

    for name, f in fisher.items():
        m = circuit_mask[name].to(f.device)
        mask_condition = (f <= cutoff_val).float() * m
        damping = torch.pow(1.0 + f / (med_val + eps), -alpha_F)
        preconditioner[name] = mask_condition * damping

    return preconditioner

def orthogonalize_gradients(
    g_f: torch.Tensor,
    g_r: torch.Tensor,
    lambda_ret: float = 1.0,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, float]:
    """
    Adaptive Gradient Orthogonalization (AGTAO):
    If <g_f, g_r> < 0:
        g_total = (g_f - (<g_f, g_r> / (||g_r||_2^2 + eps)) * g_r) + lambda_ret * g_r
    Else:
        g_total = g_f + lambda_ret * g_r
    Returns:
        (g_total, dot_product)
    """
    dot = torch.sum(g_f * g_r).item()
    if dot < 0:
        norm_sq = torch.sum(g_r ** 2).item()
        denom = norm_sq if norm_sq > eps else eps
        proj = (dot / denom) * g_r
        g_f_orth = g_f - proj
        g_total = g_f_orth + lambda_ret * g_r
    else:
        g_total = g_f + lambda_ret * g_r

    return g_total, dot
