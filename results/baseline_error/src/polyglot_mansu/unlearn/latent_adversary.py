"""
src/polyglot_mansu/unlearn/latent_adversary.py
Mid-Layer Latent Adversarial Perturbation (AGTAO):
Computes 1-step PGD perturbation in residual stream to prevent refusal bypasses,
prefill attacks, and linear probe extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class LatentAdversaryHook:
    """
    Forward hook attaching to residual stream at layer l_hub to inject
    or compute worst-case latent perturbation delta*.
    """
    def __init__(self, rho: float = 0.08, eps: float = 1e-6):
        self.rho = rho
        self.eps = eps
        self.delta: Optional[torch.Tensor] = None
        self.cached_activation: Optional[torch.Tensor] = None
        self.handle = None

    def hook_fn(self, module, inputs, output):
        # In transformer decoders, output is typically a tuple (hidden_states, ...) or tensor
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = ()

        self.cached_activation = h

        if self.delta is not None and self.delta.shape == h.shape:
            h_perturbed = h + self.delta
        else:
            h_perturbed = h

        if rest:
            return (h_perturbed,) + rest
        return h_perturbed

    def attach(self, layer_module: nn.Module):
        self.handle = layer_module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


def compute_latent_perturbation(
    model: nn.Module,
    hook: LatentAdversaryHook,
    batch: dict,
    device: torch.device
) -> torch.Tensor:
    """
    1-step PGD inner loop: finds delta* with ||delta*|| <= rho * ||h||
    maximizing forbidden target likelihood.
    """
    input_ids = batch["input_ids"].to(device)
    labels = batch.get("labels", input_ids).to(device)

    # 1. Forward pass with zero perturbation to capture h
    hook.delta = None
    outputs = model(input_ids=input_ids, labels=labels)
    h = hook.cached_activation

    # Create leaf variable delta initialized to 0
    delta = torch.zeros_like(h, requires_grad=True, device=device)
    hook.delta = delta

    # 2. Forward pass with delta requiring grad
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    # We want to MAXIMIZE target likelihood, so we minimize loss w.r.t delta (gradient points toward higher logp)
    loss.backward()

    # 3. Compute normalized perturbation step
    grad_delta = delta.grad
    if grad_delta is None:
        grad_delta = torch.randn_like(h)

    norm_h = torch.norm(h.detach(), dim=-1, keepdim=True)
    norm_grad = torch.norm(grad_delta, dim=-1, keepdim=True) + hook.eps
    
    # delta* = - rho * ||h|| * (grad / ||grad||) to decrease loss w.r.t labels
    delta_star = - hook.rho * norm_h * (grad_delta / norm_grad)
    hook.delta = delta_star.detach()

    return hook.delta
