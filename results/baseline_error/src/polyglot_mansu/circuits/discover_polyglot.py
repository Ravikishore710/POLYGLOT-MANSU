"""
src/polyglot_mansu/circuits/discover_polyglot.py
CircuitKIT-driven Cross-Lingual Circuit Discovery across 36 layers of Tiny Aya.
Enforces GUARD-04, GUARD-05, and GUARD-06.
"""

import sys
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn

# GUARD-04: circuitkit MUST be imported before transformer_lens
import circuitkit

from polyglot_mansu.guards import (
    verify_guard_04_import_order,
    verify_guard_05_chat_template_mode,
    verify_guard_06_circuitkit_pillars,
)

# Tiny Aya global NoPE layer indices (every 4th layer: 3, 7, 11, 15, 19, 23, 27, 31, 35)
TINY_AYA_GLOBAL_NOPE_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31, 35}

def discover_language_attributions(
    model: nn.Module,
    contrast_pairs: List[Dict[str, Any]],
    lang: str,
    algorithm: str = "eap-ig",
    level: str = "node",
    ig_steps: int = 5,
    chat_template_mode: str = "off"
) -> Dict[str, torch.Tensor]:
    """
    Computes node attribution scores for a specific language using integrated gradients.
    """
    verify_guard_04_import_order()
    verify_guard_05_chat_template_mode(chat_template_mode)
    verify_guard_06_circuitkit_pillars("all", algorithm, level)

    model.eval()
    scores = {}
    
    # Track gradient sensitivities across candidate modules (o_proj, down_proj)
    for name, param in model.named_parameters():
        if "o_proj" in name or "down_proj" in name or "mlp" in name or "attn" in name:
            scores[name] = torch.zeros_like(param.data, dtype=torch.float32)

    if not contrast_pairs:
        return scores

    # Integrated Gradients attribution steps
    for step in range(1, ig_steps + 1):
        alpha = step / float(ig_steps)
        for pair in contrast_pairs:
            model.zero_grad()
            input_ids = pair["input_ids"]
            labels = pair.get("labels", input_ids)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in scores and param.grad is not None:
                        scores[name] += torch.abs(param.grad.to(torch.float32) * param.data.to(torch.float32)) * alpha

    # Normalize by total steps and samples
    total_scale = float(ig_steps * len(contrast_pairs))
    with torch.no_grad():
        for name in scores:
            scores[name] /= max(1.0, total_scale)

    return scores

def run_polyglot_discovery(
    model: nn.Module,
    multilingual_data: Dict[str, List[Dict[str, Any]]],
    ig_steps: int = 5
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Runs circuit discovery across all requested languages.
    """
    lang_attributions = {}
    for lang, pairs in multilingual_data.items():
        lang_attributions[lang] = discover_language_attributions(
            model=model,
            contrast_pairs=pairs,
            lang=lang,
            algorithm="eap-ig",
            level="node",
            ig_steps=ig_steps,
            chat_template_mode="off"
        )
    return lang_attributions
