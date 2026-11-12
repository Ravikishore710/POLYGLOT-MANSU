"""
src/polyglot_mansu/guards.py
Runtime enforcement and validation of the 8 mandatory Lexsi Stack & Cohere2 runtime guards.
"""

import sys
import importlib
from typing import Dict, Any, Optional

def verify_guard_01_max_len(max_len: Optional[int]) -> bool:
    """
    GUARD-01: Tiny Aya chat template injects a ~366-token system preamble.
    Setting max_len < 512 leaves zero supervised tokens and crashes dataloader.
    """
    if max_len is not None and max_len < 512:
        raise ValueError(
            f"GUARD-01 Violation: max_len={max_len} is < 512. Tiny Aya system preamble is ~366 tokens. "
            f"Use max_len >= 512 (recommended 1024) or None."
        )
    return True

def verify_guard_02_environment() -> Dict[str, Any]:
    """
    GUARD-02: Python >= 3.12, torchao conflicts avoided.
    """
    py_ver = sys.version_info
    if py_ver < (3, 12):
        raise RuntimeError(
            f"GUARD-02 Violation: Python {py_ver.major}.{py_ver.minor} detected. SafeTune requires Python >= 3.12."
        )
    
    # Check if torchao is present
    torchao_present = importlib.util.find_spec("torchao") is not None
    return {
        "python_version": f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}",
        "torchao_present": torchao_present,
        "valid": True
    }

def verify_guard_03_audit_badge(trainer_name: str, badge: str) -> bool:
    """
    GUARD-03: Programmatically inspect and log each baseline trainer's SafeTune audit badge
    ('Faithful', 'Simplified', 'Variant', 'Wrong/Stub').
    """
    valid_badges = {"Faithful", "Simplified", "Variant", "Wrong/Stub"}
    if badge not in valid_badges:
        raise ValueError(f"GUARD-03: Invalid SafeTune audit badge '{badge}' for {trainer_name}.")
    return True

def verify_guard_04_import_order() -> bool:
    """
    GUARD-04: circuitkit MUST be imported before transformer_lens.
    """
    tl_imported = "transformer_lens" in sys.modules
    ck_imported = "circuitkit" in sys.modules
    
    if tl_imported and not ck_imported:
        raise ImportError(
            "GUARD-04 Violation: transformer_lens was imported before circuitkit! "
            "circuitkit applies patch 0006-cohere to TransformerLens on import."
        )
    return True

def verify_guard_05_chat_template_mode(mode: str) -> bool:
    """
    GUARD-05: chat_template_mode must be 'off' inside discover().
    """
    if mode != "off":
        raise ValueError(
            f"GUARD-05 Violation: chat_template_mode='{mode}'. Must be 'off' inside discover() "
            f"to prevent nested template wrapping and corrupted circuit geometry."
        )
    return True

def verify_guard_06_circuitkit_pillars(pillars: Any, algorithm: str, level: str) -> bool:
    """
    GUARD-06: pillars='all', algorithm='eap-ig', level='node', avoiding OOM on 3.35B.
    """
    if algorithm != "eap-ig":
        raise ValueError(f"GUARD-06 Violation: algorithm='{algorithm}', expected 'eap-ig'")
    if level != "node":
        raise ValueError(f"GUARD-06 Violation: level='{level}', expected 'node'")
    if pillars != "all":
        # Check if list contains all 6 pillars
        expected_pillars = {"patching", "ablation", "stability", "robustness", "baselines", "generalization"}
        if isinstance(pillars, (list, tuple, set)):
            if not expected_pillars.issubset(set(pillars)):
                raise ValueError(f"GUARD-06 Violation: pillars must include all 6 pillars or be 'all'. Got: {pillars}")
        else:
            raise ValueError(f"GUARD-06 Violation: pillars must be 'all'. Got: {pillars}")
    return True

def verify_guard_07_curatorkit_export(task_type: str, export_format: str) -> bool:
    """
    GUARD-07: Strict pairing of task_type and export_format in CuratorKIT.
    """
    qa_tasks = {"qa", "adversarial_qa"}
    pref_tasks = {"preference", "adversarial_preference"}
    
    if task_type in qa_tasks and export_format not in {"alpaca", "sharegpt"}:
        raise ValueError(
            f"GUARD-07 Violation: task_type '{task_type}' paired with invalid export_format '{export_format}'. "
            f"QA tasks must use 'alpaca' or 'sharegpt'."
        )
    if task_type in pref_tasks and export_format != "dpo":
        raise ValueError(
            f"GUARD-07 Violation: task_type '{task_type}' paired with invalid export_format '{export_format}'. "
            f"Preference tasks must use 'dpo'."
        )
    return True

def verify_guard_08_judge_vram_decoupling(device_target: str, device_judge: str, is_decoupled: bool = False) -> bool:
    """
    GUARD-08: Prevent CUDA OOM between Tiny Aya (3.35B) and WildGuard (7B).
    Must be on different devices or run in two-phase decoupled mode.
    """
    if not is_decoupled and device_target == device_judge and "cuda" in device_target:
        raise RuntimeError(
            f"GUARD-08 Violation: Target model and judge both assigned to '{device_target}'. "
            f"This will cause CUDA OOM on a 16 GB GPU. Use separate GPUs (cuda:0 vs cuda:1) "
            f"or enable two-phase decoupled evaluation."
        )
    return True
