"""
tests/test_guards.py
Unit tests for the 8 runtime guards in src/polyglot_mansu/guards.py.
"""

import pytest
from polyglot_mansu.guards import (
    verify_guard_01_max_len,
    verify_guard_02_environment,
    verify_guard_03_audit_badge,
    verify_guard_04_import_order,
    verify_guard_05_chat_template_mode,
    verify_guard_06_circuitkit_pillars,
    verify_guard_07_curatorkit_export,
    verify_guard_08_judge_vram_decoupling,
)

def test_guard_01_max_len():
    # >= 512 is allowed
    assert verify_guard_01_max_len(512) is True
    assert verify_guard_01_max_len(1024) is True
    assert verify_guard_01_max_len(None) is True
    
    # < 512 must raise ValueError
    with pytest.raises(ValueError, match="GUARD-01 Violation"):
        verify_guard_01_max_len(256)
    with pytest.raises(ValueError, match="GUARD-01 Violation"):
        verify_guard_01_max_len(511)

def test_guard_02_environment():
    env_info = verify_guard_02_environment()
    assert env_info["valid"] is True
    assert "python_version" in env_info

def test_guard_03_audit_badge():
    for badge in ["Faithful", "Simplified", "Variant", "Wrong/Stub"]:
        assert verify_guard_03_audit_badge("RMUTrainer", badge) is True
    with pytest.raises(ValueError, match="Invalid SafeTune audit badge"):
        verify_guard_03_audit_badge("RMUTrainer", "UnknownBadge")

def test_guard_04_import_order():
    # Currently circuitkit is imported before or at same time
    assert verify_guard_04_import_order() is True

def test_guard_05_chat_template_mode():
    assert verify_guard_05_chat_template_mode("off") is True
    with pytest.raises(ValueError, match="GUARD-05 Violation"):
        verify_guard_05_chat_template_mode("auto")

def test_guard_06_circuitkit_pillars():
    assert verify_guard_06_circuitkit_pillars("all", "eap-ig", "node") is True
    six = ["patching", "ablation", "stability", "robustness", "baselines", "generalization"]
    assert verify_guard_06_circuitkit_pillars(six, "eap-ig", "node") is True
    
    with pytest.raises(ValueError, match="pillars"):
        verify_guard_06_circuitkit_pillars(["patching", "ablation"], "eap-ig", "node")
    with pytest.raises(ValueError, match="algorithm"):
        verify_guard_06_circuitkit_pillars("all", "act-patch", "node")

def test_guard_07_curatorkit_export():
    assert verify_guard_07_curatorkit_export("qa", "alpaca") is True
    assert verify_guard_07_curatorkit_export("adversarial_qa", "sharegpt") is True
    assert verify_guard_07_curatorkit_export("preference", "dpo") is True
    
    with pytest.raises(ValueError, match="GUARD-07 Violation"):
        verify_guard_07_curatorkit_export("qa", "dpo")
    with pytest.raises(ValueError, match="GUARD-07 Violation"):
        verify_guard_07_curatorkit_export("preference", "alpaca")

def test_guard_08_judge_vram_decoupling():
    assert verify_guard_08_judge_vram_decoupling("cuda:0", "cuda:1") is True
    assert verify_guard_08_judge_vram_decoupling("cuda:0", "cuda:0", is_decoupled=True) is True
    assert verify_guard_08_judge_vram_decoupling("cpu", "cpu") is True
    
    with pytest.raises(RuntimeError, match="GUARD-08 Violation"):
        verify_guard_08_judge_vram_decoupling("cuda:0", "cuda:0", is_decoupled=False)
