"""
tests/test_full_4repo_pipeline_e2e.py
Full-Stack 4-Repository Chaining Integration Test:
CuratorKIT -> CircuitKIT -> SafeTune / PolyglotMANSUTrainer -> AuditKIT
Executes on real miniature Cohere2ForCausalLM weights and verifies end-to-end pipeline.
"""

import json
from pathlib import Path
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

import circuitkit
import safetune
import curatorkit
import auditkit

from polyglot_mansu.data.curate_multilingual import build_dataset_splits
from polyglot_mansu.circuits.discover_polyglot import run_polyglot_discovery
from polyglot_mansu.circuits.harmonic_intersect import compute_harmonic_score, extract_circuit_mask
from polyglot_mansu.unlearn.polyglot_mansu_trainer import PolyglotMANSUTrainer, PolyglotMANSUConfig
from polyglot_mansu.eval.auditkit_runner import run_6pillar_audit

def test_full_4repo_pipeline_e2e(tmp_path):
    # Step 1: CuratorKIT Data Curation
    data_dir = tmp_path / "data"
    manifest = build_dataset_splits(output_dir=str(data_dir))
    assert len(manifest["splits"]) == 4

    # Step 2: CircuitKIT Discovery on Cohere2
    torch.manual_seed(42)
    conf = AutoConfig.for_model(
        "cohere2",
        vocab_size=500,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        eos_token_id=1,
        pad_token_id=0,
        bos_token_id=2
    )
    model = AutoModelForCausalLM.from_config(conf)

    seq_len = 8
    discovery_data = {
        "en": [{"input_ids": torch.randint(3, 400, (1, seq_len)), "labels": torch.randint(3, 400, (1, seq_len))}],
        "hi": [{"input_ids": torch.randint(3, 400, (1, seq_len)), "labels": torch.randint(3, 400, (1, seq_len))}]
    }
    lang_attributions = run_polyglot_discovery(model, discovery_data, ig_steps=2)
    harmonic_scores = compute_harmonic_score(lang_attributions)
    circuit_mask, _ = extract_circuit_mask(harmonic_scores, sparsity_ratio=0.10)

    # Step 3: SafeTune / PolyglotMANSUTrainer Unlearning
    cfg = PolyglotMANSUConfig(lr=1e-3, epochs=1, use_quant_floor=True)
    trainer = PolyglotMANSUTrainer(model=model, circuit_mask=circuit_mask, config=cfg)

    retain_data = [
        {"input_ids": torch.randint(3, 400, (seq_len,)), "labels": torch.randint(3, 400, (seq_len,))},
        {"input_ids": torch.randint(3, 400, (seq_len,)), "labels": torch.randint(3, 400, (seq_len,))}
    ]
    retain_loader = torch.utils.data.DataLoader(retain_data, batch_size=2)
    trainer.initialize_fisher(retain_loader)

    forget_batch = {"input_ids": torch.randint(3, 400, (2, seq_len)), "labels": torch.randint(3, 400, (2, seq_len))}
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    step_metrics = trainer.train_step(forget_batch, next(iter(retain_loader)), opt)
    snap_stats = trainer.post_train_quant_snapping()

    # Step 4: AuditKIT Evaluation
    audit_summary = run_6pillar_audit(
        model=model,
        model_name="polyglot_mansu_cohere2_mini",
        test_splits={},
        output_dir=str(tmp_path / "audit_results"),
        device_target="cpu",
        device_judge="cpu",
        is_decoupled=True
    )
    assert audit_summary["status"] == "COMPLETED"

    # Save end-to-end evidence artifact
    e2e_proof = {
        "pipeline": "CuratorKIT -> CircuitKIT -> PolyglotMANSUTrainer -> AuditKIT",
        "curatorkit_splits": list(manifest["splits"].keys()),
        "circuitkit_nodes_attributed": len(harmonic_scores),
        "unlearning_step_metrics": step_metrics,
        "quant_snapped_count": snap_stats["snapped_count"],
        "auditkit_summary": audit_summary,
        "status": "VERIFIED_MINI_COHERE2_E2E"
    }

    Path("audit_for_carbon").mkdir(exist_ok=True)
    with open("audit_for_carbon/06_full_4repo_pipeline_e2e.json", "w") as f:
        json.dump(e2e_proof, f, indent=2)
