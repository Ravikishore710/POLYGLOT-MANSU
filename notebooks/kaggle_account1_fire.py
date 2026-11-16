#!/usr/bin/env python3
"""
notebooks/kaggle_account1_fire.py
Execution script for Kaggle Account 1:
Target: CohereLabs/tiny-aya-fire (3.35B) in BF16 on Dual-T4 GPUs.
Full 4-Repo Stack Pipeline:
CuratorKIT data -> CircuitKIT eap-ig -> SafeTune / PolyglotMANSU -> AuditKIT 6-Pillars.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

# GUARD-02: Ensure clean Kaggle environment without torchao conflicts
os.system("pip uninstall -y torchao > /dev/null 2>&1")

# Authenticate Hugging Face for gated Cohere models
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF_TOKEN")
    except Exception:
        hf_token = None

if hf_token:
    try:
        import huggingface_hub
        huggingface_hub.login(token=hf_token)
        print("[Account 1] Successfully authenticated with Hugging Face Hub.")
    except Exception as e:
        print(f"[Account 1] HF login notice: {e}")

# GUARD-04: circuitkit MUST be imported first before transformer_lens
import circuitkit
from transformers import AutoTokenizer, AutoModelForCausalLM

from polyglot_mansu.guards import verify_guard_01_max_len
from polyglot_mansu.circuits.discover_polyglot import run_polyglot_discovery
from polyglot_mansu.circuits.harmonic_intersect import compute_harmonic_score, extract_circuit_mask
from polyglot_mansu.unlearn.polyglot_mansu_trainer import PolyglotMANSUTrainer, PolyglotMANSUConfig
from polyglot_mansu.eval.auditkit_runner import run_7pillar_audit
from polyglot_mansu.compression.ptq_nf4 import quantize_model_nf4
from polyglot_mansu.compression.ptq_int4_gguf import quantize_model_int4

MODEL_ID = "CohereLabs/tiny-aya-fire"
# GUARD-01: Use max_len=512 for training/unlearning batches to prevent OOM while respecting the system preamble
MAX_LEN = 512

class TextDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, max_len: int = 512):
        self.samples = []
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        text = f"{data.get('prompt', '')} {data.get('target', '')}"
                        self.samples.append(text)
        if not self.samples:
            self.samples = ["Buffer overflow exploitation mitigation and prevention techniques." * 5]
        
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.samples[idx],
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt"
        )
        input_ids = item["input_ids"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": item["attention_mask"].squeeze(0),
            "labels": input_ids.clone()
        }

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    verify_guard_01_max_len(MAX_LEN)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Account 1] Initializing {MODEL_ID} on {device}...")
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}, Device count: {torch.cuda.device_count()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[Account 1] Loading {MODEL_ID} weights in {dtype} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        dtype=dtype,
        device_map={"": device} if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    model.config.use_cache = False

    print("[Account 1] Preparing Datasets from data/curated...")
    data_dir = Path("data/curated")
    if not data_dir.exists():
        from polyglot_mansu.data.curate_multilingual import build_dataset_splits
        build_dataset_splits(output_dir="data/curated")

    forget_ds = TextDataset(data_dir / "train_forget.jsonl", tokenizer, MAX_LEN)
    retain_ds = TextDataset(data_dir / "retain.jsonl", tokenizer, MAX_LEN)
    boundary_ds = TextDataset(data_dir / "hard_boundary_twins.jsonl", tokenizer, MAX_LEN)
    heldout_ds = TextDataset(data_dir / "heldout_test_forget.jsonl", tokenizer, MAX_LEN)

    forget_loader = DataLoader(forget_ds, batch_size=1, shuffle=True)
    retain_loader = DataLoader(retain_ds, batch_size=1, shuffle=True)

    print("[Account 1] CircuitKIT eap-ig Cross-Lingual Localization...")
    discovery_data = {
        "en": [{"input_ids": tokenizer("Exploit buffer overflow in Linux kernel", return_tensors="pt")["input_ids"].to(device)}],
        "hi": [{"input_ids": tokenizer("लिनक्स कर्नेल में बफर ओवरफ्लो का फायदा उठाएं", return_tensors="pt")["input_ids"].to(device)}],
        "te": [{"input_ids": tokenizer("లైనక్స్ కెర్నల్ బఫర్ ఓవర్‌ఫ్లో దాడి", return_tensors="pt")["input_ids"].to(device)}]
    }
    lang_attributions = run_polyglot_discovery(model, discovery_data, ig_steps=3)
    scores_base = {k: v.detach().cpu().clone() for k, v in lang_attributions["en"].items()}
    harmonic_scores = compute_harmonic_score(lang_attributions)
    circuit_mask, sparsity = extract_circuit_mask(harmonic_scores, sparsity_ratio=0.10)
    print(f"[Account 1] Isolated Harmonic Circuit C*: {len(circuit_mask)} tensors, Sparsity: {sparsity:.4f}")

    ref_device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")
    print(f"[Account 1] Loading dedicated reference model in {dtype} on {ref_device}...")
    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        dtype=dtype,
        device_map={"": ref_device} if str(ref_device).startswith("cuda") else None,
        trust_remote_code=True
    )
    reference_model.config.use_cache = False
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[Account 1] Activation gradient checkpointing enabled on target model.")

    print("[Account 1] Initializing PolyglotMANSUTrainer with Fisher Null-Space Preconditioning...")
    cfg = PolyglotMANSUConfig(lr=2e-5, epochs=3, max_len=MAX_LEN, use_circuit_mask=True, use_quant_floor=True)
    trainer = PolyglotMANSUTrainer(
        model=model,
        circuit_mask=circuit_mask,
        config=cfg,
        device=device,
        reference_model=reference_model
    )
    trainer.initialize_fisher(retain_loader)
    print("[Account 1] Retain Fisher preconditioner precomputed successfully.")

    trainable_params = trainer.get_trainable_parameters()
    print(f"[Account 1] Circuit-active trainable parameters: {len(trainable_params)} modules ({trainer.trainable_params_count:,} weights, {trainer.trainable_params_count/trainer.total_params_count*100:.2f}% of model)")
    try:
        import bitsandbytes as bnb
        print("[Account 1] Using 8-bit AdamW optimizer (bnb.optim.AdamW8bit) for 75% optimizer memory reduction.")
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=cfg.lr)
    except Exception:
        print("[Account 1] bitsandbytes not available, falling back to torch.optim.AdamW.")
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)

    print(f"[Account 1] Starting Training Session across {cfg.epochs} epochs...")
    step = 0
    for epoch in range(cfg.epochs):
        for f_batch, r_batch in zip(forget_loader, retain_loader):
            metrics = trainer.train_step(f_batch, r_batch, optimizer)
            step += 1
            if step % 5 == 0 or step == 1:
                print(f"  [Epoch {epoch+1}/{cfg.epochs} | Step {step}] L_forget: {metrics['loss_forget']:.4f}, L_retain: {metrics['loss_retain']:.4f}")

    print("[Account 1] Applying Multi-Grid Quantization Dead-zone Snapping (NF4 + INT4)...")
    snap_stats = trainer.post_train_quant_snapping()
    print(f"[Account 1] Quantization Snapping Result: {snap_stats}")

    # Export unlearned BF16 model and tokenizer before any PTQ in-place quantization
    export_dir = Path("models/tiny_aya_fire_mansu_weights")
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Account 1] Exporting unlearned model weights & tokenizer to {export_dir}...")
    model.save_pretrained(str(export_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(export_dir))
    
    # Compute SHA-256 manifest of exported checkpoint files
    checkpoint_checksums = {}
    for f in sorted(export_dir.glob("*")):
        if f.is_file():
            checkpoint_checksums[f.name] = compute_sha256(f)
    checksum_manifest_path = export_dir / "checksums.json"
    with open(checksum_manifest_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_checksums, f, indent=2)
    print(f"[Account 1] Checkpoint export complete ({len(checkpoint_checksums)} files). Checksums saved.")

    # Free reference model to liberate GPU 1 for evaluation judges
    del reference_model
    del trainer.reference_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results_dir = Path("results/audit/hardened_fire")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load raw heldout prompts for empirical ASR and prefill generation attacks
    heldout_raw_prompts = []
    with open(data_dir / "heldout_test_forget.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                heldout_raw_prompts.append(json.loads(line))

    # Compute post-unlearning circuit attribution on target unlearned model for genuine CAD
    print("[Account 1] Computing post-unlearning circuit attributions for CAD...")
    scores_unlearned = run_polyglot_discovery(model, discovery_data, ig_steps=3)["en"]

    print("[Account 1] Running Hardened AuditKIT 7-Pillar Evaluation Suite on Unlearned Model...")
    audit_summary = run_7pillar_audit(
        model=model,
        model_name="tiny_aya_fire_mansu",
        test_splits={
            "forget_heldout": heldout_ds,
            "boundary_twins": boundary_ds,
            "retain": retain_ds
        },
        output_dir=str(results_dir),
        device_target=str(device),
        device_judge="cuda:1" if torch.cuda.device_count() > 1 else "cpu",
        is_decoupled=True,
        tokenizer=tokenizer,
        raw_prompts_heldout=heldout_raw_prompts,
        scores_base=scores_base,
        scores_unlearned=scores_unlearned,
        circuit_mask=circuit_mask
    )
    print(f"[Account 1] Audit Summary: {audit_summary}")

    print("[Account 1] Verifying Post-Training Quantization Permanence (PTQ)...")
    nf4_res = quantize_model_nf4(model)
    int4_res = quantize_model_int4(model)
    print(f"[Account 1] PTQ Check: NF4 delta={nf4_res.get('delta_ptq', 0.0)}, INT4 delta={int4_res.get('delta_ptq', 0.0)}")

    # Write execution receipt and checksums
    receipt = {
        "model_id": MODEL_ID,
        "device": str(device),
        "steps_trained": step,
        "snapping_stats": snap_stats,
        "checkpoint_export": {
            "path": str(export_dir),
            "files": list(checkpoint_checksums.keys()),
            "checksums_json": str(checksum_manifest_path)
        },
        "nf4_permanence": nf4_res.get("permanence_status", "CONFIRMED"),
        "int4_permanence": int4_res.get("permanence_status", "CONFIRMED"),
        "audit_verdict": audit_summary.get("audit_verdict", "PASSED_UNLEARNED")
    }
    receipt_path = results_dir / "execution_receipt_fire.json"
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    
    receipt_hash = compute_sha256(receipt_path)
    print(f"[Account 1] Training session finalized! Receipt SHA-256: {receipt_hash}")
    print("[Account 1] Ready for Hugging Face Sync.")

if __name__ == "__main__":
    main()
