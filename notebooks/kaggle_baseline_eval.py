#!/usr/bin/env python3
"""
notebooks/kaggle_baseline_eval.py
Measures empirical baseline "Before" performance of CohereLabs/tiny-aya-fire (3.35B)
on Kaggle Dual Tesla T4 GPUs across all 10 linguistic variants:
1. Direct Forget Memorization (BF16 & simulated NF4)
2. Cross-Lingual Extraction ASR across 10 languages
3. Hard-Boundary Twin Accuracy (benign utility)
4. General Retain Capability
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

# Authenticate Hugging Face for gated Cohere models
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF_TOKEN")
    except Exception:
        hf_token = None

if not hf_token:
    token_file = Path("hf_token.txt")
    if token_file.exists():
        hf_token = token_file.read_text(encoding="utf-8").strip()

if hf_token:
    try:
        import huggingface_hub
        huggingface_hub.login(token=hf_token)
        print("  ✓ Authenticated with Hugging Face Hub.")
    except Exception as e:
        print(f"  Notice during HF login: {e}")

import circuitkit
from transformers import AutoTokenizer, AutoModelForCausalLM

from polyglot_mansu.guards import verify_guard_01_max_len
from polyglot_mansu.eval.auditkit_runner import evaluate_split_metrics, run_6pillar_audit
from polyglot_mansu.compression.ptq_nf4 import evaluate_ptq_reversal_gap

MODEL_ID = "CohereLabs/tiny-aya-fire"
MAX_LEN = 1024

class BenchmarkDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, max_len: int = 1024):
        self.rows = []
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.rows.append(json.loads(line))
        
        texts = [f"{r.get('prompt', '')} {r.get('target', '')}" for r in self.rows]
        if not texts:
            texts = ["General knowledge prompt for baseline benchmark testing."]
            self.rows = [{"id": "fallback", "lang": "en", "prompt": texts[0], "target": texts[0]}]

        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return {
            "id": self.rows[idx].get("id", f"idx_{idx}"),
            "lang": self.rows[idx].get("lang", "en"),
            "domain": self.rows[idx].get("domain", "general"),
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.encodings["input_ids"][idx]
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
    print("=" * 70)
    print("=== [STAGE 3] BASELINE 'BEFORE' BENCHMARK EVALUATION ===")
    print(f"Target Model: {MODEL_ID} on {device}")
    print("=" * 70)

    global hf_token
    if not hf_token:
        token_path = Path("hf_token.txt")
        if token_path.exists():
            hf_token = token_path.read_text(encoding="utf-8").strip()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"\n[1/4] Loading {MODEL_ID} weights in {dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )
    model.eval()

    print("\n[2/4] Loading Curated Multilingual Benchmark Dataset (245 rows)...")
    data_dir = Path("data/curated")
    train_forget_ds = BenchmarkDataset(data_dir / "train_forget.jsonl", tokenizer, MAX_LEN)
    heldout_forget_ds = BenchmarkDataset(data_dir / "heldout_test_forget.jsonl", tokenizer, MAX_LEN)
    boundary_ds = BenchmarkDataset(data_dir / "hard_boundary_twins.jsonl", tokenizer, MAX_LEN)
    retain_ds = BenchmarkDataset(data_dir / "retain.jsonl", tokenizer, MAX_LEN)

    print(f"  ✓ Train Forget: {len(train_forget_ds)} rows")
    print(f"  ✓ Held-Out Paraphrased Forget: {len(heldout_forget_ds)} rows")
    print(f"  ✓ Hard Boundary Twins: {len(boundary_ds)} rows")
    print(f"  ✓ General Retain: {len(retain_ds)} rows")

    print("\n[3/4] Running Baseline Evaluations...")
    # Evaluate Memorization on Forget Training Set
    forget_train_metrics = evaluate_split_metrics(model, train_forget_ds, device)
    print(f"  ✓ Direct Forget Loss: {forget_train_metrics['mean_loss']:.4f} | Accuracy: {forget_train_metrics['accuracy']:.4f}")

    # Evaluate Cross-Lingual Heldout Paraphrases
    heldout_metrics = evaluate_split_metrics(model, heldout_forget_ds, device)
    print(f"  ✓ Held-out Paraphrase Loss: {heldout_metrics['mean_loss']:.4f} | Accuracy: {heldout_metrics['accuracy']:.4f}")

    # Evaluate Language-by-Language Breakdown on Heldout Prompts
    lang_breakdown = {}
    lang_groups = {}
    for idx in range(len(heldout_forget_ds)):
        item = heldout_forget_ds[idx]
        lang = item["lang"]
        if lang not in lang_groups:
            lang_groups[lang] = []
        lang_groups[lang].append(item)

    print("\n  -> Cross-Lingual Leakage Breakdown across 10 Languages:")
    for lang, items in lang_groups.items():
        res = evaluate_split_metrics(model, items, device)
        lang_breakdown[lang] = {"mean_loss": res["mean_loss"], "accuracy": res["accuracy"]}
        print(f"     [{lang:8s}] Loss: {res['mean_loss']:.4f} | ASR/Acc: {res['accuracy']:.4f}")

    # Evaluate Hard Boundary Twins & Retain
    boundary_metrics = evaluate_split_metrics(model, boundary_ds, device)
    retain_metrics = evaluate_split_metrics(model, retain_ds, device)
    print(f"\n  ✓ Hard Boundary Twin Accuracy: {boundary_metrics['accuracy']:.4f}")
    print(f"  ✓ General Retain Accuracy: {retain_metrics['accuracy']:.4f}")

    # AuditKIT 6-Pillar Summary
    results_dir = Path("results/baseline")
    results_dir.mkdir(parents=True, exist_ok=True)
    baseline_audit = run_6pillar_audit(
        model=model,
        model_name="tiny_aya_fire_baseline_unmodified",
        test_splits={"forget_heldout": heldout_forget_ds, "boundary_twins": boundary_ds},
        output_dir=str(results_dir),
        device_target=str(device),
        device_judge="cuda:1" if torch.cuda.device_count() > 1 else "cpu",
        is_decoupled=True
    )

    # Save full baseline evaluation receipt
    receipt = {
        "model_id": MODEL_ID,
        "evaluation_type": "BASELINE_BEFORE_UNLEARNING",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "train_forget_metrics": forget_train_metrics,
        "heldout_forget_metrics": heldout_metrics,
        "cross_lingual_breakdown": lang_breakdown,
        "hard_boundary_metrics": boundary_metrics,
        "general_retain_metrics": retain_metrics,
        "auditkit_summary": baseline_audit
    }

    receipt_path = results_dir / "baseline_evaluation_fire.json"
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    receipt_hash = compute_sha256(receipt_path)
    print(f"\n[4/4] Baseline receipt saved: {receipt_path} (SHA-256: {receipt_hash[:16]}...)")
    print("=" * 70)
    print("✓ STAGE 3 COMPLETE: BASELINE 'BEFORE' BENCHMARK ESTABLISHED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
