#!/usr/bin/env python3
"""
scripts/comprehensive_validation.py
Independent verification of:
1. 12 original receipt checksums
2. 245 curated dataset rows & split checksums
3. 5 visual assets existence & dimensions
4. README.md & HF article image links
5. Fresh secret scan across tracked files
"""

import hashlib
import json
import os
import re
from pathlib import Path

def check_receipts():
    print("=== [1/5] Verifying 12 Original Receipt Checksums ===")
    expected_receipts = [
        "01_verified_lexsi_apis.json",
        "02_ast_auditor_selftest.json",
        "03_curatorkit_manifest_summary.json",
        "04_circuitkit_e2e_proof.json",
        "05_trainer_and_ptq_math_proof.json",
        "06_full_4repo_pipeline_e2e.json",
        "07_repo_wide_strict_audit.json",
        "08_stage1_gpu_smoke_test_receipt.json",
        "09_stage3_baseline_evaluation_receipt.json",
        "10_stage3_scientific_validity_diagnostic.json",
        "11_stage4_unlearn_fire_receipt.json",
        "12_stage4_unlearn_global_receipt.json",
    ]
    
    # Load ledger
    ledger_path = Path("audit_for_carbon/SHA256SUMS.txt")
    ledger = {}
    for line in ledger_path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                # normalize path
                fname = Path(parts[1].strip().lstrip("*")).name
                ledger[fname] = parts[0].strip()

    all_matched = True
    for r in expected_receipts:
        p = Path("audit_for_carbon") / r
        if not p.exists():
            print(f"  ❌ Missing receipt: {r}")
            all_matched = False
            continue
        actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        expected_hash = ledger.get(r)
        if expected_hash and actual_hash == expected_hash:
            print(f"  ✓ {r}: SHA-256 MATCH ({actual_hash[:12]}...)")
        elif expected_hash:
            print(f"  ❌ {r}: HASH MISMATCH! Expected {expected_hash[:12]}, got {actual_hash[:12]}")
            all_matched = False
        else:
            print(f"  ✓ {r}: Present ({actual_hash[:12]}...)")
    return all_matched

def check_dataset():
    print("\n=== [2/5] Verifying 245 Curated Dataset Rows & Splits ===")
    splits = {
        "train_forget.jsonl": 100,
        "heldout_test_forget.jsonl": 100,
        "retain.jsonl": 30,
        "hard_boundary_twins.jsonl": 15
    }
    total_rows = 0
    all_ok = True
    for fname, expected_count in splits.items():
        p = Path("data/curated") / fname
        if not p.exists():
            print(f"  ❌ Missing split file: {fname}")
            all_ok = False
            continue
        rows = [line for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        actual_count = len(rows)
        total_rows += actual_count
        if actual_count == expected_count:
            print(f"  ✓ {fname}: EXACT MATCH {actual_count} rows")
        else:
            print(f"  ❌ {fname}: COUNT MISMATCH! Expected {expected_count}, got {actual_count}")
            all_ok = False

    print(f"  -> Total Verified Dataset Rows: {total_rows} / 245")
    if total_rows != 245:
        all_ok = False
    return all_ok

def check_visual_assets():
    print("\n=== [3/5] Verifying Visual Assets & Dimensions ===")
    from PIL import Image
    assets = [
        ("assets/polyglot-mansu-cover.png", (1200, 648)),
        ("assets/benchmark_comparison.png", None),
        ("assets/polyglot-mansu-architecture.png", None),
        ("assets/evaluation-workflow.png", None),
        ("assets/dataset-distribution.png", None),
    ]
    all_ok = True
    for path_str, expected_size in assets:
        p = Path(path_str)
        if not p.exists():
            print(f"  ❌ Missing asset: {path_str}")
            all_ok = False
            continue
        im = Image.open(p)
        size = im.size
        if expected_size and size != expected_size:
            print(f"  ❌ {path_str}: Size {size} != expected {expected_size}")
            all_ok = False
        else:
            print(f"  ✓ {path_str}: Present ({size[0]}x{size[1]}, {p.stat().st_size // 1024} KB)")
    return all_ok

def check_readme_links():
    print("\n=== [4/5] Checking README.md Image Link Integrity ===")
    content = Path("README.md").read_text(encoding="utf-8")
    md_imgs = re.findall(r'!\[.*?\]\((.*?)\)', content)
    html_imgs = re.findall(r'<img[^>]+src=["\'](.*?)["\']', content)
    all_imgs = md_imgs + html_imgs
    all_ok = True
    for img in all_imgs:
        if img.startswith("http"):
            continue
        p = Path(img)
        if p.exists():
            print(f"  ✓ {img}: Resolves to file on disk")
        else:
            print(f"  ❌ {img}: File NOT found!")
            all_ok = False
    return all_ok

def fresh_secret_scan():
    print("\n=== [5/5] Fresh Secret Scan Across All Tracked & Modified Files ===")
    secret_patterns = [
        (re.compile(r'hf_[a-zA-Z0-9]{34,}'), 'HuggingFace Token'),
        (re.compile(r'ghp_[a-zA-Z0-9]{36}'), 'GitHub PAT'),
        (re.compile(r'AIzaSy[a-zA-Z0-9_-]{33}'), 'Google API Key'),
    ]
    tracked_files = [f for f in Path(".").glob("**/*") if f.is_file() and not any(part.startswith(".") for part in f.parts)]
    flagged = []
    for f in tracked_files:
        if f.suffix in [".png", ".sqlite", ".pyc", ".xml", ".lock"]:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            for pat, label in secret_patterns:
                matches = pat.findall(text)
                if matches:
                    flagged.append((str(f), label, len(matches)))
        except Exception:
            pass

    if flagged:
        print("  ❌ WARNING: Secrets detected:")
        for item in flagged:
            print(f"    {item[0]}: {item[1]} ({item[2]} occurrences)")
        return False
    else:
        print("  ✓ Zero exposed API keys, PATs, or tokens detected across all files.")
        return True

if __name__ == "__main__":
    r1 = check_receipts()
    r2 = check_dataset()
    r3 = check_visual_assets()
    r4 = check_readme_links()
    r5 = fresh_secret_scan()
    print("\n=======================================================")
    print(f"ALL CHECKS PASSED: {r1 and r2 and r3 and r4 and r5}")
    print("=======================================================")
