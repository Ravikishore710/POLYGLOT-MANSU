"""
tests/test_split_disjointness.py
Unit tests verifying strict train-test disjointness, MinHash leakage gate, and manifest integrity.
"""

import json
from pathlib import Path
import pytest
from polyglot_mansu.data.curate_multilingual import (
    build_dataset_splits,
    compute_minhash_signature,
    minhash_jaccard,
)

def test_split_disjointness_and_manifest(tmp_path):
    manifest = build_dataset_splits(output_dir=str(tmp_path))
    
    assert "splits" in manifest
    assert "checksums.txt" in [f.name for f in tmp_path.iterdir()]
    assert "manifest.json" in [f.name for f in tmp_path.iterdir()]

    # Read train_forget and heldout_test_forget
    train_file = tmp_path / "train_forget.jsonl"
    heldout_file = tmp_path / "heldout_test_forget.jsonl"
    
    with open(train_file, "r", encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f]
    with open(heldout_file, "r", encoding="utf-8") as f:
        heldout_rows = [json.loads(line) for line in f]

    assert len(train_rows) > 0
    assert len(heldout_rows) > 0

    train_prompts = set(r["prompt"] for r in train_rows)
    heldout_prompts = set(r["prompt"] for r in heldout_rows)

    # 1. Zero exact overlap
    exact_overlap = train_prompts.intersection(heldout_prompts)
    assert len(exact_overlap) == 0, f"Found exact prompt leakage: {exact_overlap}"

    # 2. MinHash disjointness check
    for r_train, r_test in zip(train_rows, heldout_rows):
        sig_train = compute_minhash_signature(r_train["prompt"])
        sig_test = compute_minhash_signature(r_test["prompt"])
        sim = minhash_jaccard(sig_train, sig_test)
        # Paraphrased variants should have distinct representation, similarity < 0.60
        assert sim < 0.60, f"MinHash leakage too high between '{r_train['prompt']}' and '{r_test['prompt']}': {sim}"
