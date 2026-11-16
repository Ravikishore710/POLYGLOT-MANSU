# STAGE A.2 — ACTUAL CHECKPOINT VERIFICATION & FORENSIC AUDIT REPORT

**Date:** November 16, 2026  
**Project:** Polyglot-MANSU (Lexsi AI Safety & Alignment Hackathon, Track 2b)  
**Target Architectures:** `CohereLabs/tiny-aya-fire` (3.35B) & `CohereLabs/tiny-aya-global` (3.35B)  
**Verification Level:** Forensic Zero-Tolerance Artifact Audit  

---

## 1. Executive Summary & Decision Matrix

In accordance with strict forensic guidelines (Law 2: Zero tolerance for mock data or unverified assumptions), a comprehensive search of all local workspace paths, Kaggle cloud kernel output bundles, and Hugging Face repositories was conducted to establish whether genuine serialized post-unlearning weight checkpoints exist.

| Model Target | Checkpoint Found | Authenticity Verified | Reload Successful | Training Provenance Verified | Final Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tiny Aya Fire (3.35B)** | **NO** | **NO** | **NO** | **NO (Artifacts Absent)** | **NOT READY (Retraining Required)** |
| **Tiny Aya Global (3.35B)** | **NO** | **NO** | **NO** | **NO (Artifacts Absent)** | **NOT READY (Retraining Required)** |

---

## 2. Forensic Investigation of Candidate Locations

Every potential storage location across the project, cloud environments, and model hubs was systematically inspected:

### A. Local Workspace Storage
* **`models/tiny_aya_fire_mansu_weights/`:** Directory exists (created during Stage A directory preparation), but contains **0 bytes** (empty).
* **`models/tiny_aya_global_mansu_weights/`:** Directory exists, but contains **0 bytes** (empty).
* **`results/` subdirectories (`kaggle_fire_completed`, `kaggle_global_completed`, `diagnostic_success`, `smoke_test_success`):** Contain execution logs (`*.log`), configuration YAMLs (`*.yaml`), SQLite audit databases (`*.sqlite`), and evaluation JSON summaries (`*.json`), but **zero model weight files** (`.safetensors`, `.bin`, `.pt`, `.pth`).
* **Root Workspace Search:** A recursive search for all model weight extensions (`*.safetensors`, `*.bin`, `*.pt`, `*.pth`) across the repository (excluding `.git` and `.venv`) returned **zero matches**.
* **Local Hugging Face Cache (`~/.cache/huggingface/hub/`):** Contains only tokenizer and metadata configurations (`tokenizer.json`, `config.json`) for `CohereLabs/tiny-aya-fire`. Zero model weight shards exist in local cache.

### B. Kaggle Cloud Kernel Outputs
* **Account 1 Kernel (`rahulratan12/polyglot-mansu-fire-run`):**
  - Downloaded and inspected via `kaggle kernels output rahulratan12/polyglot-mansu-fire-run -p scratch/check_kaggle_output`.
  - Contents: `polyglot-mansu-fire-run.log`, `results/audit/execution_receipt_fire.json`, `results/audit/tiny_aya_fire_mansu_audit_summary.json`, `results/audit/experiment_db.sqlite`, `data/curated/`, `configs/`.
  - **Model weight shards present:** **NONE (0 bytes)**.
* **Account 2 Kernel (`rahulratan12/polyglot-mansu-global-run`):**
  - Downloaded and inspected via `kaggle kernels output rahulratan12/polyglot-mansu-global-run -p scratch/check_kaggle_global_output`.
  - Contents: `results/audit/execution_receipt_global.json`, `results/audit/experiment_db.sqlite/tiny_aya_global_mansu.json`, `data/curated/`, `configs/`.
  - **Model weight shards present:** **NONE (0 bytes)**.

### C. Hugging Face Hub Repositories
* **Repository `Ravikishore710/polyglot-mansu-tiny-aya-fire`:**
  - Inspected via `HfApi().list_repo_files("Ravikishore710/polyglot-mansu-tiny-aya-fire")`.
  - Files present: `README.md`, `LICENSE`, `ATTRIBUTION.md`, `polyglot_mansu_fire.yaml`, JSON evaluation receipts (`11_stage4_unlearn_fire_receipt.json`, etc.), and visualization PNGs (`benchmark_comparison.png`, etc.).
  - **Model weight shards present:** **NONE (0 bytes)**.
* **Repository `Ravikishore710/polyglot-mansu-tiny-aya-global`:**
  - Inspected via `HfApi().list_models(author="Ravikishore710")`.
  - **Repository does not exist on Hugging Face.**

---

## 3. Training Provenance & Root Cause Analysis

### Verified Historical Facts:
1. **In-Memory Training Did Occur:** Historical execution logs (`scratch/check_kaggle_output/polyglot-mansu-fire-run.log` and `audit_for_carbon/11_stage4_unlearn_fire_receipt.json`) prove that on November 13, 2026, the PolyglotMANSUTrainer executed 90 unlearning steps across 3 epochs on Kaggle Dual Tesla T4 GPUs over `data/curated/train_forget.jsonl` (100 rows) and `data/curated/retain.jsonl` (30 rows).
2. **In-Memory Snapping and Audit Ran:** Snapping statistics (`snap_stats = {'snapped_count': 1, 'status': 'CONSOLIDATED'}` for Fire; `{'snapped_count': 0}` for Global) and in-memory evaluations were recorded into JSON summaries.

### The Missing Step (Root Cause):
* In the original training script (`notebooks/kaggle_account1_fire.py` and `kaggle_account2_global.py` as executed on Nov 13), lines 237–241 transitioned directly from `trainer.post_train_quant_snapping()` to `run_6pillar_audit()` without invoking:
  ```python
  model.save_pretrained(export_dir, safe_serialization=True)
  tokenizer.save_pretrained(export_dir)
  ```
* Because the unlearned model was held **strictly in volatile GPU VRAM**, the moment the Kaggle kernel session terminated and the VM was deallocated, the unlearned weights were discarded.
* This was explicitly confirmed by `scripts/reconcile_pillar6_evaluation.py` on Nov 13:
  `"model_artifact_status": "IN_MEMORY_KAGGLE_ONLY (No serialized weights exported to disk)"`.

---

## 4. Checkpoint Authenticity & Reload Testing Status

* **Fire Post-Unlearning Weights:** Cannot be authenticated or reloaded because no serialized files exist on disk or in cloud object storage.
* **Global Post-Unlearning Weights:** Cannot be authenticated or reloaded because no serialized files exist on disk or in cloud object storage.
* **Scratch/Mock Checkpoints:** The miniature GPT-2 checkpoint generated during Stage A in `scratch/test_checkpoint_export.py` was verified and subsequently cleaned up; it was never represented as a 3.35B model.

---

## 5. Minimum Safe Retraining & Evaluation Plan (A100 Execution)

Because neither unlearned checkpoint exists in persistent storage, **an actual unlearning run must be executed to produce and serialize the true 3.35B checkpoints**. 

We have already completed all preparatory code hardening in Stage A:
1. `model.save_pretrained(export_dir, safe_serialization=True)` and `tokenizer.save_pretrained(export_dir)` are positioned immediately post-snapping and pre-PTQ in [`scripts/run_a100_stage_b.py`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/scripts/run_a100_stage_b.py).
2. SHA-256 streaming manifest calculation covers every exported SafeTensors shard and config file.
3. The evaluation suite is upgraded to `run_7pillar_audit()` with $95\%$ confidence intervals and per-language metrics.

### Proposed Single-Pass Execution Protocol (NVIDIA A100 80GB):

1. **Step 1: Hardware Smoke Test (< 1 min, $0.05):**
   Execute [`scripts/smoke_test_a100_bf16.py`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/scripts/smoke_test_a100_bf16.py) to certify native BF16 Tensor Core GEMM and autograd functionality.
2. **Step 2: Tiny Aya Fire Unlearning & Checkpoint Serialization (~5 mins, $0.26):**
   - Execute 90 steps of Polyglot-MANSU on `CohereLabs/tiny-aya-fire` in native BF16.
   - Apply multi-grid dead-zone snapping.
   - **Immediately export true unlearned BF16 weights** to `models/tiny_aya_fire_mansu_weights/` (`model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`, `config.json`, tokenizer, and `checksums.json`).
   - Execute the hardened 7-Pillar AuditKIT suite.
3. **Step 3: Tiny Aya Global Unlearning & Checkpoint Serialization (~5 mins, $0.26):**
   - Execute 90 steps of Polyglot-MANSU on `CohereLabs/tiny-aya-global` in native BF16.
   - Apply multi-grid dead-zone snapping.
   - **Immediately export true unlearned BF16 weights** to `models/tiny_aya_global_mansu_weights/`.
   - Execute the hardened 7-Pillar AuditKIT suite.
4. **Step 4: Immediate Deprovisioning (< 1 min):**
   - Revert Space hardware to `CPU_BASIC` to terminate all GPU billing immediately.
   - **Total Estimated Cost:** $\approx \$0.79$ (Well within the $20.00 ceiling).
5. **Step 5: Post-Execution Verification:**
   - Run a reload test on the saved SafeTensors shards to verify state integrity and compute final SHA-256 hashes.

---

## 6. Strict Boundary Compliance Confirmation

* **GPU Execution Incurred:** 0 GPU seconds.
* **Paid Credits Consumed:** $0.00.
* **Repositories Pushed:** 0 commits to GitHub; 0 commits to Hugging Face.
* **Historical Receipts:** Untouched and intact in `audit_for_carbon/`.
