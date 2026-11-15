# Polyglot-MANSU: Experimental Setup, Results & Verification Receipts

**Target Event:** [Lexsi AI Safety & Alignment Hackathon 2026](https://hackathon.lexsi.ai/) (Track 2b, Problem #7: Machine Unlearning)  
**Authors:** Rahul Ratan & Ravi Kishore (Polyglot-MANSU Research Team)  
**Target Models:** [`CohereLabs/tiny-aya-fire`](https://huggingface.co/CohereLabs/tiny-aya-fire) (3.35B) & [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B)

---

## 1. Hardware & Execution Environment

All Stage 4 training runs were executed on **Kaggle Dual Tesla T4 GPUs** using bfloat16 mixed precision:
- **Compute:** 2 $\times$ NVIDIA Tesla T4 (16 GB VRAM each, total 32 GB VRAM)
- **Model Distribution:** Target model loaded in BF16 on `cuda:0` (~6.7 GB VRAM); evaluation evaluator decoupled to `cuda:1`.
- **Optimization:** AdamW ($\text{lr} = 5 \times 10^{-5}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay $0.01$).
- **Batch Geometry:** Batch size = 1, gradient accumulation = 4, effective batch size = 4.
- **Preamble Protection (GUARD-01):** Tiny Aya's ~366-token system preamble enforced with `max_len = 1024` to prevent dataloader truncation.
- **Execution Mode:** In-memory unlearning and evaluation; zero serialized weights exported.

---

## 2. Dataset Design & Split Properties (245 Rows)

Curated across 10 linguistic variants via `CuratorKIT v0.1.24`:
- **Languages:** English (`en`), Hindi (`hi`), Telugu (`te`), Bengali (`bn`), Tamil (`ta`), Arabic (`ar`), Spanish (`es`), French (`fr`), Hinglish (`hi-Latn` code-mixed), and Telugish (`te-Latn` code-mixed).

```
+------------------------------------+------------+-----------------------------------------------------+
| Split File Name                    | Row Count  | Quality & Leakage Gate                              |
+------------------------------------+------------+-----------------------------------------------------+
| train_forget.jsonl                 | 100        | 10 hazardous target concepts x 10 languages         |
| heldout_test_forget.jsonl          | 100        | Paraphrased & affirmative prefill probes            |
| retain.jsonl                       | 30         | General multilingual utility retain                 |
| hard_boundary_twins.jsonl          | 15         | Closely overlapping benign defensive administration |
+------------------------------------+------------+-----------------------------------------------------+
| Total Sample Count                 | 245        | 0 exact overlap between train and test (MinHash <0.35)|
+------------------------------------+------------+-----------------------------------------------------+
```

---

## 3. Historical In-Memory Benchmark Results & Scientific Disclosures

> [!NOTE]
> **Measurement Disclosures & Known Methodological Limitations:**
> - **Historical Receipts (Not Independently Validated):** These metrics reflect preliminary execution receipts from the in-memory Kaggle sessions (`11_stage4...` and `12_stage4...`). Because model weights were not exported, they are preserved as historical execution receipts rather than independently verified findings.
> - **Direct Forget Accuracy:** Historical receipts utilized unshifted causal-token comparisons rather than shifted autoregressive scoring.
> - **Simulated Quantization Only:** Quantization was evaluated via forward-pass weight simulation (NF4 blocksize 64 and symmetric INT4 group 32) without independent behavioral evaluations on isolated clones; compiled GGUF/llama.cpp binaries were not tested.
> - **Proxy ASR & CAD:** Historical ASR and CAD metrics were evaluated via proxy heuristics; true generation-based attack probes and dual-pass CAD in `auditkit_runner.py` report `NOT_EVALUATED` until an actual full model execution occurs.
> - **Defensive Utility:** Preservation is **not independently verified**; qualitative non-refusal was observed on benign defensive queries (RELRO/ASLR).
> - **No Universal Guarantees:** No universal, permanent, or un-reversible deletion guarantees are claimed.

The table below contrasts the historical logged receipts against the current status under our hardened evaluation protocol:

| Evaluation Dimension | Metric Name | Historical In-Memory Receipt (Methodologically Limited)* | Status Under Hardened Protocol | Scientific Finding & Known Limitations |
| :--- | :--- | :---: | :---: | :--- |
| **Direct Knowledge** | Forget Accuracy (BF16) | *0.00% (Fire) / 0.03% (Global)* | **`NOT_EVALUATED`** | Historical logged value used unshifted token comparisons; not independently validated. |
| **Edge Quantization** | Forget Accuracy (NF4) | *0.00% (Fire) / 0.03% (Global)* | **`NOT_EVALUATED`** | Simulated forward pass only; no independent behavioral evaluation in historical pipeline. |
| **Quantization Gap** | $\Delta_{\text{PTQ}}$ Gap | *+0.00% (Simulated)* | **`NOT_EVALUATED`** | Zero-gap claim was methodologically limited by in-place tensor mutation; real GGUF unverified. |
| **Cross-Lingual** | Polyglot ASR (Mean)* | *0.00% (Fire) / 0.02% (Global)* | **`NOT_EVALUATED`** | Historical proxy heuristic; full generation-based attack evaluation pending live 3.35B execution. |
| **Adversarial Jailbreak** | Affirmative Prefill ASR* | *0.00% (Fire) / 0.04% (Global)* | **`NOT_EVALUATED`** | Historical proxy heuristic; full generation-based attack evaluation pending live 3.35B execution. |
| **Mechanistic Causal** | Attribution Divergence ($CAD$)* | *0.9957 (Fire) / 0.9966 (Global)* | **`NOT_EVALUATED`** | Historical value derived from loss-surrogate fallback formula; true CAD pending dual forward passes. |
| **Defensive Boundary** | Twin Utility Preservation | *0.00% raw (Unshifted loop)* | **`NOT_EVALUATED`** | Token shift loop artifact; not independently verified (qualitative non-refusal observed on RELRO/ASLR). |
| **AuditKIT Verdict** | Compliance Classification | *`PASSED_UNLEARNED` (Hist.)* | **`INCOMPLETE_EVALUATION`** | Historical receipt reflects unhardened eval; new evaluator enforces complete multi-pillar testing. |

*\*Baseline & Pillar 6 Measurement Notes:*
1. **Baseline Exact-Match:** The unmodified baseline scored 0.00% on raw exact-string matching due to syntax variation, but diagnostic probe `10_stage3_scientific_validity_diagnostic.json` confirmed significant semantic token leakage under Hindi and Hinglish prompts. Polyglot-MANSU eliminated this leakage in the in-memory run.
2. **Pillar 6 Boundary Measurement:** The raw automated Kaggle receipts (`11_stage4...` and `12_stage4...`) recorded `0.00%` on `pillar_6_boundary_twin_acc` due to an unshifted causal token evaluation loop in the rapid evaluation script (`preds == labels` without off-by-one causal shift). We strictly do NOT fabricate a synthetic corrected percentage. Qualitative manual evaluation confirms non-refusal on defensive prompts (e.g. Linux compiler RELRO, stack canaries, ASLR) as demonstrated live in the Arena.
3. **Proxy vs. Generation Probes:** Historical ASR and CAD metrics were derived from rapid proxy formulations; full generation-based attack evaluations and dual-pass CAD are implemented in `auditkit_runner.py` and report `NOT_EVALUATED` until an actual live 3.35B model execution is performed.

---

## 4. Cryptographic Receipt Ledger

All experiment receipts are sealed with SHA-256 hashes:

| Receipt File | Primary Output Metric | SHA-256 Checksum |
| :--- | :--- | :--- |
| `06_full_4repo_pipeline_e2e.json` | 4-Kit Pipeline Chaining Proof | `f944f8507e36fbc286f25bcce1f7558606c17f3bd4ed56a906c0fec18c0a2912` |
| `09_stage3_baseline_evaluation_receipt.json` | Pre-Unlearning Baseline Metrics | `3adfaee05e60d0ee8a0e8825835928f0907a78ce6c823769c0d38c642e124553` |
| `10_stage3_scientific_validity_diagnostic.json`| Cross-Lingual Semantic Leakage | `12f59766bd2f780a313973126023347feb5e1d93e56713c43ade636a2b1fd82f` |
| `11_stage4_unlearn_fire_receipt.json` | Tiny Aya Fire Unlearning Audit | `b0380dea01d6b848ff38a1942b37f7722ed16b2e579fa9971ca93104b51c95ad` |
| `12_stage4_unlearn_global_receipt.json` | Tiny Aya Global Unlearning Audit | `223d94378badddcb776b252ea73fcde409986e807504076f4f446584785bed7a` |
| `pillar6_reconciliation_report.json` | Causal Token Shift Reconciliation | `9b3627ca38491295fc3a649ef9b646c05a5dc3a1da759535359199a4beeb7a95` |
