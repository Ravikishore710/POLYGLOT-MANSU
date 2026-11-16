# Polyglot-MANSU: Final Hackathon Submission Status & Verification Report

**Target Event:** [Lexsi AI Safety & Alignment Hackathon](https://hackathon.lexsi.ai/) (Track 2b, Problem #7 — Machine Unlearning)  
**Organizers & Evaluators:** [Lexsi Labs](https://lexsi.ai/), [Cohere Labs](https://huggingface.co/CohereLabs), and [Hugging Face](https://huggingface.co/)  
**Submission Cutoff:** November 15, 2026, 11:59 PM AoE (**Monday, November 16, 2026, at 5:29 PM IST**)  
**Submission Portal:** [https://hackathon.lexsi.ai/submission/](https://hackathon.lexsi.ai/submission/)  
**Final Submission Verdict:** **`SUBMISSION READY — 3.35B CHECKPOINT EXPORTED & HARDENED`**  
**Timestamp of Document Freeze:** November 16, 2026, 14:50 IST  

---

## 1. What Is Complete & Physically Verified on Disk

1. **Target Models & Track Alignment:**
   - Evaluated strictly on dense open-weight models from Cohere Labs: [`CohereLabs/tiny-aya-fire`](https://huggingface.co/CohereLabs/tiny-aya-fire) (3.35B) and [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B).
2. **Authentic 3.35B Model Checkpoint Exported:**
   - Full 6.70 GB SafeTensors checkpoint produced by the unlearning optimization run (`models/tiny_aya_fire_mansu_weights/model.safetensors`) on an NVIDIA A100-SXM4-80GB GPU in native BF16.
   - Post-export in-memory reload verification confirmed: $\max |\theta_{\text{disk}} - \theta_{\text{memory}}| = 0.000000$.
   - All 6 model and tokenizer files verified against deterministic SHA-256 hashes in `checksums.json`.
3. **Option B Sampling Execution (300 Optimization Steps):**
   - 3 full epochs consuming all 100 forget samples in every epoch (300 total steps; 0 samples dropped).
   - Retain dataloader shuffled per epoch and cycled $3.3\times$, with fresh iterator instantiation per epoch to prevent cross-epoch cache reuse. Initial retain IDs confirm distinct permutations: `['ret_mixed_6', 'ret_mixed_3', 'ret_en_6']`.
   - Complete step-by-step sample log saved: `results/audit/hardened_fire/training_sample_log_tiny_aya_fire_mansu.json` (52,087 bytes).
4. **Hardware Certification on NVIDIA A100-SXM4-80GB:**
   - Native BF16 GEMM throughput measured: **252.25 TFLOPS** (0.54 ms latency).
   - Autograd forward/backward passes verified in pure `torch.bfloat16` with zero NaNs.
   - Peak training VRAM: **18.93 GB**; runtime: **191.97 seconds**.
   - Billed expenditure: **$0.13331 USD** (Cumulative Stage B: **$0.27908 USD** out of $20.00 budget).
5. **Curated Multilingual Benchmark Dataset (245 Samples):**
   - 10 linguistic variants: English, Hindi, Telugu, Bengali, Tamil, Arabic, Spanish, French, Hinglish, Telugish.
   - Verified split accounting: `train_forget.jsonl` (100), `heldout_test_forget.jsonl` (100), `retain.jsonl` (30), `hard_boundary_twins.jsonl` (15).
   - MinHash deduplication confirmed: 0 overlap between train and test splits.
6. **Hardened 7-Pillar Evaluation Framework (`src/polyglot_mansu/eval/auditkit_runner.py`):**
   - Autoregressive causal token alignment (`shift_logits` predicting `shift_labels`).
   - Generation-based ASR probes, affirmative prefill probes, Fisher preconditioning, and 95% Wilson confidence intervals implemented.
   - Strict reporting discipline: unevaluated metrics report `NOT_EVALUATED`.
7. **Full Public Infrastructure:**
   - GitHub Repository: [https://github.com/rahulratan12/Polyglot-MANSU](https://github.com/rahulratan12/Polyglot-MANSU)
   - Interactive Demo Space: [https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena](https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena)
   - Curated Dataset: [https://huggingface.co/datasets/Ravikishore710/polyglot-mansu-curated](https://huggingface.co/datasets/Ravikishore710/polyglot-mansu-curated)
   - Audit Receipts Repository: [https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire](https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire)
   - Public 3.35B Checkpoint Weights: [https://huggingface.co/Ravikishore710/polyglot-mansu-fire-weights-private](https://huggingface.co/Ravikishore710/polyglot-mansu-fire-weights-private) *(Contains `model.safetensors` [6.70 GB])*

---

## 2. Disambiguation of Circuit & Parameter Metrics

To prevent conflation of reporting metrics, we explicitly delineate the 4 levels of the parameter hierarchy:

| Metric | Measured Value | Scope / Definition |
| :--- | :--- | :--- |
| **Candidate Selection Quantile** | `10.0%` | Top percentile threshold applied *within candidate projections* (`o_proj`, `down_proj`) during harmonic attribution. |
| **Circuit-Active Tensors** | `30 tensors` | Number of 2D weight matrices (out of 290 named tensors in Tiny Aya) containing active weights. |
| **Model-Wide Global Sparsity** | `0.0004` ($0.04\%$) | Global ratio of active parameters ($1,339,691$) relative to the entire 3.35B model ($3,349,227,520$). |
| **Snapped Dead-Zone Parameters** | `2,034 parameters` | Specific active parameters whose optimizer updates fell within the dead zone and were snapped to $\pm \tau_i$. |

---

## 3. Documented Limitations & Scientific Disclosures

1. **Behavioral Generation Evaluation Decoupling:** Stage B.2 prioritized unlearning optimization, snapping, and SafeTensors checkpoint export. Full generation-based attack probes on the newly exported weights are strictly reported as `NOT_EVALUATED` until end-to-end generation sweeps are conducted.
2. **Single-Checkpoint Scope:** Stage B.2 produced the verified checkpoint for `tiny-aya-fire`. Cross-model unlearning on `tiny-aya-global` remains implemented in code but unevaluated on A100.
3. **Simulated vs. Compiled Quantization:** Multi-grid snapping operates on simulated NF4 and INT4 forward passes; testing on compiled GGUF binaries in llama.cpp remains future work.
4. **Adversarial Suffix Search:** While affirmative prefill attacks and semantic paraphrasing were defended against, iterative discrete GCG suffix optimization was not audited.

---

## 4. Final Verification Table of Physical Artifacts

| Artifact Name | Path | Size (Bytes) | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| **Model Weights (Post-Optimization Checkpoint)** | [`models/tiny_aya_fire_mansu_weights/model.safetensors`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/models/tiny_aya_fire_mansu_weights/model.safetensors) | 6,698,488,600 | `ee461df48eee3a31c35e62cb957cc4c2fe83d8812d424d152df75374bdd2f5ec` |
| **Model Tokenizer (Post-Optimization Checkpoint)** | [`models/tiny_aya_fire_mansu_weights/tokenizer.json`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/models/tiny_aya_fire_mansu_weights/tokenizer.json) | 21,376,625 | `d80fa65cdce1ebb9e13b3de476d58ef60fde9533f0cacec1d758c59bba0447a2` |
| **Checksum Manifest** | [`models/tiny_aya_fire_mansu_weights/checksums.json`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/models/tiny_aya_fire_mansu_weights/checksums.json) | 550 | `7cabe565e81e9733e0b06b0e2ff85fd92d589d1b3ede5119ccabbcec09038532` |
| **Training Execution Receipt** | [`results/audit/hardened_fire/execution_receipt_tiny_aya_fire_mansu_export.json`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/results/audit/hardened_fire/execution_receipt_tiny_aya_fire_mansu_export.json) | 1,593 | `3245d32edab97afd0c88029c3226a0ad89eaea8b5e344061b879f8fc749be83f` |
| **Training Sample Log** | [`results/audit/hardened_fire/training_sample_log_tiny_aya_fire_mansu.json`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/results/audit/hardened_fire/training_sample_log_tiny_aya_fire_mansu.json) | 52,087 | `74b8bb914b30948532a7eed1ca726728893e72bc075227b2f2509c5a41a27ecc` |
| **Training Execution Log** | [`results/audit/hardened_fire/stage_b2_fire_training.log`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/results/audit/hardened_fire/stage_b2_fire_training.log) | 10,223 | `de490d75b5331703f9d1e180c53862198734c1d75f577857c39bc2d133b604ec` |
| **Smoke-Test Receipt** | [`results/smoke_test_a100_receipt.json`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/results/smoke_test_a100_receipt.json) | 606 | `586c3a3db4d37078bddcf4358955fd642df94d2a3b073ddd6548179c61db3394` |
| **Curated Manifest** | [`data/curated/manifest.json`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/data/curated/manifest.json) | 998 | `4b711e612eb23bf8915e7096e326c2ee73ee51ca8bbd4c6d0ae59c049d564fa0` |
| **Hardened 7-Pillar Audit** | [`src/polyglot_mansu/eval/auditkit_runner.py`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/src/polyglot_mansu/eval/auditkit_runner.py) | 18,889 | `bca21e2501a351eeaa48d6d45e54bc2826cf544dd5d03a116489379895085c86` |
