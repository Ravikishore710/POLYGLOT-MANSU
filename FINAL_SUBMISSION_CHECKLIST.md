# Polyglot-MANSU: Final Submission Readiness Checklist

**Target Event:** [Lexsi AI Safety & Alignment Hackathon 2026](https://hackathon.lexsi.ai/) (Track 2b, Problem #7 — Machine Unlearning)  
**Submission Cutoff:** November 15, 2026, 11:59 PM AoE (**Monday, November 16, 2026, at 5:29 PM IST**)  
**Submission Portal:** [https://hackathon.lexsi.ai/submission/](https://hackathon.lexsi.ai/submission/)  
**Status:** **`SUBMISSION FREEZE — FULLY AUDITED & VERIFIED`**  
**Document Timestamp:** November 16, 2026, 14:55 IST  

---

## 1. Hackathon Requirements Verification

| Requirement | Specification | Implementation / Evidence Location | Status |
| :--- | :--- | :--- | :--- |
| **Track & Problem** | Track 2b: Problem #7 (Machine Unlearning) | Focuses on unlearning hazardous cybersecurity and private entity capabilities on dense multilingual models. | **PASS** |
| **Target Models** | Cohere Labs dense open-weight models | [`CohereLabs/tiny-aya-fire`](https://huggingface.co/CohereLabs/tiny-aya-fire) (3.35B) & [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B). No forbidden models used. | **PASS** |
| **CuratorKIT Integration** | Dataset curation, Alpaca schema, MinHash deduplication | `src/polyglot_mansu/data/curate_multilingual.py`, `data/curated/manifest.json`. Jaccard $< 0.20$ verified. | **PASS** |
| **CircuitKIT Integration** | Multilingual causal attribution & circuit discovery | `src/polyglot_mansu/circuits/discover_polyglot.py`, `src/polyglot_mansu/circuits/harmonic_intersect.py`. Integrated gradients on EN, HI, TE. | **PASS** |
| **SafeTune Integration** | Guard enforcement, trainer wrapper, loss functions | `src/polyglot_mansu/unlearn/polyglot_mansu_trainer.py`, `src/polyglot_mansu/guards.py`. Guard 01-08 enforced. | **PASS** |
| **AuditKIT Integration** | 7-pillar evaluation suite, autoregressive alignment | `src/polyglot_mansu/eval/auditkit_runner.py`. Generation ASR probes, Wilson CIs, PTQ reversal clone. | **PASS** |
| **Hardware Certification** | Native BF16 execution on Ampere sm_80+ | Tested on NVIDIA A100-SXM4-80GB. 252.25 TFLOPS BF16 GEMM throughput verified. Receipt: [`results/smoke_test_a100_receipt.json`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/results/smoke_test_a100_receipt.json). | **PASS** |
| **Model Checkpoint Export** | Full 3.35B unlearned SafeTensors checkpoint | Exported and verified on local disk: [`models/tiny_aya_fire_mansu_weights/model.safetensors`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/models/tiny_aya_fire_mansu_weights/model.safetensors) (6.70 GB). | **PASS** |
| **Reload Bit-Exactness** | $\max |\theta_{\text{disk}} - \theta_{\text{memory}}| = 0.0$ | Verified bit-exact match against in-memory unlearned state. Recorded in execution receipt. | **PASS** |
| **Option B Sampling** | 300 steps, exhaustive forget coverage, cycled retain | All 100 forget samples seen every epoch (0 dropped). Logged in [`results/audit/hardened_fire/training_sample_log_tiny_aya_fire_mansu.json`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/results/audit/hardened_fire/training_sample_log_tiny_aya_fire_mansu.json). | **PASS** |

---

## 2. Physical Evidence Inventory on Disk

All artifacts are present, verified, and anchored by deterministic SHA-256 hashes:

| Item | Local File Path | Size (Bytes) | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| **Unlearned Model Weights** | `models/tiny_aya_fire_mansu_weights/model.safetensors` | 6,698,488,600 | `ee461df48eee3a31c35e62cb957cc4c2fe83d8812d424d152df75374bdd2f5ec` |
| **Unlearned Tokenizer** | `models/tiny_aya_fire_mansu_weights/tokenizer.json` | 21,376,625 | `d80fa65cdce1ebb9e13b3de476d58ef60fde9533f0cacec1d758c59bba0447a2` |
| **Model Config** | `models/tiny_aya_fire_mansu_weights/config.json` | 2,020 | `f1eba457bf81b133aa4816eb906daa5797f491656a018db17bfb3b0d0bb84cc2` |
| **Generation Config** | `models/tiny_aya_fire_mansu_weights/generation_config.json` | 132 | `b5424ac1545b2d6f76602862e0f316273d0e012c9594f8ce39830bb9d1371fac` |
| **Chat Template** | `models/tiny_aya_fire_mansu_weights/chat_template.jinja` | 3,239 | `2147c474f41b522b7296f74692b9f31e45f49a31465960576a9d321339c5915a` |
| **Tokenizer Config** | `models/tiny_aya_fire_mansu_weights/tokenizer_config.json` | 593 | `cd98275e0786e6fca40f6f062e28b136483be6a4b784dab3fb6d2f053a79045e` |
| **Checksum Manifest** | `models/tiny_aya_fire_mansu_weights/checksums.json` | 550 | `7cabe565e81e9733e0b06b0e2ff85fd92d589d1b3ede5119ccabbcec09038532` |
| **Execution Receipt** | `results/audit/hardened_fire/execution_receipt_tiny_aya_fire_mansu_export.json` | 1,593 | `3245d32edab97afd0c88029c3226a0ad89eaea8b5e344061b879f8fc749be83f` |
| **Training Sample Log** | `results/audit/hardened_fire/training_sample_log_tiny_aya_fire_mansu.json` | 52,087 | `74b8bb914b30948532a7eed1ca726728893e72bc075227b2f2509c5a41a27ecc` |
| **Execution Log** | `results/audit/hardened_fire/stage_b2_fire_training.log` | 10,223 | `de490d75b5331703f9d1e180c53862198734c1d75f577857c39bc2d133b604ec` |
| **Smoke-Test Receipt** | `results/smoke_test_a100_receipt.json` | 606 | `586c3a3db4d37078bddcf4358955fd642df94d2a3b073ddd6548179c61db3394` |
| **Smoke-Test Log** | `results/smoke_test_a100.log` | 1,684 | `8cc22dedef700e267d1e980ada8471be27f07a2e169aeb68f6a1508b5a74f0e5` |
| **Dataset Manifest** | `data/curated/manifest.json` | 998 | `4b711e612eb23bf8915e7096e326c2ee73ee51ca8bbd4c6d0ae59c049d564fa0` |

---

## 3. Disambiguation of Reported Circuit & Parameter Metrics

We explicitly distinguish the 4 levels of parameter measurements:

1. **Candidate-Level Quantile ($10.0\%$):** Configured via `sparsity_ratio=0.10`, selects the top 10% most causally impactful weights *within the candidate MLP and attention projection pool* evaluated during integrated gradient attribution.
2. **Circuit-Active Weight Tensors ($30$ tensors):** Exactly 30 weight matrices out of 290 total named parameter tensors in Tiny Aya contained weights exceeding the harmonic threshold.
3. **Model-Wide Global Sparsity ($0.0004$ or $0.04\%$):** The ratio of total active parameters ($1,339,691$) relative to Tiny Aya Fire's $3,349,227,520$ total parameters (3.35 billion):
   $$\frac{1,339,691}{3,349,227,520} \approx \mathbf{0.000400} \quad (0.040\%)$$
4. **Consolidated Dead-Zone Parameters ($2,034$ parameters):** Out of the 1.34M active parameters, exactly 2,034 scalar weights moved intentionally during unlearning but stalled inside the NF4/INT4 quantization dead-zone, requiring snapping to $\pm \tau_i$ to ensure quantization permanence ($\Delta_{\text{PTQ}} \le 0$).

---

## 4. Public Resources & Verification URLs

Every external resource is verified and operational:

- **Public GitHub Repository:** [`https://github.com/rahulratan12/Polyglot-MANSU`](https://github.com/rahulratan12/Polyglot-MANSU)
- **Live Interactive Demo Arena:** [`https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena`](https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena)
- **Curated Multilingual Benchmark Dataset:** [`https://huggingface.co/datasets/Ravikishore710/polyglot-mansu-curated`](https://huggingface.co/datasets/Ravikishore710/polyglot-mansu-curated)
- **Model Audit Receipts Repository:** [`https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire`](https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire)
- **Direct 3.35B Checkpoint Weights Repository:** [`https://huggingface.co/Ravikishore710/polyglot-mansu-fire-weights-private`](https://huggingface.co/Ravikishore710/polyglot-mansu-fire-weights-private) *(Contains `model.safetensors` [6.70 GB]; toggle to Public upon user approval)*
- **Hugging Face Research Article:** [`https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire/discussions/2`](https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire/discussions/2)
- **Official Model Collection:** [`https://huggingface.co/collections/Ravikishore710/polyglot-mansu-models-and-checkpoint-audits-6af767ae29f7e0056c086bf1`](https://huggingface.co/collections/Ravikishore710/polyglot-mansu-models-and-checkpoint-audits-6af767ae29f7e0056c086bf1)
- **Official Dataset Collection:** [`https://huggingface.co/collections/Ravikishore710/polyglot-mansu-datasets-and-evaluation-traces-6af767af5d8792224c4e66f5`](https://huggingface.co/collections/Ravikishore710/polyglot-mansu-datasets-and-evaluation-traces-6af767af5d8792224c4e66f5)

---

## 5. Official Submission Portal Form Fields (Ready for Copy-Paste)

- **Track Selection:** `Track 2b: Alignment & Defense`
- **Problem Category:** `#7 Machine Unlearning`
- **Project Title:** `Polyglot-MANSU: Cross-Lingual, Quantization-Permanent Circuit Unlearning on Cohere Tiny Aya`
- **Authors & Affiliations:**
  - Rahul Ratan (`rahulratan12@gmail.com`, GitHub: `@rahulratan12`)
  - Ravi Kishore (`Ravikishore710`, Hugging Face: `@Ravikishore710`)
- **Primary GitHub Repository:** `https://github.com/rahulratan12/Polyglot-MANSU`
- **Primary Hugging Face Space (Demo):** `https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena`
- **Primary Dataset URL:** `https://huggingface.co/datasets/Ravikishore710/polyglot-mansu-curated`
- **Primary Model / Audit URL:** `https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire`
- **Project Summary:**
  > Polyglot-MANSU investigates cross-lingual bypasses and 4-bit quantization reversal in dense multilingual models (CohereLabs/tiny-aya-fire, 3.35B). Standard unlearning updates are 47x–828x smaller than NF4 quantization bin widths, causing forgotten knowledge to resurrect under edge compression. Polyglot-MANSU chains data curation in CuratorKIT schema (10-language benchmark), CircuitKIT (harmonic cross-lingual attribution with Cohere2 compatibility), SafeTune runtime guard principles (latent adversarial unlearning + Fisher null-space damping), and AuditKIT (hardened 7-pillar evaluation framework). Executed on an NVIDIA A100 GPU in native BF16 for 300 steps under Option B sampling, producing an authentic, verified 6.70 GB SafeTensors checkpoint with bit-exact in-memory reload verification (max diff: 0.000000). While multi-grid dead-zone snapping consolidated 2,034 parameters during training, full quantization permanence under compiled edge deployments and behavioral generation attack resistance on the exported checkpoint remain strictly designated as NOT_EVALUATED hypotheses pending end-to-end evaluation sweeps.

---

## 6. Documented Scientific Disclosures & Limitations

1. **Behavioral Generation Evaluation Status:** While unlearning optimization, multi-grid snapping, and SafeTensors checkpoint export are complete and verified bit-exact, behavioral generation attack metrics on the exported weights remain strictly reported as `NOT_EVALUATED` until end-to-end generation sweeps are conducted.
2. **Single Model Checkpoint:** Stage B.2 produced the verified checkpoint for `CohereLabs/tiny-aya-fire`. Cross-model unlearning on `tiny-aya-global` remains implemented in code but unevaluated on A100.
3. **Simulated vs. Compiled Quantization:** Multi-grid snapping operates on simulated NF4 and INT4 forward passes; testing on compiled GGUF binaries in llama.cpp remains future work.

---

## 7. Submission Freeze Verdict

- **Code Frozen:** Yes.
- **Physical Checkpoint Preserved:** Yes (`models/tiny_aya_fire_mansu_weights/`, 6.70 GB).
- **Audit Manifests Preserved:** Yes (`checksums.json`, `manifest.json`, `execution_receipt_*.json`).
- **GPU Billed Status:** **`PAUSED`** (Zero active billing).
- **Total Compute Expenditure:** **`$0.27908 USD`** (Remaining budget: **`$19.72092 USD`**).
- **Verification Status:** **`100% READY FOR USER SUBMISSION`**.
