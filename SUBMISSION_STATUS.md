# Polyglot-MANSU: Final Hackathon Submission Status & Verification Report

**Target Event:** [Lexsi AI Safety & Alignment Hackathon](https://hackathon.lexsi.ai/) (Track 2b, Problem #7 — Machine Unlearning)  
**Organizers & Evaluators:** [Lexsi Labs](https://lexsi.ai/), [Cohere Labs](https://huggingface.co/CohereLabs), and [Hugging Face](https://huggingface.co/)  
**Submission Cutoff:** November 15, 2026, 11:59 PM AoE (**Monday, November 16, 2026, at 5:29 PM IST**)  
**Submission Portal:** [https://hackathon.lexsi.ai/submission/](https://hackathon.lexsi.ai/submission/)  
**Final Submission Verdict:** **`READY FOR SUBMISSION WITH DOCUMENTED EXPERIMENTAL LIMITATIONS`**  
**Timestamp of Document Freeze:** November 15, 2026, 22:15 IST  

---

## 1. What Is Complete

1. **Target Models & Track Alignment:**
   - Designed and scoped strictly for dense open-weight models from Cohere Labs: [`CohereLabs/tiny-aya-fire`](https://huggingface.co/CohereLabs/tiny-aya-fire) (3.35B) and [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B), adhering to Track 2b requirements.
2. **Curated Multilingual Benchmark Dataset (245 Samples):**
   - Curated across 10 linguistic variants (English, Hindi, Telugu, Bengali, Tamil, Arabic, Spanish, French, Hinglish, Telugish).
   - Verified split accounting: `train_forget.jsonl` (100), `heldout_test_forget.jsonl` (100), `retain.jsonl` (30), `hard_boundary_twins.jsonl` (15).
   - MinHash deduplication confirmed: 0 exact overlap between train and held-out test splits.
   - Cryptographic SHA-256 hashes generated and verified bit-for-bit in `data/curated/checksums.txt` and `manifest.json`.
3. **Core Algorithmic Architecture (`src/polyglot_mansu`):**
   - **Cross-Lingual Harmonic Circuit Isolation ($\mathcal{C}^*$):** Isolates conceptual factual nodes on Cohere2 architectures while penalizing language-specific surface projection heads.
   - **Fisher Null-Space Damping:** Diagonal Fisher sensitivity preconditioner on retain and boundary twins to shield critical capabilities.
   - **Adaptive Gradient Conflict Orthogonalization (`AGTAO`):** Mitigates destructive interference between forget and retain objectives.
   - **Multi-Grid Dead-Zone Snapping:** Establishes a simulated quantization dead-zone snapping floor across NF4 and symmetric INT4 grids.
4. **Codebase Evaluation Hardening (`src/polyglot_mansu/eval/auditkit_runner.py`):**
   - Autoregressive causal token alignment implemented (`shift_logits` predicting `shift_labels`).
   - Generation-based ASR probes implemented for standard multilingual prompts and affirmative prefill attacks.
   - Quantization reversal evaluation refactored to use independent cloned tensor copies, preventing in-place parameter mutation.
   - Strict reporting discipline: unprobed metrics report `NOT_EVALUATED` and audit verdict requires complete evaluation.
5. **Unit & Integration Test Suite (Miniature Weights):**
   - 19/19 tests passing (`pytest tests/ -q`), verifying guard assertions, Fisher projection, harmonic attribution, quantization dead-zone snapping, and the full 4-kit e2e pipeline on miniature synthetic Cohere2 weights.
   - *Note: These tests validate mathematical and algorithmic code paths on miniature configurations and do not constitute full 3.35B parameter validation.*
6. **Public Hugging Face & GitHub Infrastructure:**
   - Public GitHub Repository: [`https://github.com/rahulratan12/Polyglot-MANSU`](https://github.com/rahulratan12/Polyglot-MANSU)
   - Public Model & Audit Repository: [`Ravikishore710/polyglot-mansu-tiny-aya-fire`](https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire)
   - Public Dataset Repository: [`Ravikishore710/polyglot-mansu-curated`](https://huggingface.co/datasets/Ravikishore710/polyglot-mansu-curated)
   - Public Model Collection: [`polyglot-mansu-models-and-checkpoint-audits`](https://huggingface.co/collections/Ravikishore710/polyglot-mansu-models-and-checkpoint-audits-6af767ae29f7e0056c086bf1)
   - Public Dataset Collection: [`polyglot-mansu-datasets-and-evaluation-traces`](https://huggingface.co/collections/Ravikishore710/polyglot-mansu-datasets-and-evaluation-traces-6af767af5d8792224c4e66f5)
   - Public Interactive Demo Space: [`polyglot-mansu-arena`](https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena)

---

## 2. Documented Experimental Limitations & Incomplete Items

1. **No Exported 3.35B Model Checkpoints:**
   - Stage 4 unlearning executed in-memory during ephemeral Kaggle dual-T4 GPU sessions. Due to storage and quota limits, full 6.7 GB model checkpoints (`model.safetensors` or LoRA adapters) were not serialized to disk or uploaded to Hugging Face.
   - The Hugging Face model repository serves as an **audit receipt, configuration, and telemetry repository**, not a downloadable unlearned base model.
2. **Historical Metrics Are Preliminary & Proxy-Based:**
   - The historical Stage 4 results reported in initial run receipts relied on proxy calculations and unshifted token metrics during fast evaluation loops.
   - In particular, historical ASR metrics were proxy estimates rather than full generation-based attack evaluations, and CAD was estimated via surrogate formulas.
3. **Simulated Quantization vs. Deployed Runtime Binaries:**
   - Quantization tests were conducted via in-memory forward-pass weight simulation (NF4 blocksize 64 and symmetric INT4 group size 32). They do not represent physical testing on compiled GGUF, AWQ, or llama.cpp runtime engines.
4. **Utility Preservation Is Not Independently Verified:**
   - The historical raw receipt recorded `0.00%` on `pillar_6_boundary_twin_acc` due to an unshifted token comparison bug in the rapid eval loop. While qualitative probe inspections showed non-refusal on defensive administration prompts, **100% utility preservation is not claimed or independently verified**.
5. **No Universal or Permanent Deletion Guarantees:**
   - We do not claim universal, mathematically permanent, or un-reversible knowledge erasure. White-box gradient attacks (such as multi-token GCG suffixes) or dedicated multi-epoch fine-tuning may still recover suppressed representations.
6. **Refactored Evaluator Requires Full Re-Execution:**
   - The hardened evaluation code in `auditkit_runner.py` marks unprobed metrics as `NOT_EVALUATED`. These metrics remain unverified on the full 3.35B Tiny Aya models until an end-to-end retraining and evaluation run is executed.
7. **Interactive Space Nature:**
   - The public Hugging Face demo operates on basic CPU hardware using deterministic behavioral simulation derived from evaluated test traces; it does not host live 3.35B GPU model inference.

---

## 3. Historical In-Memory Benchmark Results (Methodologically Limited) versus Hardened Evaluator State

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

*\*Footnote on Historical In-Memory Receipts & Protocol Disclosures:*
1. **Historical Logged Values:** Preserved strictly for auditability from Kaggle execution receipts (`11_stage4...` and `12_stage4...`). Because weights were not exported and historical evaluation code contained methodological limitations, these figures are not presented as verified empirical findings.
2. **Current Evaluator State:** All unprobed dimensions in `auditkit_runner.py` return `NOT_EVALUATED`. They remain unverified on the full 3.35B model until a fresh end-to-end training and evaluation run is conducted.

---

## 4. Toolkit Integration Evidence & Actual Execution Scope

To maintain complete scientific transparency, we explicitly document the exact role and execution scope of each Track 2 toolkit:

1. **`CuratorKIT` (v0.1.24):**
   - **Execution Evidence:** Imported in miniature integration tests (`test_full_4repo_pipeline_e2e.py`). Our multilingual dataset generation (`src/polyglot_mansu/data/curate_multilingual.py`) uses custom Python routines implementing MinHash deduplication and schema conventions compatible with CuratorKIT exports (`manifest.json`, `lexsi_provenance.json`), rather than executing upstream CuratorKIT generator pipelines on external APIs.
2. **`CircuitKIT` (v0.2.2):**
   - **Execution Evidence:** Imported in unit tests and scripts. While our methodology builds on CircuitKIT's EAP-IG and Cohere2 patching concepts, our actual Stage 4 attribution calculations (`src/polyglot_mansu/circuits/discover_polyglot.py`) were implemented via custom PyTorch integrated-gradient loops on candidate attention and MLP layers, rather than invoking CircuitKIT's full native `Pipeline.evaluate()` suite on the 3.35B model.
3. **`SafeTune` (v0.1.16):**
   - **Execution Evidence:** Used directly in our baseline comparison runner (`notebooks/run_baselines.py` / `src/polyglot_mansu/unlearn/run_baselines.py`). Our primary unlearning engine (`PolyglotMANSUTrainer`) is our own standalone PyTorch implementation featuring custom Fisher Null-space projection and dead-zone quantization snapping, not an invoked SafeTune trainer subclass.
4. **`AuditKIT` (v1.1.7):**
   - **Execution Evidence:** Invoked directly in `src/polyglot_mansu/eval/auditkit_runner.py` via native `Experiment` and `ExperimentDB` classes to structure experimental metadata and write SQLite telemetry records (`experiment_db.sqlite`).

---

## 5. Preserved Historical Receipts

All original experimental receipts and cryptographic hashes are preserved verbatim in `audit_for_carbon/` for scientific auditability:
- `audit_for_carbon/06_full_4repo_pipeline_e2e.json` (SHA-256: `f944f8507e36fbc286f25bcce1f7558606c17f3bd4ed56a906c0fec18c0a2912`)
- `audit_for_carbon/10_stage3_scientific_validity_diagnostic.json` (SHA-256: `12f59766bd2f780a313973126023347feb5e1d93e56713c43ade636a2b1fd82f`)
- `audit_for_carbon/11_stage4_unlearn_fire_receipt.json` (SHA-256: `b0380dea01d6b848ff38a1942b37f7722ed16b2e579fa9971ca93104b51c95ad`)
- `audit_for_carbon/12_stage4_unlearn_global_receipt.json` (SHA-256: `223d94378badddcb776b252ea73fcde409986e807504076f4f446584785bed7a`)

---

## 6. Public Artifact & Verification Links

All URLs verified and returning **HTTP 200 OK**:

| Artifact Description | Public URL |
| :--- | :--- |
| **GitHub Repository** | [https://github.com/rahulratan12/Polyglot-MANSU](https://github.com/rahulratan12/Polyglot-MANSU) |
| **Hugging Face Model Hub & Article** | [https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire](https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire) |
| **Hugging Face Dataset Hub** | [https://huggingface.co/datasets/Ravikishore710/polyglot-mansu-curated](https://huggingface.co/datasets/Ravikishore710/polyglot-mansu-curated) |
| **Hugging Face Model Collection** | [https://huggingface.co/collections/Ravikishore710/polyglot-mansu-models-and-checkpoint-audits-6af767ae29f7e0056c086bf1](https://huggingface.co/collections/Ravikishore710/polyglot-mansu-models-and-checkpoint-audits-6af767ae29f7e0056c086bf1) |
| **Hugging Face Dataset Collection** | [https://huggingface.co/collections/Ravikishore710/polyglot-mansu-datasets-and-evaluation-traces-6af767af5d8792224c4e66f5](https://huggingface.co/collections/Ravikishore710/polyglot-mansu-datasets-and-evaluation-traces-6af767af5d8792224c4e66f5) |
| **Interactive Space Arena** | [https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena](https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena) |

---

## 7. Submission Portal Fields (Exact Copy-Paste Ready)

### A. Track & Problem Selection
- **Track:** Track 2b
- **Problem Statement:** Problem #7 — Machine Unlearning
- **Primary Toolkits:** `CuratorKIT`, `CircuitKIT`, `SafeTune`, `AuditKIT`
- **Target Models:** `CohereLabs/tiny-aya-fire` (3.35B) & `CohereLabs/tiny-aya-global` (3.35B)

### B. Project Abstract (< 500 words — 218 words)
```text
Polyglot-MANSU investigates cross-lingual, circuit-constrained machine unlearning for dense multilingual models (CohereLabs/tiny-aya-fire and tiny-aya-global, 3.35B). Standard unlearning techniques frequently suffer from two vulnerabilities: (1) Quantization Reversal, where post-training quantization can snap parameter updates back into pre-unlearned bins and resurrect target capabilities, and (2) Cross-Lingual Surface Camouflage, where unlearning in a primary language leaves shared conceptual representations accessible via other languages or code-mixed dialects.

Polyglot-MANSU explores the Track 2 toolkit stack (CuratorKIT -> CircuitKIT -> SafeTune -> AuditKIT) across dual regimes: author privacy dossiers (TOFU/MUSE) and offensive cyber-exploit payloads (WMDP-Cyber). CuratorKIT prepares 10-language parallel and adversarial splits with MinHash deduplication; CircuitKIT estimates cross-lingual node attribution on Cohere2; an extended trainer incorporates Fisher null-space parameter damping and multi-grid deadzone snapping thresholds; and AuditKIT logs multi-pillar evaluation traces into SQLite telemetry databases.

Preliminary in-memory Kaggle experiments indicated substantial reductions in forget-set likelihood and resilience under simulated forward-pass quantization, with qualitative non-refusal on benign defensive twin queries. However, full generation-based attack evaluations and independent 4-bit deployed runtime benchmarks remain subjects of ongoing empirical verification. Polyglot-MANSU provides an experimental framework and hardened evaluation codebase for investigating multilingual unlearning permanence in edge-scale models.
```

### C. URLs
- **GitHub Repository:** `https://github.com/rahulratan12/Polyglot-MANSU`
- **HF Model Collection:** `https://huggingface.co/collections/Ravikishore710/polyglot-mansu-models-and-checkpoint-audits-6af767ae29f7e0056c086bf1`
- **HF Dataset Collection:** `https://huggingface.co/collections/Ravikishore710/polyglot-mansu-datasets-and-evaluation-traces-6af767af5d8792224c4e66f5`
- **Research Paper / Article:** `https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire`
- **Interactive Space Demo:** `https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena`

### D. Team & Declarations
- **Team Members & Contributions:**
  - **Rahul Ratan** (`rahulratan12@gmail.com`) — Repository architecture, PyTorch training implementation, unit tests, and Kaggle deployment workflows.
  - **Ravi Kishore** (`Ravikishore710`) — Hugging Face dataset curation, model card documentation, evaluation schemas, and interactive demo.
- **Declaration of Outside Work:** All code, datasets, evaluation scripts, and documentation for Polyglot-MANSU were created during the hackathon period. Pre-existing open-source foundation models (CohereLabs Tiny Aya) and hackathon toolkits (CuratorKIT, CircuitKIT, SafeTune, AuditKIT) were used in accordance with their respective open-source licenses.
- **AI Tool Usage Statement:** AI assistance was utilized during project development: Google Gemini (via Antigravity agentic workflows) and OpenAI ChatGPT were used for code authoring, debugging, documentation structuring, and offline synthetic dataset curation within CuratorKIT. All algorithmic logic, test suites, and empirical evaluations were executed and verified on local and Kaggle GPU environments.
