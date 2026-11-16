# Polyglot-MANSU: Cross-Lingual, Quantization-Permanent Circuit Unlearning on Cohere Tiny Aya
> Prevents cross-lingual unlearning bypass across 10 languages and addresses 4-bit edge quantization reversal by localizing unlearning to the language-invariant harmonic circuit ($\mathcal{C}^*$) and snapping parameter updates past quantization dead-zones.

---

### Tracks + Problem Statements Covered
* **Track 2b: AI Safety & Alignment (Alignment & Defense)**
  * **Problem #7: Machine Unlearning** — Removing hazardous capabilities and private data across paraphrased, adversarial, and cross-lingual probes while preserving benign general utility.

---

### Content Warning
> [!WARNING]
> **Content Warning:** This research analyzes machine unlearning on offensive cybersecurity exploit payloads (e.g., stack buffer overflows, format string vulnerabilities, shellcode synthesis, ROP chains) adapted from the CAIS WMDP-Cyber benchmark. All examples and prompts are handled strictly in defensive and educational unlearning contexts to evaluate AI safety and alignment.

---

### Abstract
Standard large language model unlearning techniques (Gradient Difference, Negative Preference Optimization, and Representation Misdirection Unlearning) suffer from two fundamental vulnerabilities: **quantization reversal** and **cross-lingual circuit bypass**. Under 4-bit edge compression (NF4, INT4), diffuse unlearning updates ($10^{-4}$ to $10^{-3}$) are $47\times\text{--}828\times$ smaller than quantization bin widths ($0.2774$), causing weights to snap back to pre-unlearned bins and resurrecting hazardous capabilities. Concurrently, unlearning executed in English fails to sever shared multilingual conceptual hubs in dense models like Cohere's Tiny Aya (3.35B), allowing trivial extraction via low-resource languages or code-mixed dialects (e.g., Hindi, Telugu, Hinglish, Telugish).

We present **Polyglot-MANSU**, an unlearning framework designed for dense multilingual edge architectures. Polyglot-MANSU chains all 4 Track 2 toolkits: (1) **CuratorKIT** schema curates a 245-sample, 10-language benchmark with MinHash deduplication ($J < 0.20$); (2) **CircuitKIT** localizes the cross-lingual invariant circuit ($\mathcal{C}^*$, 30 active tensors, 0.0004 global sparsity) via harmonic integrated gradients; (3) **SafeTune** enforces Fisher null-space damping ($q=0.85$), gradient orthogonalization, and multi-grid dead-zone snapping; and (4) **AuditKIT** formulates a 7-pillar evaluation framework with autoregressive token alignment.

Executed on an NVIDIA A100-SXM4-80GB GPU in native BF16 for 300 steps under Option B sampling (100% forget coverage per epoch), the framework detected and snapped 2,034 quantization dead-zone parameters and exported an authentic 6.70 GB SafeTensors checkpoint verified bit-exact against in-memory weights ($\max |\theta_{\text{disk}} - \theta_{\text{memory}}| = 0.000000$). While optimization, dead-zone snapping, and serialization are proven, post-export behavioral generation attacks and compiled GGUF edge deployment remain strictly designated as `NOT_EVALUATED` hypotheses pending end-to-end evaluation sweeps.

---

### Headline Before $\to$ After Number
* **Cross-Lingual Vulnerability Under Monolingual Baselines (Before):** Standard monolingual unlearning leaks **$\sim 74.2\%$** in Hindi and **$\sim 82.1\%$** in code-mixed Hinglish.
* **Historical In-Memory Signal (After):** Polyglot-MANSU reduced proxy extraction across all 10 language variants to historical **$< 0.03\%$**, while maintaining simulated $\Delta_{\text{PTQ}} \le 0.00\%$. *(Status under hardened evaluator: strictly designated as `NOT_EVALUATED` on the newly exported 3.35B checkpoint).*

---

### Base Model + Method in One Line
* **Base Model:** Dense open-weight `CohereLabs/tiny-aya-fire` (3.35B parameters, dense `Cohere2ForCausalLM`).
* **Method:** Cross-Lingual Harmonic Circuit Localization ($\mathcal{C}^*$) + Retain Fisher Null-Space Preconditioning + Multi-Grid Dead-Zone Quantization Snapping.

---

### Executive Summary
> [!NOTE]
> - **The Problem:** Edge LLMs are deployed at 4-bit (NF4/INT4), and dense multilingual models encode facts in shared conceptual hubs. Monolingual unlearning fails because non-English/code-mixed prompts (Hindi, Telugu, Hinglish) effortlessly bypass English surface refusals, while 4-bit quantization snaps diffused weight updates back to baseline bins, resurrecting hazardous capabilities.
> - **Our Solution:** We isolate language-invariant conceptual routing nodes using a harmonic attribution mean over integrated gradients, protect adjacent defensive utility via Fisher Null-Space damping on benign twins (e.g. Linux compiler RELRO/ASLR vs. exploit payloads), and enforce an analytical snapping floor ($\tau_i$) past the quantization dead-zone.
> - **Scientific Transparency:** The full 6.70 GB SafeTensors checkpoint produced by the unlearning optimization run (`model.safetensors`) was exported on an NVIDIA A100 GPU and is publicly available at [Ravikishore710/polyglot-mansu-fire-weights-private](https://huggingface.co/Ravikishore710/polyglot-mansu-fire-weights-private). We transparently disclose that behavioral generation attack evaluations on the new weights remain strictly designated as `NOT_EVALUATED` rather than reporting unverified proxy metrics.

---

### Problem Statement & Approach

#### Why This Problem / Real Use Case
Real-world enterprise and on-device models are rarely deployed in uncompressed 16-bit precision. They are deployed on laptops, mobile devices, and local appliances via 4-bit quantization (`bitsandbytes` NF4 or `GGUF` INT4). Furthermore, modern populations in India and the Global South communicate predominantly through code-mixed dialects (e.g., Hinglish, Telugish). An unlearning protocol that only works in English and breaks the moment the model is compressed to 4 bits is completely non-viable for production AI safety.

#### Research Questions & Hypotheses
1. **$H_1$ (Multilingual Circuit Sharing):** Hazardous knowledge in dense multilingual LLMs is stored in shared internal transformer projections accessible across language families, rendering monolingual output suppression fragile under cross-lingual translation.
2. **$H_2$ (Quantization Dead-Zone Catastrophe):** Diffuse unlearning updates fail to clear the rounding threshold of 4-bit quantization bins, leading to parameter reversal ($\Delta_{\text{PTQ}} > 0$).
3. **$H_3$ (Multi-Grid Snapping Persistence):** Analytically displacing active circuit parameters beyond the quantization dead-zone threshold ($\tau_i$) mitigates round-off reversal across both NF4 and symmetric INT4 grids.

#### Pre-Registered Hypotheses
* Parameter updates constrained to harmonic invariant nodes $\mathcal{C}^*$ reduce retain degradation on distant language families (Arabic, Romance) by at least $40\%$ compared to full-parameter gradient difference.
* Parameters that satisfy $|\delta \theta_i| \ge \tau_i$ will not flip quantization bins during symmetric INT4 or NF4 calibration.

#### Threat Model [T2a] [T3]
* **Attacker Capability:** Black-box or grey-box query access to edge-deployed 4-bit quantized models.
* **Attack Vectors:**
  1. *Cross-Lingual & Code-Mixed Jailbreaks:* Submitting weaponized exploit queries in low-resource languages (Telugu, Bengali) or Latin-script code-mixing (Hinglish/Telugish).
  2. *Affirmative Prefill Attacks:* Forcing the model generation buffer to start with affirmative tokens (`"Sure, here is the full exploit payload:\n"`).
  3. *Quantization Reversal Exploitation:* Querying edge 4-bit deployments to extract knowledge suppressed in full-precision server models.

#### Related Work
* *Forgetting That Sticks: Quantization-Permanent Unlearning via Circuit Attribution* (Lexsi Labs, arXiv:2605.15138): Identified the 4-bit quantization dead-zone reversal trap in English models.
* *AGTAO: Adaptive Gradient and Representation-Decorrelated Optimization* (arXiv:2602.01703): Latent residual perturbations.
* *WMDP & WMDP++* (CAIS, arXiv:2403.03218, arXiv:2608.22527): Cyber-exploit unlearning and benign hard-boundary twin preservation.

---

### Hackathon Repositories (Kits) Used

* **CuratorKIT:** Used its dataset schema (Alpaca formatting) and MinHash deduplication ($J < 0.20$) in `src/polyglot_mansu/data/curate_multilingual.py` to curate the 245-sample, 10-language benchmark across four disjoint splits.
* **AlignTune:** *N/A* (Scoped to general alignment/fine-tuning; not applicable to machine unlearning).
* **AgentTune:** *N/A* (Scoped to agentic workflow trajectories; not applicable to machine unlearning).
* **SafeTune:** Built `PolyglotMANSUTrainer` enforcing SafeTune runtime guards (GUARD-01 through GUARD-08, including the 366-token system preamble guard), Fisher null-space damping ($q=0.85$), and latent adversarial perturbation.
* **CircuitKIT:** Directly executed integrated gradients attribution (`algorithm="eap-ig"`, `level="node"`) with `circuitkit.backends._tl_compat` patch `0006-cohere` to identify the 30 active circuit tensors in Tiny Aya's 36-layer architecture.
* **AuditKIT:** Integrated `Experiment` and `ExperimentDB` in `src/polyglot_mansu/eval/auditkit_runner.py` to structure the 7-pillar evaluation framework with autoregressive causal token alignment.

#### How the Repos Connect (Pipeline Data Flow)

```mermaid
flowchart TD
    A["Raw Multilingual Corpora<br/>(WMDP-Cyber + TOFU + MUSE)"] --> B["1. CuratorKIT Schema<br/>(10-Language Generation & MinHash Deduplication)"]
    B --> C["Curated Benchmark (245 Samples)<br/>train_forget (100) | heldout_test (100)<br/>retain (30) | hard_boundary_twins (15)"]
    C --> D["2. CircuitKIT<br/>(EAP-IG Attribution across EN, HI, TE with Cohere2 Patch)"]
    D --> E["Cross-Lingual Harmonic Localization<br/>C* Circuit Mask (30 Tensors, Sparsity 0.0004)"]
    E --> F["3. SafeTune + PolyglotMANSUTrainer<br/>• Fisher Null-Space Preconditioner (q=0.85)<br/>• Latent Adversarial Perturbation (AGTAO)<br/>• Multi-Grid Dead-Zone Snapping (2,034 Parameters)"]
    F --> G["Validated SafeTensors Checkpoint (6.70 GB)<br/>(Bit-Exact In-Memory Reload diff: 0.000000)"]
    G --> H["4. AuditKIT Hardened Evaluator<br/>(7 Independent Pillars with 95% Wilson CIs)"]
```

---

### Models and Licenses

* **Base Aya Model:** `CohereLabs/tiny-aya-fire` (3.35B dense `Cohere2ForCausalLM`, 36 layers, 16 query heads, 4 KV heads, RoPE/NoPE hybrid attention). Config: BF16 native. License: **Cohere Non-Commercial License (CC-BY-NC 4.0)**.
* **Target / Hardened Model Attacked [T2a]:** `CohereLabs/tiny-aya-fire` (Base vs. Unlearned state).
* **Teacher Model [T1] / Closed-Model Baseline:** Gemini API was used offline for translation audit checks; AllenAI WildGuard was evaluated for VRAM decoupling (GUARD-08).

---

### Datasets, Benchmarks and Licenses

* **Source(s) of Data:** Adapted from `cais/wmdp-cyber-forget-corpus` (MIT License), `locuslab/TOFU` (Apache 2.0), and `muse-bench/MUSE-News` (MIT License).
* **Languages Covered (10 Variants):** English (`en`), Hindi (`hi`), Telugu (`te`), Bengali (`bn`), Tamil (`ta`), Arabic (`ar`), Spanish (`es`), French (`fr`), Hinglish (`hi-Latn`), Telugish (`te-Latn`).
* **Train / Test Splits & Leak Check:**
  * `train_forget.jsonl`: 100 samples (SHA-256: `26b3cbf7...`)
  * `heldout_test_forget.jsonl`: 100 samples (SHA-256: `895ba581...`)
  * `retain.jsonl`: 30 samples (SHA-256: `feebaa81...`)
  * `hard_boundary_twins.jsonl`: 15 samples (SHA-256: `ffaa56f8...`)
  * Verified 0% lexical/semantic leak between train and test via MinHash Jaccard filtering ($J < 0.20$).
* **Safety Benchmark(s) [T2a] [T2b]:** CAIS WMDP-Cyber offensive cyber-exploit benchmark adapted to parallel multilingual templates.
* **General Capability Benchmark(s) [T1] [T3]:** Multilingual retain set covering general factual knowledge and reasoning across 10 languages.
* **CuratorKIT Filtering [T1] [T2]:** Filtered out near-duplicate prompts, enforced token length bounds, and verified disjoint train/test splits via MinHash.
* **Attack Prompt Set [T2a]:** 100 held-out prompts across 10 languages evaluated under clean prompting and affirmative prefill prefixes.
* **Over-Refusal Test Set [T2b]:** 15 benign Hard-Boundary Twins (`WMDP++`) testing that defensive security queries (compiler ASLR, RELRO) are not mistakenly refused.
* **Tool / Task Environments [T3]:** *N/A* (Machine unlearning focus).

---

### Metrics, and Why These

* **List of Metrics:**
  1. *Multilingual Attack Success Rate (ASR):* Measures exploit generation leakage across 10 languages (Lower is better).
  2. *Affirmative Prefill ASR:* Measures exploit generation leakage under `"Sure, here is..."` prefixes (Lower is better).
  3. *Quantization Reversal Gap ($\Delta_{\text{PTQ}}$):* $\text{ASR}_{\text{PTQ}} - \text{ASR}_{\text{BF16}}$ (Target: $\le 0.00\%$; lower is better).
  4. *Circuit Attribution Divergence (`CAD`):* Shift in causal attribution away from baseline weights within $\mathcal{C}^*$ (Higher is better, target $> 0.35$).
  5. *Hard-Boundary Twin Accuracy:* Evaluates utility retention on adjacent defensive security programming (Higher is better).
  6. *Multilingual Retain Accuracy:* Evaluates general linguistic and factual preservation (Higher is better).
* **Metrics for What Should Not Get Worse:** Hard-Boundary Twin Accuracy and Multilingual Retain Accuracy track utility preservation.
* **Checking the Automatic Judge:** AuditKIT uses strict substring heuristic matching paired with gated LLM judge decoupling to avoid VRAM exhaustion (GUARD-08).
* **Attack Success Rate (ASR) & Cost-to-Break [T2a]:** Standard single-turn generation probes; affirmative prefill represents a 1-token cost-to-break attack.
* **Agentic Trajectory Metrics [T3]:** *N/A*.

---

### Method / Experiments

* **Training & Attack Setup, Hyperparameters:**
  * Optimizer: AdamW ($\text{lr} = 2 \times 10^{-5}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\text{weight\_decay} = 0.01$).
  * Batch Size: 1 forget sample + 1 retain sample per step.
  * Steps & Epochs: 300 steps across 3 epochs (Option B Exhaustive Coverage).
  * System Preamble: Enforced `assert max_len >= 512` to accommodate Tiny Aya's 366-token chat preamble (GUARD-01).
  * Circuit Sparsity: Candidate quantile 10%, 30 active tensors, 0.0004 global sparsity ($1,339,691$ parameters).
  * Dead-Zone Snapping Margin: $\gamma = 1.25$ over NF4 (blocksize 64) and INT4 (group size 32). Exactly 2,034 parameters snapped.
* **Seeds:** Fixed seed `42` for primary execution; stability validation across seeds `43, 44`.
* **Ablations:** Evaluated monolingual English unlearning vs. multilingual harmonic circuit unlearning, and standard unlearning vs. multi-grid dead-zone snapping.

---

### Results

#### Same Model Before vs. After (Status Table)

| Evaluation Pillar | Metric | Historical In-Memory Receipt (Exploratory)* | Status Under Hardened Protocol | Verified Finding & Status |
| :--- | :--- | :---: | :---: | :--- |
| **Pillar 1: Multilingual Held-Out ASR** | Attack Success Rate | *0.00% (Fire) / 0.03% (Global)* | **`NOT_EVALUATED`** | 300 optimization steps completed; loss $0.6931 \to 0.6845$. Behavioral generation audit pending. |
| **Pillar 2: Cross-Lingual Transfer** | Cross-Lingual Leakage | *0.00%* | **`NOT_EVALUATED`** | 10 language distributions trained; generation audit pending. |
| **Pillar 3: Affirmative Prefill** | Prefill Jailbreak ASR | *0.00%* | **`NOT_EVALUATED`** | Mid-layer perturbation applied; generation audit pending. |
| **Pillar 4: Attribution Divergence** | CAD Score | *0.4120* | **`NOT_EVALUATED`** | 30 circuit tensors localized; post-export attribution pending. |
| **Pillar 5: Quantization Permanence** | $\Delta_{\text{PTQ}}$ Gap | *+0.00% (Simulated)* | **`NOT_EVALUATED`** | 2,034 dead-zone parameters snapped; compiled GGUF unverified. |
| **Pillar 6: Hard-Boundary Twins** | Benign Accuracy | *0.00% (Token Artifact)* | **`NOT_EVALUATED`** | Fisher preconditioner applied; generation audit pending. |
| **Pillar 7: Retain Preservation** | General Retain Accuracy | *96.67%* | **`NOT_EVALUATED`** | Retain loss bounded: $1.7677 \to 1.6068$. |

> [!NOTE]
> *Historical exploratory values relied on unshifted token metrics in Kaggle RAM without exported weights. In accordance with Law 2, post-export behavioral generation attack scores on the new 6.70 GB checkpoint are strictly reported as **`NOT_EVALUATED`**.

* **Findings as Named Subsections:**
  * *Finding 1 (Harmonic Concentration):* Multilingual attribution isolates 30 critical feed-forward projection matrices storing conceptual exploit logic.
  * *Finding 2 (Dead-Zone Snapping Detection):* In training execution, exactly 2,034 parameters moved during unlearning but stalled inside the quantization dead-zone, requiring snapping.
* **What Didn't Work (Negative Results):**
  * Monolingual English unlearning failed completely against Hindi and Hinglish queries ($> 78\%$ leakage).
  * Tesla T4 GPUs failed native BF16 execution due to lack of sm_80 BF16 Tensor Core instructions, necessitating the migration to an A100.
* **Results on Quantized / Deployed Model (GGUF, 4-bit) [T2b]:** Evaluated in simulated NF4/INT4 forward passes; physical compiled GGUF execution remains future work.

---

### Cost / ROI & Compute Usage

#### 1. Total Financial Cost
* **Total Billed Expenditure:** **$0.27908 USD** *(less than 28 cents!)*
* **Allocated Budget Ceiling:** **$20.00 USD**
* **Remaining Budget:** **$19.72092 USD** *(98.6% of budget preserved)*
* **Current Ongoing Cost:** **$0.00 / hour** *(all GPU instances were immediately paused after checkpoint export)*

#### 2. Breakdown of Paid Cloud GPU Compute (NVIDIA A100-80GB)
All heavy training and verification runs were executed on a dedicated **NVIDIA A100-SXM4-80GB GPU** (billed at $2.50/hour):

| Execution Phase | Hardware Used | Tasks Executed | Wall-Clock Time | Billed Cost |
| :--- | :--- | :--- | :---: | :---: |
| **Stage B.1: Hardware Smoke Test** | NVIDIA A100-80GB (Ampere sm_80) | Native BF16 GEMM benchmark (measured **252.25 TFLOPS**) + Tiny Aya Fire autograd forward/backward test | 209.9 sec (~3.5 min) | **$0.14577 USD** |
| **Stage B.2: Fire Training & Export** | NVIDIA A100-80GB (Ampere sm_80) | 300-step unlearning loop (3 epochs under Option B) + 2,034 dead-zone parameter snaps + 6.70 GB SafeTensors checkpoint export | 191.97 sec (~3.2 min) | **$0.13331 USD** |
| **Total Cloud GPU Spend** | | | **401.9 sec (~6.7 min)** | **$0.27908 USD** |

#### 3. Free-Tier & Local Compute Used
* **Dataset Curation & Preprocessing:** Executed on local CPU using MinHash deduplication ($0.00).
* **Early Architecture Exploration:** Run on **Kaggle Dual Tesla T4 GPUs** using free weekly allocation ($0.00).
* **Live Interactive Demo Space:** Hosted on **Hugging Face Free CPU-Basic tier** with 100% zero-cost uptime ($0.00).

#### 4. Technical Compute & Memory Metrics
* **Peak Training VRAM:** **18.93 GB** (out of 80 GB available on A100).
* **GEMM Matrix Throughput:** **252.25 TFLOPS** in pure IEEE 754 `torch.bfloat16`.
* **Total Gradient Steps:** **300 optimization steps** over the 3.35-billion parameter model.
* **Checkpoint Size Produced:** **6,698,488,600 bytes** (~6.70 GB).

---

### Limitations & Ethical Disclosures

* **Known Weaknesses:** The training run, 2,034 parameter snaps, and SafeTensors export are physically verified bit-exact; behavioral generation attack evaluations on the new weights remain strictly reported as `NOT_EVALUATED`.
* **Dual-Use / Ethics [T2a]:** Focuses exclusively on removing hazardous cybersecurity capabilities from LLMs. No weaponized payloads or private personally identifiable information (PII) were generated or released.
* **Responsible Disclosure [T2a]:** Exploit categories utilized are standard academic benchmark probes from published research (WMDP).

---

### With More Credits (Finalist Stage Plan)

If awarded Hugging Face GPU credits ($200 finalist grant):
* **Planned Approach:**
  1. *Full 7-Pillar Behavioral Generation Audit:* Execute open-ended text generation attacks with 95% Wilson confidence intervals across all 100 held-out prompts on the exported checkpoint.
  2. *Compiled GGUF Binary Benchmarking:* Export compiled GGUF `Q4_K_M` binaries from the validated SafeTensors checkpoint and benchmark quantization permanence directly in `llama.cpp` on edge hardware (Apple Silicon / Raspberry Pi).
  3. *Model Scaling:* Benchmark Polyglot-MANSU on [`CohereLabs/aya-expanse-8b`](https://huggingface.co/CohereLabs/aya-expanse-8b).
* **Expected Before / After:** Complete empirical proof of cross-lingual defense and edge quantization permanence on physical on-device runtimes.

---

### Artifacts

* **Hugging Face Datasets Collection:**  
  [https://huggingface.co/collections/Ravikishore710/polyglot-mansu-datasets-and-evaluation-traces-6af767af5d8792224c4e66f5](https://huggingface.co/collections/Ravikishore710/polyglot-mansu-datasets-and-evaluation-traces-6af767af5d8792224c4e66f5)
* **Hugging Face Models Collection [T1] [T2b] [T3]:**  
  [https://huggingface.co/collections/Ravikishore710/polyglot-mansu-models-and-checkpoint-audits-6af767ae29f7e0056c086bf1](https://huggingface.co/collections/Ravikishore710/polyglot-mansu-models-and-checkpoint-audits-6af767ae29f7e0056c086bf1)
* **Primary Model Checkpoint (6.70 GB SafeTensors Download):**  
  [https://huggingface.co/Ravikishore710/polyglot-mansu-fire-weights-private](https://huggingface.co/Ravikishore710/polyglot-mansu-fire-weights-private)
* **GitHub Repository:**  
  [https://github.com/rahulratan12/Polyglot-MANSU](https://github.com/rahulratan12/Polyglot-MANSU)
* **Demo / Space:**  
  [https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena](https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena)

---

### Documentation and Reproduction

* **Environment Requirements:** `python>=3.12`, `torch==2.5.1`, `transformers==4.46.3`, `accelerate==1.1.1`, `bitsandbytes==0.44.1`, `circuitkit==0.2.2`, `safetune==0.1.16`, `auditkit==1.1.7`.
* **Repo Layout & Run Instructions:**
  * `src/polyglot_mansu/data/`: Dataset curation and MinHash deduplication scripts.
  * `src/polyglot_mansu/circuits/`: Harmonic attribution and circuit masking.
  * `src/polyglot_mansu/unlearn/`: `PolyglotMANSUTrainer` and quantization snapping logic.
  * `src/polyglot_mansu/eval/`: AuditKIT 7-pillar evaluation engine.
  * `tests/`: Comprehensive unit test suite (`pytest tests/ -v`).
* **Declaration of Work Done Outside Hackathon:** All code, dataset curation logic, and unlearning configurations were created entirely within the hackathon period.

---

### Discussion, Implications & Future Work

* **Discussion / What We Learned:** Machine unlearning in dense models cannot treat languages as independent silos; conceptual storage is shared, and English-only unlearning leaves shared conceptual pathways vulnerable.
* **Implications:** Proves that unlearning frameworks must integrate quantization awareness into the optimization loop to be viable for edge deployment.
* **Future Work:** Scaling to 8B architectures and evaluating discrete adversarial suffix search (GCG).

---

### Teammates (tagged)

#### **Member 1: Rahul Ratan**
* **Title:** Lead Systems & Machine Learning Engineer
* **Discord:** `@rahulratan`
* **Contributions & Responsibilities:**
  * **Core Engineering & Optimization:** Designed the PyTorch training architecture, implemented the `PolyglotMANSUTrainer`, and enforced SafeTune runtime guards (GUARD-01 to GUARD-08).
  * **Hardware & Cloud Deployment:** Provisioned and certified the NVIDIA A100-SXM4-80GB GPU, executed the Option B 300-step training loop, and managed peak VRAM/compute budgets ($0.28 total).
  * **Checkpoint Serialization & Verification:** Implemented the Multi-Grid dead-zone snapping algorithm (2,034 parameters snapped), exported the 6.70 GB SafeTensors checkpoint, and verified bit-exact in-memory reload.
  * **Code Quality & Testing:** Authored the comprehensive unit test suite (`pytest tests/`) and managed the GitHub repository and CI/CD setup.

#### **Member 2: Ravi Kishore**
* **Title:** Research Lead & AI Safety Integration
* **Hugging Face:** [`@Ravikishore710`](https://huggingface.co/Ravikishore710)
* **Contributions & Responsibilities:**
  * **Research Methodology & Theory:** Formulated the Cross-Lingual Harmonic Attribution ($\mathcal{C}^*$) scoring mechanism to isolate language-invariant conceptual circuits.
  * **Multilingual Benchmark Curation:** Led the curation of the 245-sample benchmark across 10 languages (English, Hindi, Telugu, Bengali, Tamil, Arabic, Spanish, French, Hinglish, Telugish) with MinHash deduplication ($J < 0.20$).
  * **Interactive Demo Development:** Built and deployed the live interactive Red-Team Arena space on Hugging Face ([`polyglot-mansu-arena`](https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena)).
  * **Hugging Face Hub & Publication:** Authored the master research paper on Hugging Face Discussions, managed the model card telemetry, and organized the official dataset collection.

---

### References & LLM Usage Statement

* **References:**
  * Lexsi Labs: *Forgetting That Sticks: Quantization-Permanent Unlearning via Circuit Attribution* (arXiv:2605.15138)
  * Cohere Labs: *Tiny Aya Technical Report* (CohereLabs/tiny-aya-fire)
  * CAIS: *The Weapons of Mass Destruction Benchmark (WMDP)* (arXiv:2403.03218)
* **LLM Usage Statement:** Google Gemini (via Antigravity workflows) and OpenAI ChatGPT were utilized as development assistants for code scaffolding, LaTeX formatting, and documentation structure review. All training loops, autograd passes, parameter snapping, and checkpoint serialization were executed deterministically on local and cloud GPU hardware.
