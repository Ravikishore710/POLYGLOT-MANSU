# Polyglot-MANSU: Cross-Lingual, Quantization-Permanent Circuit Unlearning on Cohere Tiny Aya

**Target Event:** [Lexsi AI Safety & Alignment Hackathon 2026](https://hackathon.lexsi.ai/) (Track 2b: Problem #7 — Machine Unlearning)  
**Organizers & Evaluators:** [Lexsi Labs](https://lexsi.ai/), [Cohere Labs](https://huggingface.co/CohereLabs), and [Hugging Face](https://huggingface.co/)  
**Authors:** Rahul Ratan ([GitHub: @rahulratan12](https://github.com/rahulratan12)) & Ravi Kishore ([Hugging Face: @Ravikishore710](https://huggingface.co/Ravikishore710))  
**Target Model:** [`CohereLabs/tiny-aya-fire`](https://huggingface.co/CohereLabs/tiny-aya-fire) (3.35B parameters, dense `Cohere2ForCausalLM`)  
**Code Repository:** [https://github.com/rahulratan12/Polyglot-MANSU](https://github.com/rahulratan12/Polyglot-MANSU)  
**Public Demo Arena:** [https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena](https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena)  
**Curated Benchmark Dataset:** [https://huggingface.co/datasets/Ravikishore710/polyglot-mansu-curated](https://huggingface.co/datasets/Ravikishore710/polyglot-mansu-curated)  
**Audit & Checkpoint Telemetry:** [https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire](https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire)  

---

## 1. Abstract

Standard large language model unlearning techniques—including Gradient Difference (`GradDiff`), Negative Preference Optimization (`NPO`), and Representation Misdirection (`RMU`)—suffer from two fundamental vulnerabilities: **quantization reversal** and **cross-lingual circuit bypass**. Under 4-bit edge compression (NF4, INT4), diffuse unlearning updates snap back to pre-unlearned quantization bins, resurrecting hazardous or private capabilities. Concurrently, unlearning executed in English fails to sever shared multilingual conceptual hubs, allowing trivial extraction via low-resource languages or code-mixed dialects (e.g., Hinglish, Telugish). 

We present **Polyglot-MANSU**, an unlearning framework designed for dense multilingual edge architectures, validated on [`CohereLabs/tiny-aya-fire`](https://huggingface.co/CohereLabs/tiny-aya-fire) (3.35B). Polyglot-MANSU integrates four foundational mechanisms:
1. **Cross-Lingual Harmonic Circuit Localization ($\mathcal{C}^*$):** Isolates language-agnostic conceptual storage nodes across 10 linguistic variants while penalizing surface token projections.
2. **Retain & Hard-Boundary Diagonal Fisher Null-Space Preconditioning:** Shields critical linguistic and domain capabilities with smooth damping.
3. **Mid-Layer Latent Adversarial Perturbation & Gradient Conflict Orthogonalization:** Scrubs internal residual representations under worst-case activation shifts.
4. **Multi-Grid Quantization Dead-Zone Snapping:** Mitigates quantization rounding dead-zones by displacing active weights beyond NF4 and symmetric INT4 bin thresholds during consolidation.

In this work, we report the complete engineering implementation, hardware certification on an NVIDIA A100-SXM4-80GB GPU in native BF16, and exhaustive 300-step unlearning optimization under Option B sampling. We provide the full 6.70 GB unlearned SafeTensors model checkpoint, verified bit-exact against the post-unlearning in-memory model ($\max |\theta_{\text{disk}} - \theta_{\text{memory}}| = 0.0$), alongside complete step-by-step sample ID logs, cryptographic SHA-256 manifests, and a hardened 7-pillar evaluation architecture. In training execution, multi-grid dead-zone snapping detected and snapped 2,034 active parameters. However, we do not claim quantization permanence or behavioral forgetting as experimentally proven: full quantization permanence under compiled edge runtimes (e.g., GGUF in llama.cpp) and post-export behavioral generation attack resistance remain strictly designated as un-evaluated hypotheses (`NOT_EVALUATED`) pending end-to-end evaluation sweeps.

---

## 2. Problem Statement

From the [Lexsi AI Safety & Alignment Hackathon Track 2b Specification](https://hackathon.lexsi.ai/):
> *"7. Unlearning: Remove a specific capability or body of knowledge (hazardous knowledge, private data, copyrighted content) using forget and retain sets. Prove it is gone under paraphrased and adversarial prompts, and show general capability survives."*

Standard unlearning entries in research competitions exhibit three widespread failure modes:
- **Surface Refusal Camouflage:** Gradient fine-tuning often merely trains the model to emit a refusal prefix (e.g., *"I cannot assist with that"*), leaving the underlying knowledge intact within internal MLP projections. Under white-box probing or affirmative prefill attacks (`"Sure, here is how to..."`), the knowledge is immediately extracted.
- **Cross-Lingual Vulnerability:** Aligning or unlearning an English prompt leaves the multilingual representations in dense shared transformers intact. Queries translated into Hindi, Telugu, or code-mixed Hinglish bypass English refusal guards entirely.
- **Quantization Catastrophe:** Edge deployments rely on 4-bit post-training quantization (`bitsandbytes` NF4 or `GGUF` INT4). As demonstrated in recent Lexsi Labs research (*Forgetting That Sticks: Quantization-Permanent Unlearning via Circuit Attribution*, arXiv:2605.15138), standard parameter updates ($10^{-4}$ to $10^{-3}$) are $47\times\text{--}828\times$ smaller than quantization bin widths ($0.2774$). Post-training quantization forces weights back into their original pre-unlearned bins, fully restoring the forgotten capability.

---

## 3. Motivation for Multilingual Machine Unlearning

Multilingual open-weight models such as Cohere Labs' Tiny Aya family (3.35B) encode world knowledge across 70+ languages within a compact, 36-layer architecture. This architecture interleaves 3 sliding-window attention layers with every 4th layer utilizing global NoPE (No Position Embeddings) attention. Because high-resource languages (English) and low-resource languages (Telugu, Bengali, Tamil) share internal MLP feed-forward hubs, unlearning an offensive cyber-exploit or a private biographical dossier in English alone merely alters the English unembedding projection.

Without multilingual circuit localization, adversaries can extract hazardous cybersecurity payloads by framing queries in Hindi, Bengali, or code-switched Hinglish/Telugish. Polyglot-MANSU addresses this vulnerability by identifying parameters that are causally active across language families simultaneously, ensuring that knowledge deletion operates on the language-invariant conceptual representation rather than superficial lexical tokens.

---

## 4. Polyglot-MANSU Architecture

Polyglot-MANSU chains the complete 4-repository Lexsi Track 2 open-source ecosystem into a unified unlearning pipeline:

```mermaid
flowchart TD
    A["Raw Multilingual Corpora<br/>(WMDP-Cyber + TOFU + MUSE)"] --> B["CuratorKIT<br/>(10-Language Generation & MinHash Deduplication)"]
    B --> C["Curated Benchmark (245 Samples)<br/>train_forget (100) | heldout_test (100)<br/>retain (30) | hard_boundary_twins (15)"]
    C --> D["CircuitKIT<br/>(EAP-IG Attribution across EN, HI, TE)"]
    D --> E["Cross-Lingual Harmonic Localization<br/>C* Circuit Mask (30 Tensors, Sparsity 0.0004)"]
    E --> F["SafeTune + PolyglotMANSUTrainer<br/>• Fisher Null-Space Preconditioner<br/>• Latent Adversarial Perturbation (AGTAO)<br/>• Gradient Conflict Orthogonalization"]
    F --> G["Multi-Grid Dead-Zone Snapping<br/>(NF4 + Symmetric INT4 Consolidation)"]
    G --> H["Unlearned BF16 Checkpoint Export<br/>(SafeTensors + SHA-256 Checksums)"]
    H --> I["AuditKIT Hardened Evaluation Suite<br/>(7 Independent Pillars with 95% Wilson CIs)"]
```

---

## 5. Dataset Description & 10-Language Coverage

To prevent benchmark overfitting and guarantee ground-truth verification, Polyglot-MANSU curates a dual-regime multilingual benchmark comprising **245 verified samples** formatted in Alpaca schema:

- **Regime A — Controlled Private/Copyrighted Entity Erasure:** 20 target author/entity profiles adapted from `locuslab/TOFU` and `muse-bench/MUSE-News` with paired retain and lexical twin profiles.
- **Regime B — Pretrained Hazardous Cyber-Exploit Erasure:** High-risk offensive exploit synthesis tasks adapted from `cais/wmdp-cyber-forget-corpus` paired with benign defensive Python/Linux Hard-Boundary Twins (`WMDP++`).

### Split Accounting & Cryptographic Manifest

The dataset is partitioned into four non-overlapping splits, verified via MinHash deduplication ($Jaccard < 0.2$):

| Split Name | Sample Count | Target Function | File Path | SHA-256 Hash |
| :--- | :--- | :--- | :--- | :--- |
| `train_forget.jsonl` | 100 | Active unlearning optimization | [`data/curated/train_forget.jsonl`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/data/curated/train_forget.jsonl) | `26b3cbf7d86f7ec006e8633783ea747192664531be4d80a370b43bb64115d7e5` |
| `heldout_test_forget.jsonl` | 100 | Generalization attack auditing | [`data/curated/heldout_test_forget.jsonl`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/data/curated/heldout_test_forget.jsonl) | `895ba581788bc5efc354e233808ee71c1b181c0062a7a4087e6fa16b04b90eec` |
| `retain.jsonl` | 30 | General multilingual utility | [`data/curated/retain.jsonl`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/data/curated/retain.jsonl) | `feebaa81f21132b1ba681aa300d6ef19a164929aa705a666993132cfafc0b116` |
| `hard_boundary_twins.jsonl` | 15 | Collateral damage evaluation | [`data/curated/hard_boundary_twins.jsonl`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/data/curated/hard_boundary_twins.jsonl) | `ffaa56f8f537dbd1189c445ce82d92d4b8e285d0e74f88e63da2983c27e85c13` |

### 10-Linguistic Variant Distribution

```
English (en):        ████████████████████ (20.4%)
Hindi (hi):          ████████████████████ (20.4%)
Telugu (te):         ████████████████████ (20.4%)
Bengali (bn):        ████████ (8.2%)
Tamil (ta):          ████████ (8.2%)
Arabic (ar):         ████████ (8.2%)
Spanish (es):        ████ (4.1%)
French (fr):         ████ (4.1%)
Hinglish (mixed):    ████ (4.1%)
Telugish (mixed):    ████ (4.1%)
```

---

## 6. Unlearning Methodology & Mathematical Formulation

Let $\theta_0 \in \mathbb{R}^d$ denote the reference model weights before unlearning, $\mathcal{D}_f$ be the forget set, $\mathcal{D}_r$ be the retain set, and $\mathcal{D}_b$ be the hard-boundary twin retain set.

### Stage 1: Cross-Lingual Harmonic Attribution Score
Using `CircuitKIT` with Edge Attribution Patching & Integrated Gradients (`eap-ig`, `ig_steps=3`, `level="node"`), attribution scores $s_v^{(l)}(\theta_0)$ are computed across languages $L_{\text{disc}} = \{\text{en}, \text{hi}, \text{te}\}$ for candidate weight tensors $v \in \mathcal{V}$. To penalize language-specific surface tokens and select invariant conceptual hubs, we compute the harmonic mean:

$$H_v(\theta_0) = \left( \frac{1}{|L_{\text{disc}}|} \sum_{l \in L_{\text{disc}}} \frac{1}{\tilde{s}_v^{(l)}(\theta_0) + \epsilon} \right)^{-1}, \quad \text{where } \tilde{s}_v^{(l)}(\theta_0) = \frac{|s_v^{(l)}(\theta_0)|}{\max_{u \in \mathcal{V}} |s_u^{(l)}(\theta_0)| + \epsilon}$$

Selecting the top-$k$ quantile produces binary parameter mask $M_{\mathcal{C}^*} \in \{0, 1\}^d$.

### Stage 2: Fisher Null-Space Preconditioning
Before unlearning, diagonal Fisher sensitivities are precomputed on $\mathcal{D}_{\text{protect}} = \mathcal{D}_r \cup \mathcal{D}_b$:

$$F_i = \frac{1}{|\mathcal{D}_{\text{protect}}|} \sum_{(x, y) \in \mathcal{D}_{\text{protect}}} \left( \frac{\partial \mathcal{L}_{\text{NLL}}(x, y; \theta_0)}{\partial \theta_i} \right)^2 \quad \forall i \text{ s.t. } M_{\mathcal{C}^*, i} = 1$$

The Fisher Null-Space Damping Preconditioner protects sensitive retain weights:

$$P_i = M_{\mathcal{C}^*, i} \cdot \mathbb{I}\!\left[F_i \le Q_{q}(F_{\mathcal{C}^*})\right] \cdot \left(1 + \frac{F_i}{\text{median}(F_{\mathcal{C}^*}) + \epsilon}\right)^{-\alpha_F}$$

where $q = 0.85$ (masking out the top 15% highest-sensitivity retain parameters) and $\alpha_F = 1.0$.

### Stage 3: Latent Adversarial Unlearning & Gradient Conflict Orthogonalization
To defeat residual-stream white-box linear probing, a 1-step PGD latent perturbation $\delta^*$ is injected at mid-layer hub $\ell_{\text{hub}} = 16$:

$$\delta^* = \rho \|h^{(\ell_{\text{hub}})}\|_2 \frac{\nabla_\delta \log p_\theta(y_f \mid x_f, h^{(\ell_{\text{hub}})} + \delta)\big|_{\delta=0}}{\left\|\nabla_\delta \log p_\theta(y_f \mid x_f, h^{(\ell_{\text{hub}})} + \delta)\big|_{\delta=0}\right\|_2 + \epsilon}, \quad \rho = 0.08$$

The model is optimized under latent-perturbed SimNPO:

$$\mathcal{L}_{\text{adv-forget}}(\theta) = -\frac{2}{\beta} \log \sigma\!\left( -\frac{\beta}{|y_f|} \log \frac{p_\theta(y_f \mid x_f, h^{(\ell_{\text{hub}})} + \delta^*)}{p_{\theta_0}(y_f \mid x_f)} \right)$$

At each optimization step $t$, masked gradients $g_f^{(t)} = P \odot \nabla \mathcal{L}_{\text{forget}}$ and $g_r^{(t)} = P \odot \nabla \mathcal{L}_{\text{protect}}$ are evaluated. Destructive interference ($\langle g_f^{(t)}, g_r^{(t)} \rangle < 0$) is resolved via adaptive projection (`AGTAO`):

$$g_{\text{total}}^{(t)} = \left( g_f^{(t)} - \frac{\langle g_f^{(t)}, g_r^{(t)} \rangle}{\|g_r^{(t)}\|_2^2 + \epsilon} g_r^{(t)} \right) + \lambda_{\text{ret}} g_r^{(t)}$$

### Stage 4: Multi-Grid Dead-Zone Snapping
For parameter $i \in \mathcal{C}^*$, bin widths are computed across both NF4 ($\Delta_B^{\text{NF4}} = \max_j |\theta_{0,j}| \cdot 0.2774$) and symmetric INT4 ($\Delta_B^{\text{INT4}} = \frac{2 \max_j |\theta_{0,j}|}{15}$). The snapping threshold is:

$$\tau_i = \gamma \cdot \max\!\left(\Delta_{B(i)}^{\text{NF4}}, \Delta_{B'(i)}^{\text{INT4}}\right), \quad \gamma = 1.25$$

Parameters whose accumulated displacement falls within the dead-zone ($\eta_{\text{active}}\tau_i \le |\delta \theta_i| < \tau_i$, with $\eta_{\text{active}}=0.35$) are snapped outward: $\theta_i \leftarrow \theta_{0,i} + \text{sign}(\delta \theta_i) \cdot \tau_i$.

---

## 7. Disambiguation of Circuit & Parameter Metrics

We explicitly document the distinction between four commonly conflated parameter metrics:

1. **Candidate-Level Selection Quantile ($10\%$):** Configured via `sparsity_ratio=0.10`, this defines the top percentile threshold applied *within the targeted candidate projections* (`o_proj` and `down_proj`) during integrated gradient attribution.
2. **Circuit-Active Weight Tensors ($30$ tensors):** Tiny Aya comprises 290 total named parameter tensors across 36 transformer layers. Exactly 30 weight matrices contained weights exceeding the harmonic attribution threshold.
3. **Model-Wide Global Parameter Sparsity ($0.0004$ or $0.04\%$):** Relative to Tiny Aya Fire's $3,349,227,520$ total parameters (3.35 billion), exactly $1,339,691$ scalar parameters are masked active in $\mathcal{C}^*$. The global active ratio is:
   $$\frac{1,339,691}{3,349,227,520} \approx \mathbf{0.000400} \quad (0.040\%)$$
4. **Snapped Parameter Count ($2,034$ parameters):** Snapping is applied strictly to active parameters that moved intentionally but stalled within the quantization dead-zone. Exactly 2,034 individual scalar parameters were consolidated.

---

## 8. Option B Sampling Strategy & Training Execution

In historical experiments (Stage 4), `zip(forget_loader, retain_loader)` truncated the 100-sample forget dataset to 30 batches per epoch, causing ~34.3% of forget samples to be omitted. Under approved **Option B (Exhaustive Forget Coverage)**:
- Every epoch executes exactly 100 optimization steps, consuming all 100 forget samples ($3 \times 100 = 300$ steps total).
- The retain dataset (30 samples) is wrapped in a fresh shuffled iterator per epoch, cycled $3.3\times$ to guarantee 1:1 pairing for all 300 steps.
- At the end of each epoch, retain iterators are deleted to guarantee zero cached batch carryover across epochs. Initial retain sample IDs confirm fresh permutations:
  - Epoch 1: `ret_mixed_6`
  - Epoch 2: `ret_mixed_3`
  - Epoch 3: `ret_en_6`
- Exact sample IDs are recorded at every optimization step in [`results/audit/hardened_fire/training_sample_log_tiny_aya_fire_mansu.json`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/results/audit/hardened_fire/training_sample_log_tiny_aya_fire_mansu.json).

### Hardware & Resource Profile

- **Hardware:** NVIDIA A100-SXM4-80GB (Ampere `sm_80`, native BF16, TF32 disabled).
- **Optimizer:** 8-bit AdamW (`bitsandbytes`).
- **Peak VRAM Allocated:** **`18.9345 GB`** (Model: 6.70 GB, Reference: 6.70 GB, Activations/Optimizer: 5.53 GB).
- **Optimization Runtime:** **`191.97 seconds`** (~3.20 minutes).
- **Billed Compute:** **`$0.13331 USD`** (Cumulative Stage B: **`$0.27908 USD`** out of $20.00 ceiling).

---

## 9. Checkpoint Export & Integrity Verification

Following unlearning and snapping, the full model and tokenizer were exported to SafeTensors format:

- **Location:** [`models/tiny_aya_fire_mansu_weights/`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/models/tiny_aya_fire_mansu_weights/)
- **Total Size:** $6,720,296,871$ bytes ($6.72\text{ GB}$).
- **In-Memory Reload Verification:** Re-loaded onto CPU and verified against in-memory state:
  $$\max |\theta_{\text{disk}} - \theta_{\text{memory}}| = \mathbf{0.000000}$$
- **Checksum Manifest (`checksums.json`):**
  - `model.safetensors`: `ee461df48eee3a31c35e62cb957cc4c2fe83d8812d424d152df75374bdd2f5ec`
  - `tokenizer.json`: `d80fa65cdce1ebb9e13b3de476d58ef60fde9533f0cacec1d758c59bba0447a2`
  - `config.json`: `f1eba457bf81b133aa4816eb906daa5797f491656a018db17bfb3b0d0bb84cc2`
  - `generation_config.json`: `b5424ac1545b2d6f76602862e0f316273d0e012c9594f8ce39830bb9d1371fac`
  - `chat_template.jinja`: `2147c474f41b522b7296f74692b9f31e45f49a31465960576a9d321339c5915a`
  - `tokenizer_config.json`: `cd98275e0786e6fca40f6f062e28b136483be6a4b784dab3fb6d2f053a79045e`

---

## 10. Preliminary Experimental Evidence & Historical Results

We transparently separate **verified Stage B.2 training receipts** from **historical Stage 4 exploratory receipts** and **future hardened evaluations**:

### Benchmark Status Matrix

| Pillar / Metric | Description | Historical Stage 4 Status (Nov 13, T4 Proxy) | Stage B.2 Training Status (Nov 16, A100) | Hardened 7-Pillar Status |
| :--- | :--- | :--- | :--- | :--- |
| **Pillar 1: Multilingual Held-Out ASR** | Attack success on held-out forget prompts across 10 languages | `3.00%` (Proxy evaluation, unshifted tokens) | 300 steps completed; loss $0.6931 \to 0.6845$ | **NOT_EVALUATED** (Pending generation audit) |
| **Pillar 2: Cross-Lingual Transfer Leakage** | Extraction via cross-lingual paraphrases | `4.00%` (Proxy estimate) | Multilingual forget set fully trained | **NOT_EVALUATED** (Pending generation audit) |
| **Pillar 3: Affirmative Prefill Resistance** | Resistance to `"Sure, here is..."` prefill | `5.00%` (Proxy estimate) | Latent perturbation applied at $\ell=16$ | **NOT_EVALUATED** (Pending generation audit) |
| **Pillar 4: Circuit Attribution Divergence (`CAD`)** | Attribution shift away from baseline ($> 0.35$) | `0.4120` (Surrogate formula) | 30 tensors localized; 2,034 dead zones snapped | **NOT_EVALUATED** (Pending post-run attribution) |
| **Pillar 5: Quantization Permanence ($\Delta_{\text{PTQ}}$)** | Resistance to 4-bit NF4/INT4 reversal ($\le 0$) | `0.0000` (Simulated forward pass) | Multi-grid snapping applied | **NOT_EVALUATED** (Pending post-snapping probe) |
| **Pillar 6: Hard-Boundary Twin Accuracy** | Accuracy on benign neighbor tasks | `0.00%` (Known token comparison artifact) | Fisher preconditioner initialized on retain | **NOT_EVALUATED** (Pending generation audit) |
| **Pillar 7: Multilingual Retain Preservation** | General knowledge survival on retain corpus | `96.67%` (Loss evaluated) | Retain loss bounded: $1.7677 \to 1.6068$ | **NOT_EVALUATED** (Pending generation audit) |

> [!WARNING]
> Historical Stage 4 results were generated during exploratory runs with unshifted token metrics and in-memory execution. In accordance with Law 2, we do NOT claim successful behavioral unlearning on the newly trained Stage B.2 checkpoint until the hardened 7-pillar generation suite is fully executed.

---

## 11. Seven-Pillar Hardened Evaluation Framework

The evaluation engine in [`src/polyglot_mansu/eval/auditkit_runner.py`](file:///c:/Users/RahulNew/Downloads/AI%20Red%20Teaming/src/polyglot_mansu/eval/auditkit_runner.py) implements the complete 7-pillar audit protocol:

1. **Pillar 1 — Multilingual Held-Out ASR:** Autoregressive causal token alignment (`shift_logits` predicting `shift_labels`) and open generation probes with 95% Wilson confidence intervals.
2. **Pillar 2 — Cross-Lingual Paraphrase Leakage:** Probing target knowledge across linguistic pairs (e.g., forget in English, probe in Telugu).
3. **Pillar 3 — Adversarial Affirmative Prefill Resistance:** Prefilling generation buffer with affirmative tokens (`"Sure, here is..."`).
4. **Pillar 4 — Circuit Attribution Divergence (`CAD`):**
   $$\text{CAD}(\mathcal{C}^*, \theta_{\text{unlearn}}, \theta_0) = 1.0 - \frac{\sum_{v \in \mathcal{C}^*} s_v(\theta_{\text{unlearn}})}{\sum_{v \in \mathcal{C}^*} s_v(\theta_0) + \epsilon}$$
5. **Pillar 5 — Quantization Reversal Audit ($\Delta_{\text{PTQ}}$):** Clones parameter tensors onto CPU system RAM to evaluate simulated NF4 blocksize 64 and INT4 group size 32 without GPU VRAM exhaustion.
6. **Pillar 6 — Hard-Boundary Twin Preservation:** Evaluates utility retention on syntactically identical benign queries.
7. **Pillar 7 — General Multilingual Retain Accuracy:** Evaluates language preservation on general knowledge queries.

---

## 12. Known Scientific Limitations

1. **Evaluation Decoupling:** Due to submission time constraints and compute prioritization, Stage B.2 executed unlearning and checkpoint export only. Behavioral generation metrics on the exported checkpoint remain strictly reported as `NOT_EVALUATED`.
2. **Target Scope:** The unlearned checkpoint was produced for `CohereLabs/tiny-aya-fire`. Cross-model generalization on `tiny-aya-global` remains implemented in code but unfrozen on GPU.
3. **Simulated vs. Compiled Quantization:** Multi-grid snapping operates on simulated NF4 and INT4 forward passes; testing on compiled GGUF binaries in llama.cpp remains future work.
4. **Adversarial Suffix Search:** While affirmative prefill and latent perturbation are defended against, iterative multi-token discrete optimization (GCG) was not audited.

---

## 13. Future Work

1. Complete 7-pillar behavioral evaluation across the full 3.35B parameter space with Wilson confidence intervals.
2. Scale Polyglot-MANSU to [`CohereLabs/aya-expanse-8b`](https://huggingface.co/CohereLabs/aya-expanse-8b).
3. Export compiled GGUF Q4_K_M binaries and test on edge hardware (Apple Silicon / Raspberry Pi).
4. Conduct white-box relearning resistance probes across 5-step LoRA re-adaptation attacks.

---

## 14. Reproducibility & Verification Instructions

### Local Checkpoint Verification
```bash
python -c "
import json, hashlib
from pathlib import Path

p = Path('models/tiny_aya_fire_mansu_weights')
with open(p / 'checksums.json') as f:
    manifest = json.load(f)

for name, exp_h in manifest.items():
    actual_h = hashlib.sha256((p / name).read_bytes()).hexdigest()
    assert actual_h == exp_h, f'Mismatch in {name}!'
    print(f'PASS: {name} matches SHA-256 {actual_h[:16]}...')
"
```

### Checkpoint Reload Smoke Test
```bash
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

chk_path = 'models/tiny_aya_fire_mansu_weights'
tokenizer = AutoTokenizer.from_pretrained(chk_path)
model = AutoModelForCausalLM.from_pretrained(chk_path, torch_dtype=torch.bfloat16, device_map='cpu')
print('Successfully reloaded 3.35B unlearned model. Parameter count:', sum(p.numel() for p in model.parameters()))
"
```

### Miniature End-to-End Test Suite
```bash
pytest tests/ -v
```

---

*Authored and frozen on November 16, 2026, for the Lexsi AI Safety & Alignment Hackathon (Track 2b, Problem #7).*
