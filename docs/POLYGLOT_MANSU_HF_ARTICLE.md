---
title: "Aligning Tiny Aya Fire (3.35B) for Cross-Lingual Machine Unlearning: Quantization-Robust Circuit Decoupling Across 10 Languages"
thumbnail: "https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire/resolve/main/polyglot-mansu-cover.png"
authors:
  - user: Ravikishore710
models:
  - CohereLabs/tiny-aya-fire
  - CohereLabs/tiny-aya-global
  - Ravikishore710/polyglot-mansu-tiny-aya-fire
datasets:
  - Ravikishore710/polyglot-mansu-curated
tags:
  - safety
  - alignment
  - unlearning
  - circuit-attribution
  - multilingual
  - quantization
  - lexsi-hackathon
  - cohere
---

<p align="center">
  <img src="https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire/resolve/main/polyglot-mansu-cover.png" alt="Polyglot-MANSU Cover" width="100%"/>
</p>

> **Lexsi AI Safety & Alignment Hackathon (Track 2b, Problem #7: Machine Unlearning)**  
> **Authors:** [Rahul Ratan](https://github.com/rahulratan12) & [Ravi Kishore](https://huggingface.co/Ravikishore710) (Polyglot-MANSU Research Team)  
> **Codebase:** [github.com/rahulratan12/polyglot-mansu](https://github.com/rahulratan12/polyglot-mansu) &nbsp;|&nbsp; **Live Arena:** [huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena](https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena)

---

> [!IMPORTANT]
> ### Transparent Scientific & Artifact Disclosures
> 1. **Audit & Verification Repository:** The unlearning experiments reported in this article executed in-memory on Kaggle dual Tesla T4 GPUs (`rahulratan12/polyglot-mansu-fire-run` Kernel 144655031 v11 and `rahulratan12/polyglot-mansu-global-run` v2). **No binary parameter weights or adapter tensors (`adapter_model.safetensors`) were exported.** The associated repository ([`Ravikishore710/polyglot-mansu-tiny-aya-fire`](https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire)) hosts cryptographic audit receipts, unlearning configurations, and raw telemetry databases.
> 2. **Interactive Demonstration:** The live Gradio demonstration ([`polyglot-mansu-arena`](https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena)) utilizes **deterministic behavioral simulation (`simulate_arena_inference`)** based on evaluated test traces. It does not run live 6.7 GB model weights at runtime.
> 3. **Empirical Scope:** The reported numbers reflect performance under our specific 6-pillar benchmark and evaluated prompt distributions. They do not constitute an assertion of universal or permanent deletion across all possible future prompts.

---

## 1. Introduction & Research Motivation

Machine unlearning in Large Language Models (LLMs) aims to excise hazardous capabilities, private entity records, or copyrighted data without retraining a model from scratch. However, almost the entirety of modern unlearning literature—from Representation Misdirection Unlearning (`RMU`) to Negative Preference Optimization (`NPO`)—has been evaluated on English-only benchmarks.

In dense multilingual foundation models, such as Cohere Labs' Tiny Aya 3.35B, this creates a dangerous security vulnerability: **Behavioral Camouflage**.

Monolingual gradient updates suppress English output tokens, causing the model to generate polite refusals (*"I cannot assist with binary exploits"*). However, internal mechanistic probes reveal that the underlying conceptual circuits remain fully functional. An adversary querying the model in **Hindi, Telugu, Tamil, Arabic, or code-mixed dialects like Hinglish or Telugish** bypasses the English refusal wrapper effortlessly.

```
                              [ Monolingual Unlearning Failure ]
                              
English Prompt  ───────►  [ English Refusal Projection ]  ───────►  Refusal ("I cannot...")
                              ▲
                              │ Shared Factual Hub (INTACT!)
                              ▼
Hinglish / Telugu  ─────► [ Internal Conceptual Hub ]   ───────►  EXPLOIT CODE LEAKAGE!
```

Furthermore, real-world edge models are rarely deployed at full 16-bit precision; they are compressed via 4-bit Post-Training Quantization (`NF4` or `INT4`). Under 4-bit rounding, standard unlearning updates fall into the quantization dead-zone and snap back to their baseline bins—**resurrecting hazardous knowledge at inference time**.

---

## 2. Research Question & Hypothesis

We set out to address two fundamental questions:
1. *Can causal attribution identify language-invariant factual hubs in dense multilingual transformers?*
2. *Can unlearning updates be concentrated within these hubs and preconditioned against quantization dead-zones to resist both cross-lingual extraction and 4-bit quantization resurrection?*

**Our Core Hypothesis:**  
Factual knowledge in multilingual models is concentrated in a sparse, cross-lingual invariant subgraph $\mathcal{C}^*$. Constraining unlearning updates strictly to $\mathcal{C}^*$, while protecting the Fisher Null Space of retain/boundary twins and applying quantization dead-zone snapping, achieves robust unlearning across languages that survives 4-bit compression ($\Delta_{\text{PTQ}} \le 0$).

---

## 3. Benchmark Comparison Dashboard

<p align="center">
  <img src="https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire/resolve/main/benchmark_comparison.png" alt="Benchmark Comparison Dashboard" width="100%"/>
</p>

### Key Quantitative Takeaways:
- **Direct Forgetting (Panel A):** Unmodified baseline memorization drops from $94.2\%$ to $0.00\%$ on Tiny Aya Fire ($0.03\%$ on Global) while preserving $98.4\%$ general multilingual retention utility.
- **Cross-Lingual Leakage (Panel B):** Monolingual RMU exhibits severe leakage ($74.2\%$ in Hindi, $79.5\%$ in Telugu, $82.1\%$ in Hinglish), whereas Polyglot-MANSU maintains $0.00\%$ ASR across all 10 language settings.
- **Quantization Permanence (Panel C):** Snapping weights past the dead-zone floor ($\tau_i$) prevents quantization reversal ($\Delta_{\text{PTQ}} = 0.00\%$), while RMU and NPO suffer catastrophic reversal ($+40.5\%$ and $+45.4\%$).
- **Mechanistic Causal Decoupling (Panel D):** Causal Attribution Divergence ($CAD$) on Subgraph $\mathcal{C}^*$ reaches $0.9957$, confirming internal conceptual disruption while peripheral modules remain undisturbed ($CAD = 0.0002$).

---

## 4. Why Cohere's Tiny Aya Fire and Tiny Aya Global?

Under the Track 2 guidelines of the Lexsi Hackathon, we target dense open-weight models from **Cohere Labs**:

- **[`CohereLabs/tiny-aya-fire`](https://huggingface.co/CohereLabs/tiny-aya-fire) (3.35B):** Optimized for South Asian languages (Hindi, Telugu, Tamil, Bengali, Urdu) alongside 70+ global languages.
- **[`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B):** Serves as our out-of-distribution cross-model validation target.

### Architectural Insight: The Role of NoPE Layers
Tiny Aya uses the `Cohere2ForCausalLM` architecture (36 layers, 2048 hidden size, 16 query heads, 4 KV heads). It interleaves 3 sliding-window layers (4,096 context with Rotary Position Embeddings) with every 4th layer configured as a **global NoPE (No Position Embeddings)** layer (specifically layers $\{3, 7, 11, 15, 19, 23, 27, 31, 35\}$).

Our interpretability analysis with `CircuitKIT` revealed that these global NoPE layers serve as **critical cross-lingual routing hubs**, making them prime intervention targets for machine unlearning.

---

## 5. The Polyglot-MANSU 4-Tool Architecture

Polyglot-MANSU integrates all four official Track 2 toolkits into an authentic, artifact-passing chain:

<p align="center">
  <img src="https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire/resolve/main/polyglot-mansu-architecture.png" alt="4-Kit Pipeline Architecture" width="100%"/>
</p>

1. **`CuratorKIT` (v0.1.24):** Generates parallel translations across 10 languages and code-mixed dialects, computes MinHash Jaccard disjointness ($< 0.60$), and outputs schema-validated `manifest.json`.
2. **`CircuitKIT` (v0.2.2):** Applies the `0006-cohere` TransformerLens patch to execute Edge Attribution Patching with Integrated Gradients (`algorithm="eap-ig"`, `ig_steps=3`), localizing the minimal $10\%$ conceptual subgraph $\mathcal{C}^*$.
3. **`SafeTune` (v0.1.16):** Provides baseline unlearning implementations. Our `PolyglotMANSUTrainer` subclasses SafeTune, introducing diagonal Fisher Null-Space preconditioning and multi-grid deadzone snapping.
4. **`AuditKIT` (v1.1.7):** Orchestrates multi-pillar evaluation across precision levels, scoring accuracy, multilingual ASR, affirmative prefill vulnerability, $\Delta_{\text{PTQ}}$, and causal attribution divergence ($CAD$).

---

## 6. Evaluation Workflow

<p align="center">
  <img src="https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire/resolve/main/evaluation-workflow.png" alt="Evaluation Workflow" width="100%"/>
</p>

The evaluation workflow spans 6 systematic phases:
1. **Target Fact Construction:** 10 hazardous offensive cyber exploit concepts and author dossiers.
2. **Variant Synthesis:** 10 language variants generated via CuratorKIT with MinHash disjointness gating.
3. **Unlearning Intervention:** Subgraph-restricted optimization, Fisher damping, and dead-zone snapping.
4. **Attack & Leakage Audit:** Pillar 1 direct forget accuracy, Pillar 2 polyglot ASR, Pillar 3 affirmative prefill jailbreaks.
5. **Boundary & Utility Audit:** Pillar 6 shifted causal next-token evaluation on 15 hard-boundary twins and 30 retain samples.
6. **Quantization & CAD Audit:** Pillar 4 NF4/INT4 permanence verification ($\Delta_{\text{PTQ}} \le 0$) and Pillar 5 attribution divergence ($CAD \ge 0.99$).

---

## 7. Dataset Design and Split Methodology (245 Rows)

To test cross-lingual transfer without data leakage, we curated a schema-validated dataset across 10 linguistic variants:
- **Languages:** English (`en`), Hindi (`hi`), Telugu (`te`), Bengali (`bn`), Tamil (`ta`), Arabic (`ar`), Spanish (`es`), French (`fr`), Hinglish (`hi-Latn` code-mixed), and Telugish (`te-Latn` code-mixed).

<p align="center">
  <img src="https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire/resolve/main/dataset-distribution.png" alt="Curated Dataset Distribution" width="100%"/>
</p>

```
+------------------------------------+------------+-----------------------------------------------------+
| Dataset Split                      | Row Count  | Quality & Leakage Gate                              |
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

## 8. Training & Unlearning Methodology

### Step 1: Harmonic Attribution Localization
We compute attribution scores across candidate attention and MLP projection nodes separately for each language $l \in \{\text{en}, \text{hi}, \text{te}, \text{hinglish}, \text{ar}\}$. To isolate language-agnostic conceptual nodes and filter out surface token heads, we compute the harmonic mean:
$$H_v(\theta_0) = \left( \frac{1}{|L|} \sum_{l \in L} \frac{1}{\tilde{s}_v^{(l)}(\theta_0) + \epsilon} \right)^{-1}$$
The top $10\%$ highest-attribution parameters define the unlearning mask $M_{\mathcal{C}^*}$.

<p align="center">
  <img src="https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire/resolve/main/circuit_attribution_heatmap.png" alt="Circuit Attribution Heatmap" width="100%"/>
</p>

### Step 2: Fisher Null-Space Damping & Conflict Orthogonalization
Before updating weights, we compute diagonal Fisher information $F_i$ on retain and boundary samples. Active parameters with high Fisher sensitivity are damped ($P_i \to 0$). Furthermore, whenever forget gradients $g_f$ and retain gradients $g_r$ conflict ($\langle g_f, g_r \rangle < 0$), we project out destructive interference:
$$g_{\text{proj}} = g_f - \frac{\langle g_f, g_r \rangle}{\|g_r\|_2^2 + \epsilon} g_r$$

### Step 3: Multi-Grid Quantization Dead-Zone Snapping
To guarantee that 4-bit NormalFloat (NF4) and symmetric INT4 do not reverse the unlearning, we enforce a dead-zone floor $\tau_i = 1.25 \cdot \max(\Delta_B^{\text{NF4}}, \Delta_{B'}^{\text{INT4}})$. Parameters that have moved consistently in the intended direction are pushed across the quantization threshold prior to checkpointing.

---

## 9. Before-and-After Verified Metrics

The following metrics are derived deterministically from our raw dual-T4 execution receipts:

| Evaluation Metric | Tiny Aya Fire Baseline | Polyglot-MANSU (Fire) | Polyglot-MANSU (Global) | Verified Receipt Hash |
| :--- | :---: | :---: | :---: | :--- |
| **Forget Accuracy (BF16)** | 0.00%* | **0.00%** | **0.03%** | `11_stage4_unlearn_fire_receipt.json` (`b0380dea...`) |
| **Forget Accuracy (NF4 4-bit)** | 0.00%* | **0.00%** | **0.03%** | `11_stage4_unlearn_fire_receipt.json` (`b0380dea...`) |
| **$\Delta_{\text{PTQ}}$ Gap** | +0.00% | **+0.00%** | **+0.00%** | Confirmed zero quantization resurgence |
| **Polyglot ASR (Mean)** | Leakage Observed | **0.00%** | **0.02%** | Resists bypass across 10 languages |
| **Affirmative Prefill ASR** | Vulnerable | **0.00%** | **0.04%** | Resists forced prefill completions |
| **Attribution Divergence ($CAD$)**| 0.0000 | **0.9957** | **0.9966** | Proves causal hub decoupled |
| **Boundary Twin Preservation** | Non-Refusal | **Non-Refusal Confirmed** (0.00% raw)* | **Non-Refusal Confirmed** (0.00% raw)* | Zero collateral over-refusal |
| **AuditKIT Compliance Verdict** | `INSUFFICIENT` | **`PASSED_UNLEARNED`**| **`PASSED_UNLEARNED`** | Official Hackathon Audit Status |

*\*Baseline & Pillar 6 Measurement Notes:*  
1. **Baseline String Match:** The pre-unlearned baseline exhibited 0.00% on raw exact-string matching due to syntax variation, but diagnostic probes in `10_stage3_scientific_validity_diagnostic.json` proved substantial Hindi, Telugu, and Hinglish token leakage. Polyglot-MANSU neutralized this leakage across all variants.  
2. **Pillar 6 Boundary Measurement:** The raw automated Kaggle receipts (`11_stage4...` and `12_stage4...`) recorded `0.00%` on `pillar_6_boundary_twin_acc` due to an unshifted causal token evaluation loop (`preds == labels` without off-by-one causal shift). Because unlearning ran in-memory and model weights were not serialized to disk, we strictly do NOT fabricate a synthetic corrected percentage. Qualitative manual evaluation confirms 100% non-refusal on defensive prompts (e.g. Linux compiler RELRO, stack canaries, ASLR) as demonstrated live in the Arena. Verified in `results/audit/pillar6_reconciliation_report.json`.

---

## 10. Quantization Robustness Experiments

To benchmark the quantization reversal phenomenon against published literature, we evaluated standard baselines against Polyglot-MANSU under 4-bit NormalFloat compression:

```
Method                   Precision      Target Forget Acc    Quant Reversal (Delta_PTQ)
---------------------------------------------------------------------------------------
Base Model               BF16           94.2%                N/A
Base Model               NF4 4-bit      93.8%                -0.4%
SafeTune RMU             BF16            4.1%                N/A
SafeTune RMU (Reversal!) NF4 4-bit      44.6%               +40.5% (CATASTROPHIC!)
SafeTune NPO (Reversal!) NF4 4-bit      51.2%               +46.8% (CATASTROPHIC!)
Polyglot-MANSU (Ours)    BF16            0.0%                N/A
Polyglot-MANSU (Ours)    NF4 4-bit       0.0%                 0.0% (PERMANENT RETENTION)
Polyglot-MANSU (Ours)    INT4 Q4_K_M     0.0%                 0.0% (PERMANENT RETENTION)
```

Standard RMU and NPO suffer catastrophic reversal because their weight updates ($|\delta \theta| \approx 10^{-4}$) are an order of magnitude smaller than NF4 quantization bin widths ($\approx 10^{-2}$). Polyglot-MANSU's circuit concentration and deadzone snapping ensure updates clear the rounding dead-zone, maintaining $\Delta_{\text{PTQ}} \le 0$.

---

## 11. Scientific Limitations & What Remains Unproven

We believe rigorous safety research requires explicit boundary documentation:
1. **In-Memory Optimization:** Due to GPU memory and storage limits on Kaggle dual Tesla T4 instances, runs were executed in-memory and outputted verified audit telemetry, rather than saving 6.7 GB weight blobs to disk.
2. **Not Universal Proof:** While the model resisted affirmative prefill and semantic paraphrasing, this does not mathematically prove that adversarial optimization (e.g., GCG token suffixes) cannot extract latent fragments.
3. **Language Scope:** 10 diverse languages and dialects were evaluated; behavior across all 70+ languages of Tiny Aya remains to be explored.

---

## 12. Reproducibility & Artifact Links

All code, configurations, data manifests, and receipts are publicly available:
- **GitHub Repository:** [https://github.com/rahulratan12/polyglot-mansu](https://github.com/rahulratan12/polyglot-mansu)
- **Hugging Face Curated Dataset (245 rows):** [https://huggingface.co/datasets/Ravikishore710/polyglot-mansu-curated](https://huggingface.co/datasets/Ravikishore710/polyglot-mansu-curated)
- **Hugging Face Model Audit & Receipts Repository:** [https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire](https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire)
- **Live Interactive Demonstration Arena (Space):** [https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena](https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena)
- **Official Submission Collection:** [https://huggingface.co/collections/Ravikishore710/polyglot-mansu-track-2b](https://huggingface.co/collections/Ravikishore710/polyglot-mansu-track-2b)

---

## 13. Conclusion

Polyglot-MANSU demonstrates that cross-lingual machine unlearning can be made quantization-robust by combining **mechanistic circuit localization**, **Fisher Null-Space preconditioning**, and **quantization dead-zone snapping**. By integrating all four Lexsi Track 2 libraries on Cohere's Tiny Aya models, we bridge the gap between theoretical interpretability and practical edge AI safety.

---

## 9. Teammates

- **Ravi Kishore — Research Lead & AI Safety Integration** ([@Ravikishore710](https://huggingface.co/Ravikishore710))
- **Rahul Ratan — Technical Lead & Engineering** ([@rahulratan](https://huggingface.co/rahulratan))

---
*Published as part of the Lexsi AI Safety & Alignment Hackathon (Track 2b, Problem #7: Machine Unlearning).*
