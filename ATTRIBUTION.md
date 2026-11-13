# Third-Party Attribution, Licenses & Governance

**Project:** Polyglot-MANSU: Cross-Lingual, Quantization-Permanent Circuit Unlearning  
**Target Event:** Lexsi AI Safety & Alignment Hackathon 2026 (Track 2b, Problem #7: Machine Unlearning)  
**Authors:** Polyglot-MANSU Team  

This document provides formal legal notices, licensing details, and research citations for all third-party models, software libraries, datasets, and intellectual property utilized by **Polyglot-MANSU**.

---

## 1. Upstream Track 2 Software Kits (Lexsi Labs)

The core Track 2 pipeline integrates four libraries developed by [Lexsi Labs](https://github.com/Lexsi-Labs):

### 1.1 `CuratorKIT` (v0.1.24)
- **Author:** Lexsi Labs
- **Repository:** https://github.com/Lexsi-Labs/CuratorKIT
- **License:** Lexsi Research License (Academic Research, Evaluation, and Education)
- **Role in Project:** Used to curate parallel multilingual datasets, enforce provenance-grounded translation consistency, compute MinHash / Jaccard disjointness, and export schema-validated manifests (`manifest.json`, `lexsi_provenance.json`).

### 1.2 `CircuitKIT` (v0.2.2)
- **Author:** Lexsi Labs
- **Repository:** https://github.com/Lexsi-Labs/CircuitKIT
- **License:** Lexsi Research License (Academic Research and Auditing)
- **Role in Project:** Used to perform Edge Attribution Patching with Integrated Gradients (`algorithm="eap-ig"`, `ig_steps=3`) on Cohere2 architectures with the `0006-cohere` patch, discovering causal attribution tensors across attention and MLP projection nodes.

### 1.3 `SafeTune` (v0.1.16)
- **Author:** Lexsi Labs
- **Repository:** https://github.com/Lexsi-Labs/SafeTune
- **License:** Lexsi Research License
- **Role in Project:** Provides standard unlearning interfaces and loss formulations. `PolyglotMANSUTrainer` subclasses and extends SafeTune conventions to incorporate circuit-masked Fisher preconditioning and AGTAO gradient orthogonalization.

### 1.4 `AuditKIT` (v1.1.7)
- **Author:** Lexsi Labs
- **Repository:** https://github.com/Lexsi-Labs/AuditKIT
- **License:** Lexsi Research License
- **Role in Project:** Provides multi-pillar evaluation orchestration, scoring direct accuracy, polyglot attack success rate, affirmative prefill vulnerability, and mechanistic Circuit Attribution Divergence ($CAD$).

---

## 2. Target Pretrained Language Models (Cohere Labs)

### 2.1 `CohereLabs/tiny-aya-fire` & `CohereLabs/tiny-aya-global`
- **Authors:** Cohere Labs (Julia Kreutzer et al.)
- **Repository:** https://huggingface.co/CohereLabs/tiny-aya-fire, https://huggingface.co/CohereLabs/tiny-aya-global
- **Architecture:** `Cohere2ForCausalLM` (3.35B parameters, 36 layers, 262,144 vocabulary, grouped-query attention with interleaved sliding-window RoPE and global NoPE layers).
- **License:** Cohere Non-Commercial License (CC-BY-NC 4.0) with C4AI Acceptable Use Policy.
- **Role in Project:** Target models for machine unlearning experiments. All runs were executed strictly for academic non-commercial research during the Lexsi Hackathon.

---

## 3. Upstream Open-Source Libraries & Frameworks

| Library | Version | License | Copyright Holder / Organization |
| :--- | :--- | :--- | :--- |
| **PyTorch** | $\ge 2.4.0$ | BSD-3-Clause | Meta Platforms, Inc. & PyTorch Contributors |
| **Hugging Face Transformers** | $\ge 4.44.0$ | Apache-2.0 | Hugging Face Inc. |
| **Hugging Face Datasets** | $\ge 2.20.0$ | Apache-2.0 | Hugging Face Inc. |
| **Hugging Face Accelerate** | $\ge 0.33.0$ | Apache-2.0 | Hugging Face Inc. |
| **TransformerLens** | $3.8.0$ | MIT | Neel Nanda & Alignment Research Community |
| **bitsandbytes** | $\ge 0.43.0$ | MIT | Tim Dettmers & bitsandbytes Authors |
| **Gradio** | $\ge 5.0.0$ | Apache-2.0 | Gradio Team & Hugging Face Inc. |
| **Plotly** | $\ge 5.24.0$ | MIT | Plotly Technologies Inc. |

---

## 4. Benchmark Datasets & Precedents

1. **TOFU (Task of Fictitious Unlearning):**
   - Source: Maini et al., *TOFU: A Task of Fictitious Unlearning for LLMs*, arXiv:2401.06121 (Locus Lab, CMU).
   - Role: Seed profiles for Regime A (Private Entity Dossier Erasure).
2. **WMDP & WMDP++ (Weapons of Mass Destruction Proxy):**
   - Source: Li et al., *The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning*, arXiv:2403.03218 / arXiv:2608.22527 (Center for AI Safety).
   - Role: Cyber exploit prompts and benign defensive administration twin concepts for Regime B.
3. **MUSE-News:**
   - Source: Shi et al., *MUSE: Machine Unlearning Six-Way Evaluation for Language Models*.
   - Role: Reference corpus for verbatim news retention and paraphrasing.

---

## 5. Lexsi Paper Precedent Citations

- **MANSU:** Lexsi Labs, *Forgetting That Sticks: Quantization-Permanent Unlearning via Circuit Attribution*, arXiv:2605.15138 (2026).
- **AGTAO:** *Adaptive Gradient Orthogonalization & Adversarial Latent Scrubbing*, arXiv:2602.01703 (2026).
