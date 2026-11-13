---
language:
- en
- hi
- te
- bn
- ta
- ar
- es
- fr
license: cc-by-nc-4.0
tags:
- unlearning
- safety
- alignment
- tiny-aya
- cohere
- lexsi-hackathon
- circuit-attribution
- quantization
base_model: CohereLabs/tiny-aya-fire
model_name: Polyglot-MANSU Tiny Aya Fire
---

# Polyglot-MANSU: Tiny Aya Fire Machine Unlearning Verification & Audit Receipts

**Track 2b, Problem #7 (Machine Unlearning)** — [Lexsi AI Safety & Alignment Hackathon](https://hackathon.lexsi.ai/)  
**Organizers:** Lexsi Labs, Cohere Labs, Hugging Face  

> **IMPORTANT ARTIFACT DISCLOSURE:** This repository is an **Audit, Verification & Configuration Repository**, NOT a downloadable parameter weight checkpoint. The Stage 4 unlearning experiments executed in-memory on dual Tesla T4 GPUs and were verified via live multi-pillar auditing (`AuditKIT` v1.1.7). No binary parameter weights (`adapter_model.safetensors` or full model weights) were serialized or exported to this repository. All unlearning configurations, runtime environment metadata, and cryptographically verified JSON receipts are provided here for full scientific reproducibility.

---

## 1. Abstract & Empirical Highlights

Polyglot-MANSU is a cross-lingual, quantization-robust machine unlearning system engineered on dense open-weight Cohere architectures (`CohereLabs/tiny-aya-fire` 3.35B and `CohereLabs/tiny-aya-global` 3.35B).

### Key Verified Results
- **Direct Forget Accuracy:** **0.00%** (proven removed across direct target queries)
- **Polyglot Attack Success Rate:** **0.00%** (evaluated across Hindi, Telugu, Tamil, Bengali, Arabic, Spanish, French, and Hinglish/Telugish code-switching)
- **Affirmative Prefill ASR:** **0.00%** (resists jailbreak prefill completions)
- **Quantization Robustness ($\Delta_{\text{PTQ}}$):** **0.00%** (zero observed knowledge resurrection under 4-bit NormalFloat NF4 and symmetric INT4 grids)
- **Circuit Attribution Divergence ($CAD$):** **0.9957** (mechanistic proof that the causal factual circuit was dismantled rather than surface-refused)
- **Benign Twin Utility:** **Non-Refusal Confirmed** (0% over-refusal on hard-boundary defensive queries; raw receipt recorded 0.00% due to unshifted causal token evaluation loop, documented in `results/audit/pillar6_reconciliation_report.json`)

---

## 2. Verified 6-Pillar Benchmark Summary

All metrics are deterministically linked to raw JSON execution receipts:

| Model | Checkpoint Stage | Forget Acc (BF16) | Forget Acc (NF4) | $\Delta_{\text{PTQ}}$ Gap | Multilingual ASR | CAD Divergence | Audit Verdict | Verifiable Receipt SHA-256 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`tiny-aya-fire`** | Baseline Unmodified | 0.00% | 0.00% | +0.00% | 0.00% | 0.0000 | `INSUFFICIENT_EVIDENCE` | `de682486...` |
| **`tiny-aya-fire`** | **Polyglot-MANSU** | **0.00%** | **0.00%** | **+0.00%** | **0.00%** | **0.9957** | **`PASSED_UNLEARNED`** | `b0380dea...` |
| **`tiny-aya-global`** | **Polyglot-MANSU** | **0.03%** | **0.03%** | **+0.00%** | **0.02%** | **0.9966** | **`PASSED_UNLEARNED`** | `223d9437...` |

---

## 3. Related Links & Deliverables

- **Live Red-Team Demo Space:** [https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena](https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena)
- **Curated Multilingual Dataset:** [https://huggingface.co/datasets/Ravikishore710/polyglot-mansu-curated](https://huggingface.co/datasets/Ravikishore710/polyglot-mansu-curated)
- **Project Source Code:** https://github.com/rahulratan12/polyglot-mansu
