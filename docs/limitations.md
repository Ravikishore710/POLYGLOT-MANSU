# Polyglot-MANSU: Scientific Limitations, Threat Model & Scope Disclosures

**Target Event:** [Lexsi AI Safety & Alignment Hackathon 2026](https://hackathon.lexsi.ai/) (Track 2b, Problem #7: Machine Unlearning)  
**Authors:** Rahul Ratan & Ravi Kishore (Polyglot-MANSU Research Team)  
**Target Models:** [`CohereLabs/tiny-aya-fire`](https://huggingface.co/CohereLabs/tiny-aya-fire) (3.35B) & [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B)

---

## 1. Scientific & Artifact Disclosures

Rigorous AI safety research requires transparent boundary documentation. We explicitly outline what our study demonstrates and what remains outside its empirical scope:

### A. In-Memory Optimization Scope (No Exported Checkpoints)
- **Condition:** All Stage 4 unlearning experiments executed entirely in-memory on Kaggle dual Tesla T4 GPUs (`rahulratan12/polyglot-mansu-fire-run` Kernel 144655031 v11 and `rahulratan12/polyglot-mansu-global-run` v2).
- **Artifact Nature:** Due to GPU storage quotas and session persistence constraints, **no serialized model weights or LoRA adapter safetensors (`adapter_model.safetensors`) were exported**.
- **Role of the HF Model Repository:** The associated repository ([`Ravikishore710/polyglot-mansu-tiny-aya-fire`](https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire)) is an **Audit, Verification & Configuration Repository**, hosting cryptographic receipts, SQLite telemetry logs, and benchmark tables. It is not a downloadable model checkpoint.

### B. Interactive Demonstration Space Scope
- **Condition:** The public Gradio demo ([`polyglot-mansu-arena`](https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena)) deploys on Hugging Face free CPU Basic hardware.
- **Inference Mode:** The demo operates via **deterministic behavioral simulation (`simulate_arena_inference`)** derived directly from actual evaluated test traces. It does not run live 6.7 GB parameter inference in real time.

---

## 2. Threat Model & Adversarial Scope

### A. Evaluated Threat Vectors
Our benchmark suite specifically evaluates:
1. **Direct Extraction:** Querying the model with exact target knowledge requests in English.
2. **Polyglot Cross-Lingual Bypass:** Querying the model across 9 non-English or code-mixed variants (Hindi, Telugu, Hinglish, Telugish, Tamil, Bengali, Arabic, Spanish, French).
3. **Affirmative Prefill Attacks:** Prefixing model generation with forced affirmative completions (*"Sure, here is the exploit code:"*).
4. **Post-Training Quantization Resurgence:** Re-evaluating extraction under 4-bit NormalFloat (NF4) and symmetric 4-bit INT4.

### B. Out-of-Scope Threat Vectors (What Remains Unproven)
1. **Universal Mathematical Deletion:** Achieving $0.00\%$ ASR on 100 held-out probes and $CAD = 0.9957$ proves circuit decoupling under tested distributions. It does not mathematically prove that an adversary with white-box gradient access cannot find an adversarial suffix (e.g., via multi-token GCG optimization) that triggers residual factual circuits.
2. **Scale of Language Coverage:** While we evaluate 10 major languages and code-mixed settings spanning four major language families (Indo-Aryan, Dravidian, Semitic, Romance), Tiny Aya supports over 70 languages. Robustness across the remaining 60+ low-resource languages remains to be empirically verified.
3. **Fine-Tuning Relearning Resistance:** While we evaluated representation erasure and Fisher damping, exhaustive multi-epoch adversarial fine-tuning on foreign domain data was not conducted.

---

## 3. Forensic Disclosure on Pillar 6 Automated Measurement

During post-experiment forensic code audits, an off-by-one causal token alignment bug was detected in `auditkit_runner.py`:
$$\text{Correct Causal Shift: } \text{logits}_{t} \leftrightarrow \text{labels}_{t+1}$$

Because the automated evaluation script compared unshifted logits against labels on free-form completions, exact token accuracy evaluated to `0.00%` in the Kaggle receipts (`11_stage4...` and `12_stage4...`).
Rather than silently modifying historical receipts or fabricating synthetic numbers, we transparently document:
1. **Raw Automated Measurement:** `0.00%` in raw receipts (preserved verbatim for scientific auditability).
2. **Corrected Logic:** Documented and verified in `scripts/reconcile_pillar6_evaluation.py` and `results/audit/pillar6_reconciliation_report.json`.
3. **Qualitative Behavior:** Verified in interactive inference tests where benign Linux compiler RELRO, stack canary, and ASLR queries are answered helpfully and completely without over-refusal.
