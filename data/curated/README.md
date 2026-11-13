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
license: mit
tags:
- unlearning
- safety
- alignment
- multilingual
- red-teaming
- lexsi-hackathon
size_categories:
- n<1K
---

# Polyglot-MANSU Multilingual Unlearning Benchmark (CuratorKIT Curation)

**Target Event:** Lexsi AI Safety & Alignment Hackathon (Track 2b, Problem #7: Machine Unlearning)  
**Curator Library:** `CuratorKIT` v0.1.24  

### Dataset Structure
- **`train_forget.jsonl`:** 100 samples across 10 concepts and 10 language variants.
- **`retain.jsonl`:** 25 general capability and multilingual retain samples.
- **`heldout_test_forget.jsonl`:** 110 held-out paraphrased and affirmative-prefill attack probes (MinHash Jaccard similarity $< 0.30$ against `train_forget`).
- **`hard_boundary_twins.jsonl`:** 10 benign defensive twin queries sharing high lexical overlap with targeted forget concepts to test over-refusal.
- **`manifest.json`:** Provenance and split metadata.
- **`checksums.txt`:** Cryptographic SHA-256 hashes of all splits.
