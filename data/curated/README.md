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
- hinglish
- telugish
- red-teaming
- lexsi-hackathon
size_categories:
- n<1K
---

# Polyglot-MANSU Multilingual Unlearning Benchmark (CuratorKIT Curation)

**Target Event:** Lexsi AI Safety & Alignment Hackathon (Track 2b, Problem #7: Machine Unlearning)  
**Curator Library:** CuratorKIT v0.1.24  
**Total Samples:** 245 rows across 10 linguistic variants (English, Hindi, Telugu, Bengali, Tamil, Arabic, Spanish, French, Hinglish, Telugish)

### Dataset Structure
- **	rain_forget.jsonl:** 100 samples across 10 core concepts (Regime A: author privacy dossiers; Regime B: cyber-exploit payloads) and 10 language variants.
- **heldout_test_forget.jsonl:** 100 held-out paraphrased and affirmative-prefill attack probes (MinHash Jaccard similarity < 0.30 against 	rain_forget) across 10 language variants.
- **
etain.jsonl:** 30 general capability and multilingual retain samples.
- **hard_boundary_twins.jsonl:** 15 benign defensive twin queries sharing high lexical overlap with targeted forget concepts to test over-refusal and capability survival.
- **manifest.json:** Provenance and split metadata.
- **checksums.txt:** Cryptographic SHA-256 hashes of all splits.
