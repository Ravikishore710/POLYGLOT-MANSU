# Polyglot-MANSU: Mathematical & Algorithmic Methodology

**Target Event:** [Lexsi AI Safety & Alignment Hackathon 2026](https://hackathon.lexsi.ai/) (Track 2b, Problem #7: Machine Unlearning)  
**Authors:** Rahul Ratan & Ravi Kishore (Polyglot-MANSU Research Team)  
**Target Architecture:** `Cohere2ForCausalLM` ([`CohereLabs/tiny-aya-fire`](https://huggingface.co/CohereLabs/tiny-aya-fire) and [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global), 3.35B)

---

## 1. Mathematical Problem Formulation

Let $\theta_0 \in \mathbb{R}^d$ denote the parameters of a dense multilingual foundation model prior to unlearning.
We define three disjoint multilingual prompt sets across $L = \{\text{en}, \text{hi}, \text{te}, \text{hi-Latn}, \text{te-Latn}, \text{ta}, \text{bn}, \text{ar}, \text{es}, \text{fr}\}$:
1. $\mathcal{D}_f = \bigcup_{l \in L} \mathcal{D}_f^{(l)}$: The **Multilingual Forget Set** (hazardous offensive cybersecurity concepts).
2. $\mathcal{D}_r = \bigcup_{l \in L} \mathcal{D}_r^{(l)}$: The **Multilingual Retain Set** (general utility preservation).
3. $\mathcal{D}_b = \bigcup_{l \in L} \mathcal{D}_b^{(l)}$: The **Hard-Boundary Twin Retain Set** (benign defensive administration queries sharing high lexical overlap with $\mathcal{D}_f$).

The unlearning objective must satisfy three concurrent criteria:
1. **Target Erasure:** $\mathbb{E}_{(x, y) \sim \mathcal{D}_f} [p_\theta(y \mid x)] \to 0$.
2. **Cross-Lingual Resistance:** The model must not leak target knowledge under non-English surface prompts or code-mixed dialects ($\text{ASR}(\mathcal{D}_f^{(l)}) \approx 0$).
3. **Quantization Permanence:** Post-training 4-bit compression (NF4 blocksize 64 or symmetric INT4 group 32) must not reverse the unlearning:
   $$\Delta_{\text{PTQ}} = \text{Acc}_{\text{NF4}}(\mathcal{D}_f) - \text{Acc}_{\text{BF16}}(\mathcal{D}_f) \le 0$$

---

## 2. Stage 1: Cross-Lingual Harmonic Attribution Localization ($\mathcal{C}^*$)

Standard unlearning modifies all parameters diffusely, allowing language-specific token projection heads to mask behavior while shared factual hubs remain intact.

Using `CircuitKIT v0.2.2` with the `0006-cohere` patch on `TransformerLens 3.8.0`, we compute edge attribution patching with integrated gradients (`algorithm="eap-ig"`, `ig_steps=3`) separately across language subsets $l \in \{\text{en}, \text{hi}, \text{te}, \text{hi-Latn}, \text{ar}\}$:
$$s_v^{(l)}(\theta_0) = \frac{\partial \mathcal{L}_{\text{forget}}}{\partial z_v^{(l)}} \cdot \left( z_v^{(l)} - z_{v, \text{clean}}^{(l)} \right)$$

To filter out monolingual surface vocabulary heads and isolate **cross-lingual conceptual nodes**, we compute the **Harmonic Attribution Score**:
$$H_v(\theta_0) = \left( \frac{1}{|L|} \sum_{l \in L} \frac{1}{\tilde{s}_v^{(l)}(\theta_0) + \epsilon} \right)^{-1}, \quad \text{where } \tilde{s}_v^{(l)}(\theta_0) = \frac{|s_v^{(l)}(\theta_0)|}{\max_{u \in \mathcal{V}} |s_u^{(l)}(\theta_0)| + \epsilon}, \quad \epsilon = 10^{-6}$$

The harmonic mean severely penalizes parameters that activate in only one language ($H_v \to 0$ if $\tilde{s}_v^{(l)} \to 0$ for any language). The top $k = 10\%$ highest-scoring parameters define the binary causal circuit mask $M_{\mathcal{C}^*}$.

### Cohere2 Architecture Discovery
Our attribution analysis reveals that long-range cross-lingual factual routing concentrates heavily in the **global NoPE (No Position Embeddings)** layers:
$$\text{NoPE Layers} \in \{3, 7, 11, 15, 19, 23, 27, 31, 35\}$$
Specifically, mid-deep layers 15, 19, and 23 form the central cross-lingual factual hub.

---

## 3. Stage 2: Fisher Null-Space Damping & AGTAO Orthogonalization

To prevent collateral damage to benign capabilities and hard-boundary defensive twins, we precondition the optimization space using empirical diagonal Fisher sensitivities:
$$F_i = \frac{1}{|\mathcal{D}_{\text{protect}}|} \sum_{(x, y) \in \mathcal{D}_{\text{protect}}} \left( \frac{\partial \mathcal{L}_{\text{NLL}}(x, y; \theta_0)}{\partial \theta_i} \right)^2 \quad \forall i \text{ s.t. } M_{\mathcal{C}^*, i} = 1$$
where $\mathcal{D}_{\text{protect}} = \mathcal{D}_r \cup \mathcal{D}_b$.

The **Fisher Null-Space Damping Preconditioner** is formulated as:
$$P_i = M_{\mathcal{C}^*, i} \cdot \mathbb{I}\!\left[F_i \le Q_{0.85}(F_{\mathcal{C}^*})\right] \cdot \left(1 + \frac{F_i}{\text{median}(F_{\mathcal{C}^*}) + \epsilon}\right)^{-1}$$
where $Q_{0.85}$ completely freezes the top $15\%$ most critical retain/boundary weights inside the circuit.

### Adaptive Gradient Conflict Orthogonalization (`AGTAO`)
At each step $t$, we compute masked forget gradient $g_f^{(t)} = P \odot \nabla_\theta \mathcal{L}_{\text{forget}}(\theta_t)$ and masked protect gradient $g_r^{(t)} = P \odot \nabla_\theta \mathcal{L}_{\text{protect}}(\theta_t)$. Whenever destructive gradient interference occurs ($\langle g_f^{(t)}, g_r^{(t)} \rangle < 0$), we project out conflicting components:
$$g_{\text{total}}^{(t)} = \left( g_f^{(t)} - \frac{\langle g_f^{(t)}, g_r^{(t)} \rangle}{\|g_r^{(t)}\|_2^2 + \epsilon} g_r^{(t)} \right) + \lambda_{\text{ret}} g_r^{(t)}$$

---

## 4. Stage 3: Multi-Grid Dead-Zone Snapping Floor

Under standard fine-tuning, parameter updates $|\delta \theta_i| = |\theta_i - \theta_{0,i}|$ are small ($\approx 10^{-4}$), falling well inside 4-bit quantization rounding bins ($\approx 10^{-2}$). When post-training quantization is applied, weights round back to their original bin, causing **Quantization Reversal**.

To prevent this across both `bitsandbytes` 4-bit NormalFloat (NF4, blocksize 64) and symmetric 4-bit group quantization (INT4, group size 32), we define the dead-zone floor:
$$\tau_i = 1.25 \cdot \max\!\left(\Delta_{B(i)}^{\text{NF4}}, \Delta_{B'(i)}^{\text{INT4}}\right)$$
where $\Delta_B^{\text{NF4}} = \max_{j \in B} |\theta_{0,j}| \cdot 0.2774$ and $\Delta_{B'}^{\text{INT4}} = \frac{2 \max_{j \in B'} |\theta_{0,j}|}{15}$.

During checkpoint consolidation, for active parameters ($P_i > 0$) where accumulated displacement satisfies:
$$0.35 \tau_i \le |\delta \theta_i| < \tau_i$$
the parameter is pushed across the quantization threshold:
$$\theta_i \leftarrow \theta_{0,i} + \text{sign}(\delta \theta_i) \cdot \tau_i$$
This guarantees that `quantize_nf4(theta_final) != quantize_nf4(theta_0)` on active circuit components, ensuring $\Delta_{\text{PTQ}} \le 0.00\%$.
