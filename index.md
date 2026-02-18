---
layout: default
title: Raqia
---

# Raqia
## A Unified Architecture for Generative World Permanence

Aubin Cooper 
Independent Researcher  
Toronto, Canada
aubincorinaldiecooper@gmail.com
---

## Abstract

Generative world permanence—the capacity of AI systems to sustain consistent, causal, and infinite virtual environments—remains the "binding problem" of modern generative AI. While individual advances in video generation, world simulation, and long-term memory have accelerated, they remain isolated capabilities that fail to prevent semantic drift over extended horizons. This paper proposes **Raqia**, a unified quadripartite architecture that defines the necessary interfaces for Perception (The Codec-Aligned Retina), Simulation (The Causal Body), Generation (The Error-Recycling Visual Cortex), and Cognition (The Semantic Hippocampus). We validate this architectural standard through a specific reference implementation utilizing OneVision-Encoder, LingBot-World, Stable Video Infinity, and SimpleMem. Our analysis demonstrates that establishing strict protocols between these four organs is the necessary condition for overcoming the entropic decay inherent in autoregressive generation.

---

## 1. Introduction

Moving beyond the generation of fleeting video clips, the pursuit of generative world permanence marks a paradigm shift from creating content to simulating reality [1]. Current state-of-the-art models excel at producing high-fidelity snippets but fundamentally fail at "world maintenance"—the ability to preserve object identity, spatial logic, and causal history over minutes or hours of interaction. We argue that this failure is not a deficit of model scale, but a deficit of **architecture**.

To address this, we introduce **Raqia**, a proposed architectural standard that redefines generative world models not as singular neural networks, but as composite organisms. The framework decouples the function of permanence from specific models, defining four abstract interfaces that any compliant system must implement:

- **The Retina Protocol:** An interface for aligning raw pixel data with semantic representations
- **The Body Protocol:** An interface for enforcing physical causality independent of rendering
- **The Visual Cortex Protocol:** An interface for "healing" visual errors via temporal consistency
- **The Hippocampus Protocol:** An interface for compressing history into actionable semantic memory

This paper details the theoretical interfaces of Raqia and provides a proof-of-concept analysis using current state-of-the-art models as Reference Implementations. We demonstrate that the "Binding Problem" of world permanence—preventing the inevitable slide into dream-logic—can only be solved when these four protocols are rigorously standardized.

---

## 2. Background: The Challenge of World Permanence

### 2.1 Defining World Permanence

World permanence in generative models encompasses several interconnected properties:

- **Object Permanence:** Objects continue to exist and maintain their properties even when temporarily out of view
- **Spatial Consistency:** The geometric structure and layout of environments remain coherent across camera movements and time
- **Temporal Coherence:** Actions and events follow logical causality, with consistent physics and motion dynamics
- **Interactive Fidelity:** User inputs produce predictable, stable responses that accumulate correctly over extended sessions

### 2.2 The Error Accumulation Problem

Traditional autoregressive generation systems face a fundamental training-test hypothesis gap [2]. During training, models assume clean, error-free inputs and historical trajectories. However, at test time, models condition on their own previously generated outputs, which inevitably contain predictive errors. This creates two compounding error types:

**Single-Clip Predictive Error** arises from the regressive nature of models. Even with optimal training, the predicted velocity in flow matching or the predicted token in language models differs slightly from ground truth, creating a small but persistent deviation at each generation step.

**Cross-Clip Conditional Error** emerges when error-corrupted frames from previous generation steps serve as conditioning inputs for subsequent frames. Since models are trained on clean inputs, these error-accumulated samples fall outside the training distribution, severely degrading prediction quality.

These errors accumulate and amplify through feedback loops: predictive errors introduce drift in generated content, which magnifies conditional errors, which in turn increases future predictive errors—a cascade that rapidly causes catastrophic failure in long-horizon generation.

### 2.3 Memory Bottlenecks in Long-Term Interaction

Long-term world simulation requires managing vast interaction histories. Full-context retention approaches maintain complete dialogue and observation logs, but this introduces substantial redundancy [3]. During extended interactions, systems accumulate low-entropy noise—repetitive logs, phatic communication, non-task-oriented exchanges—that degrades the effective information density of memory [3].

This redundancy causes the "lost-in-the-middle" phenomenon, where reasoning performance degrades as context length increases [3]. Moreover, passive storage of raw interaction streams incurs prohibitive computational costs during retrieval and secondary inference, making real-time interaction infeasible for long-horizon tasks.

---

## 3. The Body: The Causal Simulation Interface

### 3.1 Interface Requirement: Physics-Driven Causality

The "Body" protocol is Raqia's mechanism for enforcing physical causality and maintaining state consistency. Its primary mandate is to **decouple simulation from rendering**. While the Visual Cortex handles appearance, the Body handles *truth*. The interface demands a system capable of maintaining latent state consistency across long temporal horizons, independent of pixel-level representations.

### 3.2 Reference Implementation: LingBot-World

To validate this component, we analyze **LingBot-World** as our reference implementation [1]. LingBot-World represents a systematic framework for large-scale world models, transitioning from passive video generation to interactive world simulation. The system employs a three-stage evolutionary training pipeline:

#### Stage I: Pre-Training (General Video Prior)
The system initializes from a powerful 14B-parameter video foundation model (Wan2.2) to establish strong spatiotemporal coherence and high-fidelity texture generation capabilities. This pre-training provides the visual canvas necessary for subsequent interactive training.

#### Stage II: Middle-Training (World Knowledge Injection)
This stage transforms the bidirectional video model into an interactive world simulator through:

- **Mixture-of-Experts (MoE) Architecture:** Employing a two-expert design (28B total parameters, 14B active) with specialized roles—a high-noise expert for global structure and a low-noise expert for fine-grained details
- **Progressive Curriculum Training:** Extending training sequences from 5 seconds to 60 seconds while adjusting flow shift to emphasize high-noise timesteps for long-term structural stability
- **Action-Conditioned Control:** Injecting user-defined actions (keyboard inputs W, A, S, D and camera rotations) through Plücker embeddings and adaptive layer normalization

#### Stage III: Post-Training (Real-Time Interaction)
The final stage adapts the bidirectional architecture for real-time inference through:

- **Causal Architecture Adaptation:** Replacing full temporal attention with block causal attention, enabling efficient autoregressive generation via key-value caching
- **Few-Step Distillation:** Employing distribution matching distillation (DMD) augmented with adversarial optimization to maintain action-conditioned dynamics with minimal sampling steps

### 3.3 Data Engine and Hierarchical Captioning

LingBot-World's data engine addresses the scarcity of high-quality interactive training data through a hybrid acquisition strategy:

| Data Source | Characteristics |
|:------------|:----------------|
| Real-world footage | Diverse first-person and third-person perspectives of humans, animals, and vehicles |
| Game engine recordings | Precise action-contingent dynamics with RGB frames paired to control inputs |
| Synthetic data (Unreal Engine) | Collision-free, randomized camera trajectories with ground-truth poses |

A critical innovation is **hierarchical captioning**, which generates three distinct annotation layers for each video:

1. **Comprehensive Narrative Caption:** Holistic description interweaving environment, camera trajectory, and temporal evolution
2. **Scene-Static Caption:** Environment-focused description deliberately omitting motion, enabling decoupling of motion control from scene generation
3. **Dense Temporal Caption:** Fine-grained, time-aligned descriptions segmented into intervals for temporal alignment training

This hierarchical structure allows the model to learn precise action-contingent dynamics while maintaining control over static scene elements.

### 3.4 Emergent Spatial Memory

A remarkable property of LingBot-World is its emergent capability for long-term spatial consistency without explicit 3D representations [1]. The model preserves structural integrity of landmarks (statues, buildings, rock formations) even after they remain out of view for 60+ seconds. More impressively, the system exhibits reasoning about unobserved state evolution:

- When a camera moves forward and later returns to a frontal view, distant objects (e.g., bridges) appear closer, accurately reflecting the forward movement
- Vehicles that exit the frame continue their trajectories while unobserved and reappear at physically plausible locations

These behaviors indicate that LingBot-World simulates underlying spatiotemporal consistency rather than merely memorizing pixel patterns—a crucial requirement for true world permanence.

### 3.5 Performance and Capabilities

LingBot-World demonstrates state-of-the-art performance across multiple dimensions:

| Metric | Specification |
|:-------|:--------------|
| Generation Horizon | Sustains stable, high-fidelity environments for up to 10 minutes |
| Dynamic Degree | Achieves the highest motion complexity (0.8857) among interactive world models |
| Real-Time Performance | Processes at 16fps on 480p video |
| Latency | Sub-second latency (&lt;1 second for 16-frame generation) |
| Domain Generality | Handles photorealistic landscapes, scientific visualizations, cartoon styles |

### 3.6 Alternative Candidates: Genie and Oasis

While LingBot-World serves as our primary reference, other systems offer distinct approaches to the Body protocol. **Genie** (Google DeepMind) demonstrates a purely latent action-space approach, learning to simulate 2D platformer physics unsupervised from internet videos. However, its reliance on discrete latent codes limits its fidelity in complex 3D environments. **Oasis** (Etched.ai) pushes the boundaries of real-time generation using a specialized Transformer-based architecture that simulates Minecraft-like worlds at high frame rates. While impressive in speed and interactivity, Oasis currently trades high-fidelity texture coherence for latency, often exhibiting "dream-like" shifts in object identity that Raqia seeks to eliminate.

Both systems validate the Body protocol's core tenet: that simulation must be driven by causal rules (whether learned or explicit) rather than mere frame interpolation.

---

## 4. The Visual Cortex: The Error-Recycling Interface

### 4.1 Interface Requirement: The Error-Recycling Paradigm

Raqia posits that a "Visual Cortex" organ must treat error correction as its primary objective rather than error avoidance. The interface requires a mechanism that accepts accumulated degradation (blur, artifacts) as valid input states and "heals" them into coherent outputs.

The paradox this interface addresses: Why do powerful video generation models rapidly collapse under their own generation errors? The answer lies in the training-test hypothesis gap. A compliant Visual Cortex must bridge this gap by explicitly training on error-corrupted inputs.

### 4.2 Reference Implementation: Stable Video Infinity (SVI)

To satisfy this interface, we examine **Stable Video Infinity (SVI)**, which implements the protocol via Error-Recycling Fine-Tuning [2].

#### Stage 1: Error Injection and Semantic Density Gating

The system processes raw video clips through sliding windows (size $W=20$ frames), applying semantic density gating to filter redundant content. For informative windows, the system deliberately injects three types of errors:

$$\tilde{X}_{vid} = X_{vid} + I_{vid} \cdot E_{vid}$$

$$\tilde{X}_{noi} = X_{noi} + I_{noi} \cdot E_{noi}$$

$$\tilde{X}_{img} = X_{img} + I_{img} \cdot E_{img}$$

where $E_{vid}$, $E_{noi}$, and $E_{img}$ are errors resampled from memory banks, and $I \in \{0,1\}$ controls injection probability. With probability $p=0.5$, the system uses error-free inputs to preserve generation capabilities.

#### Stage 2: Bidirectional Error Curation

Given error-injected inputs and predicted velocity $\hat{\mathbf{V}}_t$, the system approximates predictions via single-step integration:

$$\hat{X}_{vid} = \tilde{X}_t + \int_t^1 \hat{V}_s \, ds$$

$$\hat{X}_{noi} = \tilde{X}_t - \int_0^t \hat{V}_s \, ds$$

Errors are calculated as residuals between approximated predictions and error-recycled ground truth:

$$E_{vid} = \hat{X}_{vid} - X^{rcy}_{vid}$$

$$E_{noi} = \hat{X}_{noi} - X^{rcy}_{noi}$$

This bidirectional calculation efficiently captures both forward (latent) and backward (noise) error dynamics without solving full ODEs.

#### Stage 3: Error Replay Memory

Calculated errors are dynamically saved into timestep-indexed replay memory banks $B_{vid}$ and $B_{noi}$. The training timesteps (typically $N_{tra}=1000$) are discretized to align with test timesteps ($N_{test}=50$), allowing selective error sampling based on timestep position.

The selective sampling strategy reflects error characteristics:
- **Video latent error $E_{vid}$:** Sampled uniformly from timestep-aligned bank
- **Noise error $E_{noi}$:** Sampled from same timestep bank due to duality between noise start and latent end  
- **Image error $E_{img}$:** Sampled across all timesteps independently, simulating cross-clip autoregression

### 4.3 Performance Results

SVI achieves state-of-the-art performance on multiple benchmarks:

| Metric | Wan 2.1 | FramePack | SVI-Shot |
|--------|---------|-----------|----------|
| Subject Consistency (50s) | 92.45 | 94.72 | **98.19** |
| Subject Consistency (250s) | 87.27 | 86.64 | **97.89** |
| Consistency Drop | -5.18 | -8.08 | **-0.30** |

Key findings:
- **Negligible Degradation:** SVI exhibits only 0.30-0.63 point drops in consistency when extending from 50 to 250 seconds, compared to 5-13 point drops for baselines
- **Creative Generation:** SVI-Film supports storyline-driven prompt streams with frequent scene transitions, achieving 84.27 scene consistency vs. 81.44 for Wan 2.1
- **Multimodal Adaptability:** SVI-Talk (audio-guided) achieves 6.12 sync accuracy; SVI-Dance (skeleton-guided) reaches 20.01 PSNR

### 4.4 Alternative Candidates: Sora and Wan

Other leading models exemplify the Visual Cortex protocol's capabilities and limitations. **OpenAI's Sora** employs a spacetime patch-based transformer architecture that scales effectively to generate highly detailed scenes. However, early analyses suggest it struggles with long-horizon object permanence—a classic symptom of unmanaged error accumulation. **Wan 2.1** represents a significant step forward with its flow-matching diffusion transformer, offering superior motion dynamics. Yet, without explicit error-recycling mechanisms, it remains susceptible to drift in autoregressive settings.

---

## 5. The Hippocampus: The Semantic Retention Interface

### 5.1 Interface Requirement: Semantic Compression

The "Hippocampus" protocol requires **semantic lossless compression**—maximizing information density while eliminating redundancy [3]. The interface dictates that passive storage of raw interaction streams is insufficient; a compliant organ must actively consolidate experience into abstract knowledge representations.

### 5.2 Reference Implementation: SimpleMem Pipeline

**SimpleMem** serves as our reference implementation, employing a three-stage compression architecture:

#### Stage 1: Semantic Structured Compression

SimpleMem employs implicit semantic density gating integrated directly into the LLM's generation process. Incoming dialogue is segmented into sliding windows ($W=20$ turns), and the system uses the foundation model as a semantic judge:

$$gate(W) \rightarrow \{m_k\} \text{ s.t. } m_k \in \{\emptyset, M\}$$

where empty set generation $\emptyset$ indicates low-density windows (e.g., phatic chitchat), which are discarded without explicit threshold tuning.

For informative windows, a unified De-linearization Transformation $F$ jointly performs:

$$m_k = F(W | H) = g_{time} \circ g_{coref} \circ g_{extract}(W)$$

This transformation:
- Resolves pronouns to specific entity names ($g_{coref}$)
- Converts relative temporal expressions to absolute ISO-8601 timestamps ($g_{time}$)
- Atomizes complex dialogue into self-contained factual statements ($g_{extract}$)

#### Stage 2: Online Semantic Synthesis

Unlike traditional systems that accumulate raw extractions additively, SimpleMem performs intra-session consolidation during the write phase. The synthesis function maps observations to consolidated entries:

$$F_{syn}(O_{session}, C_{context}) \rightarrow m_{consolidated}$$

For example, three fragments—"User wants coffee," "User prefers oat milk," "User likes it hot"—synthesize into a single entry: "User prefers hot coffee with oat milk."

Memory units are indexed through three complementary representations:

| Layer | Representation | Use |
|:------|:---------------|:----|
| Semantic | Dense vectors (1024-dim embeddings) | Fuzzy matching |
| Lexical | Sparse BM25 inverted index | Exact keyword/entity matching |
| Symbolic | SQL-based metadata | Deterministic filtering |

#### Stage 3: Intent-Aware Retrieval Planning

SimpleMem dynamically determines retrieval scope by inferring latent search intent. Given query $q$ and history $H$, the planning module $P$ decomposes information needs:

$$(q_{sem}, q_{lex}, q_{sym}, d) = P(q, H)$$

where $d$ represents adaptive retrieval depth reflecting query complexity.

### 5.3 Experimental Validation

**LoCoMo Benchmark (GPT-4.1-mini backbone):**

| Method | Multi-Hop | Temporal | Single-Hop | Average F1 |
|--------|-----------|----------|------------|------------|
| Full Context | 25.02 | 12.04 | 19.05 | 18.70 |
| Mem0 | 30.14 | 48.91 | 16.43 | 34.20 |
| LightMem | 24.96 | 20.55 | 19.21 | 33.79 |
| **SimpleMem** | **43.46** | **58.62** | **19.76** | **43.24** |

SimpleMem achieves **26.4% higher average F1** than Mem0 while reducing token consumption by 30% (531 vs. 16,910 tokens for retrieval).

### 5.4 Alternative Candidates: MemGPT and LightMem

**MemGPT** pioneers the "LLM as Operating System" metaphor, managing memory hierarchically akin to OS virtual memory paging. It excels at maintaining persona coherence over indefinite horizons but can suffer from retrieval latency in high-frequency interactive loops. **LightMem** focuses on extremely efficient, lightweight memory architectures suitable for edge deployment.

---

## 6. The Retina: The Codec-Aligned Interface

### 6.1 Interface Requirement: The Perception Bottleneck

The "Retina" protocol defines the translation layer between the generated world and the cognitive agent. The core requirement is solving the **modality gap**—the loss of fine-grained spatial detail when visual inputs are projected into language model embeddings.

### 6.2 Reference Implementation: OneVision-Encoder

**OneVision-Encoder** implements this interface through a Unified Vision-Language Encoder:

- **Preserves High-Frequency Detail:** By using a VQ-VAE-style tokenization trained end-to-end with the language model, it captures texture and geometry that standard embeddings lose
- **Zero-Shot Transfer:** Demonstrates exceptional capability in "In-Context Visual Learning," allowing the system to understand new objects or physics rules simply by "seeing" them
- **Long-Context Visual Reasoning:** Optimized for processing long sequences of visual tokens, effectively serving as the "optic nerve" that streams the generated world into the cognitive core

In the context of world permanence, OneVision acts as the critical bridge: it translates the pixel-perfect consistency of SVI into the semantic consistency of SimpleMem, closing the loop between what the world looks like and what the agent understands.

---

## 7. Raqia Architecture: Integration and Principles

### 7.1 The Anatomy of Raqia

We propose that a truly permanent generative world functions as a synthetic organism—**Raqia** (the self-sustaining system)—composed of four specialized organs working in continuous feedback loops:

| Component | Biological Analogy | Representative System | Solves |
|-----------|-------------------|---------------------|---------|
| **Simulation** | The Body | LingBot-World | **Drift:** Prevents physical law breaking |
| **Generation** | Visual Cortex | Stable Video Infinity | **Visual Decay:** Prevents blur/artifacts |
| **Cognition** | Hippocampus | SimpleMem | **Context Amnesia:** Prevents forgetting |
| **Perception** | Retina | OneVision-Encoder | **Input Blindness:** Prevents detail loss |

### 7.2 Principle 1: Error-Aware Training

Both LingBot-World and Stable Video Infinity incorporate mechanisms to handle accumulated errors:
- LingBot-World employs progressive curriculum training, extending sequences from 5 to 60 seconds
- Stable Video Infinity explicitly trains on error-injected inputs, bridging the training-test hypothesis gap

**Insight:** Models must experience and learn to correct their own mistakes during training, not just perform well on clean data.

### 7.3 Principle 2: Hierarchical Memory Organization

All three systems employ hierarchical structures:
- **LingBot-World:** Hierarchical captioning (narrative → scene-static → temporal)
- **Stable Video Infinity:** Timestep-indexed error banks
- **SimpleMem:** Multi-view indexing (semantic, lexical, symbolic)

### 7.4 Principle 3: Adaptive Computation

Computational resources scale with task complexity:
- **LingBot-World:** MoE activates only 14B of 28B parameters per timestep
- **SimpleMem:** Intent-aware retrieval planning dynamically adjusts depth

### 7.5 Principle 4: Emergent Spatial Understanding

True world permanence requires implicit spatial reasoning beyond explicit 3D representations. LingBot-World demonstrates emergent spatial memory, maintaining landmark consistency across 60+ seconds without Gaussian splatting or NeRF.

### 7.6 Principle 5: Semantic Compression Over Raw Storage

SimpleMem achieves 30× token reduction (24,000 → 800 tokens) through semantic structured compression. **Store meaning, not tokens.**

### 7.7 Principle 6: Codec-Aligned Perception

OneVision-Encoder aligns visual features directly with the LLM's token space via VQ-VAE-style quantization, preserving high-frequency detail that standard projection layers discard.

---

## 8. Open Challenges and Future Directions

### 8.1 Challenge 1: Identity Consistency Across Scene Transitions

Current systems struggle with long-term identity persistence when characters exit and re-enter frames. Future work requires:
- Persistent identity embeddings that survive scene changes
- Cross-shot feature caching mechanisms
- Scene-aware anchoring strategies

### 8.2 Challenge 2: Infinite Context Without Bounded Memory

Scaling beyond current limits (60-second temporal consistency, 10-minute videos, ~400-turn conversations) requires:
- Hierarchical memory consolidation (episodic → semantic → schema-level)
- Forgetting mechanisms that prioritize salient information
- Distributed memory architectures

### 8.3 Challenge 3: Multimodal World Coherence

Future world models must maintain coherence across vision, audio, haptics, and language simultaneously.

### 8.4 Challenge 4: Real-Time Interactive World Generation

LingBot-World achieves &lt;1 second latency for 16 frames, but many applications demand instant response.

### 8.5 Challenge 5: Evaluating Long-Horizon Consistency

Current benchmarks test memory systems, but comprehensive evaluation requires:
- Physics consistency metrics (momentum conservation, collision detection)
- User studies measuring perceptual quality over minutes/hours
- Adversarial probes targeting identity drift

---

## 9. Conclusion

**Raqia** provides a theoretical blueprint for solving the binding problem of generative world permanence. By decomposing the problem into four coupled but replaceable organs—Simulation, Generation, Cognition, and Perception—we move beyond the limitations of monolithic models.

This modularity is central to the framework's utility: while we have examined LingBot-World, Stable Video Infinity, SimpleMem, and OneVision-Encoder as our primary reference implementation, the framework is **agnostic to the specific models used**. As demonstrated by the viability of alternatives like Genie, Wan 2.1, MemGPT, and SigLIP, Raqia defines the *interfaces* (Body, Visual Cortex, Hippocampus, Retina) rather than the implementations.

The ultimate goal—AI systems that maintain consistent, interactive worlds indefinitely—is no longer a question of scaling a single model, but of orchestrating a symphony of specialized capabilities.

---

## References

1. Robbyant Team. (2026). *Advancing open-source world models: LingBot-World*. arXiv preprint arXiv:2601.20540v1.
2. Li, W., Pan, W., Luan, P.-C., Gao, Y., & Alahi, A. (2025). *Stable Video Infinity: Infinite-length video generation with error recycling*. arXiv preprint arXiv:2510.09212v1.
3. Liu, J., Su, Y., Xia, P., Han, S., Zheng, Z., Xie, C., Ding, M., & Yao, H. (2026). *SimpleMem: Efficient lifelong memory for LLM agents*. arXiv preprint arXiv:2601.02553v3.
4. Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y. (2024). *Evaluating very long-term conversational memory of LLM agents*. arXiv preprint arXiv:2402.17753.
5. Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K.-W., & Yu, D. (2024). *LongMemEval: Benchmarking chat assistants on long-term interactive memory*. arXiv preprint arXiv:2410.10813.
6. Wang, A. et al. (2025). *Wan: Open and advanced large-scale video generative models*. arXiv preprint arXiv:2503.20314.
7. Chen, B., Martí Monson, D., Du, Y., Simchowitz, M., Tedrake, R., & Sitzmann, V. (2024). *Diffusion forcing: Next-token prediction meets full-sequence diffusion*. Advances in Neural Information Processing Systems.
8. Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). *What learning systems do intelligent agents need? Complementary learning systems theory updated*. Trends in Cognitive Sciences, 20(7), 512-534.
9. Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). *Lost in the middle: How language models use long contexts*. arXiv preprint arXiv:2307.03172.
10. Packer, C., Fang, V., Patil, S. G., Lin, K., Wooders, S., & Gonzalez, J. (2023). *MemGPT: Towards LLMs as operating systems*. arXiv preprint arXiv:2310.08560.
11. LMMs-Lab, Glint Lab, AIM for Health Lab, & MVP Lab. (2026). *OneVision-Encoder: Codec-aligned sparsity as a foundational principle for multimodal intelligence*. Technical Report.

---

**Cite this work:**
```bibtex
@article{raqia2026,
  title={Raqia: A Unified Architecture for Generative World Permanence},
  author={{Raqia Team}},
  journal={arXiv preprint},
  year={2026}
}

<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
    }
  };
</script>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
