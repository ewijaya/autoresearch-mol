# Baseline Architecture Rationale

**Date:** 2026-03-17
**Purpose:** Comprehensive record of why the baseline architecture was chosen, for reference when writing the paper.
**Sources:** `docs/stress-test-technical-feasibility.md` (Sections 3.2–3.8, 6.3), `docs/PRD-recursive-mol.md` (Section 6.3)

---

## 1. The Baseline Architecture

All tracks (SMILES, Protein, NLP) share the same model architecture as the starting point for fair comparison. The only differences are data-facing parameters (vocabulary size, sequence length, device batch size).

### 1.1 Common Architecture Parameters

| Parameter | Value | Derivation |
|-----------|-------|------------|
| Depth (n_layer) | 6 | Reduced from autoresearch default of 8 for A10G throughput |
| Width (n_embd) | 320 | ASPECT_RATIO=48 → base_dim=288, rounded up via HEAD_DIM alignment |
| Heads (n_head) | 5 | model_dim / HEAD_DIM = 320 / 64 |
| KV Heads (n_kv_head) | 5 | Full multi-head attention (not grouped-query) |
| Head Dimension | 64 | Reduced from autoresearch default of 128 |
| FFN Multiplier | 5x | Hidden dim = 5 × 320 = 1,600 |
| Activation | ReluSquared | F.relu(x).square(); inherited from autoresearch, left as searchable |
| Normalization | RMSNorm | Pre-attention and pre-MLP |
| Attention | SDPA (Flash Attention 2 backend) | Replaces FA3 for A10G (Ampere) compatibility |
| Window Pattern | SSSL | 3 short-window + 1 full-causal per 4-layer cycle |
| Value Embeddings | Enabled | Alternating layers; ResFormer-style input-dependent gating |
| Weight Tying | Disabled | Embedding and unembedding are separate |
| Optimizer | MuonAdamW | Muon for 2D matrix params, AdamW for embeddings/scalars |
| Total Batch Size | 65,536 tokens | Reduced from autoresearch default of 524K for smaller datasets |
| Time Budget | 300 seconds (5 min) | Fixed; agent cannot change this |
| Estimated Params | ~8.6–8.7M | Varies slightly by vocab size |

### 1.2 Track-Specific Parameters

| Parameter | SMILES | Protein | NLP |
|-----------|--------|---------|-----|
| Sequence Length | 256 | 512 | 2,048 |
| Vocab Size | 37 | 24 | ~8,192 (BPE) |
| Short Window Size | 128 (seq_len/2) | 256 (seq_len/2) | 1,024 (seq_len/2) |
| Device Batch Size | 256 | 128 | 64 |
| Window Pattern | SSSL | SSSL | SSSL |
| Baseline val_bpb | ~0.596 | ~3.978 | — |
| Peak VRAM | ~5.3 GB | ~5.3 GB | — |

### 1.3 Autoresearch Default (Before Adaptation)

For context, the original autoresearch (Karpathy) defaults that this was adapted from:

| Parameter | Autoresearch Default | Our Baseline | Reason for Change |
|-----------|---------------------|--------------|-------------------|
| Depth | 8 | 6 | A10G throughput constraint |
| Width | 512 | 320 | Param budget constraint |
| Heads | 4 (head_dim=128) | 5 (head_dim=64) | Smaller head dim for smaller model |
| Vocab | 8,192 (BPE) | 24–37 (char-level) | Molecular char-level tokenization |
| Seq Length | 2,048 | 256–512 | Molecular sequences are short |
| Batch Size | 524,288 | 65,536 | Smaller datasets |
| Total Params | ~50.3M | ~8.6M | Epoch budget constraint |
| Attention | Flash Attention 3 | SDPA (FA2 backend) | FA3 incompatible with Ampere GPUs |
| Data | FineWeb-Edu (400B tokens) | ZINC-250K (~12.5M tokens) | Domain-specific molecular data |

---

## 2. Why This Architecture: The Constraints

The baseline architecture was not taken from a published model or a known efficient design. It was **engineered specifically for this project's constraints** through a systematic feasibility analysis.

### 2.1 Hardware Constraint: A10G GPU

**Target hardware:** AWS g5.xlarge (NVIDIA A10G, 24 GB VRAM)

| Spec | A10G | H100 (autoresearch target) | Ratio |
|------|------|---------------------------|-------|
| bf16 peak FLOPS | 31.2 TFLOPS | 989.5 TFLOPS | 31.7x slower |
| VRAM | 24 GB | 80 GB | 3.3x less |
| Memory BW | 600 GB/s | 3,350 GB/s | 5.6x less |

The A10G is ~31.7x slower than the H100 in raw compute. The autoresearch default architecture (50.3M params) was designed for H100 throughput and is far too large for the A10G to train meaningfully in 5 minutes.

**Flash Attention 3 incompatibility:** FA3 targets Hopper (sm_90) architecture. The A10G is Ampere (sm_86). The codebase was modified to use `torch.nn.functional.scaled_dot_product_attention` (which dispatches to Flash Attention 2 on Ampere) as a drop-in replacement.

### 2.2 Dataset Size Constraint: ZINC-250K

**ZINC-250K tokenized size:** ~12.5M tokens (SMILES character-level)
**UniRef50 subset:** ~15M tokens (protein character-level)

This is many orders of magnitude smaller than the autoresearch default dataset (FineWeb-Edu, 400B tokens). The critical question was: **how many epochs can the model see in 5 minutes?**

| Model Config | Params | Estimated tok/sec (A10G, 30% MFU) | Tokens in 5 min | Epochs over 12.5M |
|--------------|--------|-----------------------------------|-----------------|-------------------|
| Autoresearch default | 50.3M | ~31K | ~9.3M | **0.7** |
| SMILES char vocab | 25.4M | ~61K | ~18.4M | 1.5 |
| Target (depth=6, dim=320) | ~8.6M | ~120K | ~36M | **2.9** |
| Tiny (depth=4, dim=128) | ~2M | ~780K | ~234M | 18.7 |

### 2.3 Epoch Budget: The Core Design Constraint

The feasibility analysis identified the number of training epochs as the critical bottleneck for architecture search signal quality:

- **< 2 epochs:** Model is in the rapid learning phase. val_bpb is still improving. Architecture differences are confounded with learning speed — "how fast does this architecture learn" rather than "how well does this architecture model the data."
- **3–8 epochs:** val_bpb begins to plateau. Architecture quality differences become discriminable. **This is the target regime.**
- **> 10 epochs:** Risk of overfitting on ZINC-250K (12.5M tokens). With ~5M+ params, the model can start memorizing.

**Design target:** 6–12M parameters, achieving 3–8 epochs in 5 minutes, producing enough convergence for meaningful val_bpb signal.

The chosen ~8.6M param baseline achieves ~2.9 epochs over ZINC-250K in 5 minutes on A10G — at the lower bound of the target range, but sufficient for discriminative signal.

### 2.4 val_bpb Signal Quality

For the architecture search loop (modify → train 5 min → eval → keep/discard) to work, two conditions must hold:

1. **val_bpb must discriminate between architectures:** At the target epoch count, val_bpb reaches a regime where architecture quality differences are visible above training noise.
2. **Changes of ~0.01 BPB must be detectable:** The evaluation budget (EVAL_TOKENS = 5 × 131,072 ≈ 655K tokens) must produce stable enough estimates to detect small improvements.

Both conditions were validated in the calibration study (see PRD SC-2).

---

## 3. Why Each Specific Parameter

### 3.1 Depth = 6

Reduced from autoresearch default of 8. The primary driver was A10G throughput — fewer layers means faster forward/backward passes, allowing more training steps (and thus more epochs) within the 5-minute budget. Depth 6 was chosen as a balance: deep enough for compositional pattern learning in molecular sequences, shallow enough for adequate throughput.

### 3.2 Width = 320, Heads = 5

Derived from `ASPECT_RATIO=48` and `HEAD_DIM=64`:
- `base_dim = DEPTH × ASPECT_RATIO = 6 × 48 = 288`
- Rounded up to nearest multiple of `HEAD_DIM`: `ceil(288/64) × 64 = 320`
- `n_head = 320 / 64 = 5`

The aspect ratio was tuned to keep total parameters in the 6–12M range. HEAD_DIM was reduced from 128 (autoresearch default) to 64 to allow more heads at the smaller model width, providing more parallel attention patterns.

### 3.3 FFN Multiplier = 5x

Inherited from the autoresearch default. This gives an FFN hidden dimension of 1,600 for n_embd=320. Left as a searchable parameter for the agent — the agent may discover that a different ratio works better for molecular data.

### 3.4 Activation = ReluSquared

Inherited from the autoresearch default (`F.relu(x).square()`). Intentionally left as the default rather than pre-optimizing it, because activation function choice is one of the architectural dimensions the search should explore. The feasibility doc explicitly lists "ReluSquared → GELU or SiLU" as a low-complexity search direction.

### 3.5 Window Pattern = SSSL

The SSSL pattern (3 short-window + 1 full-causal per 4-layer cycle) was chosen based on domain-specific reasoning:

- **For proteins:** "Proteins need more long-range attention" — secondary/tertiary structure creates dependencies spanning hundreds of residues. The SSSL pattern ensures 1 in 4 layers sees the full sequence.
- **For SMILES:** The original recommendation was "SL" (alternating short/long) for chemical locality, since bond adjacency is local in SMILES strings. In practice, SSSL was used for both tracks.
- **Short window size = seq_len / 2:** Covers local chemical neighborhoods (half the sequence) in the short-window layers.

### 3.6 Value Embeddings (ResFormer-style)

Enabled on alternating layers with input-dependent gating (32 gate channels). Inherited from the autoresearch architecture. These add a small parameter overhead (~35K params) but provide auxiliary representational capacity. Left as-is rather than ablating, since the agent can modify or remove them during search.

### 3.7 Batch Size = 65,536

Reduced from autoresearch default of 524,288 (8x reduction). With only 12.5M tokens in the SMILES dataset, a 524K batch would mean only ~24 optimizer steps per epoch. The smaller batch size of 65K allows ~190 steps per epoch, providing smoother gradient estimates and more stable training curves.

### 3.8 Optimizer = MuonAdamW

Inherited from autoresearch. MuonAdamW uses the Muon optimizer (orthogonalized momentum) for 2D matrix parameters in transformer blocks, and AdamW for 1D parameters (embeddings, scalars, biases). This hybrid was designed for efficient transformer training at small scale. Left as a searchable parameter.

---

## 4. Design Philosophy

The baseline was designed with a specific philosophy for the architecture search experiment:

1. **Maximize search signal, not absolute performance.** The goal is not to build the best molecular transformer — it's to create conditions where the architecture search agent can discover meaningful improvements. This means the baseline should be "good enough" to learn the data but leave room for improvement.

2. **Inherit reasonable defaults, don't pre-optimize.** Parameters like activation function, FFN ratio, and attention pattern were deliberately kept at their autoresearch defaults rather than hand-tuned for molecular data. This gives the agent a fair starting point and avoids biasing the search.

3. **Identical architecture across tracks.** The model architecture (depth, width, heads, activation, etc.) is the same for SMILES, protein, and NLP tracks. Only data-facing parameters (vocab size, sequence length, batch size) differ. This enables cross-track comparison of what the agent discovers for each domain.

4. **Constrained by hardware, not by design preference.** Every parameter choice traces back to the A10G + 5-minute + small-dataset constraints, not to architectural preferences from the literature. This is an important point for the paper: the baseline is a principled engineering choice, not an arbitrary one.

---

## 5. Paper Framing Notes

### For the Methods section:

> "The baseline architecture was derived from the autoresearch framework [Karpathy, 2026] and systematically adapted for our experimental constraints. The original 50.3M-parameter architecture was scaled down to ~8.6M parameters to achieve 3–8 training epochs within the 5-minute budget on A10G hardware, following a feasibility analysis of the compute-data-convergence tradeoff (see Appendix). All architectural parameters (depth, width, activation, attention pattern, FFN ratio, optimizer) are shared across the three tracks (SMILES, protein, NLP); only data-facing parameters (vocabulary, sequence length, batch size) differ. This ensures that any architectural divergence discovered by the agent is attributable to domain-specific optimization, not baseline differences."

### For reviewer defense:

**Q: Why not use a published molecular transformer architecture as the baseline?**

A: The experiment tests whether an LLM agent can discover domain-specific architectural improvements through iterative search. Starting from a domain-agnostic baseline (adapted for hardware constraints but not for molecular data) is essential to this design — it gives the agent room to discover molecular-specific features. A pre-optimized molecular baseline would conflate human domain knowledge with agent discovery.

**Q: Is 8.6M params too small to be meaningful?**

A: The model size was chosen to maximize architecture search signal quality (3–8 epochs in 5 minutes on A10G), not to achieve SOTA performance. The 8.6M scale is sufficient to learn non-trivial molecular patterns (baseline val_bpb of 0.596 on SMILES, well below random at ~5.3). The contribution is the search methodology and relative improvement, not absolute model quality.

**Q: Why ReluSquared and not GELU/SiLU?**

A: Intentionally kept as the autoresearch default to avoid pre-optimizing the search space. Activation function is one of the architectural dimensions the agent can explore. Starting from a non-standard choice (ReluSquared) tests whether the agent can discover better alternatives.

---

## 6. References

- `docs/stress-test-technical-feasibility.md` — Full feasibility analysis with compute budget calculations, epoch analysis, and per-track recommendations (Sections 3.2–3.8, 6.3)
- `docs/PRD-recursive-mol.md` — Section 6.3: Model Configuration starting point specification
- `src/train.py` — Implementation of the baseline architecture (GPTConfig dataclass, lines 66–78; hyperparameters, lines 593–622)
