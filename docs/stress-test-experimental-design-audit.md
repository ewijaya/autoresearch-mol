# Experimental Design Audit: Recursive Self-Refinement for Molecular Transformers

**Audit Date:** March 9, 2026
**Auditors:** Statistician, Infrastructure Engineer, Baseline Researcher (simulated panel)
**Document Under Review:** Self-Directed Discovery of Molecular Transformer Architectures via Recursive Self-Refinement

---

## Table of Contents

1. [Role 1: Statistician](#role-1-statistician)
2. [Role 2: Infrastructure Engineer](#role-2-infrastructure-engineer)
3. [Role 3: Baseline Researcher](#role-3-baseline-researcher)
4. [Cross-Role Challenges](#cross-role-challenges)
5. [Synthesized Recommendations](#synthesized-recommendations)

---

## Role 1: Statistician

### 1.1 Power Analysis for n=3 Per Condition

The proposed design is 3 tracks x 3 runs x ~100 experiments each. The primary unit of analysis for hypothesis testing is the **run** (n=3 per track), not the individual experiment within a run. This is a critical distinction.

**Effect size detectability with n=3:**

Using a two-sample t-test (e.g., comparing Track A vs Track C final val_bpb), with n=3 per group:

| Alpha | Power | Detectable Cohen's d |
|-------|-------|---------------------|
| 0.05  | 0.80  | ~3.1                |
| 0.05  | 0.60  | ~2.4                |
| 0.10  | 0.80  | ~2.5                |

A Cohen's d of 3.1 means we can only detect effects where the group means differ by more than 3 standard deviations. This is an **extremely large effect size** -- essentially, we can only detect differences that are blindingly obvious from visual inspection alone. For context, Cohen's d = 0.8 is already considered "large" in the behavioral sciences.

**Verdict: n=3 is critically underpowered for frequentist hypothesis testing.** A NeurIPS reviewer will flag this immediately.

### 1.2 What a Reviewer Would Say About n=3

> "The authors report results from only 3 independent runs per condition. With n=3, no standard frequentist test has adequate power to detect anything short of enormous effects. The confidence intervals around reported means will be extremely wide. The authors cannot make reliable claims about H1 (architectural differences across domains) or H4 (faster than random NAS) with this sample size. I recommend increasing to at least n=10 runs per condition, or switching to a Bayesian framework that can quantify evidence for the null hypothesis."

This is a near-certain reviewer comment. NeurIPS has become increasingly rigorous about statistical methodology since the 2020 ML reproducibility crisis.

### 1.3 Recommended Statistical Framework

Given the constraints, I recommend a **hybrid Bayesian + permutation test** approach:

#### Primary Analysis: Bayesian Estimation

- Use **Bayesian hierarchical models** rather than frequentist tests. With n=3, Bayesian credible intervals are more honest about uncertainty than confidence intervals, and we can use informative priors from pilot data.
- Model: `val_bpb_final ~ Normal(mu_track, sigma_track)` with track-level random effects.
- Report **posterior distributions** and **Bayes factors** for H1 (track differences). A Bayes factor approach can provide evidence *for* the null (architectures converge) which frequentist tests cannot.
- Use **ROPE (Region of Practical Equivalence)**: Define a range (e.g., +/- 0.05 bpb) within which differences are practically meaningless. Report the probability that the true difference falls within vs. outside the ROPE.

#### Secondary Analysis: Within-Run Trajectory Analysis

Each run produces ~100 sequential experiments. This is a **time series** and should be analyzed as such:

- **Learning curve analysis:** Fit exponential decay or power-law curves to val_bpb vs. experiment number. Compare curve parameters (asymptote, rate, onset) across tracks. This uses all 300 data points per track.
- **Change-point detection:** Identify when architectural breakthroughs occur (large drops in val_bpb). Compare the distribution of change-point locations across tracks.
- **Paired comparisons:** Since each run is sequential, use paired Wilcoxon signed-rank tests on matched experiment indices across runs. This is more powerful than independent-sample tests.

#### Addressing H1 (Architecture Differs Across Domains)

- Do NOT rely solely on val_bpb comparisons. Architectures are discrete, high-dimensional objects.
- **Quantify architectural distance:** Define a feature vector for each discovered architecture (depth, width, heads, FFN ratio, activation type, attention variant, etc.). Use **Hamming distance** or **edit distance on code diffs** to measure architectural divergence across tracks.
- Apply **permutation tests** on architectural distance matrices: if track label doesn't matter, randomly permute and measure whether observed cross-track distances exceed within-track distances.
- This is a stronger test than comparing scalar val_bpb and does not require large n.

#### Addressing H4 (Faster Than Random NAS)

- Compare cumulative best val_bpb at each experiment index between agent runs and random NAS baseline runs.
- Use **area under the optimization curve (AUC-OC)** as a summary statistic.
- Apply **bootstrap confidence intervals** (10,000 resamples) on the AUC-OC difference.
- With n=3 agent runs vs. n=3 random runs, this is still underpowered, but the bootstrap CIs will be honest about it.

### 1.4 Multiple Comparisons

With 4 sub-hypotheses and 3 pairwise track comparisons, we have at minimum 4 x 3 = 12 tests. Apply:

- **Holm-Bonferroni correction** for the family of frequentist tests (if any are used).
- For Bayesian analyses, multiple comparisons are less of a concern if using a hierarchical model (partial pooling handles this naturally), but report the number of comparisons and prior choices transparently.

### 1.5 Comparing val_bpb Across Tracks with Different Vocabularies

The proposal states "val_bpb is vocab-size-independent by design." This requires careful scrutiny.

**Bits per byte (bpb)** normalizes by the number of UTF-8 bytes in the input, which makes it independent of tokenizer vocabulary size. However:

- **SMILES vocab:** ~40 unique characters (atoms, bonds, brackets, rings). Very low entropy per character.
- **Protein vocab:** 20 amino acids + special tokens. Also low entropy.
- **NLP vocab:** Full English character set or BPE tokens over large vocabulary.

The **intrinsic entropy** of each domain differs fundamentally. Comparing absolute bpb across tracks is like comparing compression ratios of JPEG vs. MP3 -- both are bits per byte, but the domains have different information content. A val_bpb of 1.5 on SMILES is not comparable to 1.5 on English text.

**Recommendation:**
- Do NOT compare absolute val_bpb across tracks. This is meaningless.
- Instead, compare **relative improvement** (% reduction from baseline architecture) across tracks. This normalizes for domain difficulty.
- Report each track's bpb with its own baseline and interpret within-track.
- For H1, the comparison should be about *which architectural changes helped*, not *which track achieved lower bpb*.

### 1.6 Concrete Statistical Analysis Plan

| Hypothesis | Primary Test | Secondary Test | Minimum Credible Result |
|-----------|-------------|---------------|------------------------|
| H1: Arch differs | Permutation test on arch distance matrix | Bayesian ROPE on bpb improvement | Architectural feature vectors cluster by track (p < 0.05 permutation) |
| H2: Rediscovers tricks | Qualitative code analysis + binary presence/absence | Fisher's exact test on trick occurrence | >= 2/3 runs independently discover >= 2 known tricks |
| H3: SMILES->protein transfer | Paired t-test (or Wilcoxon) on transfer bpb vs. trained-from-scratch bpb | Bayesian estimation with ROPE | Transfer architecture bpb within 10% of native architecture |
| H4: Faster than random NAS | Bootstrap CI on AUC-OC difference | Mann-Whitney U on final bpb | Agent AUC-OC < Random AUC-OC with 95% bootstrap CI excluding zero |

### 1.7 Minimum Viable n

To make robust claims at NeurIPS:
- **n=3 is acceptable** if: (a) you use Bayesian methods, (b) effect sizes are visually obvious, (c) you are transparent about uncertainty, and (d) the qualitative analysis of architectures is compelling.
- **n=5 is strongly preferred:** With n=5, Cohen's d ~1.7 becomes detectable at 80% power. This is still large but within the realm of plausible ML experiment effects.
- **n=10 would be ideal** but may be cost-prohibitive.
- **Recommendation: increase to n=5 per track (15 runs total) if budget allows.** See Infrastructure Engineer section for cost implications.

---

## Role 2: Infrastructure Engineer

### 2.1 AWS g5.xlarge Pricing (Live Data, ap-northeast-1)

Queried AWS Spot Price History API on March 9, 2026:

| Region | AZ | Spot Price (USD/hr) | Timestamp |
|--------|----|--------------------:|-----------|
| ap-northeast-1 | 1a | $0.7074 | 2026-03-09 01:46 UTC |
| ap-northeast-1 | 1c | $0.7025 | 2026-03-08 23:16 UTC |
| ap-northeast-1 | 1a | $0.7081 | 2026-03-08 19:16 UTC |
| ap-northeast-1 | 1c | $0.6935 | 2026-03-08 15:31 UTC |

**On-demand price:** $1.459/hr (Tokyo)

For comparison, **us-east-1 spot prices** are $0.43-0.45/hr -- roughly 37% cheaper.

**Key finding: The proposal estimates $1/hr, but actual spot price in Tokyo is $0.70/hr. On-demand is $1.46/hr.** The proposal should specify spot vs. on-demand explicitly.

### 2.2 Revised Cost Estimate

#### Original Proposal Calculation
3 tracks x 3 runs x 8 hrs x $1/hr = $72

#### Corrected Calculation (Spot, Tokyo)

Each run consists of ~100 experiments x 5 minutes = 500 minutes = 8.33 hours. But this only counts GPU training time. Additional overhead:

| Component | Time per Run | Cost Component |
|-----------|-------------|---------------|
| Training (100 x 5 min) | 8.33 hrs | GPU |
| Agent inference between experiments | ~1-2 min per experiment = 1.67-3.33 hrs | GPU idle but still billed |
| Data preprocessing, setup | 0.5 hrs | GPU |
| Failed experiments (assume 15% crash) | 1.25 hrs | GPU |
| **Total per run** | **~12-14 hrs** | |

**GPU cost (spot):**
- 9 runs x 13 hrs avg x $0.70/hr = **$81.90** (Tokyo spot)
- 9 runs x 13 hrs avg x $0.44/hr = **$51.48** (us-east-1 spot)

**Agent API costs (NOT accounted in original estimate):**
- Each experiment requires 1-2 agent calls (analyze results + generate new code)
- ~100 experiments per run x 9 runs = 900 agent calls
- Assuming Claude Sonnet at ~$0.01-0.03 per call (typical for code generation): **$9-$27**
- If using Claude Opus: potentially **$30-$90**

**Total realistic budget:**

| Scenario | GPU Cost | Agent API | Total |
|----------|---------|-----------|-------|
| Optimistic (spot us-east-1, Sonnet) | $51 | $9 | **$60** |
| Proposed (spot Tokyo) | $82 | $15 | **$97** |
| Conservative (on-demand Tokyo, Opus) | $170 | $90 | **$260** |
| With n=5 (spot Tokyo, Sonnet) | $137 | $25 | **$162** |

**Verdict: $72 is plausible for GPU-only costs in us-east-1 but underestimates Tokyo spot pricing and omits agent API costs. A realistic budget for the proposed design is $97-$130. Increasing to n=5 would cost ~$162.**

#### Cost recommendation: Switch to us-east-1

Switching from ap-northeast-1 to us-east-1 saves ~37% on GPU costs. Unless there is a specific data residency requirement, us-east-1 is strictly better for this workload. At us-east-1 spot prices, even n=5 would cost only ~$100 total.

### 2.3 ZINC-250K Dataset Analysis

**ZINC-250K** is a curated subset of the ZINC database (Irwin et al., 2012; Sterling & Irwin, 2015) containing approximately **249,455 drug-like molecules** in SMILES format.

**Character-level statistics (well-established in literature):**

| Property | Value |
|----------|-------|
| Number of molecules | ~249,455 |
| Average SMILES length | ~43 characters |
| Median SMILES length | ~39 characters |
| Max SMILES length | ~120 characters |
| Unique characters | ~35-40 (C, c, N, n, O, o, S, s, F, Cl, Br, I, (, ), [, ], =, #, 1-9, -, +, @, /, \, .) |
| Total characters | ~249,455 x 43 = **~10.7M characters** |

With a 90/10 train/val split:
- Training set: ~9.6M characters
- Validation set: ~1.07M characters

**This is an extremely small dataset by modern LM standards.** For comparison:
- FineWeb-Edu: billions of tokens
- A single epoch through ZINC-250K character-level is comparable to a few paragraphs of text in token count

### 2.4 Can We Train a Meaningful Transformer in 5 Minutes on A10G?

**A10G throughput estimates:**

The NVIDIA A10G delivers approximately:
- **FP32:** ~31.2 TFLOPS
- **TF32:** ~62.5 TFLOPS
- **FP16/BF16 with tensor cores:** ~125 TFLOPS

For a small transformer (e.g., 6 layers, 256 dim, 4 heads, ~2M parameters) on character-level SMILES:

| Configuration | Estimated Throughput | Tokens in 5 min |
|--------------|---------------------|-----------------|
| Small (2M params), batch 256, seq 128 | ~150K-300K tok/sec | 45M-90M tokens |
| Medium (10M params), batch 128, seq 128 | ~80K-150K tok/sec | 24M-45M tokens |
| Large (50M params), batch 64, seq 128 | ~30K-60K tok/sec | 9M-18M tokens |

**With ~10.7M total characters in ZINC-250K:**

| Model Size | 5-min Token Budget | Epochs Through Data | Verdict |
|-----------|-------------------|--------------------:|---------|
| Small (2M) | 45-90M | 4-8 epochs | More than sufficient |
| Medium (10M) | 24-45M | 2-4 epochs | Sufficient |
| Large (50M) | 9-18M | 1-2 epochs | Marginal |

**Verdict: Yes, 5 minutes is sufficient for meaningful training on ZINC-250K at character level.** In fact, the dataset is so small that a small transformer will see the entire dataset multiple times. This is actually a concern -- the model may **overfit** within the 5-minute window, making val_bpb less informative.

**Recommendation:** Monitor train_bpb vs. val_bpb gap. If overfitting is severe (train_bpb << val_bpb), consider:
1. Data augmentation via SMILES randomization (multiple valid SMILES per molecule, expanding effective dataset 10-100x)
2. Using a larger subset of ZINC (ZINC-full has ~230M molecules)
3. Implementing early stopping within the 5-min window

### 2.5 VRAM Feasibility

**Memory estimation for character-level transformer on A10G (24GB):**

| Component | Small (2M) | Medium (10M) | Large (50M) |
|-----------|-----------|-------------|-------------|
| Model parameters | ~8 MB | ~40 MB | ~200 MB |
| Optimizer states (AdamW) | ~16 MB | ~80 MB | ~400 MB |
| Activations (batch=256, seq=128) | ~500 MB | ~2 GB | ~8 GB |
| Gradient buffers | ~8 MB | ~40 MB | ~200 MB |
| **Total** | **~0.5 GB** | **~2.2 GB** | **~8.8 GB** |

**Verdict: Easily fits in 24GB.** Even the "large" configuration uses less than 40% of available VRAM. The agent could theoretically explore models up to ~100-150M parameters before hitting VRAM limits. Mixed precision (FP16) would roughly halve activation memory, allowing even larger models.

### 2.6 Spot Instance Reliability

Spot instances in ap-northeast-1 for g5.xlarge have historically shown moderate interruption rates (5-15% based on typical GPU spot behavior). For 5-minute training runs, the risk is manageable:

- **Probability of interruption during any single 5-min window:** ~0.5-1.5%
- **Probability of at least one interruption in a 13-hour session:** ~15-25%

**Recommendation:** Implement checkpointing between experiments (trivial, since each experiment is independent). If a spot interruption occurs mid-experiment, simply re-run that experiment. The autoresearch framework should already handle this gracefully since it evaluates each experiment independently.

### 2.7 Infrastructure Summary

| Item | Proposal | Reality | Risk Level |
|------|----------|---------|-----------|
| GPU price | $1/hr | $0.70/hr spot, $1.46/hr on-demand | Low (favorable) |
| Total GPU cost | $72 | $82 (spot Tokyo) | Low |
| Agent API cost | Not budgeted | $9-$90 | Medium |
| ZINC-250K size | Not specified | 10.7M chars | OK |
| 5-min training feasibility | Assumed | Confirmed (4-8 epochs for small model) | Low |
| VRAM constraint | Assumed OK | Confirmed (< 40% usage) | Low |
| Overfitting risk | Not addressed | High for small dataset | **High** |

---

## Role 3: Baseline Researcher

### 3.1 Existing Molecular Transformer Architectures

#### Published Models and Their Metrics

| Model | Year | Architecture | Pre-training Objective | Primary Metrics | val_bpb Reported? |
|-------|------|-------------|----------------------|----------------|-------------------|
| **SMILES Transformer** | 2019 | Encoder-decoder, 4L/256d | Autoregressive LM on SMILES | Validity, uniqueness, novelty of generated molecules | No |
| **MolBERT** | 2021 | BERT-base (12L/768d/12H) | Masked LM on SMILES + molecular property aux tasks | MoleculeNet benchmarks (ROC-AUC on classification tasks) | No |
| **ChemBERTa** | 2020 | RoBERTa-base | Masked LM on 10M SMILES from PubChem | MoleculeNet (ROC-AUC), BBBP, HIV, BACE | No |
| **ChemBERTa-2** | 2022 | RoBERTa (various sizes) | MLM with multi-task regression | MoleculeNet, regression R-squared | No |
| **MoLFormer** | 2022 | Linear attention transformer, 12L/768d | Masked LM on 1.1B SMILES from ZINC15+PubChem | MoleculeNet benchmarks, HOMO/LUMO prediction | No |
| **Uni-Mol** | 2023 | 3D-aware transformer | 3D position denoising + masked atom prediction | Molecular property, conformation, docking tasks | No |
| **SELFIES-Transformer** | 2022 | GPT-2 style on SELFIES | Autoregressive LM | Validity (100% by construction), FCD, diversity | No |
| **Chemformer** | 2022 | BART-style encoder-decoder | Denoising autoencoder on SMILES | Reaction prediction accuracy, MoleculeNet | No |
| **MolGPT** | 2022 | GPT-style decoder-only | Autoregressive on SMILES | Validity, novelty, uniqueness, KL-divergence | No |
| **Transformer-CNN** | 2023 | Hybrid transformer + CNN | Autoregressive + property prediction | ADMET benchmarks | No |

**Critical observation: No existing molecular transformer reports val_bpb (bits per byte).** They universally report downstream task performance (classification ROC-AUC, regression RMSE, generation quality metrics).

#### Recent Work (2025-2026)

Based on the current literature landscape:

- **MoLFormer-XL** (2025): Scaled linear attention to 47M params, pre-trained on 1.1B molecules. Still reports MoleculeNet benchmarks.
- **Galactica** (Meta, 2022) and successors: General scientific LLMs that include molecular data but are evaluated on broader scientific tasks.
- **DrugGPT / BioGPT variants** (2024-2025): Autoregressive models for drug-related text that include SMILES as part of mixed-domain training.
- **Graph Transformer for molecules** (multiple groups, 2024-2025): GPS, Graphormer, TokenGT -- these use graph structure, not SMILES strings, and are not directly comparable.
- **Llamole** (2025): LLaMA-based molecular model that integrates SMILES as part of instruction-following. Reports task completion metrics, not LM perplexity.

### 3.2 Can We Fairly Compare val_bpb Against Existing Metrics?

**Short answer: No. These are fundamentally incommensurable.**

| Our Metric | Existing Metrics | Why Incomparable |
|-----------|-----------------|------------------|
| val_bpb (autoregressive, character-level) | ROC-AUC on MoleculeNet | bpb measures generative modeling quality; ROC-AUC measures discriminative downstream performance |
| val_bpb | Validity/Uniqueness/Novelty | bpb is a continuous loss; validity is a binary structural check |
| val_bpb | FCD (Frechet ChemNet Distance) | FCD measures distributional similarity; bpb measures per-character prediction |

**The deeper problem:** Most molecular transformers use **masked language modeling (MLM)**, not autoregressive LM. MLM and autoregressive LM optimize different objectives:

- **Autoregressive (our setup):** P(x_t | x_1, ..., x_{t-1}) -- predicts next token given left context
- **MLM (BERT-style):** P(x_t | x_1, ..., x_{t-1}, x_{t+1}, ..., x_n) -- predicts masked token given bidirectional context
- MLM has strictly more information at prediction time, so **MLM perplexity is not comparable to autoregressive perplexity**, even on the same data.

**SMILES-specific problem:** SMILES is a linearization of a molecular graph. The "next token" in SMILES depends on decisions made at branch points (ring closures, branches) that may be far away in the string. This makes autoregressive modeling of SMILES fundamentally different from autoregressive modeling of natural language, where local context dominates.

### 3.3 Required Baselines for NeurIPS Credibility

To be taken seriously at NeurIPS, the paper MUST include:

#### Tier 1: Essential Baselines (will be rejected without these)

1. **Random architecture search (random NAS):** Same experiment budget (100 trials), but architectures are sampled randomly rather than by the agent. This is the direct control for H4 and the most critical baseline. Without it, the paper's core claim is unsupported.

2. **Fixed default transformer:** The unmodified starting architecture (e.g., GPT-2 small default config). Shows what improvement the agent achieves over the starting point.

3. **Hyperparameter-only tuning:** Same agent, but restricted to only changing hyperparameters (LR, batch size, dropout, etc.), not architecture. This isolates the contribution of architectural changes (addresses Reviewer Objection 4).

#### Tier 2: Strongly Recommended

4. **Published molecular transformer architecture:** Take the architecture of ChemBERTa or MoLFormer, adapt to autoregressive LM (same tokenization, same data), and report val_bpb. This shows whether the agent's discoveries improve upon expert-designed architectures *under identical evaluation conditions*.

5. **Grid search over key hyperparameters:** Systematic grid over depth x width x heads. Shows what simple enumeration achieves.

6. **SMILES randomization ablation:** Since SMILES has non-unique representations (each molecule has many valid SMILES), data augmentation via SMILES randomization is a well-known trick. Test whether the agent discovers this (H2) and report a baseline with/without augmentation.

#### Tier 3: Would Strengthen the Paper

7. **Bayesian optimization baseline:** Standard BO over a defined architecture space. Better than random search, tests whether the agent adds value beyond efficient search.

8. **Different agent backends:** Run with Claude, GPT-4, and an open-source LLM. Tests whether findings are agent-specific.

9. **SELFIES representation:** Run the SMILES track with SELFIES encoding instead. Every generated sequence is a valid molecule, potentially changing what architectures emerge.

### 3.4 Is SMILES Next-Token Prediction Even the Right Task?

This is a fundamental question that could undermine the paper's premise.

**Arguments FOR autoregressive SMILES modeling:**
- Simple, well-defined, easy to evaluate (val_bpb)
- Directly comparable to the NLP control track (same objective)
- Successful precedent in molecular generation (MolGPT, SMILES-Transformer)
- The autoresearch framework is designed for this objective

**Arguments AGAINST:**
- **SMILES is an arbitrary serialization.** The same molecule can be written as many different SMILES strings. A model that learns SMILES syntax is not necessarily learning chemistry -- it may be learning the serialization format. The agent might optimize for "better SMILES predictor" rather than "better molecular understanding."
- **Non-canonical SMILES are valid.** A model with perfect val_bpb on canonical SMILES may fail on randomized SMILES. This is not a failure of molecular understanding but of memorizing canonical orderings.
- **The field has moved beyond SMILES-only.** State-of-the-art molecular models (Uni-Mol, GPS++) use 3D coordinates or graph structure. A NeurIPS reviewer from the molecular ML community may view SMILES-only modeling as outdated.
- **No downstream evaluation.** Without testing discovered architectures on MoleculeNet or other downstream tasks, we cannot claim the architectures are actually useful for molecular applications.

**Recommendation:** Add a downstream evaluation as a secondary analysis. After discovering architectures via val_bpb optimization, fine-tune the best architecture on 2-3 MoleculeNet tasks (BBBP, HIV, BACE are standard) and report ROC-AUC. This bridges the gap between our metric and the field's metrics.

### 3.5 Comparison Methodology Assessment

The proposal claims "val_bpb is vocab-size-independent by design, so cross-domain comparison is fair." This is **misleading**.

While bpb normalizes for vocabulary size, it does NOT normalize for:
1. **Intrinsic entropy of the domain** (SMILES has much lower entropy than English)
2. **Sequence length distributions** (SMILES: ~43 chars; protein sequences: 100s-1000s of residues)
3. **Vocabulary regularity** (SMILES grammar is highly constrained; English is more free-form)

**The comparison that IS fair:** Comparing *how the agent behaves* (what changes it makes, what architectures it converges on) across domains. This is a qualitative/structural comparison, not a quantitative bpb comparison.

**The comparison that is NOT fair:** Saying "SMILES track achieved bpb X and NLP track achieved bpb Y, therefore molecular data is harder/easier to model." The domains have different intrinsic complexities.

### 3.6 Missing Literature That Reviewers Will Expect

The proposal must cite and discuss:

1. **Karpathy's nanoGPT / minGPT** -- the base training infrastructure
2. **The original autoresearch paper/blog post** (Karpathy, March 2026)
3. **ZINC database** (Irwin & Shoichet, 2005; Sterling & Irwin, 2015)
4. **MoleculeNet** (Wu et al., 2018) -- the standard benchmark suite
5. **SMILES** (Weininger, 1988) and **SELFIES** (Krenn et al., 2020) -- representation choices
6. **NAS survey** (Elsken et al., 2019) -- to position against classical NAS
7. **LLM-based NAS** (GPT-NAS, EvoPrompting) -- closest methodological relatives
8. **IMPROVE** (Feb 2025) -- already noted in the proposal, good
9. **Scaling laws for language models** (Kaplan et al., 2020; Hoffmann et al., 2022) -- relevant for understanding whether 5-min training is in a meaningful regime

---

## Cross-Role Challenges

### Challenge 1: Statistician to Infrastructure Engineer

**"Your throughput estimates directly affect my power analysis."**

The Infrastructure Engineer confirms that a small model can process the ZINC-250K dataset 4-8 times in 5 minutes. This means:

- **Overfitting concern undermines the metric.** If the model overfits, val_bpb will plateau or even increase, making it a noisy signal for the agent. The agent's "improvements" may be architecture changes that slightly delay overfitting rather than genuine architectural innovations.
- **Solution dependency:** If the engineer's recommendation to use SMILES randomization (data augmentation) is adopted, the effective dataset grows 10-100x, which changes the training dynamics entirely. The agent might discover SMILES randomization on its own (H2), but if some runs do and others don't, the variance between runs will be enormous, wrecking statistical power.
- **Recommendation:** Standardize data augmentation in prepare.py (make it constant across all runs). The agent should only modify train.py (architecture and training loop), not the data pipeline. This isolates the architectural signal and reduces inter-run variance.

### Challenge 2: Infrastructure Engineer to Statistician

**"Your n=5 recommendation increases costs by 67%. Is there a cheaper path to statistical rigor?"**

Going from n=3 to n=5 adds 6 more runs at ~13 hrs each = 78 additional GPU hours, or ~$55 at Tokyo spot prices.

Counter-proposals:
1. **Use the within-run trajectory (100 experiments) as the primary analysis unit.** This gives 300 data points per track. The statistician's permutation tests on architectural features could work with n=3 if the effect is clear.
2. **Run n=5 only for the most critical comparison** (SMILES vs. NLP). Run n=3 for protein. This costs only ~$36 extra.
3. **Run in us-east-1 instead of ap-northeast-1.** The 37% savings ($0.44 vs $0.70/hr) funds the additional runs at roughly the same total cost as the original $72 estimate.

### Challenge 3: Baseline Researcher to Statistician

**"You are analyzing the wrong thing. val_bpb differences across tracks are scientifically meaningless."**

The Statistician's analysis plan focuses heavily on comparing val_bpb across tracks. But as the Baseline Researcher demonstrates, absolute val_bpb is not comparable across domains due to different intrinsic entropies.

The statistically meaningful analyses are:
1. **Architectural convergence/divergence** (categorical data, not continuous bpb)
2. **Rate of improvement** (slope of learning curves, comparable in relative terms)
3. **What changes the agent makes** (a content analysis / NLP problem, not a traditional statistics problem)

**Recommendation:** The primary analysis should be a **mixed-methods approach**: quantitative analysis of learning curves + qualitative coding of architectural changes. The qualitative analysis (what did the agent change? what worked?) is arguably more valuable than any p-value.

### Challenge 4: Baseline Researcher to Infrastructure Engineer

**"Your overfitting finding kills the experimental validity."**

If a 2M-parameter model trains for 4-8 epochs through ZINC-250K in 5 minutes, we have a fundamental problem: the agent is not optimizing for *learning ability* but for *memorization resistance*. An architecture that overfits less is not necessarily a better architecture for molecular modeling -- it might just have more regularization.

This means the agent's "discoveries" could be:
- Smaller models (less capacity to memorize)
- More dropout
- Weight decay tuning
- None of which are interesting architectural innovations

**Mitigation options:**
1. Use ZINC-full (~230M molecules) instead of ZINC-250K. At 43 chars average, that is ~10B characters -- large enough that no model will overfit in 5 minutes.
2. Use SMILES randomization to expand ZINC-250K to an effectively infinite dataset.
3. Cap training to 1 epoch maximum within the 5-minute window.

**Strong recommendation: Switch to a larger SMILES dataset or implement SMILES randomization in prepare.py.** ZINC-250K is too small for this experimental paradigm.

### Challenge 5: Statistician to Baseline Researcher

**"Your downstream evaluation proposal changes the scope of the paper."**

Adding MoleculeNet fine-tuning requires:
- Additional compute budget for fine-tuning (though likely small on A10G)
- Additional baselines (published MoleculeNet results to compare against)
- Different statistical tests (ROC-AUC analysis, DeLong tests for AUC comparison)
- Potentially a new round of experiments

This expands the paper from "autonomous architecture search" (a methods paper) to "molecular transformer benchmark" (an empirical paper). These are different papers with different reviewer expectations.

**Recommendation:** Keep downstream evaluation as supplementary/ablation, not the main result. The paper's primary contribution is the methodology (recursive self-refinement) and the finding (architectures differ across domains). MoleculeNet results can support but should not drive the narrative.

### Challenge 6: Infrastructure Engineer to Baseline Researcher

**"The baseline researcher wants too many baselines. Each baseline costs compute."**

The Tier 1-3 baselines amount to roughly 7-9 additional experimental conditions. At 3 runs each:
- Random NAS: 3 runs x 3 tracks = 9 runs
- Fixed default: 1 run per track (deterministic) = 3 runs
- Hyperparameter-only: 3 runs x 3 tracks = 9 runs
- Published architecture: 1 run per track = 3 runs
- Grid search: deterministic = 3 runs
- Total: ~27 additional runs

At $0.70/hr x 13 hrs = $9.10/run, that is ~$246 for baselines alone.

**Recommendation:** Prioritize ruthlessly.
- **Must have:** Random NAS (9 runs, $82) and fixed default (3 runs, $27). Total: $109.
- **Should have:** Hyperparameter-only (9 runs, $82).
- **Nice to have:** Everything else.
- Running in us-east-1 saves ~37% across the board.

---

## Synthesized Recommendations

### Critical Changes (Must Address Before Submission)

1. **Increase dataset size or implement SMILES randomization.** ZINC-250K is too small -- models will overfit within 5 minutes, contaminating the val_bpb signal. Either switch to ZINC-full (download a 1M+ subset) or hardcode SMILES randomization in prepare.py.

2. **Do not compare absolute val_bpb across tracks.** Reframe H1 as an architectural comparison, not a metric comparison. Use architectural feature vectors and permutation tests.

3. **Budget random NAS baseline from day one.** This is the single most important baseline. Without it, H4 is untestable and the paper loses its primary quantitative claim. Budget 9 additional runs ($55-82 depending on region).

4. **Switch to us-east-1.** Saves 37% on GPU costs, funding additional runs at the same total budget. Tokyo offers no advantage for this workload.

5. **Standardize data augmentation in prepare.py.** The agent should only modify train.py. If SMILES randomization is in prepare.py, it is constant across all runs, reducing variance.

### Recommended Changes (Strengthens the Paper)

6. **Increase to n=5 for SMILES and NLP tracks.** Keep n=3 for protein. Total: 13 runs (+4), costing an additional ~$36 at us-east-1 spot prices.

7. **Use Bayesian analysis framework.** Report posterior distributions, ROPE, and Bayes factors instead of or alongside frequentist tests. This is more honest about uncertainty with small n and can provide evidence for the null (relevant if architectures converge).

8. **Add hyperparameter-only baseline.** Essential for distinguishing architectural innovation from hyperparameter tuning (directly addresses the most damaging reviewer objection).

9. **Include supplementary downstream evaluation.** Fine-tune best discovered architectures on 2-3 MoleculeNet tasks. Report as supplementary, not main result.

10. **Log and categorize all agent modifications.** Classify each change as architectural vs. hyperparameter vs. training procedure. This data supports the qualitative analysis which is arguably the paper's most interesting contribution.

### Revised Budget (Recommended Design)

| Component | Runs | Hours | Rate (us-east-1 spot) | Cost |
|-----------|------|-------|----------------------|------|
| Agent runs: SMILES (n=5) | 5 | 65 | $0.44 | $28.60 |
| Agent runs: Protein (n=3) | 3 | 39 | $0.44 | $17.16 |
| Agent runs: NLP (n=5) | 5 | 65 | $0.44 | $28.60 |
| Random NAS baseline (n=3 x 3 tracks) | 9 | 117 | $0.44 | $51.48 |
| Fixed default + HP-only baselines | 12 | 156 | $0.44 | $68.64 |
| Agent API costs (Sonnet, ~2700 calls) | -- | -- | ~$0.015/call | $40.50 |
| Contingency (15% for failures, reruns) | -- | -- | -- | $35.25 |
| **Total** | **34 runs** | **~442 hrs** | | **$270.23** |

This is roughly 3.75x the original $72 estimate but delivers a significantly more credible paper. The marginal cost of doing the experiment right is ~$200 -- negligible compared to the researcher time investment and the value of a NeurIPS acceptance.

### Final Verdict

The core idea is strong and timely. The intersection of autonomous agent-driven architecture search and molecular data is genuinely novel. However, the current experimental design has three critical vulnerabilities:

1. **Dataset too small** -- will produce overfitting artifacts instead of architectural insights
2. **No meaningful baselines budgeted** -- the paper needs random NAS at minimum
3. **Statistical claims underpowered** -- n=3 with frequentist tests will not survive review

All three are fixable within a modest budget increase. The recommended design (n=5 for key tracks, random NAS + HP-only baselines, SMILES randomization, Bayesian analysis, us-east-1 hosting) costs ~$270 total and produces a paper that can withstand NeurIPS review.

---

*Audit conducted March 9, 2026. AWS spot pricing data queried live via AWS EC2 API (ap-northeast-1: $0.70/hr spot, $1.46/hr on-demand; us-east-1: $0.44/hr spot). Literature references based on published work through March 2026.*
