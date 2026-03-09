# Product Requirements Document: recursive-mol

**Project:** Autonomous Discovery of Domain-Specific Transformer Designs for Molecular Sequences
**Codename:** recursive-mol
**Repository:** github.com/ewijaya/recursive-mol
**Author:** Rex
**Date:** March 9, 2026
**Status:** APPROVED — Conditional Go
**Target:** NeurIPS 2026 (May 15 deadline) | arXiv preprint by April 27

---

## 1. Problem Statement

Transformer architectures for molecular sequence modeling (SMILES strings, protein sequences) are overwhelmingly borrowed from NLP without systematic investigation of whether molecular data demands fundamentally different designs. Standard Neural Architecture Search (NAS) operates over predefined cell spaces too constrained to discover novel architectural patterns. Meanwhile, Karpathy's autoresearch framework (released March 7, 2026) demonstrated that an LLM coding agent can iteratively refine transformer architectures via unbounded code edits — but only for natural language data.

**No one has applied autonomous LLM-driven architecture search to molecular or biological sequence data.** This intersection is confirmed empty as of March 9, 2026.

### Core Research Question

> Do molecular sequences (SMILES, proteins) induce fundamentally different optimal transformer architectures than natural language, when discovered through autonomous agent-driven code-level search?

---

## 2. Project Goals

### 2.1 Primary Goal

Produce a NeurIPS 2026-quality empirical study that reveals what an autonomous LLM agent discovers about molecular transformer architecture when given free rein to modify model code, evaluated across three domains (SMILES, protein, NLP control).

### 2.2 Secondary Goals

| Goal | Priority | Metric |
|------|----------|--------|
| Establish first-mover priority in molecular autoresearch | P0 | arXiv preprint posted by April 27 |
| Open-source molecular autoresearch framework | P1 | Public repo with prepare.py, program.md, reproducible configs |
| Introduce "Recursive Self-Refinement" (RSR) as citable concept | P2 | Term defined and differentiated in Section 2 of paper |
| Validate cross-domain architectural transfer (SMILES ↔ protein) | P1 | Transfer matrix with statistical significance |
| Demonstrate agent superiority over random search baselines | P1 | AUC-OC improvement with 95% bootstrap CI |

---

## 3. Success Criteria

### 3.1 Hard Success Criteria (Paper Submission Gate)

Every criterion below must be met before NeurIPS submission on May 15. Failure on any single criterion triggers the corresponding pivot (Section 10).

| ID | Criterion | Measurement | Threshold | Verification |
|----|-----------|-------------|-----------|--------------|
| **SC-1** | Molecular pipeline functional | SMILES `prepare.py` runs end-to-end on ZINC-250K with SMILES enumeration; val_bpb computed | val_bpb < 4.0 on first baseline run (below uniform random ~5.3) | Automated test: `python prepare.py && python train.py` completes on g5.xlarge |
| **SC-2** | Proxy validity confirmed | Spearman rank correlation between 5-min val_bpb and 2-hr val_bpb across 20 random architectures | rho > 0.5 | Calibration study output logged to `results/calibration/` |
| **SC-3** | Agent makes architectural changes | >= 1 of 5 SMILES agent runs produces >= 3 architectural modifications (not just HP tuning) | Code diff categorization: architectural = new layers, attention patterns, activation functions, model structure changes | Manual review of `results/smiles/run_*/diffs/` |
| **SC-4** | All 13 agent runs complete | 5 SMILES + 3 protein + 5 NLP runs finish with >= 80 experiments each | 13/13 runs completed; each produces >= 80 rows in `results.tsv` | Automated: `wc -l results/*/results.tsv` |
| **SC-5** | All Tier 1 baselines complete | Random NAS (9 runs), Fixed default (3 runs), HP-only (9 runs) | 21/21 baseline runs completed | Automated check |
| **SC-6** | Transfer matrix computed | Cross-evaluate top architecture from each track on all 3 datasets | 3×3 matrix with mean + std from 3 replicates each | `results/transfer/matrix.json` exists and is complete |
| **SC-7** | MoleculeNet validation complete | Top 3 SMILES architectures fine-tuned on BBBP, HIV, BACE | ROC-AUC reported for all 9 combinations (3 archs × 3 tasks) | `results/moleculenet/scores.json` exists |
| **SC-8** | Statistical analysis complete | Bayesian hierarchical model fitted; permutation test on architectural distance matrices | All 4 hypotheses have quantified evidence (Bayes factors or p-values) | `results/analysis/hypothesis_tests.json` exists |
| **SC-9** | arXiv preprint posted | Paper uploaded to arXiv cs.LG + q-bio.QM | Live arXiv URL | URL accessible |

### 3.2 Soft Success Criteria (Paper Quality Gate)

These determine submission target (main conference vs. workshop) but do not block submission.

| ID | Criterion | Target | Fallback if missed |
|----|-----------|--------|-------------------|
| **SQ-1** | Architectural clustering by domain | Permutation test p < 0.05: cross-track architectural distance > within-track | Reframe as "universal transformer" finding |
| **SQ-2** | Agent outperforms random NAS | AUC-OC agent < AUC-OC random, 95% bootstrap CI excludes zero | Report as marginal; emphasize qualitative agent behavior analysis |
| **SQ-3** | At least one molecular "trick" approximated | Post-hoc alignment score >= 2 of 5 known techniques partially matched | Drop H2; focus on H1, H3, H4 |
| **SQ-4** | Asymmetric transfer (H3a) | Protein→SMILES degrades < SMILES→Protein by >= 5% relative | Report symmetric transfer as finding |
| **SQ-5** | MoleculeNet ROC-AUC correlates with val_bpb rank | Spearman rho > 0.5 between val_bpb ranking and MoleculeNet ranking | Report as limitation; val_bpb is imperfect proxy |
| **SQ-6** | Open-source LLM replication | At least 1 track replicated with open-source agent (DeepSeek/Llama) | Acknowledge closed-source limitation; release all logs |

### 3.3 Competition Success Criterion

| ID | Criterion | Threshold |
|----|-----------|-----------|
| **CC-1** | No competing molecular autoresearch preprint exists on arXiv when we post | Weekly fork monitoring shows zero molecular/bio forks with papers |

---

## 4. Scope

### 4.1 In Scope

| Component | Description |
|-----------|-------------|
| Fork and adapt autoresearch | Fork `karpathy/autoresearch`, fix FA3→SDPA, implement molecular prepare.py |
| Three-track experiments | SMILES (Track A), Protein (Track B), NLP Control (Track C) |
| Calibration study | 20 architectures × {5min, 2hr} to validate proxy |
| Agent runs | 13 runs total (5+3+5) with ~100 experiments each |
| Tier 1 baselines | Random NAS (9), Fixed default (3), HP-only agent (9) |
| Transfer experiments | Cross-evaluate best architectures; layer freezing analysis |
| Downstream validation | MoleculeNet fine-tuning (BBBP, HIV, BACE) |
| Statistical analysis | Bayesian hierarchical models, permutation tests, bootstrap CIs |
| Paper writing | Full NeurIPS-format paper with supplementary material |
| arXiv preprint | Posted by April 27 regardless of result completeness |

### 4.2 Out of Scope (v1)

| Component | Rationale |
|-----------|-----------|
| 3D molecular representations (GNNs, equivariant models) | Different paradigm; future work |
| Masked language modeling objective | Autoresearch uses autoregressive; changing objective breaks the framework |
| Multiple LLM agent backends (Claude vs. GPT-4 vs. open-source) | Time constraint; acknowledged as limitation. Open-source replication is SQ-6. |
| p4d.24xlarge scaling experiments | Cost-prohibitive for v1; mentioned as future work |
| Protein structure prediction downstream tasks | Requires additional infrastructure; out of scope |
| SELFIES representation | Interesting alternative to SMILES but adds complexity |
| Tier 2 baselines (published mol-transformer arch, grid search, Bayesian optimization) | Nice-to-have; implement only if time permits after Tier 1 |

---

## 5. Hypotheses (Revised)

Each hypothesis has been refined based on the stress-test panels to be specific, falsifiable, and publishable regardless of outcome.

### H1: Domain-Specific Architecture Clustering

> **Statement:** Agent-discovered architectures will cluster by domain (SMILES, protein, NLP) when measured via architectural feature vectors in a permutation test on distance matrices.

| Aspect | Detail |
|--------|--------|
| **Test** | Encode each final architecture as a feature vector (depth, width, heads, FFN ratio, activation, attention variant, window size, normalization, optimizer changes). Compute pairwise Hamming/edit distance. Permutation test: do cross-track distances exceed within-track distances? |
| **Positive result** | p < 0.05: architectures cluster by domain |
| **Negative result** | p > 0.05: "universal transformer" finding — architectures converge regardless of domain |
| **Publication narrative (positive)** | "Molecular data induces systematically different transformer designs" |
| **Publication narrative (negative)** | "At this scale, optimal transformer architecture is invariant to domain" |

### H2: Rediscovery of Domain Knowledge

> **Statement:** Agent modifications will correlate with known molecular modeling principles in a post-hoc alignment analysis, scoring >= 2/5 known techniques as partially matched across >= 2/3 of SMILES runs.

| Known Technique | What to Look For in Agent Code |
|-----------------|-------------------------------|
| Local attention for bonded atoms | Reduced attention window, sliding window patterns |
| SMILES augmentation / randomization | Any data-side trick (note: agent can't modify prepare.py, so this is unlikely) |
| Smaller embedding dimension for tiny vocab | Reduced embedding size, asymmetric embed/model dims |
| Positional encoding changes for molecular grammar | Custom PE, relative PE, rotary PE adjustments |
| Shallower/wider models for short sequences | Reduced depth, increased width relative to NLP baseline |

| **Positive result** | >= 2 techniques partially matched in >= 2/3 runs |
| **Negative result** | Agent makes no domain-informed changes → report as "agent applies generic optimization strategies regardless of domain" |

### H3: Cross-Domain Transfer (4 sub-hypotheses)

> **H3a (asymmetry):** Transfer from protein-optimized architecture to SMILES data degrades val_bpb by < 15%, while SMILES-optimized architecture on protein data degrades by > 15%.
>
> **H3b (layer specificity):** Early transformer layers (1-2) transfer with < 5% val_bpb degradation; late layers degrade > 10%.
>
> **H3c (length dominance):** In length-controlled transfer experiments, val_bpb degradation drops by > 50% when sequence lengths are matched, demonstrating length mismatch as the primary transfer barrier.
>
> **H3d (innovation classification):** Agent-discovered architectural innovations can be classified as "universal" (30-40%) vs. "domain-specific" (60-70%) via cross-domain ablation.

### H4: Search Efficiency

> **Statement:** Agent achieves lower area under the optimization curve (AUC-OC) than random NAS baseline with 95% bootstrap confidence interval excluding zero.

| **Positive result** | Agent AUC-OC significantly lower → agent search is more efficient |
| **Negative result** | No significant difference → agent's value is in interpretability/analysis, not search efficiency |

---

## 6. Technical Architecture

### 6.1 System Overview

```
┌─────────────────────────────────────────────────────┐
│                  recursive-mol                       │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐   │
│  │prepare.py│    │ train.py │    │  program.md  │   │
│  │ (FIXED)  │    │(AGENT    │    │  (HUMAN      │   │
│  │          │    │ EDITS)   │    │   WRITTEN)   │   │
│  └────┬─────┘    └────┬─────┘    └──────┬───────┘   │
│       │               │                 │            │
│       ▼               ▼                 ▼            │
│  ┌─────────────────────────────────────────────┐    │
│  │           Autoresearch Loop                  │    │
│  │                                              │    │
│  │  Agent reads program.md + results.tsv        │    │
│  │       ↓                                      │    │
│  │  Agent modifies train.py                     │    │
│  │       ↓                                      │    │
│  │  Train 5 min on g5.xlarge (A10G)            │    │
│  │       ↓                                      │    │
│  │  Evaluate val_bpb via prepare.py             │    │
│  │       ↓                                      │    │
│  │  Log to results.tsv → keep/discard           │    │
│  │       ↓                                      │    │
│  │  Repeat ~100 times                           │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  Three tracks run independently:                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐      │
│  │ Track A  │  │ Track B  │  │   Track C    │      │
│  │ SMILES   │  │ Protein  │  │ NLP Control  │      │
│  │(ZINC+enum│  │(UniRef50)│  │ (FineWeb-Edu)│      │
│  └──────────┘  └──────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────┘
```

### 6.2 Data Pipeline Specifications

#### Track A: SMILES (ZINC-250K + Enumeration)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Source | `250k_rndm_zinc_drugs_clean_3.csv` | Standard molecular generation benchmark |
| Molecules | 249,455 | Drug-like subset |
| SMILES enumeration | 5 random SMILES per molecule | Expands to ~62.5M tokens; avoids overfitting |
| Tokenization | Character-level | Matches SMILES grammar; universal in literature |
| Vocabulary | ~45 unique characters + 4 special tokens | `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>` |
| Total tokens | ~62.5M (with enumeration) | 3-5 epochs in 5 min with 12M-param model |
| Train/val split | 90/10 random molecule split | Split before enumeration — no molecule leakage |
| MAX_SEQ_LEN | 256 | Covers longest SMILES (~120 chars) with margin |
| val_bpb interpretation | Bits per SMILES character | Theoretical max: log2(45) = 5.5 bits |

#### Track B: Protein (UniRef50 Subset)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Source | UniRef50 (ftp.uniprot.org) | Standard protein sequence database |
| Filter | Length 50-500 residues | Excludes fragments and giants |
| Sequences | 50,000 random after filtering | Matches SMILES token count |
| Tokenization | Character-level (1 char = 1 amino acid) | Natural and universal for proteins |
| Vocabulary | ~25 tokens (20 AA + specials) | Smallest of all tracks |
| Total tokens | ~12.5M | 50K × ~250 avg residues |
| Train/val split | 90/10 random | |
| MAX_SEQ_LEN | 512 | Covers proteins up to 500 residues |

#### Track C: NLP Control (FineWeb-Edu)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Source | karpathy/climbmix-400b-shuffle (existing) | Original autoresearch dataset |
| Tokenization | BPE, vocab_size=8192 (existing) | Unchanged from autoresearch |
| MAX_SEQ_LEN | 2048 (existing, but may reduce for A10G) | |
| Modification | FA3→SDPA only; reduce model size to match tracks | Fair cross-track comparison |

### 6.3 Model Configuration (Starting Point)

The agent will modify `train.py` from this starting configuration. The starting point must be identical across all three tracks for fair comparison.

| Parameter | Value | Notes |
|-----------|-------|-------|
| DEPTH | 6 | Reduced from 8 for A10G throughput |
| ASPECT_RATIO | 48 | base_dim = 288 |
| HEAD_DIM | 64 | → model_dim = 320, num_heads = 5 |
| Estimated params | ~8-10M | With SMILES vocab: ~8M (tiny embedding) |
| Attention | SDPA (Flash Attention 2 backend) | Replaces FA3 for A10G compatibility |
| Activation | ReluSquared | Agent may change |
| Normalization | RMSNorm | Agent may change |
| Optimizer | MuonAdamW | Agent may change |
| TOTAL_BATCH_SIZE | 65,536 | Reduced from 524K for smaller dataset |
| DEVICE_BATCH_SIZE | 256 (SMILES), 128 (protein), 64 (NLP) | Adjusted for sequence length |
| TIME_BUDGET | 300 seconds | Fixed; agent cannot change |

### 6.4 Infrastructure

| Resource | Specification | Quantity | Region |
|----------|--------------|----------|--------|
| GPU instance | g5.xlarge (NVIDIA A10G, 24GB VRAM) | 3 concurrent (1 per track) | us-east-1 |
| Pricing | Spot: $0.44/hr | | |
| Agent backend | Claude Sonnet (primary) | ~$0.015/call | |
| Storage | EBS gp3, 100GB per instance | | |
| Total GPU hours | ~454 hrs | | |
| Total budget | ~$278 (including 15% contingency) | | |

---

## 7. The 7 Mandatory Changes

These changes are non-negotiable requirements derived from the stress-test panels. Each addresses a vulnerability that would cause NeurIPS rejection.

### Change 1: Add Downstream Molecular Benchmarks

**Problem:** val_bpb on SMILES is not meaningful to the computational biology community. No existing molecular transformer reports val_bpb. Without downstream task evaluation, the paper fails to demonstrate molecular utility.

**Requirements:**

| ID | Requirement | Acceptance Criterion |
|----|-------------|---------------------|
| C1.1 | Fine-tune top 3 discovered SMILES architectures on MoleculeNet BBBP task | ROC-AUC computed and logged; comparison with random NAS baseline architectures |
| C1.2 | Fine-tune top 3 discovered SMILES architectures on MoleculeNet HIV task | ROC-AUC computed and logged |
| C1.3 | Fine-tune top 3 discovered SMILES architectures on MoleculeNet BACE task | ROC-AUC computed and logged |
| C1.4 | Report molecular generation metrics for best SMILES architecture | Validity, uniqueness, novelty, and FCD computed from 10K generated molecules |
| C1.5 | Compute Spearman rank correlation between val_bpb ranking and MoleculeNet ranking | Correlation reported in paper (even if low) |

**Implementation:** Add `evaluate_downstream.py` script that takes a trained model checkpoint and runs MoleculeNet fine-tuning using DeepChem or PyTorch Geometric.

**Effort:** 1 day implementation + ~$20 compute
**Deadline:** Phase 4 (April 6)

### Change 2: Scale Up Dataset via SMILES Enumeration

**Problem:** ZINC-250K produces only ~12.5M tokens. A 6M-param model trains 6+ epochs in 5 minutes, creating overfitting risk. The agent would optimize for memorization resistance, not genuine architectural quality.

**Requirements:**

| ID | Requirement | Acceptance Criterion |
|----|-------------|---------------------|
| C2.1 | Implement SMILES enumeration in `prepare.py` using RDKit | `MolToSmiles(MolFromSmiles(smi), doRandom=True)` generates valid random SMILES |
| C2.2 | Generate 5 random SMILES per molecule at data preparation time | Total tokens ~62.5M (verified by `prepare.py` output log) |
| C2.3 | Ensure train/val split is by molecule, not by SMILES string | No molecule appears in both train and val (verified by set intersection check) |
| C2.4 | Protein track dataset: 50K UniRef50 sequences, length 50-500 | ~12.5M tokens (verified); download and filter script in `prepare_protein.py` |
| C2.5 | Verify overfitting regime: train_bpb vs val_bpb gap < 0.3 after 5 min | Logged in baseline run output |

**Effort:** 4 hours (SMILES) + 2 hours (protein)
**Deadline:** Phase 1 (March 16)

### Change 3: Shrink Model and Fix Flash Attention

**Problem:** FA3 crashes on A10G (Ampere sm_86). Default model (50.3M params) processes only 0.7 epochs in 5 minutes on the target dataset — insufficient signal.

**Requirements:**

| ID | Requirement | Acceptance Criterion |
|----|-------------|---------------------|
| C3.1 | Replace all FA3 calls with `torch.nn.functional.scaled_dot_product_attention` | `train.py` runs without importing `kernels` package; no FA3 references |
| C3.2 | Handle sliding window attention removal or custom mask | Training completes without error; document which attention pattern is used |
| C3.3 | Default model size: 6-12M parameters | `sum(p.numel() for p in model.parameters())` in range [6M, 12M] |
| C3.4 | Model achieves 3-8 epochs over SMILES data in 5 minutes | Epoch counter in training log shows 3-8 at t=300s |
| C3.5 | VRAM usage < 50% of 24GB | `torch.cuda.max_memory_allocated()` < 12GB |
| C3.6 | Set MAX_SEQ_LEN=256 for SMILES, 512 for protein | Configured per-track in prepare.py |

**Effort:** 2 hours (FA3→SDPA) + 1 hour (model scaling)
**Deadline:** Phase 1 (March 14) — this is the day-one blocker

### Change 4: Add Essential Baselines

**Problem:** Without baselines, hypotheses H1 and H4 are untestable and the paper will be rejected.

**Requirements:**

| ID | Requirement | Acceptance Criterion |
|----|-------------|---------------------|
| C4.1 | **Random NAS baseline:** 9 runs (3 tracks × 3 replicates). Each run: 100 random architecture modifications (random depth, width, heads, activation sampled from predefined distributions), same 5-min training, keep-best. | 9 `results.tsv` files with >= 80 rows each; final val_bpb logged |
| C4.2 | **Fixed default baseline:** 3 runs (1 per track). Train the unmodified starting architecture for 5 minutes, 3 times. | Mean and std of val_bpb for each track |
| C4.3 | **HP-only agent baseline:** 9 runs (3 tracks × 3 replicates). Same agent, but `program.md` instructs: "Only modify hyperparameters (learning rate, batch size, dropout, weight decay, warmup steps, optimizer params). Do NOT change model architecture." | 9 `results.tsv` files; code diffs verified to contain zero architectural changes |
| C4.4 | Implement `random_nas.py` script that generates random architecture configs | Script produces valid `train.py` modifications that compile and train |
| C4.5 | All baselines use identical data, tokenization, and hardware | Same `prepare.py`, same instance type, same TIME_BUDGET |

**Effort:** 1 day implementation + ~$119 compute (Tier 1)
**Deadline:** Baselines start Phase 2, complete by Phase 3

### Change 5: Increase Statistical Rigor

**Problem:** n=3 can only detect Cohen's d > 3.1. NeurIPS reviewers will reject for insufficient statistical power.

**Requirements:**

| ID | Requirement | Acceptance Criterion |
|----|-------------|---------------------|
| C5.1 | Run n=5 agent sessions for SMILES and NLP tracks | 5 completed runs per track with results.tsv |
| C5.2 | Keep n=3 for protein track | 3 completed runs |
| C5.3 | Implement Bayesian hierarchical model for H1 | PyMC/Stan model with posterior distributions; ROPE defined at +/- 0.05 relative bpb improvement |
| C5.4 | Implement permutation test for architectural distance matrices | 10,000 permutations; p-value computed |
| C5.5 | Implement bootstrap CIs for H4 (AUC-OC comparison) | 10,000 bootstrap resamples; 95% CI reported |
| C5.6 | Apply Holm-Bonferroni correction for multiple comparisons | Adjusted p-values for all 12+ tests |
| C5.7 | Do NOT report absolute val_bpb comparisons across tracks | Paper text reviewed: no claim like "SMILES bpb X vs NLP bpb Y" |
| C5.8 | Report relative improvement from baseline within each track | % reduction from fixed default, with CI |

**Effort:** 1 day statistical analysis implementation + ~$36 additional compute for extra runs
**Deadline:** Analysis framework ready by Phase 4; results by Phase 5

### Change 6: Reframe as Empirical Study

**Problem:** "Recursive Self-Refinement" framing claims methodological novelty that doesn't survive scrutiny against EvoPrompting, FunSearch, LM-Searcher. Reviewer 1 scored 4/10 specifically on novelty.

**Requirements:**

| ID | Requirement | Acceptance Criterion |
|----|-------------|---------------------|
| C6.1 | Paper title does NOT contain "Recursive Self-Refinement" | Title is descriptive and concrete (see Section 12 of this PRD) |
| C6.2 | RSR defined as internal concept in Section 2, not as paper's primary contribution | Section 2 defines RSR; abstract and intro do not lead with it |
| C6.3 | Related work cites >= 15 papers including all must-cites | EvoPrompting, FunSearch, LM-Searcher, Self-Refine, IMPROVE, The AI Scientist, MoleculeNet, MoLFormer — all present |
| C6.4 | Novelty claims position on NAS spectrum, not as new paradigm | No claim of "fundamentally different from NAS"; instead "extends LLM-guided NAS to molecular domain" |
| C6.5 | Primary contribution framed as empirical findings about domain-specific architectures | Abstract leads with scientific question, not method |
| C6.6 | Acknowledge autoresearch-robotics as concurrent domain adaptation | Footnote or related work mention |

**Effort:** Integrated into paper writing (Phase 5-6)
**Deadline:** Paper draft reviewed for framing by Phase 5

### Change 7: Validate the 5-Minute Training Proxy

**Problem:** The entire experiment rests on 5-minute val_bpb being a reliable proxy for architectural quality. Unvalidated.

**Requirements:**

| ID | Requirement | Acceptance Criterion |
|----|-------------|---------------------|
| C7.1 | Design 20 random architecture variants (vary depth 3-8, width 128-512, heads 2-8, activation {ReLU, GELU, SiLU, ReluSquared}, attention {full, windowed}) | Config JSON for each of 20 architectures |
| C7.2 | Train each architecture for 5 minutes on SMILES data | 20 val_bpb values at t=300s |
| C7.3 | Train each architecture for 2 hours on SMILES data | 20 val_bpb values at t=7200s |
| C7.4 | Compute Spearman rank correlation between 5-min and 2-hr val_bpb | Correlation coefficient and p-value |
| C7.5 | **Decision gate:** If rho > 0.7 → proceed. If 0.4-0.7 → proceed with caution, increase n. If < 0.4 → increase TIME_BUDGET to 15-30 min. | Decision logged in `results/calibration/decision.md` |
| C7.6 | Report calibration results in paper (Section 5.1 or supplementary) | Figure: scatter plot of 5-min vs 2-hr val_bpb with trend line and rho |

**Effort:** ~$15 compute + 2 hours analysis
**Deadline:** Phase 1 (March 16) — must complete before main experiments

---

## 8. Deliverables

### 8.1 Code Deliverables

| Deliverable | Description | Location | Phase |
|-------------|-------------|----------|-------|
| `prepare_smiles.py` | SMILES data pipeline with enumeration, char-level tokenizer, BPB evaluation | `src/prepare_smiles.py` | Phase 1 |
| `prepare_protein.py` | Protein data pipeline, UniRef50 subsetting, char-level tokenizer | `src/prepare_protein.py` | Phase 1 |
| `train.py` (base) | Modified autoresearch train.py: FA3→SDPA, scaled-down model | `src/train.py` | Phase 1 |
| `program.md` (molecular) | Agent instructions tailored for molecular architecture search | `src/program.md` | Phase 1 |
| `program_hponly.md` | Agent instructions restricting to HP-only changes (for baseline) | `src/program_hponly.md` | Phase 2 |
| `random_nas.py` | Random architecture generation script for baseline | `src/random_nas.py` | Phase 2 |
| `calibration.py` | Proxy validation: train 20 archs × {5min, 2hr} | `src/calibration.py` | Phase 1 |
| `evaluate_downstream.py` | MoleculeNet fine-tuning evaluation | `src/evaluate_downstream.py` | Phase 4 |
| `analyze.py` | Statistical analysis: Bayesian models, permutation tests, bootstrap CIs | `src/analyze.py` | Phase 5 |
| `transfer_eval.py` | Cross-domain transfer experiment runner | `src/transfer_eval.py` | Phase 4 |
| `categorize_diffs.py` | Classify agent code diffs as architectural vs. HP changes | `src/categorize_diffs.py` | Phase 3 |

### 8.2 Data Deliverables

| Deliverable | Description | Location | Phase |
|-------------|-------------|----------|-------|
| ZINC-250K + enumeration | Preprocessed SMILES tokens (train/val .pkl) | `data/smiles/` | Phase 1 |
| UniRef50 subset | 50K filtered protein sequences (train/val .pkl) | `data/protein/` | Phase 1 |
| Calibration results | 20 archs × {5min, 2hr} val_bpb | `results/calibration/` | Phase 1 |
| Agent run results | 13 runs × results.tsv + all train.py versions | `results/{smiles,protein,nlp}/run_*/` | Phase 2 |
| Baseline results | 21 baseline runs × results.tsv | `results/baselines/` | Phase 3 |
| Transfer matrix | 3×3 cross-evaluation results | `results/transfer/` | Phase 4 |
| MoleculeNet scores | ROC-AUC for top architectures | `results/moleculenet/` | Phase 4 |
| Statistical analysis | Posterior distributions, test results, figures | `results/analysis/` | Phase 5 |

### 8.3 Paper Deliverables

| Deliverable | Description | Phase |
|-------------|-------------|-------|
| Paper draft v1 | All sections drafted with placeholder figures | Phase 5 |
| Paper draft v2 | Figures complete, statistical analysis integrated | Phase 5 |
| arXiv preprint | Final version uploaded to arXiv cs.LG + q-bio.QM | Phase 6 |
| NeurIPS submission | Camera-ready with supplementary materials | Phase 7 |
| Supplementary materials | Full agent logs, code diffs, architecture evolution animations | Phase 6 |

---

## 9. Timeline and Milestones

### 9.1 Phase Overview

| Phase | Name | Dates | Duration | Gate |
|-------|------|-------|----------|------|
| **Phase 1** | Infrastructure & Validation | Mar 9-16 | 1 week | Checkpoint 1 |
| **Phase 2** | Main Experiments | Mar 16-23 | 1 week | Checkpoint 2 |
| **Phase 3** | Baselines & Initial Analysis | Mar 23-30 | 1 week | Checkpoint 3 |
| **Phase 4** | Transfer & Downstream Evaluation | Mar 30 - Apr 6 | 1 week | Checkpoint 4 |
| **Phase 5** | Statistical Analysis & Writing | Apr 6-20 | 2 weeks | — |
| **Phase 6** | Finalize & Preprint | Apr 20-27 | 1 week | arXiv posted |
| **Phase 7** | NeurIPS Submission | Apr 27 - May 15 | ~3 weeks | Submitted |

### 9.2 Phase Details

---

#### Phase 1: Infrastructure & Validation (March 9-16)

**Goal:** Build the molecular data pipeline, fix hardware blockers, validate the 5-minute training proxy, and confirm end-to-end feasibility on all 3 tracks.

**Depends on:** Nothing (project start)
**Blocks:** Phase 2, Phase 3, Phase 4

| Step | Task | Owner | Blocker? |
|------|------|-------|----------|
| 1.1 | Fork autoresearch; fix FA3→SDPA; remove `kernels` dependency | Engineering | **P0 BLOCKER** |
| 1.2 | Scale down model config (DEPTH=6, dim=288-320) | Engineering | |
| 1.3 | Implement `prepare_smiles.py` with SMILES enumeration | Engineering | |
| 1.4 | Implement `prepare_protein.py` with UniRef50 subsetting | Engineering | |
| 1.5 | Run baseline on all 3 tracks (verify end-to-end) | Engineering | |
| 1.6 | Run calibration study (20 archs × 5min + 2hr) | Engineering | |
| 1.7 | Write `program.md` for molecular architecture search | Engineering | |

**Checkpoint 1 (March 16) — GATE:**
- [ ] `python prepare_smiles.py` completes without error
- [ ] `python train.py` completes 5-minute training on SMILES data
- [ ] Baseline val_bpb < 4.0 (significantly below random ~5.3)
- [ ] Proxy calibration Spearman rho > 0.5
- [ ] Protein `prepare_protein.py` functional
- [ ] All 3 tracks produce valid val_bpb on a single baseline run

**Kill condition:** FA3 fix fails and no workaround → switch to p4d.24xlarge or abandon

**Agent instructions:** "Execute Phase 1 of the PRD. Fork autoresearch, fix FA3→SDPA, implement prepare_smiles.py and prepare_protein.py, scale down the model, run baseline validation on all 3 tracks, and run the calibration study. Report Checkpoint 1 criteria status when complete."

---

#### Phase 2: Main Experiments (March 16-23)

**Goal:** Run all 13 agent sessions across 3 tracks. Start Tier 1 baselines in parallel.

**Depends on:** Phase 1 (Checkpoint 1 passed)
**Blocks:** Phase 3, Phase 4

| Step | Task | Runs | Est. Time | Parallel? |
|------|------|------|-----------|-----------|
| 2.1 | Agent runs: SMILES (n=5) | 5 | ~65 hrs total | Yes (across 3 instances) |
| 2.2 | Agent runs: Protein (n=3) | 3 | ~39 hrs total | Yes |
| 2.3 | Agent runs: NLP (n=5) | 5 | ~65 hrs total | Yes |
| 2.4 | Random NAS baseline (start) | 9 | ~117 hrs total | Yes |
| 2.5 | HP-only baseline (start) | 9 | ~117 hrs total | Yes |
| 2.6 | Write `program_hponly.md` for HP-only baseline | — | 1 hr | — |
| 2.7 | Implement `random_nas.py` for random baseline | — | 3 hrs | — |

**Checkpoint 2 (March 23) — GATE:**
- [ ] >= 10 of 13 agent runs complete
- [ ] >= 1 SMILES run shows >= 3 architectural modifications (not just HP)
- [ ] No systematic crashes or data pipeline failures

**Kill condition:** Agent makes zero architectural changes across all 5 SMILES runs → pivot to workshop paper framed as "HP optimization study"

**Agent instructions:** "Execute Phase 2 of the PRD. Launch all 13 agent sessions (5 SMILES, 3 protein, 5 NLP) and all Tier 1 baselines (random NAS, HP-only). Monitor for completion. Categorize early code diffs to check for architectural vs HP changes. Report Checkpoint 2 criteria status when complete."

---

#### Phase 3: Baselines & Initial Analysis (March 23-30)

**Goal:** Complete all baselines. Categorize agent modifications. Run initial hypothesis tests (H1 permutation test).

**Depends on:** Phase 2 (Checkpoint 2 passed)
**Blocks:** Phase 5

| Step | Task | Description |
|------|------|-------------|
| 3.1 | Complete remaining baselines | All 21 Tier 1 baseline runs finished |
| 3.2 | Fixed default baseline | 3 runs (1 per track), unmodified starting architecture |
| 3.3 | Categorize agent modifications | Run `categorize_diffs.py` on all agent run code diffs |
| 3.4 | Build architectural feature vectors | Encode each final architecture as feature vector |
| 3.5 | Run permutation tests (H1) | Cross-track vs. within-track architectural distances |
| 3.6 | Begin agent behavior analysis | What strategies does the agent employ? What changes first? |
| 3.7 | Compute AUC-OC for H4 | Agent vs. random NAS optimization curves |

**Checkpoint 3 (March 30) — GATE:**
- [ ] All 21 baselines complete
- [ ] At least one qualitative architectural difference between SMILES and NLP tracks visible
- [ ] Modification categorization complete: % architectural vs. % HP changes computed
- [ ] H1 permutation test p-value computed
- [ ] H4 AUC-OC comparison computed

**Kill condition:** No visible architectural difference between any tracks → pivot to "universal transformer" framing

**Agent instructions:** "Execute Phase 3 of the PRD. Ensure all baseline runs are complete. Run categorize_diffs.py on all agent runs. Build architectural feature vectors and run the permutation test for H1. Compute AUC-OC for H4. Report Checkpoint 3 criteria status."

---

#### Phase 4: Transfer & Downstream Evaluation (March 30 - April 6)

**Goal:** Run cross-domain transfer experiments (H3). Fine-tune on MoleculeNet (Change 1). Compute generation metrics.

**Depends on:** Phase 2 (agent runs complete), Phase 3 (best architectures identified)
**Blocks:** Phase 5

| Step | Task | Description |
|------|------|-------------|
| 4.1 | Transfer experiments | Cross-evaluate best architecture from each track on all 3 datasets (36 runs) |
| 4.2 | Layer freezing experiments | Progressive layer freezing for H3b (18 runs) |
| 4.3 | MoleculeNet fine-tuning | Top 3 SMILES architectures × 3 tasks × 3 replicates (9 runs) |
| 4.4 | Generation metrics | Generate 10K molecules from best SMILES architecture; compute validity, uniqueness, novelty, FCD |
| 4.5 | Length-controlled transfer | Match SMILES/protein lengths for H3c ablation |
| 4.6 | Implement `evaluate_downstream.py` | MoleculeNet fine-tuning script using DeepChem |
| 4.7 | Implement `transfer_eval.py` | Cross-domain evaluation runner |

**Checkpoint 4 (April 6) — GATE:**
- [ ] Transfer matrix complete (3×3 with mean + std)
- [ ] MoleculeNet ROC-AUC computed for all 9 conditions
- [ ] Generation metrics (validity, uniqueness, novelty, FCD) computed
- [ ] Layer freezing curves plotted for H3b
- [ ] Length-controlled transfer results available for H3c

**Kill condition:** MoleculeNet ROC-AUC shows zero correlation with val_bpb → report as limitation, do not anchor claims on downstream performance

**Agent instructions:** "Execute Phase 4 of the PRD. Run all transfer experiments (cross-evaluation matrix, layer freezing). Implement and run MoleculeNet fine-tuning for top 3 SMILES architectures on BBBP, HIV, BACE. Generate 10K molecules and compute generation metrics. Report Checkpoint 4 criteria status."

---

#### Phase 5: Statistical Analysis & Writing (April 6-20)

**Goal:** Complete all statistical analyses. Write full NeurIPS-format paper with figures.

**Depends on:** Phase 3 (H1, H4 preliminary results), Phase 4 (transfer + downstream results)
**Blocks:** Phase 6

| Step | Task | Description |
|------|------|-------------|
| 5.1 | Bayesian hierarchical model | Fit PyMC model for H1; compute posterior distributions and ROPE |
| 5.2 | Bootstrap CIs for H4 | 10,000 bootstrap resamples on AUC-OC difference |
| 5.3 | H3 sub-hypothesis tests | Statistical tests for H3a (asymmetry), H3b (layer), H3c (length), H3d (classification) |
| 5.4 | Multiple comparison correction | Apply Holm-Bonferroni across all 12+ tests |
| 5.5 | Attention analysis | Extract and classify attention patterns from best architectures per track |
| 5.6 | Paper draft v1 | All sections drafted with placeholder figures |
| 5.7 | Figure creation | Architecture evolution plots, transfer heatmaps, val_bpb curves, attention visualizations, calibration scatter plot |
| 5.8 | Paper draft v2 | Figures complete, statistical analysis integrated, all results tables filled |
| 5.9 | Implement `analyze.py` | Unified statistical analysis script |

**No formal gate — continuous progress toward paper completion.**

**Agent instructions:** "Execute Phase 5 of the PRD. Implement analyze.py with Bayesian hierarchical models (PyMC), permutation tests, and bootstrap CIs. Run all statistical tests for H1-H4. Generate all paper figures. Draft the full paper in NeurIPS format."

---

#### Phase 6: Finalize & Preprint (April 20-27)

**Goal:** Finalize paper. Post arXiv preprint to establish priority.

**Depends on:** Phase 5 (paper draft v2 complete)
**Blocks:** Phase 7

| Step | Task | Description |
|------|------|-------------|
| 6.1 | Internal review | Co-author review of full paper draft |
| 6.2 | Revise | Address review comments; tighten framing per Change 6 |
| 6.3 | Verify citation completeness | All 15 must-cite papers present (Section 12.3 of PRD) |
| 6.4 | Prepare supplementary materials | Agent logs, code diffs, architecture evolution animations, statistical tables |
| 6.5 | **Post arXiv preprint** | **Upload to arXiv cs.LG + q-bio.QM by April 27** |
| 6.6 | Open-source repo cleanup | Ensure `recursive-mol` repo has clean README, reproducible configs, data download scripts |

**Gate: arXiv preprint live by April 27.**

**Agent instructions:** "Execute Phase 6 of the PRD. Review the paper for framing compliance (Change 6), citation completeness (15 must-cites), and statistical rigor. Prepare supplementary materials. Compile LaTeX for arXiv submission. Clean up the repo for public release."

---

#### Phase 7: NeurIPS Submission (April 27 - May 15)

**Goal:** Incorporate any early feedback. Submit to NeurIPS 2026 main conference.

**Depends on:** Phase 6 (preprint posted)

| Step | Task | Description |
|------|------|-------------|
| 7.1 | Monitor preprint feedback | Track arXiv comments, Twitter/X reactions, competing work |
| 7.2 | Incorporate feedback | Address substantive critiques in paper |
| 7.3 | Final polish | Camera-ready quality; proofread; verify all numbers |
| 7.4 | Submit NeurIPS 2026 | Main conference, ML track |
| 7.5 | Prepare workshop fallback | If main conference confidence is low, also submit to ML4Drug Discovery workshop |

**Gate: NeurIPS submission by May 15.**

**Agent instructions:** "Execute Phase 7 of the PRD. Review any feedback on the arXiv preprint. Make final revisions. Compile camera-ready LaTeX. Submit to NeurIPS 2026."

---

## 10. Kill Criteria and Pivots

| Phase | Kill Condition | Pivot Strategy | Fallback Target |
|-------|---------------|----------------|-----------------|
| Phase 1 | FA3 fix fails; model doesn't train on molecular data | Switch to p4d.24xlarge (A100) or abandon project | — |
| Phase 1 | Proxy calibration rho < 0.4 | Increase TIME_BUDGET to 15-30 min; reduce to 30-50 experiments per run | NeurIPS (delayed 1 phase) |
| Phase 2 | Agent makes zero architectural changes (HP only) across all SMILES runs | Reframe as "LLM-Guided Hyperparameter Optimization for Molecular Transformers" | NeurIPS workshop |
| Phase 3 | No architectural difference between molecular and NLP tracks | Pivot to "universal transformer" framing: "at this scale, domain doesn't matter" | NeurIPS main (different narrative) |
| Phase 4 | MoleculeNet ROC-AUC uncorrelated with val_bpb | Report as limitation; val_bpb is an imperfect proxy for molecular utility | NeurIPS (weaker downstream claims) |
| Phase 6 | Competing preprint appears on arXiv | Accelerate posting; differentiate on analysis depth or additional tracks | NeurIPS (cite concurrent work) |

---

## 11. Risk Register

| Risk | Probability | Impact | Mitigation | Owner |
|------|-------------|--------|------------|-------|
| FA3 kernel incompatible with A10G | HIGH | CRITICAL | Replace with SDPA (Change 3) | Engineering |
| ZINC-250K overfitting | HIGH | HIGH | SMILES enumeration (Change 2) | Engineering |
| n=3 underpowered → reviewer rejection | HIGH | HIGH | Increase to n=5 (Change 5) | Rex |
| Agent only tunes hyperparameters | MEDIUM-HIGH | HIGH | Constrained program.md encouraging arch changes; this itself is a finding | Rex |
| Competing preprint appears before April 27 | MEDIUM | HIGH | Move fast; post partial results preprint early if threatened | Rex |
| Muon optimizer unstable at small scale | LOW | MEDIUM | Fallback to pure AdamW | Engineering |
| Spot instance interruption mid-run | MEDIUM | LOW | Checkpoint between experiments; re-run interrupted experiment | Engineering |
| `kernels` package dependency breaks entirely | HIGH | MEDIUM | Remove dependency; use only PyTorch native ops | Engineering |
| val_bpb too noisy for architecture discrimination | LOW-MEDIUM | HIGH | Increase EVAL_TOKENS; run 2x evaluation; use calibration to quantify noise | Engineering |
| Protein sequences too long for 5-min budget | LOW | MEDIUM | Filter to <500 residues (Change 2) | Engineering |

---

## 12. Paper Specification

### 12.1 Title Options (Ranked)

1. **"Autonomous Discovery of Domain-Specific Transformer Designs for Molecular Sequences"** (recommended — concrete, searchable, no controversy)
2. "What Do Language Agents Discover About Molecular Transformer Architectures?" (question-form — strong for oral)
3. "Language Agents as Architecture Researchers: Automated Discovery of Molecular Transformer Designs" (narrative — highlights agent angle)

### 12.2 Abstract Template

> We investigate whether molecular sequence data (SMILES, proteins) induces fundamentally different optimal transformer architectures compared to natural language. Using an autonomous language agent that iteratively modifies, trains, and evaluates transformer architectures via unbounded code-level search, we conduct a controlled three-track experiment across SMILES strings (ZINC), protein sequences (UniRef50), and English text (FineWeb-Edu). We find that [KEY FINDING ON H1]. Agent-discovered architectures [RESULT ON H4] compared to random search baselines, and [RESULT ON H2 — domain knowledge]. Cross-domain transfer experiments reveal [H3 RESULT], with sequence length mismatch as the primary transfer barrier. Downstream evaluation on MoleculeNet confirms [CORRELATION RESULT]. Our analysis of agent search trajectories provides insights into how autonomous systems explore architecture space for unfamiliar scientific domains. Code, data, and complete agent interaction logs are available at [URL].

### 12.3 Required Citations (15 must-appear)

1. Karpathy (2026) — autoresearch
2. Chen et al. (NeurIPS 2023) — EvoPrompting
3. Romera-Paredes et al. (Nature 2024) — FunSearch
4. Hu et al. (EMNLP 2025) — LM-Searcher
5. Madaan et al. (NeurIPS 2023) — Self-Refine
6. IMPROVE (Feb 2025)
7. Lu et al. (ICML 2024) — The AI Scientist
8. Wu et al. (2018) — MoleculeNet
9. Ross et al. (Nature MI 2022) — MoLFormer
10. Lehman et al. (2024) — OpenELM
11. Nasir et al. (2023) — LLMatic
12. Weininger (1988) — SMILES
13. Elsken et al. (2019) — NAS survey
14. Lin et al. (2023) — ESM-2
15. Zhou et al. (ICLR 2023) — Uni-Mol

---

## 13. Competition Monitoring Protocol

| Frequency | Action | Tool |
|-----------|--------|------|
| Daily (Phase 1-2) | Search GitHub for new autoresearch forks mentioning molecular/bio/chemistry/drug/protein/SMILES | GitHub API search |
| Weekly (Phase 3+) | Search arXiv for "autoresearch" OR "molecular architecture search" OR "LLM NAS molecular" | arXiv API / Semantic Scholar |
| Weekly | Check autoresearch fork count and scan top-starred new forks | GitHub |
| On trigger | If competing preprint found: accelerate arXiv posting to within 48 hours | Manual |

---

## 14. Open Questions (To Resolve During Execution)

| ID | Question | Resolution Deadline | Decision Path |
|----|----------|-------------------|---------------|
| OQ-1 | Character-level vs. SMILES-aware tokenizer (treating `Cl`, `Br` as single tokens)? | Phase 1 | Start with character-level; implement SMILES-aware as P1 enhancement if time permits |
| OQ-2 | Should NLP control track use same model size as molecular tracks? | Phase 1 | Yes — fair comparison requires identical starting architecture |
| OQ-3 | Which Claude model version for agent? Pin specific version? | Phase 1 | Use Claude Sonnet; log exact model ID; acknowledge non-reproducibility |
| OQ-4 | Should program.md include any molecular domain hints? | Phase 1 | No hints for primary runs (tests H2); run 1 additional "hinted" run as ablation |
| OQ-5 | Sliding window attention: drop entirely or implement custom mask for SDPA? | Phase 1 | Try custom causal mask first; fall back to full attention if complex |
| OQ-6 | Open-source LLM replication (SQ-6): attempt or defer? | Phase 3 | Attempt 1 SMILES run with DeepSeek-V2 if API available; otherwise defer |

---

## 15. Approval and Sign-off

| Role | Status | Date |
|------|--------|------|
| Project Lead (Rex) | APPROVED | March 9, 2026 |
| Stress-Test Review | COMPLETE (6 panels, 22 roles) | March 9, 2026 |
| Budget Approval | APPROVED ($278 revised) | March 9, 2026 |
| Go/No-Go Decision | **CONDITIONAL GO** (65% confidence) | March 9, 2026 |

**Conditions for continued Go:** Checkpoint 1 must pass by March 16. See Section 10 for kill criteria.

---

*PRD version 1.0 — March 9, 2026*
*Derived from: stress-test-final-recommendation.md + Self-Directed-Discovery-of-Molecular-Transformer-Architectures.md*
*Next review: Phase 1 Checkpoint (March 16)*
