# PRD: Paper Writing (Phase 5–6)

**Project:** recursive-mol
**Goal:** Write a NeurIPS 2026 submission-quality paper and post to arXiv by April 27 (SC-9)
**Author:** Rex
**Date:** 2026-03-29
**Status:** READY TO EXECUTE
**Depends on:** SC-1 through SC-8 all complete
**Deadline:** arXiv preprint by April 27, 2026; NeurIPS submission by May 15, 2026

---

## 1. Paper Identity

### 1.1 Title

**"When Does Architecture Search Matter? Decomposing LLM-Guided Transformer Design Across Molecular and Language Domains"**

Alternative titles (for author review):
1. "Autonomous Discovery of Domain-Specific Transformer Designs for Molecular Sequences"
2. "Architecture Search vs. Hyperparameter Tuning: An Empirical Decomposition Across SMILES, Protein, and NLP Domains"
3. "What an AI Agent Discovers About Molecular Transformer Design — And What Transfers"

The recommended title leads with the paper's strongest finding (the decomposition) rather than the method.

### 1.2 Venue

- **Primary:** NeurIPS 2026 main conference (deadline May 15)
- **Fallback:** NeurIPS Datasets & Benchmarks track, or ML4Drug Discovery workshop
- **arXiv:** cs.LG + q-bio.QM by April 27

### 1.3 Core Contribution Statement

This paper contributes:
1. **The first empirical decomposition of architecture search vs. HP tuning across molecular and language domains**, using a 4-condition controlled design (agent, random NAS, HP-only, fixed default) with 3,106 experiments
2. **Evidence that the value of architecture search is domain-dependent**: architecture search adds 81% of improvement for NLP but negative value for SMILES, where HP tuning alone suffices
3. **A surprising universality finding**: while architectures cluster by domain (p=0.004), all 41 discovered innovations transfer freely across domains with < 1% degradation
4. **Practical guidance**: a decision framework for when to invest in architecture search vs. HP tuning, based on domain complexity (vocab size, sequence length)
5. **An open-source framework** for autonomous transformer architecture search across arbitrary sequence domains

---

## 2. Complete Results Summary

All numbers below are final and come from the completed analyses.

### 2.1 Phase 2 Data

- **34 tasks, 3,106 experiments** across 4 conditions × 3 tracks
- Conditions: agent (13 runs), random_nas (9 runs), hp_only (9 runs), fixed_default (3 runs)
- Tracks: SMILES (ZINC-250K), protein (UniRef50), NLP (FineWeb-Edu)

### 2.2 H1: Architecture Clustering

- Permutation test on architectural distance matrices: **p = 0.0037**
- Observed cross-track/within-track distance ratio: 1.43
- Architectures cluster by domain — the agent discovers different things for different data

### 2.3 H2: Domain Knowledge Rediscovery

- 4 of 5 known molecular techniques partially matched across all 5 SMILES runs
- Criterion met (≥2 techniques in ≥2 runs)
- Agent independently rediscovered: gated MLPs, attention pattern changes, value embedding strategies, depth/width rebalancing

### 2.4 H3: Cross-Domain Transfer

- **H3a (Asymmetric transfer):** Not supported — all degradation < 1%
- **H3b (Layer specificity):** Partially supported — monotonic degradation with frozen layers; early layers transfer cleanly; late layers mostly < 10% degradation (except nlp→smiles at 16%)
- **H3c (Length matching):** Not supported — truncating NLP sequences to shorter lengths actively hurts
- **H3d (Innovation classification):** 41/41 innovations classified as universal (binomial p = 2×10⁻¹⁹ against predicted 35% universal). All innovations transfer.

### 2.5 H4: Search Efficiency & Decomposition

**Agent vs Random NAS (AUC-OC):**

| Track | Bootstrap CI | p (adj) | Cohen's d | Significant? |
|-------|-------------|---------|-----------|-------------|
| SMILES | [-0.66, -0.16] | 0.044 | -1.50 | Yes |
| Protein | [-0.42, -0.17] | 0.137 | -2.86 | No (after correction) |
| NLP | [-0.54, +0.31] | 0.632 | -1.35 | No |

**Agent vs HP-only (AUC-OC):**

| Track | Bootstrap CI | p (adj) | Cohen's d | Significant? |
|-------|-------------|---------|-----------|-------------|
| SMILES | [+0.55, +1.03] | 0.015 | +1.41 | Yes — **HP-only wins** |
| Protein | [-0.77, +0.01] | 0.635 | -1.07 | No |
| NLP | [-2.05, -1.23] | 0.005 | -4.45 | Yes — **agent wins** |

**Decomposition (% of total improvement from fixed_default to agent best):**

| Track | HP contribution | Architecture contribution | Interpretation |
|-------|----------------|--------------------------|----------------|
| SMILES | 151% (p=0.001) | -51% (n.s.) | HP tuning alone exceeds agent; arch search hurts |
| Protein | 6% (n.s.) | 94% (n.s.) | Neither component significant; margins tiny |
| NLP | 19% (p=0.022) | 81% (p=0.008) | Arch search adds major value beyond HP tuning |

### 2.6 SC-6: Transfer Matrix

| Architecture → Data | SMILES | Protein | NLP |
|---------------------|--------|---------|-----|
| SMILES arch | identity | -0.08% | -0.02% |
| Protein arch | -0.71% | identity | +0.80% |
| NLP arch | +0.05% | -0.15% | identity |

### 2.7 SC-7: MoleculeNet & Generation

| Architecture | BBBP | HIV | BACE | Mean ROC-AUC |
|-------------|------|-----|------|-------------|
| Agent #1 (bpb 0.5808) | 0.702 | 0.758 | 0.805 | 0.755 |
| Agent #2 (bpb 0.5834) | 0.690 | 0.731 | 0.795 | 0.739 |
| Agent #3 (bpb 0.5839) | 0.711 | 0.735 | 0.798 | 0.748 |

Spearman ρ = 0.5 (bpb rank vs ROC-AUC rank; n=3, p=0.67)

Generation: 95.2% validity, 100% uniqueness, 99.96% novelty

### 2.8 Proxy Calibration

Spearman ρ = 0.54 (5-min vs 2-hr val_bpb; n=20, p=0.014)

---

## 3. The Narrative

### 3.1 The Question

> Do molecular sequences require fundamentally different transformer architectures than natural language? And if an autonomous agent can search the architecture space, does the search add value beyond simple hyperparameter tuning?

### 3.2 The Answer (nuanced)

**It depends on the domain.**

The paper's central finding is that the value of architecture search is **domain-dependent**, and a 4-condition decomposition can precisely quantify how much.

- **NLP (complex domain: 8K vocab, 2048 tokens):** Architecture search contributes 81% of improvement. HP tuning alone barely helps (19%). The full agent — which can modify both architecture and hyperparameters — is essential.

- **SMILES (simple domain: 50 chars, 256 tokens):** HP tuning alone captures 151% of the total improvement. Architecture search actually hurts — the agent wastes budget exploring architectural changes when it should focus on hyperparameters. A constrained HP-only agent outperforms the full agent.

- **Protein (intermediate):** Margins are too small (< 0.3% spread) for any condition to achieve significance. The domain may be constrained by factors orthogonal to architecture.

### 3.3 The Surprise

Despite architectures clustering by domain (H1, p=0.004), **all innovations transfer freely** (H3d, 100% universal). This means:

- The agent discovers different architectures for different domains, but the architectural innovations themselves are domain-agnostic
- The clustering reflects optimization path dependence (what the agent tries first), not fundamental domain requirements
- At this model scale (~10M parameters), there are no domain-locked innovations

### 3.4 The Practical Implication

The decomposition gives practitioners a **decision framework**:

| Your domain looks like... | Recommended strategy | Why |
|--------------------------|---------------------|-----|
| Small vocab, short sequences (SMILES-like) | HP tuning only | Architecture search wastes budget; HP-only agent achieves the best results |
| Large vocab, long sequences (NLP-like) | Full architecture search | 81% of gains come from architectural changes |
| Small vocab, medium sequences (protein-like) | Either — margins are thin | The domain may not benefit substantially from either |

**Additional practical takeaways:**
- Architectural innovations discovered in one domain can be directly applied to another (100% transfer rate, < 1% degradation)
- Context window size is a genuine domain constraint — truncating to shorter sequences actively hurts performance
- The 5-minute training proxy is moderately reliable (ρ=0.54) for architecture ranking — use it for coarse filtering, not final selection
- Gated MLPs (SwiGLU), aggressive KV head compression (GQA with n_kv_head=1), and learned residual scaling are universally beneficial innovations worth trying on any domain

---

## 4. Paper Structure

NeurIPS format: 9 pages main text + unlimited references + unlimited supplementary.

### Section 1: Introduction (~1.5 pages)

**Content:**
1. **Opening hook:** Molecular transformers borrow NLP architectures without questioning fit. Nobody has systematically asked whether molecules need different designs.
2. **The gap:** Traditional NAS operates over fixed search spaces. LLM-guided search can explore open-ended code modifications, but has only been applied to NLP.
3. **Our contribution:** First application of autonomous LLM-guided architecture search to molecular data (SMILES, protein), with a controlled 4-condition experimental design that decomposes the contributions of architecture search vs. HP tuning.
4. **Key finding preview:** Architecture search matters for complex domains (NLP) but not simple ones (SMILES). All innovations transfer universally.
5. **Practical significance:** A decision framework for when to invest in architecture search.

**Key figure:** Figure 1 — System overview diagram showing the 3-track experiment with the 4-condition decomposition.

### Section 2: Related Work (~1.5 pages)

**Must-cite papers (15):**

| Paper | Why it matters | How we differ |
|-------|---------------|---------------|
| Karpathy (2026) — autoresearch | Base framework; NLP-only | We extend to molecular domains; add baselines and decomposition |
| Chen et al. (NeurIPS 2023) — EvoPrompting | LLM-guided program synthesis for NAS | They use evolution; we use iterative refinement. They don't decompose arch vs HP |
| Romera-Paredes et al. (Nature 2024) — FunSearch | LLM discovers math programs | Different domain (combinatorics vs. transformers); similar methodology |
| Hu et al. (EMNLP 2025) — LM-Searcher | LLM-based NAS for NLP | NLP-only; no molecular domains; no decomposition |
| Madaan et al. (NeurIPS 2023) — Self-Refine | Iterative refinement with LLMs | General framework; we apply to architecture search |
| IMPROVE (Feb 2025) | LLM improves ML code | General ML; we focus on architecture specifically |
| Lu et al. (ICML 2024) — The AI Scientist | Autonomous scientific discovery | Broader scope; we do controlled experiment on architecture |
| Wu et al. (2018) — MoleculeNet | Standard molecular benchmark | We use for downstream validation |
| Ross et al. (Nature MI 2022) — MoLFormer | Linear attention for molecules | Domain-specific architecture designed by hand; we discover automatically |
| Lehman et al. (2024) — OpenELM | Evolution through LLMs | Evolution-based; we use iterative refinement |
| Nasir et al. (2023) — LLMatic | LLM for quality-diversity optimization | Different approach (MAP-Elites); we use sequential search |
| Weininger (1988) — SMILES | SMILES representation | Defines our data domain |
| Elsken et al. (2019) — NAS survey | NAS landscape | Positions our work in NAS literature |
| Lin et al. (2023) — ESM-2 | Protein language models | Relevant molecular transformer |
| Zhou et al. (ICLR 2023) — Uni-Mol | 3D molecular transformer | Alternative molecular modeling approach |

**Additional related work areas:**
- Hyperparameter optimization (Optuna, Hyperband) — contrast with open-ended code search
- Molecular transformers (ChemBERTa, MolBERT, Chemformer) — all use borrowed NLP architectures
- Transfer learning across domains — existing work on NLP→bio transfer

### Section 3: Methodology (~2 pages)

**3.1 Experimental Design**

| Condition | Description | Purpose |
|-----------|-------------|---------|
| Agent | LLM-guided arch + HP search | Full capability |
| Random NAS | Random architecture sampling | Controls for search strategy |
| HP-only | LLM-guided HP search, fixed arch | Controls for architecture search |
| Fixed default | No search at all | Baseline floor |

This 4-condition design enables a clean decomposition:
- agent vs. random_nas → value of LLM guidance
- agent vs. hp_only → value of architecture search
- hp_only vs. fixed_default → value of HP tuning alone
- random_nas vs. fixed_default → value of any architecture variation

**3.2 Tracks and Data**

| Track | Dataset | Vocab | Seq len | Experiments |
|-------|---------|-------|---------|-------------|
| SMILES | ZINC-250K + enumeration | ~50 chars | 256 | 5 agent + 3 NAS + 3 HP + 1 fixed |
| Protein | UniRef50 (50K seqs) | ~25 AA | 512 | 3 agent + 3 NAS + 3 HP + 1 fixed |
| NLP | FineWeb-Edu | ~8K BPE | 2048 | 5 agent + 3 NAS + 3 HP + 1 fixed |

**3.3 Agent Search Process**

- Agent: Claude Sonnet, given `program.md` instructions
- Each experiment: agent modifies train.py → 5-min training on A10G → val_bpb evaluation → keep/discard
- ~100 experiments per run
- HP-only agent: restricted to HP changes via `program_hponly.md`

**3.4 Evaluation Metrics**

- Primary: val_bpb (bits per byte)
- AUC-OC (area under optimization curve) for search efficiency
- Keep rate (fraction of experiments that improve)
- Downstream: MoleculeNet ROC-AUC (BBBP, HIV, BACE)

**3.5 Proxy Validation**

- Calibration study: 20 architectures × {5-min, 2-hr} training
- Spearman ρ = 0.54 (p=0.014) — moderate reliability

**Key figure:** Figure 2 — Experimental design matrix (4 conditions × 3 tracks) with run counts.

### Section 4: Results (~3 pages)

This is the core of the paper. Present results in order of importance.

**4.1 The Decomposition (main finding)**

**Key figure:** Figure 3 — Decomposition bar chart (`figures/h4_decomposition.png`). Stacked bars showing HP contribution vs. architecture contribution per track.

Present the decomposition table (from Section 2.5 above). This is the paper's headline result.

Discuss each track:
- NLP: arch contribution 81% (p=0.008) — architecture search is essential
- SMILES: HP contribution 151% (p=0.001), arch contribution negative — HP-only agent wins
- Protein: neither significant — domain is "architecture-insensitive" at this scale

**4.2 Search Efficiency**

**Key figures:**
- Figure 4 — Best-so-far curves per track (`figures/h4_best_so_far_*.png`), showing all 4 conditions
- Figure 5 — AUC comparison bar chart (`figures/h4_auc_comparison.png`)

Agent vs NAS: significant on SMILES (p=0.044, d=-1.50), not on others after correction.
Agent vs HP-only: HP-only wins on SMILES (p=0.015), agent wins on NLP (p=0.005, d=-4.45).

**4.3 Architecture Clustering (H1)**

**Key figure:** Figure 6 — PCA of architectures colored by track (`figures/h1_architecture_pca.png`)

Architectures cluster by domain (p=0.004). The agent discovers different designs for different data:
- SMILES: shallower, wider, full attention, SwiGLU
- NLP: aggressive KV compression (GQA n_kv_head=1), mostly full attention, post-activation RMSNorm
- Protein: per-layer value head variation, alternating window patterns, sigmoid gates

**4.4 Transfer Universality (H3)**

**Key figure:** Figure 7 — Transfer heatmap (`figures/h3_transfer_heatmap.png`)

The surprise: despite clustering, **all innovations transfer** (< 1% degradation for most pairs, 41/41 universal). Layer freezing confirms early layers transfer cleanly; late layers show mild degradation.

Length matching doesn't help — context window size is a genuine architectural constraint.

**4.5 Domain Knowledge Rediscovery (H2)**

4/5 known molecular techniques rediscovered across all 5 SMILES runs. The agent independently converges on gated MLPs, attention pattern changes, and depth/width rebalancing — without domain-specific prompting.

**4.6 Downstream Validation (MoleculeNet)**

Agent-discovered SMILES architectures achieve ROC-AUC 0.70–0.81 on BBBP/HIV/BACE. Moderate correlation with val_bpb ranking (ρ=0.5, n=3). Generation: 95.2% validity.

### Section 5: Practical Implications (~0.5 pages)

**This section directly addresses the "so what?" for practitioners.**

**5.1 When to Use Architecture Search vs. HP Tuning**

Present the decision framework:

> **Rule of thumb:** If your domain has a small vocabulary (< 100 tokens) and short sequences (< 500 tokens), start with HP tuning only — it's cheaper and likely sufficient. If your domain has a large vocabulary (> 1K tokens) and long sequences (> 1K tokens), invest in architecture search — it accounts for the majority of improvement.

Quantify the cost savings:
- HP-only agent: ~100 experiments × 5 min × $0.44/hr = ~$3.60 in GPU + ~$1.50 in API = $5 total
- Full agent: same GPU cost but architecture exploration adds risk of wasted experiments
- Random NAS: no API cost, same GPU cost

**5.2 Transferable Innovations**

List the universally beneficial innovations discovered:
1. **Grouped Query Attention (n_kv_head=1):** 5:1 key-value sharing. Reduces parameters with negligible quality loss.
2. **SwiGLU/GeGLU activation:** Gated MLP consistently beats ReluSquared.
3. **Learned residual scaling:** Per-layer attention and MLP scaling parameters improve training.
4. **Value embeddings on every layer:** Better than alternating-layer strategy.

These can be applied to any transformer architecture regardless of domain.

**5.3 Framework Reusability**

The open-source framework can be pointed at any new sequence domain:
- Glycan sequences, reaction SMILES, ADMET strings, gene sequences
- Cost: ~$5-10 per architecture search
- Time: overnight on a single GPU

### Section 6: Discussion & Limitations (~0.5 pages)

**Limitations to acknowledge:**
1. **Small model scale (~10M params).** The universality finding may not hold at larger scales where domain-specific patterns become more important.
2. **Short training proxy (5 min).** Calibration ρ=0.54 means rankings can flip with longer training.
3. **Single agent backend (Claude Sonnet).** Results may differ with other LLMs.
4. **n=3-5 runs per condition.** Low statistical power for some comparisons (protein track).
5. **No SELFIES or 3D representations.** SMILES is one of several molecular representations.
6. **Closed-source agent.** Exact reproducibility is limited by API non-determinism.

**Discussion points:**
- Why does architecture search hurt on SMILES? Likely because the domain is "simple enough" that the default architecture is near-optimal, and exploring changes wastes experiments that could improve HP.
- The clustering-but-universal paradox: agents follow different search paths for different data, but the innovations they discover along those paths happen to be universally helpful. This suggests the search dynamics (path dependence) matter more than fundamental domain requirements at this scale.
- Implications for the "universal transformer" hypothesis at larger scales.

### Section 7: Conclusion (~0.25 pages)

Summarize: architecture search value is domain-dependent. Provide the decision framework. State that all code, data, and 3,106 experiment logs are open-source.

### References (~1 page, unlimited)

15 must-cite + ~20–30 additional references.

### Supplementary Material (unlimited)

- Full decomposition tables with bootstrap CIs
- All best-so-far curves
- Architectural feature vectors for all 13 agent runs
- Transfer matrix with individual replicate values
- Layer freezing curves
- MoleculeNet per-task per-replicate scores
- Generation metrics details
- Training dynamics analysis (convergence, stability, MFU)
- Distribution violin plots
- Crash rate analysis
- Agent prompt (program.md and program_hponly.md)
- Complete list of architectural modifications per run

---

## 5. Figure Plan

### Main Paper Figures (target: 6-8 figures in 9 pages)

| Fig # | Content | Source file | Purpose |
|-------|---------|-------------|---------|
| 1 | System overview + experimental design | NEW — to be created | Introduces the 3-track, 4-condition setup |
| 2 | **Decomposition bar chart** | `figures/h4_decomposition.png` | **Headline result** — HP vs arch contribution per track |
| 3 | Best-so-far curves (3-panel: SMILES, protein, NLP) | `figures/h4_best_so_far_*.png` | Search trajectories across conditions |
| 4 | Architecture PCA + distance heatmap (2-panel) | `figures/h1_architecture_pca.png` + `figures/h1_distance_heatmap.png` | H1 clustering result |
| 5 | Transfer heatmap | `figures/h3_transfer_heatmap.png` | H3 universality finding |
| 6 | MoleculeNet bar chart | `figures/moleculenet_bar.png` | Downstream validation |
| 7 | Layer freezing curves | `figures/h3_layer_freezing.png` or `figures/layer_freezing_curves.png` | H3b layer specificity |

### Supplementary Figures

| Fig # | Content | Source file |
|-------|---------|-------------|
| S1 | AUC comparison bars | `figures/h4_auc_comparison.png` |
| S2 | Keep rate curves | `figures/h4_keep_rate.png` |
| S3 | Anytime performance table | `figures/h4_anytime_table.png` |
| S4 | Distribution violins (3 panels) | `figures/distribution_violin_*.png` |
| S5 | H2 technique heatmap | `figures/h2_technique_heatmap.png` |
| S6 | Permutation null distribution | `figures/h1_permutation_null.png` |
| S7 | Training dynamics | `figures/training_dynamics_*.png` |
| S8 | Innovation pie chart | `figures/innovation_pie.png` |
| S9 | bpb vs ROC-AUC scatter | `figures/bpb_vs_rocauc_scatter.png` |

### Figures to Create

1. **Figure 1: System overview** — diagram showing:
   - The autoresearch loop (agent → modify train.py → train 5 min → evaluate → keep/discard)
   - Three parallel tracks (SMILES, protein, NLP)
   - Four conditions (agent, random NAS, HP-only, fixed default)
   - Use TikZ or matplotlib

2. **Practitioner decision flowchart** (optional, for Section 5) — simple flowchart:
   - Is vocab > 1K tokens? → Yes: architecture search. No: →
   - Is sequence length > 1K? → Yes: architecture search. No: HP tuning only.

---

## 6. Tables Plan

### Main Paper Tables

| Table # | Content | Section |
|---------|---------|---------|
| 1 | Experimental design: conditions × tracks × runs × experiments | Section 3 |
| 2 | **Decomposition results**: total improvement, HP%, arch%, p-values per track | Section 4.1 |
| 3 | Agent vs baselines: AUC-OC, bootstrap CI, Cohen's d, significance per track | Section 4.2 |
| 4 | Transfer matrix: 3×3 with degradation percentages | Section 4.4 |
| 5 | Decision framework: domain characteristics → recommended strategy | Section 5.1 |
| 6 | MoleculeNet ROC-AUC per architecture × task | Section 4.6 |

### Supplementary Tables

| Table # | Content |
|---------|---------|
| S1 | All 34 tasks with run details, best_val_bpb, num_crash |
| S2 | Architectural feature vectors for 13 agent-best runs |
| S3 | Anytime performance (best-so-far at N=5,10,20,50,100) |
| S4 | Time-to-threshold analysis |
| S5 | H2 technique × run binary matrix |
| S6 | Layer freezing degradation per pair × freeze level |
| S7 | All p-values with raw and Holm-Bonferroni adjusted |
| S8 | Crash rates per condition × track |

---

## 7. Writing Guidelines

### 7.1 Tone

- Empirical, not hype. "We find that..." not "We demonstrate the superiority of..."
- Honest about negative results: SMILES HP-only beating agent is a real finding, not a failure
- The decomposition is the contribution — frame it as enabling practitioners to make informed decisions

### 7.2 Claims to Avoid

- Do NOT claim "agent always outperforms baselines" — it doesn't on SMILES
- Do NOT compare absolute val_bpb across tracks — different intrinsic entropy
- Do NOT claim methodological novelty for the autoresearch framework — it's Karpathy's; our novelty is the molecular application and 4-condition decomposition
- Do NOT overclaim from n=3 comparisons — be transparent about power limitations
- Do NOT present the protein results as significant — they're not after correction

### 7.3 Claims to Make

- The decomposition is novel: no prior work separates architecture search from HP tuning with this design
- The domain-dependent finding is actionable: practitioners can use it to allocate compute budgets
- Universal innovation transfer is surprising and practically useful
- The framework is reusable for arbitrary domains at low cost
- The qualitative agent behavior analysis (H2) shows genuine reasoning about domain structure

### 7.4 Framing the Negative Results

SMILES HP-only > agent is **not a failure**. Frame as:

> "For domains with small vocabularies and short sequences, architecture search is counterproductive — it consumes experimental budget on structural changes that don't improve over the near-optimal default. This finding is directly actionable: practitioners working on SMILES-like domains should constrain their agents to hyperparameter-only search."

### 7.5 NeurIPS Format Requirements

- 9 pages main text (excluding references and supplementary)
- LaTeX template: `neurips_2026.sty`
- Double-blind review (anonymize author names, institution, repo URL)
- Supplementary material in same PDF or separate file
- Ethics statement required
- Reproducibility checklist required

---

## 8. Required Citations

### Must-Appear (15 — from PRD)

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

### Strongly Recommended Additional

16. Irwin & Shoichet (2005) — ZINC database
17. Kaplan et al. (2020) — Scaling laws
18. Hoffmann et al. (2022) — Chinchilla scaling
19. Touvron et al. (2023) — LLaMA (GQA reference)
20. Shazeer (2019) — Multi-query attention / GQA
21. Dauphin et al. (2017) — Gated linear units / SwiGLU origin
22. Shazeer (2020) — GLU variants
23. Vaswani et al. (2017) — Transformer (obviously)
24. Chithrananda et al. (2020) — ChemBERTa
25. Ahmad et al. (2022) — ChemBERTa-2
26. Fabian et al. (2020) — MolBERT
27. Irwin et al. (2022) — Chemformer
28. Li & Hoiem (2017) — Learning without forgetting (transfer learning)
29. Bergstra & Bengio (2012) — Random search for HPO
30. Falkner et al. (2018) — BOHB

---

## 9. Practitioner-Facing Content Plan

This section details how to make the paper useful beyond academia. The PRD emphasizes this because the project's impact depends on practitioners actually changing their behavior based on the findings.

### 9.1 Decision Framework (Section 5.1 of paper)

**The core deliverable for practitioners is a simple decision tree:**

```
Given a new sequence modeling domain:

1. What is the vocabulary size?
   - < 100 tokens (like SMILES, amino acids): → HP tuning is likely sufficient
   - 100–1000 tokens: → Try HP-only first; architecture search if budget allows
   - > 1000 tokens (like NLP): → Architecture search recommended

2. What is the typical sequence length?
   - < 500 tokens (like SMILES): → HP tuning is likely sufficient
   - 500–1000 tokens: → Domain-dependent; protein-like domains show thin margins
   - > 1000 tokens (like NLP): → Architecture search recommended

3. Budget-aware recommendation:
   - If budget < 20 experiments: → HP-only agent (agent finds good HPs faster)
   - If budget 20–50 experiments: → HP-only for simple domains, full agent for complex
   - If budget > 50 experiments: → Full agent for complex domains only
```

**Why this framework works:** The SMILES/NLP contrast provides the extremes, and protein fills the middle. The vocab size and sequence length are observable before starting — they don't require running experiments first.

### 9.2 Transferable Innovation Catalog (Section 5.2 of paper)

Present each universally beneficial innovation with:
- **What it is** (one sentence)
- **How to implement** (one code snippet or config change)
- **Expected impact** (observed improvement range)
- **Where discovered** (which track, which run)

Example:

> **Grouped Query Attention (n_kv_head=1)**
> Reduce key-value heads to 1 while keeping 5 query heads (5:1 compression).
> Discovered by NLP agent (run_3, exp081). Transfers to SMILES and protein with < 0.2% degradation.
> Implementation: set `n_kv_head=1` in GPTConfig.
> Saves ~15% attention parameters with negligible quality loss.

### 9.3 Cost-Benefit Analysis (Section 5 or Supplementary)

| Strategy | GPU cost | API cost | Total | Best SMILES bpb | Best NLP bpb |
|----------|---------|---------|-------|-----------------|-------------|
| Fixed default (0 search) | $0 | $0 | $0 | 0.5961 | 1.1528 |
| HP-only agent (100 exp) | ~$3.50 | ~$1.50 | ~$5 | **0.5801** | 1.1470 |
| Random NAS (100 exp) | ~$3.50 | $0 | ~$3.50 | 0.5906 | 1.1297 |
| Full agent (100 exp) | ~$3.50 | ~$1.50 | ~$5 | 0.5808 | **1.1151** |

**Key insight:** For SMILES, the $5 HP-only search achieves the best result. For NLP, the $5 full agent search achieves the best result. Random NAS ($3.50, no API) is a strong middle ground.

### 9.4 "What Should I Try on My Data?" Checklist

For a practitioner reading the paper:

1. Start with the default transformer config and measure baseline val_bpb
2. Run HP-only tuning for 20-50 experiments (~$2-3)
3. If improvement saturates: your domain is SMILES-like. Stop here.
4. If improvement continues: your domain may benefit from architecture search. Run full agent.
5. If you discover a good innovation: it probably transfers to your other domains (100% transfer rate in our experiments)

---

## 10. Source File Reference

This section maps each paper section to the codebase files that contain the authoritative data, code, and documentation the writer should consult.

### 10.1 Results Data (JSON)

All statistical results live in `results/analysis/`. Always extract exact numbers from these files — do not copy from the tables in this PRD without verifying against the JSON.

| File | Content | Used in paper section |
|------|---------|----------------------|
| `results/analysis/hypothesis_tests.json` | Master summary of all p-values, effect sizes, CIs | Sections 4.1–4.5 |
| `results/analysis/h1_permutation_test.json` | Architecture clustering p-value, distance ratio | Section 4.3 (H1) |
| `results/analysis/h1_distance_matrix.json` | Within/cross-track distance matrices | Section 4.3 (H1) |
| `results/analysis/h1_bayesian_posterior.json` | Bayesian clustering posterior | Supplementary |
| `results/analysis/h2_technique_matrix.json` | Domain knowledge rediscovery per run | Section 4.5 (H2) |
| `results/analysis/h2_fisher_tests.json` | Fisher exact tests for technique enrichment | Section 4.5 (H2) |
| `results/analysis/h3_transfer_tests.json` | All H3a–H3d sub-hypothesis results | Section 4.4 (H3) |
| `results/analysis/h4_auc_values.json` | AUC-OC values per condition per track | Section 4.2 (H4) |
| `results/analysis/h4_decomposition.json` | HP vs arch contribution percentages | Section 4.1 (headline result) |
| `results/analysis/multiple_comparisons.json` | Bonferroni-corrected p-values | All results sections |
| `results/analysis/crash_rates.json` | Training failure rates per condition | Supplementary |
| `results/analysis/distribution_stats.json` | Performance distributions | Supplementary |
| `results/analysis/training_dynamics.json` | Convergence, stability, MFU metrics | Supplementary |

### 10.2 Figures

All publication-ready figures are in `figures/`. See Section 5 of this PRD for the complete figure plan mapping figure numbers to files.

### 10.3 Analysis Scripts

| File | Purpose | Relevant paper section |
|------|---------|----------------------|
| `scripts/analyze_phase2.py` | Generates all hypothesis tests, figures, and JSON results | Section 3 (methodology), Section 4 (all results) |
| `scripts/analyze_training_dynamics.py` | Training convergence and stability analysis | Supplementary |
| `scripts/transfer_eval.py` | Layer freezing and cross-domain transfer | Section 4.4 (H3) |
| `scripts/moleculenet_eval.py` | Downstream MoleculeNet evaluation | Section 4.6 |
| `scripts/_eval_common.py` | Shared evaluation utilities | — |

### 10.4 Training Code & Agent Prompts

| File | Purpose | Relevant paper section |
|------|---------|----------------------|
| `src/train.py` | Main training loop — describes the model architecture, training procedure, and eval metrics | Section 3 (methodology) |
| `src/program.md` | Full agent system prompt (arch + HP search) | Section 3.3, Supplementary (agent prompt appendix) |
| `src/program_hponly.md` | HP-only agent prompt | Section 3.3, Supplementary |
| `src/prepare_smiles.py` | SMILES data preparation pipeline | Section 3.2 (data) |
| `src/prepare_protein.py` | Protein data preparation pipeline | Section 3.2 (data) |
| `src/prepare.py` | NLP data preparation pipeline | Section 3.2 (data) |
| `src/calibration.py` | Proxy validation (5-min vs 2-hr) | Section 3.5 |
| `src/phase2_runner.py` | Agent orchestrator — describes how the LLM agent runs experiments | Section 3.3 |
| `src/random_nas.py` | Random NAS baseline generator | Section 3.1 (conditions) |

### 10.5 Experiment Results (Raw)

| Directory | Content |
|-----------|---------|
| `results/smiles/` | All SMILES agent runs (1-5) + baselines |
| `results/protein/` | All protein agent runs (1-3) + baselines |
| `results/nlp/` | All NLP agent runs (1-5) + baselines |
| `results/baselines/` | Random NAS and HP-only runs |
| `results/calibration/` | Proxy validation experiments |
| `results/transfer/` | Layer freezing and cross-domain transfer data |
| `results/moleculenet/` | MoleculeNet downstream evaluation data |

### 10.6 Strategic & Review Preparation Documents

These docs in `docs/` help anticipate reviewer questions and frame the narrative:

| File | Purpose | Relevant paper section |
|------|---------|----------------------|
| `docs/baseline-architecture-rationale.md` | Justification for the 4-condition design | Section 3.1 |
| `docs/stress-test-adversarial-reviews.md` | Pre-rebuttal Q&A for likely reviewer questions | Discussion, rebuttal prep |
| `docs/stress-test-novelty-prior-art.md` | Positioning vs LLM-guided NAS papers | Section 2 (related work) |
| `docs/stress-test-experimental-design-audit.md` | Design critique and validation | Section 3, Section 6 |
| `docs/stress-test-transfer-hypothesis.md` | Transfer findings robustness | Section 4.4, Section 6 |
| `docs/stress-test-paper-positioning.md` | Narrative framing guidance | Introduction, Discussion |
| `docs/stress-test-technical-feasibility.md` | Reproducibility concerns | Section 6, reproducibility checklist |
| `docs/stress-test-final-recommendation.md` | Go/no-go assessment | Overall framing |
| `docs/llm_guided_search_vs_hpo.md` | Conceptual foundation for HP-only baseline | Section 2, Section 3 |
| `docs/PRD-SC8-statistical-analysis.md` | H1–H4 hypothesis test methodology | Section 3.4 |
| `docs/PRD-SC8-H3-transfer-analysis.md` | Transfer experiment protocol | Section 3, Section 4.4 |
| `docs/PRD-recursive-mol.md` | Original project spec with research questions | Background context |

---

## 11. Implementation Plan

### 11.1 File Structure

```
manuscript/
  main.tex                    — Master file that \input's all sections
  neurips_2026.sty            — NeurIPS style file
  references.bib              — BibTeX references
  sections/
    abstract.tex              — Abstract
    introduction.tex          — Section 1: Introduction
    related_work.tex          — Section 2: Related Work
    methodology.tex           — Section 3: Methodology
    results.tex               — Section 4: Results
    practical.tex             — Section 5: Practical Implications
    discussion.tex            — Section 6: Discussion & Limitations
    conclusion.tex            — Section 7: Conclusion
    supplementary.tex         — Supplementary Material (appended or separate)
    ethics.tex                — Ethics Statement
    reproducibility.tex       — Reproducibility Checklist
  figures/                    — Symlink or copy from project figures/
  tables/                     — Generated LaTeX table files
```

**`main.tex` structure:**

```latex
\documentclass{article}
\usepackage[final]{neurips_2026}  % or [preprint] for arXiv
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{xcolor}

\title{When Does Architecture Search Matter? Decomposing LLM-Guided Transformer Design Across Molecular and Language Domains}

% --- FOR DOUBLE-BLIND SUBMISSION: use this ---
% \author{Anonymous}

% --- FOR CAMERA-READY: uncomment this block ---
\author{
  Edward Wijaya \\
  StemRIM, Inc. \\
  \texttt{wijaya@stemrim.com}
}

\begin{document}
\maketitle

\input{sections/abstract}
\input{sections/introduction}
\input{sections/related_work}
\input{sections/methodology}
\input{sections/results}
\input{sections/practical}
\input{sections/discussion}
\input{sections/conclusion}

\bibliographystyle{plainnat}
\bibliography{references}

\appendix
\input{sections/ethics}
\input{sections/reproducibility}
\input{sections/supplementary}

\end{document}
```

Each section file should contain only the section content (no `\documentclass`, `\begin{document}`, etc.) — just the `\section{...}` and body text.

### 11.2 Writing Order

Write sections in this order (dependencies flow downward). Each section is a separate file in `manuscript/sections/`.

1. **`methodology.tex`** — most mechanical, establishes notation
2. **`results.tex`** — fill in numbers from completed analyses
3. **`practical.tex`** — synthesize results into practitioner guidance
4. **`introduction.tex`** — can now reference results and frame the story
5. **`related_work.tex`** — position against literature
6. **`discussion.tex`** — reflect on findings and limitations
7. **`conclusion.tex`** — summarize
8. **`abstract.tex`** — write last, when the story is crystallized
9. **`supplementary.tex`** — compile all supporting material
10. **`ethics.tex`** + **`reproducibility.tex`** — NeurIPS requirements

### 11.3 Timeline

| Date | Milestone |
|------|-----------|
| Mar 30 – Apr 5 | Sections 3, 4 drafted; all tables and figures placed |
| Apr 5 – Apr 12 | Sections 1, 2, 5, 6, 7 drafted; full first draft complete |
| Apr 12 – Apr 19 | Internal review, revision, figure polish |
| Apr 19 – Apr 25 | Final revision, supplementary compiled |
| Apr 26 | LaTeX compiled, proofread |
| **Apr 27** | **arXiv posted (SC-9)** |
| May 1 – May 14 | Incorporate any early feedback; prepare NeurIPS submission |
| **May 15** | **NeurIPS submission** |

### 11.4 Anonymization for Double-Blind

- Replace "recursive-mol" with "our framework" or "[FRAMEWORK]"
- Remove author names and affiliations
- Replace GitHub URL with "available upon acceptance"
- Remove any references to StemRIM, ISDD, or specific internal projects
- Ensure supplementary material doesn't leak identity

---

## 12. Verification Checklist

Before arXiv submission:

- [ ] All 15 must-cite papers present in references
- [ ] No absolute val_bpb comparisons across tracks
- [ ] Decomposition table matches `hypothesis_tests.json` exactly
- [ ] All p-values reported with both raw and adjusted values
- [ ] SMILES HP-only > agent finding prominently discussed (not hidden)
- [ ] Protein non-significance honestly reported
- [ ] NeurIPS format: 9 pages main text
- [ ] Figures are legible at print resolution
- [ ] Ethics statement included
- [ ] Reproducibility checklist completed
- [ ] Supplementary includes complete experiment logs
- [ ] Code/data availability statement present (anonymized)
- [ ] No identity-revealing information in double-blind version
- [ ] Decision framework in Section 5 is concrete and actionable
- [ ] Transferable innovations listed with implementation guidance

---

*PRD version 1.0 — March 29, 2026*
*Derived from: PRD-recursive-mol.md (Sections 12, Phase 5-6), all SC results, pitch.md, contingency-narratives.md*
