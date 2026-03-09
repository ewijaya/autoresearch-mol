# Final Comprehensive Recommendation: Recursive Self-Refinement for Molecular Transformers

**Date:** March 9, 2026
**Synthesized from:** 6 independent stress-test panels (22 simulated expert roles)
**Verdict:** **CONDITIONAL GO — with mandatory redesign of 7 critical elements**

---

## Executive Summary

Six stress-test teams evaluated the paper proposal "Self-Directed Discovery of Molecular Transformer Architectures via Recursive Self-Refinement" across novelty, feasibility, experimental design, positioning, and hypothesis rigor. The core scientific question — *do molecular sequences induce different optimal transformer architectures than natural language?* — is genuinely interesting and timely. The intersection of {LLM coding agent} × {architecture discovery} × {molecular data} is confirmed empty. However, the current experimental design has **critical vulnerabilities** that would result in a NeurIPS rejection (aggregate adversarial score: 3.3/10).

The good news: every vulnerability is fixable within the proposed timeline and a modest budget increase ($72 → ~$300). The project should proceed, but **not as currently designed**.

---

## Verdict by Dimension

| Dimension | Score | Status | Key Issue |
|-----------|-------|--------|-----------|
| **Novelty** | 7/10 | GREEN (3 claims) / YELLOW (1 claim) | Triple intersection is empty; method claim is weak vs. EvoPrompting/LM-Searcher |
| **Technical Feasibility** | 6/10 | GO with fixes | FA3 kernel blocker on A10G; model too large; dataset too small |
| **Experimental Design** | 3/10 | MUST REDESIGN | n=3 underpowered; no baselines; overfitting risk; val_bpb incomparable across tracks |
| **Molecular Relevance** | 2/10 | CRITICAL GAP | No downstream benchmarks; val_bpb meaningless to comp bio community |
| **Paper Positioning** | 5/10 | NEEDS REFRAMING | RSR title risky; contribution is empirical, not methodological |
| **Transfer Hypothesis (H3)** | 7/10 | WELL-DESIGNED | Strong distributional analysis; sequence length (7.7x) is the key barrier |
| **Competition Risk** | HIGH | URGENT | 1,300 forks in 3 days; robotics fork already exists; preprint needed by late April |

---

## The 7 Mandatory Changes

These are non-negotiable for a credible submission. Skip any one and the paper will be rejected.

### 1. Fix the Evaluation Crisis: Add Downstream Molecular Benchmarks

**Problem:** val_bpb on SMILES is not a meaningful metric for the computational biology community. No existing molecular transformer reports val_bpb — they all use MoleculeNet ROC-AUC, generation quality metrics, or binding affinity. A paper optimizing val_bpb alone will be dismissed as optimizing an irrelevant proxy.

**Fix:**
- After architecture search, fine-tune the top 3 discovered architectures on **3 MoleculeNet classification tasks** (BBBP, HIV, BACE — standard, fast to evaluate)
- Report ROC-AUC as **secondary validation** that val_bpb improvements translate to molecular utility
- For the SMILES track, also report **molecular generation metrics**: validity, uniqueness, novelty, and FCD (Fréchet ChemNet Distance)
- Present val_bpb as the search objective and MoleculeNet as the validation — this bridges the ML and comp bio communities

**Effort:** ~1 day implementation + ~$20 additional compute

### 2. Scale Up the Dataset: ZINC-250K Is a Toy

**Problem:** ZINC-250K produces only ~12.5M character-level tokens. A 6M-parameter model trains 6+ epochs in 5 minutes, creating severe overfitting risk. The agent would optimize for memorization resistance rather than genuine architectural quality. By 2026 standards, 250K molecules is a debugging dataset.

**Fix (choose one or combine):**
- **Option A (recommended):** Implement SMILES enumeration (randomization) in `prepare.py` — generate 5 random SMILES per molecule, expanding to ~62.5M tokens effectively for free. This is also a well-known molecular modeling technique, giving the agent something meaningful to "discover" (H2).
- **Option B:** Use a larger ZINC subset (ZINC-1M or ZINC-2M from ZINC15/ZINC22) — ~50-100M tokens.
- **Option C:** Use the GuacaMol benchmark set (~1.6M molecules from ChEMBL).

**For protein track:** Subset UniRef50 to ~50K sequences filtered to <500 residues → ~12.5M tokens (matches SMILES scale).

**Effort:** ~4 hours for SMILES enumeration; ~2 hours for protein subsetting

### 3. Shrink the Model and Fix Flash Attention

**Problem:** The default autoresearch model is 50.3M parameters with Flash Attention 3 (Hopper-only). On A10G (Ampere, sm_86):
- FA3 will crash on import — this is a **day-one blocker**
- The 50.3M model processes only ~0.7 epochs of ZINC-250K in 5 minutes — insufficient for learning signal
- With a tiny SMILES vocabulary (~45 chars), embedding parameters drop from ~25M to ~0.1M, making the 50.3M budget wildly misallocated

**Fix:**
- Replace FA3 with `torch.nn.functional.scaled_dot_product_attention` (SDPA) — automatically dispatches to Flash Attention 2 on Ampere. Note: SDPA doesn't natively support sliding window; use custom causal mask or drop windowed attention.
- Scale model to **6-12M parameters** (DEPTH=6, dim=288-320, 4-5 heads). This achieves 3-8 epochs in 5 minutes — the sweet spot for architecture search signal.
- Set `MAX_SEQ_LEN=256` for SMILES (sequences average ~43 chars), `MAX_SEQ_LEN=512` for proteins.
- Reduce `TOTAL_BATCH_SIZE=65536` (from 524K) since the dataset is smaller.

**Effort:** ~2 hours for FA3→SDPA; ~1 hour for model scaling

### 4. Add Essential Baselines From Day One

**Problem:** Without baselines, no hypothesis is testable. The paper currently proposes zero baselines to run alongside agent experiments.

**Fix — Tier 1 (will be rejected without these):**

| Baseline | Purpose | Runs | Cost (us-east-1 spot) |
|----------|---------|------|----------------------|
| **Random NAS** | Control for H4 (agent vs. random) | 9 (3 tracks × 3) | $51 |
| **Fixed default transformer** | Shows agent improvement over starting point | 3 (1 per track) | $17 |
| **Hyperparameter-only agent** | Isolates architectural vs. HP contributions | 9 (3 tracks × 3) | $51 |

**Fix — Tier 2 (strongly recommended):**

| Baseline | Purpose | Runs | Cost |
|----------|---------|------|------|
| **Published molecular transformer** | ChemBERTa/MoLFormer arch adapted to autoregressive | 3 | $17 |
| **Grid search** (depth × width × heads) | What simple enumeration achieves | 3 | $17 |

**Total baseline cost:** ~$119 (Tier 1) or ~$153 (Tier 1+2) at us-east-1 spot pricing

### 5. Increase Statistical Rigor

**Problem:** n=3 per condition can only detect Cohen's d > 3.1 (enormous effects). NeurIPS reviewers will flag this immediately. The 100 experiments within each run are correlated (sequential, each depends on prior agent state), so they are not independent samples.

**Fix:**
- Increase to **n=5 for SMILES and NLP tracks** (the key comparison for H1). Keep n=3 for protein track. Total: 13 agent runs (up from 9).
- Use **Bayesian hierarchical models** with ROPE (Region of Practical Equivalence) instead of frequentist tests. With small n, Bayesian credible intervals are more honest about uncertainty, and Bayes factors can provide evidence *for* the null (relevant if architectures converge — the "universal transformer" finding).
- For H1, the primary test should be a **permutation test on architectural distance matrices** — encode each architecture as a feature vector (depth, width, heads, activation, attention variant...) and test whether cross-track distances exceed within-track distances. This uses all the data and doesn't require large n.
- **Do NOT compare absolute val_bpb across tracks.** Different domains have different intrinsic entropy. Compare **relative improvement from baseline** and **architectural feature vectors**.
- Apply Holm-Bonferroni correction for the family of 12+ tests across 4 hypotheses × 3 track comparisons.

**Effort:** Statistical analysis plan adds ~1 day to analysis phase; additional runs add ~$36 at us-east-1 spot

### 6. Reframe the Paper: Empirical Study, Not Methods Paper

**Problem:** The "Recursive Self-Refinement" framing claims methodological novelty that doesn't survive scrutiny against EvoPrompting (NeurIPS 2023), FunSearch (Nature 2024), LM-Searcher (EMNLP 2025), and OpenELM (2024). These all combine LLMs with iterative architecture/program search. Claiming RSR as a new paradigm will antagonize NAS-savvy reviewers.

**Fix:**
- **Change the title.** Drop "Recursive Self-Refinement" from the title. Use a concrete, descriptive title:
  > *"What Do Language Agents Discover About Molecular Transformer Architectures? An Empirical Study via Autonomous Code-Level Search"*

  or the more concise:
  > *"Autonomous Discovery of Domain-Specific Transformer Designs for Molecular Sequences"*

- **Keep RSR as an internal concept** (Section 2: "We formalize this process as Recursive Self-Refinement (RSR)...") for citation value, but don't lead with it.
- **Frame the contribution as empirical/scientific:** What does autonomous architecture search reveal about how molecular data shapes transformer design? The control condition (NLP track) transforms this from "we applied X to Y" into "we reveal systematic domain differences via controlled autonomous experimentation."
- **Cite aggressively:** Add EvoPrompting, FunSearch, LM-Searcher, LLMatic, OpenELM, GENIUS, and The AI Scientist to related work. Position on a spectrum from constrained NAS → LLM-guided NAS → fully autonomous code-level search. Don't claim to be different in kind — claim to be the first to apply this paradigm to molecular data and analyze what it reveals.

**Novelty claim adjustment:**

| Original Claim | Revised Claim |
|----------------|---------------|
| "First systematic study of autonomous AI-driven architecture search for molecular language models" | Keep — confirmed empty intersection |
| "Recursive Self-Refinement is a novel paradigm" | Downgrade to "RSR is a useful framing for iterative agent-artifact refinement" |
| "Fundamentally different from standard NAS" | Replace with "Extends LLM-guided NAS to unbounded code-level search in a new domain" |
| "'Recursive Self-Refinement' has zero prior use" | Keep — confirmed. But cite Self-Refine (Madaan 2023) as conceptual ancestor |

### 7. Validate the 5-Minute Training Proxy

**Problem:** The entire experiment assumes that 5-minute val_bpb is a reliable proxy for architectural quality. This is likely partially false — short training favors architectures that converge quickly (good initialization sensitivity, simple loss landscapes), not necessarily architectures that learn the best representations. The proxy is the foundation of the entire experimental design and it is unvalidated.

**Fix:**
- Run a **calibration study** before the main experiment: train 15-20 architectures (random variations of depth, width, heads, activation) for both 5 minutes and 2 hours on SMILES data. Measure **Spearman rank correlation** between 5-min val_bpb and 2-hr val_bpb.
- If correlation > 0.7: the proxy is valid, proceed with confidence.
- If correlation 0.4-0.7: proxy is noisy but usable. Acknowledge limitation, increase n.
- If correlation < 0.4: the proxy is broken. Increase training budget to 15-30 minutes per experiment (reduces experiments per run from ~100 to ~30-60, still viable).
- Report the calibration result in the paper (Section 3.1 or supplementary). This directly addresses the most damaging reviewer objection.

**Effort:** ~$15 additional compute for calibration; ~2 hours analysis

---

## Revised Experimental Design

### Design Overview

| Component | Original | Revised | Rationale |
|-----------|----------|---------|-----------|
| Dataset (SMILES) | ZINC-250K (12.5M tokens) | ZINC-250K + 5x SMILES enumeration (~62.5M tokens) | Avoid overfitting |
| Dataset (Protein) | UniRef50 subset (unspecified) | 50K sequences <500 residues (~12.5M tokens) | Match SMILES token count |
| Model size | 50.3M params (default) | 6-12M params (DEPTH=6, dim=288-320) | 3-8 epochs in 5 min |
| Attention | Flash Attention 3 | SDPA (Flash Attention 2 backend) | A10G compatibility |
| Agent runs | 3 tracks × 3 runs = 9 | SMILES: 5, Protein: 3, NLP: 5 = 13 | Statistical power |
| Baselines | None | Random NAS (9), Fixed default (3), HP-only (9) = 21 | Essential controls |
| Calibration | None | 20 architectures × 5min + 2hr = 40 runs | Proxy validation |
| Evaluation | val_bpb only | val_bpb + MoleculeNet (BBBP, HIV, BACE) + generation metrics | Molecular relevance |
| Analysis | Frequentist | Bayesian hierarchical + permutation tests + ROPE | Honest uncertainty |
| AWS region | ap-northeast-1 (Tokyo) | us-east-1 (Virginia) | 37% cost savings |

### Revised Budget

| Component | Runs | GPU Hours | Rate (us-east-1 spot) | API Cost | Total |
|-----------|------|-----------|----------------------|----------|-------|
| **Calibration study** | 40 | 7 | $0.44/hr | $3 | **$6** |
| **Agent runs (SMILES n=5)** | 5 | 65 | $0.44/hr | $8 | **$37** |
| **Agent runs (Protein n=3)** | 3 | 39 | $0.44/hr | $5 | **$22** |
| **Agent runs (NLP n=5)** | 5 | 65 | $0.44/hr | $8 | **$37** |
| **Random NAS baseline** | 9 | 117 | $0.44/hr | $0 | **$51** |
| **Fixed default + HP-only** | 12 | 156 | $0.44/hr | $18 | **$87** |
| **MoleculeNet fine-tuning** | 9 | 5 | $0.44/hr | $0 | **$2** |
| **Contingency (15%)** | — | — | — | — | **$36** |
| **Total** | **~83 runs** | **~454 hrs** | | | **~$278** |

This is 3.9x the original $72 estimate but delivers a paper that can survive NeurIPS review. The marginal $206 is trivial compared to researcher time.

### Revised Hypotheses

| ID | Original | Revised | Rationale |
|----|----------|---------|-----------|
| H1 | "Optimal architectures will differ" | "Agent-discovered architectures will cluster by domain when measured via architectural feature vectors" | Testable with permutation test; doesn't require val_bpb comparison |
| H2 | "Agent will rediscover known tricks" | "Agent modifications will correlate with known molecular modeling principles in a post-hoc alignment analysis" | Softer but more honest; agent can't modify prepare.py where most tricks live |
| H3 | "SMILES→protein partial transfer" | Split into 4 sub-hypotheses: H3a (asymmetric transfer), H3b (early layers transfer better), H3c (length > grammar as barrier), H3d (30-40% of innovations are universal) | Specific, falsifiable claims; publishable regardless of outcome |
| H4 | "Faster than random NAS" | "Agent achieves lower AUC-OC (area under optimization curve) than random NAS baseline with 95% bootstrap CI" | Formally specified; but H4 alone is too weak to anchor the paper |

---

## Revised Timeline

| Week | Tasks | Checkpoint |
|------|-------|------------|
| **1** (Mar 9-16) | Fork autoresearch, fix FA3→SDPA, implement SMILES `prepare.py` with enumeration, implement protein `prepare.py`, scale down model. Run calibration study (20 archs × 5min + 2hr). | **Checkpoint 1:** Molecular pipeline working, baseline val_bpb improving, proxy correlation > 0.5 |
| **2** (Mar 16-23) | Run all 13 agent sessions (3 tracks, n=5/3/5) overnight on spot instances. Start random NAS baselines in parallel. Start HP-only baselines. | **Checkpoint 2:** Evidence of architectural (not just HP) changes in at least 1 track |
| **3** (Mar 23-30) | Complete all baselines. Begin analysis: categorize agent modifications, build architectural feature vectors, run permutation tests. | **Checkpoint 3:** At least one qualitative arch difference between molecular and NLP tracks visible |
| **4** (Mar 30 - Apr 6) | Transfer experiments (cross-evaluate best architectures). MoleculeNet fine-tuning. Layer freezing experiments for H3. | **Checkpoint 4:** Transfer matrix computed; MoleculeNet results available |
| **5-6** (Apr 6-20) | Statistical analysis (Bayesian models, ROPE, bootstrap CIs). Attention pattern analysis. Write paper. Prepare figures: architecture evolution plots, transfer heatmaps, val_bpb curves. | |
| **7** (Apr 20-27) | Finalize paper. Internal review. Post **arXiv preprint** (establishes priority). | **Preprint posted by Apr 27** |
| **8** (Apr 27 - May 15) | Revise based on feedback. Submit NeurIPS 2026. | **NeurIPS submission by May 15** |

### Kill Criteria

| Checkpoint | Kill Condition | Pivot |
|------------|---------------|-------|
| Week 1 | FA3 fix fails; model doesn't train on molecular data | Abandon or switch to p4d.24xlarge |
| Week 2 | Agent makes zero architectural changes (only HPs) across all 5 SMILES runs | Reframe as "LLM-Guided HP Optimization" study; target workshop |
| Week 2 | Proxy calibration correlation < 0.4 | Increase training budget to 15-30 min; reduce to ~30-50 experiments per run |
| Week 3 | No visible architectural difference between molecular and NLP tracks | Pivot to "universal transformer" framing (still publishable) |
| Week 4 | MoleculeNet ROC-AUC shows zero correlation with val_bpb ranking | Report as limitation; downweight downstream claims |

---

## Competition Strategy

### Threat Assessment

| Threat | Severity | Timeline | Mitigation |
|--------|----------|----------|------------|
| AI Twitter blog posts/tweets | HIGH | Days | Not papers — but erode "first" claim. Move fast. |
| autoresearch-robotics (already exists) | MEDIUM | Active now | Cannot claim "first domain extension." Claim "first molecular/biological." |
| Independent researchers (arXiv preprint) | HIGH | 3-5 weeks | Post arXiv preprint by late April regardless of result quality. |
| Academic ML labs (Hutter's group, etc.) | MEDIUM | 2-4 months | They'll do it more rigorously but slower. Speed is our advantage. |
| Molecular ML labs (Barzilay, etc.) | LOW | 3-6 months | Different research culture; likely won't use autoresearch directly. |

### Priority Actions

1. **Start experiments today.** Every day of delay erodes first-mover advantage.
2. **Post arXiv preprint by April 27** — even with preliminary results, this establishes priority for the molecular autoresearch intersection.
3. **Monitor autoresearch forks weekly** — check for molecular/bio forks among the rapidly growing fork count (~1,300 and accelerating).
4. **Do NOT wait for perfect results** to post a preprint. A clear methodology paper with initial results beats a polished paper that arrives second.

---

## Paper Structure Recommendation

### Recommended Title
> **"Autonomous Discovery of Domain-Specific Transformer Designs for Molecular Sequences"**

### Recommended Structure

1. **Introduction** — Frame as empirical study: what does autonomous architecture search reveal about molecular transformer design?
2. **Background & Related Work** — Position on spectrum: classical NAS → LLM-guided NAS (EvoPrompting, FunSearch, LM-Searcher) → autonomous code-level search (autoresearch, IMPROVE) → our domain application. Cite 15+ related works.
3. **Recursive Self-Refinement Framework** — Define RSR as internal concept. Describe the agent loop. Distinguish from RSI.
4. **Experimental Design** — 3 tracks, baselines, proxy validation, statistical analysis plan.
5. **Results:**
   - 5.1 Proxy Validation (calibration study results)
   - 5.2 Architecture Search Trajectories (what did the agent change? HP vs. architectural modifications?)
   - 5.3 Domain-Specific Architectures (H1: permutation test on architectural feature vectors)
   - 5.4 Rediscovery of Domain Knowledge (H2: alignment with known molecular modeling principles)
   - 5.5 Cross-Domain Transfer (H3a-d: transfer matrix, layer freezing, attention analysis)
   - 5.6 Search Efficiency (H4: agent vs. baselines AUC-OC)
   - 5.7 Downstream Validation (MoleculeNet ROC-AUC)
6. **Analysis of Agent Behavior** — The most interesting contribution: how does the agent explore architecture space for unfamiliar domains? What strategies does it employ?
7. **Discussion** — Universal vs. domain-specific architectures. Limitations (closed-source agent, small scale, proxy validity). Future work.
8. **Supplementary** — Full agent interaction logs, all code diffs, architecture evolution animations, complete statistical tables.

### Submission Strategy

- **Primary:** NeurIPS 2026 main conference (ML track, not CompBio track)
- **Fallback 1:** NeurIPS 2026 ML4Drug Discovery workshop (70-80% acceptance probability)
- **Fallback 2:** ICML 2027 (6 months more results)
- **Fallback 3:** Nature Computational Science or Bioinformatics (different audience, different bar)

---

## Missing Citations to Add

### Must-Cite (failure to cite would be noticed by reviewers)

| Paper | Why |
|-------|-----|
| EvoPrompting (Chen et al., NeurIPS 2023) | LLM + evolutionary NAS with code generation — closest prior art |
| FunSearch (Romera-Paredes et al., Nature 2024) | LLM-guided iterative program search |
| LM-Searcher (Hu et al., EMNLP 2025) | Iterative, history-informed LLM-based NAS — newly discovered threat |
| Self-Refine (Madaan et al., NeurIPS 2023) | Established "self-refinement" concept in NLP |
| IMPROVE (Feb 2025) | LLM agent iterative ML pipeline refinement — already cited |
| The AI Scientist (Lu et al., ICML 2024) | End-to-end automated ML research |
| MoleculeNet (Wu et al., 2018) | Standard molecular benchmark suite |
| MoLFormer (Ross et al., Nature MI 2022) | Most relevant molecular transformer |

### Should-Cite

| Paper | Why |
|-------|-----|
| LLMatic (Nasir et al., 2023) | QD + LLM for NAS |
| OpenELM (Lehman et al., 2024) | Evolution through Large Models |
| GENIUS (2024) | LLM-based NAS |
| ChemBERTa / ChemBERTa-2 | Molecular transformer baselines |
| ESM-2 (Lin et al., 2023) | Protein language model — baseline for protein track |
| Uni-Mol (Zhou et al., ICLR 2023) | 3D molecular pretraining — represents where the field has moved |
| autoresearch-robotics (jellyheadandrew, Mar 2026) | Concurrent domain adaptation — acknowledge in footnote |

---

## Final Go/No-Go Decision Matrix

| Factor | Weight | Score (1-10) | Weighted |
|--------|--------|-------------|----------|
| Scientific interest of core question | 20% | 8 | 1.6 |
| Novelty (triple intersection confirmed empty) | 20% | 7 | 1.4 |
| Feasibility (with modifications) | 15% | 7 | 1.05 |
| Timeline risk (8 weeks to NeurIPS) | 15% | 5 | 0.75 |
| Competition risk | 10% | 4 | 0.4 |
| Downside protection (workshop fallback) | 10% | 8 | 0.8 |
| Cost | 10% | 9 | 0.9 |
| **Total** | **100%** | | **6.9/10** |

### **VERDICT: GO — with the 7 mandatory changes implemented.**

**Confidence: 65%** for NeurIPS main conference acceptance (up from ~30% without changes).
**Confidence: 85%** for workshop acceptance.
**Confidence: 95%** for a citable arXiv preprint that establishes priority.

The cost is bounded ($278), the downside is bounded (workshop paper at worst), and the upside is meaningful (first molecular autoresearch paper at a top venue, citable RSR concept, open-source framework). The single most important action right now: **fix FA3, implement SMILES prepare.py, and run the first baseline today.**

---

## Appendix: Source Documents

| Document | Lines | Key Contribution |
|----------|-------|-----------------|
| `stress-test-adversarial-reviews.md` | 335 | 4 reviewer perspectives; aggregate 3.3/10; identified 3 fatal critiques |
| `stress-test-experimental-design-audit.md` | 543 | Statistical power analysis; AWS pricing validation; baseline requirements |
| `stress-test-novelty-prior-art.md` | 358 | 5-role prior art search; confirmed empty triple intersection; LM-Searcher threat |
| `stress-test-technical-feasibility.md` | 948 | Autoresearch codebase analysis; prepare.py pseudocode; FA3 blocker; model sizing |
| `stress-test-paper-positioning.md` | 434 | NeurIPS strategy; framing alternatives; results prediction; competition analysis |
| `stress-test-transfer-hypothesis.md` | 489 | SMILES/protein distributional analysis; 6 transfer experiments; 4 sub-hypotheses |

---

*Synthesized March 9, 2026. This document integrates findings from all 6 stress-test panels and should be treated as the canonical project decision document. Reassess at each weekly checkpoint.*
