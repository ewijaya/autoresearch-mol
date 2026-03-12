# Contingency Narratives: If Agent Search Is Not Statistically Significant

Written on March 12, 2026, while Phase 2 is running (task 2/34 complete).

## Context

The original manuscript thesis is: **"LLM-guided architecture search discovers better domain-specific transformer architectures than baselines."** This document outlines alternative narratives if the agent does not achieve statistically significant improvement over baselines on val_bpb.

### Early warning signs (SMILES run_1, n=1 each)

| Method | Best val_bpb | Best experiment | Keep rate |
|--------|-------------|-----------------|-----------|
| Agent | 0.5918 | exp071 | 14/100 (14%) |
| Random NAS | 0.5906 | exp093 | 6/97 (6%) |

Random NAS is currently beating the agent on best score. This is only one run each (no statistical power), but it motivates preparing alternative framings now rather than after all 34 tasks finish.

---

## Story 1: Negative result — LLMs don't add value over random search

### Framing

If a capable AI agent with full reasoning ability cannot beat random sampling over the same architecture search space, that is a strong empirical claim worth publishing. It tells the community: **don't waste API credits on agent-driven NAS — random search is just as good.**

### Why it's publishable

- Negative results are undervalued but important for calibrating community expectations around AI-for-science.
- The 4-condition experimental design (agent, HP-only, random NAS, fixed default) with multiple replicates per condition is rigorous enough to support the claim.
- Timely: the AI-for-science hype cycle needs grounding data.

### What to emphasize

- Methodological rigor: calibrated proxy metric, statistical replication, controlled baselines.
- Honest analysis of *why* the agent fails — does it get stuck in local optima? Is the search space too smooth? Does incremental editing limit exploration?
- Cost analysis: agent API cost vs random NAS compute-only cost for equivalent results.

### Venue fit

NeurIPS datasets & benchmarks track, ICML position papers, or a workshop paper. Top venues increasingly accept well-executed negative results.

---

## Story 2: Sample efficiency — the agent finds good architectures faster

### Framing

Even if the agent's *best* result after 100 experiments is not better than random NAS's best, the agent may reach a competitive threshold much sooner.

### How to measure

Plot **best-so-far curves**: val_bpb (y-axis) vs experiment number (x-axis) for each method. If the agent's curve dominates in the first 10-30 experiments, the story becomes: **"The agent is more sample-efficient — it finds competitive architectures in fewer trials."**

### Example scenario

- Agent: reaches 0.593 by experiment 10, converges to 0.592 by experiment 40
- Random NAS: reaches 0.593 by experiment 60, lucky hit of 0.591 at experiment 93

Even though random NAS wins at experiment 100, the agent was better for the first 60 experiments. If you only have budget for 20 experiments (a realistic constraint), the agent dominates.

### Why it matters

Each experiment costs ~5 minutes of GPU time plus (for agent runs) API credits. If the agent reaches 95% of the best possible score in 10 experiments vs 100, that is a 10x compute saving. For expensive training regimes (longer time budgets, larger models), sample efficiency is the primary bottleneck.

### Additional metrics

- **Area Under the best-so-far Curve (AUC)**: Integrates performance over the full search trajectory. The agent could have better AUC even with worse final-point performance.
- **Time-to-threshold**: For a given target val_bpb (e.g., 1% improvement over baseline), how many experiments does each method need?
- **Anytime performance**: At experiment N (for N = 5, 10, 20, 50, 100), what is each method's best-so-far? Report as a table.

### Data availability

Can be computed from existing results.tsv files immediately — no additional experiments needed.

---

## Story 3: Molecular sequences don't need special architectures

### Framing

Flip the research question. If neither agent nor random NAS finds dramatically better architectures for SMILES/protein compared to NLP, the finding is: **standard transformer architectures are already near-optimal for molecular sequences.** The architecture search landscape is flat near the default configuration.

### Why it's interesting

The molecular ML community frequently proposes bespoke architectures (graph transformers, equivariant networks, domain-specific attention patterns). If a thorough search over architecture space yields marginal improvements, that undermines the motivation for custom architectures and suggests effort is better spent on data, scale, or pre-training objectives.

### Evidence needed

- Show that the variance in val_bpb across 100 random architectures is small (tight distribution around the default).
- Show this holds across all three tracks — not just SMILES.
- Compare the *spread* of random NAS results across tracks: if SMILES has the same spread as NLP, molecular data isn't special.

### Supporting analysis

- Histogram of val_bpb across all 100 random NAS experiments per track.
- Effect size of architecture changes: what fraction of the random configs improve over default?
- Sensitivity analysis: which architecture dimensions (depth, width, activation, attention) have the largest effect on val_bpb?

---

## Story 4: The benchmark and methodology contribution

### Framing

Frame the paper as introducing the **recursive self-improvement evaluation framework** itself, with the agent results as one data point rather than the main claim.

### Contributions

1. **A reproducible pipeline** for evaluating AI-guided architecture search across domains.
2. **Three standardized tracks** (SMILES, protein, NLP) with calibrated 5-minute proxy training (validated by Spearman correlation between 5-min and 2-hour runs).
3. **Four controlled baselines** with statistical replication (agent, HP-only, random NAS, fixed default).
4. **Open-source tooling** for others to plug in their own agents (GPT-5, Gemini, open-source models) and compare.
5. **Calibration study** demonstrating that short proxy runs preserve architecture rankings.

### Why it's publishable

The NeurIPS Datasets & Benchmarks track explicitly values evaluation frameworks. The molecular domain adds novelty — existing NAS benchmarks (NAS-Bench-101, NAS-Bench-201) focus on image classification. A molecular NAS benchmark fills a gap.

### What to emphasize

- Reproducibility: fixed seeds, deterministic configs, version-controlled train.py snapshots with diffs.
- Extensibility: the framework supports any agent that can edit Python files.
- The calibration study (Phase 1) as a methodological contribution in its own right.

---

## Story 5: Qualitative architecture analysis — what the agent discovers vs what random finds

### Framing

Even if quantitative results are similar, the *types* of architectures found may be fundamentally different. The agent may discover **coherent, interpretable design patterns** while random NAS finds **one lucky outlier with no generalizable insight.**

### Evidence to build

1. **Keep rate comparison**: Agent has 14% keep rate vs random's 6%. The agent's search is more directed — it proposes improvements more often, even if the single best point is similar.

2. **Architecture clustering**: Do the agent's kept experiments converge to a consistent design pattern (e.g., "shallow + wide + SwiGLU works best for SMILES")? Or are they scattered? Cluster the kept configs by their architecture parameters and visualize.

3. **Diff analysis**: The agent's diffs (stored in `diffs/`) show a coherent reasoning trajectory — each edit builds on the previous best. Random NAS has no trajectory; each config is independent. Visualize the agent's exploration path through architecture space.

4. **Pareto frontier (val_bpb vs memory_gb)**: The agent might find architectures that are better on the efficiency-accuracy tradeoff even if not on raw accuracy alone. Plot both methods' experiments on a Pareto chart.

5. **Interpretable discoveries**: Extract the agent's descriptions from results.tsv. Do they reveal domain-specific insights? For example:
   - "SwiGLU gating helps SMILES but not NLP" → domain-specific finding
   - "Alternating value embeddings are critical for short sequences" → generalizable insight

   Random NAS can never produce such insights.

### Why it matters

In practice, researchers don't just want a good architecture — they want to understand *why* it works. An agent that produces interpretable, building-block discoveries has value beyond the final val_bpb number, even if random search matches the endpoint.

---

## Story 6: Cross-domain transfer patterns

### Framing

Test whether architectures optimized for one domain transfer to others. This is interesting regardless of agent-vs-baseline performance.

### Transfer matrix design

Take the best architecture from each {method} x {track} combination and evaluate it on all three tracks:

|  | Eval: SMILES | Eval: Protein | Eval: NLP |
|--|-------------|---------------|-----------|
| Arch from SMILES | native | transfer | transfer |
| Arch from Protein | transfer | native | transfer |
| Arch from NLP | transfer | transfer | native |

### Possible findings

- **Domain specificity**: SMILES-optimal architectures degrade on NLP (and vice versa) → molecular sequences need different architectures, supporting the paper's original motivation even if the agent isn't the one finding them.
- **Universality**: All tracks prefer similar architectures → standard transformers generalize, architecture search is unnecessary for domain adaptation.
- **Asymmetric transfer**: NLP architectures transfer well to SMILES but not vice versa → molecular data is a subset of language patterns.

### Data requirements

Requires Phase 4 experiments. Plan for this early — reserve GPU time.

---

## Story 7: Decomposing the value of architecture vs hyperparameter tuning

### Framing

The 4-condition design enables a clean decomposition:

```
Agent (arch + HP)  vs  HP-only (HP only)     → value of architecture changes
Agent (arch + HP)  vs  Random NAS (arch only) → value of guided vs random search
HP-only            vs  Fixed default          → value of HP tuning alone
Random NAS         vs  Fixed default          → value of random architecture changes
```

### Possible findings

- **HP-only ≈ Agent**: Architecture changes don't matter much; it's all in the hyperparameters. Provocative claim that would generate discussion.
- **HP-only >> Fixed default but Agent ≈ HP-only**: Hyperparameter tuning is the primary lever; architecture is secondary.
- **Random NAS ≈ Agent >> HP-only**: Architecture matters, but you don't need intelligence to find good ones — random search suffices.
- **Agent >> Random NAS ≈ HP-only**: Both guided architecture search AND hyperparameter tuning contribute, but guided search adds more.

### Why it matters

The NAS community rarely controls for HP tuning vs architecture changes. This decomposition is a methodological contribution regardless of which cell "wins."

---

## Viability ranking

| Story | Publishability | Novelty | Requires additional data? | Risk |
|-------|---------------|---------|--------------------------|------|
| 2 (sample efficiency) | High | Medium | No — replot existing results | Low |
| 4 (benchmark contribution) | High | High | No | Low |
| 5 (qualitative analysis) | Medium-High | Medium | No — diffs/descriptions exist | Low |
| 3 (architectures don't matter) | Medium-High | High | Need cross-track comparison | Medium |
| 7 (HP decomposition) | Medium-High | Medium-High | Need HP-only runs to finish | Medium |
| 1 (negative result) | Medium | Medium | No | Low |
| 6 (transfer patterns) | Medium | High | Need Phase 4 | High |

---

## Recommended immediate actions

1. **Plot best-so-far curves now** for SMILES run_1 (agent vs random NAS). This takes minutes and immediately tells you if Story 2 is viable.

2. **Don't change the experimental plan.** All 34 tasks need to complete regardless of which story you tell. The 4-condition design supports every narrative above.

3. **Start collecting cost data.** Track API credits spent on agent sessions vs compute-only cost of random NAS. This supports both Story 1 (negative: agent is expensive for no gain) and Story 2 (positive: agent is cheaper per good architecture found).

4. **After 2-3 agent runs complete**, compute preliminary cross-run variance. If agent variance is high (some runs great, some bad), that's a different problem than consistently mediocre results.

5. **Archive this document** and revisit after task 10/34 (roughly when you'll have multi-run data for SMILES). Update the viability rankings based on actual data.

---

## Combinability

These stories are not mutually exclusive. The strongest paper would combine 2-3:

- **Primary claim**: Sample efficiency (Story 2) or benchmark contribution (Story 4)
- **Supporting analysis**: Qualitative architecture insights (Story 5) + HP decomposition (Story 7)
- **Discussion section**: Transfer patterns (Story 6) + implications for molecular ML (Story 3)

The experimental design already supports all of these. The question is which to lead with, and that depends on how the remaining 32 tasks turn out.
