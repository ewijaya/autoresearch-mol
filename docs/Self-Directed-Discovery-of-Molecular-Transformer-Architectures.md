# Self-Directed Discovery of Molecular Transformer Architectures via Recursive Self-Refinement

> **Research paper proposal:** Recursive self-refinement framework for autonomous discovery of molecular transformer architectures. Built on Karpathy's autoresearch paradigm, extended to molecular sequence modeling.

---

## Central Hypothesis

> Through recursive self-refinement — where an autonomous AI agent iteratively modifies, evaluates, and refines neural architectures — molecular sequence modeling tasks will converge on architecturally distinct transformer designs compared to natural language. Unlike recursive self-improvement (where the agent itself becomes more capable), our framework keeps the agent fixed while the research artifact undergoes compounding refinement cycles, yielding domain-specific architectural innovations that reflect the structural properties of molecular data.

## Sub-Hypotheses (Testable)

1. **H1:** Optimal architectures for SMILES strings will differ from those for natural language (e.g., different depth/width ratios, attention head counts, activation functions)
2. **H2:** The agent will independently discover or approximate known molecular modeling tricks (e.g., augmentation via SMILES randomization, local attention for bonded atoms) without being told about them
3. **H3:** Architectures discovered on SMILES will partially transfer to protein sequences (shared sequential molecular grammar) but not fully (different alphabet size, structural constraints)
4. **H4:** The autonomous agent will find competitive architectures faster (fewer experiments) than random NAS or grid search baselines

---

## Method

### Framework

Fork Karpathy's autoresearch (github.com/karpathy/autoresearch) as the base recursive self-refinement engine. Replace FineWeb-Edu with molecular datasets. Keep the core loop: agent refines train.py → 5-min training → measure val_bpb → keep/discard → repeat. The agent remains fixed; only the research artifact (model architecture) undergoes compounding refinement cycles.

### Datasets (3 tracks)

- **Track A — SMILES:** ZINC-250K or ChEMBL subset (small molecules as SMILES strings). Metric: val_bpb on held-out molecules.
- **Track B — Protein sequences:** UniRef50 subset (amino acid sequences). Same metric.
- **Track C — Natural language (original FineWeb-Edu):** Control condition. Same agent, same budget, different data.

### Experimental Design

1. Run 3 independent agent sessions per track (to measure variance), each ~100 experiments (overnight run)
2. Compare final architectures across tracks: what did the agent change for molecules vs. text?
3. Cross-transfer test: take best SMILES architecture → evaluate on protein data (and vice versa)
4. Baseline comparisons: (a) random architecture search, (b) published molecular transformer architectures (MolBERT, ChemBERTa)

### AWS Infrastructure

- **Primary:** g5.xlarge (A10G, 24GB VRAM) — cost-effective for 5-min runs, ~$1/hr on-demand
- **Scaling test:** p4d.24xlarge (A100) for one track to compare hardware-dependent architecture emergence
- **Region:** ap-northeast-1 (existing fleet). Use spot instances for bulk runs.
- **Estimated cost:** 3 tracks × 3 runs × 8hrs × $1/hr ≈ $72 for core experiments

---

## Expected Contributions

1. First systematic study of autonomous AI-driven architecture search for molecular language models
2. Quantitative evidence for whether molecular data demands fundamentally different architectures than natural language
3. Open-source molecular autoresearch framework (fork + datasets + program.md for molecular research)
4. Analysis of AI agent research trajectories: how does an agent explore architecture space for an unfamiliar domain?

## Why This Is Publishable

- **Novel intersection:** No one has applied the autoresearch paradigm to molecular/biological data yet
- **Timely:** Karpathy released autoresearch March 2026; first domain-specific paper would be a strong fast-follow
- **Dual audience:** appeals to both ML (autonomous agents, NAS) and computational biology (molecular modeling) communities
- **Win either way:** if architectures differ → interesting finding about domain specificity. If they converge → evidence for universal transformer architectures.

## Target Venues

- **Tier 1:** NeurIPS 2026 (deadline May), ICML 2027
- **Domain:** Nature Computational Science, Bioinformatics (Oxford)
- **Workshop:** NeurIPS ML4Drug Discovery, ICML CompBio
- **Preprint:** arXiv cs.LG + q-bio.QM (dual listing)

---

## Modifications to prepare.py

The original autoresearch locks prepare.py as read-only. For molecular data, we need to modify it to:
- Replace FineWeb-Edu download with ZINC/ChEMBL/UniRef50 data pipeline
- Train a domain-appropriate BPE tokenizer (or use character-level for SMILES)
- Keep val_bpb as the metric (it is vocab-size-independent by design, so cross-domain comparison is fair)

This is a one-time fork modification. The agent still only touches train.py during experiments.

---

## Proposed Timeline

- **Week 1-2:** Fork repo, set up molecular datasets in prepare.py, validate baseline runs on AWS
- **Week 3-4:** Run all 9 agent sessions (3 tracks × 3 runs), collect results
- **Week 5-6:** Analysis — architecture diff, transfer experiments, comparison with baselines
- **Week 7-8:** Write paper, prepare figures (architecture evolution plots, val_bpb curves per track)

## Key Risks & Mitigations

- **Risk:** Agent gets stuck / no improvement on molecular data → **Mitigation:** iterate on program.md with domain hints (this itself becomes a finding)
- **Risk:** 5-min budget too short for meaningful molecular model training → **Mitigation:** adjust dataset size to ensure convergence signal within budget
- **Risk:** Architectures converge to same solution across all tracks → **Mitigation:** this is still a publishable negative result ("universal transformers")

---

## Conceptual Framing: Recursive Self-Refinement

### Why Not "AutoResearch" or "NAS"?

"AutoResearch" is Karpathy's project name, not a generalizable concept. Standard Neural Architecture Search (NAS) uses random sampling, evolutionary strategies, or RL controllers. Our system is fundamentally different: a reasoning agent with domain understanding makes deliberate, hypothesis-driven modifications. It reads code, understands what it changed, and decides what to try next based on accumulated experimental history. This is closer to how a human researcher works than how NAS works.

### Why "Recursive Self-Refinement"?

The system has a genuinely recursive structure: the output of experiment N (the improved train.py) becomes the starting point for experiment N+1. Each cycle builds on accumulated improvements. The architecture is refining itself through the agent as proxy.

We deliberately choose "self-refinement" over the loaded term "self-improvement" (Bostrom, Good, Yudkowsky). The distinction is precise and important:
- **"Self-improvement"** implies the agent gets smarter each cycle (it does not — the LLM is frozen)
- **"Self-refinement"** implies the artifact gets better through iterative cycles (it does — the architecture improves)
- This framing is technically accurate, avoids AI safety alarm bells, and claims a distinct conceptual space

### Proposed Paper Introduction

> "We introduce a recursive self-refinement framework where an autonomous AI agent iteratively modifies, evaluates, and refines neural architectures for molecular language modeling. Unlike recursive self-improvement — where the agent itself becomes more capable — our approach keeps the agent fixed while the research artifact (model architecture) undergoes compounding refinement cycles. We apply this framework to molecular sequence data (SMILES, protein sequences) and demonstrate that the architectures discovered through self-refinement differ systematically from those found for natural language, suggesting that molecular data demands fundamentally different transformer designs."

### Project Identity

- **Paper title:** "Self-Directed Discovery of Molecular Transformer Architectures via Recursive Self-Refinement"
- **Core concept:** Recursive Self-Refinement (RSR)
- **GitHub repo:** recursive-mol (github.com/ewijaya/recursive-mol)
- **Track naming:** RSR-Mol (SMILES), RSR-Prot (protein), RSR-NLP (language baseline)

### Strategic Positioning

1. "Recursive" captures the compounding, iterative energy that distinguishes this from one-shot NAS
2. Explicitly contrasting with RSI shows mastery of the literature and disarms safety-minded reviewers
3. "Self-Directed" in the title emphasizes agent autonomy — this is not random search, it is deliberate
4. Claiming "Recursive Self-Refinement" as our term gives us a citable concept others can build on

---

## Novelty Validation (Stress Test — March 9, 2026)

> **Clear air confirmed:** The intersection of {autonomous LLM coding agent} x {architecture discovery} x {molecular data} is completely empty. The term "recursive self-refinement" has zero prior use in ML literature.

### Key Findings

- "recursive self-refinement" + neural network + architecture → 0 results across all search engines
- No domain-specific forks of Karpathy autoresearch exist (only macOS/MLX ports)
- No prior work uses an LLM coding agent to iteratively discover transformer architectures for molecular sequence data
- Autoresearch itself was released March 7, 2026 — we would be the first domain-specific extension

### Related Work (Must Cite)

1. **GNN Architecture Search for Molecular Property Prediction** (Jiang et al., 2020) — Standard NAS (evolutionary/RL) on GNNs for molecular property prediction. Different: they search graph architectures via predefined cell spaces; we search transformer architectures via unbounded code edits on SMILES sequences.

2. **IMPROVE: Iterative Model Pipeline Refinement and Optimization** (Feb 2025) — Closest conceptual neighbor. LLM agents iteratively refine ML pipelines for image classification. Different: pipeline/hyperparameter tuning, not architecture discovery; not molecular domain. Must cite as related work.

3. **Self-Refine** (Madaan et al., 2023) — LLM iterative self-refinement for text outputs (code, reasoning). The term "self-refine" exists in NLP. Different: refines LLM outputs, not neural network architectures. Cite to differentiate our use of "self-refinement".

4. **A Self-Improving Coding Agent** (Apr 2025) — Agent that improves its own code, closer to actual RSI. Different: improves the agent itself, not a separate trained model.

5. **GPT-NAS** (IEEE, 2024) — Uses GPT to guide neural architecture search. Different: vision tasks only; LLM as search controller selecting from predefined cells, not as autonomous code-editing researcher.

6. **ChemCrow / DrugChat / Llamole** — LLM agents using chemistry tools for molecule generation and property prediction. Different paradigm entirely: they use LLMs to design molecules, we use LLMs to design the architectures that model molecules.

### Anticipated Reviewer Objections

**Objection 1: "Why not just use standard NAS (DARTS, ENAS)?"**

> Standard NAS searches a predefined cell space (choice of operations, connection patterns). Our agent makes arbitrary code changes — new activation functions, custom attention patterns, optimizer modifications, training loop restructuring — that are not expressible in any predefined search space. The search space is unbounded, constrained only by what compiles and runs. This is a fundamentally different and more expressive form of architecture search.

**Objection 2: "5-minute training budget is too short for meaningful molecular model training."**

> Mitigation: Use character-level tokenization on SMILES (small vocabulary, fast convergence). ZINC-250K has ~250K molecules; at character level this is a manageable corpus. The fixed time budget is a feature, not a bug — it forces the agent to find architectures that learn efficiently, which is the real-world constraint. We can also calibrate dataset size to ensure convergence signal within budget.

**Objection 3: "The agent (Claude/Codex) is non-deterministic and closed-source. How is this reproducible?"**

> Mitigation: (a) Run 3 independent replicates per condition and report mean + variance. (b) Log every agent prompt and response for full transparency. (c) The discovered architectures (train.py) are fully open and deterministic — anyone can retrain them without the agent. (d) Test with multiple agent backends (Claude, Codex, open-source) to measure agent-dependence.

**Objection 4: "This is just automated hyperparameter tuning, not architecture search."**

> Mitigation: Explicitly track and categorize changes in results.tsv as architectural (new layers, attention patterns, model structure) vs. hyperparameter (learning rate, batch size). Show that the agent makes both types and that architectural changes drive the majority of improvement. Include code diffs as supplementary material.

---

*Drafted by Rex · March 9, 2026*
