# recursive-mol: The Big-Pharma Pitch

## The Problem They Already Feel

Every pharma company using AI for drug discovery is running transformers designed for English text. ChemBERTa, MolBERT, protein language models — they all copy GPT/BERT architectures and just swap the dataset. Nobody has systematically asked: *should a transformer that reads SMILES strings even look like a transformer that reads Wikipedia?*

## What This Project Does

We let an AI agent redesign the transformer architecture from scratch, specifically for molecular data. Not fine-tuning weights — rewriting the actual model code. The agent runs ~100 iterations, each time modifying the architecture (attention patterns, layer depth, activation functions, etc.), training for 5 minutes, measuring performance, and deciding what to try next. We do this for SMILES, proteins, and NLP as a control.

## Why a Pharma VP Should Care — Three Scenarios

1. **Architectures diverge** (our main hypothesis): The agent discovers that molecules want fundamentally different transformers — maybe shallower, wider, with local attention for bonded atoms. This means every molecular AI model your company runs today is leaving performance on the table by using NLP hand-me-downs. The discovered architectures become a direct asset.

2. **Architectures converge**: That's also valuable. It means your current approach of borrowing from NLP is actually fine, and you can stop worrying about architecture and focus R&D dollars elsewhere (data, objectives, training scale).

3. **The method itself**: An autonomous agent that can search architecture space for any new modality in ~$120 of compute. Next quarter you want a transformer for ADMET prediction, or for glycan sequences, or for reaction SMILES — you point this framework at it and let it run overnight instead of paying a team for months of manual experimentation.

## Concrete Deliverables They'd Care About

- A transfer matrix showing which architectural innovations are molecular-specific vs. universal
- MoleculeNet validation (BBBP, HIV, BACE) proving the discovered architectures actually improve drug-relevant tasks, not just perplexity
- An open-source framework they can fork and point at their proprietary chemical data

## The Cost/Speed Argument

The entire experimental budget is ~$200 in AWS compute over 6 weeks. Compare that to a typical pharma ML team spending months on architecture ablations. Even if the findings are incremental, the ROI on the method alone is absurd.

## The Timing Argument

Karpathy released the base framework March 7, 2026. Nobody has applied it to molecules yet. The first team to do this and publish sets the benchmark everyone else cites.

## Why This Is Strategic for StemRIM

StemRIM is a 1-person computational division (ISDD) supporting 7+ active ML projects — bioactivity prediction, generative drug design, peptide-enzyme interaction modeling, protein language models, and more — all feeding into StemRIM's Regeneration-Inducing Medicine pipeline (Redasemtide, TRIM3/4/5).

This project is strategic on four axes:

### 1. Direct pipeline acceleration

StemRIM's drug candidates are peptides. Every ISDD project — hemagglutination prediction, MSC proliferation modeling, PDGF-BB secretion prediction, peptide stability optimization — depends on neural networks that process molecular or protein sequences. If this project discovers that molecular data wants fundamentally different architectures than NLP, that finding immediately upgrades every model in the ISDD portfolio. Better architectures mean better bioactivity predictions, which means faster candidate selection for TRIM3/4/5.

### 2. Capability multiplier for a resource-constrained team

ISDD is one person doing the work that Astellas, Daiichi Sankyo, and Takeda assign to entire ML engineering teams. An autonomous agent that searches architecture space overnight — without human supervision — is the single highest-leverage tool a 1-person division can have. Instead of manually running ablation studies for weeks, you deploy the agent, go home, and review the results in the morning. This project proves that workflow works, and benchmarks it rigorously.

### 3. Competitive positioning and investor signal

StemRIM is pre-revenue (EPS: -¥34.42, market cap: ~¥6.2B). The stock has declined 66% from its 2022 peak. Investor confidence depends on demonstrating that StemRIM's R&D capabilities justify the burn rate. Publishing at a top ML venue (NeurIPS/ICML/ICLR) — from a 1-person division at a small Japanese biotech — sends an outsized signal: this company is doing frontier computational work, not just running off-the-shelf models. That matters for:

- **Partnership negotiations** (Shionogi, potential new licensees): demonstrates computational sophistication
- **Investor relations**: concrete, peer-reviewed evidence of AI-driven R&D capability
- **Recruiting**: attracts computational talent who want to publish, not just run pipelines

### 4. Generalizable framework across the pipeline

The methodology is modality-agnostic. Once validated on SMILES, proteins, and NLP, the same agent framework can be pointed at any new task:

- Glycan sequence modeling for the cell collection device (PJ4)
- Gene therapy target prediction for SR-GT1 (PJ5)
- ADMET property prediction for next-generation peptide optimization
- Any new modality StemRIM encounters as the pipeline expands

The $200 compute cost per search means ISDD can afford to run architecture optimization for every new project, not just the highest-priority ones.

### The internal elevator pitch

*"We built an AI system that automatically discovers optimal neural network architectures for molecular modeling. It costs $200 per run and works overnight. This lets our 1-person computational division deliver models that would normally require a dedicated ML engineering team — directly accelerating peptide discovery for Redasemtide, TRIM3, TRIM4, and TRIM5. The research also positions StemRIM as a credible AI-driven drug discovery company at a time when investor confidence depends on demonstrating exactly that."*

---

**The one-liner:** "You're running billion-dollar drug programs on transformer architectures nobody bothered to optimize for chemistry. We built an AI that does that optimization automatically for $200. Here's what it found."
