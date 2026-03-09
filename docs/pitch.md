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

---

**The one-liner:** "You're running billion-dollar drug programs on transformer architectures nobody bothered to optimize for chemistry. We built an AI that does that optimization automatically for $200. Here's what it found."
