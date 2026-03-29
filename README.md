# autoresearch-mol

Autonomous discovery of domain-specific transformer architectures for molecular sequences.

## What This Project Does

An AI coding agent autonomously redesigns transformer model architecture for molecular data. The agent modifies model code, trains for 5 minutes, evaluates performance, keeps improvements, discards regressions, and repeats — running ~100 experiments unattended.

We run this process across three domains:

- **SMILES** — small-molecule drug-like compounds (ZINC-250K)
- **Protein** — amino acid sequences (UniRef50)
- **NLP** — natural language text as a control (FineWeb)

## Questions We Are Answering

1. **Can an AI agent self-improve transformer architecture for molecular sequences?**
   The agent starts from a generic transformer and iteratively discovers architectural improvements (attention patterns, layer structure, activation functions, etc.) that lower validation loss on molecular data.

2. **Do molecular sequences demand fundamentally different architectures than natural language?**
   By running the same agent-driven search on SMILES, protein, and NLP tracks, we compare what the agent converges to in each domain. If the discovered architectures diverge, molecular data has distinct structural requirements that standard NLP-derived models miss.

3. **Do the discovered improvements transfer to real-world molecular tasks?**
   We validate the top architectures on downstream drug-discovery benchmarks (MoleculeNet: BBBP, HIV, BACE) to confirm that better sequence modeling translates to better molecular property prediction.

## How It Works

1. **Calibration** — Verify that short (5-minute) training runs reliably rank architectures the same way as long (2-hour) runs, so the agent can explore cheaply.
2. **Agent search** — The agent runs ~100 iterations per track, each time modifying the model architecture, training, and evaluating. Only improvements are kept.
3. **Baselines** — Random architecture search and hyperparameter-only tuning baselines establish whether the agent adds value beyond brute-force exploration.
4. **Analysis** — Cross-domain transfer experiments, downstream task validation, and statistical tests quantify what the agent discovered and whether it is domain-specific.

## Quick Start

**Requirements:** Single NVIDIA GPU (tested on A10G 24GB), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
cd src && uv sync

# Prepare data
uv run prepare_smiles.py   # SMILES track
uv run prepare_protein.py  # Protein track
uv run prepare.py          # NLP track

# Run a single training experiment
RECURSIVE_MOL_TRACK=smiles uv run train.py
```

## Project Phases

| Phase | Timeline | Status |
|-------|----------|--------|
| 1. Infrastructure | Mar 9–16 | ✅ Complete |
| 2. Agent Search | Mar 16–28 | ✅ Complete (3,106 experiments) |
| 3. Baselines | Integrated with Phase 2 | ✅ Complete |
| 4. Analysis | Mar 28 | ✅ Complete (H1–H4 hypothesis tests) |
| 5. Paper | Mar 30–May 15 | 🔄 In progress (NeurIPS 2026) |

## License

MIT
