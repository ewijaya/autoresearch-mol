# recursive-mol

Autonomous discovery of domain-specific transformer architectures for molecular sequences.

An AI agent iteratively modifies transformer architecture code, trains for 5 minutes, measures validation bits-per-byte (val_bpb), and keeps or discards each change. The core question: **do molecular sequences (SMILES, proteins) induce fundamentally different optimal transformer architectures than natural language?**


## Project Structure

```
src/
  train.py              # Model, optimizer, training loop (agent modifies this)
  program.md            # Agent instructions for architecture search
  calibration.py        # Validates that 5-min runs predict 2-hr rankings
  prepare.py            # NLP track: FineWeb data + BPE tokenizer (do not modify)
  prepare_smiles.py     # SMILES track: ZINC-250K + RDKit enumeration (do not modify)
  prepare_protein.py    # Protein track: UniRef50 subset (do not modify)
  prepare_char.py       # Shared character-level tokenizer utilities
  pyproject.toml        # Dependencies
docs/
  PRD-recursive-mol.md  # Full project requirements
  phase-prompts.md      # Phase-by-phase agent prompts
  pitch.md              # One-pager for stakeholders
data/
  smiles/               # ZINC-250K processed data
  protein/              # UniRef50 processed data
results/
  calibration/          # Calibration study outputs
autoresearch/           # Reference training code
```

## Tracks

| Track | Dataset | Tokenizer | MAX_SEQ_LEN | Source |
|-------|---------|-----------|-------------|--------|
| SMILES | ZINC-250K (250K molecules, 5x enumeration) | Character-level (~45 chars) | 256 | `prepare_smiles.py` |
| Protein | UniRef50 (50K sequences, length 50-500) | Character-level (20 amino acids) | 512 | `prepare_protein.py` |
| NLP (control) | FineWeb | BPE (50K vocab) | 2048 | `prepare.py` |

Select a track with `RECURSIVE_MOL_TRACK={smiles,protein,nlp}`.

## Quick Start

**Requirements:** Single NVIDIA GPU (tested on A10G 24GB), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
cd src && uv sync

# Download data and prepare tokenizer
uv run prepare_smiles.py   # SMILES track
uv run prepare_protein.py  # Protein track
uv run prepare.py          # NLP track

# Run a single 5-minute training experiment
RECURSIVE_MOL_TRACK=smiles uv run train.py
```

## Key Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `RECURSIVE_MOL_TRACK` | `nlp` | Track selection: `smiles`, `protein`, `nlp` |
| `RECURSIVE_MOL_TIME_BUDGET` | `300` | Training wall-clock budget in seconds |
| `RECURSIVE_MOL_DEPTH` | `6` | Transformer depth (number of layers) |
| `RECURSIVE_MOL_MODEL_DIM` | - | Model dimension (overrides aspect ratio) |
| `RECURSIVE_MOL_NUM_HEADS` | - | Number of attention heads |
| `RECURSIVE_MOL_HEAD_DIM` | `64` | Dimension per attention head |
| `RECURSIVE_MOL_ACTIVATION` | `ReluSquared` | Activation function: `ReLU`, `GELU`, `SiLU`, `ReluSquared` |
| `RECURSIVE_MOL_ATTENTION` | `full` | Attention variant: `full`, `windowed` |
| `WANDB_PROJECT` | `recursive-mol` | W&B project name |
| `WANDB_DISABLED` | `false` | Set to `1` to disable W&B logging |

## How Training Works

1. The training loop runs for a fixed **wall-clock time budget** (default 5 minutes)
2. The learning rate schedule (warmup, constant, cooldown) is tied to `progress = elapsed_time / time_budget`
3. At the end, `val_bpb` is computed on a held-out validation set
4. Metrics are logged to W&B every 10 steps

The time-budget approach ensures fair comparison across architectures: a small model processes more tokens in 5 minutes than a large model, but both get equal compute.

## Calibration

Before running the full agent search, the calibration study validates that 5-minute proxy runs reliably rank architectures the same way as 2-hour runs:

```bash
cd src && uv run calibration.py
```

This trains 20 random architectures at both 300s and 7200s budgets, computes Spearman rank correlation on val_bpb, and outputs a go/no-go decision:
- rho > 0.7: proceed with 5-min proxy
- rho 0.4-0.7: proceed with caution
- rho < 0.4: increase proxy budget

## Autonomous Agent Mode

Point your AI coding agent at `src/program.md`:

```
Read program.md and start experimenting.
```

The agent will:
1. Establish a baseline val_bpb
2. Make one architectural change to `train.py`
3. Train for 5 minutes and measure val_bpb
4. Keep improvements, revert regressions
5. Repeat indefinitely

## W&B Metrics

**Per-step (every 10 steps):**
- `train/loss` — EMA-smoothed training loss
- `train/mfu` — model FLOPs utilization (%)
- `train/tok_per_sec` — throughput

**Final (end of run):**
- `val/bpb` — validation bits-per-byte (primary metric, lower is better)
- `val/peak_vram_mb` — peak GPU memory usage
- `val/mfu_percent` — steady-state MFU over the full run

## Project Phases

| Phase | Timeline | Description |
|-------|----------|-------------|
| 1. Infrastructure | Mar 9-16 | Set up training pipeline, implement SMILES/protein tracks, calibration |
| 2. Agent Search | Mar 16-Apr 6 | Run autonomous agents on all 3 tracks (5 SMILES + 3 protein + 5 NLP) |
| 3. Baselines | Apr 6-13 | Random NAS, fixed default, HP-only baselines |
| 4. Analysis | Apr 13-20 | Transfer matrix, MoleculeNet validation, statistical tests |
| 5. Paper | Apr 20-May 15 | Write and submit to NeurIPS 2026 |

## License

MIT
