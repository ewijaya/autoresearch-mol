# recursive-mol autoresearch program (HP-only)

This program drives autonomous hyperparameter search for short-sequence next-token prediction.

## Scope

Read these files before starting:

- `README.md`
- `prepare.py`, `prepare_smiles.py`, and `prepare_protein.py` to understand the fixed data/eval harness for each track
- `train.py`, which is the only file you are allowed to modify during search

The active track is selected externally with `RECURSIVE_MOL_TRACK={smiles,protein,nlp}`.

## Fixed rules

- Only edit `train.py`.
- Do not modify any `prepare*.py` file.
- Do not change the evaluation metric or the dataset split.
- Only modify hyperparameters: learning rate, batch size, dropout, weight decay, warmup steps, optimizer params.
- Do NOT change model architecture: no new layers, no attention pattern changes, no activation function changes, no model structure changes.
- Keep the fixed loop structure:
  1. edit `train.py`
  2. run a 5-minute training job
  3. read `val_bpb`
  4. keep or discard the change

## What to optimize

Primary metric: lowest `val_bpb`.

Secondary constraints:

- Runs must finish cleanly within the fixed time budget.
- Avoid dramatic VRAM regressions.
- Prefer changes with a clear accuracy-to-complexity tradeoff.

## Search guidance

Architectural edits are forbidden in this program.

High-value areas to explore:

- learning rate scales and ratios
- batch size and gradient accumulation
- weight decay
- optimizer betas and momentum settings
- warmup steps and warmup / warmdown scheduling
- scalar optimizer settings that do not alter model structure

## Experiment loop

1. Establish a baseline by running the unmodified starting `train.py`.
2. Make one coherent hyperparameter change at a time.
3. Run training and capture the log.
4. Extract `val_bpb` and peak VRAM.
5. Record the result in `results.tsv`.
6. Keep improvements.
7. Revert regressions or crashes.
8. Continue until interrupted or the experiment budget is exhausted.

## Logging

Use `results.tsv` with columns:

`commit	val_bpb	memory_gb	status	description`

Status values:

- `keep`
- `discard`
- `crash`

Descriptions should say what changed, with emphasis on the hyperparameter intent.
