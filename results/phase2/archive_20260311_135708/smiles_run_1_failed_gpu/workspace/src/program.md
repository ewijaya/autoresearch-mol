# recursive-mol autoresearch program

This program drives autonomous architecture search for short-sequence next-token prediction.

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

Do not limit yourself to hyperparameters. Architectural changes are explicitly encouraged.

High-value areas to explore:

- depth / width balance
- attention layout and windowing patterns
- head structure, KV structure, and projection choices
- activation functions and MLP structure
- normalization strategy
- residual pathways and gating
- embedding / unembedding structure
- optimizer grouping and scheduling when they interact with architecture

Pure hyperparameter tuning is allowed, but it should not dominate the search.

## Experiment loop

1. Establish a baseline by running the unmodified starting `train.py`.
2. Make one coherent change at a time.
3. Run training and capture the log.
4. Extract `val_bpb` and peak VRAM.
5. Record the result in `results.tsv`.
6. Keep improvements.
7. Revert regressions or crashes.
8. Continue indefinitely until interrupted.

## Logging

Use `results.tsv` with columns:

`commit	val_bpb	memory_gb	status	description`

Status values:

- `keep`
- `discard`
- `crash`

Descriptions should say what changed, with emphasis on architectural intent when applicable.
