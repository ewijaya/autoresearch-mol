# Phase 2 Experiment Note

Snapshot updated on March 11, 2026.

## Planned workload

Phase 2 is scheduled to run `34` total sessions:

- `13` agent sessions
- `9` random NAS baseline sessions
- `9` HP-only baseline sessions
- `3` fixed-default baseline sessions

### Agent sessions

- SMILES: `5` runs x `~100` experiments = `~500`
- Protein: `3` runs x `~100` experiments = `~300`
- NLP: `5` runs x `~100` experiments = `~500`

Planned agent total: `~1,300` experiments.

### Tier 1 baselines

- Random NAS: `9` runs x `100` experiments = `900`
- HP-only agent: `9` runs x `~100` experiments = `~900`
- Fixed default: `3` runs x `1` experiment = `3`

Planned baseline total: `~1,803` experiments.

## Expected total

Planned Phase 2 total: about `3,103` experiments.

Because the PRD uses "`~100`" for the agent and HP-only sessions, the exact final count may be slightly lower or higher than `3,103`, but this is the working expectation for the current queue.

## Execution model

Phase 2 is still running sequentially on a single GPU.

The queue now uses one short-lived Codex session per experiment row for agent-driven runs instead of one long-lived Codex session per 100-experiment run. This reduces failure risk from context exhaustion or stalled post-run analysis.

The runner also now verifies that the expected number of rows is actually recorded in `results.tsv` before it treats a task as complete.

## Current live status

At the time this note was written:

- the Phase 2 runner is on task `1/34`
- active task: SMILES agent `run_1`
- completed experiments in the active run: `6`
- current recorded rows:
  - `exp001`: baseline, `val_bpb = 0.596350`, `keep`
  - `exp002`: full attention for all layers at seq len 256, `val_bpb = 0.597899`, `discard`
  - `exp003`: disable alternating value embeddings, `val_bpb = 0.598981`, `discard`
  - `exp004`: increase default depth from 6 to 8 with matched aspect-ratio width, `val_bpb = 0.601616`, `discard`
  - `exp005`: reduce default depth from 6 to 5 for a faster smaller model, `val_bpb = 0.597927`, `discard`
  - `exp006`: replace dense FFN with parameter-matched SwiGLU gating, `val_bpb = 0.594814`, `keep`
- the runner has resumed with the patched per-experiment Codex flow
- the next experiment is `exp007`

## Where to check progress

- Runner state: `results/phase2/queue_state.json`
- Runner log: `results/phase2/runner.log`
- Active SMILES run log: `results/smiles/run_1/agent_session.log`
- Active SMILES results: `results/smiles/run_1/results.tsv`
