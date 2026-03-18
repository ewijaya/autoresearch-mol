# Phase 2 Experiment Note

Snapshot updated on March 18, 2026.

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

The runner is also quota-aware for ChatGPT Plus usage. If Codex hits a usage limit, the queue records a `paused_rate_limit` state in `results/phase2/queue_state.json`, keeps any script-only tasks available for execution, and can later resume from the recorded row counts in each run's `results.tsv`.

If the weekly Codex limit hits `0%`, follow `docs/phase2-weekly-limit-playbook.md` for the stop-and-resume procedure.

## Completed tasks (13/34)

| # | Kind | Track | Run | Experiments | Best val_bpb | Notes |
|---|------|-------|-----|-------------|-------------|-------|
| 1 | agent | SMILES | run_1 | 100 | 0.5918 | |
| 2 | agent | SMILES | run_2 | 100 | 0.5808 | Overall best SMILES |
| 3 | agent | SMILES | run_3 | 100 | 0.5839 | |
| 4 | agent | SMILES | run_4 | 100 | 0.5892 | |
| 5 | random_nas | SMILES | run_1 | 100 | 0.5906 | |
| 6 | random_nas | SMILES | run_2 | 100 | 0.5923 | |
| 7 | random_nas | SMILES | run_3 | 100 | 0.5914 | |
| 8 | agent | SMILES | run_5 | 100 | 0.5834 | |
| 9 | random_nas | protein | run_1 | 103 | 3.9719 | Slightly over 100 |
| 10 | random_nas | protein | run_2 | 100 | 3.9710 | |
| 11 | agent | protein | run_1 | 100 | 3.9656 | Best protein overall; 3 crashes |
| 12 | random_nas | protein | run_3 | 100 | 3.9693 | |
| 13 | agent | protein | run_2 | 100 | 3.9684 | 3 crashes |

**Total completed experiments:** `1,303`

## Current live status

- the Phase 2 runner is on task `14/34`
- active task: random_nas NLP `run_1`
- completed experiments in the active run: `11/100`
- current best: `exp001`, `val_bpb = 1.1431`
- first NLP track run; all protein runs (agent + random_nas) now complete

## Remaining tasks (21/34)

- **Agent:** protein run_3, NLP runs 1–5 = `6` tasks
- **Random NAS:** NLP runs 1 (in progress), 2, 3 = `3` tasks
- **HP-only agent:** SMILES runs 1–3, protein runs 1–3, NLP runs 1–3 = `9` tasks
- **Fixed default:** SMILES, protein, NLP = `3` tasks

## Where to check progress

- Runner state: `results/phase2/queue_state.json`
- Runner log: `logs/phase2-resume-20260316_103926.log`
- Active NLP run results: `results/baselines/random_nas/nlp/run_1/results.tsv`
- Active NLP run summary: `results/baselines/random_nas/nlp/run_1/summary.json`
