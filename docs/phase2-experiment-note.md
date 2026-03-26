# Phase 2 Experiment Note

Snapshot updated on March 26, 2026.

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

## Completed tasks (28/34)

| # | Kind | Track | Run | Experiments | Best val_bpb | Notes |
|---|------|-------|-----|-------------|-------------|-------|
| 1 | agent | SMILES | run_1 | 100 | 0.5918 | 1 crash |
| 2 | agent | SMILES | run_2 | 100 | 0.5808 | 3 crashes; overall best SMILES |
| 3 | agent | SMILES | run_3 | 100 | 0.5839 | 4 crashes |
| 4 | agent | SMILES | run_4 | 100 | 0.5892 | |
| 5 | random_nas | SMILES | run_1 | 100 | 0.5906 | |
| 6 | random_nas | SMILES | run_2 | 100 | 0.5923 | |
| 7 | random_nas | SMILES | run_3 | 100 | 0.5914 | |
| 8 | agent | SMILES | run_5 | 100 | 0.5834 | 1 crash |
| 9 | random_nas | protein | run_1 | 103 | 3.9719 | Slightly over 100 |
| 10 | random_nas | protein | run_2 | 100 | 3.9710 | |
| 11 | agent | protein | run_1 | 100 | 3.9656 | 3 crashes; best protein overall |
| 12 | random_nas | protein | run_3 | 100 | 3.9693 | |
| 13 | agent | protein | run_2 | 100 | 3.9684 | 3 crashes |
| 14 | random_nas | NLP | run_1 | 100 | 1.1297 | 3 crashes |
| 15 | agent | protein | run_3 | 100 | 3.9666 | 5 crashes |
| 16 | random_nas | NLP | run_2 | 100 | 1.1301 | 2 crashes |
| 17 | agent | NLP | run_1 | 100 | 1.1188 | 4 crashes |
| 18 | random_nas | NLP | run_3 | 100 | 1.1306 | 1 crash |
| 19 | agent | NLP | run_2 | 100 | 1.1277 | |
| 20 | hp_only | SMILES | run_1 | 100 | 0.5807 | |
| 21 | agent | NLP | run_3 | 100 | 1.1151 | 2 crashes; best NLP overall |
| 22 | hp_only | SMILES | run_2 | 100 | 0.5801 | best hp_only SMILES |
| 23 | agent | NLP | run_4 | 100 | 1.1212 | |
| 24 | hp_only | SMILES | run_3 | 100 | 0.5810 | 1 crash |
| 25 | agent | NLP | run_5 | 100 | 1.1314 | 21 crashes |
| 26 | hp_only | protein | run_1 | 100 | 3.9901 | 8 crashes |
| 27 | hp_only | protein | run_2 | 100 | 3.9699 | best hp_only protein |
| 28 | hp_only | protein | run_3 | 100 | 3.9684 | |

**Total completed experiments:** `2,803`

## Current live status

- the Phase 2 runner is on task `29/34`
- active task: hp_only NLP `run_1`
- completed experiments in the active run: `12/~100`
- current best: `exp011`, `val_bpb = 1.1507`
- all agent runs complete (SMILES 5/5, protein 3/3, NLP 5/5)
- all random_nas runs complete (SMILES 3/3, protein 3/3, NLP 3/3)
- hp_only SMILES runs 1–3 complete; hp_only protein runs 1–3 complete; hp_only NLP run_1 in progress

## Remaining tasks (6/34)

- **HP-only:** NLP runs 1 (in progress, 12/~100), 2, 3 = `3` tasks
- **Fixed default:** SMILES, protein, NLP = `3` tasks

## Where to check progress

- Runner state: `results/phase2/queue_state.json`
- Runner log: `logs/phase2-resume-20260316_103926.log`
- Active run results: `results/baselines/hp_only/nlp/run_1/results.tsv`
- Active run summary: `results/baselines/hp_only/nlp/run_1/summary.json`
