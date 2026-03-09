# Phase 1 Resume Prompt

Use this if codex runs out of context before Phase 1 completes.

## Start a new codex session

```bash
tmux new -s phase1b
cd ~/storage1/recursive-mol/src
codex --full-auto
```

## Paste this prompt

```
You are working on the recursive-mol project (PRD: docs/PRD-recursive-mol.md).
Phase 1 is partially complete. Check what has already been done and resume from where it left off.

Check existing progress:
- ls ../results/phase1/ to see which baseline logs exist
- ls ../data/smiles/ and ../data/protein/ to see which datasets are prepared
- ls results/calibration/ to see if calibration has started

Use .venv/bin/python for ALL python commands (the venv is already set up in src/).

Resume any incomplete steps from this list:

STEP 4 — Verify prepare_protein.py:
- If ../data/protein/ is empty or missing, run: .venv/bin/python prepare_protein.py
- Verify output files exist and ~12.5M total tokens

STEP 6 — Run baseline validation:
- If baseline logs are missing, train the unmodified architecture for 5 minutes on each track:
  RECURSIVE_MOL_TRACK=smiles .venv/bin/python train.py
  RECURSIVE_MOL_TRACK=protein .venv/bin/python train.py
  RECURSIVE_MOL_TRACK=nlp .venv/bin/python train.py
- Verify val_bpb < 4.0 on SMILES
- Verify all 3 tracks produce valid val_bpb without errors

STEP 7 — Run calibration study:
- If results/calibration/ is empty or incomplete, run .venv/bin/python calibration.py
- 20 random architecture variants, train each for 5 min on SMILES → record val_bpb
- Train each for 2 hours on SMILES → record val_bpb
- Compute Spearman rank correlation between 5-min and 2-hr val_bpb
- Save results to results/calibration/
- DECISION GATE: rho > 0.7 → proceed. rho 0.4-0.7 → proceed with caution. rho < 0.4 → increase TIME_BUDGET to 15-30 min.

After completing all steps, report the status of each Checkpoint 1 criterion:
- python prepare_smiles.py completes without error
- python train.py completes 5-minute training on SMILES data
- Baseline val_bpb < 4.0 (below random ~5.3)
- Proxy calibration Spearman rho > 0.5
- Protein prepare_protein.py functional
- All 3 tracks produce valid val_bpb on a single baseline run
- VRAM usage < 12GB on A10G
- Model parameter count in 8-12M range

FINALLY: Stop the EC2 instance by running: /home/ubuntu/bin/stopinstance
```
