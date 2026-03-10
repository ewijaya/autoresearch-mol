# PRD: Fix Calibration Overfitting

**Date:** 2026-03-10
**Status:** Ready for execution
**Blocks:** Phase 1 completion (calibration study, Step 7)

---

## Problem

The calibration study compares 5-minute vs 2-hour training runs to validate that short proxy runs can rank architectures reliably (Spearman rho > 0.5 required). Currently, **2-hour runs produce WORSE val_bpb than 5-minute runs** due to severe overfitting on the small SMILES dataset.

### Evidence

With 5 of 20 calibration variant pairs completed:

| Variant | Config | Params | 5-min val_bpb | 2-hr val_bpb | Train epochs (2hr) |
|---------|--------|--------|---------------|--------------|-------------------|
| v00 | d8, dim=160, SiLU | 2.9M | 0.579 | 0.600 (+0.021) | ~122x |
| v01 | d3, dim=448, ReLU | 8.5M | 0.594 | 0.643 (+0.049) | ~83x |
| v02 | d3, dim=128, GELU | 0.7M | 0.629 | 0.619 (-0.010) | ~459x |
| v03 | d7, dim=224, ReluSquared | 5.0M | 0.577 | 0.616 (+0.039) | ~94x |
| v04 | d6, dim=416, ReLU | 14.6M | 0.587 | 0.615 (+0.028) | ~43x |

Train loss keeps decreasing (0.39 → 0.33) while val_bpb increases — classic overfitting. Spearman rho = 0.60 on these 5 pairs, but the correlation is unreliable because the 2-hour metric measures overfitting tendency, not architecture quality.

### Root cause

The SMILES training set has only **54.3M tokens** (train stream). With TOTAL_BATCH_SIZE=65536 and seq_len=256, each step consumes 65,536 tokens. A 2-hour run at ~400k tok/sec processes 2,880M tokens, cycling through the training data **53+ times**. Even at 5 minutes, some small models see the data 4-19x.

The protein dataset is even smaller: **8.9M train tokens** — the same problem will be worse there.

---

## Solution

A two-part fix: (1) increase dataset size via more SMILES enumerations, and (2) add an epoch cap to `train.py` to halt training when the model has cycled through data too many times.

### Part A: Increase SMILES dataset (prepare_smiles.py)

**Change:** Increase `NUM_ENUMERATIONS` from 5 to 20.

- Current: 224,509 train molecules × 5 enumerations = 1,122,545 sequences → 54.3M tokens
- After: 224,509 train molecules × 20 enumerations = ~4,490,180 sequences → ~217M tokens
- Val set stays at 5 enumerations (no change to evaluation, avoids inflating val set unnecessarily)

RDKit SMILES enumeration is deterministic given a random seed. Each call to `Chem.MolToSmiles(mol, canonical=False, doRandom=True)` produces a valid non-canonical SMILES. The ZINC-250K molecules average ~46 chars, and most molecules with >5 heavy atoms have well over 20 valid random SMILES orderings, so 20 enumerations will produce mostly unique strings.

**File:** `src/prepare_smiles.py`
- Change line 25: `NUM_ENUMERATIONS = 5` → `NUM_ENUMERATIONS = 20`
- Add a separate constant for val enumerations: `VAL_ENUMERATIONS = 5`
- Modify `build_sequences()` to accept an enumeration count parameter
- Update `main()` to use `NUM_ENUMERATIONS` for train and `VAL_ENUMERATIONS` for val

**Validation:**
- Run `python prepare_smiles.py` and verify:
  - `train_stream_tokens` is ~200-220M (4x increase)
  - `val_stream_tokens` is still ~6M (unchanged)
  - `canonical_overlap_count` is still 0
  - Script completes without error

### Part B: Increase protein dataset (prepare_protein.py)

**Change:** Increase `TARGET_SEQUENCES` from 50,000 to 200,000.

- Current: 45,000 train sequences → 8.9M tokens
- After: 180,000 train sequences → ~35.6M tokens
- UniRef50 has >50M filtered candidates, so sampling 200K is trivial

**File:** `src/prepare_protein.py`
- Change line 29: `TARGET_SEQUENCES = 50_000` → `TARGET_SEQUENCES = 200_000`

**Validation:**
- Run `python prepare_protein.py` and verify:
  - `train_stream_tokens` is ~35M
  - Script completes without error
- NOTE: This re-downloads from UniProt FTP. Ensure network access.

### Part C: Add epoch cap to train.py

**Change:** Add a `MAX_EPOCHS` environment variable (default: 10) that stops training when the dataloader has cycled through the training data this many times.

**File:** `src/train.py`

Add at config section (near line 621):
```python
MAX_EPOCHS = env_int("RECURSIVE_MOL_MAX_EPOCHS", 10)
```

Modify the training loop break condition (line 854-856). Currently:
```python
step += 1
if step > WARMUP_STEPS and total_training_time >= TIME_BUDGET:
    break
```

Change to:
```python
step += 1
if step > WARMUP_STEPS and total_training_time >= TIME_BUDGET:
    break
if epoch > MAX_EPOCHS:
    print(f"\nEpoch cap reached: {epoch} > {MAX_EPOCHS}")
    break
```

The `epoch` variable is already tracked by `make_stream_dataloader` — it increments each time the stream position wraps to 0. The dataloader yields `(x, y, epoch)` and `epoch` is already used in the print statement (line 832).

**Validation:**
- Run a 5-minute train on SMILES with `MAX_EPOCHS=3` and verify it stops early
- Run a 5-minute train without setting MAX_EPOCHS and verify default (10) doesn't interfere with short runs

### Part D: Re-run calibration study

**After Parts A-C are done:**

1. Delete stale calibration results:
   ```
   rm -rf results/calibration/logs/variant_*_7200.log
   ```
   Keep the 5-min logs (`variant_*_300.log`) — they will be re-run anyway since the dataset changed.

2. Actually, delete ALL calibration results since the dataset has changed:
   ```
   rm -rf results/calibration/logs/ results/calibration/results.json results/calibration/summary.json results/calibration/decision.md
   ```
   Keep `variants.json` (the variant configs are still valid).

3. Re-run the full calibration:
   ```
   cd src && python calibration.py --count 20 --seed 42 --budgets 300 7200
   ```

4. Verify:
   - All 20 variants complete for both 300s and 7200s budgets
   - `results/calibration/summary.json` contains `spearman_rho`
   - val_bpb for 2-hr runs is LOWER (better) than 5-min runs for most variants
   - Spearman rho > 0.5 (gate threshold)

### Part E: Re-run Phase 1 baselines

Since the datasets changed, re-run the 3 baseline validations:

```bash
cd src

# SMILES baseline
RECURSIVE_MOL_TRACK=smiles RECURSIVE_MOL_TIME_BUDGET=300 python train.py > ../results/phase1/baseline_smiles.log 2>&1

# Protein baseline
RECURSIVE_MOL_TRACK=protein RECURSIVE_MOL_TIME_BUDGET=300 python train.py > ../results/phase1/baseline_protein.log 2>&1

# NLP baseline (dataset unchanged, but re-run for consistency)
RECURSIVE_MOL_TRACK=nlp RECURSIVE_MOL_TIME_BUDGET=300 python train.py > ../results/phase1/baseline_nlp.log 2>&1
```

Verify: val_bpb < 4.0 on SMILES (should still pass easily).

---

## Files to modify

| File | Change |
|------|--------|
| `src/prepare_smiles.py` | NUM_ENUMERATIONS 5→20, add VAL_ENUMERATIONS=5, parametrize build_sequences |
| `src/prepare_protein.py` | TARGET_SEQUENCES 50K→200K |
| `src/train.py` | Add MAX_EPOCHS env var and epoch cap break condition |

## Files NOT to modify

- `src/prepare_char.py` — no changes needed (shared infra is correct)
- `src/calibration.py` — no changes needed (logic is correct, just needs re-run)
- `src/prepare.py` — NLP track, unchanged

## Execution order

1. Modify `prepare_smiles.py` (Part A)
2. Run `python prepare_smiles.py` — regenerate SMILES data (~2-5 min)
3. Modify `prepare_protein.py` (Part B)
4. Run `python prepare_protein.py` — regenerate protein data (~5-10 min, network-bound)
5. Modify `train.py` (Part C)
6. Quick smoke test: `RECURSIVE_MOL_TRACK=smiles RECURSIVE_MOL_TIME_BUDGET=60 RECURSIVE_MOL_MAX_EPOCHS=2 python train.py` — verify epoch cap works
7. Delete stale calibration results (Part D)
8. Run full calibration study (Part D) — ~11 hours (20 variants × 300s + 20 × 7200s)
9. Re-run baselines (Part E) — ~15 min
10. Report final Spearman rho and Phase 1 checkpoint status

## Estimated time

- Code changes: ~15 min
- Data regeneration: ~15 min
- Calibration re-run: ~11 hours (can run unattended)
- Baselines: ~15 min
- **Total wall time: ~12 hours** (dominated by calibration)

## Risk

- **Low risk:** More enumerations cannot hurt — it's strictly more training data with the same molecule split, so no data leakage concerns.
- **Low risk:** Epoch cap defaults to 10, which won't trigger for most 5-min runs (they see data 2-5x). Only affects long runs on small datasets, which is exactly the overfitting scenario.
- **Medium risk:** Protein data re-download depends on UniProt FTP availability. If it fails, increase enumerations or use a local cached copy if available at `data/protein/raw/`.

## Success criteria

After re-running calibration with the larger datasets + epoch cap:
- 2-hr val_bpb should be LOWER than 5-min val_bpb for ≥15 of 20 variants
- Spearman rho > 0.5 (gate pass)
- Ideally rho > 0.7 (proceed confidently)
