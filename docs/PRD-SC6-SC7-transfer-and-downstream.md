# PRD: SC-6 Transfer Matrix & SC-7 MoleculeNet Validation

**Project:** recursive-mol
**Criteria:** SC-6 (transfer matrix computed) and SC-7 (MoleculeNet validation complete)
**Author:** Rex
**Date:** 2026-03-28
**Status:** READY TO EXECUTE
**Depends on:** Phase 2 complete (all 34/34 tasks done)
**Blocks:** H3 sub-hypothesis tests in SC-8, paper writing (Phase 5)

---

## 1. Objective

Implement two scripts and run the GPU experiments they require:

1. **`scripts/transfer_eval.py`** — Cross-domain transfer matrix (SC-6) + layer freezing + length-controlled transfer
2. **`scripts/moleculenet_eval.py`** — MoleculeNet fine-tuning (SC-7) + generation metrics

### SC-6 acceptance criterion (from PRD)

> Cross-evaluate top architecture from each track on all 3 datasets. 3×3 matrix with mean + std from 3 replicates each. `results/transfer/matrix.json` exists and is complete.

### SC-7 acceptance criterion (from PRD)

> Top 3 SMILES architectures fine-tuned on BBBP, HIV, BACE. ROC-AUC reported for all 9 combinations (3 archs × 3 tasks). `results/moleculenet/scores.json` exists.

---

## 2. Scope and Ordering

SC-6 and SC-7 share infrastructure (modified train.py execution, checkpoint saving) but are otherwise independent. Execute SC-6 first because:

- SC-6 results feed H3 tests in SC-8 (which is already running but will skip H3)
- SC-7 needs a retrain-with-checkpoint capability that SC-6 also needs → build once, use for both
- SC-6 output (innovation classification) contextualizes SC-7 results in the paper

**Execution order:**
1. Build shared infrastructure (checkpoint saving, transfer runner)
2. SC-6: transfer matrix → layer freezing → length-controlled → innovation classification
3. SC-7: MoleculeNet fine-tuning → generation metrics → correlation analysis

**Total GPU time estimate:** ~8–10 hours on A10G

---

## 3. Shared Infrastructure

### 3.1 Problem: No checkpoints exist

The current `train.py` trains from scratch and reports val_bpb but **does not save model weights**. Both SC-6 and SC-7 need checkpoints. Two approaches:

**Option A (preferred): Retrain and save.** Take each best architecture's train.py, add `torch.save(model.state_dict(), path)` after training, run for 5 minutes. This produces a checkpoint. Cost: 5 min per architecture.

**Option B: Modify train.py to always save.** Add checkpoint saving to the base train.py. But this modifies a file that agent experiments depend on, which is risky.

**Use Option A.** Create a wrapper script that:
1. Copies the best architecture's train.py to a temp location
2. Appends checkpoint saving after the evaluation block
3. Runs it with the appropriate TRACK env var
4. Collects the checkpoint

### 3.2 Identifying the best architectures

From Phase 2 summary.json files, the best architecture per track per condition:

**Agent (best per track — used for SC-6 transfer matrix):**

| Track | Run | Best Experiment | val_bpb | Source |
|-------|-----|-----------------|---------|--------|
| SMILES | run_2 | exp047 | 0.5808 | `results/smiles/run_2/train_versions/exp047_keep.py` |
| protein | run_1 | exp099 | 3.9656 | `results/protein/run_1/train_versions/exp099_keep.py` |
| NLP | run_3 | exp081 | 1.1151 | `results/nlp/run_3/train_versions/exp081_keep.py` |

**Agent top 3 SMILES (used for SC-7 MoleculeNet):**

| Rank | Run | Best Experiment | val_bpb | Source |
|------|-----|-----------------|---------|--------|
| 1 | run_2 | exp047 | 0.5808 | `results/smiles/run_2/train_versions/exp047_keep.py` |
| 2 | run_5 | exp099 | 0.5834 | `results/smiles/run_5/train_versions/exp099_keep.py` |
| 3 | run_3 | exp051 | 0.5839 | `results/smiles/run_3/train_versions/exp051_keep.py` |

**IMPORTANT:** Verify these by reading the actual summary.json files before running. The values above are from the last documentation update and should be confirmed.

### 3.3 How cross-track execution works

Each agent's best train.py was evolved for its native track. To run architecture X on track Y:

1. Copy X's train.py
2. Set `RECURSIVE_MOL_TRACK=Y` environment variable
3. The script dynamically imports the correct `prepare_*.py` module, tokenizer, and data
4. Vocabulary size, sequence length, and batch size adjust automatically
5. The model architecture (depth, width, attention pattern, etc.) stays as coded

**Key constraint:** Some agent-modified train.py files may have hardcoded track-specific values (e.g., explicit vocab_size, seq_len). The transfer script must:
- Check if vocab_size is hardcoded and override it with the target track's tokenizer
- Check if MAX_SEQ_LEN is hardcoded and override it for the target track
- Log any such overrides

### 3.4 Track-specific parameters

| Parameter | SMILES | Protein | NLP |
|-----------|--------|---------|-----|
| MAX_SEQ_LEN | 256 | 512 | 2048 |
| DEVICE_BATCH_SIZE | 256 | 128 | 32 |
| Vocab size (approx) | ~50 | ~25 | ~8192 |
| Data module | `prepare_smiles` | `prepare_protein` | `prepare` |

---

## 4. SC-6: Transfer Matrix

### 4.1 Step 1 — 3×3 Cross-Domain Transfer Matrix

Take the 3 best agent architectures (one per track) and evaluate each on all 3 datasets. This produces a 3×3 matrix.

**Runs required:**

| Architecture source | Target track | Replicates | Total runs |
|--------------------:|:------------|:----------:|:----------:|
| SMILES arch | SMILES data | 3 | 3 (identity — retrain to verify) |
| SMILES arch | protein data | 3 | 3 |
| SMILES arch | NLP data | 3 | 3 |
| protein arch | SMILES data | 3 | 3 |
| protein arch | protein data | 3 | 3 |
| protein arch | NLP data | 3 | 3 |
| NLP arch | SMILES data | 3 | 3 |
| NLP arch | protein data | 3 | 3 |
| NLP arch | NLP data | 3 | 3 |
| **Total** | | | **27 runs** |

Each run: train from scratch for 5 minutes, report val_bpb. Different random seeds per replicate (seeds: 42, 137, 2026).

**GPU time:** 27 × 5 min = 135 min ≈ 2.3 hours

**Implementation:**

```python
# Pseudocode for transfer_eval.py
for arch_track in ["smiles", "protein", "nlp"]:
    train_py = load_best_architecture(arch_track)
    for data_track in ["smiles", "protein", "nlp"]:
        for seed in [42, 137, 2026]:
            val_bpb = run_training(
                train_py=train_py,
                track=data_track,
                seed=seed,
                time_budget=300,
            )
            results[arch_track][data_track].append(val_bpb)
```

**Output:**

```json
// results/transfer/matrix.json
{
  "generated_at": "2026-03-...",
  "architectures": {
    "smiles": {"source": "results/smiles/run_2/train_versions/exp047_keep.py", "native_bpb": 0.5808},
    "protein": {"source": "results/protein/run_1/train_versions/exp099_keep.py", "native_bpb": 3.9656},
    "nlp": {"source": "results/nlp/run_3/train_versions/exp081_keep.py", "native_bpb": 1.1151}
  },
  "matrix": {
    "smiles_arch": {
      "smiles_data": {"mean": 0.582, "std": 0.001, "runs": [0.581, 0.583, 0.582]},
      "protein_data": {"mean": 3.978, "std": 0.002, "runs": [3.976, 3.979, 3.978]},
      "nlp_data": {"mean": 1.145, "std": 0.003, "runs": [1.143, 1.147, 1.145]}
    },
    "protein_arch": {...},
    "nlp_arch": {...}
  },
  "degradation_matrix": {
    "smiles_arch": {
      "smiles_data": {"pct_degradation": 0.0, "note": "identity"},
      "protein_data": {"pct_degradation": 2.3},
      "nlp_data": {"pct_degradation": 1.5}
    },
    ...
  }
}
```

**Degradation calculation:**

For architecture A evaluated on track T:

```
baseline_T = fixed_default val_bpb for track T
native_A  = fixed_default val_bpb for arch A's native track
transfer_bpb = mean val_bpb when running A on T

pct_degradation = 100 * (transfer_bpb - baseline_T) / baseline_T
```

A positive pct_degradation means the transferred architecture is worse than the target track's default. A negative value means it's better (the transferred architecture helps).

### 4.2 Step 2 — Layer Freezing (H3b)

For each of the 6 cross-domain pairs (excluding identity), progressively freeze layers to identify which layers are domain-specific vs. universal.

**Freeze levels:** freeze first 1, 2, 3, 4, 5 layers (out of 6 total). That's 5 levels per pair.

**But this requires pretrained weights.** The procedure is:
1. Train architecture A on its native track T_A for 5 min → save checkpoint
2. Load checkpoint, set track to T_B (target)
3. Freeze first N layers (set `requires_grad=False`)
4. Train unfrozen layers for 5 min on T_B data
5. Report val_bpb

**Runs required:** 6 pairs × 5 freeze levels × 1 replicate = 30 runs (plus 3 checkpoint-creation runs)

**GPU time:** 33 × 5 min = 165 min ≈ 2.8 hours

**Simplification:** Given time pressure, run only 3 freeze levels (freeze 1, 3, 5 layers) instead of all 5. This cuts to 6 × 3 = 18 runs.

**GPU time (simplified):** 21 × 5 min = 105 min ≈ 1.8 hours

**Output:**

```json
// results/transfer/layer_freezing.json
{
  "generated_at": "...",
  "pairs": [
    {
      "arch_source": "smiles",
      "data_target": "protein",
      "freeze_levels": [
        {"frozen_layers": 1, "val_bpb": 3.972},
        {"frozen_layers": 3, "val_bpb": 3.985},
        {"frozen_layers": 5, "val_bpb": 4.012}
      ],
      "no_freeze_baseline": 3.978,
      "native_baseline": 3.9767
    },
    ...
  ]
}
```

### 4.3 Step 3 — Length-Controlled Transfer (H3c)

Test whether sequence length mismatch is the primary transfer barrier.

**Procedure:**
1. Identify the worst-performing transfer pairs from Step 1 (highest degradation)
2. Re-run those pairs with matched sequence lengths:
   - When transferring to a shorter-sequence track: truncate source track sequences to match
   - When transferring to a longer-sequence track: pad/truncate target data to match source's typical length
3. Compare degradation with vs. without length matching

**Practical implementation:** For SMILES→NLP or NLP→SMILES transfers, set MAX_SEQ_LEN to the shorter value (256) for both. For protein→NLP or NLP→protein, use 512.

**Runs:** Top 3 worst pairs × 1 replicate × 1 length-matched run = 3 runs (plus comparison against Step 1 results)

**GPU time:** 3 × 5 min = 15 min

**Output:**

```json
// results/transfer/length_controlled.json
{
  "pairs": [
    {
      "arch_source": "nlp",
      "data_target": "smiles",
      "matched_seq_len": 256,
      "val_bpb_unmatched": 0.598,
      "val_bpb_matched": 0.589,
      "degradation_reduction_pct": 52.3,
      "h3c_criterion_met": true
    },
    ...
  ]
}
```

### 4.4 Step 4 — Innovation Classification (H3d)

Using the transfer matrix from Step 1, classify each distinct architectural change as "universal" or "domain-specific."

**Procedure:**
1. For each track's best architecture, identify the architectural differences from the default (parse train.py diffs)
2. For each difference, check whether it helps on other tracks (from the transfer matrix):
   - If cross-domain degradation < 10% relative to native: **universal**
   - If cross-domain degradation >= 10%: **domain-specific**
3. Compute the proportion split

**This is an analysis step, not a GPU step.** It reads the transfer matrix output.

**Output:**

```json
// results/transfer/innovation_classification.json
{
  "innovations": [
    {
      "source_track": "nlp",
      "description": "n_kv_head reduced from 5 to 1 (5:1 GQA)",
      "type": "architectural",
      "cross_domain_degradation": {"smiles": 3.2, "protein": 5.1},
      "classification": "universal"
    },
    {
      "source_track": "protein",
      "description": "Per-layer KV head variation with n_v_head",
      "type": "architectural",
      "cross_domain_degradation": {"smiles": 12.4, "nlp": 15.7},
      "classification": "domain_specific"
    },
    ...
  ],
  "summary": {
    "total_innovations": 8,
    "universal_count": 3,
    "domain_specific_count": 5,
    "universal_pct": 37.5,
    "domain_specific_pct": 62.5
  }
}
```

### 4.5 SC-6 total GPU time

| Step | Runs | Time |
|------|------|------|
| Transfer matrix (27 runs) | 27 | 2.3 hrs |
| Checkpoint creation (3 runs) | 3 | 0.3 hrs |
| Layer freezing (18 runs) | 18 | 1.5 hrs |
| Length-controlled (3 runs) | 3 | 0.3 hrs |
| Innovation classification | 0 (CPU) | — |
| **SC-6 total** | **51 runs** | **~4.4 hrs** |

---

## 5. SC-7: MoleculeNet Validation

### 5.1 Overview

Fine-tune pretrained SMILES architectures on 3 MoleculeNet classification tasks to validate that val_bpb improvements translate to downstream molecular property prediction.

### 5.2 Prerequisites

**DeepChem or PyTorch Geometric** must be installed for MoleculeNet data loading and scaffold splitting.

```bash
pip install deepchem
# or, if deepchem has dependency issues:
pip install ogb  # Open Graph Benchmark has MoleculeNet loaders too
```

**Fallback:** If DeepChem is too heavy, use the raw MoleculeNet CSV files (available from the MoleculeNet website) with a manual scaffold split. The datasets are small:

| Task | Molecules | Task type | Metric | Split |
|------|-----------|-----------|--------|-------|
| BBBP | 2,039 | Binary classification | ROC-AUC | Scaffold |
| HIV | 41,127 | Binary classification | ROC-AUC | Scaffold |
| BACE | 1,513 | Binary classification | ROC-AUC | Scaffold |

### 5.3 Fine-tuning approach

There are two viable approaches for downstream evaluation:

**Approach A: Feature extraction (preferred, simpler)**
1. Train the SMILES architecture for 5 min on ZINC-250K (pretrain) → save checkpoint
2. Load checkpoint, freeze all transformer layers
3. Add a classification head: `Linear(n_embd, 1)` on top of mean-pooled hidden states
4. Fine-tune only the classification head on MoleculeNet task
5. Report ROC-AUC on scaffold test split

**Approach B: Full fine-tuning**
1. Same pretrain step
2. Load checkpoint, add classification head
3. Fine-tune entire model with small LR on MoleculeNet task
4. Report ROC-AUC

**Use Approach A** — it's more standard for evaluating pretrained representations, and faster. Fall back to Approach B only if Approach A gives near-random results across all architectures.

### 5.4 Implementation: `scripts/moleculenet_eval.py`

```python
# High-level structure

class MoleculeNetEvaluator:
    def __init__(self, train_py_path, task_name, seed=42):
        """
        train_py_path: path to the best architecture's train.py
        task_name: "bbbp", "hiv", or "bace"
        """
        self.train_py_path = train_py_path
        self.task_name = task_name
        self.seed = seed

    def pretrain(self):
        """Train on ZINC-250K for 5 min, save checkpoint."""
        # Run train.py with RECURSIVE_MOL_TRACK=smiles
        # Append torch.save() to save model state dict
        # Return checkpoint path

    def load_moleculenet_data(self):
        """Load BBBP/HIV/BACE with scaffold split."""
        # Use DeepChem or raw CSV
        # Convert SMILES to token sequences using prepare_smiles tokenizer
        # Return train/val/test splits

    def extract_features(self, smiles_list):
        """Run pretrained model on SMILES, extract mean-pooled hidden states."""
        # Tokenize each SMILES
        # Forward pass through frozen model
        # Mean-pool over sequence length → [batch, n_embd] features
        # Return feature matrix

    def train_classifier(self, train_features, train_labels):
        """Train logistic regression on extracted features."""
        # sklearn LogisticRegression or simple Linear + BCE
        # Return trained classifier

    def evaluate(self, test_features, test_labels):
        """Compute ROC-AUC on test set."""
        # Return ROC-AUC score
```

### 5.5 Runs required

| Architecture | Task | Replicates | Runs |
|-------------|------|:----------:|:----:|
| SMILES agent #1 (run_2 exp047) | BBBP | 3 | 3 |
| SMILES agent #1 | HIV | 3 | 3 |
| SMILES agent #1 | BACE | 3 | 3 |
| SMILES agent #2 (run_5 exp099) | BBBP | 3 | 3 |
| SMILES agent #2 | HIV | 3 | 3 |
| SMILES agent #2 | BACE | 3 | 3 |
| SMILES agent #3 (run_3 exp051) | BBBP | 3 | 3 |
| SMILES agent #3 | HIV | 3 | 3 |
| SMILES agent #3 | BACE | 3 | 3 |
| **Total** | | | **27 runs** |

**But pretrain only needs to happen once per architecture** (3 pretrain runs, not 27). The feature extraction + classification is CPU-only and fast.

**GPU time:** 3 architectures × 5 min pretrain = 15 min GPU. The rest (feature extraction on small MoleculeNet datasets + logistic regression) is CPU-only and takes seconds.

**Optionally also evaluate:**
- Fixed default architecture (as baseline) — 1 additional pretrain run
- Best random_nas SMILES architecture — 1 additional pretrain run

This enables comparing agent-discovered vs baseline vs random on downstream tasks too. Cost: +10 min GPU.

### 5.6 Generation metrics

Load the best SMILES architecture checkpoint (already created for MoleculeNet). Generate 10K SMILES strings autoregressively.

**Procedure:**
1. Load pretrained checkpoint
2. Generate 10,000 SMILES by sampling from the model autoregressively (temperature=1.0, top-k or nucleus sampling)
3. For each generated SMILES, check validity using RDKit:
   ```python
   from rdkit import Chem
   mol = Chem.MolFromSmiles(generated_smiles)
   valid = mol is not None
   ```
4. Compute metrics:
   - **Validity:** % of generated SMILES that parse to a valid molecule
   - **Uniqueness:** % of valid molecules that are unique (by canonical SMILES)
   - **Novelty:** % of unique valid molecules not in the training set
   - **FCD (Fréchet ChemNet Distance):** distributional distance between generated and training molecules (requires `fcd` package: `pip install fcd`)

**GPU time:** ~5 min for generation (10K sequences × ~50 tokens each, batch generation)

### 5.7 Correlation analysis

After all MoleculeNet results are collected:

1. Rank the 3 architectures by val_bpb (from Phase 2)
2. Rank the 3 architectures by mean ROC-AUC (averaged across tasks)
3. Compute Spearman rank correlation between the two rankings

With only 3 architectures, Spearman ρ can only be -1, -0.5, 0, 0.5, or 1. This is acknowledged as low-power; it's supplementary evidence.

If we add fixed_default and random_nas (5 architectures total), the correlation becomes more meaningful.

### 5.8 SC-7 Output

```json
// results/moleculenet/scores.json
{
  "generated_at": "2026-03-...",
  "architectures": [
    {
      "name": "agent_smiles_run2_exp047",
      "source": "results/smiles/run_2/train_versions/exp047_keep.py",
      "val_bpb": 0.5808,
      "tasks": {
        "bbbp": {"roc_auc_mean": 0.723, "roc_auc_std": 0.012, "runs": [0.715, 0.728, 0.726]},
        "hiv": {"roc_auc_mean": 0.681, "roc_auc_std": 0.015, "runs": [0.669, 0.695, 0.678]},
        "bace": {"roc_auc_mean": 0.756, "roc_auc_std": 0.018, "runs": [0.742, 0.769, 0.757]}
      }
    },
    ...
  ],
  "correlation": {
    "val_bpb_ranking": [1, 2, 3],
    "roc_auc_ranking": [1, 3, 2],
    "spearman_rho": 0.5,
    "spearman_p": 0.667,
    "note": "With n=3, rank correlation has very low power. Interpret qualitatively."
  }
}
```

```json
// results/moleculenet/generation_metrics.json
{
  "architecture": "agent_smiles_run2_exp047",
  "num_generated": 10000,
  "temperature": 1.0,
  "sampling": "top_k_50",
  "validity": 0.87,
  "uniqueness": 0.94,
  "novelty": 0.82,
  "fcd": 12.3,
  "note": "FCD computed against ZINC-250K training set"
}
```

---

## 6. Implementation Plan

### 6.1 Script 1: `scripts/transfer_eval.py`

**What it does:** All SC-6 work (transfer matrix, layer freezing, length-controlled transfer, innovation classification).

**Capabilities needed:**
1. Load a train.py file and execute it as a subprocess with modified environment variables
2. Parse the output to extract val_bpb
3. Save and load model checkpoints (for layer freezing)
4. Classify architectural innovations by parsing code diffs

**Running a single transfer experiment:**

```python
def run_transfer(train_py_path, target_track, seed=42, time_budget=300,
                 checkpoint_path=None, freeze_layers=0, seq_len_override=None):
    """
    Run a single transfer experiment.

    1. Copy train_py_path to a temp working directory
    2. Optionally inject checkpoint loading + layer freezing code
    3. Optionally inject checkpoint saving code
    4. Set RECURSIVE_MOL_TRACK=target_track
    5. Set seed
    6. Execute and parse val_bpb from stdout
    """
    env = {
        "RECURSIVE_MOL_TRACK": target_track,
        "RECURSIVE_MOL_TIME_BUDGET": str(time_budget),
        "RECURSIVE_MOL_SEED": str(seed),
        "WANDB_DISABLED": "1",
    }
    if seq_len_override:
        env["RECURSIVE_MOL_MAX_SEQ_LEN"] = str(seq_len_override)

    # ... inject checkpoint load/save/freeze as needed
    # ... run subprocess
    # ... parse stdout for val_bpb line
    return val_bpb
```

**Critical implementation detail — modifying train.py for checkpoint support:**

The train.py files from agent runs don't save checkpoints. To add this, inject code after the evaluation block (after `val_bpb = evaluate_bpb(...)`) but before the print block:

```python
# Inject after evaluate_bpb
import os as _os
_ckpt_path = _os.environ.get("RECURSIVE_MOL_CHECKPOINT_SAVE", "")
if _ckpt_path:
    torch.save(model.state_dict(), _ckpt_path)
    print(f"checkpoint_saved: {_ckpt_path}")
```

For loading checkpoints + layer freezing, inject code after `model.init_weights()`:

```python
_ckpt_load = _os.environ.get("RECURSIVE_MOL_CHECKPOINT_LOAD", "")
if _ckpt_load:
    _state = torch.load(_ckpt_load, map_location=device, weights_only=True)
    model.load_state_dict(_state, strict=False)
    print(f"checkpoint_loaded: {_ckpt_load}")

_freeze_n = int(_os.environ.get("RECURSIVE_MOL_FREEZE_LAYERS", "0"))
if _freeze_n > 0:
    for _i, _block in enumerate(model.blocks):
        if _i < _freeze_n:
            for _p in _block.parameters():
                _p.requires_grad = False
    print(f"frozen_layers: {_freeze_n}")
```

**IMPORTANT:** Use `strict=False` in `load_state_dict` because vocabulary size will differ across tracks (the embedding and LM head dimensions change). The transfer loads only the transformer body weights. Log any missing/unexpected keys.

### 6.2 Script 2: `scripts/moleculenet_eval.py`

**What it does:** All SC-7 work.

**Dependencies:**
- `rdkit` — for SMILES validation and molecule manipulation (likely already installed for prepare_smiles.py)
- `deepchem` or manual CSV loading — for MoleculeNet data
- `scikit-learn` — for LogisticRegression classifier and ROC-AUC scoring
- `fcd` — for Fréchet ChemNet Distance (optional; skip if not installable)

**Check if rdkit is already available:**
```bash
python -c "from rdkit import Chem; print('rdkit OK')"
```

**If deepchem is problematic, use manual data loading:**
MoleculeNet datasets are available as CSVs. Download from:
- BBBP: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv`
- HIV: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv`
- BACE: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv`

Implement scaffold split manually using RDKit's `Chem.Scaffolds.MurckoScaffold`.

### 6.3 Script execution

The whole pipeline runs as:

```bash
# SC-6
cd /home/ubuntu/storage1/recursive-mol
python scripts/transfer_eval.py

# SC-7 (can start immediately after, or in parallel on separate GPU)
python scripts/moleculenet_eval.py
```

Each script should:
1. Print clear progress logs
2. Save intermediate results as it goes (so partial runs are recoverable)
3. Skip already-completed runs (check if output files exist)
4. Produce a final summary JSON

---

## 7. Output File Summary

### SC-6 outputs

```
results/transfer/matrix.json                    — 3×3 transfer matrix with mean ± std
results/transfer/layer_freezing.json            — degradation vs frozen layers for 6 pairs
results/transfer/length_controlled.json         — length-matched transfer results
results/transfer/innovation_classification.json — universal vs domain-specific labels
results/transfer/checkpoints/                   — saved model checkpoints (3 files)
results/transfer/raw/                           — per-run val_bpb results
figures/transfer_heatmap.png                    — annotated 3×3 heatmap
figures/layer_freezing_curves.png               — degradation vs frozen layers plot
figures/innovation_pie.png                      — universal vs domain-specific pie chart
```

### SC-7 outputs

```
results/moleculenet/scores.json                 — ROC-AUC for all 9 conditions + correlation
results/moleculenet/generation_metrics.json     — validity, uniqueness, novelty, FCD
results/moleculenet/features/                   — cached feature matrices (optional)
figures/moleculenet_bar.png                     — ROC-AUC bar chart by architecture × task
figures/bpb_vs_rocauc_scatter.png               — val_bpb vs ROC-AUC correlation plot
```

---

## 8. Total GPU Budget

| Component | Runs | Time per run | Total time |
|-----------|------|:------------:|:----------:|
| **SC-6: Transfer matrix** | 27 | 5 min | 2.3 hrs |
| **SC-6: Checkpoint creation** | 3 | 5 min | 0.3 hrs |
| **SC-6: Layer freezing** | 18 | 5 min | 1.5 hrs |
| **SC-6: Length-controlled** | 3 | 5 min | 0.3 hrs |
| **SC-7: Pretrain architectures** | 3–5 | 5 min | 0.3–0.4 hrs |
| **SC-7: Generation** | 1 | 5 min | 0.1 hrs |
| **SC-7: Feature extraction** | — | CPU | negligible |
| **Buffer (10%)** | — | — | 0.5 hrs |
| **Grand total** | ~55–57 | | **~5.3–5.4 hrs** |

This fits comfortably in a single session on the A10G.

---

## 9. Edge Cases and Failure Modes

| Issue | How to handle |
|-------|---------------|
| **Agent train.py has hardcoded vocab_size** | Detect via regex; override using env var or code injection |
| **Agent train.py has custom classes not in base** | Copy the full file, not just config vars — the custom classes ARE the architecture |
| **Cross-track model has mismatched embedding dims** | Expected — embedding/LM_head weights won't transfer. Use `strict=False`, reinitialize embedding layers for target vocab |
| **DeepChem installation fails** | Fall back to manual CSV + scaffold split |
| **FCD package unavailable** | Skip FCD metric; report other 3 metrics |
| **Layer freezing + different vocab causes crash** | Reinitialize embedding layers after loading checkpoint; only freeze transformer body |
| **NLP architecture on SMILES data OOMs** | NLP uses seq_len=2048 which may OOM with SMILES batch size. Set DEVICE_BATCH_SIZE lower for cross-track runs |
| **val_bpb parsing fails** | Look for line matching `val_bpb:\s+[\d.]+` in stdout. If not found, check stderr for crash |
| **Run times out (>10 min)** | Set subprocess timeout to 600s. Log timeout as failure. |
| **Partial completion** | Each step saves results independently. Re-running skips completed steps. |

---

## 10. Verification Checklist

### SC-6

- [ ] `results/transfer/matrix.json` exists with all 9 cells populated (mean + std + 3 runs each)
- [ ] Degradation percentages are computed correctly (positive = worse than baseline)
- [ ] Identity cells (arch on its own track) show near-zero degradation
- [ ] Layer freezing shows monotonically increasing degradation as more layers are frozen
- [ ] Length-controlled results show reduced degradation vs unmatched
- [ ] Innovation classification totals match number of distinct architectural differences
- [ ] All 3 heatmap/figure files exist

### SC-7

- [ ] `results/moleculenet/scores.json` exists with all 9 conditions (3 archs × 3 tasks)
- [ ] ROC-AUC values are between 0.5 and 1.0 (above random chance)
- [ ] ROC-AUC values are reasonable for these tasks (BBBP: ~0.65-0.85, HIV: ~0.65-0.80, BACE: ~0.70-0.85)
- [ ] Spearman correlation computed (even if not significant)
- [ ] Generation metrics show validity > 50% (ideally > 80%)
- [ ] Uniqueness and novelty computed for valid molecules only
- [ ] All figure files exist

---

## 11. Prompt for New Session

Paste this into a new session to execute:

---

Execute the PRD at `docs/PRD-SC6-SC7-transfer-and-downstream.md`. Read it fully before writing any code.

Your job:

1. Implement `scripts/transfer_eval.py` (SC-6: transfer matrix, layer freezing, length-controlled transfer, innovation classification)
2. Implement `scripts/moleculenet_eval.py` (SC-7: MoleculeNet fine-tuning, generation metrics, correlation)
3. Run both scripts and verify outputs pass the checklist in Section 10

Key points:
- Read the PRD first — it specifies the exact architecture sources, data paths, output formats, and edge cases
- The main challenge is injecting checkpoint save/load/freeze code into agent-modified train.py files
- Use `strict=False` for cross-track checkpoint loading (vocab mismatch is expected)
- If DeepChem is hard to install, fall back to manual CSV + scaffold split (see Section 5.2)
- Save intermediate results so partial runs are recoverable
- Do not modify any existing result files — only create new outputs in `results/transfer/` and `results/moleculenet/`

---

*PRD version 1.0 — March 28, 2026*
*Derived from: PRD-recursive-mol.md (SC-6, SC-7, Phase 4), phase-prompts.md (Phase 4), stress-test-transfer-hypothesis.md*
