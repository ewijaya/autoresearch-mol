# PRD: SC-8 Statistical Analysis

**Project:** recursive-mol
**Criterion:** SC-8 — Statistical analysis complete
**Author:** Rex
**Date:** 2026-03-28
**Status:** READY TO EXECUTE
**Depends on:** Phase 2 complete (all 34/34 tasks done, 3,106 experiments)
**Blocks:** Phase 5 paper writing, SC-9 arXiv preprint

---

## 1. Objective

Produce the file `results/analysis/hypothesis_tests.json` containing quantified evidence (Bayes factors, p-values, bootstrap CIs, effect sizes) for all 4 hypotheses (H1–H4), plus all supporting figures, tables, and intermediate artifacts needed for the paper. Implement this as `scripts/analyze_phase2.py` — a single standalone script that reads the completed Phase 2 data and produces all outputs.

### SC-8 acceptance criterion (from PRD)

> Bayesian hierarchical model fitted; permutation test on architectural distance matrices. All 4 hypotheses have quantified evidence (Bayes factors or p-values). `results/analysis/hypothesis_tests.json` exists.

---

## 2. Available Data

All data is already on disk. **Do not re-run any experiments.** The analysis script only reads existing files.

### 2.1 Data inventory

| Condition | Track | Runs | Experiments/run | Location |
|-----------|-------|------|-----------------|----------|
| agent | SMILES | 5 (run_1–5) | 100 each | `results/smiles/run_{1..5}/` |
| agent | protein | 3 (run_1–3) | 100 each | `results/protein/run_{1..3}/` |
| agent | NLP | 5 (run_1–5) | 100 each | `results/nlp/run_{1..5}/` |
| random_nas | SMILES | 3 (run_1–3) | 100 each (run_1: 103) | `results/baselines/random_nas/smiles/run_{1..3}/` |
| random_nas | protein | 3 (run_1–3) | 100 each (run_1: 103) | `results/baselines/random_nas/protein/run_{1..3}/` |
| random_nas | NLP | 3 (run_1–3) | 100 each | `results/baselines/random_nas/nlp/run_{1..3}/` |
| hp_only | SMILES | 3 (run_1–3) | 100 each | `results/baselines/hp_only/smiles/run_{1..3}/` |
| hp_only | protein | 3 (run_1–3) | 100 each | `results/baselines/hp_only/protein/run_{1..3}/` |
| hp_only | NLP | 3 (run_1–3) | 100 each | `results/baselines/hp_only/nlp/run_{1..3}/` |
| fixed_default | SMILES | 1 | 1 | `results/baselines/fixed_default/smiles/` |
| fixed_default | protein | 1 | 1 | `results/baselines/fixed_default/protein/` |
| fixed_default | NLP | 1 | 1 | `results/baselines/fixed_default/nlp/` |

**Total: 3,106 experiments across 34 runs.**

### 2.2 Per-run file structure

Each run directory contains:

| File | Description | Used by |
|------|-------------|---------|
| `results.tsv` | TSV with columns: `commit`, `val_bpb`, `memory_gb`, `status`, `description` | H4 (trajectories), keep rates, all per-experiment analysis |
| `summary.json` | `best_experiment`, `best_val_bpb`, `num_crash`, `num_discard`, `num_experiments`, `num_keep` | Summary statistics |
| `diffs/exp*.diff` | Code diffs between consecutive experiments | H1 (architectural feature extraction) |
| `train_versions/exp*_candidate.py` | Full train.py source for each experiment | H1 (architecture parsing), H2 (domain knowledge) |
| `train_versions/exp*_keep.py` | Kept versions (only for status=keep) | H1 best architecture extraction |
| `logs/exp*.log` | Per-step training logs (loss, lr, MFU, throughput) | Training dynamics analysis |

### 2.3 Key constraints

- **Do NOT compare absolute val_bpb across tracks.** Tracks have different intrinsic entropy. Compare relative improvement from fixed_default floor within each track.
- **The unit of analysis for between-condition comparisons is the run** (n=3 to 5), not the individual experiment.
- **Within-run trajectories** (100 experiments each) can be analyzed as time series for within-run analyses.

---

## 3. Hypothesis Tests

### 3.1 H1: Domain-Specific Architecture Clustering

> Agent-discovered architectures cluster by domain when measured via architectural feature vectors in a permutation test on distance matrices.

#### 3.1.1 Architecture feature extraction

Parse each kept experiment's `train_versions/exp*_keep.py` (or the final best experiment's candidate.py) to extract an architectural feature vector. Extract from the **best architecture per run** (13 architectures total: 5 SMILES + 3 protein + 5 NLP).

Feature vector dimensions (extract from train.py source code):

| Feature | Type | How to extract |
|---------|------|---------------|
| `depth` (num_layers) | int | `DEPTH` or `num_layers` variable |
| `model_dim` | int | `base_dim * ASPECT_RATIO` or direct dim |
| `num_heads` | int | `model_dim // HEAD_DIM` or explicit |
| `head_dim` | int | `HEAD_DIM` variable |
| `ffn_ratio` | float | FFN hidden dim / model_dim |
| `activation` | categorical | activation function class name |
| `attention_type` | categorical | full / sliding_window / linear / other |
| `window_size` | int or None | if sliding window |
| `normalization` | categorical | RMSNorm / LayerNorm / other |
| `optimizer` | categorical | Muon / AdamW / MuonAdamW / other |
| `learning_rate` | float | peak LR |
| `weight_decay` | float | weight_decay param |
| `dropout` | float | dropout rate (0 if not present) |
| `batch_size` | int | DEVICE_BATCH_SIZE or TOTAL_BATCH_SIZE |
| `warmup_steps` | int | warmup steps |

For categorical features, one-hot encode them. For numerical features, standardize (z-score) before computing distances.

#### 3.1.2 Architectural distance matrix

- Compute pairwise **Gower distance** (handles mixed categorical + numerical features) between all 13 best architectures.
- Alternative: use **Hamming distance** on a discretized/binned version of the feature vector.

#### 3.1.3 Permutation test

- **Null hypothesis:** Track labels (SMILES, protein, NLP) are exchangeable — cross-track distances are not greater than within-track distances.
- **Test statistic:** Ratio of mean cross-track distance to mean within-track distance.
- **Procedure:** Permute track labels 10,000 times. Compute test statistic each time. p-value = fraction of permutations where test statistic >= observed.
- **Report:** p-value, observed ratio, null distribution histogram.

#### 3.1.4 Supplementary: Bayesian hierarchical model

- Fit a Bayesian model: `feature_j ~ Normal(μ_track, σ_track)` for each numeric feature j, with track-level random effects.
- Use PyMC or NumPyro.
- Report posterior distributions of track-level means.
- Compute ROPE: define practical equivalence as ±1 standard deviation of the pooled feature distribution. Report probability that track differences fall within vs. outside ROPE.
- Report Bayes factors for track effect ≠ 0 for each feature.

#### 3.1.5 Output

```
results/analysis/h1_architecture_features.json    — 13 feature vectors
results/analysis/h1_distance_matrix.json           — 13×13 pairwise distances
results/analysis/h1_permutation_test.json          — p-value, observed ratio, null distribution
results/analysis/h1_bayesian_posterior.json         — posterior summaries per feature per track
figures/h1_distance_heatmap.png                     — annotated heatmap of distance matrix
figures/h1_permutation_null.png                     — null distribution with observed value marked
figures/h1_architecture_pca.png                     �� PCA/t-SNE of architectures colored by track
```

---

### 3.2 H2: Rediscovery of Domain Knowledge

> Agent modifications correlate with known molecular modeling principles.

#### 3.2.1 Known techniques checklist

| Technique | What to look for in code diffs | Search pattern |
|-----------|-------------------------------|---------------|
| Local/sliding attention for bonded atoms | Reduced attention window, sliding window patterns | `window`, `local_attn`, `sliding` |
| Smaller embedding dim for tiny vocab | Embedding dim < model_dim, decoupled embedding | `embed_dim`, `vocab_size.*embed`, `n_embd` reduced |
| Positional encoding changes | Custom PE, rotary PE adjustments, relative PE | `RoPE`, `rotary`, `pos_emb`, `position` |
| Shallower/wider for short sequences | Decreased depth with increased width vs. NLP baseline | compare depth/width ratios SMILES vs NLP |
| Regularization for small data | Increased dropout, weight decay adjustments | `dropout`, `weight_decay` changes |

#### 3.2.2 Scoring procedure

For each of the 5 SMILES agent runs and each of the 5 known techniques:

1. Parse all diffs in `results/smiles/run_*/diffs/exp*.diff`
2. Classify each diff as matching/not-matching each technique (binary)
3. A technique is "partially matched" in a run if at least one experiment diff matches it AND the experiment was kept (status=keep)

**Success criterion:** ≥2 techniques partially matched in ≥2/3 SMILES runs (i.e., ≥2 of 5 runs).

#### 3.2.3 Statistical test

- Fisher's exact test: for each technique, 2×2 contingency table (matched/not × SMILES/NLP). Are molecular-relevant techniques more common in SMILES runs than NLP runs?
- Report as supplementary (this is primarily qualitative).

#### 3.2.4 Output

```
results/analysis/h2_technique_matrix.json    — 5 runs × 5 techniques binary matrix
results/analysis/h2_fisher_tests.json        — per-technique Fisher's exact test results
results/analysis/h2_diff_classifications.json — all classified diffs
figures/h2_technique_heatmap.png              — heatmap of technique × run presence
```

---

### 3.3 H3: Cross-Domain Transfer

> Note: H3 depends on SC-6 (transfer matrix), which is NOT YET DONE. This PRD covers only the **statistical analysis of transfer results once SC-6 data exists**. If SC-6 data is not yet available when this script runs, skip H3 tests and log a warning.

H3 has 4 sub-hypotheses. The statistical tests to apply once transfer data exists in `results/transfer/`:

| Sub-hypothesis | Test | Data needed |
|----------------|------|-------------|
| H3a: Asymmetric transfer | Paired t-test or Wilcoxon on degradation percentages | `results/transfer/matrix.json` |
| H3b: Layer specificity | Regression of degradation % vs. layer depth | `results/transfer/layer_freezing/` |
| H3c: Length dominance | Paired comparison: length-matched vs. unmatched transfer | `results/transfer/length_controlled/` |
| H3d: Innovation classification | Proportion test (30-40% universal vs. 60-70% domain-specific) | `results/transfer/innovation_labels.json` |

#### 3.3.1 Implementation

- Check if `results/transfer/matrix.json` exists. If not, skip all H3 tests, set `h3_status: "skipped_no_data"` in output.
- If data exists, run each sub-test and report.

#### 3.3.2 Output

```
results/analysis/h3_transfer_tests.json   — test results for H3a-d (or skip status)
figures/h3_transfer_heatmap.png            — 3×3 transfer matrix heatmap (if data exists)
figures/h3_layer_freezing.png              — degradation vs. layer depth curve (if data exists)
```

---

### 3.4 H4: Search Efficiency

> Agent achieves lower AUC-OC than baselines with 95% bootstrap CI excluding zero.

This is the **primary quantitative claim** and must be computed for all 3 tracks × 3 comparisons (agent vs random_nas, agent vs hp_only, hp_only vs fixed_default, random_nas vs fixed_default).

#### 3.4.1 Best-so-far curves

For each run, compute the **best-so-far** (cumulative minimum) trajectory of val_bpb across experiments 1–100, reading from `results.tsv`.

Discard crash rows (status=crash) — they don't update the best-so-far. Use the val_bpb column directly for keep/discard rows.

Implementation note: the best-so-far at experiment k = min(val_bpb[1], ..., val_bpb[k]), considering only non-crash experiments up to index k.

#### 3.4.2 AUC-OC (Area Under the Optimization Curve)

For each run, compute AUC using the trapezoidal rule over the best-so-far curve (experiments 1–100). Lower AUC = better cumulative search performance.

For fixed_default runs (1 experiment only): the "curve" is a flat line at val_bpb for all 100 positions → AUC = val_bpb × 100. This provides the ceiling reference.

#### 3.4.3 Per-track comparisons

For each track (SMILES, protein, NLP), compute:

**A. Agent vs Random NAS**

| Metric | Method | Report |
|--------|--------|--------|
| AUC-OC difference | Bootstrap 95% CI (10,000 resamples) | CI, mean difference, whether CI excludes 0 |
| AUC-OC comparison | Welch's t-test | t-statistic, p-value |
| AUC-OC comparison | Mann-Whitney U | U-statistic, p-value |
| Effect size | Cohen's d | d value, interpretation |
| Final best val_bpb | Welch's t-test | t-statistic, p-value |
| Final best val_bpb | Bootstrap 95% CI on difference | CI |
| Keep rate | Fisher's exact test (pooled keep/discard counts) | odds ratio, p-value |

**B. Agent vs HP-only**

Same battery of tests as (A).

**C. HP-only vs Fixed Default**

- Compute relative improvement: `(fixed_default_bpb - hp_only_best_bpb) / fixed_default_bpb` for each hp_only run.
- Report mean relative improvement ± bootstrap 95% CI.
- One-sample t-test: is mean relative improvement > 0?

**D. Random NAS vs Fixed Default**

Same as (C) but with random_nas runs.

**E. Agent vs Fixed Default**

Same as (C) but with agent runs.

#### 3.4.4 Bootstrap procedure

```
For each comparison (e.g., agent_AUC vs nas_AUC):
    observed_diff = mean(agent_AUC) - mean(nas_AUC)
    for i in 1..10000:
        resample_agent = sample_with_replacement(agent_AUC, n=len(agent_AUC))
        resample_nas = sample_with_replacement(nas_AUC, n=len(nas_AUC))
        boot_diff[i] = mean(resample_agent) - mean(resample_nas)
    CI_lower = percentile(boot_diff, 2.5)
    CI_upper = percentile(boot_diff, 97.5)
    p_value = 2 * min(mean(boot_diff >= 0), mean(boot_diff <= 0))
```

#### 3.4.5 Anytime performance table

For each track, compute the mean best-so-far at budgets N = {5, 10, 15, 20, 30, 50, 75, 100} for each condition. Report as table + figure.

#### 3.4.6 Time-to-threshold analysis

For each track:
1. Define thresholds at the 25th, 50th, 75th, and 90th percentile of the fixed_default val_bpb-to-best-seen range.
2. For each condition and run, find the first experiment that reaches each threshold.
3. Report median time-to-threshold and fraction of runs that ever reach it.

#### 3.4.7 Decomposition analysis (the key Phase 2 story)

This is the unique contribution of having all 4 conditions. For each track, decompose the total improvement from fixed_default to best agent:

```
total_improvement    = fixed_default - agent_best
hp_contribution      = fixed_default - hp_only_best
arch_contribution    = hp_only_best - agent_best   (may be negative!)
nas_contribution     = fixed_default - random_nas_best
guided_vs_random     = random_nas_best - agent_best
```

Report each as absolute bpb difference and as percentage of total improvement. Bootstrap 95% CIs on each component.

**This directly answers:** "Does architecture search add value beyond HP tuning?" If `arch_contribution` is negative (as in SMILES), HP tuning alone suffices. If positive (as in NLP), architecture search matters.

#### 3.4.8 Output

```
results/analysis/h4_auc_values.json              — AUC per run per condition per track
results/analysis/h4_bootstrap_results.json        — all bootstrap CIs and p-values
results/analysis/h4_frequentist_tests.json        — t-tests, Mann-Whitney, Fisher's exact
results/analysis/h4_anytime_performance.json      — best-so-far at each budget checkpoint
results/analysis/h4_time_to_threshold.json        — threshold analysis
results/analysis/h4_decomposition.json            — contribution decomposition per track
figures/h4_best_so_far_smiles.png                 — best-so-far curves, all conditions, SMILES
figures/h4_best_so_far_protein.png                — same for protein
figures/h4_best_so_far_nlp.png                    ��� same for NLP
figures/h4_auc_comparison.png                     — bar chart of AUC by condition, per track
figures/h4_decomposition.png                      — stacked bar showing HP vs arch contribution
figures/h4_keep_rate.png                          — cumulative keep rate curves by condition
figures/h4_anytime_table.png                      — anytime performance visualization
```

---

### 3.5 Multiple Comparison Correction

Apply **Holm-Bonferroni correction** across all frequentist tests.

Count all p-values produced:
- H1: 1 permutation test
- H2: up to 5 Fisher's exact tests
- H4: per track (3 tracks × ~6 tests per comparison × 4 comparisons) = ~72 tests, but group by family

**Family grouping:** Correct within logical families, not across all tests:
1. **H1 family:** permutation test (1 test)
2. **H2 family:** 5 Fisher's exact tests (5 tests)
3. **H4 per-track family:** For each track, correct across the 4 pairwise comparisons of AUC-OC (4 tests × 3 tracks = 3 families of 4)
4. **H4 decomposition family:** 3 tracks × 3 components = 9 tests

Report both raw and adjusted p-values.

#### Output

```
results/analysis/multiple_comparisons.json  — raw p-values, adjusted p-values, family grouping
```

---

## 4. Supplementary Analyses

These are not required for SC-8 but add value for the paper. Include them in the script.

### 4.1 Training dynamics (null result confirmation)

Re-run the training dynamics analysis from `scripts/analyze_training_dynamics.py` but now covering **all 34 runs** (not just SMILES). Confirm that agent vs baseline differences in convergence, stability, and MFU are negligible.

- Parse per-step training logs from `logs/exp*.log` for all runs.
- Compare convergence rate, stability (loss variance), and compute efficiency (MFU) between conditions.

#### Output

```
results/analysis/training_dynamics.json
figures/training_dynamics_convergence.png
figures/training_dynamics_stability.png
```

### 4.2 Distribution analysis

For each track × condition, compute:
- Best, median, mean, std, IQR of val_bpb across all experiments in condition
- Violin plots or box plots comparing conditions within each track

#### Output

```
results/analysis/distribution_stats.json
figures/distribution_violin_smiles.png
figures/distribution_violin_protein.png
figures/distribution_violin_nlp.png
```

### 4.3 Crash rate analysis

Per condition × track: crash count, crash rate, Fisher's exact test comparing agent crash rate vs each baseline.

#### Output

```
results/analysis/crash_rates.json
```

---

## 5. Implementation Specification

### 5.1 Script: `scripts/analyze_phase2.py`

Single Python script. No Jupyter notebooks. Must be runnable as:

```bash
cd /home/ubuntu/storage1/recursive-mol
python scripts/analyze_phase2.py
```

No command-line arguments needed — paths are hardcoded relative to the repo root.

### 5.2 Dependencies

Only use packages that are likely already installed or trivially installable:

| Package | Purpose | Install |
|---------|---------|---------|
| numpy | Array ops, bootstrap | likely installed |
| scipy | t-tests, Mann-Whitney, Fisher's exact, Spearman | likely installed |
| pandas | TSV loading | likely installed |
| matplotlib | All figures | likely installed |
| seaborn | Heatmaps, violin plots | likely installed |
| scikit-learn | PCA, Gower distance, StandardScaler | likely installed |
| pymc (v5) | Bayesian hierarchical model | `pip install pymc` if needed |

If PyMC is not installed or fails to import, the script should:
1. Print a warning: "PyMC not available; skipping Bayesian analysis for H1"
2. Skip the Bayesian model
3. Still produce all other outputs
4. Set `h1_bayesian_status: "skipped_pymc_not_available"` in output

The script must not fail if PyMC is missing. All other analyses use scipy/numpy only.

### 5.3 Architecture feature extraction

Parsing train.py files to extract architecture features is the most complex part. Strategy:

1. For each agent run, find the best experiment from `summary.json` → `best_experiment` field.
2. Read `train_versions/{best_experiment}_keep.py` (or `_candidate.py` if keep doesn't exist).
3. Use regex to extract feature values from Python source. Key patterns:
   - `DEPTH = <int>` or `num_layers = <int>`
   - `HEAD_DIM = <int>`
   - `ASPECT_RATIO = <int>` (model_dim = base_dim * ASPECT_RATIO)
   - Look for class definitions: `class Attention`, `class MLP`, etc.
   - Look for activation: `F.relu`, `F.gelu`, `ReluSquared`, etc.
   - Look for optimizer: `Muon`, `AdamW`
   - Look for normalization: `RMSNorm`, `LayerNorm`
4. For random_nas runs, extract from the best experiment's candidate.py.
5. For hp_only runs, architecture should be identical to default (verify this!).
6. For fixed_default, use the starting architecture config.

**Fallback:** If regex extraction fails for a file, log a warning and use `None` for that feature. Do not crash.

### 5.4 Output structure

All JSON output goes to `results/analysis/`. All figures go to `figures/`. Create both directories if they don't exist.

The **master output** is `results/analysis/hypothesis_tests.json`:

```json
{
  "generated_at": "2026-03-28T...",
  "sc8_status": "complete",
  "h1": {
    "permutation_test": {
      "p_value": 0.023,
      "p_value_adjusted": 0.023,
      "observed_ratio": 1.45,
      "n_permutations": 10000,
      "interpretation": "Architectures cluster by domain (p < 0.05)"
    },
    "bayesian": {
      "status": "complete",
      "rope_probability": {...},
      "bayes_factors": {...}
    }
  },
  "h2": {
    "techniques_matched": 3,
    "runs_with_matches": 4,
    "criterion_met": true,
    "fisher_tests": {...}
  },
  "h3": {
    "status": "skipped_no_data",
    "note": "Requires SC-6 transfer matrix. Run again after transfer experiments."
  },
  "h4": {
    "smiles": {
      "agent_vs_nas": {
        "auc_bootstrap_ci": [-0.82, -0.12],
        "auc_ci_excludes_zero": true,
        "auc_p_value": 0.037,
        "cohens_d": -1.91,
        "final_best_p_value": 0.049,
        "keep_rate_p_value": 0.004
      },
      "agent_vs_hp_only": {...},
      "decomposition": {
        "total_improvement": 0.0153,
        "hp_contribution": 0.0160,
        "hp_contribution_pct": 104.6,
        "arch_contribution": -0.0007,
        "arch_contribution_pct": -4.6,
        "interpretation": "HP tuning alone exceeds agent; architecture search adds no value on SMILES"
      }
    },
    "protein": {...},
    "nlp": {...}
  },
  "multiple_comparisons": {
    "method": "Holm-Bonferroni",
    "families": [...],
    "all_raw_p_values": [...],
    "all_adjusted_p_values": [...]
  }
}
```

### 5.5 Figure specifications

All figures should use:
- `figsize=(10, 6)` default, `(12, 8)` for multi-panel
- `dpi=150` for PNG output
- Font size 12 for labels, 10 for tick labels
- Color palette: use a consistent 4-color scheme for conditions:
  - agent: `#2196F3` (blue)
  - random_nas: `#FF9800` (orange)
  - hp_only: `#4CAF50` (green)
  - fixed_default: `#9E9E9E` (gray)
- Include significance stars on bar charts: `*` p<0.05, `**` p<0.01, `***` p<0.001
- Save both PNG and PDF versions of all figures

### 5.6 Logging

Print progress to stdout as the script runs:

```
[H1] Extracting architecture features from 13 runs...
[H1] Computing distance matrix...
[H1] Running permutation test (10,000 permutations)...
[H1] Permutation test: p=0.023, observed ratio=1.45
[H1] Fitting Bayesian hierarchical model...
[H1] Bayesian model complete.
...
[H4/SMILES] Computing best-so-far curves...
[H4/SMILES] Agent vs NAS: AUC bootstrap CI = [-0.82, -0.12], p=0.037
...
[DONE] All results written to results/analysis/hypothesis_tests.json
[DONE] 15 figures saved to figures/
```

---

## 6. Expected Results and Interpretation Guide

Based on the completed Phase 2 data, here is what the analysis should find. Use this to verify the script is working correctly.

### 6.1 H4 expected results by track

**SMILES:**
- Agent (mean best): ~0.5858, HP-only (mean best): ~0.5806, NAS (mean best): ~0.5914, Fixed default: 0.5961
- **Agent vs NAS:** Agent wins (lower bpb). Expect significant AUC difference (prior interim: p=0.037).
- **Agent vs HP-only:** HP-only wins! This is a real finding. Expect agent AUC to be worse than hp_only AUC.
- **Decomposition:** HP contribution > total improvement → architecture search hurts on SMILES.

**Protein:**
- Agent (mean best): ~3.9669, HP-only (mean best): ~3.9761, NAS (mean best): ~3.9704, Fixed default: 3.9767
- **Margins are tiny** (~0.3% relative spread). Significance will be hard to achieve.
- **Agent vs NAS:** Agent slightly better. May or may not be significant.
- **Agent vs HP-only:** Agent slightly better than hp_only. Marginal.

**NLP:**
- Agent (mean best): ~1.1228, HP-only (mean best): ~1.1470, NAS (mean best): ~1.1301, Fixed default: 1.1528
- **Agent vs NAS:** Agent clearly better. Should be significant.
- **Agent vs HP-only:** Agent clearly better. Architecture search adds substantial value on NLP.
- **Decomposition:** arch_contribution is positive and large → architecture search matters for NLP.

### 6.2 Narrative implications

The decomposition analysis is the most important new result. It tells a **nuanced story**:

1. **NLP:** Architecture search adds clear value beyond HP tuning → agent's full capability matters
2. **SMILES:** HP tuning alone suffices; architecture search adds no value → the domain is "easy" for HP-only
3. **Protein:** Margins too small to distinguish → domain may be constrained by other factors

This is more interesting than "agent always wins" and is a genuine empirical contribution.

---

## 7. Verification Checklist

After the script runs, verify:

- [ ] `results/analysis/hypothesis_tests.json` exists and is valid JSON
- [ ] All 4 hypotheses have entries (H3 may be "skipped")
- [ ] H1 permutation test p-value is between 0 and 1
- [ ] H4 has results for all 3 tracks
- [ ] H4 bootstrap CIs are properly formatted [lower, upper]
- [ ] H4 decomposition sums are consistent: hp_contribution + arch_contribution ≈ total_improvement
- [ ] Multiple comparison corrections are applied
- [ ] All expected figures exist in `figures/`
- [ ] No NaN or null values in any reported statistics (except H3 if skipped)
- [ ] Raw p-values and adjusted p-values are both present
- [ ] Script runs without errors (warnings about PyMC are OK)

---

## 8. Known Edge Cases

| Case | How to handle |
|------|---------------|
| random_nas protein run_1 has 103 experiments | Use all 103; AUC computed over first 100 for fair comparison |
| fixed_default has only 1 experiment | AUC = val_bpb × 100; it's a flat ceiling line |
| Agent NLP run_5 had 21 crashes | Include in analysis; crashes don't count as experiments for best-so-far |
| PyMC not installed | Skip Bayesian H1; all other analyses still run |
| H3 transfer data doesn't exist yet | Skip H3; set status to "skipped_no_data" |
| Architecture feature extraction fails for a run | Log warning; exclude that run from H1 distance matrix; proceed with remaining runs |
| hp_only architecture differs from default | This would be a bug — log a warning; the architecture should be identical |

---

## 9. Relationship to Existing Analysis

The interim analysis in `docs/analysis-sample-efficiency-interim-20260316.md` and the script `scripts/analyze_training_dynamics.py` covered SMILES-only agent vs random_nas. The new script **supersedes** that analysis by:

1. Covering all 3 tracks (not just SMILES)
2. Including hp_only and fixed_default conditions
3. Adding the decomposition analysis (the key new insight)
4. Adding H1 and H2 tests (not previously computed)
5. Adding multiple comparison correction
6. Producing paper-ready figures

The old script and interim doc remain as historical records but should not be cited in the paper. The new `scripts/analyze_phase2.py` is the authoritative analysis.

---

## 10. Timeline

This is a single-session task. Expected runtime:

| Step | Time |
|------|------|
| Data loading and validation | ~1 min |
| H1: Feature extraction + permutation test | ~5 min |
| H1: Bayesian model (if PyMC available) | ~10 min |
| H2: Diff classification | ~5 min |
| H4: All curves, bootstrap, tests | ~10 min |
| Figure generation | ~5 min |
| **Total** | **~30-40 min** |

No GPU needed. CPU-only analysis.

---

*PRD version 1.0 — March 28, 2026*
*Derived from: PRD-recursive-mol.md (SC-8, C5.3-C5.6), stress-test-experimental-design-audit.md (Section 1)*
