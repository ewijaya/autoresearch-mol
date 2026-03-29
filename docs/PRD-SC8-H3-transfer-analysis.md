# PRD: SC-8 H3 Transfer Analysis (Completion)

**Project:** recursive-mol
**Criterion:** SC-8 H3 — Fill in the skipped H3 transfer hypothesis tests
**Author:** Rex
**Date:** 2026-03-29
**Status:** READY TO EXECUTE
**Depends on:** SC-6 complete (`results/transfer/` populated), SC-8 complete (all other hypotheses done)
**Blocks:** Paper writing (H3 results needed for Section 5)

---

## 1. Objective

The SC-8 analysis script (`scripts/analyze_phase2.py`) was run before SC-6 transfer data existed. The H3 section in `results/analysis/hypothesis_tests.json` currently reads:

```json
"h3": {
    "status": "skipped_no_data",
    "note": "Requires SC-6 transfer matrix. Run again after transfer experiments."
}
```

SC-6 is now complete. All transfer data exists:

- `results/transfer/matrix.json` — 3x3 cross-domain transfer matrix (27 runs, 3 replicates each)
- `results/transfer/layer_freezing.json` — 6 cross-domain pairs x 3 freeze levels
- `results/transfer/length_controlled.json` — 3 worst-performing pairs with matched sequence lengths
- `results/transfer/innovation_classification.json` — 41 innovations classified

**Task:** Implement the H3 analysis in `scripts/analyze_phase2.py` by replacing the stub in `analyze_h3()`, then re-run the script. All existing H1, H2, H4 results must be preserved exactly — only the H3 section of `hypothesis_tests.json` should change.

---

## 2. Available Data

### 2.1 Transfer matrix (`results/transfer/matrix.json`)

A 3x3 matrix of val_bpb values. Each cell has `mean`, `std`, and `runs` (3 replicates with seeds 42, 137, 2026).

Structure:
```
matrix[arch_source][data_target] = {mean, std, runs: [r1, r2, r3]}
```

Where `arch_source` and `data_target` are each one of: `smiles_arch`, `protein_arch`, `nlp_arch` (for source) and `smiles_data`, `protein_data`, `nlp_data` (for target).

Also includes:
- `degradation_matrix[arch][data] = {pct_degradation, reference_bpb}` — pre-computed degradation percentages
- `baseline_bpbs = {smiles: 0.59607, protein: 3.976676, nlp: 1.152764}` — fixed_default floor per track

**Key finding from the data:** Degradation is near-zero across all cross-domain pairs:

| Architecture → Data | SMILES | Protein | NLP |
|---------------------|--------|---------|-----|
| SMILES arch | identity | -0.08% | -0.02% |
| Protein arch | -0.71% | identity | +0.80% |
| NLP arch | +0.05% | -0.15% | identity |

### 2.2 Layer freezing (`results/transfer/layer_freezing.json`)

6 cross-domain pairs, each with 3 freeze levels (1, 3, 5 layers frozen out of 6 total).

Structure per pair:
```json
{
  "arch_source": "smiles",
  "data_target": "protein",
  "freeze_levels": [
    {"frozen_layers": 1, "val_bpb": 3.968894},
    {"frozen_layers": 3, "val_bpb": 3.972734},
    {"frozen_layers": 5, "val_bpb": 3.978239}
  ],
  "native_baseline": 3.976676,
  "no_freeze_baseline": 3.973496
}
```

### 2.3 Length-controlled transfer (`results/transfer/length_controlled.json`)

3 pairs tested with matched sequence lengths:

| Pair | Unmatched degradation | Matched degradation | Reduction | H3c met? |
|------|----------------------|--------------------|-----------| ---------|
| protein→nlp (seq=512) | +0.80% | +4.35% | -447% (worse!) | No |
| nlp→smiles (seq=256) | +0.05% | -0.02% | +144% | Yes |
| smiles→nlp (seq=256) | -0.02% | +8.01% | null (baseline was negative) | No |

### 2.4 Innovation classification (`results/transfer/innovation_classification.json`)

41 total innovations, **all classified as universal** (0 domain-specific). This is a surprising null/negative result for H3d.

---

## 3. H3 Sub-Hypothesis Tests

### 3.1 H3a: Asymmetric Transfer

> Protein→SMILES transfer degrades < 15%; SMILES→Protein degrades > 15%.

**Test:** Compare degradation percentages from the transfer matrix.

**Procedure:**
1. Extract the 6 off-diagonal degradation values from `degradation_matrix`
2. For each directional pair (A→B vs B→A), test whether degradation is asymmetric
3. **Paired Wilcoxon signed-rank test** on the 3 directional pairs: is |degradation(A→B)| ≠ |degradation(B→A)|?
4. Since n=3 pairs is very small, also report the raw values and interpret qualitatively

**Specific sub-tests:**
- protein→smiles (-0.71%) vs smiles→protein (-0.08%): is this difference significant across replicates?
- Use the 3 individual replicate val_bpb values from `matrix.json` for each cell
- Welch's t-test on the 3 replicates of protein_arch/smiles_data vs smiles_arch/protein_data

**Expected finding:** Degradation is near-zero in all directions, so asymmetry is unlikely to be significant. The original H3a prediction (protein→SMILES < 15%, SMILES→protein > 15%) is **not supported** — both are < 1%. Report as negative result.

**Output fields:**
```json
"h3a": {
    "status": "complete",
    "prediction_met": false,
    "protein_to_smiles_degradation_pct": -0.71,
    "smiles_to_protein_degradation_pct": -0.08,
    "wilcoxon_p_value": ...,
    "interpretation": "No significant asymmetry; all cross-domain degradation < 1%"
}
```

### 3.2 H3b: Layer Specificity

> Early transformer layers transfer with < 5% val_bpb degradation; late layers degrade > 10%.

**Test:** Linear regression of degradation vs. number of frozen layers.

**Procedure:**
1. For each of the 6 cross-domain pairs, compute degradation at each freeze level relative to the no-freeze baseline:
   ```
   degradation[pair, level] = (freeze_val_bpb - no_freeze_val_bpb) / no_freeze_val_bpb * 100
   ```
2. Pool all 6 pairs × 3 levels = 18 data points
3. Fit linear regression: `degradation ~ frozen_layers`
4. Report slope, R², p-value for slope ≠ 0
5. Also test the specific thresholds: is degradation at freeze=1 < 5%? Is degradation at freeze=5 > 10%?

**Expected finding from the data:**

| Pair | Freeze 1 | Freeze 3 | Freeze 5 | Trend |
|------|----------|----------|----------|-------|
| smiles→protein | -0.12% | -0.02% | +0.12% | increasing |
| smiles→nlp | -0.07% | +1.00% | +2.83% | increasing |
| protein→smiles | +0.58% | +1.02% | +2.32% | increasing |
| protein→nlp | +0.64% | +1.64% | +3.92% | increasing |
| nlp→smiles | +0.08% | +0.55% | +16.12% | increasing (steep jump at 5) |
| nlp→protein | +0.03% | +0.08% | +0.72% | increasing |

The trend is consistently monotonically increasing (early layers transfer better than late layers). H3b prediction is **partially supported**: early layers degrade < 5% (yes, all freeze=1 values are < 1%), but late layers rarely exceed 10% (only nlp→smiles at freeze=5 = 16.1%).

**Output fields:**
```json
"h3b": {
    "status": "complete",
    "regression_slope": ...,
    "regression_r_squared": ...,
    "regression_p_value": ...,
    "early_layers_under_5pct": true,
    "late_layers_over_10pct": false,
    "prediction_met": "partial",
    "per_pair_degradation": {...},
    "interpretation": "Monotonic degradation with frozen layers confirmed; early layers universally transferable; late layer degradation mostly < 10% except nlp→smiles"
}
```

### 3.3 H3c: Length Dominance

> Length-matched transfer experiments show > 50% degradation reduction.

**Test:** Compare degradation with and without length matching.

**Procedure:**
1. For each of the 3 length-controlled pairs, compute:
   ```
   reduction_pct = (unmatched_degradation - matched_degradation) / unmatched_degradation * 100
   ```
   (Already pre-computed in `length_controlled.json` as `degradation_reduction_pct`)
2. Test whether mean reduction > 50% using one-sample t-test (n=3 is very small)
3. Report individual pair results

**Expected finding from the data:**
- protein→nlp: reduction = -447% (length matching made it **much worse**)
- nlp→smiles: reduction = +144% (length matching helped)
- smiles→nlp: reduction = null (baseline was already negative)

H3c is **not supported**. Length matching does not consistently reduce degradation. In fact, for protein→nlp, matching to 512 tokens made things dramatically worse (the NLP model is designed for 2048 context; truncating to 512 cripples it).

**Output fields:**
```json
"h3c": {
    "status": "complete",
    "prediction_met": false,
    "pairs": [
        {"pair": "protein→nlp", "reduction_pct": -446.5, "note": "length matching worsened transfer"},
        {"pair": "nlp→smiles", "reduction_pct": 144.2, "note": "length matching helped"},
        {"pair": "smiles→nlp", "reduction_pct": null, "note": "baseline degradation was negative"}
    ],
    "mean_reduction_pct": null,
    "t_test_p_value": null,
    "interpretation": "Length matching does not consistently reduce transfer degradation. Truncating NLP sequences to shorter lengths actively hurts, suggesting context window is a critical architectural property, not just a data artifact."
}
```

### 3.4 H3d: Innovation Classification

> Innovations classified as 30-40% universal vs 60-70% domain-specific.

**Test:** Binomial proportion test.

**Procedure:**
1. Read `results/transfer/innovation_classification.json`
2. Count universal vs domain-specific innovations
3. Binomial test: is the universal proportion significantly different from the predicted 35% (midpoint of 30-40%)?
4. Also report 95% CI on the universal proportion

**Expected finding from the data:** 41/41 innovations are universal (100%). The prediction of 60-70% domain-specific is **strongly rejected**.

**Output fields:**
```json
"h3d": {
    "status": "complete",
    "prediction_met": false,
    "total_innovations": 41,
    "universal_count": 41,
    "domain_specific_count": 0,
    "universal_pct": 100.0,
    "binomial_test_p_value": ...,
    "predicted_universal_pct": 35.0,
    "ci_95": [91.4, 100.0],
    "interpretation": "All 41 innovations classified as universal. The prediction of 60-70% domain-specific innovations is strongly rejected. At this model scale, architectural innovations transfer freely across molecular and NLP domains."
}
```

---

## 4. Implementation

### 4.1 What to modify

Edit `scripts/analyze_phase2.py`. Replace the `analyze_h3()` function (currently a stub that returns `skipped_no_data` or `not_implemented`) with a full implementation.

### 4.2 The existing stub (to replace)

```python
def analyze_h3() -> dict[str, Any]:
    transfer_matrix = RESULTS_DIR / "transfer" / "matrix.json"
    if not transfer_matrix.exists():
        payload = {
            "status": "skipped_no_data",
            "note": "Requires SC-6 transfer matrix. Run again after transfer experiments.",
        }
        write_json(ANALYSIS_DIR / "h3_transfer_tests.json", payload)
        return {"master": payload, "payload": payload}
    payload = {
        "status": "not_implemented",
        "note": "Transfer data exists but H3 analysis was not required in this workspace snapshot.",
    }
    write_json(ANALYSIS_DIR / "h3_transfer_tests.json", payload)
    return {"master": payload, "payload": payload}
```

### 4.3 New implementation requirements

The new `analyze_h3()` must:

1. Read all 4 transfer data files:
   - `results/transfer/matrix.json`
   - `results/transfer/layer_freezing.json`
   - `results/transfer/length_controlled.json`
   - `results/transfer/innovation_classification.json`
2. If any file is missing, skip only the sub-test that needs it (don't skip all of H3)
3. Run H3a, H3b, H3c, H3d tests as specified in Section 3
4. Return a dict that gets inserted into `hypothesis_tests.json` under the `h3` key
5. Save detailed results to `results/analysis/h3_transfer_tests.json`
6. Generate 2 figures:
   - `figures/h3_transfer_heatmap.png` — 3×3 degradation matrix heatmap with values annotated
   - `figures/h3_layer_freezing.png` — 6 lines (one per cross-domain pair) showing degradation vs frozen layers

### 4.4 Statistical functions needed

All available in scipy/numpy (no new dependencies):

| Test | scipy function |
|------|---------------|
| Wilcoxon signed-rank (H3a) | `scipy.stats.wilcoxon` |
| Welch's t-test (H3a replicate comparison) | `scipy.stats.ttest_ind(equal_var=False)` |
| Linear regression (H3b) | `scipy.stats.linregress` |
| One-sample t-test (H3c) | `scipy.stats.ttest_1samp` |
| Binomial test (H3d) | `scipy.stats.binomtest` |
| Binomial CI (H3d) | `scipy.stats.binomtest(...).proportion_ci()` |

### 4.5 Figure specifications

Follow the same style as existing figures in the script:
- `figsize=(10, 6)`, `dpi=150`
- Font size 12 for labels, 10 for tick labels
- Save both PNG and PDF

**`h3_transfer_heatmap.png`:**
- 3×3 heatmap with rows = architecture source, columns = data target
- Cell values = degradation percentage (annotated in cells)
- Color scale: diverging (green = negative/good, red = positive/bad), centered at 0
- Identity cells (diagonal) marked distinctly

**`h3_layer_freezing.png`:**
- X-axis: number of frozen layers (0, 1, 3, 5)
- Y-axis: val_bpb degradation (%) relative to no-freeze baseline
- 6 lines, one per cross-domain pair, with labels
- Horizontal dashed lines at 5% and 10% thresholds (H3b criteria)

### 4.6 Master output format

The `h3` section of `results/analysis/hypothesis_tests.json` should become:

```json
"h3": {
    "status": "complete",
    "h3a": {
        "status": "complete",
        "prediction_met": false,
        ...
    },
    "h3b": {
        "status": "complete",
        "prediction_met": "partial",
        ...
    },
    "h3c": {
        "status": "complete",
        "prediction_met": false,
        ...
    },
    "h3d": {
        "status": "complete",
        "prediction_met": false,
        ...
    },
    "overall_interpretation": "H3 predictions largely not supported. Cross-domain transfer shows near-zero degradation (<1% for most pairs), all innovations are universal, and length matching does not consistently help. The dominant finding is architectural universality at this model scale, not domain specificity."
}
```

---

## 5. Preserving Existing Results

**CRITICAL:** When re-running the script, all existing H1, H2, H4, and multiple_comparisons results must remain identical. The script should:

1. Not re-run H1, H2, H4 analyses (they are deterministic except for bootstrap, which uses a seed)
2. Only update the `h3` key in `hypothesis_tests.json`
3. Not regenerate existing figures (or regenerate them identically)

**Preferred approach:** Modify only the `analyze_h3()` function. Re-run the full script. Since H1, H2, H4 use the same data and (presumably) fixed random seeds, outputs should be identical.

**Alternative approach (safer):** Read the existing `hypothesis_tests.json`, run only H3 analysis, update the `h3` key, and write back. This guarantees H1/H2/H4 are untouched.

Use the alternative approach if the full re-run produces any differences in H1/H2/H4 results.

---

## 6. Verification Checklist

After running:

- [ ] `results/analysis/hypothesis_tests.json` has `h3.status: "complete"` (not `skipped_no_data`)
- [ ] All 4 sub-hypotheses (h3a, h3b, h3c, h3d) have results
- [ ] H3a has Wilcoxon p-value and per-pair degradation values
- [ ] H3b has regression slope, R², p-value, and per-pair freeze-level degradation
- [ ] H3c has per-pair reduction percentages and interpretation
- [ ] H3d has binomial test p-value and 95% CI on universal proportion
- [ ] `figures/h3_transfer_heatmap.png` exists and shows 3×3 matrix
- [ ] `figures/h3_layer_freezing.png` exists and shows 6 degradation curves
- [ ] `results/analysis/h3_transfer_tests.json` exists with detailed results
- [ ] H1, H2, H4 results in `hypothesis_tests.json` are unchanged from before
- [ ] No NaN or null in reported statistics (except H3c mean where expected)

---

## 7. Expected Narrative

The H3 results tell a surprising but publishable story:

> **At this model scale (~8-10M parameters, 5-minute training), transformer architectural innovations transfer freely across molecular and NLP domains.** The predicted asymmetric transfer and domain-specific innovations did not materialize. All 41 agent-discovered innovations are universal. Cross-domain degradation is < 1% for most pairs. Early layers transfer with near-zero degradation; late layers show mild degradation that rarely exceeds 10%.
>
> This is consistent with the H1 result (architectures DO cluster by domain) but with an important nuance: while agents discover different architectures for different domains, the innovations themselves are not domain-locked — they improve performance broadly. The clustering may reflect optimization path dependence rather than fundamental domain requirements.
>
> The one exception is length-related: truncating NLP sequences to match molecular sequence lengths severely hurts performance, confirming that context window size is a genuine domain constraint, not an arbitrary hyperparameter.

---

*PRD version 1.0 — March 29, 2026*
*Derived from: PRD-SC8-statistical-analysis.md (Section 3.3), PRD-SC6-SC7-transfer-and-downstream.md (completed outputs)*
