# Phase Prompts for recursive-mol

Copy-paste each prompt to your AI agent when starting the corresponding phase.
Check off items as they complete. Do NOT start a phase until the previous gate passes.

---

## Phase 1: Infrastructure & Validation (Mar 9-16)

**AWS Instance:** `g5.xlarge` (A10G 24GB, spot ~$0.44/hr) — 1 instance is enough. All tasks are sequential builds + short validation runs.

**Estimated compute:** ~$15-20 (calibration study is the most expensive part: 20 archs × 2hr = 40 GPU-hrs)

### Prompt

```
You are working on the recursive-mol project (PRD: docs/PRD-recursive-mol.md).
Execute Phase 1: Infrastructure & Validation. Here are your tasks in order:

STEP 1 — Fork and fix autoresearch:
- Fork karpathy/autoresearch into this repo's src/ directory
- Replace ALL Flash Attention 3 (FA3) calls with torch.nn.functional.scaled_dot_product_attention
- Remove the `kernels` package dependency entirely
- For sliding window attention: try a custom causal mask first; fall back to full attention if too complex
- Verify: train.py runs without importing `kernels`

STEP 2 — Scale down the model:
- Set DEPTH=6, ASPECT_RATIO=48 (base_dim=288), HEAD_DIM=64 (model_dim=320, num_heads=5)
- Target: 8-12M parameters total
- Set MAX_SEQ_LEN=256 for SMILES, 512 for protein
- Set TOTAL_BATCH_SIZE=65536
- Verify: model fits in <12GB VRAM on A10G

STEP 3 — Implement prepare_smiles.py:
- Download ZINC-250K (250k_rndm_zinc_drugs_clean_3.csv)
- Implement SMILES enumeration using RDKit: 5 random SMILES per molecule
- Character-level tokenizer (~45 chars + 4 special tokens: PAD, BOS, EOS, UNK)
- Train/val split by MOLECULE (90/10), then enumerate — no molecule leakage
- Output: train/val .pkl files in data/smiles/
- Verify: total tokens ~62.5M, no molecule appears in both splits

STEP 4 — Implement prepare_protein.py:
- Download UniRef50 subset from ftp.uniprot.org
- Filter: length 50-500 residues, randomly sample 50K sequences
- Character-level tokenizer (20 amino acids + special tokens)
- 90/10 random split
- Output: train/val .pkl files in data/protein/
- Verify: ~12.5M total tokens

STEP 5 — Write program.md:
- Agent instructions for molecular architecture search
- Do NOT include molecular domain hints (we're testing if the agent discovers them)
- Encourage architectural changes (not just HP tuning)
- Keep the autoresearch loop structure: modify train.py → 5min train → measure val_bpb → keep/discard

STEP 6 — Run baseline validation:
- Train the unmodified starting architecture for 5 minutes on each of the 3 tracks (SMILES, protein, NLP)
- Verify val_bpb < 4.0 on SMILES
- Verify all 3 tracks produce valid val_bpb without errors

STEP 7 — Run calibration study:
- Implement calibration.py
- Design 20 random architecture variants (vary depth 3-8, width 128-512, heads 2-8, activation {ReLU, GELU, SiLU, ReluSquared}, attention {full, windowed})
- Train each for 5 minutes on SMILES → record val_bpb
- Train each for 2 hours on SMILES → record val_bpb
- Compute Spearman rank correlation between 5-min and 2-hr val_bpb
- Save results to results/calibration/
- DECISION GATE: rho > 0.7 → proceed. rho 0.4-0.7 → proceed with caution. rho < 0.4 → increase TIME_BUDGET to 15-30 min.

After completing all steps, report the status of each Checkpoint 1 criterion below.

FINALLY: Stop the EC2 instance by running: /home/ubuntu/bin/stopinstance
```

### Checkpoint 1 — GATE

- [ ] `python prepare_smiles.py` completes without error
- [ ] `python train.py` completes 5-minute training on SMILES data
- [ ] Baseline val_bpb < 4.0 (below random ~5.3)
- [ ] Proxy calibration Spearman rho > 0.5
- [ ] Protein `prepare_protein.py` functional
- [ ] All 3 tracks produce valid val_bpb on a single baseline run
- [ ] VRAM usage < 12GB on A10G
- [ ] Model parameter count in 8-12M range

**Kill condition:** FA3 fix fails and no workaround → switch to p4d.24xlarge or abandon

---

## Phase 2: Main Experiments (Mar 16-23)

**AWS Instance:** `g5.xlarge` × 3 (one per track, spot ~$0.44/hr each). Run agent sessions in parallel across instances to finish within 1 week.

**Estimated compute:** ~$85 (13 agent runs × ~13hrs × $0.44/hr + baseline runs starting)

### Prompt

```
You are working on the recursive-mol project (PRD: docs/PRD-recursive-mol.md).
Phase 1 is complete. Checkpoint 1 passed. Execute Phase 2: Main Experiments.

STEP 1 — Write program_hponly.md:
- Copy program.md but restrict instructions to: "Only modify hyperparameters (learning rate, batch size, dropout, weight decay, warmup steps, optimizer params). Do NOT change model architecture (no new layers, no attention pattern changes, no activation function changes, no model structure changes)."

STEP 2 — Implement random_nas.py:
- Script that generates random architecture configs by sampling: depth (3-8), width (128-512), heads (2-8), activation (ReLU/GELU/SiLU/ReluSquared), attention (full/windowed)
- Each config must produce a valid train.py that compiles and trains
- Output: modified train.py files saved per run

STEP 3 — Launch agent runs (13 total):
- Track A (SMILES): 5 independent agent sessions, each ~100 experiments
- Track B (Protein): 3 independent agent sessions, each ~100 experiments
- Track C (NLP): 5 independent agent sessions, each ~100 experiments
- Each session uses program.md, the autoresearch loop, TIME_BUDGET=300s
- Save all results to results/{smiles,protein,nlp}/run_{1..N}/
- Save every version of train.py the agent produces

STEP 4 — Launch Tier 1 baselines (in parallel where possible):
- Random NAS: 9 runs (3 tracks × 3 replicates), 100 random archs each
- HP-only agent: 9 runs (3 tracks × 3 replicates), using program_hponly.md
- Fixed default: 3 runs (1 per track), unmodified starting architecture
- Save to results/baselines/{random_nas,hp_only,fixed_default}/

STEP 5 — Early monitoring:
- After the first 2 SMILES agent runs complete, check code diffs
- Categorize changes as architectural vs. HP-only
- Flag if agent is making zero architectural changes (kill condition)

After all runs complete, report Checkpoint 2 criteria status.

FINALLY: Stop the EC2 instance by running: /home/ubuntu/bin/stopinstance
```

### Checkpoint 2 — GATE

- [ ] >= 10 of 13 agent runs complete with >= 80 experiments each
- [ ] >= 1 SMILES run shows >= 3 architectural modifications (not just HP)
- [ ] No systematic crashes or data pipeline failures
- [ ] All agent run results saved to results/{smiles,protein,nlp}/run_*/
- [ ] All train.py versions saved per run
- [ ] Baseline runs launched (may still be running)

**Kill condition:** Agent makes zero architectural changes across all 5 SMILES runs → pivot to workshop paper ("HP optimization study")

---

## Phase 3: Baselines & Initial Analysis (Mar 23-30)

**AWS Instance:** `g5.xlarge` × 2-3 (spot, to finish remaining baselines). Once baselines complete, analysis can run on a `c5.xlarge` (CPU-only, ~$0.17/hr) or locally — no GPU needed for statistical analysis.

**Estimated compute:** ~$50 (remaining baseline runs) + ~$2 (CPU analysis)

### Prompt

```
You are working on the recursive-mol project (PRD: docs/PRD-recursive-mol.md).
Phase 2 is complete. Checkpoint 2 passed. Execute Phase 3: Baselines & Initial Analysis.

STEP 1 — Complete all remaining baselines:
- Ensure all 21 Tier 1 baseline runs are finished:
  - 9 random NAS (3 tracks × 3 replicates)
  - 9 HP-only agent (3 tracks × 3 replicates)
  - 3 fixed default (1 per track)
- Verify each has results.tsv with >= 80 rows

STEP 2 — Implement categorize_diffs.py:
- Input: all agent run directories containing train.py versions
- For each modification, classify as:
  - Architectural: new layers, attention patterns, activation functions, model structure, normalization changes
  - Hyperparameter: learning rate, batch size, dropout, weight decay, warmup, optimizer params
- Output: per-run summary with % architectural vs % HP changes
- Save to results/analysis/modification_categories.json

STEP 3 — Build architectural feature vectors:
- For each final architecture (13 agent + 9 random NAS + 9 HP-only = 31 total), extract:
  - depth, width, heads, FFN ratio, activation type, attention variant, window size, normalization type, optimizer
- Encode as numerical feature vector
- Save to results/analysis/architecture_vectors.json

STEP 4 — Run H1 permutation test:
- Compute pairwise Hamming/edit distance between all architectural feature vectors
- Permutation test (10,000 permutations): do cross-track distances exceed within-track distances?
- Report p-value
- Save to results/analysis/h1_permutation_test.json

STEP 5 — Compute H4 AUC-OC:
- For each agent run and each random NAS run, compute the area under the optimization curve (val_bpb over experiment number)
- Compare agent AUC-OC vs. random NAS AUC-OC per track
- 95% bootstrap CI (10,000 resamples)
- Save to results/analysis/h4_auc_oc.json

STEP 6 — Agent behavior analysis:
- What changes does the agent try first across runs?
- Are there common strategies per track?
- What fraction of agent experiments improve val_bpb?
- Save qualitative summary to results/analysis/agent_behavior.md

After completing all steps, report Checkpoint 3 criteria status.

FINALLY: Stop the EC2 instance by running: /home/ubuntu/bin/stopinstance
```

### Checkpoint 3 — GATE

- [ ] All 21 baselines complete with valid results.tsv
- [ ] Modification categorization complete: % architectural vs. % HP per run
- [ ] Architectural feature vectors extracted for all 31 runs
- [ ] H1 permutation test p-value computed
- [ ] H4 AUC-OC comparison computed with bootstrap CI
- [ ] At least one qualitative architectural difference between SMILES and NLP tracks visible
- [ ] Agent behavior summary written

**Kill condition:** No visible architectural difference between any tracks → pivot to "universal transformer" framing

---

## Phase 4: Transfer & Downstream Evaluation (Mar 30 - Apr 6)

**AWS Instance:** `g5.xlarge` × 2 (spot). Transfer experiments and MoleculeNet fine-tuning both need GPU. Run transfer matrix on one instance, MoleculeNet on another.

**Estimated compute:** ~$40 (transfer: 36 runs × 5min + layer freezing: 18 runs × 5min + MoleculeNet: 27 runs × ~15min each)

### Prompt

```
You are working on the recursive-mol project (PRD: docs/PRD-recursive-mol.md).
Phase 3 is complete. Checkpoint 3 passed. Execute Phase 4: Transfer & Downstream Evaluation.

STEP 1 — Implement transfer_eval.py:
- Takes a trained architecture (train.py) and a dataset track
- Trains from scratch on the target dataset for 5 minutes
- Reports val_bpb
- Supports layer freezing (freeze first N layers)

STEP 2 — Run cross-domain transfer matrix:
- Take the best architecture from each track (SMILES, protein, NLP) — 3 architectures
- Evaluate each on all 3 datasets — 9 conditions
- 3 replicates each — 27 runs total (but quick: 5 min each)
- Plus the identity condition (arch on its own track) already exists from Phase 2
- Save 3×3 matrix (mean ± std) to results/transfer/matrix.json

STEP 3 — Layer freezing experiments (H3b):
- For each cross-domain pair (6 pairs), progressively freeze layers 1, 1-2, 1-3, etc.
- Measure val_bpb degradation at each freeze level
- 18 runs total (6 pairs × 3 freeze levels)
- Save to results/transfer/layer_freezing.json

STEP 4 — Length-controlled transfer (H3c):
- Re-run the worst-performing transfer pairs but with matched sequence lengths
- Truncate the longer sequences to match the shorter track's distribution
- Measure if val_bpb degradation drops by > 50%
- Save to results/transfer/length_controlled.json

STEP 5 — Implement evaluate_downstream.py:
- Fine-tune a pretrained architecture on MoleculeNet tasks using DeepChem or PyTorch Geometric
- Tasks: BBBP, HIV, BACE
- Report ROC-AUC per task
- Support multiple architectures as input

STEP 6 — Run MoleculeNet fine-tuning:
- Top 3 SMILES architectures (by val_bpb) × 3 tasks × 3 replicates = 27 runs
- Report ROC-AUC (mean ± std) for each architecture × task
- Compute Spearman rank correlation between val_bpb ranking and MoleculeNet ROC-AUC ranking
- Save to results/moleculenet/scores.json

STEP 7 — Generation metrics:
- Load best SMILES architecture checkpoint
- Generate 10K SMILES strings autoregressively
- Compute: validity (% parseable by RDKit), uniqueness, novelty (vs training set), FCD
- Save to results/moleculenet/generation_metrics.json

STEP 8 — Classify innovations as molecular-specific vs. universal (H3d):
- Using the 3×3 transfer matrix from STEP 2, identify which architectural innovations
  transfer well across domains (universal) vs. which degrade significantly (domain-specific)
- For each distinct architectural change discovered in Phase 2, label it as:
  - "universal": cross-domain val_bpb degradation < 10%
  - "domain-specific": cross-domain val_bpb degradation >= 10%
- Compute the percentage split (pitch target: ~30-40% universal, ~60-70% domain-specific)
- Save classification to results/transfer/innovation_classification.json

After completing all steps, report Checkpoint 4 criteria status.

FINALLY: Stop the EC2 instance by running: /home/ubuntu/bin/stopinstance
```

### Checkpoint 4 — GATE

- [ ] Transfer matrix complete (3×3 with mean ± std) with innovations classified as molecular-specific vs. universal
- [ ] Layer freezing curves computed for all 6 cross-domain pairs
- [ ] Length-controlled transfer results available
- [ ] Innovation classification saved to results/transfer/innovation_classification.json
- [ ] MoleculeNet ROC-AUC computed for all 9 conditions (3 archs × 3 tasks)
- [ ] val_bpb vs ROC-AUC Spearman correlation computed
- [ ] Generation metrics (validity, uniqueness, novelty, FCD) computed
- [ ] All results saved to results/transfer/ and results/moleculenet/

**Kill condition:** MoleculeNet ROC-AUC shows zero correlation with val_bpb → report as limitation, do not anchor claims on downstream performance

---

## Phase 5: Statistical Analysis & Writing (Apr 6-20)

**AWS Instance:** `c5.2xlarge` (CPU-only, ~$0.34/hr) for statistical analysis. No GPU needed. PyMC/Stan runs on CPU. Paper writing needs no compute.

**Estimated compute:** ~$5 (Bayesian models + bootstrap: ~15 CPU-hours)

### Prompt

```
You are working on the recursive-mol project (PRD: docs/PRD-recursive-mol.md).
Phase 4 is complete. Checkpoint 4 passed. Execute Phase 5: Statistical Analysis & Writing.

STEP 1 — Implement analyze.py:
- Unified statistical analysis script that reads all results from results/ and produces:

  a) Bayesian hierarchical model for H1 (use PyMC):
     - Model: track-level random effects on architectural feature vectors
     - ROPE: ±0.05 relative bpb improvement
     - Output: posterior distributions, Bayes factors

  b) Permutation tests for H1 (already computed in Phase 3, refine here):
     - Finalize with all data including baselines

  c) Bootstrap CIs for H4:
     - AUC-OC difference (agent vs random NAS) per track
     - 10,000 bootstrap resamples, 95% CI

  d) H3 sub-hypothesis tests:
     - H3a (asymmetry): compare protein→SMILES vs SMILES→protein degradation
     - H3b (layer specificity): test if early layers degrade < 5%, late layers > 10%
     - H3c (length dominance): test if degradation drops > 50% with length matching
     - H3d (innovation classification): classify each architectural change as universal vs domain-specific via cross-domain ablation

  e) Holm-Bonferroni correction across all 12+ tests

  f) H2 post-hoc alignment: score agent modifications against 5 known molecular techniques

- Save all results to results/analysis/hypothesis_tests.json

STEP 2 — Generate all paper figures:
- Figure 1: Architecture evolution plots (val_bpb over experiment number, one per track, with agent + baselines)
- Figure 2: Architectural clustering visualization (PCA/t-SNE of feature vectors, colored by track)
- Figure 3: Transfer heatmap (3×3 matrix with color intensity) annotated with molecular-specific vs. universal innovation labels
- Figure 4: Layer freezing curves (val_bpb degradation vs frozen layers, per cross-domain pair)
- Figure 5: Calibration scatter plot (5-min vs 2-hr val_bpb with trend line and rho)
- Figure 6: Attention pattern visualizations from best architecture per track
- Figure 7: Agent modification timeline (what types of changes, when)
- Save all figures as publication-quality PDFs to figures/

STEP 3 — Write the paper (NeurIPS format, LaTeX):
- Use the abstract template from PRD Section 12.2, filling in actual results
- Title: "Autonomous Discovery of Domain-Specific Transformer Designs for Molecular Sequences" (or option 2/3 from PRD Section 12.1 based on results)
- Structure: Abstract, 1-Introduction, 2-Background, 3-Method, 4-Experimental Setup, 5-Results, 6-Analysis, 7-Discussion, 8-Conclusion
- Include all 15 must-cite references (PRD Section 12.3)
- Frame as empirical study (Change 6): lead with scientific question, not method
- Do NOT claim RSR as primary contribution — define it in Section 2 as internal concept
- Do NOT compare absolute val_bpb across tracks (Change 5, C5.7)
- Report relative improvement from baseline within each track with CIs
- Paper draft v1: all sections with placeholder figures
- Paper draft v2: integrate actual figures and complete results tables

Save paper to paper/main.tex with sections in paper/sections/.

FINALLY: Stop the EC2 instance by running: /home/ubuntu/bin/stopinstance
```

### Phase 5 Checklist (no formal gate)

- [ ] analyze.py implemented and runs end-to-end
- [ ] All 4 hypotheses (H1-H4) have quantified evidence
- [ ] Holm-Bonferroni correction applied to all tests
- [ ] All 7+ figures generated as publication-quality PDFs
- [ ] Paper draft v1 complete (all sections)
- [ ] Paper draft v2 complete (figures + results integrated)
- [ ] All 15 must-cite references included
- [ ] Framing compliance: empirical study, not method paper

---

## Phase 6: Finalize & Preprint (Apr 20-27)

**AWS Instance:** None needed. All work is writing, review, and repo cleanup. Use local machine or a `t3.medium` (~$0.04/hr) for LaTeX compilation if needed.

**Estimated compute:** ~$1

### Prompt

```
You are working on the recursive-mol project (PRD: docs/PRD-recursive-mol.md).
Phase 5 is complete. Execute Phase 6: Finalize & Preprint.

STEP 1 — Review paper for compliance:
- Verify framing (Change 6): title does NOT contain "Recursive Self-Refinement"; abstract leads with scientific question; RSR defined in Section 2 only
- Verify all 15 must-cite references present (PRD Section 12.3):
  Karpathy, EvoPrompting, FunSearch, LM-Searcher, Self-Refine, IMPROVE, AI Scientist, MoleculeNet, MoLFormer, OpenELM, LLMatic, SMILES (Weininger), NAS survey (Elsken), ESM-2, Uni-Mol
- Verify no cross-track absolute val_bpb comparisons
- Verify all statistical claims have CIs or Bayes factors
- Verify autoresearch-robotics acknowledged as concurrent work

STEP 2 — Prepare supplementary materials:
- Full agent interaction logs (anonymized if needed)
- Complete code diffs for all 13 agent runs
- Architecture evolution animations (if feasible) or detailed per-experiment tables
- Extended statistical tables (all permutation test details, all bootstrap samples)
- Save to paper/supplementary/

STEP 3 — Clean up the repo for public release:
- README.md with: project description, installation, data download, reproduction steps
- requirements.txt with pinned versions
- All scripts have docstrings and usage instructions
- data/ directory has download scripts (not raw data)
- Remove any hardcoded paths, API keys, or personal references
- Verify: a fresh clone can reproduce the pipeline

STEP 4 — Compile final paper:
- LaTeX compiles without errors
- All figures render correctly
- Page count within NeurIPS limits (9 pages + unlimited references/appendix)
- Generate paper/main.pdf

STEP 5 — Post arXiv preprint:
- Categories: cs.LG (primary) + q-bio.QM (cross-list)
- Verify PDF renders correctly on arXiv
- Record the arXiv URL

Deadline: arXiv preprint MUST be live by April 27.

FINALLY: Stop the EC2 instance by running: /home/ubuntu/bin/stopinstance
```

### Phase 6 Checklist

- [ ] Paper passes framing compliance check
- [ ] All 15 must-cite references verified present
- [ ] Supplementary materials prepared
- [ ] Repo cleaned for public release (README, requirements.txt, no secrets)
- [ ] LaTeX compiles, PDF renders, within page limits
- [ ] arXiv preprint posted (cs.LG + q-bio.QM)
- [ ] arXiv URL recorded: _______________

---

## Phase 7: NeurIPS Submission (Apr 27 - May 15)

**AWS Instance:** None. Possibly a `g5.xlarge` for 1-2 days if you need to re-run any experiment based on feedback (~$10 contingency).

**Estimated compute:** ~$0-10

### Prompt

```
You are working on the recursive-mol project (PRD: docs/PRD-recursive-mol.md).
Phase 6 is complete. arXiv preprint is live. Execute Phase 7: NeurIPS Submission.

STEP 1 — Monitor preprint feedback (ongoing):
- Check arXiv comments
- Search Twitter/X for mentions of the paper or competing work
- Search arXiv weekly for competing molecular autoresearch preprints

STEP 2 — Incorporate feedback:
- Address any substantive critiques from early readers
- If a competing preprint appears, add it to related work and differentiate

STEP 3 — Final polish:
- Proofread entire paper
- Verify all numbers in text match results tables and figures
- Check figure quality (300 DPI minimum, readable at print size)
- Verify supplementary materials are complete and referenced

STEP 4 — Submit to NeurIPS 2026:
- Main conference track (ML)
- Upload paper + supplementary
- Write author response template (anticipate reviewer questions from stress-test reviews in docs/)

STEP 5 — Workshop fallback:
- If confidence in main conference is low, ALSO prepare a submission for NeurIPS ML4Drug Discovery workshop
- Shorter format, emphasize molecular results over methodology

Deadline: NeurIPS submission by May 15.

FINALLY: Stop the EC2 instance by running: /home/ubuntu/bin/stopinstance
```

### Phase 7 Checklist

- [ ] Preprint feedback monitored and addressed
- [ ] No competing preprint undermines novelty (or differentiation added)
- [ ] All numbers verified (text ↔ tables ↔ figures)
- [ ] Camera-ready quality achieved
- [ ] NeurIPS 2026 main conference submission complete
- [ ] Workshop submission prepared (if applicable)
- [ ] Author response template drafted

---

## Cost Summary

| Phase | Instance | Count | Est. Hours | Spot $/hr | Est. Cost |
|-------|----------|-------|------------|-----------|-----------|
| Phase 1 | g5.xlarge | 1 | ~45 | $0.44 | ~$20 |
| Phase 2 | g5.xlarge | 3 | ~170 total | $0.44 | ~$85 |
| Phase 3 | g5.xlarge | 2 | ~100 total | $0.44 | ~$50 |
| Phase 3 | c5.xlarge (analysis) | 1 | ~5 | $0.17 | ~$1 |
| Phase 4 | g5.xlarge | 2 | ~80 total | $0.44 | ~$40 |
| Phase 5 | c5.2xlarge (analysis) | 1 | ~15 | $0.34 | ~$5 |
| Phase 6 | — | — | — | — | ~$1 |
| Phase 7 | g5.xlarge (contingency) | 1 | ~20 | $0.44 | ~$10 |
| **Total** | | | | | **~$212** |

All instances in `us-east-1`. Use spot instances for all GPU work. Budget includes 15% contingency → **$244 ceiling**.
