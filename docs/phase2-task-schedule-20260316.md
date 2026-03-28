# Phase 2 Task Schedule

**Date**: 2026-03-16 (updated 2026-03-28)
**Status**: Complete — all 34 of 34 tasks finished

---

## Task Queue (Full Listing)

The runner interleaves agent runs with baseline runs to ensure incremental data collection across conditions.

### Completed (Tasks 1-34)

| # | Kind | Track | Run | Status | Best val_bpb | Notes |
|---|------|-------|-----|--------|-------------|-------|
| 1 | agent | smiles | 1 | Done | 0.5918 (exp071) | 1 crash |
| 2 | random_nas | smiles | 1 | Done | 0.5906 (exp093) | |
| 3 | agent | smiles | 2 | Done | 0.5808 (exp047) | 3 crashes |
| 4 | random_nas | smiles | 2 | Done | 0.5923 (exp082) | |
| 5 | agent | smiles | 3 | Done | 0.5839 (exp051) | 4 crashes |
| 6 | random_nas | smiles | 3 | Done | 0.5914 (exp084) | |
| 7 | agent | smiles | 4 | Done | 0.5892 (exp092) | 0 crashes |
| 8 | random_nas | protein | 1 | Done | 3.9719 (exp047) | 103 experiments |
| 9 | agent | smiles | 5 | Done | 0.5834 (exp099) | 1 crash |
| 10 | random_nas | protein | 2 | Done | 3.9710 (exp064) | |
| 11 | agent | protein | 1 | Done | 3.9656 (exp099) | 3 crashes; best protein overall |
| 12 | random_nas | protein | 3 | Done | 3.9693 (exp078) | |
| 13 | agent | protein | 2 | Done | 3.9684 (exp099) | 3 crashes |
| 14 | random_nas | nlp | 1 | Done | 1.1297 (exp016) | 3 crashes |
| 15 | agent | protein | 3 | Done | 3.9666 (exp078) | 5 crashes |
| 16 | random_nas | nlp | 2 | Done | 1.1301 (exp052) | 2 crashes |
| 17 | agent | nlp | 1 | Done | 1.1188 (exp089) | 4 crashes |
| 18 | random_nas | nlp | 3 | Done | 1.1306 (exp055) | 1 crash |
| 19 | agent | nlp | 2 | Done | 1.1277 (exp099) | 0 crashes |
| 20 | hp_only | smiles | 1 | Done | 0.5807 (exp095) | 0 crashes |
| 21 | agent | nlp | 3 | Done | 1.1151 (exp081) | 2 crashes; best NLP overall |
| 22 | hp_only | smiles | 2 | Done | 0.5801 (exp090) | 0 crashes; best SMILES overall |
| 23 | agent | nlp | 4 | Done | 1.1212 (exp099) | 0 crashes |
| 24 | hp_only | smiles | 3 | Done | 0.5810 (exp073) | 1 crash |
| 25 | agent | nlp | 5 | Done | 1.1314 (exp098) | 21 crashes |
| 26 | hp_only | protein | 1 | Done | 3.9901 (exp068) | 8 crashes |
| 27 | hp_only | protein | 2 | Done | 3.9699 (exp018) | 0 crashes |
| 28 | hp_only | protein | 3 | Done | 3.9684 (exp038) | 0 crashes |
| 29 | hp_only | nlp | 1 | Done | 1.1462 (exp094) | 0 crashes |
| 30 | hp_only | nlp | 2 | Done | 1.1470 (exp086) | 0 crashes; best hp_only NLP |
| 31 | hp_only | nlp | 3 | Done | 1.1477 (exp060) | 0 crashes |
| 32 | fixed_default | smiles | 1 | Done | 0.5961 (exp001) | 1 experiment; baseline floor |
| 33 | fixed_default | protein | 1 | Done | 3.9767 (exp001) | 1 experiment; baseline floor |
| 34 | fixed_default | nlp | 1 | Done | 1.1528 (exp001) | 1 experiment; baseline floor |

---

## Time Estimates

| Estimate | Value |
|----------|-------|
| Agent runs (~8 min/experiment, 100 experiments) | ~13 hrs each |
| Random NAS runs (~6 min/experiment, 100 experiments) | ~10 hrs each |
| Fixed default (single architecture, one training run) | ~10 hrs |
| **Total remaining (0 tasks)** | **0 hrs — complete** |

**Assumptions**:
- Continuous runtime with no API rate-limit pauses (optimistic; we've already hit one weekly limit pause during SMILES run_4)
- Each experiment is 5 minutes of GPU training plus agent overhead (~3 min for agent runs, ~1 min for NAS/fixed)
- Fixed default estimate assumes it trains 100 experiments of the same architecture (to be verified)

---

## Experimental Design Summary

| Condition | Description | Runs per track | Tracks | Total runs |
|-----------|-------------|----------------|--------|------------|
| **agent** | LLM-guided architecture + HP search | 5 (SMILES, NLP) / 3 (protein) | 3 | 13 |
| **random_nas** | Random architecture sampling (depth, width, heads, activation, attention) | 3 | 3 | 9 |
| **hp_only** | LLM-guided HP search only (architecture fixed) | 3 | 3 | 9 |
| **fixed_default** | No search; baseline architecture as-is | 1 | 3 | 3 |
| | | | **Total** | **34** |

The 4-condition design enables a decomposition analysis (Story 7):
- **agent vs random_nas**: Does LLM guidance beat random search?
- **agent vs hp_only**: Does architecture search add value beyond HP tuning?
- **hp_only vs fixed_default**: Does HP tuning alone help?
- **random_nas vs fixed_default**: Does any form of architecture variation help?

---

## Risk Factors

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Weekly API rate limits** | N/A | All tasks complete |
| **Disk space** | Low | Results are small (~50 MB/run); datasets already downloaded |
| **GPU availability** | Low | Single A10G, no contention currently |

---

## Progress Summary

- **SMILES track**: All runs complete (5 agent + 3 random_nas + 3 hp_only + 1 fixed_default). Best agent: 0.5808 (run_2). Best hp_only: 0.5801 (run_2). Fixed default floor: 0.5961.
- **Protein track**: All runs complete (3 agent + 3 random_nas + 3 hp_only + 1 fixed_default). Best agent: 3.9656 (run_1). Best hp_only: 3.9684 (run_3). Fixed default floor: 3.9767.
- **NLP track**: All runs complete (5 agent + 3 random_nas + 3 hp_only + 1 fixed_default). Best agent: 1.1151 (run_3). Best hp_only: 1.1470 (run_2). Best random_nas: 1.1297 (run_1). Fixed default floor: 1.1528.
