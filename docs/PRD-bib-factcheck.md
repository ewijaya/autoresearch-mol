# PRD: BibTeX Fact-Check via Agent Teams

**Project:** autoresearch-mol
**Goal:** Verify all 30 entries in `manuscript/references.bib` against real publication metadata and ensure every in-text citation is coherent with what the cited paper actually contributes
**Author:** Rex
**Date:** 2026-03-30
**Status:** READY TO EXECUTE
**Depends on:** Manuscript draft complete (all .tex files written)

---

## 1. Objective

Run a multi-agent fact-check on `manuscript/references.bib` and all `\citep`/`\cite` references in the manuscript. The check has two parts:

1. **Bibliographic accuracy:** Do the BibTeX entries (title, authors, year, venue, volume, pages) match the actual published papers?
2. **Citation coherence:** Does each in-text citation accurately describe what the cited paper does?

The output is a single report file (`docs/bib-factcheck-report.md`) listing every issue found, categorized by severity.

---

## 2. Why Agent Teams

This task is naturally parallel and benefits from independent investigation:

- Each citation can be verified independently (no dependencies between checks)
- Different skills are needed: bibliographic metadata lookup vs. reading the manuscript narrative
- Competing perspectives help: one agent may catch a misattribution another misses
- The work is read-only (no file edits), so there are no file conflict risks

A single agent checking 30 references sequentially would be slow and context-heavy. Three teammates splitting the work finish faster and keep smaller context windows.

---

## 3. Team Structure

### 3.1 Team Lead

- Coordinates the three teammates
- Collects all findings into the final report
- Resolves disagreements if two teammates flag the same citation differently

### 3.2 Teammates

| Teammate | Role | Scope |
|----------|------|-------|
| **bib-checker** | Bibliographic metadata verification | All 30 BibTeX entries |
| **citation-checker-1** | Citation coherence for Sections 1-3 | introduction.tex, related_work.tex, methodology.tex |
| **citation-checker-2** | Citation coherence for Sections 4-7 | results.tex, practical.tex, discussion.tex, conclusion.tex, supplementary.tex |

### 3.3 Model Selection

- Lead: default (Opus or Sonnet, whichever is active)
- Teammates: Sonnet (sufficient for web lookups and text comparison; lower cost)

---

## 4. Verification Tasks

### 4.1 Bibliographic Accuracy (bib-checker)

For each of the 30 entries in `manuscript/references.bib`:

| Check | What to verify | How |
|-------|---------------|-----|
| Title | Exact match (modulo LaTeX escaping) | Web search for the paper by DOI or title |
| Authors | First author correct; "et al." entries have real author list | Cross-reference with venue proceedings |
| Year | Matches actual publication year | Check venue/arXiv date |
| Venue | Correct conference/journal name | Verify against publisher page |
| Volume/pages | Match if provided | Check against publisher metadata |
| DOI/arXiv ID | Valid if present | Attempt to resolve |
| BibTeX key | Consistent with content (no key/content mismatch) | Manual inspection |

**Known risks to check:**
- `suzgun2024zinc250k` (line 229): key says "suzgun2024" but the entry describes the ZINC database paper by Irwin et al. (2012). This looks like a key/content mismatch.
- `suzgun2024uniref` (line 240): similar issue; key says "suzgun2024" but content is Suzek et al. (2015) UniRef paper.
- `jordan2024muon`: arXiv preprint with no arXiv ID specified. Verify the paper exists and add the arXiv ID.
- `lehman2024openelm`: arXiv ID is "2206.08896" but year says 2024. Check if this was updated/published later.
- `hu2025lmsearcher`: author list is just "Hu, Yifan and others". Verify full author list.
- `huang2025improve`: author list is just "Huang, Sida and others". Verify full author list.

**Output format per entry:**
```
### [bibtex_key]
- Status: OK | WARNING | ERROR
- Issues: (list of specific problems)
- Suggested fix: (corrected BibTeX if needed)
```

### 4.2 Citation Coherence (citation-checker-1, citation-checker-2)

For each `\citep{...}` or `\cite{...}` in the assigned .tex files:

| Check | What to verify |
|-------|---------------|
| Claim matches paper | Does the text accurately describe what the cited paper does/shows? |
| Attribution correct | Is the contribution attributed to the right paper? (e.g., not crediting GQA to the wrong paper) |
| Context appropriate | Is the citation used in an appropriate context? (e.g., not citing a molecular paper for an NLP claim) |
| Missing citations | Are there claims that should have a citation but don't? |
| Overcitation | Are there citations that don't support the surrounding text? |

**Specific checks by section:**

**Introduction (citation-checker-1):**
- `\citep{chithrananda2020chemberta,ross2022molformer}` for "property prediction": do both papers actually do property prediction?
- `\citep{lin2023esm2}` for "protein structure": does ESM-2 predict protein structure or just protein language modeling?
- `\citep{zhou2023unimol}` for "3D molecular representations": correct?
- `\citep{elsken2019nas}` for "NAS offers a principled way": appropriate survey citation?
- `\citep{chen2023evoprompting,romeraparedes2024funsearch,hu2025lmsearcher}` for "LLMs as architecture search agents": do all three do this?
- `\citep{karpathy2026autoresearch}` for autoresearch framework: correct attribution?

**Related Work (citation-checker-1):**
- Every paper description matches what the paper actually does
- EvoPrompting described as "evolutionary search": correct?
- FunSearch described as "mathematical programs": correct?
- Self-Refine described as "iterative LLM refinement": correct?
- IMPROVE described as "ML code optimization": correct?
- MoLFormer described as "linear attention for molecules": does it actually use linear attention?
- ChemBERTa described as "BERT architecture to SMILES": correct?
- `\citep{shazeer2019multiquery,touvron2023llama}` for GQA: is Shazeer 2019 the right paper for multi-query attention? Is LLaMA the right citation for GQA adoption?
- `\citep{dauphin2017glu,shazeer2020glu}` for gated linear units: correct pair of citations?

**Methodology (citation-checker-1):**
- `\citep{irwin2005zinc}` for ZINC-250K: is this the right ZINC paper?
- `\citep{suzgun2024uniref}` for UniRef50: correct paper?
- `\citep{penedo2024fineweb}` for FineWeb-Edu: correct paper?
- `\citep{weininger1988smiles}` for SMILES representation: correct?
- `\citep{vaswani2017attention}` for "GPT-style": Vaswani 2017 is the Transformer paper, not GPT specifically. Is this appropriate?
- `\citep{jordan2024muon}` for MuonAdamW: does this paper describe Muon optimizer?

**Results (citation-checker-2):**
- `\citep{wu2018moleculenet}` for MoleculeNet benchmark: correct?

**Output format per citation:**
```
### [section.tex] line N: \citep{key}
- Context: "surrounding text..."
- Claim: [what the text claims about this reference]
- Actual: [what the paper actually does, from web search]
- Status: OK | WARNING | ERROR
- Issue: (description if not OK)
```

---

## 5. Execution Plan

### 5.1 Prompt for Team Lead

```
Create an agent team to fact-check our manuscript bibliography.

We need to verify 30 BibTeX entries in manuscript/references.bib and check
that every \citep citation in the manuscript accurately describes the cited
paper.

Spawn three teammates:

1. "bib-checker" - Verify bibliographic metadata (title, authors, year,
   venue) for all 30 entries in manuscript/references.bib. Use web search
   to check each entry against the actual publication. Flag any mismatches
   in title, author, year, or venue. Pay special attention to entries with
   suspicious keys (suzgun2024zinc250k, suzgun2024uniref) and incomplete
   entries (hu2025lmsearcher, huang2025improve, jordan2024muon).

2. "citation-checker-1" - Check citation coherence in Sections 1-3
   (sections/introduction.tex, sections/related_work.tex,
   sections/methodology.tex). For each \citep, verify that the surrounding
   text accurately describes what the cited paper does. Use web search to
   confirm paper content when unsure.

3. "citation-checker-2" - Check citation coherence in Sections 4-7
   (sections/results.tex, sections/practical.tex, sections/discussion.tex,
   sections/conclusion.tex, sections/supplementary.tex). Same approach as
   citation-checker-1.

Each teammate should produce a structured report. When all three are done,
synthesize their findings into a single report at
docs/bib-factcheck-report.md with sections: Summary, Errors (must fix),
Warnings (should review), and OK entries.

Use Sonnet for all teammates.
```

### 5.2 Expected Runtime

- bib-checker: ~10-15 min (30 web searches)
- citation-checker-1: ~10 min (15-20 citations in Sections 1-3)
- citation-checker-2: ~5 min (fewer citations in Sections 4-7)
- Lead synthesis: ~5 min

Total: ~20-25 min with parallelism.

### 5.3 Token Budget

- 3 teammates x ~50K tokens each = ~150K teammate tokens
- Lead: ~30K tokens for coordination and synthesis
- Total: ~180K tokens

---

## 6. Output

### 6.1 Primary Output

`docs/bib-factcheck-report.md` with this structure:

```markdown
# BibTeX Fact-Check Report

Generated: 2026-03-30
Checked: 30 BibTeX entries, N in-text citations

## Summary

- Errors (must fix): X
- Warnings (should review): Y
- OK: Z

## Errors

### [entry or citation details]
...

## Warnings

### [entry or citation details]
...

## All Entries (full checklist)

| Key | Title OK | Authors OK | Year OK | Venue OK | Status |
|-----|----------|------------|---------|----------|--------|
| ... | ... | ... | ... | ... | ... |
```

### 6.2 Action Items

After the report is generated:
1. Fix all ERROR entries in `references.bib`
2. Review all WARNING entries and fix as needed
3. Recompile manuscript to verify no broken citations
4. Re-run `bibtex main` to check for warnings

---

## 7. Known Issues to Pre-Flag

These are issues I already suspect based on reading the .bib file. The agents should confirm or refute each:

1. **`suzgun2024zinc250k`**: BibTeX key suggests Suzgun 2024 but content is Irwin et al. 2012 ZINC paper. Key should probably be `irwin2012zinc`.
2. **`suzgun2024uniref`**: Key suggests Suzgun 2024 but content is Suzek et al. 2015. Key should probably be `suzek2015uniref`.
3. **`lehman2024openelm`**: arXiv:2206.08896 was posted in 2022, not 2024. Year may need correction.
4. **`hu2025lmsearcher`**: Incomplete author list ("Hu, Yifan and others"). Need full authors.
5. **`huang2025improve`**: Incomplete author list ("Huang, Sida and others"). Need full authors.
6. **`jordan2024muon`**: Missing arXiv ID. Need to add it.
7. **`lin2023esm2`**: Cited for "protein structure" in introduction, but ESM-2 is a protein language model. The structure prediction capability is downstream; the primary paper is about language modeling. Check if attribution is accurate.
8. **`vaswani2017attention`**: Cited for "GPT-style" architecture. Vaswani 2017 is the original Transformer; GPT is Radford et al. 2018. Check if this is appropriate (it is, since GPT is a decoder-only Transformer).

---

## 8. Prerequisites

- Claude Code v2.1.32+ (for agent teams)
- `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in settings or environment
- Web search access for all teammates (to verify papers)
- Read access to `manuscript/` directory

---

## 9. Verification

After the fact-check is complete:

- [ ] `docs/bib-factcheck-report.md` exists
- [ ] All 30 BibTeX entries have been checked
- [ ] All in-text citations have been checked
- [ ] No ERROR items remain unfixed in references.bib
- [ ] Manuscript recompiles after any fixes
- [ ] `bibtex main` produces no warnings about missing entries

---

*PRD version 1.0 -- March 30, 2026*
