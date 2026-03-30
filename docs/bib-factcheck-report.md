# BibTeX Fact-Check Report

Generated: 2026-03-30
Checked: 30 BibTeX entries, ~25 in-text citations across 8 .tex files

## Summary

- **Errors (must fix): 9**
- **Warnings (should review): 7**
- **OK: 18**

(Some entries have both a bibliographic error and a citation coherence error, counted once per distinct issue.)

---

## Errors

### 1. `suzgun2024zinc250k` — Key/content mismatch + fabricated title

**Source:** bib-checker

- BibTeX key says "suzgun2024" but the entry describes the ZINC 2012 paper by Irwin, Sterling, Mysinger, Bolstad, Coleman. No author named "Suzgun" is involved.
- Title in bib ("ZINC-250K: A curated subset for molecular generation benchmarks") does not match the actual paper. The real 2012 Irwin et al. paper is titled "ZINC: A Free Tool to Discover Chemistry for Biology".
- The ZINC-250K subset does not have its own dedicated publication; it was popularized by the molecular generation community (e.g., Jin et al. 2018 JT-VAE).
- The metadata (authors, year 2012, JCIM vol 52, no. 7, pages 1757-1768) does match the real 2012 paper, but the title and key are wrong.

**Suggested fix:**
```bibtex
@article{irwin2012zinc,
  title={{ZINC}: A free tool to discover chemistry for biology},
  author={Irwin, John J and Sterling, Teague and Mysinger, Michael M and Bolstad, Erin S and Coleman, Ryan G},
  journal={Journal of Chemical Information and Modeling},
  volume={52},
  number={7},
  pages={1757--1768},
  year={2012},
  publisher={ACS Publications},
}
```
**Action:** Rename key from `suzgun2024zinc250k` to `irwin2012zinc`. Update all `\citep{suzgun2024zinc250k}` references in .tex files (none found in sections 1-7; key is only used internally or via `irwin2005zinc` for ZINC-250K).

---

### 2. `suzgun2024uniref` — Key/content mismatch + wrong title

**Source:** bib-checker, citation-checker-1

- BibTeX key says "suzgun2024" but entry is the Suzek et al. 2015 UniRef paper. No author named "Suzgun" is involved.
- Title in bib ("UniRef: Comprehensive and non-redundant UniProt reference clusters") matches the **2007** paper, but the metadata (vol 31, no. 6, pp 926-932, year 2015) matches the **2015** paper whose actual title is "UniRef clusters: a comprehensive and scalable alternative for improving sequence similarity searches".
- Year mismatch in key ("2024") vs content (2015).
- Used in methodology.tex line 53: `\citep{suzgun2024uniref}` for UniRef50.

**Suggested fix:**
```bibtex
@article{suzek2015uniref,
  title={{UniRef} clusters: a comprehensive and scalable alternative for improving sequence similarity searches},
  author={Suzek, Baris E and Wang, Yuqi and Huang, Hongzhan and McGarvey, Peter B and Wu, Cathy H and {UniProt Consortium}},
  journal={Bioinformatics},
  volume={31},
  number={6},
  pages={926--932},
  year={2015},
  publisher={Oxford University Press},
}
```
**Action:** Rename key to `suzek2015uniref`. Update `\citep{suzgun2024uniref}` to `\citep{suzek2015uniref}` in methodology.tex.

---

### 3. `huang2025improve` — Unverifiable entry (likely hallucinated)

**Source:** bib-checker, citation-checker-1

- Author listed as "Huang, Sida and others" — no ICLR 2025 paper matching this title and author could be found.
- Title in bib ("Iterative model-led program refinement via optimizing and validating expressions") does not match the actual IMPROVE paper found on arXiv (arXiv:2502.18530, "Iterative Model Pipeline Refinement and Optimization Leveraging LLM Experts" by Eric Xue et al.).
- Both the title expansion and the first author appear fabricated or confused.

**Suggested fix:** If citing the IMPROVE pipeline-refinement paper:
```bibtex
@article{xue2025improve,
  title={{IMPROVE}: Iterative Model Pipeline Refinement and Optimization Leveraging {LLM} Experts},
  author={Xue, Eric and Chen, Ke and Huang, Zeyi and others},
  journal={arXiv preprint arXiv:2502.18530},
  year={2025},
}
```
**Action:** Verify which IMPROVE paper was intended and correct the entry entirely.

---

### 4. `hu2025lmsearcher` — Wrong author name + wrong title

**Source:** bib-checker

- First author is **Yuxuan Hu**, not "Yifan Hu" as listed.
- Actual title is "LM-Searcher: Cross-domain Neural Architecture Search with LLMs via Unified Numerical Encoding" (not "LLM-based neural architecture search for natural language processing").
- Missing page numbers (actual: 9408-9421).

**Suggested fix:**
```bibtex
@inproceedings{hu2025lmsearcher,
  title={{LM-Searcher}: Cross-domain Neural Architecture Search with {LLMs} via Unified Numerical Encoding},
  author={Hu, Yuxuan and Liu, Jihao and Wang, Ke and Zheng, Jinliang and Shi, Weikang and Zhang, Manyuan and Dou, Qi and Liu, Rui and Zhou, Aojun and Li, Hongsheng},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={9408--9421},
  year={2025},
}
```

---

### 5. `lehman2024openelm` — Wrong year + misleading key + author name error

**Source:** bib-checker

- arXiv:2206.08896 was submitted June **2022**, not 2024. The year 2024 may reflect a later Springer book chapter, but the bib cites the arXiv version.
- Key says "openelm" but the paper is "Evolution through Large Models" (ELM). "OpenELM" is a separate follow-up.
- Author "Jain, Shyamal" — the actual name appears to be "Jain, Shawn" on arXiv.

**Suggested fix:**
```bibtex
@article{lehman2022elm,
  title={Evolution through large models},
  author={Lehman, Joel and Gordon, Jonathan and Jain, Shawn and Ndousse, Kamal and Yeh, Cathy and Stanley, Kenneth O},
  journal={arXiv preprint arXiv:2206.08896},
  year={2022},
}
```
**Action:** Rename key to `lehman2022elm`. Update all `\citep{lehman2024openelm}` references.

---

### 6. `dauphin2017glu` — Wrong author name

**Source:** bib-checker

- Fourth author listed as "Gresse, David" but the actual author is **David Grangier**.

**Suggested fix:**
```bibtex
author={Dauphin, Yann N and Fan, Angela and Auli, Michael and Grangier, David},
```

---

### 7. GQA citation error — `shazeer2019multiquery` + `touvron2023llama` cited for GQA

**Source:** citation-checker-1

- **Location:** related_work.tex line 20: `\citep{shazeer2019multiquery,touvron2023llama}` for "grouped query attention (GQA)"
- Shazeer 2019 introduced **multi-query attention (MQA)**, not GQA. GQA was introduced by Ainslie et al. 2023 (arXiv:2305.13245), which is missing from the bibliography entirely.
- `touvron2023llama` is LLaMA 1, which uses standard MHA, not GQA. GQA was adopted in **LLaMA 2** (Touvron et al., arXiv:2307.09288), a different paper not in the bib.

**Suggested fix:** Add `ainslie2023gqa` to the bibliography and cite it. Replace or supplement `touvron2023llama` with `touvron2023llama2` for GQA adoption:
```bibtex
@article{ainslie2023gqa,
  title={{GQA}: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
  author={Ainslie, Joshua and Lee-Thorp, James and de Jong, Michiel and Zemlyanskiy, Yury and Lebron, Federico and Sanghai, Sumit},
  journal={arXiv preprint arXiv:2305.13245},
  year={2023},
}
```

---

### 8. FunSearch misattributed as architecture search — introduction.tex

**Source:** citation-checker-1

- **Location:** introduction.tex line 6: `\citep{chen2023evoprompting,romeraparedes2024funsearch,hu2025lmsearcher}` for "large language models can serve as architecture search agents"
- FunSearch (Romera-Paredes et al. 2024) discovers mathematical programs for combinatorial problems, **not** neural architectures. It is not an architecture search paper.
- The related_work.tex description of FunSearch ("uses LLMs to discover novel mathematical programs") is correct, but the introduction groups it incorrectly with architecture search agents.

**Suggested fix:** Either remove `romeraparedes2024funsearch` from this citation group, or broaden the framing to "program search agents" rather than "architecture search agents".

---

### 9. `jordan2024muon` — Entirely wrong author list

**Source:** bib-checker

- Bib lists: Jordan, Keller and Chen, Yuchen and Goldblum, Micah and Saunders, George and Wilson, Andrew Gordon.
- Actual Muon authors: Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, Jeremy Bernstein.
- 4 of 5 co-authors are wrong. Only Keller Jordan is correct.
- Muon was published as a blog post, not an arXiv paper. No arXiv ID exists.

**Suggested fix:**
```bibtex
@misc{jordan2024muon,
  title={Muon: An optimizer for hidden layers in neural networks},
  author={Jordan, Keller and Jin, Yuchen and Boza, Vlado and You, Jiacheng and Cesista, Franz and Newhouse, Laker and Bernstein, Jeremy},
  year={2024},
  howpublished={\url{https://kellerjordan.github.io/posts/muon/}},
}
```

---

## Warnings

### 1. `touvron2023llama` — Wrong venue + author name typo

**Source:** bib-checker

- Listed as `@inproceedings` at ICML, but LLaMA 1 was never published at ICML. It is an arXiv preprint (arXiv:2302.13971).
- Author "Gorat, Naman" should be **"Goyal, Naman"**.

**Suggested fix:** Change to `@article` with `journal={arXiv preprint arXiv:2302.13971}` and fix author name to Goyal.

---

### 2. `lu2024aiscientist` — Wrong venue

**Source:** bib-checker

- Listed as `@inproceedings` at ICML 2024. The paper (arXiv:2408.06292, submitted August 2024) was released after ICML 2024 deadlines and has not been confirmed as an ICML proceedings paper.

**Suggested fix:** Change to `@article` with `journal={arXiv preprint arXiv:2408.06292}`.

---

### 3. `lin2023esm2` cited for "protein structure" — introduction.tex line 4

**Source:** citation-checker-1

- The text says "protein structure \citep{lin2023esm2}". The Lin et al. 2023 paper IS about structure prediction (ESMFold), but ESM-2 itself is the protein language model component. The citation is defensible but could confuse readers about ESM-2 (the LM) vs. ESMFold (the structure predictor).

**Suggested fix:** Consider adding "(ESMFold)" parenthetically, or citing both ESM-2 and ESMFold explicitly.

---

### 4. `vaswani2017attention` cited for "GPT-style" — methodology.tex line 73

**Source:** citation-checker-1

- The text says "a GPT-style \citep{vaswani2017attention} autoregressive transformer". Vaswani 2017 is the original encoder-decoder Transformer, not the GPT decoder-only variant (Radford et al. 2018).
- This is a common shorthand but technically imprecise when the text explicitly says "GPT-style".

**Suggested fix:** Add a GPT citation (Radford et al. 2018 or 2019) alongside Vaswani, or change wording to "Transformer-based" instead of "GPT-style".

---

### 5. `irwin2005zinc` for ZINC-250K — methodology.tex line 52

**Source:** citation-checker-1

- `irwin2005zinc` (2005, Irwin & Shoichet) is a reasonable general ZINC citation, but it does not define the ZINC-250K benchmark subset. The 250K subset was popularized by Gomez-Bombarelli et al. 2018 from ZINC15 (Sterling & Irwin 2015).

**Suggested fix:** Consider citing Sterling & Irwin 2015 (ZINC15) or Gomez-Bombarelli et al. 2018 as additional references for the specific 250K subset.

---

### 6. `penedo2024fineweb` — Truncated title

**Source:** bib-checker

- Title in bib: "FineWeb: decanting the web for the finest text data at scale"
- Actual title: "**The FineWeb Datasets**: Decanting the Web for the Finest Text Data at Scale"
- Also accepted at NeurIPS 2024 Datasets and Benchmarks track, so venue could be updated.

**Suggested fix:** Fix title to include "The FineWeb Datasets".

---

### 7. `hu2025lmsearcher` described as NLP-specific — related_work.tex line 5

**Source:** citation-checker-1

- Text says "LM-Searcher applies LLM-based NAS specifically to NLP architectures" but LM-Searcher is a cross-domain NAS method (CNN, LoRA, etc.), not NLP-specific.

**Suggested fix:** Change "specifically to NLP architectures" to "across multiple architecture domains".

---

## Missing Citations (should add)

| Location | Claim | Suggested citation |
|----------|-------|--------------------|
| practical.tex line 29 | GQA as a discovered innovation | Add `\citep{shazeer2019multiquery}` or new `ainslie2023gqa` |
| practical.tex line 29 | SwiGLU/GeGLU as discovered innovations | Add `\citep{shazeer2020glu}` |
| results.tex line 169 | "95.2% valid, 100% unique, 99.96% novel SMILES" metric framework | Cite GuacaMol (Brown et al. 2019) or MOSES (Polykovskiy et al. 2020) |
| conclusion.tex line 6 | "framework...publicly available" | Add code repository URL or anonymous link |

---

## All Entries (full checklist)

| Key | Title OK | Authors OK | Year OK | Venue OK | Status |
|-----|----------|------------|---------|----------|--------|
| `karpathy2026autoresearch` | Yes | Yes | Yes | Yes | OK |
| `chen2023evoprompting` | Yes | Yes | Yes | Yes | OK |
| `romeraparedes2024funsearch` | Yes | Yes | Yes | Yes | OK |
| `hu2025lmsearcher` | No | No | Yes | Yes | **ERROR** |
| `madaan2023selfrefine` | Yes | Truncated | Yes | Yes | OK |
| `huang2025improve` | No | No | Yes | Unverified | **ERROR** |
| `lu2024aiscientist` | Yes | Yes | Yes | No (not ICML) | WARNING |
| `wu2018moleculenet` | Yes | Yes | Yes | Yes | OK |
| `ross2022molformer` | Yes | Yes | Yes | Yes | OK |
| `lehman2024openelm` | Yes | No | No (2022) | N/A | **ERROR** |
| `nasir2023llmatic` | Yes | Yes | Yes | Yes | OK |
| `weininger1988smiles` | Yes | Yes | Yes | Yes | OK |
| `elsken2019nas` | Yes | Yes | Yes | Yes | OK |
| `lin2023esm2` | Yes | Yes | Yes | Yes | OK |
| `zhou2023unimol` | Yes | Yes | Yes | Yes | OK |
| `irwin2005zinc` | Yes | Yes | Yes | Yes | OK |
| `kaplan2020scaling` | Yes | Yes | Yes | Yes | OK |
| `touvron2023llama` | Yes | No (Gorat) | Yes | No (not ICML) | WARNING |
| `shazeer2019multiquery` | Yes | Yes | Yes | Yes | OK |
| `dauphin2017glu` | Yes | No (Gresse) | Yes | Yes | **ERROR** |
| `shazeer2020glu` | Yes | Yes | Yes | Yes | OK |
| `vaswani2017attention` | Yes | Yes | Yes | Yes | OK |
| `chithrananda2020chemberta` | Yes | Yes | Yes | Yes | OK |
| `irwin2022chemformer` | Yes | Yes | Yes | Yes | OK |
| `bergstra2012random` | Yes | Yes | Yes | Yes | OK |
| `falkner2018bohb` | Yes | Yes | Yes | Yes | OK |
| `hoffmann2022chinchilla` | Yes | Yes | Yes | Yes | OK |
| `suzgun2024zinc250k` | No | Yes* | No (key) | Yes | **ERROR** |
| `suzgun2024uniref` | No | Yes* | No (key) | Yes | **ERROR** |
| `penedo2024fineweb` | Truncated | Yes | Yes | Yes | WARNING |
| `jordan2024muon` | Yes | No | Yes | No arXiv ID | **ERROR** |

\* Authors in entry body are correct; key name is wrong.

---

## Citation Coherence Issues

| Section | Citation | Status | Issue |
|---------|----------|--------|-------|
| introduction.tex:4 | `chithrananda2020chemberta` | OK | Correct for property prediction |
| introduction.tex:4 | `ross2022molformer` | OK | Correct for property prediction |
| introduction.tex:4 | `lin2023esm2` | WARNING | ESM-2 is a protein LM; structure prediction is ESMFold |
| introduction.tex:4 | `zhou2023unimol` | OK | Correct for 3D molecular representations |
| introduction.tex:6 | `elsken2019nas` | OK | Canonical NAS survey |
| introduction.tex:6 | `chen2023evoprompting` | OK | Correct for LLM-based NAS |
| introduction.tex:6 | `romeraparedes2024funsearch` | **ERROR** | Not architecture search; mathematical program discovery |
| introduction.tex:6 | `hu2025lmsearcher` | WARNING | Works within predefined spaces, not "beyond" them |
| introduction.tex:6 | `karpathy2026autoresearch` | OK | Correct attribution |
| related_work.tex:5 | `chen2023evoprompting` | OK | Correct description |
| related_work.tex:5 | `romeraparedes2024funsearch` | OK | Correct here ("mathematical programs") |
| related_work.tex:5 | `hu2025lmsearcher` | WARNING | Cross-domain, not "specifically NLP" |
| related_work.tex:5 | `madaan2023selfrefine` | OK | Correct description |
| related_work.tex:5 | `huang2025improve` | **ERROR** | Bib entry title/author don't match real paper |
| related_work.tex:14 | `chithrananda2020chemberta` | OK | Correct for BERT on SMILES |
| related_work.tex:14 | `irwin2022chemformer` | OK | Correct for encoder-decoder on SMILES |
| related_work.tex:14 | `ross2022molformer` | OK | Correct for linear attention |
| related_work.tex:14 | `lin2023esm2` | OK | Correct here ("protein language models at billion scale") |
| related_work.tex:14 | `zhou2023unimol` | OK | Correct for 3D structural information |
| related_work.tex:20 | `shazeer2019multiquery` + `touvron2023llama` | **ERROR** | Shazeer=MQA not GQA; LLaMA 1 doesn't use GQA |
| related_work.tex:20 | `dauphin2017glu` + `shazeer2020glu` | OK | Correct pair for GLUs |
| methodology.tex:52 | `irwin2005zinc` | WARNING | 2005 paper, not specific to ZINC-250K subset |
| methodology.tex:53 | `suzgun2024uniref` | **ERROR** | Key mismatch (see bib error #2) |
| methodology.tex:54 | `penedo2024fineweb` | OK | Correct for FineWeb-Edu |
| methodology.tex:60 | `weininger1988smiles` | OK | Canonical SMILES paper |
| methodology.tex:73 | `vaswani2017attention` | WARNING | Transformer paper, not GPT specifically |
| methodology.tex:108 | `jordan2024muon` | OK | Description of Muon is accurate despite bib errors |
| results.tex:151 | `wu2018moleculenet` | OK | Correct for MoleculeNet benchmark |

---

*Report generated by 3-agent fact-check team. Agents: bib-checker (bibliographic metadata), citation-checker-1 (Sections 1-3), citation-checker-2 (Sections 4-7).*
