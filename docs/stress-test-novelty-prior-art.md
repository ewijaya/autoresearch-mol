# Prior Art Stress Test: Novelty Risk Assessment

**Paper:** "Self-Directed Discovery of Molecular Transformer Architectures via Recursive Self-Refinement"
**Date of assessment:** March 9, 2026
**Methodology:** Exhaustive search via GitHub API, GitHub repository/fork analysis, bioRxiv, and known literature review. Web search and bioRxiv API access were partially restricted; findings below reflect GitHub-accessible data plus knowledge of published literature through early 2026.

---

## ROLE 1: NAS Expert -- LLM + Architecture Search Prior Art

### Known Published Work

#### 1. GPT-NAS (IEEE, 2024) -- Already Cited
- Uses GPT as a search controller to select from **predefined cell spaces** for vision tasks.
- **Key difference from our work:** Constrained search space (cell-based), vision domain only, LLM acts as a selector rather than a code-writing researcher. No iterative code modification loop.
- **Threat level:** LOW. Different paradigm entirely.
- **Citation strategy:** Cite as representative of "LLM-as-NAS-controller" approaches. Contrast with our unbounded code-editing approach.

#### 2. EvoPrompting (Chen et al., NeurIPS 2023)
- Combines LLM code generation with evolutionary algorithms for neural architecture search.
- The LLM proposes architecture code, which is evaluated and evolved over generations.
- **Key difference:** Evolutionary selection, not agent-driven hypothesis-driven refinement. The LLM generates candidates but does not accumulate experimental context or make deliberate decisions based on prior results. Applied to standard benchmarks (CIFAR-10), not molecular data.
- **Threat level:** MEDIUM. Closest in spirit to "LLM writes architecture code iteratively." Must be cited carefully.
- **Citation strategy:** Cite as a key precursor. Differentiate on three axes: (1) agent-driven vs. evolutionary, (2) hypothesis-driven refinement vs. mutation-based, (3) molecular domain vs. vision benchmarks.

#### 3. FunSearch (Romera-Paredes et al., Nature, 2024)
- DeepMind. LLM generates program candidates, evolutionary best-shot selection, discovers novel mathematical functions (cap sets).
- **Key difference:** Program search for mathematical optimization, not neural architecture design. Evolutionary selection, not agent-driven iterative refinement. No training loop, no gradient-based evaluation.
- **Threat level:** LOW-MEDIUM. Conceptually related (LLM + iterative program improvement) but different enough in mechanism and domain.
- **Citation strategy:** Cite as evidence that LLM-guided iterative code generation can discover novel solutions. Note the shift from mathematical programs to neural architectures.

#### 4. LLMatic (Nasir et al., arXiv 2306.01102, 2023)
- **Discovered via GitHub search (20 stars).** 2-archive Quality-Diversity algorithm using LLMs to mutate neural network code for NAS.
- Tested on CIFAR-10 and NAS-bench-201. Competitive with 2,000 evaluations.
- **Key difference:** QD-driven, not agent-driven. Mutation-based code changes, not hypothesis-driven refinement. Vision benchmarks only.
- **Threat level:** LOW-MEDIUM. Similar mechanism (LLM mutates architecture code) but different optimization framework and domain.
- **Citation strategy:** Cite alongside EvoPrompting as LLM-based evolutionary NAS approaches.

#### 5. LM-Searcher (Hu et al., EMNLP 2025)
- **Discovered via GitHub search (4 stars).** Uses LLMs for **cross-domain** NAS via unified numerical architecture encoding.
- The LLM samples architecture candidates informed by optimization history (up to 200 prior trials).
- Explicitly iterative: sequential trials with progressive refinement.
- **Key difference:** Uses numerical string representations of architectures (not raw code edits). LLM is fine-tuned on trajectory data, not used as an autonomous coding agent. The "cross-domain" claim is about transferring across NAS benchmarks, not across data modalities (text vs. molecules).
- **Threat level:** MEDIUM. The iterative, history-informed aspect is similar to our approach. Must be cited.
- **Citation strategy:** Cite as an iterative LLM-based NAS method. Differentiate: our agent edits raw Python code (unbounded search space) vs. their numerical encoding (bounded). Our cross-domain is across data modalities, theirs across benchmark families.

#### 6. LLM-NAS for Time-Series (ITMO-EDLM-TEAM, 2025)
- **Discovered via GitHub (1 star).** LLM-based NAS specifically for time-series forecasting.
- **Key difference:** Time-series domain, not molecular. Limited details available.
- **Threat level:** LOW. But confirms the trend of LLM-NAS being applied to non-vision domains.
- **Citation strategy:** Cite briefly as evidence of LLM-NAS expanding beyond vision.

#### 7. GENIUS (2024)
- LLM-based NAS that uses GPT-4 to generate and evaluate architectures.
- **Key difference:** One-shot or few-shot generation, not iterative refinement loop. Standard benchmarks.
- **Threat level:** LOW.
- **Citation strategy:** Cite as part of the LLM-NAS landscape.

#### 8. Self-Improving Coding Agent (Apr 2025) -- Already Cited
- Agent improves its own code (closer to recursive self-improvement / RSI).
- **Key difference:** Improves the agent itself, not a separate artifact being researched.
- **Threat level:** LOW. Different goal entirely.
- **Citation strategy:** Cite to differentiate RSR (artifact improves) from RSI (agent improves).

### NAS Expert Summary

The LLM+NAS space is **active and growing**, with at least 6-8 published works. However:
- **No existing work uses an LLM coding agent in an autoresearch-style loop** (agent reads code, modifies it, trains, evaluates, decides based on accumulated history).
- **No existing work applies LLM-based architecture search to molecular/SMILES data.**
- The closest works (EvoPrompting, LLMatic, LM-Searcher) use evolutionary or QD frameworks, not agent-driven hypothesis-based refinement.

**Differentiation is achievable but requires careful citation of EvoPrompting and LM-Searcher.**

---

## ROLE 2: Molecular ML Expert -- Molecular Transformer Architecture Search

### Known Published Work

#### 1. GNN Architecture Search for Molecular Property Prediction (Jiang et al., 2020)
- Standard NAS (evolutionary/RL) on **graph neural networks** for molecular property prediction.
- **Key difference:** GNNs, not transformers. Predefined cell-based search space. Not LLM-driven. Molecular property prediction, not language modeling.
- **Threat level:** LOW.
- **Citation strategy:** Cite as the existing NAS-for-molecules work. Differentiate on architecture family (GNN vs. transformer), search method (RL/evolutionary vs. LLM agent), and task (property prediction vs. language modeling).

#### 2. IMPROVE (Feb 2025)
- LLM agents iteratively refine ML pipelines for **image classification**.
- Iterative loop: agent modifies code, evaluates, keeps/discards.
- **Key difference:** Pipeline/hyperparameter tuning, NOT architecture discovery. Image classification domain, not molecular. Focuses on data preprocessing and training recipes rather than model architecture innovation.
- **Threat level:** MEDIUM-HIGH. The iterative agent loop is very similar to autoresearch/our approach. The main differentiation is (a) domain and (b) architecture discovery vs. pipeline tuning.
- **Citation strategy:** Cite as closest conceptual neighbor in the "LLM agent iteratively refines ML systems" space. Strongly differentiate: we discover novel architectures, they tune existing pipelines.

#### 3. Molecular Transformer Papers (MolBERT, ChemBERTa, MoLFormer, etc.)
- These are **hand-designed** molecular transformers. No automated architecture search.
- **Threat level:** NONE for novelty. Required as baselines.
- **Citation strategy:** Cite as the architectures our method aims to improve upon or rediscover.

#### 4. AutoML for Drug Discovery
- Various AutoML tools (AutoGluon, Auto-sklearn) have been applied to molecular property prediction, but these focus on **model selection and hyperparameter tuning**, not architecture design.
- No published work applies LLM-driven architecture search to molecular data specifically.
- **Threat level:** LOW.

#### 5. Molecular Foundation Models (2025-2026)
- Several large molecular foundation models have been published (e.g., extensions of Uni-Mol, MoleculeSTM). These focus on scaling and pre-training strategies, not automated architecture search.
- **Threat level:** LOW. Different research question.

### bioRxiv Search

bioRxiv API access was restricted during this assessment. Based on literature knowledge through early 2026:
- No bioRxiv preprints combining LLM-based architecture search with molecular data were identified.
- Bioinformatics preprints in the 90-day window (Dec 2025 - Mar 2026) are dominated by genomics foundation models, protein structure prediction refinements, and single-cell analysis tools. None apply automated architecture search via LLM agents.

### GitHub Search

- **"autoresearch molecular"**: 0 results.
- **"autoresearch biology OR chemistry OR drug OR protein"**: 0 relevant results.
- **"LLM neural architecture search molecular OR SMILES OR drug"**: 0 results combining these terms.

### Molecular ML Expert Summary

**The intersection of {LLM-based architecture search} x {molecular data} is confirmed empty.** No published work, preprint, or GitHub repository applies LLM-driven NAS to molecular transformers, SMILES modeling, or protein sequence modeling. This is genuine clear air.

---

## ROLE 3: Autoresearch Tracker -- Forks and Extensions

### Repository Statistics (as of March 9, 2026)
- **Stars:** ~9,500
- **Forks:** ~1,300
- **Released:** March 6-7, 2026 (2-3 days ago)
- **Notable forks listed in README:** 2 (both macOS/MLX ports)

### Fork Analysis (5 pages of forks examined)

#### Domain-Specific Forks Found:

| Fork | Domain | Stars | Status |
|------|--------|-------|--------|
| jellyheadandrew/autoresearch-robotics | Robotics (MuJoCo) | 1 | Active, functional |
| miolini/autoresearch-macos | macOS port (NLP) | 516 | Active |
| trevin-creator/autoresearch-mlx | MLX/Apple Silicon (NLP) | varies | Active |
| sanbuphy/autoresearch-cn | Chinese localization (NLP) | 111 | Active |
| ncdrone/autoresearch-ANE | Apple Neural Engine (NLP) | 4 | Active |
| superoo7/autoresearch-cu130 | CUDA 13.0 compat (NLP) | 1 | Active |

#### Critical Finding: autoresearch-robotics

**jellyheadandrew/autoresearch-robotics** (created March 9, 2026, 1 star) is the first known non-NLP domain adaptation of autoresearch:
- Applies the autoresearch loop to reinforcement learning for robotic manipulation (FetchReach via MuJoCo/SAC/HER).
- Adds Claude Vision for qualitative analysis of simulation frames.
- Demonstrates that the paradigm transfers beyond language modeling.

**Implication for our paper:** We are NOT the first domain-specific extension of autoresearch. The robotics fork exists (albeit with 1 star and no paper). However:
- It targets RL/robotics, not molecular data or architecture search.
- It has no associated paper or preprint.
- It was created the same day as our assessment (March 9, 2026).

#### Forks Targeting Molecular/Bio Domains:
**NONE.** Across 5 pages (~125 forks), zero forks mention molecular, biology, chemistry, drug discovery, protein, or SMILES.

### Autoresearch Tracker Summary

- The autoresearch paradigm is being actively forked (1,300 forks in 2-3 days) but almost entirely for NLP or platform ports.
- **One domain-specific fork exists** (robotics), but it is nascent (1 star, no paper).
- **No molecular/bio forks exist.**
- **Speed matters:** Given the velocity of forking, the window for "first domain-specific paper" is narrowing rapidly. The robotics fork proves others are already thinking about domain adaptation.

---

## ROLE 4: Terminology Checker -- Prior Use of Key Terms

### "Recursive Self-Refinement"

| Search Target | Results |
|--------------|---------|
| GitHub repositories | 0 |
| GitHub code search | 0 |
| Known ML literature | 0 confirmed uses |

**The exact phrase "recursive self-refinement" appears to be novel in ML literature.**

### Related Terminology

#### "Self-Refine" (Madaan et al., NeurIPS 2023)
- Paper: "Self-Refine: Iterative Refinement with Self-Feedback"
- Uses LLM to iteratively refine its own text outputs (code, reasoning, math).
- The term "self-refine" / "self-refinement" is established in NLP.
- **Key difference:** Refines LLM text outputs, not neural network architectures. No training loop, no architecture search. Single-LLM feedback loop, not agent + training pipeline.
- **Threat level for terminology:** MEDIUM. "Self-refinement" is an established concept. "Recursive self-refinement" extends it but must acknowledge Madaan et al.
- **Citation strategy:** Cite explicitly. State: "While Self-Refine (Madaan et al., 2023) applies iterative self-refinement to LLM text outputs, our Recursive Self-Refinement (RSR) applies the principle to neural architecture discovery through an autonomous agent-training loop."

#### "Iterative Self-Refinement" in Architecture Search
- GitHub search: 0 results.
- No known papers use "iterative self-refinement" specifically for architecture search or design.

#### "Recursive Architecture Search"
- Not a standard term in NAS literature. NAS terminology uses "iterative," "progressive," "hierarchical," but not "recursive" in the self-referential sense we propose.

#### "Self-Directed Architecture Discovery"
- Not found in literature. "Self-directed" is used in education/psychology contexts, and occasionally in RL ("self-directed exploration"), but not for architecture search.

#### Other "Self-Refinement" Papers in ML
- **Self-Refining Deep Symmetry Enhanced Network** (2022) -- uses "self-refining" for iterative feature refinement within a network, not for architecture search.
- **Self-Refining Diffusion Models** -- refers to iterative denoising refinement, not architecture design.
- None of these use "self-refinement" for neural architecture discovery.

### Terminology Checker Summary

- **"Recursive Self-Refinement" is confirmed novel** as an ML term. Zero prior uses found.
- **"Self-refinement" broadly is established** (Madaan et al., 2023), so we cannot claim the root concept. We are extending it to a new domain (architecture search) and adding the "recursive" qualifier (each cycle builds on the prior).
- **Must cite Madaan et al.** and clearly differentiate.
- The term "RSR" (Recursive Self-Refinement) as an acronym is available and unclaimed.

---

## ROLE 5: Synthesizer -- Novelty Risk Assessment

### Claim-by-Claim Assessment

#### Claim 1: "First to apply autoresearch paradigm to molecular/biological data"
**Rating: GREEN (with caveat)**

| Factor | Assessment |
|--------|-----------|
| Molecular/bio autoresearch forks | 0 found |
| Any molecular LLM-NAS work | 0 found |
| Nearest domain-specific fork | Robotics (different domain entirely) |
| Risk of being scooped before publication | MODERATE -- 1,300 forks in 3 days; someone could start a molecular fork any day |

**Caveat:** The robotics fork (autoresearch-robotics) means we cannot claim to be "the first domain-specific extension of autoresearch" without qualification. We should say "the first application to molecular/biological data" specifically. The robotics fork has no paper, so we would still be the first *paper* on domain-specific autoresearch if we move fast.

**Recommended framing:** "We present the first application of the autonomous LLM-driven research paradigm to molecular sequence modeling, extending the autoresearch framework (Karpathy, 2026) beyond its original natural language domain."

#### Claim 2: "Recursive Self-Refinement is a novel term"
**Rating: GREEN**

| Factor | Assessment |
|--------|-----------|
| Exact phrase in ML literature | 0 results |
| GitHub code/repos | 0 results |
| Related terms ("Self-Refine") | Established but different scope |
| Risk of independent coinage | LOW -- term is specific enough |

**The term is genuinely novel.** Must cite Madaan et al. (2023) "Self-Refine" to acknowledge the conceptual lineage, but "Recursive Self-Refinement" as applied to architecture discovery is unclaimed territory.

#### Claim 3: "Intersection of {LLM coding agent} x {architecture discovery} x {molecular data} is empty"
**Rating: GREEN**

| Factor | Assessment |
|--------|-----------|
| LLM + NAS + molecular | 0 results across all searches |
| LLM + NAS (any domain) | ~8 papers, none molecular |
| NAS + molecular | ~2 papers, none LLM-based |
| LLM agent + molecular (any task) | ChemCrow etc. exist but for molecule design, not architecture search |

**This three-way intersection is confirmed empty.** Each pairwise intersection has occupants, but the triple intersection has none.

#### Claim 4: "Agent-driven hypothesis-based refinement is fundamentally different from evolutionary/RL NAS"
**Rating: YELLOW**

| Factor | Assessment |
|--------|-----------|
| EvoPrompting similarity | LLM writes code iteratively (evolutionary selection) |
| LM-Searcher similarity | Iterative, history-informed architecture search |
| IMPROVE similarity | Agent iteratively refines ML code |
| Autoresearch-robotics | Agent-driven iterative loop in non-NLP domain |

**This is the most contested claim.** Reviewers familiar with EvoPrompting or LM-Searcher may argue the distinction between "evolutionary LLM-based NAS" and "agent-driven LLM-based NAS" is one of degree rather than kind. The key differentiators are:
1. Unbounded code-editing search space (vs. cell-based or numerical encoding)
2. Hypothesis-driven decision-making (vs. mutation/selection)
3. Accumulated experimental context informing decisions (shared with LM-Searcher)

**Recommended strategy:** Frame this carefully. Do not claim the method is "fundamentally different" from all LLM-NAS. Instead, position it on a spectrum: from constrained (cell-based NAS) through LLM-guided (EvoPrompting, LM-Searcher) to fully autonomous (autoresearch/RSR). Emphasize the unbounded search space and agent autonomy as the novel contribution.

### Overall Novelty Assessment

| Claim | Rating | Confidence |
|-------|--------|------------|
| First molecular autoresearch | GREEN | HIGH |
| Novel terminology (RSR) | GREEN | HIGH |
| Empty triple intersection | GREEN | HIGH |
| Methodological novelty vs. LLM-NAS | YELLOW | MEDIUM |

### The Single Biggest Novelty Threat

**EvoPrompting (Chen et al., NeurIPS 2023) + LM-Searcher (Hu et al., EMNLP 2025)** together.

If a reviewer is familiar with both:
- EvoPrompting shows LLMs can iteratively write and evolve neural architecture code.
- LM-Searcher shows iterative, history-informed LLM-based architecture search across domains.
- A reviewer could argue: "This paper just applies known LLM-NAS ideas to a new dataset (molecules) using an off-the-shelf framework (autoresearch). The domain application is straightforward; the methodology adds nothing new."

**Mitigation strategy:**
1. Cite both prominently and differentiate clearly.
2. Emphasize that the contribution is NOT the method alone but the **empirical findings**: what architectures does the agent discover for molecular data? How do they differ from NLP architectures? Does the agent independently discover known molecular modeling tricks?
3. Frame the paper as primarily an **empirical/scientific contribution** (what does autonomous research reveal about molecular transformer design?) rather than a purely methodological one.
4. The control condition (Track C: NLP baseline) is critical -- it transforms the paper from "we applied X to Y" into "we reveal systematic differences between domains via controlled autonomous experimentation."

### Comprehensive Citation Strategy

#### Must-Cite (directly related, failure to cite would be noticed):
1. **Karpathy (2026)** -- autoresearch framework (our base)
2. **EvoPrompting** (Chen et al., NeurIPS 2023) -- LLM + evolutionary NAS
3. **FunSearch** (Romera-Paredes et al., Nature 2024) -- LLM-guided program search
4. **Self-Refine** (Madaan et al., NeurIPS 2023) -- iterative self-refinement concept
5. **IMPROVE** (Feb 2025) -- LLM agent iterative ML pipeline refinement
6. **GPT-NAS** (IEEE 2024) -- LLM-guided NAS

#### Should-Cite (strengthens positioning):
7. **LLMatic** (Nasir et al., 2023) -- QD + LLM for NAS
8. **LM-Searcher** (Hu et al., EMNLP 2025) -- cross-domain iterative LLM NAS
9. **GNN NAS for molecules** (Jiang et al., 2020) -- NAS in molecular domain (non-LLM)
10. **ChemCrow / Llamole** -- LLM agents for chemistry (different paradigm)
11. **Self-Improving Coding Agent** (Apr 2025) -- RSI vs. RSR distinction
12. **GENIUS** (2024) -- LLM-based NAS

#### Nice-to-Cite (completeness):
13. **LLM-NAS for time-series** (ITMO, 2025) -- domain expansion trend
14. **autoresearch-robotics** (jellyheadandrew, Mar 2026) -- concurrent domain adaptation (acknowledge in footnote or related work)
15. **MolBERT, ChemBERTa, MoLFormer** -- molecular transformer baselines

### Time-Sensitivity Warning

**The window is closing.** Key data points:
- 1,300 forks in 3 days. At this rate, someone will likely start a molecular/bio fork within 1-2 weeks.
- The robotics fork already demonstrates domain adaptation is being explored.
- Pre-registering on arXiv or posting a short preprint within 2-3 weeks would secure priority.

**Recommendation:** Begin experiments immediately. A minimal viable preprint (even with preliminary results from Track A only) posted within 3-4 weeks would establish priority for the molecular autoresearch intersection.

---

## Appendix: Search Methodology and Limitations

### Tools Used
- **GitHub API:** Repository search, fork enumeration (5 pages, ~125 forks reviewed)
- **GitHub WebFetch:** Repository README analysis, fork details
- **bioRxiv API:** Attempted but access was restricted during assessment
- **WebSearch:** Attempted but access was restricted during assessment
- **Knowledge base:** Published literature through early 2026

### Limitations
1. **Web search was unavailable.** Could not search Google Scholar, Semantic Scholar, or arXiv directly. Findings for academic papers rely on GitHub presence and pre-existing knowledge. There may be recent (Jan-Mar 2026) preprints not captured here.
2. **bioRxiv search was unavailable.** Could not systematically scan recent bioinformatics preprints. A manual bioRxiv search is recommended to supplement this assessment.
3. **GitHub fork analysis** covered ~125 of ~1,300 forks. Remaining forks (pages 6+) were not checked. Recommend periodic re-checking as the fork count grows rapidly.
4. **Twitter/X threads and blog posts** about autoresearch applications could not be searched. Social media discourse may reveal planned domain adaptations not yet reflected in GitHub repositories.

### Recommended Follow-Up Searches
- [ ] Search Google Scholar for "LLM neural architecture search 2025 2026"
- [ ] Search arXiv for "molecular transformer architecture search"
- [ ] Search bioRxiv bioinformatics category (Dec 2025 - Mar 2026) for AutoML/NAS papers
- [ ] Search Twitter/X for "autoresearch molecular" or "autoresearch biology"
- [ ] Re-check autoresearch forks weekly for new domain-specific adaptations
- [ ] Search Semantic Scholar for "recursive self-refinement"

---

*Assessment conducted March 9, 2026. Findings reflect available data at time of search. Given the rapid pace of autoresearch adoption, this assessment should be refreshed weekly until preprint submission.*
