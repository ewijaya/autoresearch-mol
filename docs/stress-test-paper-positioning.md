# Paper Positioning War Game: Stress Test

**Paper:** "Self-Directed Discovery of Molecular Transformer Architectures via Recursive Self-Refinement"
**Target:** NeurIPS 2026 (May deadline)
**Date:** March 9, 2026
**Time since autoresearch release:** 2 days (released March 7, 2026)

---

## Role 1: NeurIPS Strategist

### Main Conference vs. Workshop

**Recommendation: Submit to main conference, prepare workshop fallback.**

The paper sits at a triple intersection -- autonomous AI agents, neural architecture search, and molecular ML -- that is timely and interdisciplinary. NeurIPS values novelty and timeliness. Being the first domain-specific extension of autoresearch to molecular data is a genuine first-mover claim. However, the risk is that reviewers see this as "ran someone else's tool on new data" rather than a methodological contribution.

- **Main conference:** Viable if the paper delivers strong empirical results AND frames the contribution as a systematic study of how LLM agents explore architecture space for domain-specific data. The methodological contribution must be the analysis framework, not just the results.
- **Workshop (ML4Drug Discovery, CompBio):** High acceptance probability (~70-80%). Use as fallback if main conference reviews are borderline.
- **Datasets & Benchmarks track:** Not a natural fit unless the paper is reframed around releasing a benchmark for evaluating autonomous architecture search across domains. This could work but requires a different paper structure.

### Track Assessment

**Best fit: Main conference, ML track (not CompBio track).**

The core audience is the ML/AutoML community, not computational biologists. The molecular domain is the experimental testbed, not the contribution. Positioning it as CompBio would invite reviewers who care about SOTA molecular property prediction, which this paper does not deliver.

### Score Threshold

NeurIPS 2025/2026 acceptance typically requires an average reviewer score of 5.5-6.0/10, with no reviewer below 4. For this paper:

- **Optimistic case (strong results, good framing):** 6-7 range. Acceptance likely.
- **Base case (moderate results, clear presentation):** 5-6 range. Borderline, depends on champion reviewer.
- **Pessimistic case (negative results or weak differentiation from NAS):** 4-5 range. Rejection likely.

### Spotlight vs. Poster

For spotlight (top ~3% of submissions), the paper would need:

1. A genuinely surprising finding (e.g., the agent independently invents a known molecular trick like SMILES augmentation, or discovers an architecture that significantly outperforms hand-designed molecular transformers)
2. A compelling narrative arc (the "agent as scientist" story)
3. Ablations showing the LLM agent meaningfully outperforms non-LLM baselines (random search, evolutionary NAS)
4. Reproducibility with open-source agents (not just Claude/GPT-4)

For poster, the bar is lower: novel experimental setup + interesting analysis of agent behavior + some positive signal on at least 2 of 4 hypotheses.

### Contribution Type

**Primarily empirical, with a conceptual framing contribution.**

The method (autoresearch + molecular data) is an application/extension, not a new algorithm. The novelty is:
1. The experimental design (3-track comparison)
2. The analysis of what the agent discovers
3. The "recursive self-refinement" framing as a conceptual contribution

This means the paper lives or dies on the quality and interestingness of the empirical findings. If the agent just tunes learning rates, the paper is weak. If the agent makes genuinely creative architectural changes, the paper is strong.

### Comparison to NAS Papers at NeurIPS

Typical NAS papers at NeurIPS (DARTS, ENAS, ProxylessNAS, etc.) contribute:
- A new search algorithm with theoretical or empirical improvements
- SOTA results on standard benchmarks (CIFAR-10, ImageNet)
- Efficiency improvements (search cost reduction)

This paper does NOT contribute a new search algorithm -- the "algorithm" is Claude/GPT-4 reading code and making changes. This is both a weakness (no algorithmic novelty) and a strength (completely different paradigm that sidesteps the NAS search space design problem). Reviewers familiar with NAS will either find this refreshingly different or dismiss it as "just prompt engineering."

**Key NeurIPS 2025 reference points:**
- LLM-driven research papers (FunSearch by DeepMind, published 2024; likely follow-ups at NeurIPS 2025) set a precedent for "LLM discovers things" papers
- AlphaCode-style papers show LLMs can write meaningful code
- AutoML papers increasingly use LLMs as components (GPT-NAS, LLM-assisted hyperparameter tuning)

The bar has been set: the community accepts LLM-driven discovery as a valid research paradigm. This paper needs to show it works in a new domain with new insights.

### Is 8 Weeks Realistic?

**Tight but feasible, with conditions:**

- Weeks 1-2 (setup): Feasible IF the molecular data pipeline is straightforward. SMILES character-level tokenization avoids BPE training complexity. Risk: debugging Flash Attention compatibility with different sequence characteristics.
- Weeks 3-4 (experiments): 9 overnight runs (3 tracks x 3 replicates) can run in parallel on 3 GPUs. Feasible in 1 week if infrastructure works. Risk: runs crash or produce trivial results, requiring reruns.
- Weeks 5-6 (analysis): This is where the paper gets made or broken. Need to categorize every agent change, build transfer experiments, run baselines. This is the most underestimated phase.
- Weeks 7-8 (writing): Two weeks for a NeurIPS paper is tight but doable for experienced authors.

**Critical path risk:** If molecular runs don't produce interesting architectural changes by end of week 4, there's insufficient time to iterate on program.md or try different agent configurations.

---

## Role 2: Framing Critic

### "Recursive Self-Refinement" -- Assessment

**Verdict: Defensible but risky. The term is precise but triggers association with loaded concepts.**

Strengths:
- Technically accurate: the output of experiment N feeds experiment N+1
- The "self-refinement vs. self-improvement" distinction is genuine and well-articulated
- Zero prior use in ML literature (confirmed by novelty search) -- this is claimable territory
- Creates a citable concept (RSR) that others can reference

Weaknesses:
- "Recursive" will trigger RSI (Recursive Self-Improvement) associations for ~30% of ML reviewers, regardless of the disclaimer
- The disclaimer in the intro may read as protesting too much -- if you need a paragraph explaining why your term is NOT the scary thing, maybe pick a different term
- "Self-refinement" implies the system is refining itself, but actually it is the agent (fixed) refining a separate artifact (the architecture). The "self" is misleading.
- Buzzword risk: sounds like it is overselling what is fundamentally "iterative optimization with an LLM in the loop"

### Evaluation of 5 Alternative Framings

| Framing | Novelty | Clarity | Controversy Risk | Publishability |
|---------|---------|---------|-----------------|----------------|
| 1. Autonomous Architecture Discovery | Medium | High | Low | Good |
| 2. Agent-Driven NAS | Medium | High | Low | Good |
| 3. Iterative Code-Level Architecture Search | Low | Very High | Very Low | Medium |
| 4. LLM-Guided Architecture Evolution | Medium | High | Low | Good |
| 5. Programmatic Architecture Search via Language Agents | High | Medium | Low | Very Good |

**Detailed analysis:**

**1. "Autonomous Architecture Discovery"**
- Pro: Clean, accurate, no baggage. "Autonomous" is a strong but non-controversial claim.
- Con: Does not convey the iterative/compounding nature. Could describe a one-shot method.
- Verdict: Safe choice, slightly generic.

**2. "Agent-Driven NAS"**
- Pro: Immediately positions in the NAS literature. Reviewers know what to compare against.
- Con: Invites direct comparison with DARTS/ENAS on standard benchmarks, which this paper will not win. "NAS" comes with expectations of search efficiency analysis.
- Verdict: Dangerous. Sets up comparisons you do not want.

**3. "Iterative Code-Level Architecture Search"**
- Pro: Most technically precise. "Code-level" distinguishes from cell-space NAS. No buzzwords.
- Con: Boring. Does not convey that an LLM agent is doing the searching. Reads like a systems paper.
- Verdict: Too conservative. Undersells the novelty.

**4. "LLM-Guided Architecture Evolution"**
- Pro: "Evolution" captures the iterative improvement well. "LLM-Guided" is specific and accurate.
- Con: "Evolution" invites comparison with evolutionary NAS methods. Technically, this is not evolutionary (no population, no crossover, no selection pressure beyond keep/discard).
- Verdict: Good, but the evolution metaphor is slightly misleading.

**5. "Programmatic Architecture Search via Language Agents"**
- Pro: "Programmatic" is precise (the agent writes programs/code). "Language Agents" is the correct ML term for the paradigm. Positions well in the emerging "language agents" literature.
- Con: Longer and more academic. "Programmatic" may be confused with "Programmatic NAS" (a different thing).
- Verdict: Strong for a ML audience. Best novelty-to-controversy ratio.

### Recommendation

**Best framing: Option 5, with elements of Option 1.**

Proposed title: **"Programmatic Architecture Search for Molecular Transformers via Autonomous Language Agents"**

This title:
- Leads with the method (Programmatic Architecture Search)
- Specifies the domain (Molecular Transformers)
- Names the mechanism (Autonomous Language Agents)
- Avoids RSI associations entirely
- Is concrete and searchable
- Positions in both the NAS and language agents literatures

**However**, if the team is committed to claiming "Recursive Self-Refinement" as a named concept for long-term citation value, the current title is defensible. The question is whether the citation value of owning the term outweighs the reviewer friction it creates.

**Compromise:** Keep "Recursive Self-Refinement" as a named concept WITHIN the paper (Section 2: "We formalize this process as Recursive Self-Refinement (RSR)...") but do not put it in the title. The title should be concrete and descriptive; the conceptual branding happens inside.

### Would a More Concrete Title Be Better?

Yes. The current title is concept-forward ("Self-Directed Discovery... via Recursive Self-Refinement"). A results-forward title would be stronger:

- "What Do Language Agents Discover About Molecular Transformer Architectures?"
- "Autonomous Discovery of Domain-Specific Transformer Designs for Molecular Sequences"
- "Language Agents as Architecture Researchers: Automated Discovery of Molecular Transformer Designs"

The last option has the best balance of novelty signal and concreteness.

---

## Role 3: Results Predictor

### H1: Different Architectures for Different Domains

**Most likely outcome: Moderate differences, mostly in hyperparameters, with 1-2 genuine architectural differences.**

Prediction with confidence levels:
- **Very likely (80%):** The agent will converge on different depth/width ratios for SMILES vs. NLP. SMILES sequences are shorter and more structured; expect shallower, wider models.
- **Likely (65%):** Different attention patterns. SMILES has explicit bonding structure; the agent may discover that shorter attention windows work better (or worse -- SMILES needs global context for ring closures).
- **Possible (40%):** Different activation functions or normalization schemes.
- **Unlikely (20%):** Genuinely novel architectural components (new attention mechanisms, custom layers).

**If architectures DON'T differ:** This is the "universal transformer" finding. Frame as: "Despite operating on fundamentally different sequential data, the optimal 5-minute transformer architecture is invariant to domain, suggesting that at this scale, architecture matters less than optimization." This is publishable but less exciting. Strengthen it by showing that the convergent architecture differs from the standard GPT configuration (i.e., the agent finds improvements, they are just the same improvements for all domains).

**Key risk:** The 5-minute training budget and small model size may mean that at this scale, all domains look the same. Larger-scale differences may only emerge with longer training. This limits the strength of any claim.

### H2: Rediscovery of Known Tricks

**This is the highest-upside hypothesis but hardest to validate.**

What counts as "rediscovery"?
- **Strong rediscovery:** Agent explicitly implements SMILES randomization (canonical vs. non-canonical SMILES), or adds positional encodings that respect ring structure, or implements atom-level attention masking.
- **Weak rediscovery:** Agent adjusts sequence length, changes tokenization-adjacent parameters, or tunes parameters that happen to align with known best practices.

**Prediction:** The agent will NOT rediscover domain-specific tricks because:
1. The agent only modifies train.py, not prepare.py. It cannot change tokenization or data augmentation.
2. SMILES-specific tricks (randomization, augmentation) are data pipeline operations, not architecture changes.
3. The agent has no explicit molecular domain knowledge in its context (unless program.md provides it).

**Mitigation:** Reframe H2 as "Does the agent discover architectural patterns that correlate with known molecular modeling principles?" This is softer but more honest. Look for:
- Local attention patterns (correlate with bonded-atom interactions)
- Smaller vocabulary utilization patterns
- Different embedding dimension choices

**Verification method:** Post-hoc alignment analysis. After experiments complete, compare the agent's architectural choices against a checklist of known molecular modeling techniques. Use cosine similarity or categorical overlap metrics. This is inherently subjective, so present it as qualitative analysis, not a hard test.

### H3: Transfer Learning (SMILES to Protein)

**Prediction: Weak positive transfer, not statistically significant.**

Reasoning:
- SMILES and protein sequences share: sequential structure, finite alphabet, local grammar rules
- They differ in: alphabet size (SMILES ~40 chars, proteins 20 amino acids), sequence length distributions, structural semantics
- At the architecture level, a good SMILES architecture is probably a reasonable protein architecture (both benefit from similar depth/width ratios for sequential data), but the improvement over a baseline protein architecture will be small
- The interesting result would be ASYMMETRIC transfer: SMILES-to-protein transfers well but protein-to-SMILES does not (or vice versa). This would suggest directional architectural compatibility.

**Expected result:** Transfer architectures perform within 5% of domain-specific architectures. This is a weak finding unless the 5% matters for a specific downstream task.

**How to make this interesting:** Do not just report val_bpb. Show which specific architectural components transfer and which do not. A component-level transfer analysis is more publishable than a single aggregate metric.

### H4: Faster Than Random Search

**Prediction: Yes, easily. This is too weak a claim to anchor the paper.**

The LLM agent will outperform random NAS because:
1. It starts from a reasonable baseline (the default train.py) and makes incremental improvements
2. It uses code understanding to avoid obviously broken configurations
3. It has implicit priors about what works from pretraining data

A random search baseline will waste most of its budget on configurations that crash or diverge. The agent avoids this entirely. Showing agent > random is like showing that a human researcher outperforms a coin-flipping monkey.

**How to strengthen H4:**
- Compare against SMARTER baselines: Bayesian optimization (e.g., BOHB), evolutionary NAS with a well-designed cell space, grid search over known-good hyperparameter ranges
- Show sample efficiency: how many experiments does the agent need vs. baselines to reach X% of final performance?
- Show that the agent's search trajectory is qualitatively different from random (e.g., it explores systematically, not randomly)

### What If the Agent Just Tunes Hyperparameters?

**This is the most likely failure mode and must be planned for.**

Based on autoresearch's NLP results, the agent tends to:
1. First: adjust learning rate, batch size, model dimensions (hyperparameters)
2. Then: modify activation functions, normalization (minor architectural)
3. Rarely: invent new components or fundamentally restructure the model

If the agent only tunes hyperparameters:
- The paper becomes "LLM-Guided Hyperparameter Optimization for Molecular Transformers" -- less exciting but still publishable at a workshop
- Mitigation: modify program.md to explicitly encourage architectural changes (e.g., "Focus on structural modifications to the transformer architecture, not just hyperparameter tuning"). This itself becomes an ablation (constrained vs. unconstrained agent)
- Track the ratio of architectural vs. hyperparameter changes across tracks. If the agent makes more architectural changes for molecular data than NLP data, that is itself a finding.

### Writing a Compelling Paper with Negative Results

If most hypotheses are negative, structure the paper as:

1. **Lead with the framework:** The contribution is the experimental methodology, not the results
2. **"Surprising null result" framing:** "Despite the structural differences between molecular and natural language sequences, autonomous architecture search converges on remarkably similar designs..."
3. **Emphasize the process analysis:** What did the agent TRY? The trajectory of failed experiments is as interesting as the final result
4. **Connect to the universality debate:** Does this support the "transformer is all you need" hypothesis?

### Minimum Publishable Result

The absolute minimum for a main conference paper:
1. The agent produces measurable improvement over baseline on at least 2 of 3 tracks
2. At least one qualitative architectural difference between tracks (even if small)
3. The agent outperforms random search by a meaningful margin
4. A clean analysis of the agent's search trajectory with interpretable takeaways

For a workshop paper, even weaker results suffice: "We applied autoresearch to molecular data. Here is what happened. Here are lessons for the community."

---

## Role 4: Competing Groups Assessment

### Immediate Threat: Karpathy Followers / AI Twitter

**Threat level: HIGH. Time horizon: Days to weeks.**

Autoresearch was released March 7. By March 9 (today), the AI Twitter community has certainly noticed it. The obvious extensions are:
1. Run it on different datasets (code, math, multilingual, molecular, music, DNA)
2. Run it with different agents (Claude, GPT-4, Gemini, open-source)
3. Run it on different hardware (MLX/Apple Silicon, AMD, multi-GPU)

The MLX forks already exist (miolini/autoresearch-macos, trevin-creator/autoresearch-mlx). Domain-specific forks are the next obvious step.

**Who specifically:**
- Independent ML researchers/influencers who follow Karpathy
- PhD students looking for quick publications
- AI labs doing "weekend projects"

**Their advantage:** Speed. Someone could fork autoresearch, point it at a molecular dataset, run overnight, and post results on Twitter within 48 hours.

**Their disadvantage:** A Twitter thread or blog post is not a paper. They likely will not do the systematic 3-track comparison, the transfer experiments, the baseline comparisons, or the careful framing. But they will take the "first to do it" claim.

**Assessment:** Someone will almost certainly run autoresearch on non-NLP data within the next 1-2 weeks. The question is whether they will write a paper or just tweet about it.

### Molecular ML Groups

**Threat level: MEDIUM. Time horizon: Weeks to months.**

Key groups:

- **MIT (Barzilay/Jaakkola lab):** Deep expertise in molecular ML. Published foundational work on molecular transformers and graph neural networks. They would approach this with rigor but likely view autoresearch as too informal/unprincipled for their research taste. More likely to build a proper benchmark than to use Karpathy's tool directly. Time to paper: 3-6 months.

- **Stanford (Pande lab successors / Dror lab / Leskovec group):** Strong in molecular ML and graph learning. Similar assessment to MIT -- they would want a cleaner methodology. The Leskovec group might be interested in the "LLM agent" angle but would probably focus on GNNs, not transformers for SMILES.

- **DeepMind AlphaFold team:** Not a threat. They focus on protein structure prediction with massive compute. Autoresearch's 5-minute-per-experiment paradigm is too small-scale for their research agenda.

- **Molecular transformer practitioners (ChemBERTa authors at DeepChem, MolBERT):** They know the domain but are unlikely to pivot to LLM-driven architecture search. Different research culture.

**Assessment:** Traditional molecular ML groups are not a near-term threat. They will see this paper and either (a) dismiss it as not rigorous enough, or (b) cite it as motivation for their own, more thorough follow-up.

### AutoML Groups

**Threat level: MEDIUM-HIGH. Time horizon: Weeks to months.**

Key groups:

- **Frank Hutter's group (Freiburg):** Leading AutoML group. Published Auto-PyTorch, BOHB, and foundational NAS work. They are VERY likely to notice autoresearch and frame it within their research program. However, their approach would be to build a principled framework on top of it (benchmarks, comparisons with existing NAS), not to rush a domain-specific paper. Time to paper: 2-4 months.

- **Google Brain/DeepMind AutoML:** Published NASNet, AmoebaNet, EfficientNet. They have the compute to run autoresearch at scale. However, internal review processes slow them down. Time to paper: 3-6 months.

- **Microsoft Research (AutoGen team):** They have the LLM agent expertise (AutoGen, TaskWeaver) and could connect it to NAS. But they focus on agent frameworks, not domain-specific applications. Lower threat.

**Assessment:** AutoML groups are the most dangerous medium-term competitors. They will do this more rigorously with proper baselines and benchmarks. Our advantage is speed and the specific molecular angle.

### Groups with BOTH LLM Agent AND Molecular ML Expertise

**This intersection is small, which is our key advantage.**

Very few groups combine:
1. Hands-on experience with LLM coding agents
2. Published work in molecular ML
3. Infrastructure to run GPU experiments quickly

Possible candidates:
- **Recursion Pharmaceuticals:** AI-native pharma company, uses ML extensively. They have both LLM and molecular expertise but focus on drug discovery, not ML methodology papers.
- **Insilico Medicine:** Similar profile. Publishing in Nature-family journals, not NeurIPS.
- **Isomorphic Labs (DeepMind spinoff):** Protein-focused, massive compute, but slow to publish.
- **Individual researchers** at the intersection: hardest to track, highest speed potential.

**Assessment:** No single group dominates this intersection. This is the paper's strongest positioning argument.

### Speed of Execution

| Group Type | Time to Run Experiments | Time to Paper | Total |
|------------|------------------------|---------------|-------|
| AI Twitter (blog post) | 1-3 days | 0 (tweet/blog) | 1-3 days |
| Independent researcher (arXiv) | 1-2 weeks | 2-3 weeks | 3-5 weeks |
| Academic lab (conference paper) | 2-4 weeks | 4-8 weeks | 6-12 weeks |
| Industry lab (conference paper) | 2-4 weeks | 8-16 weeks | 10-20 weeks |

### Realistic Time Advantage

Autoresearch released: March 7, 2026
Today: March 9, 2026
NeurIPS deadline: ~May 15, 2026

**Our window:** ~10 weeks to submission.
**Competitor window:** Same, but they have not started yet (probably).

**Realistic assessment:**
- We are 2 days into a 10-week timeline. That is not a meaningful head start.
- Anyone who starts this week has the same shot at NeurIPS 2026.
- The real question is: who else is working on this RIGHT NOW?

### Signs of Existing Competition

Based on available information:
- Only 2 forks exist (both macOS/MLX ports, not domain-specific)
- No molecular or biology-specific forks visible
- No arXiv preprints on autoresearch extensions yet (too early, given 2-day window)
- No Twitter threads announcing molecular autoresearch experiments (as of search date)

**But this means nothing.** Anyone working on this silently would not announce it until they have results. The 2-day window is too short to draw conclusions from absence of evidence.

---

## Synthesized Recommendation

### GO/NO-GO: CONDITIONAL GO

**Confidence: MEDIUM (60%)**

### Decision Matrix

| Factor | Assessment | Impact |
|--------|-----------|--------|
| Novelty of intersection | Strong -- empty space confirmed | +++ |
| Time pressure (competition) | High -- 2-day head start is negligible | -- |
| NeurIPS acceptance probability | 35-50% for main conference | + |
| Workshop acceptance probability | 70-80% | ++ |
| Risk of trivial results | Medium-high (agent may only tune hyperparams) | -- |
| 8-week timeline feasibility | Tight but possible | +/- |
| Framing/positioning clarity | Needs refinement (RSR terminology risky) | - |
| Infrastructure readiness | Reasonable (g5.xlarge, existing fleet) | + |
| Cost | Low (~$72-200 for core experiments) | ++ |

### Conditions for GO

1. **Week 1 checkpoint (by March 16):** Molecular data pipeline (SMILES character-level tokenization) is working and a baseline autoresearch run completes on ZINC-250K with meaningful val_bpb improvement. If not: reassess timeline.

2. **Week 3 checkpoint (by March 30):** At least one track shows the agent making genuine architectural changes (not just hyperparameter tuning). If all three tracks show only hyperparameter tuning: pivot to workshop paper or reframe as "hyperparameter optimization" study.

3. **Week 4 checkpoint (by April 6):** At least one qualitative difference between molecular and NLP architectures is visible. If all tracks converge: the "universal transformer" framing must be developed as the primary narrative.

4. **Framing decision (by March 16):** Decide on title and terminology before experiments begin. Recommendation: adopt "Programmatic Architecture Search" framing for the title, keep "Recursive Self-Refinement" as an internal concept. Reduces reviewer friction.

5. **Preprint strategy:** Post to arXiv by week 6 (late April) regardless of NeurIPS submission status. This establishes priority even if the paper needs more work for the conference.

### Risk-Adjusted Strategy

**Primary target:** NeurIPS 2026 main conference (submit May)
**Fallback 1:** NeurIPS 2026 ML4Drug Discovery workshop (later deadline, ~June-July)
**Fallback 2:** ICML 2027 (submit ~January 2027, with 6 months more results)
**Fallback 3:** Domain journal (Nature Computational Science, Bioinformatics) -- different audience, different standards

### Three Scenarios

**Best case (20% probability):** Agent discovers genuinely surprising architectural differences for molecular data. At least one known trick is approximately rediscovered. Transfer results show asymmetric patterns. Paper is a clear accept at NeurIPS main conference.

**Base case (50% probability):** Agent finds moderate differences (mostly depth/width, some attention pattern changes). No trick rediscovery. Transfer results are inconclusive. Paper is borderline for main conference; strong accept at workshop. ArXiv preprint establishes priority.

**Worst case (30% probability):** Agent only tunes hyperparameters. No meaningful architectural differences across tracks. Paper is not competitive for main conference. Reframe as workshop contribution or pivot to a different angle (e.g., "Benchmarking LLM Agents as Architecture Researchers").

### Final Verdict

**GO, with eyes open.** The cost is low ($72-200 + 8 weeks of effort), the downside is bounded (workshop paper + arXiv preprint at worst), and the upside is meaningful (first domain-specific autoresearch paper at a top venue). The key risk is not competition -- it is result quality. Establish checkpoints and be willing to pivot framing if results do not support the original hypotheses.

The single most important action right now: **start running experiments today.** Every day of delay erodes the first-mover advantage. The molecular data pipeline should be the week 1 priority, with the first overnight agent run happening by March 14 at the latest.

---

*War game conducted March 9, 2026. Reassess at each checkpoint.*
