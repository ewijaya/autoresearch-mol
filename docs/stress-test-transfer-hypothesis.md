# Stress-Test of Hypothesis H3: SMILES-to-Protein Transfer

> **H3:** "Architectures discovered on SMILES will partially transfer to protein sequences (shared sequential molecular grammar) but not fully (different alphabet size, structural constraints)."

This document provides a rigorous distributional analysis of both sequence domains, predicts which architectural features should and should not transfer, and designs experiments to make H3 publishable regardless of outcome.

---

## Part 1: SMILES Linguist — Formal Analysis of SMILES Grammar

### 1.1 SMILES Syntax Rules

SMILES (Simplified Molecular-Input Line-Entry System) is a **context-free grammar** with the following core productions:

| Rule | Description | Example |
|------|-------------|---------|
| **Atoms** | Organic subset: B, C, N, O, P, S, F, Cl, Br, I (and lowercase aromatic: b, c, n, o, p, s) | `C`, `c`, `N` |
| **Bonds** | Single (implicit or `-`), double (`=`), triple (`#`), aromatic (`:`) | `C=O`, `C#N` |
| **Branches** | Parentheses denote side chains off the main chain | `CC(=O)O` (acetic acid) |
| **Rings** | Digit pairs mark ring-closure bonds; same digit opens and closes | `C1CCCCC1` (cyclohexane) |
| **Stereo** | `@`, `@@` for tetrahedral; `/`, `\` for cis/trans | `C(/F)=C/Cl` |
| **Brackets** | `[...]` for non-organic-subset atoms, charges, isotopes, hydrogen counts | `[NH4+]`, `[13C]` |
| **Disconnection** | `.` separates disconnected fragments | `[Na+].[Cl-]` |

SMILES is formally context-free: brackets and parentheses must balance, and ring-closure digits must pair. However, ring closures create a form of **non-local cross-serial dependency** analogous to Swiss German word order — a digit at position i must match a digit at position j, creating long-range constraints that a purely local model cannot capture.

### 1.2 Character Vocabulary

The practical SMILES character set for drug-like molecules:

| Category | Characters | Count |
|----------|-----------|-------|
| Organic atoms (upper) | C, N, O, S, P, B, F, I | 8 |
| Aromatic atoms (lower) | c, n, o, s, p, b | 6 |
| Halogens (multi-char) | Cl, Br (treated as 2 chars or 1 token) | 2-4 |
| Bonds | -, =, #, : | 4 |
| Branches | (, ) | 2 |
| Ring digits | 0-9, % (for rings >9) | 11 |
| Brackets | [, ] | 2 |
| Stereo | @, /, \ | 3 |
| Charges | +, - (reused) | (counted above) |
| Dot | . | 1 |
| Hydrogens, isotopes | H, 0-9 (reused inside brackets) | 1 |
| **Total unique characters** | | **~42-55** |

At character level on ZINC-250K, the effective vocabulary is approximately **45-50 unique characters** after deduplication. If using byte-level encoding, this rises slightly; if using atom-level tokenization (Cl, Br as single tokens), the vocabulary is ~50-55 tokens.

### 1.3 Sequence Length Distribution (ZINC-250K)

| Statistic | Value |
|-----------|-------|
| Mean length | ~43 characters |
| Median length | ~40 characters |
| 10th percentile | ~25 characters |
| 90th percentile | ~65 characters |
| Max length | ~120 characters |
| Standard deviation | ~14 characters |

Drug-like molecules (Lipinski's Rule of Five) are compact: typically 250-500 Da, translating to 30-60 SMILES characters. This is **much shorter** than natural language sentences (mean ~20 words = ~100 characters) or protein sequences.

### 1.4 Character Frequency Distribution

Approximate character frequencies in ZINC-250K (character-level):

| Rank | Character | Frequency (%) | Role |
|------|-----------|---------------|------|
| 1 | C | ~20-25% | Carbon (backbone) |
| 2 | c | ~10-15% | Aromatic carbon |
| 3 | ( | ~8-10% | Branch open |
| 4 | ) | ~8-10% | Branch close |
| 5 | 1 | ~4-6% | Ring closure |
| 6 | N | ~4-5% | Nitrogen |
| 7 | = | ~3-5% | Double bond |
| 8 | O | ~3-5% | Oxygen |
| 9 | n | ~2-4% | Aromatic nitrogen |
| 10 | 2 | ~2-3% | Ring closure |
| 11 | S | ~1-2% | Sulfur |
| 12 | F | ~1-2% | Fluorine |
| 13-20 | Cl, Br, #, @, etc. | <1% each | Rare characters |

**Key observation:** The distribution is highly skewed. Carbon alone accounts for ~35% of all characters (aromatic + aliphatic). Structural characters (parentheses, ring digits) account for ~25%. The effective entropy is lower than vocabulary size suggests.

### 1.5 Dependency Structure

| Dependency Type | Range | Example | Frequency |
|-----------------|-------|---------|-----------|
| **Immediate bonding** | 1-2 positions | `C=O` | Very high |
| **Branch structure** | 2-15 positions | `CC(CC(=O)O)CC` | High |
| **Ring closures** | 3-30+ positions | `c1ccc2c(c1)cccc2` (naphthalene) | Moderate |
| **Stereochemistry** | 1-5 positions | `C(/F)=C/Cl` | Low |
| **Bracket matching** | 1-10 positions | `[NH2+]` | Moderate |

**Bracket depth analysis:** Typical drug-like molecules have parenthesis nesting depth of 2-4. Polycyclic molecules can reach depth 5-6. Ring closures create the most challenging long-range dependencies — in a molecule like naphthalene (`c1ccc2c(c1)cccc2`), digit 1 spans 5 characters and digit 2 spans 9 characters. In complex natural products, ring spans can exceed 20 characters.

### 1.6 Optimal Attention Patterns for SMILES

1. **Local attention (window 3-5):** Captures immediate bond context, atom environments
2. **Parenthesis-matching attention:** Heads that learn to connect matching `(` and `)` pairs
3. **Ring-closure attention:** Heads that learn to connect matching digits (the hardest pattern)
4. **Global sparse attention:** For capturing molecular-level properties (charge balance, aromaticity)

A SMILES-optimal architecture would likely benefit from **relatively shallow models** (sequences are short, so deep representations are not as critical) with **specialized attention heads** for bracket and ring matching.

---

## Part 2: Protein Linguist — Formal Analysis of Protein Sequences

### 2.1 Amino Acid Alphabet

| Category | Characters | Count |
|----------|-----------|-------|
| Standard amino acids | A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y | 20 |
| Rare amino acids | U (selenocysteine), O (pyrrolysine) | 2 |
| Ambiguity codes | B (D/N), Z (E/Q), X (any), J (I/L) | 4 |
| **Practical vocab** | | **~25** |

In practice, >99% of sequences use only the 20 standard amino acids. The practical effective vocabulary is **20-22 characters** for most datasets.

### 2.2 Sequence Length Distribution (UniRef50)

| Statistic | Value |
|-----------|-------|
| Mean length | ~330 residues |
| Median length | ~250 residues |
| 10th percentile | ~80 residues |
| 90th percentile | ~650 residues |
| Max length | ~35,000+ residues (titin) |
| Standard deviation | ~300 residues |

The length distribution is **heavily right-skewed** (log-normal). For a practical training subset, truncating at 512 or 1024 residues captures ~85-95% of sequences. This is **6-8x longer** than typical SMILES strings.

### 2.3 Amino Acid Frequency Distribution

| Rank | Amino Acid | Frequency (%) | Properties |
|------|-----------|---------------|------------|
| 1 | L (Leucine) | ~9.5% | Hydrophobic, aliphatic |
| 2 | A (Alanine) | ~8.3% | Small, hydrophobic |
| 3 | G (Glycine) | ~7.1% | Smallest, flexible |
| 4 | V (Valine) | ~6.8% | Hydrophobic, branched |
| 5 | E (Glutamate) | ~6.3% | Negative charge |
| 6 | S (Serine) | ~6.1% | Polar, small |
| 7 | I (Isoleucine) | ~5.9% | Hydrophobic |
| 8 | K (Lysine) | ~5.8% | Positive charge |
| 9 | R (Arginine) | ~5.5% | Positive charge |
| 10 | D (Aspartate) | ~5.4% | Negative charge |
| 11 | T (Threonine) | ~5.3% | Polar |
| 12 | P (Proline) | ~4.7% | Rigid, helix-breaker |
| 13 | N (Asparagine) | ~4.0% | Polar |
| 14 | Q (Glutamine) | ~3.9% | Polar |
| 15 | F (Phenylalanine) | ~3.9% | Aromatic |
| 16 | Y (Tyrosine) | ~2.9% | Aromatic, polar |
| 17 | M (Methionine) | ~2.4% | Sulfur-containing |
| 18 | H (Histidine) | ~2.3% | Aromatic, pH-sensitive |
| 19 | C (Cysteine) | ~1.4% | Disulfide bonds |
| 20 | W (Tryptophan) | ~1.1% | Largest aromatic |

**Key observation:** The distribution is **much more uniform** than SMILES. The most frequent amino acid (Leu, ~9.5%) is far less dominant than carbon in SMILES (~25%). Shannon entropy per position is higher for proteins (~4.0 bits) vs. SMILES (~3.3 bits).

### 2.4 Dependency Structure

| Dependency Type | Range (residues) | Example | Frequency |
|-----------------|-----------------|---------|-----------|
| **Local bonding** | 1 (peptide bond) | Always adjacent | Very high |
| **Secondary structure** | 3-6 residues | Alpha helix (period 3.6), beta turns (4 residues) | Very high |
| **Beta sheet pairing** | 5-50+ residues | Parallel/antiparallel strands hydrogen bonding | High |
| **Disulfide bonds** | 10-200+ residues | Cysteine pairs (C...C) | Moderate |
| **Tertiary contacts** | 10-500+ residues | Hydrophobic core packing, salt bridges | High |
| **Domain boundaries** | 50-300 residues | Multi-domain proteins | Moderate |

**Critical difference from SMILES:** Protein dependencies are **not syntactically marked**. There is no character in the sequence that signals a disulfide bond or tertiary contact — these are emergent physical properties. The sequence is a flat string of amino acids with no brackets, no parentheses, no explicit structural annotation. A transformer must learn these dependencies entirely from statistical co-occurrence.

### 2.5 Optimal Attention Patterns for Proteins

1. **Local attention (window 5-10):** Captures secondary structure motifs (helix periodicity, beta turns)
2. **Periodic attention:** Heads with period ~3.6 for helix recognition
3. **Long-range symmetric attention:** For contact prediction (residue i contacts residue j implies both attend to each other)
4. **Segment-level attention:** For domain-level patterns
5. **Position-independent attention:** For conserved motifs regardless of location

A protein-optimal architecture would benefit from **deeper models** (longer sequences need more layers of abstraction) with **wider attention spans** and likely **more attention heads** to capture the diversity of dependency types.

---

## Part 3: Comparative Distributional Analysis

### 3.1 Head-to-Head Comparison Table

| Property | SMILES (Track A) | Protein (Track B) | Implication for Transfer |
|----------|-----------------|-------------------|--------------------------|
| **Vocabulary size** | ~45-55 chars | ~20-25 chars | Similar order of magnitude; embedding layers nearly interchangeable |
| **Effective alphabet entropy** | ~3.3 bits/char | ~4.0 bits/char | Proteins are more uniform; SMILES more predictable per-character |
| **Mean sequence length** | ~43 chars | ~330 residues | **7.7x longer** — this is the dominant difference |
| **Sequence length variance** | Low (SD ~14) | Very high (SD ~300) | Protein arch must handle extreme length variation |
| **Most frequent char** | C (~25%) | L (~9.5%) | SMILES much more skewed |
| **Top-5 concentration** | ~65% of chars | ~38% of chars | SMILES distribution more concentrated |
| **Grammar type** | Context-free (brackets, rings) | No explicit grammar | Fundamentally different parsing needs |
| **Explicit structural markers** | Yes (parentheses, digits, brackets) | No | SMILES has syntactic shortcuts; proteins do not |
| **Local dependency range** | 1-5 chars (bonds) | 1-6 residues (secondary structure) | **Similar** — both benefit from local attention |
| **Long-range dependency type** | Ring closures (syntactically marked) | Tertiary contacts (implicit) | Different discovery mechanisms for attention |
| **Long-range dependency span** | 3-30 chars | 10-500+ residues | Protein long-range is much further |
| **Positional meaning** | Weak (SMILES is non-canonical) | Strong (N-to-C terminus has meaning) | Different positional encoding needs |
| **Data augmentation** | SMILES randomization (many valid strings per molecule) | Minimal (sequence is canonical) | Augmentation tricks will not transfer |
| **Sequence structure** | Tree-like (branches) | Linear (flat sequence) | Different inductive biases needed |

### 3.2 Shared Properties (Support Transfer)

1. **Small discrete vocabulary:** Both domains have vocabularies of 20-55 characters, orders of magnitude smaller than NLP (50K+ BPE tokens). Embedding strategies, output projection layers, and softmax temperature tuning should transfer directly.

2. **Character-level modeling:** Both benefit from character/residue-level tokenization rather than subword BPE. Architectural innovations around character-level transformers should transfer.

3. **Local dependencies dominate:** In both domains, the most informative context is within 5-10 positions. Attention patterns with local bias (windowed attention, relative position encoding) should benefit both.

4. **Small model sufficiency:** Given the small vocabularies and (for SMILES) short sequences, both domains likely favor smaller models than NLP. Width and depth preferences discovered on one may partially apply to the other.

5. **Repetitive motifs:** Both have recurring structural patterns (functional groups in SMILES, secondary structure elements in proteins) that benefit from pattern-matching attention heads.

### 3.3 Divergent Properties (Impede Transfer)

1. **Sequence length (7.7x difference):** This is the single largest barrier. SMILES architectures optimized for length ~43 will have:
   - Insufficient depth for length ~330 (need more layers of abstraction)
   - Positional encodings that degrade or are untested beyond 100
   - Memory/compute budgets calibrated wrong for 7x longer sequences
   - Attention patterns that assume full quadratic attention is cheap

2. **Explicit vs. implicit structure:** SMILES encodes structure syntactically (parentheses, ring digits); proteins encode structure only through amino acid identity and position. An architecture that learns bracket-matching attention heads will find those heads useless on proteins.

3. **Character frequency distribution:** SMILES's carbon-dominated distribution means the model can often "predict C" as a strong baseline. Protein models must distribute predictions more evenly. This affects:
   - Loss landscape shape
   - Gradient magnitudes per output position
   - Optimal temperature/softmax scaling

4. **Positional semantics:** SMILES has weak positional meaning (randomized SMILES are equivalent molecules). Proteins have strong positional meaning (N-terminus vs. C-terminus, signal peptides at position 1-30, etc.). Positional encoding schemes will transfer poorly.

5. **Dependency structure:** SMILES ring closures are syntactically paired (like matching parentheses in code). Protein long-range contacts are statistically learned with no syntactic marker. Architectures that excel at one pattern may not help with the other.

6. **Length variance:** SMILES lengths are tightly distributed (SD/mean = 0.33). Protein lengths span 3 orders of magnitude (SD/mean = 0.91). Architectures must handle extreme length heterogeneity for proteins but not SMILES.

---

## Part 4: Transfer Theorist — Predictions and Experimental Design

### 4.1 Quantitative Transfer Predictions

Based on the distributional analysis, the following predictions are made:

| Transfer Direction | val_bpb Degradation | Confidence | Reasoning |
|-------------------|---------------------|------------|-----------|
| SMILES arch to protein data | 15-30% worse than protein-native | High | Sequence length mismatch dominates |
| Protein arch to SMILES data | 5-15% worse than SMILES-native | Medium | Protein arch is "oversized" but functional |
| NLP arch to SMILES data | 20-40% worse | Medium | Massive vocab mismatch, wrong scale |
| NLP arch to protein data | 10-25% worse | Medium | Closer in sequence length, still wrong vocab |

**Asymmetry prediction:** Transfer from protein to SMILES should work better than SMILES to protein, because:
- A model designed for length 330 can handle length 43 (just underutilized capacity)
- A model designed for length 43 will struggle at length 330 (positional encoding degrades, insufficient depth)
- This mirrors the general principle that "scaling down is easier than scaling up"

### 4.2 Layer-by-Layer Transfer Analysis

**Experiment:** For the best SMILES and protein architectures, train both on both datasets. Then analyze which layers contribute most to performance.

| Layer Type | Expected Transfer Quality | Rationale |
|-----------|--------------------------|-----------|
| **Embedding layer** | High | Small vocab means small embeddings; easy to reinitialize |
| **Early attention layers (1-2)** | High | Local patterns are similar in both domains |
| **Middle attention layers (3-5)** | Medium | Domain-specific patterns emerge here |
| **Late attention layers (6+)** | Low | Highly specialized to sequence-level patterns |
| **Output projection** | Must reinitialize | Different vocabulary |
| **Positional encoding** | Low | Different length scales and positional semantics |
| **Layer norm parameters** | High | Distribution-normalizing; relatively domain-agnostic |

**Proposed protocol:** Freeze layers progressively (embed-only, embed+layer1, etc.) and measure degradation. The "transfer cliff" — where freezing one more layer causes sharp degradation — reveals the domain-specificity boundary.

### 4.3 Attention Head Analysis

**Experiment:** After training best architectures on both domains, extract attention patterns and classify heads by function.

| Head Type | Expected in SMILES | Expected in Protein | Shared? |
|-----------|-------------------|---------------------|---------|
| Local window (plus/minus 3) | Yes | Yes | **Yes** |
| Bracket-matching | Yes | No | No |
| Ring-closure pairing | Yes | No | No |
| Periodic (period ~3.6) | No | Yes (helix) | No |
| Long-range symmetric | Moderate | Yes (contacts) | Partial |
| Positional (attend to fixed positions) | No | Yes (termini, signals) | No |
| Copy/retrieval | Yes (repeated substructures) | Yes (conserved motifs) | **Yes** |

**Prediction:** ~30-40% of attention heads will serve analogous functions across domains (local context, copy). ~60-70% will be domain-specific.

### 4.4 Comprehensive Experimental Design

#### Experiment 1: Direct Cross-Evaluation (Core H3 Test)

```
For arch in {best_SMILES_arch, best_protein_arch, best_NLP_arch, GPT2-small_baseline}:
    For data in {SMILES, Protein, NLP}:
        Train arch on data for 5 minutes
        Record val_bpb

Result: 4x3 = 12 conditions, 3 replicates each = 36 runs
Output: 4x3 performance matrix
```

**Analysis:** Compute transfer efficiency = val_bpb(transferred) / val_bpb(native). Values close to 1.0 indicate good transfer.

#### Experiment 2: Progressive Layer Freezing

```
For source in {SMILES, Protein}:
    Train best_{source}_arch on source data (full training)
    For n_frozen in {0, 1, 2, ..., all_layers}:
        Freeze first n_frozen layers
        Train on target data
        Record val_bpb

Result: Transfer curve showing domain-specificity boundary
```

#### Experiment 3: Attention Pattern Probing

```
For arch in {best_SMILES_arch, best_protein_arch}:
    Train on native data
    Extract attention maps for 1000 test sequences
    For each head:
        Compute: local_score (attention within plus/minus 5)
        Compute: periodic_score (FFT of attention pattern)
        Compute: long_range_score (attention beyond plus/minus 20)
        Compute: bracket_match_score (SMILES only)
        Classify head type

Result: Head-type distribution comparison
```

#### Experiment 4: Architectural Feature Ablation

Identify the top 5 architectural innovations the agent discovered for each track. Systematically ablate:

```
For each innovation I in {SMILES_innovations}:
    arch_without_I = remove innovation I from best_SMILES_arch
    Train arch_without_I on SMILES data -> delta_native
    Train arch_without_I on Protein data -> delta_transfer
    If delta_native >> delta_transfer: innovation is SMILES-specific
    If delta_native is approx delta_transfer: innovation is universal

Result: Classification of each innovation as domain-specific vs universal
```

#### Experiment 5: Vocabulary-Controlled Transfer

To isolate the effect of architecture from vocabulary:

```
Map both SMILES and Protein to a shared 26-character alphabet (A-Z)
Retrain architectures on mapped data
Re-run cross-evaluation

Result: Transfer performance with vocabulary confound removed
```

#### Experiment 6: Length-Controlled Transfer

To isolate sequence length effects:

```
Create length-matched subsets:
    SMILES_long: select SMILES with length 80-120 (longest molecules)
    Protein_short: select proteins with length 40-80 (shortest proteins)

Train architectures on matched-length data
Compare transfer performance

Result: Is length or grammar the bigger transfer barrier?
```

### 4.5 Controls

| Control | Purpose |
|---------|---------|
| **GPT-2 small (124M)** | Standard NLP architecture as transfer baseline |
| **Random architecture search** | Same budget but random modifications instead of agent-directed |
| **Shuffled sequences** | Destroy all structure; measures how much architecture depends on real dependencies |
| **Reversed SMILES** | Valid molecules, different character order; tests positional encoding sensitivity |
| **Homopolymer proteins** | Poly-A, Poly-L sequences; tests if architecture handles low-complexity regions |

### 4.6 Making H3 Publishable Regardless of Outcome

#### Scenario A: Partial Transfer (Expected — Confirms H3)

**Story:** "We find that ~40% of architectural innovations discovered by RSR transfer across molecular domains, primarily local attention patterns and embedding strategies. However, innovations related to sequence length handling, long-range dependency structure, and domain-specific syntax patterns do not transfer. This provides the first empirical evidence that molecular sequence domains require distinct transformer architectures despite superficial similarities."

**Key figures:**
1. Transfer efficiency matrix (4x3 heatmap)
2. Layer freezing curves showing domain-specificity boundary
3. Attention head type distributions (side-by-side)
4. Ablation chart: universal vs. domain-specific innovations

#### Scenario B: Strong Transfer (Surprising — Refutes H3)

**Story:** "Contrary to our hypothesis, architectures discovered for SMILES transfer well to protein sequences (and vice versa), suggesting a universal 'molecular transformer' architecture. The shared properties — small vocabulary, character-level modeling, local dependency dominance — create sufficient architectural convergence despite surface-level differences in grammar and sequence length."

**Key figures:**
1. Transfer matrix showing near-diagonal performance
2. Architecture comparison showing convergent features across tracks
3. Comparison with NLP architecture transfer (which should still fail)
4. Analysis: which shared properties drive transferability?

#### Scenario C: No Transfer (Also Publishable — Strengthens H3)

**Story:** "We find that architectures are highly domain-specific even between related molecular sequence types. SMILES architectures exploit syntactic structure (parentheses, ring digits) that has no analog in proteins. Protein architectures develop long-range contact-prediction capabilities irrelevant to short SMILES strings. This demonstrates that the autoresearch agent discovers genuinely domain-adapted architectures, not generic language models."

**Key figures:**
1. Transfer matrix showing strong diagonal dominance
2. Qualitative analysis of discovered innovations per domain
3. Attention pattern visualizations showing domain-specific heads
4. Code diff analysis: what did the agent change differently?

### 4.7 Statistical Rigor

- **3 replicates minimum** per condition (9 for core experiments if budget allows)
- **Report:** mean, standard deviation, and 95% confidence intervals for all val_bpb values
- **Statistical tests:** Paired t-test or Wilcoxon signed-rank for native vs. transferred performance
- **Effect size:** Cohen's d for transfer degradation
- **Multiple comparison correction:** Bonferroni or FDR for the full experiment matrix

---

## Part 5: Existing Literature on SMILES-Protein Transfer

While a comprehensive web-based literature search was not possible at time of writing, the following related work is known and should be investigated:

### Known Related Work

1. **Unified Molecular Representations:**
   - **MolFormer** (IBM, 2022): Pre-trained on 1.1B SMILES, but not transferred to proteins.
   - **ESM-2** (Meta, 2022): Pre-trained on proteins, not transferred to SMILES.
   - No known work trains a single architecture on both SMILES and protein sequences from scratch.

2. **Multi-Modal Molecular Models:**
   - **Galactica** (Meta, 2022): Trained on scientific text including SMILES and protein sequences simultaneously. Suggests shared representations are possible but uses massive scale.
   - **BioTranslator** (2022): Translates between biological modalities but uses separate encoders.

3. **Chemical-Biological Interface:**
   - **DrugBank interaction models** use separate encoders for drugs (SMILES) and targets (proteins), implicitly acknowledging that unified architectures are difficult.
   - **MolTrans** (2021): Drug-target interaction prediction with separate molecular and protein encoders.

4. **Key Gap:** No prior work has:
   - Used an autonomous agent to discover architectures for both domains
   - Systematically compared architectural preferences across SMILES and protein data
   - Performed controlled transfer experiments between agent-discovered architectures

This gap strongly supports the novelty of H3 testing within the RSR framework.

---

## Part 6: Summary of Predictions

### Transfer Scorecard

| Architectural Feature | SMILES to Protein | Protein to SMILES | Confidence |
|----------------------|------------------|------------------|------------|
| Embedding dimension | Transfers well | Transfers well | High |
| Number of layers | Too shallow | Acceptable (overbuilt) | High |
| Attention head count | May be insufficient | Acceptable | Medium |
| Local attention window | Transfers well | Transfers well | High |
| Positional encoding scheme | Transfers poorly | Transfers poorly | High |
| Activation function | Transfers well | Transfers well | Medium |
| Learning rate schedule | Partial transfer | Partial transfer | Medium |
| Specialized attention (rings/contacts) | Does not transfer | Does not transfer | High |
| Layer normalization strategy | Transfers well | Transfers well | High |
| Dropout / regularization | Needs recalibration | Needs recalibration | Medium |

### Bottom Line

**H3 is well-formulated and testable.** The distributional analysis predicts partial transfer driven by shared vocabulary scale and local dependency structure, with transfer failure driven by sequence length mismatch and structural grammar differences. The asymmetry prediction (protein-to-SMILES transfers better than SMILES-to-protein) provides a specific, falsifiable claim beyond the general H3 statement.

**Recommended additions to H3:**
- **H3a:** Transfer efficiency is asymmetric (protein-to-SMILES > SMILES-to-protein)
- **H3b:** Early layers transfer better than late layers in both directions
- **H3c:** Sequence length is a larger transfer barrier than vocabulary/grammar differences (testable via Experiment 6)
- **H3d:** The agent's architectural innovations can be cleanly separated into "universal" (~30-40%) and "domain-specific" (~60-70%) categories

These sub-hypotheses make the paper substantially stronger by providing multiple specific, falsifiable predictions rather than a single vague claim about "partial transfer."

---

*Analysis prepared for recursive-mol project, March 9, 2026*
*Stress-testing H3 of "Self-Directed Discovery of Molecular Transformer Architectures via Recursive Self-Refinement"*
