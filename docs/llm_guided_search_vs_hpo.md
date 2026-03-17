# LLM-Guided Architecture Search vs. Hyperparameter Optimization

This document provides a detailed comparison between the recursive-mol autoresearch approach and traditional hyperparameter optimization frameworks such as Optuna, Ray Tune, or Hyperband. It is intended as source material for the manuscript's related-work and methodology sections.

---

## 1. Nature of the Search Space

### Traditional HPO (Optuna, etc.)

The search space is **predefined and closed**. The researcher declares every tunable dimension upfront:

```python
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    depth = trial.suggest_int("depth", 3, 8)
    activation = trial.suggest_categorical("activation", ["relu", "gelu", "silu"])
    ...
```

The optimizer can only sample within these declared bounds. Adding a new dimension (e.g., a novel gating mechanism) requires the researcher to stop, redesign the search space, and restart.

### recursive-mol (LLM-Guided Search)

The search space is **open-ended and defined by code**. The LLM agent directly edits `train.py` source code, meaning it can:

- Invent entirely new architectural patterns never enumerated in advance
- Compose multiple changes into a single coherent modification
- Introduce structural novelty (e.g., progressive attention windowing schedules, post-branch RMS normalization, auxiliary value-embedding gates with learned attenuation)

The "search space" is effectively the space of all valid Python programs that define a transformer model and train within 5 minutes on a single GPU. This is qualitatively larger and more expressive than any parameterized search space.

**Concrete example from SMILES run_1**: The agent discovered a progressive local attention window schedule (`96/192/192/224` tokens across layers) combined with attenuated value-embedding gates (`0.75x sigmoid`) and post-branch RMS normalization. This composite architectural pattern spans multiple interacting components and would be extremely difficult to express as a Cartesian product of independent hyperparameter dimensions.

---

## 2. Search Strategy: Reasoning vs. Statistical Sampling

### Traditional HPO

Optimizers use principled statistical methods:

- **TPE (Tree-structured Parzen Estimators)**: Models the conditional probability of good vs. bad configurations
- **CMA-ES**: Covariance matrix adaptation for continuous spaces
- **Random search**: Uniform sampling with no learning
- **Bayesian optimization**: Gaussian process surrogate models
- **Hyperband / ASHA**: Early stopping + successive halving for resource-efficient search

All of these treat the model as a **black box**. They observe `(configuration, metric)` pairs and use statistical models to propose the next configuration. They have no understanding of *why* a change helped or hurt.

### recursive-mol

The LLM agent operates as a **white-box reasoner**. It:

1. Reads the current `train.py` source code and understands the architecture
2. Reads the results history (`results.tsv`) and identifies trends
3. Forms hypotheses grounded in architectural understanding (e.g., "local attention windows are helping; let me try a progressive schedule where early layers see narrow context and later layers see wider context")
4. Implements a targeted change and evaluates it
5. Interprets the result in context (e.g., "narrowing from 96 to 64 tokens hurt, so 96 is near-optimal for the first layer")

This is closer to how a human ML researcher works: forming hypotheses, testing them, and updating beliefs based on results. The agent's search is **informed by architectural semantics**, not just statistical correlations between configurations and metrics.

**Evidence from the experiment log**: The agent's trajectory shows coherent exploration strategies. After discovering that wider local attention windows helped (exp011-exp013), it systematically explored the optimal window size per layer (exp032-exp045), eventually converging on a progressive `96/192/192/224` schedule through a series of principled ablations.

---

## 3. Types of Changes: Structural vs. Parametric

### Traditional HPO

Changes are limited to what can be expressed as **numeric or categorical parameters**:

- Learning rate, weight decay, dropout rate (continuous)
- Batch size, depth, width, number of heads (integer)
- Activation function, optimizer type (categorical from a predefined menu)

Even "architecture search" in HPO frameworks typically means choosing from predefined building blocks (e.g., "use 4 or 8 attention heads"), not creating new ones.

### recursive-mol

The agent makes **qualitative code-level transformations** that go far beyond parameter tuning:

| Category | Example from SMILES run_1 | HPO equivalent |
|----------|--------------------------|----------------|
| Activation structure | Replace dense FFN with parameter-matched SwiGLU gating (exp006) | Could be categorical, but parameter-matching requires code changes |
| Attention layout | Progressive local window schedule: 96/192/192/224 (exp032-041) | No natural HPO encoding for per-layer window sizes that interact |
| Value embeddings | Learned value embeddings with attenuated sigmoid gates (exp007, exp040, exp048) | Novel mechanism; not in any standard search space |
| Normalization | RMS-normalize gated MLP hidden activations before output projection (exp056) | Requires inserting new code, not toggling a flag |
| Normalization | RMS-normalize merged attention output before projection (exp057) | Same: structural code insertion |
| Residual pathways | Attempted parallel residual updates (exp061), depth-decayed x0 skip (exp062) | Requires rewriting the forward pass |

Many of these changes introduce entirely new operations (e.g., post-branch RMS normalization) rather than selecting from a menu.

---

## 4. Accumulation Model: Ratcheting vs. Independent Trials

### Traditional HPO

Each trial is typically **independent**:

1. Sample a configuration from the search space
2. Train a model from scratch with that configuration
3. Record the metric
4. Use the result to update the surrogate model / sampling distribution
5. Repeat

The configurations don't build on each other. Trial 50 doesn't inherit the architectural innovations from trial 20. The "state" is in the optimizer's statistical model, not in the model architecture itself.

### recursive-mol

Improvements **accumulate through a ratcheting mechanism**:

1. Start from a baseline `train.py`
2. Make a change and train
3. If `val_bpb` improves: **keep** the modified `train.py` as the new baseline
4. If it worsens: **discard** and revert to the previous best
5. The next experiment starts from the current best

This means experiment 57 (best result: `val_bpb = 0.5925`) incorporates all 13 previously kept improvements:

```
exp001: baseline (0.5964)
  +exp006: SwiGLU gating
    +exp007: learned value embeddings in every layer
      +exp011: wider local attention (192 tokens)
        +exp013: wider local attention (224 tokens)
          +exp032: progressive window schedule
            +exp033: narrower first-layer window (128)
              +exp035: third layer held at 192
                +exp040: attenuated value-embedding gates (0.5x)
                  +exp041: tightest first-layer window (96)
                    +exp048: further gate attenuation (0.75x)
                      +exp056: RMS norm on MLP hidden
                        +exp057: RMS norm on attention output (0.5925)
```

Each kept experiment is a **permanent architectural improvement** that compounds with all previous ones. This is fundamentally different from HPO, where each trial is a point in a static search space.

---

## 5. Handling of Interactions and Compositions

### Traditional HPO

Most HPO methods assume **limited interactions** between parameters. TPE models marginal distributions independently. Grid search treats dimensions as orthogonal. Even methods that model interactions (e.g., SMAC with random forests) struggle with high-order interactions because the number of possible interactions grows combinatorially.

### recursive-mol

The LLM agent naturally handles **compositional changes** because it reads and understands the full source code. It can:

- Recognize that a normalization strategy interacts with a gating mechanism
- Adjust the value-embedding gate range *because* it already changed the attention window schedule
- Understand that tying embeddings is destructive for a small vocabulary (exp016: `val_bpb = 2.50`, correctly discarded)

The agent's search trajectory shows awareness of interactions. For example, after introducing SwiGLU gating (exp006), the agent later specifically tested whether ReluSquared would work as the gate activation (exp019, exp047) — both times finding it harmful, and correctly keeping SiLU. This reflects an understanding that the gate activation interacts with the SwiGLU structure.

---

## 6. Experimental Controls and Baselines

The recursive-mol project explicitly controls for the hypothesis that LLM-guided search adds value over simpler methods by running three baselines:

### A. Random NAS (9 runs: 3 per track)

Random sampling from a fixed architectural search space (depth, width, heads, activation, attention type). 100 independent configurations per run, each trained for 5 minutes. Uses `random_nas.py` with the same `sample_configs()` function as the calibration study.

This is the closest analogue to random search in HPO. It tests whether structured exploration adds value over brute-force sampling.

### B. HP-Only Agent (9 runs: 3 per track)

Same LLM agent loop, but restricted by `program_hponly.md` to only modify hyperparameters:

> "Only modify hyperparameters: learning rate, batch size, dropout, weight decay, warmup steps, optimizer params. Do NOT change model architecture."

This isolates the contribution of **architectural** changes by giving the agent the same reasoning capability but constraining it to the parametric search space.

**Important distinction: HP-Only Agent ≠ Traditional HPO.** The HP-only agent is *not* equivalent to running Optuna or Ray Tune. It is still an LLM-guided search that reads source code, reasons about prior results, and proposes targeted changes through a cumulative ratcheting process. The only restriction is that it cannot modify the model architecture — it must work within the hyperparameter space.

| Aspect | HP-Only Agent (our baseline) | Traditional HPO (Optuna, etc.) |
|--------|------------------------------|-------------------------------|
| Search strategy | LLM reasoning over code + results | Statistical surrogate model (TPE, BO) |
| Accumulation | Ratcheting (keeps best, builds on it) | Independent trials |
| Parameter interactions | Understands how params interact via code reading | Models marginal distributions independently |
| Cost per trial | LLM API call + training | Training only |
| Trial structure | Sequential, informed by full history | Parallel-friendly, informed by surrogate |

The HP-only agent is therefore a **stronger baseline** than traditional HPO — it has the LLM's reasoning advantage applied to hyperparameters. If the full agent (with architectural freedom) outperforms the HP-only agent, that is strong evidence that **architectural changes specifically** (not just better hyperparameter tuning embedded within architectural modifications) drive the improvement. This is a more conservative test than comparing against Optuna, which would lack the LLM's reasoning ability entirely.

### C. Fixed Default (3 runs: 1 per track)

Single run of the unmodified baseline model. Establishes the floor.

### What This Controls For

| Comparison | Tests |
|-----------|-------|
| Agent vs. Random NAS | Does LLM reasoning add value over random sampling? |
| Agent vs. HP-Only Agent | Do architectural changes add value beyond hyperparameter tuning? (Conservative test: both sides use LLM reasoning) |
| Agent vs. Fixed Default | Does any search at all improve on the default? |
| HP-Only Agent vs. Random NAS | Does LLM-guided HP search beat random architecture sampling? |
| HP-Only Agent vs. Fixed Default | Does LLM-guided HP tuning alone improve the baseline? |

**Note on missing baseline:** The project does not include a traditional HPO baseline (e.g., Optuna with TPE). The HP-only agent is a *stronger* control because it uses LLM reasoning, making the Agent vs. HP-Only comparison more conservative. If a reviewer requests an Optuna comparison, the HP-Only Agent results provide an upper bound on what HPO could achieve — any improvement the full agent shows over the HP-only agent would also hold against Optuna.

---

## 7. Calibration: Ensuring Fair Comparison

A key methodological concern is whether 5-minute training runs reliably rank architectures. The project addresses this with a formal calibration study (`calibration.py`):

1. Sample 20 random architecture variants
2. Train each for both 300 seconds (5 min) and 7,200 seconds (2 hours)
3. Compute Spearman rank correlation between short-run and long-run `val_bpb` rankings
4. Decision thresholds:
   - rho > 0.7: proceed confidently
   - rho in [0.4, 0.7]: proceed with caution
   - rho < 0.4: increase the time budget

Result: **rho = 0.541** ("proceed with caution"). The 5-minute proxy is moderately reliable for ranking architectures, which is sufficient for a search process that runs 100 experiments and keeps only improvements.

This calibration step has no analogue in standard HPO — Optuna doesn't validate that its evaluation budget is sufficient for ranking.

---

## 8. Cross-Domain Comparative Design

### Traditional HPO

Optimizes a single objective for a single task. If you want to compare architectures across domains, you run separate HPO campaigns and manually compare results.

### recursive-mol

The same autonomous search loop runs across three domains (SMILES, protein, NLP) with the explicit goal of answering:

> "Do molecular sequences demand fundamentally different architectures than natural language?"

This is a **scientific question**, not an engineering optimization. The search is a means to discover domain-specific architectural patterns, not just to find the best model. The project includes:

- Transfer experiments (train architecture discovered on SMILES, evaluate on protein and vice versa)
- Downstream validation on MoleculeNet benchmarks (BBBP, HIV, BACE)
- Statistical tests to quantify whether domain-specific improvements are significant

---

## 9. Scope Constraints: Architecture of Trust

The recursive-mol system is carefully scoped to enable reliable autonomous operation:

| Constraint | Purpose |
|-----------|---------|
| Only `train.py` is editable | Prevents the agent from gaming the evaluation |
| Fixed metric (`val_bpb`) | Ensures all experiments are comparable |
| Fixed time budget (300 seconds) | Wall-clock parity across all experiments |
| Frozen data splits | No data leakage or split manipulation |
| Frozen tokenizers and data loaders | Isolates architectural improvements |
| Single-model evaluation | No ensembling tricks |
| Deterministic train/val split | Reproducibility |

These constraints are enforced by the infrastructure (`session_tools.py`, `phase2_runner.py`), not by the agent's good behavior. The agent literally cannot modify `prepare_smiles.py` or the evaluation code.

---

## 10. Summary Table

| Dimension | Traditional HPO (Optuna) | recursive-mol (LLM-Guided) |
|-----------|-------------------------|---------------------------|
| **Search space** | Predefined, closed, parameterized | Open-ended, defined by code |
| **Search strategy** | Statistical (TPE, BO, random) | Reasoning over code and results |
| **Change granularity** | Numeric/categorical parameters | Source code transformations |
| **Trial independence** | Independent trials | Cumulative ratcheting |
| **Interaction handling** | Limited (marginal models) | Natural (reads full code) |
| **Novel structures** | Cannot create new operations | Can invent new mechanisms |
| **Domain awareness** | Black-box | Reads and understands architecture |
| **Cross-domain design** | Manual comparison | Built-in multi-track comparison |
| **Calibration** | Assumed sufficient | Formally validated (Spearman rho) |
| **Baselines** | Often missing | Random NAS + HP-only + fixed default |
| **Goal** | Find best configuration | Discover domain-specific architectures |

---

## 11. Relationship to Neural Architecture Search (NAS)

The recursive-mol approach also differs from traditional NAS methods:

- **DARTS / differentiable NAS**: Optimizes architecture parameters via gradient descent on a continuous relaxation. Limited to predefined cell structures and operation choices.
- **ENAS / weight-sharing NAS**: Shares weights across candidate architectures to amortize training cost. Requires a fixed super-network.
- **Evolutionary NAS**: Mutates and recombines architectures. Closer in spirit to recursive-mol's ratcheting, but mutations are random and predefined, not semantically informed.

The LLM agent is closest to an **evolutionary approach with semantically informed mutations**, but with the crucial difference that the "mutation operator" understands what the code does and can make targeted, compositional changes that would require many random mutations to discover.

---

## 12. Limitations and Caveats

For a balanced manuscript, the following limitations should be acknowledged:

1. **Reproducibility**: The LLM agent is stochastic and its behavior depends on the model version, prompt, and context window. Two runs of the same agent on the same track may explore different trajectories.
2. **Cost**: Each agent step involves an LLM API call (20-minute timeout), making the per-experiment cost higher than HPO, which only requires training compute.
3. **Scalability**: The approach is limited by LLM context window and reasoning capability. It may not scale to very large codebases or very complex architectural search spaces.
4. **Confounding**: The agent's improvements may partly reflect better hyperparameter tuning embedded within architectural changes. The HP-only baseline controls for this, but the separation is not perfect.
5. **Calibration uncertainty**: The Spearman rho of 0.541 means some architectural rankings may be unreliable under the 5-minute budget. Improvements that appear marginal (< 0.001 val_bpb) may not hold at longer training horizons.
6. **Single metric**: Optimizing only `val_bpb` may miss architectures that are better on downstream tasks but slightly worse on next-token prediction.
