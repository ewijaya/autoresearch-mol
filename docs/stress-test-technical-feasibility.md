# Technical Feasibility Stress-Test: Recursive Self-Refinement for Molecular Transformers

**Date:** 2026-03-09
**Project:** Fork of karpathy/autoresearch for autonomous architecture search on molecular data
**Hardware target:** g5.xlarge (NVIDIA A10G, 24 GB VRAM)
**Time budget per experiment:** 5 minutes (300 seconds wall-clock training time)

---

## Table of Contents

1. [Autoresearch Baseline Analysis](#1-autoresearch-baseline-analysis)
2. [Role 1 -- Data Engineer: SMILES Data Pipeline](#2-role-1----data-engineer-smiles-data-pipeline)
3. [Role 2 -- Architect: Model Sizing and Feasibility](#3-role-2----architect-model-sizing-and-feasibility)
4. [Role 3 -- Protein Specialist: UniRef50 Track](#4-role-3----protein-specialist-uniref50-track)
5. [Cross-Role Synthesis: Tokenization Strategy](#5-cross-role-synthesis-tokenization-strategy)
6. [Feasibility Assessment and Recommendations](#6-feasibility-assessment-and-recommendations)

---

## 1. Autoresearch Baseline Analysis

### Codebase Structure

The autoresearch repo consists of three files:

| File | Purpose | Editable? |
|------|---------|-----------|
| `prepare.py` | Data download, BPE tokenizer training, dataloader, `evaluate_bpb` | No (fixed) |
| `train.py` | GPT model, MuonAdamW optimizer, training loop | Yes (agent edits) |
| `program.md` | Agent instructions for the experiment loop | Human edits |

### Core Loop

The agent loop is: **modify train.py -> train 5 min -> measure val_bpb -> keep/discard -> repeat**. The `val_bpb` metric (bits per byte) is computed in `evaluate_bpb()` which sums per-token cross-entropy (in nats), sums target byte lengths, then converts nats/byte to bits/byte. Special tokens (byte length 0) are excluded.

### Default Architecture (from `train.py`)

```
GPTConfig defaults:
  sequence_len:   2048
  vocab_size:     32768 (overridden by tokenizer; actual default BPE = ~8196)
  n_layer:        12
  n_head:         6
  n_kv_head:      6
  n_embd:         768
  window_pattern: "SSSL"
```

**However**, the actual build uses hyperparameters at the bottom of train.py:

```
DEPTH = 8
ASPECT_RATIO = 64  -> base_dim = 8 * 64 = 512
HEAD_DIM = 128     -> model_dim = 512, num_heads = 4
```

### Computed Default Model Size

| Component | Parameters |
|-----------|-----------|
| Token embedding (wte) | 4,194,304 |
| LM head (unembedding) | 4,194,304 |
| Transformer blocks (8 layers) | 25,165,824 |
| Value embeddings (4 alternating layers) | 16,777,216 |
| VE gates | 512 |
| Per-layer scalars | 16 |
| **Total** | **~50.3M** |

### Key Design Decisions in Baseline

- **Optimizer:** MuonAdamW -- Muon (orthogonalized momentum) for 2D matrix params, AdamW for embeddings/scalars
- **Activation:** ReluSquared (`F.relu(x).square()`) in MLP
- **Normalization:** RMSNorm (applied pre-attention, pre-MLP, and post-final-block)
- **Attention:** Flash Attention 3 with sliding window pattern (SSSL = 3 short + 1 long per 4 layers)
- **Value Embedding (ResFormer):** Alternating layers have a separate value embedding with input-dependent gating
- **Logit softcap:** 15
- **Total batch size:** 524,288 tokens (~0.5M per optimizer step)
- **Data:** FineWeb-Edu (via climbmix-400b-shuffle), BPE tokenizer with vocab_size=8192

### Data Pipeline (from `prepare.py`)

- Downloads parquet shards from HuggingFace (`karpathy/climbmix-400b-shuffle`)
- Trains BPE tokenizer using `rustbpe` with GPT-4-style split pattern
- BOS-aligned dataloader with best-fit packing (100% utilization, no padding)
- Evaluation: fixed `EVAL_TOKENS = 40 * 524288` (~21M tokens) from pinned val shard
- `token_bytes.pt` lookup for BPB computation: maps each token ID to its UTF-8 byte length

---

## 2. Role 1 -- Data Engineer: SMILES Data Pipeline

### 2.1 Where to Download ZINC-250K

**Primary sources:**

| Source | Format | Access Method | Notes |
|--------|--------|---------------|-------|
| ZINC Database (zinc15.docking.org) | SMILES text files | Direct HTTP/wget | Canonical source, requires navigating subsets |
| RDKit Chem.PandasTools | SDF/SMILES | `pip install rdkit` | Can read ZINC SDF files |
| PyTorch Geometric (ZINC dataset) | Pre-processed graph | `torch_geometric.datasets.ZINC` | Graph-level, but includes SMILES strings |
| TDC (Therapeutics Data Commons) | SMILES CSV | `pip install PyTDC; from tdc import Oracle` | Clean, well-documented subset |
| MoleculeNet (via DeepChem) | SMILES CSV | `deepchem.molnet.load_zinc15` | Standard benchmark format |
| Kaggle / HuggingFace | Various CSV/parquet | Direct download | Community-uploaded copies |

**Recommendation:** Use the ZINC-250K subset from MoleculeNet/DeepChem or the widely-used `250k_rndm_zinc_drugs_clean_3.csv` file that is standard in the molecular generation literature (used by JT-VAE, CGVAE, MolGAN, etc.). This file contains 249,455 SMILES strings, pre-cleaned and drug-like.

### 2.2 Character-Level vs BPE Tokenizer for SMILES

| Criterion | Character-Level | BPE |
|-----------|----------------|-----|
| **Vocabulary size** | ~40-60 tokens | 500-8000 tokens (configurable) |
| **Chemical validity** | Every character is a valid SMILES atom/bond/bracket token | Subword tokens may split across chemical boundaries (e.g., `Cl` into `C` + `l`) |
| **Sequence length** | ~50 chars/molecule avg | ~15-25 tokens/molecule avg |
| **Semantic alignment** | Each token is a chemically meaningful unit | Tokens may span arbitrary substructures |
| **val_bpb interpretability** | BPB directly measures character-level prediction | BPB is normalized by byte length, so still comparable |
| **Literature precedent** | Standard for SMILES generation (CharRNN, REINVENT, etc.) | Rare for SMILES; more common in protein language models |
| **Overfitting risk** | Higher (longer sequences = more tokens to memorize) | Lower (shorter sequences) |
| **Ring/branch notation** | `(`, `)`, digits `1-9`, `%10-%99` are individual tokens | May merge ring tokens with adjacent atoms |

**Verdict:** Character-level tokenization is strongly preferred for SMILES. The SMILES grammar is character-level by design -- each character or small character group (like `Cl`, `Br`, `[NH]`) maps to a specific chemical entity. BPE would create semantically incoherent subword units. The SMILES generation literature universally uses character-level tokenization.

**Caveat:** A SMILES-aware tokenizer (splitting on atoms rather than raw characters) would be even better than naive character-level. For example, `Cl` should be one token, not two. The SELFIES representation could also be considered as an alternative to SMILES.

### 2.3 SMILES Token Count Estimate

```
ZINC-250K statistics:
  - 249,455 molecules
  - Average SMILES length: ~50 characters (range: ~10 to ~200)
  - Median SMILES length: ~45 characters
  - Total characters: 249,455 * 50 = ~12.5M characters

With BOS tokens:
  - 249,455 BOS tokens + 12.5M character tokens = ~12.75M tokens
```

At character level, 1 character = 1 token = 1 byte (SMILES is pure ASCII), so:
- **~12.5M tokens total**
- **~12.5M bytes total**

### 2.4 SMILES Vocabulary

The SMILES character set appearing in drug-like molecules (ZINC-250K):

**Atoms (organic subset):** `C`, `c`, `N`, `n`, `O`, `o`, `S`, `s`, `F`, `P`, `p`, `B`, `I`
**Two-character atoms (treated as single token in SMILES-aware tokenizer):** `Cl`, `Br`, `Si`, `Se`
**Bonds:** `-`, `=`, `#`, `:` (aromatic)
**Branching:** `(`, `)`
**Ring closures:** `0`-`9`, `%` (for ring closure digits >= 10)
**Brackets:** `[`, `]`
**Inside brackets:** `+`, `-`, `H`, digits, `@`, `@@`
**Stereochemistry:** `/`, `\`
**Charges:** `+`, `-`
**Misc:** `.` (disconnected fragments -- rare in ZINC)

**Estimated vocabulary sizes:**

| Tokenization | Vocab Size | Notes |
|-------------|-----------|-------|
| Raw character-level | ~40-45 unique chars | Every single character is a token |
| SMILES-aware atom-level | ~55-65 tokens | `Cl`, `Br` as single tokens; bracket groups intact |
| BPE (500 merges) | ~550 tokens | Learns common substructures |
| BPE (2000 merges) | ~2050 tokens | Learns larger fragments |

For ZINC-250K specifically, raw character-level yields approximately **40-45 unique characters**.

### 2.5 Does val_bpb Still Make Sense with a Tiny Vocabulary?

**Yes, but with important nuances.**

The BPB metric is:

```
val_bpb = total_nats / (ln(2) * total_bytes)
```

Where `total_nats` = sum of per-token cross-entropy losses (in nats), and `total_bytes` = sum of UTF-8 byte lengths of target tokens.

For character-level SMILES where every token is 1 byte:
- `val_bpb = mean_nats_per_token / ln(2)` = mean cross-entropy in bits per token = **bits per character**
- This is directly the character-level entropy of the model's predictions

**Why it still works:**
- BPB is explicitly designed to be vocabulary-size independent
- With ~40 chars, the theoretical maximum entropy is `log2(40) = 5.3 bits`, and a uniform random baseline gives `val_bpb = 5.3`
- A good character-level SMILES model should achieve `val_bpb` around 1.0-2.0 (SMILES has strong syntactic patterns)
- The metric is directly comparable across character-level and BPE tokenizations

**What changes:**
- The absolute BPB values will be different from the NLP track (FineWeb-Edu), so cross-track comparison requires care
- With a tiny vocab, the softmax is cheap but the embedding table is also tiny -- this shifts the parameter budget toward transformer layers, which is actually desirable

### 2.6 Train/Val Split Strategy

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Random split** | Random 90/10 or 80/20 split of molecules | Simple, standard | May leak structural information (similar scaffolds in both sets) |
| **Scaffold split** | Split by Bemis-Murcko scaffolds | Tests generalization to novel scaffolds | Harder to implement; may create distribution shift |
| **Time-based split** | If molecules have timestamps | Realistic for prospective discovery | ZINC-250K doesn't have clear timestamps |

**Recommendation for this project:** Use **random split (90/10)** for simplicity and to maximize signal in the 5-minute training window. The goal is architecture search, not evaluating generalization to novel scaffolds. Scaffold split is more appropriate for downstream molecular property prediction tasks.

For the val set: 10% = ~25K molecules = ~1.25M tokens. This is small but sufficient for detecting meaningful BPB differences (the autoresearch default uses ~21M eval tokens, but we can adjust `EVAL_TOKENS` proportionally).

### 2.7 Pseudocode for `prepare.py` Replacement

```python
"""
prepare.py for SMILES molecular data (ZINC-250K).
Drop-in replacement for the autoresearch prepare.py.
"""

import os, math, torch, pickle, csv
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = 512        # SMILES are short; 512 >> longest SMILES in ZINC
TIME_BUDGET = 300         # 5 minutes
EVAL_TOKENS = 10 * 131072  # ~1.3M tokens for val eval (scaled down)

CACHE_DIR = os.path.expanduser("~/.cache/recursive-mol")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

ZINC_URL = ("https://raw.githubusercontent.com/aspuru-guzik-group/"
            "chemical_vae/master/models/zinc_properties/"
            "250k_rndm_zinc_drugs_clean_3.csv")

def download_zinc():
    """Download ZINC-250K SMILES dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, "zinc250k.csv")
    if os.path.exists(filepath):
        print(f"Data: already downloaded at {filepath}")
        return filepath
    import requests
    print("Downloading ZINC-250K...")
    resp = requests.get(ZINC_URL)
    resp.raise_for_status()
    with open(filepath, "w") as f:
        f.write(resp.text)
    print(f"Data: saved to {filepath}")
    return filepath

def load_smiles(filepath: str) -> List[str]:
    """Load SMILES strings from ZINC CSV."""
    smiles_list = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get("smiles", row.get("SMILES", "")).strip()
            if smi:
                smiles_list.append(smi)
    return smiles_list

def split_data(smiles_list, val_ratio=0.1, seed=42):
    """Random train/val split."""
    import random
    rng = random.Random(seed)
    indices = list(range(len(smiles_list)))
    rng.shuffle(indices)
    split = int(len(indices) * (1 - val_ratio))
    train_idx = indices[:split]
    val_idx = indices[split:]
    train = [smiles_list[i] for i in train_idx]
    val = [smiles_list[i] for i in val_idx]
    return train, val

# ---------------------------------------------------------------------------
# Character-level tokenizer
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

class SMILESTokenizer:
    """Character-level SMILES tokenizer."""

    def __init__(self, char_to_id: dict, id_to_char: dict):
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char
        self.pad_id = char_to_id["<PAD>"]
        self.bos_id = char_to_id["<BOS>"]
        self.eos_id = char_to_id["<EOS>"]
        self.unk_id = char_to_id["<UNK>"]

    @classmethod
    def build_from_corpus(cls, smiles_list: List[str]):
        """Build vocabulary from SMILES corpus."""
        chars = set()
        for smi in smiles_list:
            chars.update(smi)
        sorted_chars = sorted(chars)
        char_to_id = {}
        for i, tok in enumerate(SPECIAL_TOKENS):
            char_to_id[tok] = i
        for i, ch in enumerate(sorted_chars):
            char_to_id[ch] = len(SPECIAL_TOKENS) + i
        id_to_char = {v: k for k, v in char_to_id.items()}
        return cls(char_to_id, id_to_char)

    def get_vocab_size(self):
        return len(self.char_to_id)

    def get_bos_token_id(self):
        return self.bos_id

    def encode(self, text, prepend=None):
        """Encode a SMILES string or list of strings."""
        if isinstance(text, str):
            ids = [self.char_to_id.get(c, self.unk_id) for c in text]
            if prepend is not None:
                ids.insert(0, prepend if isinstance(prepend, int)
                           else self.bos_id)
            return ids
        elif isinstance(text, list):
            return [self.encode(s, prepend=prepend) for s in text]

    def decode(self, ids):
        return "".join(self.id_to_char.get(i, "?") for i in ids)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "tokenizer.pkl"), "wb") as f:
            pickle.dump((self.char_to_id, self.id_to_char), f)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            char_to_id, id_to_char = pickle.load(f)
        return cls(char_to_id, id_to_char)

# Alias for compatibility with train.py imports
Tokenizer = SMILESTokenizer

# ---------------------------------------------------------------------------
# Dataloader (mirrors autoresearch BOS-aligned packing)
# ---------------------------------------------------------------------------

def _document_batches(split_data_list, tokenizer):
    """Infinite iterator over tokenized SMILES documents."""
    bos = tokenizer.get_bos_token_id()
    epoch = 1
    while True:
        for smi in split_data_list:
            yield tokenizer.encode(smi, prepend=bos), epoch
        epoch += 1

def make_dataloader(tokenizer, B, T, split):
    """
    Simplified BOS-aligned dataloader for SMILES.
    Packs multiple SMILES into rows of length T+1.
    """
    split_file = os.path.join(DATA_DIR, f"{split}_smiles.pkl")
    with open(split_file, "rb") as f:
        split_data_list = pickle.load(f)

    row_capacity = T + 1
    docs = _document_batches(split_data_list, tokenizer)
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    doc_buffer = []
    epoch = 1

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < 100:
                    tokens, epoch = next(docs)
                    doc_buffer.append(tokens)
                remaining = row_capacity - pos
                best_idx, best_len = -1, 0
                for i, doc in enumerate(doc_buffer):
                    if len(doc) <= remaining and len(doc) > best_len:
                        best_idx, best_len = i, len(doc)
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos+len(doc)] = torch.tensor(doc)
                    pos += len(doc)
                else:
                    doc = doc_buffer.pop(0)
                    row_buffer[row_idx, pos:pos+remaining] = \
                        torch.tensor(doc[:remaining])
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (BPB for character-level SMILES)
# ---------------------------------------------------------------------------

def get_token_bytes(device="cpu"):
    """For char-level SMILES: every non-special token = 1 byte (ASCII)."""
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    return torch.load(path, map_location=device)

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """BPB evaluation -- same formula as autoresearch."""
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Step 1: Download
    filepath = download_zinc()
    smiles_list = load_smiles(filepath)
    print(f"Loaded {len(smiles_list)} SMILES")

    # Step 2: Split
    train_smiles, val_smiles = split_data(smiles_list)
    print(f"Train: {len(train_smiles)}, Val: {len(val_smiles)}")

    # Save splits
    for name, data in [("train", train_smiles), ("val", val_smiles)]:
        with open(os.path.join(DATA_DIR, f"{name}_smiles.pkl"), "wb") as f:
            pickle.dump(data, f)

    # Step 3: Build tokenizer
    tokenizer = SMILESTokenizer.build_from_corpus(smiles_list)
    tokenizer.save(TOKENIZER_DIR)
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Step 4: Build token_bytes lookup
    token_bytes = []
    for i in range(tokenizer.get_vocab_size()):
        ch = tokenizer.id_to_char.get(i, "")
        if ch in SPECIAL_TOKENS:
            token_bytes.append(0)  # exclude specials from BPB
        else:
            token_bytes.append(1)  # every SMILES char = 1 byte (ASCII)
    torch.save(torch.tensor(token_bytes, dtype=torch.int32),
               os.path.join(TOKENIZER_DIR, "token_bytes.pt"))

    # Stats
    total_chars = sum(len(s) for s in smiles_list)
    avg_len = total_chars / len(smiles_list)
    print(f"Total characters: {total_chars:,}")
    print(f"Average SMILES length: {avg_len:.1f} chars")
    print(f"Max SMILES length: {max(len(s) for s in smiles_list)}")
    print("Done! Ready to train.")
```

---

## 3. Role 2 -- Architect: Model Sizing and Feasibility

### 3.1 Default Model Size Recap

With the default `DEPTH=8, ASPECT_RATIO=64`:
- **n_embd:** 512
- **n_head:** 4 (head_dim=128)
- **n_layer:** 8
- **Total params:** ~50.3M (with vocab_size=8192)

A massive proportion of parameters (~50%) is in embeddings and value embeddings. With SMILES vocab of ~45:
- **Embedding overhead drops from ~25M to ~0.1M**
- **Effective model:** ~25.4M params (almost entirely in transformer layers)

### 3.2 Compute Budget on A10G

**A10G specs:**
- bf16 peak: 31.2 TFLOPS
- VRAM: 24 GB
- Memory bandwidth: 600 GB/s

**Compared to H100 (autoresearch's target):**
- H100 bf16 peak: 989.5 TFLOPS
- **A10G is ~31.7x slower** in raw compute

**MFU expectations on A10G:**
- Flash Attention 3 (`kernels` package) targets Hopper (H100); the code falls back to `kernels-community/flash-attn3` for non-Hopper GPUs
- A10G is Ampere (sm_86), not Hopper (sm_90) -- FA3 may not be supported at all
- **Critical risk:** The FA3 kernel may crash on A10G. Fallback to `torch.nn.functional.scaled_dot_product_attention` would be needed
- Realistic MFU on A10G with standard attention: **20-30%**

### 3.3 Will the Model Overfit ZINC-250K?

**Throughput estimates (A10G, 30% MFU):**

| Model Config | Params | tok/sec | Tokens in 5 min | Epochs over 12.5M |
|-------------|--------|---------|-----------------|-------------------|
| Original (vocab=8192) | 50.3M | ~31K | ~9.3M | 0.7 |
| SMILES char (vocab=45) | 25.4M | ~61K | ~18.4M | 1.5 |
| Small (depth=4, dim=256) | ~6M | ~260K | ~78M | 6.2 |
| Tiny (depth=4, dim=128) | ~2M | ~780K | ~234M | 18.7 |

**Analysis:**

- With the **original 50.3M model**, we barely see the data once (~0.7 epochs). The model will be massively undertrained, not overfitting. val_bpb will be noisy and may not converge.
- With **SMILES char vocab** (25.4M params), we get ~1.5 epochs. Still borderline for meaningful convergence.
- At **6M params**, we get ~6 epochs -- enough to see real learning and convergence signal.
- At **2M params**, we get ~19 epochs -- this will likely overfit without regularization.

**Overfitting threshold:** With 12.5M tokens, a model with more than ~5M parameters is likely to start memorizing if trained for many epochs. However, at 1-2 epochs, even a 25M param model won't overfit -- it won't even converge.

### 3.4 Is 1-5 Epochs Enough for Meaningful val_bpb Signal?

**For the architecture search loop to work, we need:**
1. val_bpb to be in a regime where it discriminates between architectures
2. Changes of ~0.01 BPB to be detectable above noise

**At 1-2 epochs:** The model is in the rapid learning phase. val_bpb will still be improving. Architecture differences may be visible but will be confounded with "how fast does this architecture learn" rather than "how well does this architecture model the data."

**At 5+ epochs:** val_bpb starts to plateau. Architecture differences become clearer. This is the regime we want.

**Recommendation:**
- **Target 3-8 epochs** for the architecture search to produce reliable signal
- This means keeping the model small enough (6-15M params) to process 40-100M tokens in 5 minutes
- Or augmenting the dataset (more molecules, data augmentation via SMILES enumeration)

### 3.5 Should We Adjust Dataset Size or Model Size?

**Option A: Increase dataset size**
- ZINC-250K is actually small. ZINC-15 has 1.4B molecules. Even a 1M molecule subset would give ~50M tokens.
- SMILES enumeration: each molecule has multiple valid SMILES (via different atom orderings). This can 5-10x the dataset effectively.
- GuacaMol benchmark set: ~1.6M molecules from ChEMBL
- Combined ZINC-250K + ChEMBL subset: ~2M molecules = ~100M tokens

**Option B: Reduce model size**
- Depth 4-6, dim 256-384, heads 2-3
- Target 5-15M params for the sweet spot on A10G with ZINC-250K
- This is still large enough to learn meaningful SMILES patterns

**Option C: Both** (recommended)
- Use ZINC-250K with SMILES enumeration (5 random SMILES per molecule = ~62.5M tokens)
- Use a 6-12M param model
- Result: 3-8 epochs, good convergence, meaningful architecture search signal

### 3.6 SMILES-Specific Architectural Features to Search

| Feature | Rationale | Complexity |
|---------|-----------|-----------|
| **Reduced sequence length** | SMILES are short (~50 chars). MAX_SEQ_LEN=512 instead of 2048. Reduces memory, allows larger batches. | Low |
| **Smaller vocab embedding** | Vocab ~45 vs 8192. Tiny embedding table, more params in layers. | Low |
| **Local attention / short windows** | Bond adjacency is local in SMILES. Short attention windows (32-64) may suffice for most patterns. | Medium |
| **Ring-closure positional encoding** | Ring closure digits (e.g., `C1CCC1`) create long-range dependencies. Special positional signal for matched ring digits. | High |
| **Bracket-aware attention bias** | Bracket groups `[NH]` are single semantic units. Attention bias for within-bracket tokens. | Medium |
| **Branch depth encoding** | Parentheses create a tree structure. Encoding branch depth as a positional feature. | Medium |
| **SMILES syntax-constrained decoding** | At inference time, mask invalid next-character predictions. Doesn't affect training val_bpb. | N/A for training |
| **Relative position encoding for bonded atoms** | Atoms bonded in the molecule may be far apart in SMILES (due to branching). Graph-distance positional encoding. | High |
| **ReluSquared -> GELU or SiLU** | Default uses ReluSquared. Other activations may work better for this domain. | Low |
| **Deeper, narrower model** | SMILES patterns may benefit from more layers (compositional structure) with smaller hidden dim. | Low |

**Highest-value changes for the architecture search loop:**
1. Reduce `MAX_SEQ_LEN` to 256 or 512 (immediately faster training)
2. Reduce model size to 6-12M params (more epochs per run)
3. Experiment with window sizes (SMILES locality)
4. Deeper but narrower (more layers, smaller dim)

### 3.7 Flash Attention 3 on A10G -- Critical Compatibility Issue

The autoresearch code uses:
```python
cap = torch.cuda.get_device_capability()
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface
```

A10G has compute capability `(8, 6)`. The `kernels-community/flash-attn3` repo targets the `kernels` ecosystem. **FA3 is designed for Hopper (sm_90).** On Ampere (A10G), this will likely fail.

**Required fix for A10G:**
- Replace FA3 with `torch.nn.functional.scaled_dot_product_attention` (SDPA), which automatically dispatches to Flash Attention 2 on Ampere
- Or use `flash-attn` (v2) package, which supports sm_80+
- This is a **blocking issue** that must be resolved before any training can happen

### 3.8 Batch Size and Memory

With a 6M param model on A10G (24 GB):
- Model memory: ~12 MB (bf16)
- Optimizer states: ~72 MB
- Activations at batch_size=128, seq_len=512: ~2-4 GB (rough estimate)
- **Plenty of VRAM headroom** -- can increase batch size significantly

Recommended: `DEVICE_BATCH_SIZE=256` or higher, `TOTAL_BATCH_SIZE=65536` (reduced from 524K since dataset is smaller).

---

## 4. Role 3 -- Protein Specialist: UniRef50 Track

### 4.1 How to Subset UniRef50

**UniRef50 overview:**
- ~53M cluster representative sequences (as of 2025)
- Total size: ~25 GB compressed FASTA
- Available from UniProt: `https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz`

**Subsetting strategies:**

| Strategy | Sequences | Tokens (est.) | Notes |
|----------|-----------|---------------|-------|
| Random 10K | 10,000 | ~3M | Very small, fast iteration |
| Random 50K | 50,000 | ~15M | Similar scale to ZINC-250K |
| Random 200K | 200,000 | ~60M | Good for A10G throughput |
| Length-filtered (<200 aa) random 100K | 100,000 | ~10M | Short proteins, fast training |
| Swiss-Prot reviewed subset | ~570K | ~170M | High-quality, well-annotated |
| Pfam domain sequences | Variable | Variable | Focused on protein domains |

**Recommendation:** Start with **50K random sequences** filtered to length < 500 amino acids. This gives ~15M tokens, comparable to ZINC-250K's ~12.5M tokens and enabling direct throughput comparison.

**Practical subsetting:**
```bash
# Download and subsample
wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz
zcat uniref50.fasta.gz | awk '/^>/{n++} n<=50000' > uniref50_50k.fasta
```

Or use BioPython / `pyfaidx` for more controlled filtering.

### 4.2 Amino Acid Alphabet

**Standard amino acids (20):** `A, R, N, D, C, E, Q, G, H, I, L, K, M, F, P, S, T, W, Y, V`

**Non-standard / ambiguous (in UniRef50):**
- `U` -- Selenocysteine (rare, ~0.01% of sequences)
- `O` -- Pyrrolysine (extremely rare)
- `B` -- Asparagine or Aspartic acid (ambiguous)
- `Z` -- Glutamine or Glutamic acid (ambiguous)
- `X` -- Unknown amino acid
- `J` -- Leucine or Isoleucine (ambiguous)

**Special tokens for our tokenizer:**
- `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`

**Total vocabulary: ~30 tokens** (20 standard + 6 non-standard + 4 special)

This is even smaller than SMILES (~45 tokens).

### 4.3 Character-Level Tokenization for Proteins

**Character-level tokenization is the natural choice for proteins.** Each character already represents one amino acid, which is the fundamental biological unit. Unlike NLP text where characters are sub-semantic (individual letters of words), each protein "character" carries full semantic meaning:

- Each amino acid has distinct chemical properties (size, charge, hydrophobicity)
- The sequence of amino acids directly determines 3D structure and function
- There is no "subword" analog -- amino acids are already the atoms of protein language

**Literature precedent:** All major protein language models (ESM, ESM-2, ProtTrans, ProGen) use single amino acid tokenization. Some add special tokens for species tags or functional annotations, but the core is always one-token-per-residue.

**BPE for proteins?** Some work (e.g., ProtGPT2) has explored BPE on protein sequences. The learned merges often correspond to common dipeptides/tripeptides (e.g., `ALA-GLY`, `LEU-ALA`). This can reduce sequence length by ~2-3x but introduces tokens that are harder to interpret biochemically. For our architecture search, character-level is strongly preferred.

### 4.4 Protein Sequence Length Distribution

**UniRef50 statistics (approximate):**

| Metric | Value |
|--------|-------|
| Mean length | ~300-350 residues |
| Median length | ~250 residues |
| Mode | ~100-150 residues |
| 10th percentile | ~80 residues |
| 25th percentile | ~140 residues |
| 75th percentile | ~420 residues |
| 90th percentile | ~700 residues |
| 99th percentile | ~2000 residues |
| Maximum | ~36,000 residues (titin-like) |

**Key observation:** The distribution is heavily right-skewed. Most proteins are 100-500 residues, but extreme outliers (multi-domain proteins, titin) can be thousands of residues long.

### 4.5 Comparison with SMILES Length Distribution

| Metric | SMILES (ZINC-250K) | Protein (UniRef50) | Ratio (Protein/SMILES) |
|--------|-------------------|-------------------|----------------------|
| Mean length | ~50 chars | ~300-350 residues | 6-7x |
| Median | ~45 chars | ~250 residues | 5-6x |
| Max | ~200 chars | ~36,000 residues | 180x |
| Typical range | 20-100 | 50-1000 | 5-10x |
| Vocab size | ~45 tokens | ~30 tokens | 0.7x |
| Tokens per molecule/protein | ~50 | ~300-350 | 6-7x |

**Implications for training:**
- Protein sequences are **6-7x longer** on average than SMILES
- With the same number of sequences, proteins produce **6-7x more tokens**
- Longer sequences require more memory per batch element (quadratic attention cost)
- Fewer sequences fit in a batch with fixed VRAM

### 4.6 Can We Filter to Short Proteins?

**Yes, and we should.** Filtering to proteins < 200 residues:
- Captures ~35-40% of UniRef50
- Average length drops to ~120 residues
- Still 2-3x longer than SMILES
- Includes many biologically important proteins (small enzymes, peptide hormones, toxins, antimicrobial peptides)

**Recommended filtering for the protein track:**
1. Filter UniRef50 to sequences with 50-500 residues (excludes fragments and giants)
2. Random sample 50K sequences
3. Expected total tokens: 50K * 250 avg = ~12.5M tokens (matches SMILES track)

**MAX_SEQ_LEN for protein track:** 512 is reasonable (covers up to ~500 residue proteins with BOS/EOS). Longer proteins get truncated, which is acceptable for architecture search.

### 4.7 Protein-Specific Architectural Features

| Feature | Rationale | Difficulty |
|---------|-----------|-----------|
| **Contact prediction attention** | Protein structure has long-range contacts. Attention patterns should learn to predict which residues are spatially close. | Emergent (ESM shows this naturally) |
| **Secondary structure bias** | Alpha-helices have ~3.6 residue periodicity. Periodic positional encoding could help. | Medium |
| **Hydrophobicity encoding** | Embedding each amino acid with its physicochemical properties (hydrophobicity, charge, size) as auxiliary features. | Low |
| **Evolutionary coupling** | Co-evolving positions in protein families. Would require MSA input, out of scope for single-sequence modeling. | Out of scope |
| **Local window = 3-7** | Local sequence patterns (secondary structure motifs) span 3-7 residues. Short attention windows capture helices and beta strands. | Low |
| **Longer context attention** | Long-range contacts (beta sheets, disulfide bonds) require full attention. Mix of local + global attention (already in SSSL pattern). | Already supported |
| **Amino acid property embeddings** | Instead of learned embeddings, initialize with biophysical properties (AAindex). 553 published amino acid indices. | Low |
| **Sinusoidal positional encoding with helix period** | Add a sinusoidal feature with period 3.6 (alpha-helix pitch) to positional encoding. | Low-Medium |

**Highest-value changes:**
1. Physicochemical property-augmented embeddings (nearly free, potentially helpful)
2. Experiment with window sizes tailored to secondary structure motifs
3. Test whether deeper models (more layers) capture long-range contacts better

---

## 5. Cross-Role Synthesis: Tokenization Strategy

### 5.1 Shared Findings

All three tracks use fundamentally different tokenization, which affects every aspect of the pipeline:

| Aspect | Track A: SMILES | Track B: Protein | Track C: NLP (FineWeb-Edu) |
|--------|----------------|-----------------|---------------------------|
| Tokenization | Character-level | Character-level | BPE (8192 vocab) |
| Vocab size | ~45 | ~30 | ~8196 |
| Avg sequence length | ~50 tokens | ~250-350 tokens | ~2048 tokens (packed) |
| Token semantics | 1 token = 1 SMILES char | 1 token = 1 amino acid | 1 token = subword (~4 chars) |
| BPB interpretation | Bits per SMILES character | Bits per amino acid character | Bits per UTF-8 byte |
| Bytes per token | 1 (ASCII) | 1 (ASCII) | Variable (~3-5 avg) |
| Theoretical max entropy | log2(45) = 5.5 bits | log2(30) = 4.9 bits | log2(8196) = 13.0 bits, but BPB normalizes by bytes |

### 5.2 Architect's Comment on Data Engineer's Dataset Sizes

**The 12.5M token ZINC-250K dataset is workable but tight.** Here is the core tension:

- With the default 50.3M param model on A10G, we only see ~0.7 epochs in 5 minutes. This is insufficient for the architecture search loop to produce meaningful signal.
- Scaling down to 6-12M params gives us 3-8 epochs, which is the sweet spot.
- However, a 6M param model is extremely small for a transformer. The architecture search space shrinks because many sophisticated features (deep attention patterns, multi-head diversity) don't emerge at this scale.

**Recommendation from the architect to the data engineer:**
- **Augment ZINC-250K with SMILES enumeration** to 50-100M tokens. This is the single most impactful data engineering decision.
- Alternatively, use ZINC-1M or ZINC-2M for a larger base dataset.
- With ~60M tokens, a 12M param model processes 3-5 epochs in 5 min on A10G, giving clean signal for architecture search.

### 5.3 Protein Specialist's Distributional Comparison

**Key differences between SMILES and protein distributions that affect architecture search:**

1. **Sequence length variance:** Protein sequences have much higher variance (std ~300 vs ~30 for SMILES). This means:
   - Batch packing efficiency differs significantly
   - The model must handle both very short and very long sequences
   - Attention memory cost varies dramatically between batches

2. **Vocabulary structure:** Both are tiny, but amino acids have rich physicochemical properties that could be encoded as auxiliary features. SMILES characters have syntactic roles (atoms vs bonds vs branches) that are more diverse.

3. **Long-range dependencies:** In SMILES, ring closures create a few long-range dependencies per molecule. In proteins, tertiary contacts create many long-range dependencies per sequence (every helix-helix contact, beta-sheet pairing, etc.). This suggests proteins may benefit more from full (non-windowed) attention.

4. **Compositionality:** SMILES has strict grammar (parenthetical nesting, ring closure matching). Proteins have statistical regularities (secondary structure motifs) but no strict grammar. This suggests SMILES modeling may benefit from architecture features that respect its grammar, while protein modeling is more about learning statistical patterns.

### 5.4 Unified Tokenization Recommendation

For all molecular tracks (A and B), use **character-level tokenization**. For the NLP control track (C), use the existing BPE tokenizer.

The `prepare.py` replacement should:
1. Accept a `--track` flag (`smiles`, `protein`, `nlp`)
2. Build appropriate tokenizer for each track
3. Compute `token_bytes.pt` correctly (1 byte per character for molecular tracks)
4. Set `MAX_SEQ_LEN` appropriately per track (512 for SMILES, 512 for proteins, 2048 for NLP)
5. Use the same `evaluate_bpb` formula across all tracks for comparability

---

## 6. Feasibility Assessment and Recommendations

### 6.1 Go/No-Go Assessment

| Criterion | SMILES (Track A) | Protein (Track B) | NLP Control (Track C) | Verdict |
|-----------|-----------------|-------------------|----------------------|---------|
| Data availability | ZINC-250K readily available | UniRef50 available, needs subsetting | FineWeb-Edu (existing) | GO |
| Tokenization | Character-level, well-understood | Character-level, natural | BPE (existing) | GO |
| Dataset size vs model size | Tight but workable with augmentation | Workable with filtering | Already tuned | GO with caveats |
| 5-min training signal | Marginal at default model size; good at 6-12M params | Good with 50K filtered seqs | Already validated | GO with model scaling |
| Hardware compatibility | **FA3 kernel may not work on A10G** | Same issue | Same issue | BLOCKER -- needs fix |
| val_bpb metric | Valid and interpretable | Valid and interpretable | Already validated | GO |
| Architecture search space | Rich (SMILES-specific features possible) | Rich (protein-specific features possible) | Already explored by autoresearch | GO |

### 6.2 Critical Blockers

#### Blocker 1: Flash Attention 3 on A10G (Ampere)

**Severity:** Critical -- training will crash on import
**Fix:** Replace FA3 calls with `torch.nn.functional.scaled_dot_product_attention` (SDPA):
```python
# Replace:
y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
# With:
y = F.scaled_dot_product_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
    is_causal=True
).transpose(1, 2)
```
**Note:** SDPA doesn't natively support sliding window attention. Window attention would need a custom mask or be dropped.

#### Blocker 2: `kernels` Package Dependency

The `kernels` package (`get_kernel()`) is used only for FA3. On A10G, this import path may fail entirely. The `train.py` modification for molecular tracks should remove this dependency.

### 6.3 Recommended Configuration per Track

#### Track A: SMILES (ZINC-250K)

```python
# prepare.py constants
MAX_SEQ_LEN = 256          # SMILES are short
TIME_BUDGET = 300
EVAL_TOKENS = 5 * 131072   # ~655K tokens

# train.py hyperparameters
DEPTH = 6
ASPECT_RATIO = 48          # base_dim=288
HEAD_DIM = 64              # -> dim=320, heads=5
WINDOW_PATTERN = "SL"      # Alternating short/long for SMILES locality
TOTAL_BATCH_SIZE = 65536   # Smaller dataset -> smaller batch
DEVICE_BATCH_SIZE = 256    # Short sequences -> can fit more
```

**Estimated model size:** ~8-10M params
**Estimated throughput on A10G:** ~120K tok/sec at 30% MFU
**Estimated epochs in 5 min:** ~2.9 over 12.5M tokens (or ~0.6 over 60M augmented tokens)

#### Track B: Protein (UniRef50 subset)

```python
# prepare.py constants
MAX_SEQ_LEN = 512          # Covers proteins up to ~500 residues
TIME_BUDGET = 300
EVAL_TOKENS = 5 * 131072

# train.py hyperparameters
DEPTH = 6
ASPECT_RATIO = 48
HEAD_DIM = 64
WINDOW_PATTERN = "SSSL"    # Proteins need more long-range attention
TOTAL_BATCH_SIZE = 65536
DEVICE_BATCH_SIZE = 128    # Longer sequences need smaller batch
```

**Estimated model size:** ~8-10M params
**Estimated throughput on A10G:** ~120K tok/sec
**Estimated epochs in 5 min:** ~2.4 over 15M tokens

#### Track C: NLP Control (FineWeb-Edu)

```python
# Use existing autoresearch prepare.py and train.py
# Only change: replace FA3 with SDPA for A10G compatibility
MAX_SEQ_LEN = 2048
TIME_BUDGET = 300

# Scale down model for A10G
DEPTH = 6
ASPECT_RATIO = 48
```

### 6.4 Implementation Roadmap

| Phase | Task | Effort | Priority |
|-------|------|--------|----------|
| 0 | Fix FA3 -> SDPA for A10G compatibility | 2 hours | P0 (blocker) |
| 1a | Implement SMILES `prepare.py` | 4 hours | P0 |
| 1b | Implement protein `prepare.py` | 4 hours | P0 |
| 1c | Scale down `train.py` defaults for A10G | 1 hour | P0 |
| 2 | Baseline run on all 3 tracks | 15 min each | P0 |
| 3 | Validate architecture search loop (3 iterations) | 45 min | P0 |
| 4 | SMILES enumeration data augmentation | 3 hours | P1 |
| 5 | SMILES-aware tokenizer (multi-char atoms) | 3 hours | P1 |
| 6 | Protein physicochemical embeddings | 2 hours | P2 |
| 7 | Custom attention patterns for molecular data | 4 hours | P2 |

### 6.5 Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| FA3 incompatible with A10G | High | Critical | Replace with SDPA (Phase 0) |
| 12.5M tokens too small for signal | Medium | High | SMILES enumeration augmentation |
| Model overfits ZINC-250K | Medium (at small model sizes with many epochs) | Medium | Early stopping; use augmented data |
| val_bpb too noisy for architecture search | Low-Medium | High | Increase EVAL_TOKENS; run 2x eval |
| Muon optimizer unstable at small scale | Low | Medium | Fall back to pure AdamW |
| `kernels` package breaks on Ampere | High | Critical | Remove dependency entirely |
| Protein sequences too long for 5-min budget | Low (with filtering) | Medium | Filter to <500 residues |

### 6.6 Expected Outcomes

**If successful, the architecture search should discover:**

For SMILES:
- Optimal model depth/width ratio for chemical language
- Whether sliding window attention helps (SMILES locality hypothesis)
- Whether smaller batch sizes with more epochs beat larger batches with fewer epochs
- Optimal learning rates for the tiny-vocab regime

For Proteins:
- Optimal attention pattern for sequence-structure relationships
- Whether deeper models (more layers) learn better long-range contact patterns
- Whether protein-specific positional encodings improve val_bpb

**Comparable metrics across tracks** will allow us to quantify:
- How much "easier" SMILES is to model than natural language (expected: significantly easier due to strict grammar)
- How protein modeling difficulty compares to SMILES and NLP
- Whether molecular-specific architecture features transfer insights across domains

### 6.7 Final Verdict

**The project is feasible with modifications.** The core autoresearch loop (modify -> train 5 min -> eval -> keep/discard) translates well to molecular data. The main challenges are:

1. **Hardware:** FA3 kernel compatibility on A10G is a blocker but straightforward to fix
2. **Scale:** ZINC-250K is small for the default model; solved by reducing model size or augmenting data
3. **Signal quality:** With appropriate model scaling (6-12M params), 5-minute runs produce enough epochs for meaningful val_bpb signal

The three-track design (SMILES, Protein, NLP control) provides excellent cross-domain comparison opportunities. The NLP control track validates that the infrastructure works, while molecular tracks test domain-specific architectural hypotheses.

**Recommended immediate next step:** Fix the FA3/kernels dependency for A10G, implement the SMILES prepare.py, and run a baseline to validate end-to-end feasibility.
