"""
Prepare a 200K protein-sequence subset from UniRef50.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import requests

from prepare_char import (
    TIME_BUDGET,
    CharTokenizer,
    EOS_TOKEN,
    PAD_TOKEN,
    BOS_TOKEN,
    UNK_TOKEN,
    evaluate_bpb as evaluate_bpb_shared,
    make_stream_dataloader,
    save_stream,
)

MAX_SEQ_LEN = 512
MIN_LENGTH = 50
MAX_LENGTH = 500
TARGET_SEQUENCES = 200_000
SPLIT_SEED = 42
VALID_AA = "ACDEFGHIKLMNPQRSTVWY"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "protein"
RAW_DIR = DATA_DIR / "raw"
TRAIN_PATH = DATA_DIR / "train.pkl"
VAL_PATH = DATA_DIR / "val.pkl"
TOKENIZER_PATH = DATA_DIR / "tokenizer.pkl"
STATS_PATH = DATA_DIR / "stats.json"

UNIPROT_URL = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref50/uniref50.fasta.gz"


class Tokenizer(CharTokenizer):
    @classmethod
    def from_directory(cls, tokenizer_dir: str | Path = DATA_DIR) -> "Tokenizer":
        tokenizer = CharTokenizer.from_file(Path(tokenizer_dir) / "tokenizer.pkl")
        return cls(stoi=tokenizer.stoi, itos=tokenizer.itos)


def make_dataloader(tokenizer, batch_size, seq_len, split, device=None):
    split_path = TRAIN_PATH if split == "train" else VAL_PATH
    return make_stream_dataloader(split_path, batch_size, seq_len, device=device)


def evaluate_bpb(model, tokenizer, batch_size, device=None):
    return evaluate_bpb_shared(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_len=MAX_SEQ_LEN,
        split_path=VAL_PATH,
        device=device,
    )


def build_tokenizer() -> Tokenizer:
    itos = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + list(VALID_AA)
    stoi = {token: idx for idx, token in enumerate(itos)}
    return Tokenizer(stoi=stoi, itos=itos)


def iter_fasta_sequences():
    with requests.get(UNIPROT_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        with gzip.GzipFile(fileobj=response.raw) as handle:
            header = None
            seq_parts = []
            for raw_line in handle:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None:
                        yield header, "".join(seq_parts)
                    header = line[1:]
                    seq_parts = []
                else:
                    seq_parts.append(line)
            if header is not None:
                yield header, "".join(seq_parts)


def reservoir_sample_sequences() -> tuple[list[tuple[str, str]], int]:
    rng = random.Random(SPLIT_SEED)
    reservoir: list[tuple[str, str]] = []
    seen = 0
    valid_chars = set(VALID_AA)
    for header, sequence in iter_fasta_sequences():
        if not (MIN_LENGTH <= len(sequence) <= MAX_LENGTH):
            continue
        if any(residue not in valid_chars for residue in sequence):
            continue
        seen += 1
        item = (header, sequence)
        if len(reservoir) < TARGET_SEQUENCES:
            reservoir.append(item)
            continue
        replace_idx = rng.randint(0, seen - 1)
        if replace_idx < TARGET_SEQUENCES:
            reservoir[replace_idx] = item
    if len(reservoir) < TARGET_SEQUENCES:
        raise RuntimeError(
            f"Expected {TARGET_SEQUENCES} valid sequences after filtering, only found {len(reservoir)}"
        )
    return reservoir, seen


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    sampled, total_filtered = reservoir_sample_sequences()
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(sampled)
    cutoff = int(0.9 * len(sampled))
    train_samples = sampled[:cutoff]
    val_samples = sampled[cutoff:]

    tokenizer = build_tokenizer()
    tokenizer.save(TOKENIZER_PATH)

    train_sequences = [seq for _, seq in train_samples]
    val_sequences = [seq for _, seq in val_samples]
    train_stats = save_stream(TRAIN_PATH, train_sequences, tokenizer)
    val_stats = save_stream(VAL_PATH, val_sequences, tokenizer)

    stats = {
        "source_url": UNIPROT_URL,
        "filtered_candidates_seen": total_filtered,
        "sampled_sequences": len(sampled),
        "train_sequences": len(train_sequences),
        "val_sequences": len(val_sequences),
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "train_raw_tokens": train_stats["raw_token_count"],
        "val_raw_tokens": val_stats["raw_token_count"],
        "total_raw_tokens": train_stats["raw_token_count"] + val_stats["raw_token_count"],
        "train_stream_tokens": train_stats["stream_token_count"],
        "val_stream_tokens": val_stats["stream_token_count"],
        "length_filter": [MIN_LENGTH, MAX_LENGTH],
    }
    with open(STATS_PATH, "w") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)

    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
