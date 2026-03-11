"""
Prepare the SMILES track based on ZINC-250K with random enumeration.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import requests
from rdkit import Chem, RDLogger

from prepare_char import (
    TIME_BUDGET,
    CharTokenizer,
    build_tokenizer_from_texts,
    evaluate_bpb as evaluate_bpb_shared,
    make_stream_dataloader,
    save_stream,
)

MAX_SEQ_LEN = 256
NUM_ENUMERATIONS = 20
VAL_ENUMERATIONS = 5
SPLIT_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "smiles"
RAW_DIR = DATA_DIR / "raw"
RAW_FILENAME = "250k_rndm_zinc_drugs_clean_3.csv"
RAW_PATH = RAW_DIR / RAW_FILENAME
TOKENIZER_PATH = DATA_DIR / "tokenizer.pkl"
TRAIN_PATH = DATA_DIR / "train.pkl"
VAL_PATH = DATA_DIR / "val.pkl"
STATS_PATH = DATA_DIR / "stats.json"

SOURCE_URLS = [
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
    "https://raw.githubusercontent.com/molecularsets/moses/master/data/250k_rndm_zinc_drugs_clean_3.csv",
]


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


def download_dataset() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists():
        return RAW_PATH

    last_error = None
    for url in SOURCE_URLS:
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            RAW_PATH.write_bytes(response.content)
            return RAW_PATH
        except requests.RequestException as exc:
            last_error = exc
    raise RuntimeError(f"Unable to download {RAW_FILENAME}: {last_error}")


def load_unique_molecules(csv_path: Path) -> dict[str, Chem.Mol]:
    unique = {}
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        smiles_key = "smiles" if "smiles" in fieldnames else fieldnames[0]
        for row in reader:
            raw_smiles = row[smiles_key].strip()
            if not raw_smiles:
                continue
            mol = Chem.MolFromSmiles(raw_smiles)
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            unique.setdefault(canonical, mol)
    return unique


def enumerate_smiles(mol: Chem.Mol, count: int) -> list[str]:
    smiles = []
    for _ in range(count):
        smiles.append(Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True))
    return smiles


def split_molecules(canonical_to_mol: dict[str, Chem.Mol]) -> tuple[list[str], list[str]]:
    molecule_ids = list(canonical_to_mol.keys())
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(molecule_ids)
    cutoff = int(0.9 * len(molecule_ids))
    return molecule_ids[:cutoff], molecule_ids[cutoff:]


def build_sequences(
    molecule_ids: list[str],
    canonical_to_mol: dict[str, Chem.Mol],
    enumeration_count: int,
) -> tuple[list[str], list[str]]:
    enumerated = []
    canonical_ids = []
    for canonical in molecule_ids:
        variants = enumerate_smiles(canonical_to_mol[canonical], enumeration_count)
        enumerated.extend(variants)
        canonical_ids.extend([canonical] * len(variants))
    return enumerated, canonical_ids


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = download_dataset()
    canonical_to_mol = load_unique_molecules(csv_path)
    train_ids, val_ids = split_molecules(canonical_to_mol)

    train_sequences, train_canonical_ids = build_sequences(
        train_ids,
        canonical_to_mol,
        NUM_ENUMERATIONS,
    )
    val_sequences, val_canonical_ids = build_sequences(
        val_ids,
        canonical_to_mol,
        VAL_ENUMERATIONS,
    )

    tokenizer = build_tokenizer_from_texts(train_sequences + val_sequences)
    tokenizer.save(TOKENIZER_PATH)

    train_stats = save_stream(TRAIN_PATH, train_sequences, tokenizer)
    val_stats = save_stream(VAL_PATH, val_sequences, tokenizer)

    train_set = set(train_ids)
    val_set = set(val_ids)
    overlap = train_set & val_set
    stats = {
        "source_csv": str(csv_path),
        "num_unique_molecules": len(canonical_to_mol),
        "num_train_molecules": len(train_ids),
        "num_val_molecules": len(val_ids),
        "num_enumerations": NUM_ENUMERATIONS,
        "val_enumerations": VAL_ENUMERATIONS,
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "train_sequences": train_stats["num_sequences"],
        "val_sequences": val_stats["num_sequences"],
        "train_raw_tokens": train_stats["raw_token_count"],
        "val_raw_tokens": val_stats["raw_token_count"],
        "total_raw_tokens": train_stats["raw_token_count"] + val_stats["raw_token_count"],
        "train_stream_tokens": train_stats["stream_token_count"],
        "val_stream_tokens": val_stats["stream_token_count"],
        "canonical_overlap_count": len(overlap),
        "train_canonical_examples": train_canonical_ids[:5],
        "val_canonical_examples": val_canonical_ids[:5],
    }
    with open(STATS_PATH, "w") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)

    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
