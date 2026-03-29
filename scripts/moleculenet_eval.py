from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import requests
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from _eval_common import (
    FIGURES_DIR,
    PROJECT_ROOT,
    REPLICATE_SEEDS,
    RESULTS_DIR,
    ArchitectureInfo,
    ensure_dir,
    load_torch_payload,
    now_iso,
    read_json,
    run_architecture_subprocess,
    save_run_record,
    top_smiles_architectures,
    write_json,
)


MOLECULENET_DIR = RESULTS_DIR / "moleculenet"
RAW_DIR = MOLECULENET_DIR / "raw"
CHECKPOINT_DIR = MOLECULENET_DIR / "checkpoints"
FEATURE_DIR = MOLECULENET_DIR / "features"
DATASET_DIR = MOLECULENET_DIR / "datasets"
GENERATED_DIR = MOLECULENET_DIR / "generated"
SCORES_PATH = MOLECULENET_DIR / "scores.json"
GENERATION_PATH = MOLECULENET_DIR / "generation_metrics.json"

TASK_URLS = {
    "bbbp": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    "hiv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
    "bace": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
}
TASK_LABEL_KEYS = {
    "bbbp": ("p_np", "P_np", "label", "Label"),
    "hiv": ("HIV_active", "HIV_ACTIVE", "activity", "label", "Label"),
    "bace": ("Class", "class", "label", "Label"),
}
TASK_ORDER = ("bbbp", "hiv", "bace")
TASK_LABELS = {"bbbp": "BBBP", "hiv": "HIV", "bace": "BACE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SC-7 MoleculeNet evaluation")
    parser.add_argument("--smoke", action="store_true", help="Run a minimal smoke test.")
    parser.add_argument("--time-budget", type=int, default=None, help="Override the 300s pretrain budget.")
    parser.add_argument("--num-generated", type=int, default=None, help="Override the 10K-generation default.")
    return parser.parse_args()


def resolve_time_budget(args: argparse.Namespace) -> int:
    if args.time_budget is not None:
        return args.time_budget
    return 20 if args.smoke else 300


def resolve_num_generated(args: argparse.Namespace) -> int:
    if args.num_generated is not None:
        return args.num_generated
    return 128 if args.smoke else 10_000


def resolve_tasks(args: argparse.Namespace) -> tuple[str, ...]:
    return ("bbbp",) if args.smoke else TASK_ORDER


def resolve_seeds(args: argparse.Namespace) -> tuple[int, ...]:
    return (42,) if args.smoke else REPLICATE_SEEDS


def pretrain_raw_path(arch: ArchitectureInfo) -> Path:
    return RAW_DIR / "pretrain" / f"{arch.name}.json"


def pretrain_checkpoint_path(arch: ArchitectureInfo) -> Path:
    return CHECKPOINT_DIR / f"{arch.name}_seed42.pt"


def feature_input_path(task: str, split_name: str) -> Path:
    return FEATURE_DIR / "requests" / f"{task}_{split_name}.json"


def feature_output_path(arch: ArchitectureInfo, task: str, split_name: str) -> Path:
    return FEATURE_DIR / f"{arch.name}_{task}_{split_name}.pt"


def probe_raw_path(arch: ArchitectureInfo, task: str, seed: int) -> Path:
    return RAW_DIR / "probes" / f"{arch.name}_{task}_seed{seed}.json"


def generate_raw_path(arch: ArchitectureInfo) -> Path:
    return GENERATED_DIR / f"{arch.name}.pt"


def load_or_download_dataset(task: str) -> Path:
    ensure_dir(DATASET_DIR)
    path = DATASET_DIR / f"{task}.csv"
    if path.exists():
        return path
    response = requests.get(TASK_URLS[task], timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def normalize_label(raw: str) -> int | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        numeric = float(value)
    except ValueError:
        return None
    return 1 if numeric > 0 else 0


def find_smiles_key(fieldnames: list[str]) -> str:
    lowered = {field.lower(): field for field in fieldnames}
    for candidate in ("smiles", "mol", "molecule", "drug"):
        if candidate in lowered:
            return lowered[candidate]
    return fieldnames[0]


def find_label_key(task: str, fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    for candidate in TASK_LABEL_KEYS[task]:
        if candidate in fieldnames:
            return candidate
    ignore = {"name", "id", "split", "model", "mol_id"}
    smiles_key = find_smiles_key(fieldnames)
    for field in fieldnames:
        if field == smiles_key or field.lower() in ignore:
            continue
        labels = [normalize_label(row.get(field, "")) for row in rows[:50]]
        if any(label is not None for label in labels):
            return field
    raise ValueError(f"Unable to infer label column for task={task}")


def load_task_records(task: str) -> list[dict[str, Any]]:
    csv_path = load_or_download_dataset(task)
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    fieldnames = reader.fieldnames or []
    smiles_key = find_smiles_key(fieldnames)
    label_key = find_label_key(task, fieldnames, rows)
    records = []
    for row in rows:
        smiles = (row.get(smiles_key) or "").strip()
        label = normalize_label(row.get(label_key, ""))
        if not smiles or label is None:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        records.append({"smiles": smiles, "canonical_smiles": canonical, "label": label})
    if not records:
        raise ValueError(f"No valid records loaded for task={task}")
    return records


def scaffold_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    scaffold_groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        scaffold_groups[scaffold_key(record["smiles"])].append(index)
    groups = sorted(scaffold_groups.values(), key=lambda item: (len(item), item[0]), reverse=True)
    n_total = len(records)
    train_cutoff = int(0.8 * n_total)
    valid_cutoff = int(0.9 * n_total)
    train_indices: list[int] = []
    valid_indices: list[int] = []
    test_indices: list[int] = []
    for group in groups:
        if len(train_indices) + len(group) <= train_cutoff:
            train_indices.extend(group)
        elif len(train_indices) + len(valid_indices) + len(group) <= valid_cutoff:
            valid_indices.extend(group)
        else:
            test_indices.extend(group)
    return {
        "train": [records[index] for index in train_indices],
        "val": [records[index] for index in valid_indices],
        "test": [records[index] for index in test_indices],
    }


def save_feature_request(path: Path, smiles_list: list[str], *, batch_size: int) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps({"texts": smiles_list, "batch_size": batch_size}) + "\n")


def ensure_pretrain_checkpoint(arch: ArchitectureInfo, *, time_budget: int) -> Path:
    ensure_dir(CHECKPOINT_DIR)
    checkpoint = pretrain_checkpoint_path(arch)
    if checkpoint.exists():
        print(f"[pretrain] cached: {arch.name}")
        return checkpoint
    print(f"[pretrain] running: {arch.name}")
    raw_path = pretrain_raw_path(arch)
    result = run_architecture_subprocess(
        arch.source_path,
        track="smiles",
        seed=42,
        time_budget=time_budget,
        checkpoint_save=checkpoint,
    )
    result.update({"arch_name": arch.name, "arch_source_path": str(arch.source_path)})
    save_run_record(raw_path, result)
    return checkpoint


def ensure_features(
    arch: ArchitectureInfo,
    task: str,
    split_name: str,
    smiles_list: list[str],
    checkpoint: Path,
    *,
    smoke: bool,
) -> torch.Tensor:
    output_path = feature_output_path(arch, task, split_name)
    if output_path.exists():
        print(f"[features] cached: {arch.name} task={task} split={split_name}")
        payload = load_torch_payload(output_path)
        return payload["features"].float()
    print(f"[features] running: {arch.name} task={task} split={split_name}")
    batch_size = 64 if smoke else 256
    request_path = feature_input_path(task, split_name)
    save_feature_request(request_path, smiles_list, batch_size=batch_size)
    ensure_dir(output_path.parent)
    run_architecture_subprocess(
        arch.source_path,
        track="smiles",
        seed=42,
        mode="features",
        checkpoint_load=checkpoint,
        feature_input=request_path,
        feature_output=output_path,
        device_batch_size=128,
        time_budget=1,
        timeout_seconds=1800,
    )
    payload = load_torch_payload(output_path)
    return payload["features"].float()


def standardize_features(
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    test_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std


def safe_roc_auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    y_true = labels.detach().cpu().numpy()
    y_score = scores.detach().cpu().numpy()
    if len(set(y_true.tolist())) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def train_linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(train_x.size(1), 1).to(device)
    train_x = train_x.to(torch.float32)
    val_x = val_x.to(torch.float32)
    test_x = test_x.to(torch.float32)
    train_y = train_y.to(torch.float32)
    val_y = val_y.to(torch.float32)
    test_y = test_y.to(torch.float32)

    pos = float(train_y.sum().item())
    neg = float(train_y.numel() - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, generator=generator)

    best_state: dict[str, Any] | None = None
    best_auc = -math.inf
    best_epoch = 0
    patience = 10
    no_improve = 0
    max_epochs = 100 if train_x.size(0) < 10_000 else 60

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(val_x.to(device))).squeeze(1).cpu()
        val_auc = safe_roc_auc(val_y, val_scores)
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is None:
        raise RuntimeError("Linear probe failed to produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_scores = torch.sigmoid(model(test_x.to(device))).squeeze(1).cpu()
    return {
        "test_roc_auc": round(safe_roc_auc(test_y, test_scores), 6),
        "val_roc_auc": round(best_auc, 6),
        "best_epoch": best_epoch,
        "seed": seed,
    }


def build_scores_payload(
    architectures: list[ArchitectureInfo],
    tasks: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    items = []
    for arch in architectures:
        arch_payload = arch.to_json()
        task_payload: dict[str, Any] = {}
        task_means: list[float] = []
        for task in tasks:
            runs = []
            for seed in seeds:
                raw = read_json(probe_raw_path(arch, task, seed), {})
                if raw and raw.get("test_roc_auc") is not None:
                    runs.append(float(raw["test_roc_auc"]))
            if runs:
                mean = sum(runs) / len(runs)
                std = (sum((value - mean) ** 2 for value in runs) / len(runs)) ** 0.5
                task_payload[task] = {
                    "roc_auc_mean": round(mean, 6),
                    "roc_auc_std": round(std, 6),
                    "runs": [round(value, 6) for value in runs],
                }
                task_means.append(mean)
            else:
                task_payload[task] = {"roc_auc_mean": None, "roc_auc_std": None, "runs": []}
        arch_payload["tasks"] = task_payload
        arch_payload["mean_roc_auc"] = round(sum(task_means) / len(task_means), 6) if task_means else None
        items.append(arch_payload)

    usable = [item for item in items if item["mean_roc_auc"] is not None]
    correlation = {
        "val_bpb_ranking": [],
        "roc_auc_ranking": [],
        "spearman_rho": None,
        "spearman_p": None,
        "note": "With n=3, rank correlation has very low power. Interpret qualitatively.",
    }
    if len(usable) >= 2:
        bpb_sorted = sorted(usable, key=lambda item: item["native_bpb"])
        roc_sorted = sorted(usable, key=lambda item: item["mean_roc_auc"], reverse=True)
        bpb_rank = {item["name"]: index + 1 for index, item in enumerate(bpb_sorted)}
        roc_rank = {item["name"]: index + 1 for index, item in enumerate(roc_sorted)}
        order = [item["name"] for item in usable]
        bpb_ranking = [bpb_rank[name] for name in order]
        roc_ranking = [roc_rank[name] for name in order]
        rho, p_value = spearmanr(bpb_ranking, roc_ranking)
        correlation = {
            "val_bpb_ranking": bpb_ranking,
            "roc_auc_ranking": roc_ranking,
            "spearman_rho": round(float(rho), 6),
            "spearman_p": round(float(p_value), 6),
            "note": "With n=3, rank correlation has very low power. Interpret qualitatively.",
        }
    return {"generated_at": now_iso(), "architectures": items, "correlation": correlation}


def plot_scores(scores_payload: dict[str, Any]) -> None:
    ensure_dir(FIGURES_DIR)
    architectures = scores_payload["architectures"]
    x = torch.arange(len(TASK_ORDER))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, arch in enumerate(architectures):
        means = [arch["tasks"][task]["roc_auc_mean"] or 0.0 for task in TASK_ORDER]
        positions = [value + (idx - 1) * width for value in x.tolist()]
        ax.bar(positions, means, width=width, label=arch["name"])
    ax.set_xticks(x.tolist(), [TASK_LABELS[task] for task in TASK_ORDER])
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("SC-7 MoleculeNet ROC-AUC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "moleculenet_bar.png", dpi=200)
    plt.close(fig)

    usable = [arch for arch in architectures if arch["mean_roc_auc"] is not None]
    if usable:
        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        xs = [arch["native_bpb"] for arch in usable]
        ys = [arch["mean_roc_auc"] for arch in usable]
        ax.scatter(xs, ys, color="#0b6e4f", s=70)
        for arch in usable:
            ax.annotate(arch["name"], (arch["native_bpb"], arch["mean_roc_auc"]), xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("val_bpb (pretraining)")
        ax.set_ylabel("Mean ROC-AUC")
        ax.set_title("val_bpb vs MoleculeNet ROC-AUC")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "bpb_vs_rocauc_scatter.png", dpi=200)
        plt.close(fig)


def load_training_smiles_set() -> set[str]:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import prepare_smiles  # type: ignore

    canonical_to_mol = prepare_smiles.load_unique_molecules(prepare_smiles.RAW_PATH)
    train_ids, _ = prepare_smiles.split_molecules(canonical_to_mol)
    return set(train_ids)


def compute_generation_metrics(generated_path: Path, *, num_generated: int) -> dict[str, Any]:
    payload = load_torch_payload(generated_path)
    generated = payload["generated"]
    valid_smiles = []
    for smiles in generated:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        valid_smiles.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    unique_valid = sorted(set(valid_smiles))
    train_smiles = load_training_smiles_set()
    novelty = 0.0
    if unique_valid:
        novelty = sum(item not in train_smiles for item in unique_valid) / len(unique_valid)
    metrics = {
        "num_generated": num_generated,
        "temperature": float(payload["temperature"]),
        "sampling": f"top_k_{int(payload['top_k'])}",
        "validity": round(len(valid_smiles) / max(len(generated), 1), 6),
        "uniqueness": round(len(unique_valid) / max(len(valid_smiles), 1), 6),
        "novelty": round(novelty, 6),
        "fcd": None,
        "note": "FCD skipped because the optional fcd package is not installed.",
    }
    return metrics


def verify_sc7(scores_payload: dict[str, Any], generation_payload: dict[str, Any], *, tasks: tuple[str, ...], seeds: tuple[int, ...]) -> dict[str, bool]:
    architectures = scores_payload["architectures"]
    all_tasks_present = all(
        arch["tasks"][task]["roc_auc_mean"] is not None and len(arch["tasks"][task]["runs"]) == len(seeds)
        for arch in architectures
        for task in tasks
    )
    roc_values = [
        arch["tasks"][task]["roc_auc_mean"]
        for arch in architectures
        for task in tasks
        if arch["tasks"][task]["roc_auc_mean"] is not None
    ]
    figure_paths = [
        FIGURES_DIR / "moleculenet_bar.png",
        FIGURES_DIR / "bpb_vs_rocauc_scatter.png",
    ]
    return {
        "scores_complete": all_tasks_present,
        "roc_auc_above_random": all(0.5 <= value <= 1.0 for value in roc_values),
        "spearman_present": scores_payload["correlation"]["spearman_rho"] is not None,
        "generation_validity_gt_50pct": generation_payload["validity"] > 0.5,
        "uniqueness_and_novelty_on_valid_only": generation_payload["uniqueness"] <= 1.0 and generation_payload["novelty"] <= 1.0,
        "figures_exist": all(path.exists() for path in figure_paths),
    }


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    ensure_dir(MOLECULENET_DIR)
    ensure_dir(RAW_DIR)
    ensure_dir(FEATURE_DIR)
    ensure_dir(GENERATED_DIR)

    architectures = top_smiles_architectures(1 if args.smoke else 3)
    tasks = resolve_tasks(args)
    seeds = resolve_seeds(args)
    time_budget = resolve_time_budget(args)
    num_generated = resolve_num_generated(args)
    print("SC-7 architectures:")
    for arch in architectures:
        print(f"  {arch.name}: {arch.source_path} (native_bpb={arch.native_bpb:.6f})")
    print(f"SC-7 tasks={list(tasks)} time_budget={time_budget}s seeds={list(seeds)} num_generated={num_generated}")

    checkpoints = {arch.name: ensure_pretrain_checkpoint(arch, time_budget=time_budget) for arch in architectures}
    task_splits = {task: scaffold_split(load_task_records(task)) for task in tasks}
    for task, split in task_splits.items():
        print(
            f"[dataset] {task}: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}"
        )

    for arch in architectures:
        checkpoint = checkpoints[arch.name]
        for task in tasks:
            split = task_splits[task]
            train_x = ensure_features(arch, task, "train", [item["smiles"] for item in split["train"]], checkpoint, smoke=args.smoke)
            val_x = ensure_features(arch, task, "val", [item["smiles"] for item in split["val"]], checkpoint, smoke=args.smoke)
            test_x = ensure_features(arch, task, "test", [item["smiles"] for item in split["test"]], checkpoint, smoke=args.smoke)

            train_y = torch.tensor([item["label"] for item in split["train"]], dtype=torch.float32)
            val_y = torch.tensor([item["label"] for item in split["val"]], dtype=torch.float32)
            test_y = torch.tensor([item["label"] for item in split["test"]], dtype=torch.float32)
            train_x, val_x, test_x = standardize_features(train_x, val_x, test_x)

            for seed in seeds:
                raw_path = probe_raw_path(arch, task, seed)
                cached = read_json(raw_path)
                if cached and cached.get("test_roc_auc") is not None:
                    print(f"[probe] cached: {arch.name} task={task} seed={seed}")
                    continue
                print(f"[probe] running: {arch.name} task={task} seed={seed}")
                result = train_linear_probe(train_x, train_y, val_x, val_y, test_x, test_y, seed=seed)
                result.update(
                    {
                        "arch_name": arch.name,
                        "task": task,
                        "seed": seed,
                        "num_train": len(split["train"]),
                        "num_val": len(split["val"]),
                        "num_test": len(split["test"]),
                        "timestamp": now_iso(),
                    }
                )
                save_run_record(raw_path, result)
                scores_payload = build_scores_payload(architectures, tasks, seeds)
                write_json(SCORES_PATH, scores_payload)

    scores_payload = build_scores_payload(architectures, tasks, seeds)
    write_json(SCORES_PATH, scores_payload)
    plot_scores(scores_payload)

    best_arch = architectures[0]
    generated_path = generate_raw_path(best_arch)
    if not generated_path.exists():
        print(f"[generate] running: {best_arch.name}")
        run_architecture_subprocess(
            best_arch.source_path,
            track="smiles",
            seed=42,
            mode="generate",
            checkpoint_load=checkpoints[best_arch.name],
            generate_output=generated_path,
            generate_batch_size=128 if args.smoke else 256,
            num_samples=num_generated,
            top_k=50,
            temperature=1.0,
            max_new_tokens=255,
            time_budget=1,
            timeout_seconds=1800,
        )
    else:
        print(f"[generate] cached: {best_arch.name}")
    generation_payload = compute_generation_metrics(generated_path, num_generated=num_generated)
    generation_payload["generated_at"] = now_iso()
    generation_payload["architecture"] = best_arch.name
    write_json(GENERATION_PATH, generation_payload)

    verification = verify_sc7(scores_payload, generation_payload, tasks=tasks, seeds=seeds)
    print("SC-7 verification:")
    for key, value in verification.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
