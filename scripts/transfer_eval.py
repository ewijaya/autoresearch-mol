from __future__ import annotations

import argparse
import difflib
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from _eval_common import (
    FIGURES_DIR,
    PROJECT_ROOT,
    REPLICATE_SEEDS,
    RESULTS_DIR,
    ArchitectureInfo,
    baseline_bpbs,
    best_architectures_by_track,
    compute_pct_degradation,
    ensure_dir,
    now_iso,
    read_json,
    run_architecture_subprocess,
    save_run_record,
    summarize_runs,
    write_json,
)


TRANSFER_DIR = RESULTS_DIR / "transfer"
RAW_DIR = TRANSFER_DIR / "raw"
CHECKPOINT_DIR = TRANSFER_DIR / "checkpoints"
MATRIX_PATH = TRANSFER_DIR / "matrix.json"
FREEZING_PATH = TRANSFER_DIR / "layer_freezing.json"
LENGTH_PATH = TRANSFER_DIR / "length_controlled.json"
INNOVATION_PATH = TRANSFER_DIR / "innovation_classification.json"

TRACK_ORDER = ("smiles", "protein", "nlp")
FREEZE_LEVELS = (1, 3, 5)
LENGTH_CONTROL_SEED = 42
ARCH_LABELS = {"smiles": "SMILES", "protein": "Protein", "nlp": "NLP"}
ARCH_KEYWORDS = (
    "n_kv_head",
    "n_v_head",
    "window_pattern",
    "has_ve",
    "ve_gate",
    "attn_lambda",
    "mlp_lambda",
    "attn_gate_logits",
    "mlp_gate_logits",
    "attn_temp_lambdas",
    "ve_lambdas",
    "head_dim",
    "activation",
    "ffn_mult",
    "tie_embed_weights",
    "value_head_count",
    "key_head_count",
    "norm(x)",
    "attn_out",
    "mlp_out",
)
NON_ARCH_KEYWORDS = (
    "EMBEDDING_LR",
    "UNEMBEDDING_LR",
    "MATRIX_LR",
    "SCALAR_LR",
    "TOTAL_BATCH_SIZE",
    "DEVICE_BATCH_SIZE",
    "TIME_BUDGET",
    "MAX_EPOCHS",
    "WANDB",
    "tok/sec",
    "remaining:",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SC-6 transfer evaluation")
    parser.add_argument("--smoke", action="store_true", help="Run a minimal smoke test instead of the full suite.")
    parser.add_argument(
        "--time-budget",
        type=int,
        default=None,
        help="Override the 300s training budget for all train-mode runs.",
    )
    return parser.parse_args()


def transfer_raw_path(kind: str, stem: str) -> Path:
    return RAW_DIR / kind / f"{stem}.json"


def checkpoint_path(arch: ArchitectureInfo) -> Path:
    return CHECKPOINT_DIR / f"{arch.name}_seed42.pt"


def resolve_time_budget(args: argparse.Namespace) -> int:
    if args.time_budget is not None:
        return args.time_budget
    return 20 if args.smoke else 300


def resolve_seeds(args: argparse.Namespace) -> tuple[int, ...]:
    return (42,) if args.smoke else REPLICATE_SEEDS


def discover_architectures() -> dict[str, ArchitectureInfo]:
    return best_architectures_by_track()


def maybe_run_train(
    raw_path: Path,
    *,
    arch: ArchitectureInfo,
    target_track: str,
    seed: int,
    time_budget: int,
    checkpoint_save: Path | None = None,
    checkpoint_load: Path | None = None,
    freeze_layers: int = 0,
    seq_len_override: int | None = None,
) -> dict[str, Any]:
    cached = read_json(raw_path)
    if cached and cached.get("returncode") == 0 and cached.get("val_bpb") is not None:
        return cached
    result = run_architecture_subprocess(
        arch.source_path,
        track=target_track,
        seed=seed,
        time_budget=time_budget,
        checkpoint_save=checkpoint_save,
        checkpoint_load=checkpoint_load,
        freeze_layers=freeze_layers,
        seq_len_override=seq_len_override,
    )
    result.update(
        {
            "arch_source_track": arch.track,
            "arch_name": arch.name,
            "arch_source_path": str(arch.source_path),
        }
    )
    save_run_record(raw_path, result)
    return result


def is_completed_run(raw_path: Path) -> bool:
    cached = read_json(raw_path)
    return bool(cached and cached.get("returncode") == 0 and cached.get("val_bpb") is not None)


def build_matrix_payload(
    architectures: dict[str, ArchitectureInfo],
    baselines: dict[str, float],
    *,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    degradation: dict[str, Any] = {}
    for arch_track, arch in architectures.items():
        arch_key = f"{arch_track}_arch"
        matrix[arch_key] = {}
        degradation[arch_key] = {}
        for target_track in TRACK_ORDER:
            runs = []
            for seed in seeds:
                raw = read_json(transfer_raw_path("matrix", f"{arch_track}_to_{target_track}_seed{seed}"), {})
                if raw and raw.get("val_bpb") is not None:
                    runs.append(float(raw["val_bpb"]))
            summary = summarize_runs(runs)
            matrix[arch_key][f"{target_track}_data"] = summary
            if summary["mean"] is None:
                degradation[arch_key][f"{target_track}_data"] = {
                    "pct_degradation": None,
                    "reference_bpb": None,
                }
                continue
            if arch_track == target_track:
                degradation[arch_key][f"{target_track}_data"] = {
                    "pct_degradation": 0.0,
                    "reference_bpb": arch.native_bpb,
                    "note": "identity",
                }
            else:
                baseline = baselines[target_track]
                degradation[arch_key][f"{target_track}_data"] = {
                    "pct_degradation": compute_pct_degradation(summary["mean"], baseline),
                    "reference_bpb": baseline,
                }

    payload = {
        "generated_at": now_iso(),
        "architectures": {track: arch.to_json() for track, arch in architectures.items()},
        "baseline_bpbs": baselines,
        "replicate_seeds": list(seeds),
        "degradation_reference": "identity cells are pinned to 0; off-diagonal cells are relative to fixed_default target-track baselines",
        "matrix": matrix,
        "degradation_matrix": degradation,
    }
    return payload


def plot_transfer_heatmap(matrix_payload: dict[str, Any]) -> None:
    ensure_dir(FIGURES_DIR)
    values = np.zeros((3, 3), dtype=float)
    for row_idx, arch_track in enumerate(TRACK_ORDER):
        for col_idx, data_track in enumerate(TRACK_ORDER):
            cell = matrix_payload["matrix"][f"{arch_track}_arch"][f"{data_track}_data"]
            values[row_idx, col_idx] = float(cell["mean"])
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    image = ax.imshow(values, cmap="viridis")
    ax.set_xticks(range(3), [ARCH_LABELS[track] for track in TRACK_ORDER])
    ax.set_yticks(range(3), [ARCH_LABELS[track] for track in TRACK_ORDER])
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Architecture Source")
    ax.set_title("SC-6 Transfer Matrix (val_bpb)")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            ax.text(col_idx, row_idx, f"{values[row_idx, col_idx]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, shrink=0.85, label="val_bpb")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "transfer_heatmap.png", dpi=200)
    plt.close(fig)


def ensure_checkpoints(architectures: dict[str, ArchitectureInfo], *, time_budget: int) -> dict[str, Path]:
    ensure_dir(CHECKPOINT_DIR)
    paths: dict[str, Path] = {}
    for track, arch in architectures.items():
        path = checkpoint_path(arch)
        paths[track] = path
        if path.exists():
            print(f"[checkpoint] cached: {track} -> {path}")
            continue
        print(f"[checkpoint] running: {track} -> {path}")
        raw_path = transfer_raw_path("checkpoints", f"{track}_seed42")
        maybe_run_train(
            raw_path,
            arch=arch,
            target_track=track,
            seed=42,
            time_budget=time_budget,
            checkpoint_save=path,
        )
    return paths


def build_freezing_payload(
    architectures: dict[str, ArchitectureInfo],
    baselines: dict[str, float],
    checkpoints: dict[str, Path],
    matrix_payload: dict[str, Any],
) -> dict[str, Any]:
    pairs = []
    arch_tracks = [track for track in TRACK_ORDER if track in architectures]
    for arch_track in arch_tracks:
        arch = architectures[arch_track]
        for target_track in TRACK_ORDER:
            if arch_track == target_track:
                continue
            if matrix_payload["matrix"][f"{arch_track}_arch"][f"{target_track}_data"]["mean"] is None:
                continue
            freeze_levels = []
            for freeze_n in FREEZE_LEVELS:
                raw = read_json(
                    transfer_raw_path(
                        "layer_freezing",
                        f"{arch_track}_to_{target_track}_freeze{freeze_n}",
                    ),
                    {},
                )
                freeze_levels.append(
                    {
                        "frozen_layers": freeze_n,
                        "val_bpb": raw.get("val_bpb"),
                        "raw_path": str(
                            transfer_raw_path(
                                "layer_freezing",
                                f"{arch_track}_to_{target_track}_freeze{freeze_n}",
                            )
                        ),
                    }
                )
            no_freeze = matrix_payload["matrix"][f"{arch_track}_arch"][f"{target_track}_data"]["mean"]
            pairs.append(
                {
                    "arch_source": arch_track,
                    "data_target": target_track,
                    "checkpoint": str(checkpoints[arch_track]),
                    "freeze_levels": freeze_levels,
                    "no_freeze_baseline": no_freeze,
                    "native_baseline": baselines[target_track],
                }
            )
    return {"generated_at": now_iso(), "pairs": pairs}


def plot_layer_freezing(freezing_payload: dict[str, Any]) -> None:
    ensure_dir(FIGURES_DIR)
    fig, ax = plt.subplots(figsize=(9, 6))
    for pair in freezing_payload["pairs"]:
        xs = [0]
        ys = [pair["no_freeze_baseline"]]
        for item in pair["freeze_levels"]:
            if item["val_bpb"] is None:
                continue
            xs.append(item["frozen_layers"])
            ys.append(float(item["val_bpb"]))
        if len(xs) <= 1:
            continue
        label = f"{pair['arch_source']}→{pair['data_target']}"
        ax.plot(xs, ys, marker="o", linewidth=1.6, label=label)
    ax.set_xlabel("Frozen Transformer Layers")
    ax.set_ylabel("val_bpb")
    ax.set_title("SC-6 Layer Freezing")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "layer_freezing_curves.png", dpi=200)
    plt.close(fig)


def worst_transfer_pairs(matrix_payload: dict[str, Any], top_k: int = 3) -> list[tuple[str, str, float]]:
    scored: list[tuple[str, str, float]] = []
    for arch_track in TRACK_ORDER:
        for target_track in TRACK_ORDER:
            if arch_track == target_track:
                continue
            deg = matrix_payload["degradation_matrix"][f"{arch_track}_arch"][f"{target_track}_data"]["pct_degradation"]
            if deg is None:
                continue
            scored.append((arch_track, target_track, float(deg)))
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored[:top_k]


def build_length_payload(
    architectures: dict[str, ArchitectureInfo],
    baselines: dict[str, float],
    matrix_payload: dict[str, Any],
) -> dict[str, Any]:
    pairs_payload = []
    for arch_track, target_track, unmatched_deg in worst_transfer_pairs(matrix_payload, top_k=3):
        arch = architectures[arch_track]
        matched_seq_len = min(arch.native_seq_len, matrix_payload["architectures"][target_track]["native_seq_len"])
        raw_path = transfer_raw_path("length_controlled", f"{arch_track}_to_{target_track}_seq{matched_seq_len}")
        raw = read_json(raw_path, {})
        matched_bpb = raw.get("val_bpb")
        unmatched_bpb = matrix_payload["matrix"][f"{arch_track}_arch"][f"{target_track}_data"]["mean"]
        matched_deg = None
        reduction = None
        if matched_bpb is not None:
            matched_deg = compute_pct_degradation(float(matched_bpb), baselines[target_track])
            if unmatched_deg > 0:
                reduction = round(100.0 * (unmatched_deg - matched_deg) / unmatched_deg, 6)
        pairs_payload.append(
            {
                "arch_source": arch_track,
                "data_target": target_track,
                "matched_seq_len": matched_seq_len,
                "seed": LENGTH_CONTROL_SEED,
                "val_bpb_unmatched": unmatched_bpb,
                "val_bpb_matched": matched_bpb,
                "pct_degradation_unmatched": unmatched_deg,
                "pct_degradation_matched": matched_deg,
                "degradation_reduction_pct": reduction,
                "h3c_criterion_met": matched_deg is not None and matched_deg < unmatched_deg,
                "raw_path": str(raw_path),
            }
        )
    return {"generated_at": now_iso(), "pairs": pairs_payload}


def _is_architectural_chunk(lines: list[str]) -> bool:
    if not lines:
        return False
    if any(marker in line for line in lines for marker in NON_ARCH_KEYWORDS):
        return False
    return any(marker in line for line in lines for marker in ARCH_KEYWORDS)


def extract_innovations(base_source: str, arch_source: str) -> list[str]:
    base_lines = [line.rstrip() for line in base_source.splitlines()]
    arch_lines = [line.rstrip() for line in arch_source.splitlines()]
    matcher = difflib.SequenceMatcher(None, base_lines, arch_lines)
    descriptions: list[str] = []
    seen: set[str] = set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        before = [line.strip() for line in base_lines[i1:i2] if line.strip()]
        after = [line.strip() for line in arch_lines[j1:j2] if line.strip()]
        lines = before + after
        if not _is_architectural_chunk(lines):
            continue
        chosen = after[:2] if after else before[:2]
        prefix = "added" if tag in {"insert", "replace"} else "removed"
        description = f"{prefix}: {' | '.join(chosen)}"
        if description not in seen:
            descriptions.append(description)
            seen.add(description)
    return descriptions


def build_innovation_payload(architectures: dict[str, ArchitectureInfo], matrix_payload: dict[str, Any]) -> dict[str, Any]:
    base_source = (PROJECT_ROOT / "src" / "train.py").read_text()
    innovations = []
    for track, arch in architectures.items():
        cross_domain = {
            target: matrix_payload["degradation_matrix"][f"{track}_arch"][f"{target}_data"]["pct_degradation"]
            for target in TRACK_ORDER
            if target != track
        }
        max_deg = max(value for value in cross_domain.values() if value is not None)
        classification = "universal" if max_deg < 10.0 else "domain_specific"
        for description in extract_innovations(base_source, arch.source_path.read_text()):
            innovations.append(
                {
                    "source_track": track,
                    "description": description,
                    "type": "architectural",
                    "cross_domain_degradation": cross_domain,
                    "classification": classification,
                }
            )
    universal_count = sum(item["classification"] == "universal" for item in innovations)
    domain_count = sum(item["classification"] == "domain_specific" for item in innovations)
    total = len(innovations)
    payload = {
        "generated_at": now_iso(),
        "innovations": innovations,
        "summary": {
            "total_innovations": total,
            "universal_count": universal_count,
            "domain_specific_count": domain_count,
            "universal_pct": round(100.0 * universal_count / total, 6) if total else 0.0,
            "domain_specific_pct": round(100.0 * domain_count / total, 6) if total else 0.0,
        },
    }
    return payload


def plot_innovation_pie(innovation_payload: dict[str, Any]) -> None:
    ensure_dir(FIGURES_DIR)
    summary = innovation_payload["summary"]
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.pie(
        [summary["universal_count"], summary["domain_specific_count"]],
        labels=["Universal", "Domain-specific"],
        autopct="%1.1f%%",
        colors=["#4f6d7a", "#dd6e42"],
        startangle=120,
    )
    ax.set_title("SC-6 Innovation Classification")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "innovation_pie.png", dpi=200)
    plt.close(fig)


def verify_sc6(matrix_payload: dict[str, Any], freezing_payload: dict[str, Any], length_payload: dict[str, Any], innovation_payload: dict[str, Any]) -> dict[str, bool]:
    all_cells = [
        matrix_payload["matrix"][f"{arch}_arch"][f"{target}_data"]
        for arch in TRACK_ORDER
        for target in TRACK_ORDER
    ]
    matrix_complete = all(cell["mean"] is not None and len(cell["runs"]) == len(matrix_payload["replicate_seeds"]) for cell in all_cells)
    identity_zero = all(
        abs(
            matrix_payload["degradation_matrix"][f"{track}_arch"][f"{track}_data"]["pct_degradation"]
        ) < 1e-9
        for track in TRACK_ORDER
    )
    monotonic_freezing = True
    for pair in freezing_payload["pairs"]:
        values = [item["val_bpb"] for item in pair["freeze_levels"]]
        if any(value is None for value in values):
            monotonic_freezing = False
            break
        monotonic_freezing = monotonic_freezing and all(
            float(values[idx]) <= float(values[idx + 1]) + 1e-9 for idx in range(len(values) - 1)
        )
    length_improves = any(item.get("h3c_criterion_met") for item in length_payload["pairs"]) if length_payload["pairs"] else False
    figure_paths = [
        FIGURES_DIR / "transfer_heatmap.png",
        FIGURES_DIR / "layer_freezing_curves.png",
        FIGURES_DIR / "innovation_pie.png",
    ]
    return {
        "matrix_complete": matrix_complete,
        "degradation_present": all(
            matrix_payload["degradation_matrix"][f"{arch}_arch"][f"{target}_data"]["pct_degradation"] is not None
            for arch in TRACK_ORDER
            for target in TRACK_ORDER
        ),
        "identity_near_zero": identity_zero,
        "freezing_monotonic": monotonic_freezing,
        "length_control_reduces_degradation": length_improves,
        "innovation_count_matches": innovation_payload["summary"]["total_innovations"] == len(innovation_payload["innovations"]),
        "figures_exist": all(path.exists() for path in figure_paths),
    }


def main() -> None:
    args = parse_args()
    ensure_dir(TRANSFER_DIR)
    ensure_dir(RAW_DIR)
    architectures = discover_architectures()
    baselines = baseline_bpbs()
    seeds = resolve_seeds(args)
    time_budget = resolve_time_budget(args)
    print("SC-6 architectures:")
    for track, arch in architectures.items():
        print(f"  {track}: {arch.source_path} (native_bpb={arch.native_bpb:.6f})")
    print(f"SC-6 time_budget={time_budget}s seeds={list(seeds)}")

    matrix_targets = TRACK_ORDER if not args.smoke else ("smiles",)
    arch_targets = TRACK_ORDER if not args.smoke else ("smiles",)

    for arch_track in arch_targets:
        arch = architectures[arch_track]
        for target_track in matrix_targets:
            for seed in seeds:
                raw_path = transfer_raw_path("matrix", f"{arch_track}_to_{target_track}_seed{seed}")
                status = "cached" if is_completed_run(raw_path) else "running"
                print(f"[matrix] {status}: {arch_track} -> {target_track} seed={seed}")
                maybe_run_train(
                    raw_path,
                    arch=arch,
                    target_track=target_track,
                    seed=seed,
                    time_budget=time_budget,
                )
                matrix_payload = build_matrix_payload(architectures, baselines, seeds=seeds)
                write_json(MATRIX_PATH, matrix_payload)
    matrix_payload = build_matrix_payload(architectures, baselines, seeds=seeds)
    write_json(MATRIX_PATH, matrix_payload)
    plot_transfer_heatmap(matrix_payload)

    if args.smoke:
        checkpoints = ensure_checkpoints({"smiles": architectures["smiles"]}, time_budget=time_budget)
    else:
        checkpoints = ensure_checkpoints(architectures, time_budget=time_budget)

    freeze_arches = arch_targets
    freeze_targets = matrix_targets
    for arch_track in freeze_arches:
        arch = architectures[arch_track]
        for target_track in freeze_targets:
            if arch_track == target_track:
                continue
            for freeze_n in FREEZE_LEVELS:
                raw_path = transfer_raw_path("layer_freezing", f"{arch_track}_to_{target_track}_freeze{freeze_n}")
                status = "cached" if is_completed_run(raw_path) else "running"
                print(f"[freeze] {status}: {arch_track} -> {target_track} freeze={freeze_n}")
                maybe_run_train(
                    raw_path,
                    arch=arch,
                    target_track=target_track,
                    seed=42,
                    time_budget=time_budget,
                    checkpoint_load=checkpoints[arch_track],
                    freeze_layers=freeze_n,
                )
                freezing_payload = build_freezing_payload(
                    architectures if not args.smoke else {"smiles": architectures["smiles"]},
                    baselines,
                    checkpoints if not args.smoke else {"smiles": checkpoints["smiles"]},
                    matrix_payload,
                )
                write_json(FREEZING_PATH, freezing_payload)
    freezing_payload = build_freezing_payload(
        architectures if not args.smoke else {"smiles": architectures["smiles"]},
        baselines,
        checkpoints if not args.smoke else {"smiles": checkpoints["smiles"]},
        matrix_payload,
    )
    write_json(FREEZING_PATH, freezing_payload)
    plot_layer_freezing(freezing_payload)

    if not args.smoke:
        for arch_track, target_track, _ in worst_transfer_pairs(matrix_payload, top_k=3):
            matched_seq_len = min(architectures[arch_track].native_seq_len, architectures[target_track].native_seq_len)
            raw_path = transfer_raw_path("length_controlled", f"{arch_track}_to_{target_track}_seq{matched_seq_len}")
            status = "cached" if is_completed_run(raw_path) else "running"
            print(f"[length] {status}: {arch_track} -> {target_track} seq_len={matched_seq_len}")
            maybe_run_train(
                raw_path,
                arch=architectures[arch_track],
                target_track=target_track,
                seed=LENGTH_CONTROL_SEED,
                time_budget=time_budget,
                seq_len_override=matched_seq_len,
            )
    length_payload = build_length_payload(architectures, baselines, matrix_payload)
    write_json(LENGTH_PATH, length_payload)

    innovation_payload = build_innovation_payload(architectures, matrix_payload)
    write_json(INNOVATION_PATH, innovation_payload)
    plot_innovation_pie(innovation_payload)

    verification = verify_sc6(matrix_payload, freezing_payload, length_payload, innovation_payload)
    print("SC-6 verification:")
    for key, value in verification.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
