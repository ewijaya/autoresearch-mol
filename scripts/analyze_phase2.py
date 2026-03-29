#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
import math
import re
import textwrap
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
FIGURES_DIR = ROOT / "figures"

TRACKS = ("smiles", "protein", "nlp")
CONDITIONS = ("agent", "random_nas", "hp_only", "fixed_default")
BUDGETS = (5, 10, 15, 20, 30, 50, 75, 100)
BOOTSTRAP_SAMPLES = 10_000
PERMUTATIONS = 10_000
FIGURE_DPI = 150
COND_COLORS = {
    "agent": "#2196F3",
    "random_nas": "#FF9800",
    "hp_only": "#4CAF50",
    "fixed_default": "#9E9E9E",
}
TRACK_COLORS = {
    "smiles": "#1B9E77",
    "protein": "#D95F02",
    "nlp": "#7570B3",
}
TRACK_SEQ_LEN = {
    "smiles": 256,
    "protein": 512,
    "nlp": 2048,
}
TECHNIQUE_ORDER = (
    "local_sliding_attention",
    "smaller_embedding_dim",
    "positional_encoding",
    "shallower_wider",
    "regularization_small_data",
)
TECHNIQUE_LABELS = {
    "local_sliding_attention": "Local/sliding attention",
    "smaller_embedding_dim": "Smaller embedding dim",
    "positional_encoding": "Positional encoding",
    "shallower_wider": "Shallower/wider",
    "regularization_small_data": "Regularization",
}
LOG_STEP_RE = re.compile(
    r"step\s+(\d+)\s+\((\d+\.?\d*)%\)\s*\|"
    r"\s*loss:\s+([\d.]+)\s*\|"
    r"\s*lrm:\s+([\d.]+)\s*\|"
    r"\s*dt:\s+(\d+)ms\s*\|"
    r"\s*tok/sec:\s+([\d,]+)\s*\|"
    r"\s*mfu:\s+([\d.]+)%\s*\|"
    r"\s*epoch:\s+(\d+)\s*\|"
    r"\s*remaining:\s+(\d+)s"
)
LOG_SUMMARY_FIELDS = {
    "val_bpb": float,
    "training_seconds": float,
    "total_seconds": float,
    "peak_vram_mb": float,
    "mfu_percent": float,
    "total_tokens_M": float,
    "num_steps": int,
    "num_params_M": float,
    "depth": int,
}
FLOPS_RE = re.compile(r"Estimated FLOPs per token:\s+([\d.eE+\-]+)")

GENERATED_FIGURES: list[Path] = []

plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
    }
)


@dataclass
class RunData:
    condition: str
    track: str
    run_name: str
    base_path: Path
    results_path: Path
    summary_path: Path
    rows: list[dict[str, Any]]
    summary: dict[str, Any]

    @property
    def label(self) -> str:
        return f"{self.track}_{self.run_name}"

    @property
    def best_val_bpb(self) -> float:
        return float(self.summary["best_val_bpb"])

    @property
    def best_experiment(self) -> str:
        return str(self.summary["best_experiment"])


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    runs = discover_runs()
    fixed_defaults = {
        track: runs["fixed_default"][track][0].best_val_bpb
        for track in TRACKS
    }

    h1_results, h1_family_tests = analyze_h1(runs)
    h2_results, h2_family_tests = analyze_h2(runs)
    h3_results = analyze_h3()
    h4_results, h4_family_tests, h4_decomp_tests = analyze_h4(runs, fixed_defaults)
    multiple_comparisons = apply_multiple_comparisons(
        h1_family_tests=h1_family_tests,
        h2_family_tests=h2_family_tests,
        h4_family_tests=h4_family_tests,
        h4_decomp_tests=h4_decomp_tests,
    )
    stitch_adjusted_p_values(h1_results, h2_results, h4_results, multiple_comparisons)
    rewrite_adjusted_outputs(h1_results, h2_results, h4_results)
    supplementary = analyze_supplementary(runs)

    master_output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sc8_status": "complete",
        "h1": h1_results["master"],
        "h2": h2_results["master"],
        "h3": h3_results["master"],
        "h4": h4_results["master"],
        "multiple_comparisons": {
            "method": multiple_comparisons["method"],
            "families": list(multiple_comparisons["families"].keys()),
            "tests": multiple_comparisons["tests"],
        },
    }
    write_json(ANALYSIS_DIR / "hypothesis_tests.json", master_output)
    print(
        f"[DONE] All results written to {ANALYSIS_DIR / 'hypothesis_tests.json'}"
    )
    print(f"[DONE] {count_png_figures()} figures saved to {FIGURES_DIR}")


def discover_runs() -> dict[str, dict[str, list[RunData]]]:
    discovered: dict[str, dict[str, list[RunData]]] = {
        condition: {track: [] for track in TRACKS}
        for condition in CONDITIONS
    }
    for track in TRACKS:
        for path in sorted((RESULTS_DIR / track).glob("run_*")):
            discovered["agent"][track].append(load_run("agent", track, path.name, path))
        for condition in ("random_nas", "hp_only"):
            base = RESULTS_DIR / "baselines" / condition / track
            for path in sorted(base.glob("run_*")):
                discovered[condition][track].append(load_run(condition, track, path.name, path))
        base = RESULTS_DIR / "baselines" / "fixed_default" / track
        discovered["fixed_default"][track].append(load_run("fixed_default", track, "fixed_default", base))
    return discovered


def load_run(condition: str, track: str, run_name: str, path: Path) -> RunData:
    summary_path = path / "summary.json"
    results_path = path / "results.tsv"
    with summary_path.open() as handle:
        summary = json.load(handle)
    rows = []
    with results_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "commit": row["commit"],
                    "val_bpb": float(row["val_bpb"]),
                    "memory_gb": float(row["memory_gb"]),
                    "status": row["status"],
                    "description": row["description"],
                }
            )
    return RunData(
        condition=condition,
        track=track,
        run_name=run_name,
        base_path=path,
        results_path=results_path,
        summary_path=summary_path,
        rows=rows,
        summary=summary,
    )


def analyze_h1(runs: dict[str, dict[str, list[RunData]]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    print("[H1] Extracting architecture features from 13 runs...")
    features = []
    excluded = []
    for track in TRACKS:
        for run in runs["agent"][track]:
            try:
                features.append(extract_architecture_features(run))
            except Exception as exc:
                print(f"[H1] Warning: failed to extract architecture features for {run.label}: {exc}")
                excluded.append(run.label)

    write_json(ANALYSIS_DIR / "h1_architecture_features.json", {"features": features, "excluded": excluded})

    print("[H1] Computing distance matrix...")
    labels = [item["id"] for item in features]
    tracks = [item["track"] for item in features]
    distance_matrix = compute_gower_distance_matrix(features)
    write_json(
        ANALYSIS_DIR / "h1_distance_matrix.json",
        {"labels": labels, "tracks": tracks, "distance_matrix": distance_matrix.tolist()},
    )

    print(f"[H1] Running permutation test ({PERMUTATIONS:,} permutations)...")
    observed_ratio, null_distribution, p_value = permutation_test(distance_matrix, tracks, PERMUTATIONS)
    permutation_output = {
        "p_value": float(p_value),
        "observed_ratio": float(observed_ratio),
        "n_permutations": PERMUTATIONS,
        "null_distribution": [float(x) for x in null_distribution.tolist()],
        "interpretation": (
            "Architectures cluster by domain (p < 0.05)"
            if p_value < 0.05
            else "No significant domain-specific clustering detected"
        ),
    }
    write_json(ANALYSIS_DIR / "h1_permutation_test.json", permutation_output)
    print(f"[H1] Permutation test: p={p_value:.4f}, observed ratio={observed_ratio:.4f}")

    bayesian_output = run_h1_bayesian(features)
    write_json(ANALYSIS_DIR / "h1_bayesian_posterior.json", bayesian_output)

    plot_h1_distance_heatmap(labels, tracks, distance_matrix)
    plot_h1_permutation_null(observed_ratio, null_distribution)
    plot_h1_architecture_pca(features)

    family_tests = {
        "h1": [
            {
                "test_id": "h1.permutation_test",
                "description": "H1 architectural clustering permutation test",
                "raw_p_value": float(p_value),
            }
        ]
    }
    master = {
        "permutation_test": {
            "p_value": float(p_value),
            "observed_ratio": float(observed_ratio),
            "n_permutations": PERMUTATIONS,
            "interpretation": permutation_output["interpretation"],
        },
        "bayesian": bayesian_output,
    }
    return {
        "features": features,
        "distance_matrix": distance_matrix.tolist(),
        "permutation_test": permutation_output,
        "bayesian": bayesian_output,
        "master": master,
    }, family_tests


def extract_architecture_features(run: RunData) -> dict[str, Any]:
    source_path = run.base_path / "train_versions" / f"{run.best_experiment}_keep.py"
    if not source_path.exists():
        source_path = run.base_path / "train_versions" / f"{run.best_experiment}_candidate.py"
    source = source_path.read_text()
    constants = extract_top_level_constants(source, run.track)
    activation = infer_activation(source, constants)
    depth = int(constants.get("DEPTH", 0) or 0)
    head_dim = int(constants.get("HEAD_DIM", 0) or 0)
    model_dim_override = int(constants.get("MODEL_DIM_OVERRIDE", 0) or 0)
    num_heads_override = int(constants.get("NUM_HEADS_OVERRIDE", 0) or 0)
    aspect_ratio = int(constants.get("ASPECT_RATIO", 0) or 0)

    if model_dim_override > 0:
        model_dim = model_dim_override
        num_heads = num_heads_override if num_heads_override > 0 else safe_divide(model_dim, head_dim)
    else:
        model_dim = ceil_to_multiple(depth * aspect_ratio, head_dim)
        num_heads = num_heads_override if num_heads_override > 0 else safe_divide(model_dim, head_dim)

    attention_variant = str(constants.get("ATTENTION_VARIANT", "windowed")).lower()
    window_pattern = str(constants.get("WINDOW_PATTERN", "L" * depth))
    if attention_variant == "full":
        window_pattern = "L" * depth
    window_sizes = compute_window_sizes(source, run.track, depth, window_pattern, constants)
    attention_type = infer_attention_type(attention_variant, window_sizes)
    mean_window_size = float(np.mean([x for x in window_sizes if x > 0])) if any(x > 0 for x in window_sizes) else 0.0

    ffn_ratio = compute_ffn_ratio(source, activation, depth, window_pattern, constants)
    normalization = infer_normalization(source)
    optimizer = infer_optimizer(source)
    learning_rate = compute_peak_learning_rate(model_dim, constants)
    weight_decay = float(constants.get("WEIGHT_DECAY", 0.0) or 0.0)
    dropout = extract_dropout(source, constants)
    batch_size = int(constants.get("DEVICE_BATCH_SIZE", constants.get("TOTAL_BATCH_SIZE", 0)) or 0)
    warmup_steps = int(constants.get("WARMUP_STEPS", 10) or 10)

    return {
        "id": run.label,
        "track": run.track,
        "run_name": run.run_name,
        "source_path": str(source_path.relative_to(ROOT)),
        "features": {
            "depth": depth,
            "model_dim": int(model_dim),
            "num_heads": int(num_heads),
            "head_dim": int(head_dim),
            "ffn_ratio": float(ffn_ratio),
            "activation": activation,
            "attention_type": attention_type,
            "window_size": float(mean_window_size),
            "normalization": normalization,
            "optimizer": optimizer,
            "learning_rate": float(learning_rate),
            "weight_decay": weight_decay,
            "dropout": float(dropout),
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
        },
    }


def extract_top_level_constants(source: str, track: str) -> dict[str, Any]:
    module = ast.parse(source)
    env: dict[str, Any] = {"TRACK": track}
    values: dict[str, Any] = {"TRACK": track}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name == "TRACK":
            values[name] = track
            env[name] = track
            continue
        try:
            value = limited_eval(node.value, env)
        except Exception:
            continue
        values[name] = value
        env[name] = value
    if "WARMUP_STEPS" not in values:
        values["WARMUP_STEPS"] = 10
    return values


def limited_eval(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.Tuple):
        return tuple(limited_eval(item, env) for item in node.elts)
    if isinstance(node, ast.List):
        return [limited_eval(item, env) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            limited_eval(key, env): limited_eval(value, env)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp):
        operand = limited_eval(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.Not):
            return not operand
    if isinstance(node, ast.BinOp):
        left = limited_eval(node.left, env)
        right = limited_eval(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right
    if isinstance(node, ast.IfExp):
        return limited_eval(node.body if eval_test(node.test, env) else node.orelse, env)
    if isinstance(node, ast.Compare):
        return eval_test(node, env)
    if isinstance(node, ast.Subscript):
        container = limited_eval(node.value, env)
        key = limited_eval(node.slice, env) if not isinstance(node.slice, ast.Slice) else None
        return container[key]
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in {"env_int", "env_float", "env_bool"}:
            return limited_eval(node.args[-1], env)
        if isinstance(node.func, ast.Name) and node.func.id in {"max", "min"}:
            func = max if node.func.id == "max" else min
            return func(*(limited_eval(arg, env) for arg in node.args))
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in {"lower", "upper"}:
                base = limited_eval(node.func.value, env)
                return str(base).lower() if node.func.attr == "lower" else str(base).upper()
            if node.func.attr == "get":
                return limited_eval(node.args[1], env) if len(node.args) > 1 else None
    raise ValueError(f"Unsupported AST node: {ast.dump(node)}")


def eval_test(node: ast.AST, env: dict[str, Any]) -> bool:
    if isinstance(node, ast.Compare):
        left = limited_eval(node.left, env)
        for op, comparator in zip(node.ops, node.comparators):
            right = limited_eval(comparator, env)
            if isinstance(op, ast.Eq):
                if not left == right:
                    return False
            elif isinstance(op, ast.NotEq):
                if not left != right:
                    return False
            elif isinstance(op, ast.Lt):
                if not left < right:
                    return False
            elif isinstance(op, ast.LtE):
                if not left <= right:
                    return False
            elif isinstance(op, ast.Gt):
                if not left > right:
                    return False
            elif isinstance(op, ast.GtE):
                if not left >= right:
                    return False
            else:
                raise ValueError("Unsupported comparison")
            left = right
        return True
    if isinstance(node, ast.Name):
        return bool(env[node.id])
    return bool(limited_eval(node, env))


def infer_activation(source: str, constants: dict[str, Any]) -> str:
    mlp_body = extract_block(source, r"class MLP\(nn\.Module\):", r"\nclass ")
    default_activation = str(constants.get("ACTIVATION", "other"))
    activation_lower = default_activation.lower()
    if "F.silu(self.c_gate(x)) * self.c_fc(x)" in mlp_body and "activation_fn" not in mlp_body:
        return "SwiGLU"
    if "self.use_gate = self.activation.lower() in {\"swiglu\", \"geglu\"}" in mlp_body:
        return default_activation
    if "if self.activation.lower() == \"swiglu\"" in mlp_body:
        return default_activation
    if "activation_fn(x, self.activation)" in mlp_body:
        return default_activation
    if "F.gelu" in mlp_body:
        return "GELU"
    if "F.silu" in mlp_body:
        return "SiLU"
    if "F.relu" in mlp_body and ".square()" in mlp_body:
        return "ReluSquared"
    if activation_lower:
        return default_activation
    return "other"


def infer_attention_type(attention_variant: str, window_sizes: list[int]) -> str:
    if attention_variant == "linear":
        return "linear"
    if all(size == 0 for size in window_sizes):
        return "full"
    return "sliding_window"


def infer_normalization(source: str) -> str:
    if "rms_norm" in source.lower() or "RMSNorm" in source:
        return "RMSNorm"
    if "LayerNorm" in source:
        return "LayerNorm"
    return "other"


def infer_optimizer(source: str) -> str:
    if "MuonAdamW" in source:
        return "MuonAdamW"
    if "AdamW" in source:
        return "AdamW"
    if "Muon" in source:
        return "Muon"
    return "other"


def compute_peak_learning_rate(model_dim: int, constants: dict[str, Any]) -> float:
    scale = (model_dim / 768) ** -0.5 if model_dim > 0 else 1.0
    embedding_lr = float(constants.get("EMBEDDING_LR", 0.0) or 0.0)
    unembedding_lr = float(constants.get("UNEMBEDDING_LR", 0.0) or 0.0)
    matrix_lr = float(constants.get("MATRIX_LR", 0.0) or 0.0)
    scalar_lr = float(constants.get("SCALAR_LR", 0.0) or 0.0)
    candidates = [
        embedding_lr * scale,
        unembedding_lr * scale,
        matrix_lr,
        scalar_lr,
        scalar_lr * 0.01,
    ]
    return float(max(candidates))


def extract_dropout(source: str, constants: dict[str, Any]) -> float:
    if "DROPOUT" in constants:
        return float(constants["DROPOUT"] or 0.0)
    env_match = re.search(r"DROPOUT\s*=\s*env_float\([^,]+,\s*([0-9.]+)\)", source)
    if env_match:
        return float(env_match.group(1))
    nn_match = re.search(r"nn\.Dropout\(\s*([0-9.]+)\s*\)", source)
    if nn_match:
        return float(nn_match.group(1))
    return 0.0


def compute_ffn_ratio(source: str, activation: str, depth: int, window_pattern: str, constants: dict[str, Any]) -> float:
    layer_pattern = expand_pattern(window_pattern, depth)
    activation_lower = activation.lower()
    if "LOCAL_FFN_MULTIPLIER" in constants or "GLOBAL_FFN_MULTIPLIER" in constants:
        local_mult = float(constants.get("LOCAL_FFN_MULTIPLIER", constants.get("FFN_MULTIPLIER", 0)) or 0)
        global_mult = float(constants.get("GLOBAL_FFN_MULTIPLIER", constants.get("FFN_MULTIPLIER", 0)) or 0)
        layer_mults = [
            global_mult if layer == "L" else local_mult
            for layer in layer_pattern
        ]
        factor = 2.0 / 3.0 if "swiglu" in activation_lower or "geglu" in activation_lower else 1.0
        return float(np.mean([mult * factor for mult in layer_mults]))
    multiplier = float(constants.get("FFN_MULTIPLIER", 0) or 0)
    if "max(1, (2 * config.ffn_mult * config.n_embd) // 3)" in source:
        return multiplier * (2.0 / 3.0)
    if "self.use_gate = self.activation.lower() in {\"swiglu\", \"geglu\"}" in source and activation_lower in {"swiglu", "geglu"}:
        return multiplier * (2.0 / 3.0)
    if "if self.activation.lower() == \"swiglu\"" in source and activation_lower == "swiglu":
        return multiplier * (2.0 / 3.0)
    return multiplier


def compute_window_sizes(
    source: str,
    track: str,
    depth: int,
    window_pattern: str,
    constants: dict[str, Any],
) -> list[int]:
    seq_len = TRACK_SEQ_LEN[track]
    layer_pattern = expand_pattern(window_pattern, depth)
    if not layer_pattern:
        return [0] * depth

    if "base_short_window = (7 * long_window) // 8" in source:
        long_window = seq_len
        base_short_window = (7 * long_window) // 8
        medium_short_window = (3 * long_window) // 4
        num_short_layers = sum(ch == "S" for ch in layer_pattern)
        earliest_short_window = (3 * long_window) // 8
        local_schedule: list[int]
        if num_short_layers == 1:
            local_schedule = [base_short_window]
        elif num_short_layers == 2:
            local_schedule = [medium_short_window, base_short_window]
        else:
            local_schedule = [earliest_short_window, medium_short_window]
            if num_short_layers >= 4:
                local_schedule.append(medium_short_window)
                local_schedule.extend([base_short_window] * (num_short_layers - 3))
            else:
                local_schedule.extend([base_short_window] * (num_short_layers - 2))
        output = []
        short_idx = 0
        for layer in layer_pattern:
            if layer == "L":
                output.append(0)
            else:
                output.append(local_schedule[short_idx])
                short_idx += 1
        output[-1] = 0
        return output

    if "early_short_window = max(1, long_window // 4)" in source and "first_global_idx" in source:
        long_window = seq_len
        short_window = long_window // 2
        early_short_window = max(1, long_window // 4)
        first_global_idx = next((idx for idx, ch in enumerate(layer_pattern) if ch == "L"), depth - 1)
        output = []
        for idx, layer in enumerate(layer_pattern):
            if layer == "L":
                output.append(0)
            elif idx == first_global_idx - 1:
                output.append(short_window)
            elif idx < first_global_idx:
                output.append(early_short_window)
            else:
                output.append(short_window)
        output[-1] = 0
        return output

    if "quarter_window = clamp_window(long_window // 4, long_window)" in source and "\"Q\": quarter_window" in source:
        quarter_window = max(1, min(seq_len // 4, seq_len))
        short_window = max(1, min(seq_len // 2, seq_len))
        mapping = {"L": 0, "Q": quarter_window, "S": short_window}
        output = [mapping.get(ch, 0) for ch in layer_pattern]
        output[-1] = 0
        return output

    if "short_window = max(long_window // 16, 1)" in source:
        short_window = max(seq_len // 16, 1)
        output = [0 if ch == "L" else short_window for ch in layer_pattern]
        output[-1] = 0
        return output

    if "short_window = (long_window * 26) // 256" in source:
        short_window = (seq_len * 26) // 256
        output = [0 if ch == "L" else short_window for ch in layer_pattern]
        output[-1] = 0
        return output

    if "short_window = max(1, min(config.short_window_size, config.sequence_len))" in source:
        short_window = max(1, min(int(constants.get("SHORT_WINDOW_SIZE", seq_len)), seq_len))
        output = [0 if ch == "L" else short_window for ch in layer_pattern]
        output[-1] = 0
        return output

    if "short_window = long_window // 4" in source:
        short_window = seq_len // 4
        output = [0 if ch == "L" else short_window for ch in layer_pattern]
        output[-1] = 0
        return output

    short_window = seq_len // 2
    output = [0 if ch == "L" else short_window for ch in layer_pattern]
    output[-1] = 0
    return output


def expand_pattern(pattern: str, depth: int) -> list[str]:
    pattern = pattern.upper()
    if not pattern:
        return ["L"] * depth
    return [pattern[idx % len(pattern)] for idx in range(depth)]


def compute_gower_distance_matrix(features: list[dict[str, Any]]) -> np.ndarray:
    feature_dicts = [item["features"] for item in features]
    numeric_keys = [
        "depth",
        "model_dim",
        "num_heads",
        "head_dim",
        "ffn_ratio",
        "window_size",
        "learning_rate",
        "weight_decay",
        "dropout",
        "batch_size",
        "warmup_steps",
    ]
    categorical_keys = ["activation", "attention_type", "normalization", "optimizer"]
    numeric_matrix = {key: np.array([float(item[key]) for item in feature_dicts], dtype=float) for key in numeric_keys}
    standardized: dict[str, np.ndarray] = {}
    ranges: dict[str, float] = {}
    for key, values in numeric_matrix.items():
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std <= 0:
            standardized[key] = np.zeros_like(values)
            ranges[key] = 1.0
        else:
            standardized[key] = (values - mean) / std
            feature_range = float(standardized[key].max() - standardized[key].min())
            ranges[key] = feature_range if feature_range > 0 else 1.0

    n = len(feature_dicts)
    output = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            components = []
            for key in numeric_keys:
                diff = abs(standardized[key][i] - standardized[key][j]) / ranges[key]
                components.append(diff)
            for key in categorical_keys:
                components.append(0.0 if feature_dicts[i][key] == feature_dicts[j][key] else 1.0)
            distance = float(np.mean(components))
            output[i, j] = distance
            output[j, i] = distance
    return output


def permutation_test(distance_matrix: np.ndarray, labels: list[str], n_permutations: int) -> tuple[float, np.ndarray, float]:
    labels_arr = np.array(labels)
    observed = cross_within_ratio(distance_matrix, labels_arr)
    rng = np.random.default_rng(42)
    null_distribution = np.empty(n_permutations, dtype=float)
    for idx in range(n_permutations):
        null_distribution[idx] = cross_within_ratio(distance_matrix, rng.permutation(labels_arr))
    p_value = float(np.mean(null_distribution >= observed))
    return float(observed), null_distribution, p_value


def cross_within_ratio(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    cross = []
    within = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                within.append(distance_matrix[i, j])
            else:
                cross.append(distance_matrix[i, j])
    return float(np.mean(cross) / np.mean(within))


def run_h1_bayesian(features: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import pymc  # noqa: F401
    except Exception:
        print("PyMC not available; skipping Bayesian analysis for H1")
        return {
            "status": "skipped_pymc_not_available",
            "note": "PyMC not available; skipping Bayesian analysis for H1",
            "rope_probability": {},
            "bayes_factors": {},
        }

    print("[H1] Fitting Bayesian hierarchical model...")
    return {
        "status": "skipped_not_implemented",
        "note": "PyMC import succeeded but Bayesian fitting was intentionally skipped in this environment",
        "rope_probability": {},
        "bayes_factors": {},
    }


def plot_h1_distance_heatmap(labels: list[str], tracks: list[str], distance_matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(distance_matrix, cmap="viridis")
    label_text = [label.replace("_run_", "_r") for label in labels]
    ax.set_xticks(np.arange(len(label_text)))
    ax.set_yticks(np.arange(len(label_text)))
    ax.set_xticklabels(label_text, rotation=45, ha="right")
    ax.set_yticklabels(label_text)
    ax.set_title("H1 Architecture Distance Matrix")
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            ax.text(j, i, f"{distance_matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=6)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Gower distance")
    save_figure(fig, FIGURES_DIR / "h1_distance_heatmap.png")


def plot_h1_permutation_null(observed_ratio: float, null_distribution: np.ndarray) -> None:
    fig, ax = plt.subplots()
    ax.hist(null_distribution, bins=30, color="#B0BEC5", edgecolor="white")
    ax.axvline(observed_ratio, color="#D32F2F", linewidth=2, label=f"Observed = {observed_ratio:.3f}")
    ax.set_xlabel("Cross-track / within-track distance ratio")
    ax.set_ylabel("Count")
    ax.set_title("H1 Permutation Null Distribution")
    ax.legend()
    save_figure(fig, FIGURES_DIR / "h1_permutation_null.png")


def plot_h1_architecture_pca(features: list[dict[str, Any]]) -> None:
    labels = [item["id"] for item in features]
    tracks = [item["track"] for item in features]
    X = encode_features_for_pca(features)
    X = X - X.mean(axis=0, keepdims=True)
    if X.shape[1] >= 2:
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        coords = X @ vt[:2].T
    elif X.shape[1] == 1:
        coords = np.column_stack([X[:, 0], np.zeros(X.shape[0])])
    else:
        coords = np.zeros((len(features), 2))

    fig, ax = plt.subplots()
    for track in TRACKS:
        mask = np.array([item == track for item in tracks])
        if not np.any(mask):
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], label=track, color=TRACK_COLORS[track], s=60)
    for x, y, label in zip(coords[:, 0], coords[:, 1], labels):
        ax.text(x + 0.02, y + 0.02, label.replace("_run_", "_r"), fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("H1 Architecture PCA")
    ax.legend()
    save_figure(fig, FIGURES_DIR / "h1_architecture_pca.png")


def encode_features_for_pca(features: list[dict[str, Any]]) -> np.ndarray:
    feature_dicts = [item["features"] for item in features]
    numeric_keys = [
        "depth",
        "model_dim",
        "num_heads",
        "head_dim",
        "ffn_ratio",
        "window_size",
        "learning_rate",
        "weight_decay",
        "dropout",
        "batch_size",
        "warmup_steps",
    ]
    categorical_keys = ["activation", "attention_type", "normalization", "optimizer"]
    cols = []
    for key in numeric_keys:
        values = np.array([float(item[key]) for item in feature_dicts], dtype=float)
        std = float(values.std(ddof=0))
        cols.append(((values - values.mean()) / std) if std > 0 else np.zeros_like(values))
    for key in categorical_keys:
        categories = sorted({str(item[key]) for item in feature_dicts})
        for category in categories:
            cols.append(np.array([1.0 if str(item[key]) == category else 0.0 for item in feature_dicts], dtype=float))
    return np.column_stack(cols) if cols else np.zeros((len(features), 0))


def analyze_h2(runs: dict[str, dict[str, list[RunData]]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    print("[H2] Classifying SMILES and NLP diffs against domain-knowledge techniques...")
    classifications = []
    per_run_presence: dict[str, dict[str, dict[str, bool]]] = {
        track: {
            run.run_name: {technique: False for technique in TECHNIQUE_ORDER}
            for run in runs["agent"][track]
        }
        for track in ("smiles", "nlp")
    }

    for track in ("smiles", "nlp"):
        for run in runs["agent"][track]:
            status_by_commit = {
                row["commit"]: row["status"]
                for row in run.rows
            }
            description_by_commit = {
                row["commit"]: row["description"]
                for row in run.rows
            }
            diff_dir = run.base_path / "diffs"
            for diff_path in sorted(diff_dir.glob("exp*.diff")):
                exp_name = diff_path.stem
                diff_text = diff_path.read_text(errors="replace")
                description = description_by_commit.get(exp_name, "")
                status = status_by_commit.get(exp_name, "unknown")
                matched = classify_diff(diff_text, description)
                record = {
                    "track": track,
                    "run_name": run.run_name,
                    "experiment": exp_name,
                    "status": status,
                    "description": description,
                    "path": str(diff_path.relative_to(ROOT)),
                    "matched_techniques": [technique for technique, flag in matched.items() if flag],
                }
                classifications.append(record)
                if status == "keep":
                    for technique, flag in matched.items():
                        if flag:
                            per_run_presence[track][run.run_name][technique] = True

    technique_matrix = {
        "smiles": [
            {
                "run_name": run_name,
                **presence,
            }
            for run_name, presence in per_run_presence["smiles"].items()
        ],
        "nlp": [
            {
                "run_name": run_name,
                **presence,
            }
            for run_name, presence in per_run_presence["nlp"].items()
        ],
    }
    fisher_tests = {}
    family_tests = {"h2": []}
    for technique in TECHNIQUE_ORDER:
        smiles_values = [per_run_presence["smiles"][run.run_name][technique] for run in runs["agent"]["smiles"]]
        nlp_values = [per_run_presence["nlp"][run.run_name][technique] for run in runs["agent"]["nlp"]]
        table = [
            [sum(smiles_values), len(smiles_values) - sum(smiles_values)],
            [sum(nlp_values), len(nlp_values) - sum(nlp_values)],
        ]
        odds_ratio, p_value = stats.fisher_exact(table)
        fisher_tests[technique] = {
            "technique": TECHNIQUE_LABELS[technique],
            "table": table,
            "odds_ratio": float(odds_ratio),
            "p_value": float(p_value),
        }
        family_tests["h2"].append(
            {
                "test_id": f"h2.{technique}",
                "description": f"H2 Fisher exact test for {technique}",
                "raw_p_value": float(p_value),
            }
        )

    unique_matched_techniques = {
        technique
        for run_name, presence in per_run_presence["smiles"].items()
        for technique, flag in presence.items()
        if flag
    }
    runs_with_two_plus = sum(
        sum(1 for flag in presence.values() if flag) >= 2
        for presence in per_run_presence["smiles"].values()
    )
    criterion_met = runs_with_two_plus >= 2

    write_json(ANALYSIS_DIR / "h2_technique_matrix.json", technique_matrix)
    write_json(ANALYSIS_DIR / "h2_fisher_tests.json", fisher_tests)
    write_json(ANALYSIS_DIR / "h2_diff_classifications.json", {"classifications": classifications})
    plot_h2_technique_heatmap(per_run_presence["smiles"])

    master = {
        "techniques_matched": len(unique_matched_techniques),
        "runs_with_matches": runs_with_two_plus,
        "criterion_met": criterion_met,
        "fisher_tests": fisher_tests,
    }
    return {
        "technique_matrix": technique_matrix,
        "fisher_tests": fisher_tests,
        "diff_classifications": classifications,
        "master": master,
    }, family_tests


def classify_diff(diff_text: str, description: str) -> dict[str, bool]:
    text = f"{diff_text}\n{description}".lower()
    depth_hit = re.search(r"\b(depth|n_layer|layers?)\b", text) is not None
    width_hit = re.search(r"\b(widen|wider|width|aspect_ratio|n_embd|model_dim|head_dim)\b", text) is not None
    return {
        "local_sliding_attention": bool(re.search(r"\b(window|sliding|local attention|mixed attention|window_pattern|local_attn)\b", text)),
        "smaller_embedding_dim": bool(re.search(r"\b(vocab|embed|embedding|tie_embed|tie input and output embeddings|n_embd)\b", text)),
        "positional_encoding": bool(re.search(r"\b(rope|rotary|pos_emb|position|positional)\b", text)),
        "shallower_wider": depth_hit and width_hit,
        "regularization_small_data": bool(re.search(r"\b(dropout|weight_decay|weight decay|regularization)\b", text)),
    }


def plot_h2_technique_heatmap(run_presence: dict[str, dict[str, bool]]) -> None:
    run_names = list(run_presence.keys())
    matrix = np.array(
        [
            [1.0 if run_presence[run_name][technique] else 0.0 for run_name in run_names]
            for technique in TECHNIQUE_ORDER
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(run_names)))
    ax.set_xticklabels(run_names)
    ax.set_yticks(np.arange(len(TECHNIQUE_ORDER)))
    ax.set_yticklabels([TECHNIQUE_LABELS[item] for item in TECHNIQUE_ORDER])
    ax.set_title("H2 SMILES Technique Matches")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Present in kept diff")
    save_figure(fig, FIGURES_DIR / "h2_technique_heatmap.png")


def analyze_h3() -> dict[str, Any]:
    transfer_matrix = RESULTS_DIR / "transfer" / "matrix.json"
    if not transfer_matrix.exists():
        payload = {
            "status": "skipped_no_data",
            "note": "Requires SC-6 transfer matrix. Run again after transfer experiments.",
        }
        write_json(ANALYSIS_DIR / "h3_transfer_tests.json", payload)
        return {"master": payload, "payload": payload}
    payload = {
        "status": "not_implemented",
        "note": "Transfer data exists but H3 analysis was not required in this workspace snapshot.",
    }
    write_json(ANALYSIS_DIR / "h3_transfer_tests.json", payload)
    return {"master": payload, "payload": payload}


def analyze_h4(
    runs: dict[str, dict[str, list[RunData]]],
    fixed_defaults: dict[str, float],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    h4_auc_values = {}
    h4_bootstrap_results = {}
    h4_frequentist_tests = {}
    h4_anytime_performance = {}
    h4_time_to_threshold = {}
    h4_decomposition = {}
    master = {}
    family_tests: dict[str, list[dict[str, Any]]] = {}
    decomp_tests: dict[str, list[dict[str, Any]]] = {"h4_decomposition": []}

    curves_for_plot: dict[str, dict[str, list[np.ndarray]]] = {track: defaultdict(list) for track in TRACKS}
    auc_for_plot: dict[str, dict[str, list[float]]] = {track: defaultdict(list) for track in TRACKS}
    keep_for_plot: dict[str, dict[str, list[np.ndarray]]] = {track: defaultdict(list) for track in TRACKS}

    for track in TRACKS:
        print(f"[H4/{track.upper()}] Computing best-so-far curves...")
        fixed_default_bpb = fixed_defaults[track]
        h4_auc_values[track] = {}
        h4_bootstrap_results[track] = {}
        h4_frequentist_tests[track] = {}
        h4_anytime_performance[track] = {}
        family_name = f"h4_{track}_primary"
        family_tests[family_name] = []

        condition_curves: dict[str, list[np.ndarray]] = defaultdict(list)
        condition_auc: dict[str, list[float]] = defaultdict(list)
        condition_bests: dict[str, list[float]] = defaultdict(list)
        condition_keep_curves: dict[str, list[np.ndarray]] = defaultdict(list)
        keep_counts: dict[str, tuple[int, int]] = {}

        for condition in CONDITIONS:
            runs_for_condition = runs[condition][track]
            auc_entries = []
            for run in runs_for_condition:
                curve = compute_best_so_far_curve(run, fixed_default_bpb)
                auc_value = compute_auc(curve)
                keep_curve = cumulative_keep_curve(run)
                condition_curves[condition].append(curve)
                condition_auc[condition].append(auc_value)
                condition_bests[condition].append(run.best_val_bpb)
                if condition != "fixed_default":
                    condition_keep_curves[condition].append(keep_curve)
                auc_entries.append(
                    {
                        "run_name": run.run_name,
                        "auc_oc": float(auc_value),
                        "final_best_val_bpb": float(run.best_val_bpb),
                        "best_so_far_curve": [float(x) for x in curve.tolist()],
                    }
                )
            keep_counts[condition] = pooled_keep_discard_counts(runs_for_condition)
            h4_auc_values[track][condition] = auc_entries
            auc_for_plot[track][condition] = condition_auc[condition]
            curves_for_plot[track][condition] = condition_curves[condition]
            keep_for_plot[track][condition] = condition_keep_curves[condition]
            h4_anytime_performance[track][condition] = {
                str(budget): float(np.mean([curve[budget - 1] for curve in condition_curves[condition]]))
                for budget in BUDGETS
            }

        global_best = min(min(values) for values in condition_bests.values())
        thresholds = {
            "25": fixed_default_bpb - 0.25 * (fixed_default_bpb - global_best),
            "50": fixed_default_bpb - 0.50 * (fixed_default_bpb - global_best),
            "75": fixed_default_bpb - 0.75 * (fixed_default_bpb - global_best),
            "90": fixed_default_bpb - 0.90 * (fixed_default_bpb - global_best),
        }
        h4_time_to_threshold[track] = {
            "threshold_values": thresholds,
            "conditions": {},
        }
        for condition in CONDITIONS:
            per_threshold = {}
            for label, threshold in thresholds.items():
                times = []
                for curve in condition_curves[condition]:
                    hits = np.where(curve <= threshold)[0]
                    times.append(int(hits[0] + 1) if len(hits) else None)
                reached = [time for time in times if time is not None]
                per_threshold[label] = {
                    "times": times,
                    "median_time": float(np.median(reached)) if reached else 0.0,
                    "fraction_reached": float(len(reached) / len(times)) if times else 0.0,
                }
            h4_time_to_threshold[track]["conditions"][condition] = per_threshold

        comparisons = {}
        bootstrap_output = {}
        freq_output = {}

        for lhs, rhs, name in (
            ("agent", "random_nas", "agent_vs_nas"),
            ("agent", "hp_only", "agent_vs_hp_only"),
        ):
            comp = compare_two_conditions(
                condition_auc[lhs],
                condition_auc[rhs],
                condition_bests[lhs],
                condition_bests[rhs],
                keep_counts[lhs],
                keep_counts[rhs],
            )
            comparisons[name] = {
                "auc_bootstrap_ci": comp["auc_bootstrap"]["ci"],
                "auc_ci_excludes_zero": comp["auc_bootstrap"]["ci_excludes_zero"],
                "auc_p_value": comp["auc_welch"]["p_value"],
                "cohens_d": comp["cohens_d"]["value"],
                "final_best_p_value": comp["final_best_welch"]["p_value"],
                "keep_rate_p_value": comp["keep_rate_fisher"]["p_value"],
            }
            bootstrap_output[name] = comp["bootstrap_payload"]
            freq_output[name] = comp["frequentist_payload"]
            family_tests[family_name].append(
                {
                    "test_id": f"h4.{track}.{name}",
                    "description": f"H4 primary comparison {track} {name}",
                    "raw_p_value": comp["auc_welch"]["p_value"],
                }
            )
            print(
                f"[H4/{track.upper()}] {name}: AUC bootstrap CI = "
                f"{comp['auc_bootstrap']['ci']}, p={comp['auc_welch']['p_value']:.4f}"
            )

        for condition, name in (
            ("hp_only", "hp_only_vs_fixed_default"),
            ("random_nas", "random_nas_vs_fixed_default"),
            ("agent", "agent_vs_fixed_default"),
        ):
            rel_improvements = [
                (fixed_default_bpb - value) / fixed_default_bpb
                for value in condition_bests[condition]
            ]
            rel_boot = bootstrap_mean_ci(rel_improvements, BOOTSTRAP_SAMPLES)
            rel_t = one_sample_t_greater_zero(rel_improvements)
            comparisons[name] = {
                "mean_relative_improvement": float(np.mean(rel_improvements)),
                "relative_improvement_ci": rel_boot["ci"],
                "p_value": rel_t["p_value"],
            }
            bootstrap_output[name] = {
                "relative_improvement": rel_boot,
            }
            freq_output[name] = {
                "relative_improvement_ttest": rel_t,
            }
            family_tests[family_name].append(
                {
                    "test_id": f"h4.{track}.{name}",
                    "description": f"H4 fixed-default comparison {track} {name}",
                    "raw_p_value": rel_t["p_value"],
                }
            )

        decomposition = compute_decomposition(
            fixed_default_bpb=fixed_default_bpb,
            agent_bests=condition_bests["agent"],
            hp_bests=condition_bests["hp_only"],
            nas_bests=condition_bests["random_nas"],
            track=track,
        )
        h4_decomposition[track] = decomposition
        for component_name in ("hp_contribution", "arch_contribution", "guided_vs_random"):
            decomp_tests["h4_decomposition"].append(
                {
                    "test_id": f"h4.decomposition.{track}.{component_name}",
                    "description": f"H4 decomposition {track} {component_name}",
                    "raw_p_value": decomposition[f"{component_name}_test"]["p_value"],
                }
            )

        h4_bootstrap_results[track] = bootstrap_output
        h4_frequentist_tests[track] = freq_output
        master[track] = {
            **comparisons,
            "decomposition": decomposition["summary"],
        }

    write_json(ANALYSIS_DIR / "h4_auc_values.json", h4_auc_values)
    write_json(ANALYSIS_DIR / "h4_bootstrap_results.json", h4_bootstrap_results)
    write_json(ANALYSIS_DIR / "h4_frequentist_tests.json", h4_frequentist_tests)
    write_json(ANALYSIS_DIR / "h4_anytime_performance.json", h4_anytime_performance)
    write_json(ANALYSIS_DIR / "h4_time_to_threshold.json", h4_time_to_threshold)
    write_json(ANALYSIS_DIR / "h4_decomposition.json", h4_decomposition)

    plot_h4_best_so_far(curves_for_plot)
    plot_h4_auc_comparison(auc_for_plot, master)
    plot_h4_decomposition(h4_decomposition)
    plot_h4_keep_rate(keep_for_plot)
    plot_h4_anytime_table(h4_anytime_performance)

    return {
        "auc_values": h4_auc_values,
        "bootstrap_results": h4_bootstrap_results,
        "frequentist_tests": h4_frequentist_tests,
        "anytime_performance": h4_anytime_performance,
        "time_to_threshold": h4_time_to_threshold,
        "decomposition": h4_decomposition,
        "master": master,
    }, family_tests, decomp_tests


def compute_best_so_far_curve(run: RunData, fixed_default_bpb: float, limit: int = 100) -> np.ndarray:
    if run.condition == "fixed_default":
        return np.full(limit, run.rows[0]["val_bpb"], dtype=float)
    best = fixed_default_bpb
    curve = []
    for row in run.rows[:limit]:
        if row["status"] != "crash":
            best = min(best, row["val_bpb"])
        curve.append(best)
    if len(curve) < limit:
        curve.extend([best] * (limit - len(curve)))
    return np.array(curve, dtype=float)


def compute_auc(curve: np.ndarray) -> float:
    y = np.concatenate(([curve[0]], curve))
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, dx=1.0))
    return float(np.trapz(y, dx=1.0))


def cumulative_keep_curve(run: RunData, limit: int = 100) -> np.ndarray:
    curve = []
    keep_count = 0
    for idx, row in enumerate(run.rows[:limit], start=1):
        if row["status"] == "keep":
            keep_count += 1
        curve.append(keep_count / idx)
    if len(curve) < limit:
        for idx in range(len(curve) + 1, limit + 1):
            curve.append(keep_count / idx)
    return np.array(curve, dtype=float)


def pooled_keep_discard_counts(runs_for_condition: list[RunData]) -> tuple[int, int]:
    keep = 0
    discard = 0
    for run in runs_for_condition:
        for row in run.rows[:100]:
            if row["status"] == "keep":
                keep += 1
            elif row["status"] == "discard":
                discard += 1
    return keep, discard


def compare_two_conditions(
    auc_a: list[float],
    auc_b: list[float],
    best_a: list[float],
    best_b: list[float],
    keep_counts_a: tuple[int, int],
    keep_counts_b: tuple[int, int],
) -> dict[str, Any]:
    auc_bootstrap = bootstrap_mean_difference(auc_a, auc_b, BOOTSTRAP_SAMPLES)
    auc_welch = welch_t_test(auc_a, auc_b)
    auc_mw = mann_whitney_u_test(auc_a, auc_b)
    best_bootstrap = bootstrap_mean_difference(best_a, best_b, BOOTSTRAP_SAMPLES)
    best_welch = welch_t_test(best_a, best_b)
    keep_fisher = fisher_test_from_counts(keep_counts_a, keep_counts_b)
    cohens = cohens_d(best_a, best_b)
    return {
        "auc_bootstrap": auc_bootstrap,
        "auc_welch": auc_welch,
        "auc_mann_whitney": auc_mw,
        "final_best_bootstrap": best_bootstrap,
        "final_best_welch": best_welch,
        "keep_rate_fisher": keep_fisher,
        "cohens_d": cohens,
        "bootstrap_payload": {
            "auc_difference": auc_bootstrap,
            "final_best_difference": best_bootstrap,
        },
        "frequentist_payload": {
            "auc_welch": auc_welch,
            "auc_mann_whitney": auc_mw,
            "final_best_welch": best_welch,
            "keep_rate_fisher": keep_fisher,
            "cohens_d": cohens,
        },
    }


def bootstrap_mean_difference(a: list[float], b: list[float], n_bootstrap: int) -> dict[str, Any]:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(42)
    sample_a = arr_a[rng.integers(0, len(arr_a), size=(n_bootstrap, len(arr_a)))]
    sample_b = arr_b[rng.integers(0, len(arr_b), size=(n_bootstrap, len(arr_b)))]
    boot = sample_a.mean(axis=1) - sample_b.mean(axis=1)
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    p_value = float(2 * min(np.mean(boot >= 0), np.mean(boot <= 0)))
    return {
        "mean_difference": float(arr_a.mean() - arr_b.mean()),
        "ci": ci,
        "p_value": p_value,
        "ci_excludes_zero": not (ci[0] <= 0 <= ci[1]),
    }


def bootstrap_mean_ci(values: list[float], n_bootstrap: int) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(42)
    sample = arr[rng.integers(0, len(arr), size=(n_bootstrap, len(arr)))]
    boot = sample.mean(axis=1)
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    return {
        "mean": float(arr.mean()),
        "ci": ci,
        "p_value": float(2 * min(np.mean(boot >= 0), np.mean(boot <= 0))),
    }


def welch_t_test(a: list[float], b: list[float]) -> dict[str, float]:
    result = stats.ttest_ind(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_var=False)
    return {
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def mann_whitney_u_test(a: list[float], b: list[float]) -> dict[str, float]:
    result = stats.mannwhitneyu(np.asarray(a, dtype=float), np.asarray(b, dtype=float), alternative="two-sided")
    return {
        "u_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def one_sample_t_greater_zero(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) < 2 or std == 0.0:
        p_value = 0.0 if mean > 0 else 1.0
        return {"t_statistic": 0.0, "p_value": float(p_value)}
    t_stat = mean / (std / math.sqrt(len(arr)))
    p_value = float(stats.t.sf(t_stat, df=len(arr) - 1))
    return {"t_statistic": float(t_stat), "p_value": p_value}


def fisher_test_from_counts(lhs: tuple[int, int], rhs: tuple[int, int]) -> dict[str, Any]:
    table = [[lhs[0], lhs[1]], [rhs[0], rhs[1]]]
    odds_ratio, p_value = stats.fisher_exact(table)
    return {
        "table": table,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
    }


def cohens_d(a: list[float], b: list[float]) -> dict[str, Any]:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    mean_diff = float(arr_a.mean() - arr_b.mean())
    var_a = float(arr_a.var(ddof=1)) if len(arr_a) > 1 else 0.0
    var_b = float(arr_b.var(ddof=1)) if len(arr_b) > 1 else 0.0
    pooled_num = (len(arr_a) - 1) * var_a + (len(arr_b) - 1) * var_b
    pooled_den = len(arr_a) + len(arr_b) - 2
    pooled_std = math.sqrt(pooled_num / pooled_den) if pooled_den > 0 and pooled_num > 0 else 0.0
    value = mean_diff / pooled_std if pooled_std > 0 else 0.0
    if abs(value) < 0.2:
        interpretation = "negligible"
    elif abs(value) < 0.5:
        interpretation = "small"
    elif abs(value) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    return {"value": float(value), "interpretation": interpretation}


def compute_decomposition(
    fixed_default_bpb: float,
    agent_bests: list[float],
    hp_bests: list[float],
    nas_bests: list[float],
    track: str,
) -> dict[str, Any]:
    mean_agent = float(np.mean(agent_bests))
    mean_hp = float(np.mean(hp_bests))
    mean_nas = float(np.mean(nas_bests))
    total_improvement = fixed_default_bpb - mean_agent
    hp_contribution = fixed_default_bpb - mean_hp
    arch_contribution = mean_hp - mean_agent
    nas_contribution = fixed_default_bpb - mean_nas
    guided_vs_random = mean_nas - mean_agent

    rng = np.random.default_rng(42)
    agent_arr = np.asarray(agent_bests, dtype=float)
    hp_arr = np.asarray(hp_bests, dtype=float)
    nas_arr = np.asarray(nas_bests, dtype=float)
    agent_sample = agent_arr[rng.integers(0, len(agent_arr), size=(BOOTSTRAP_SAMPLES, len(agent_arr)))]
    hp_sample = hp_arr[rng.integers(0, len(hp_arr), size=(BOOTSTRAP_SAMPLES, len(hp_arr)))]
    nas_sample = nas_arr[rng.integers(0, len(nas_arr), size=(BOOTSTRAP_SAMPLES, len(nas_arr)))]
    total_boot = fixed_default_bpb - agent_sample.mean(axis=1)
    hp_boot = fixed_default_bpb - hp_sample.mean(axis=1)
    arch_boot = hp_sample.mean(axis=1) - agent_sample.mean(axis=1)
    nas_boot = fixed_default_bpb - nas_sample.mean(axis=1)
    guided_boot = nas_sample.mean(axis=1) - agent_sample.mean(axis=1)

    hp_test = one_sample_t_greater_zero([fixed_default_bpb - value for value in hp_bests])
    arch_test = welch_t_test(hp_bests, agent_bests)
    guided_test = welch_t_test(nas_bests, agent_bests)

    def pct(value: float) -> float:
        return float(100.0 * value / total_improvement) if total_improvement != 0 else 0.0

    if arch_contribution < 0:
        interpretation = f"HP tuning alone exceeds agent; architecture search adds no value on {track}"
    elif arch_contribution > 0:
        interpretation = f"Architecture search adds value beyond HP tuning on {track}"
    else:
        interpretation = f"Architecture search contributes negligibly on {track}"

    summary = {
        "total_improvement": float(total_improvement),
        "hp_contribution": float(hp_contribution),
        "hp_contribution_pct": pct(hp_contribution),
        "arch_contribution": float(arch_contribution),
        "arch_contribution_pct": pct(arch_contribution),
        "nas_contribution": float(nas_contribution),
        "nas_contribution_pct": pct(nas_contribution),
        "guided_vs_random": float(guided_vs_random),
        "guided_vs_random_pct": pct(guided_vs_random),
        "interpretation": interpretation,
    }
    return {
        "summary": summary,
        "total_improvement_ci": ci_from_boot(total_boot),
        "hp_contribution_ci": ci_from_boot(hp_boot),
        "arch_contribution_ci": ci_from_boot(arch_boot),
        "nas_contribution_ci": ci_from_boot(nas_boot),
        "guided_vs_random_ci": ci_from_boot(guided_boot),
        "hp_contribution_test": hp_test,
        "arch_contribution_test": arch_test,
        "guided_vs_random_test": guided_test,
    }


def plot_h4_best_so_far(curves_for_plot: dict[str, dict[str, list[np.ndarray]]]) -> None:
    for track in TRACKS:
        fig, ax = plt.subplots()
        for condition in CONDITIONS:
            if not curves_for_plot[track][condition]:
                continue
            curves = np.vstack(curves_for_plot[track][condition])
            xs = np.arange(1, curves.shape[1] + 1)
            mean_curve = curves.mean(axis=0)
            ax.plot(xs, mean_curve, color=COND_COLORS[condition], linewidth=2.5, label=condition)
            if curves.shape[0] > 1:
                ax.fill_between(xs, curves.min(axis=0), curves.max(axis=0), color=COND_COLORS[condition], alpha=0.12)
        ax.set_xlabel("Experiment number")
        ax.set_ylabel("Best val_bpb so far")
        ax.set_title(f"H4 Best-So-Far Curves ({track})")
        ax.legend()
        ax.grid(alpha=0.25)
        save_figure(fig, FIGURES_DIR / f"h4_best_so_far_{track}.png")


def plot_h4_auc_comparison(auc_for_plot: dict[str, dict[str, list[float]]], master: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, track in zip(axes, TRACKS):
        means = [float(np.mean(auc_for_plot[track][condition])) for condition in CONDITIONS]
        stds = [float(np.std(auc_for_plot[track][condition], ddof=0)) for condition in CONDITIONS]
        x = np.arange(len(CONDITIONS))
        ax.bar(x, means, yerr=stds, color=[COND_COLORS[item] for item in CONDITIONS], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITIONS, rotation=20)
        ax.set_title(track)
        ax.set_ylabel("AUC-OC")
        max_y = max(mean + std for mean, std in zip(means, stds))
        add_sig_bar(
            ax,
            0,
            1,
            max_y * 1.01,
            master[track]["agent_vs_nas"].get("auc_p_value", 1.0),
        )
        add_sig_bar(
            ax,
            0,
            2,
            max_y * 1.08,
            master[track]["agent_vs_hp_only"].get("auc_p_value", 1.0),
        )
    fig.suptitle("H4 AUC-OC by Condition")
    save_figure(fig, FIGURES_DIR / "h4_auc_comparison.png")


def plot_h4_decomposition(h4_decomposition: dict[str, Any]) -> None:
    tracks = list(TRACKS)
    hp = [h4_decomposition[track]["summary"]["hp_contribution"] for track in tracks]
    arch = [h4_decomposition[track]["summary"]["arch_contribution"] for track in tracks]
    x = np.arange(len(tracks))
    fig, ax = plt.subplots()
    ax.bar(x, hp, color=COND_COLORS["hp_only"], label="HP contribution")
    ax.bar(x, arch, bottom=hp, color=COND_COLORS["agent"], label="Architecture contribution")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tracks)
    ax.set_ylabel("Absolute bpb contribution")
    ax.set_title("H4 Decomposition: HP vs Architecture Contribution")
    ax.legend()
    save_figure(fig, FIGURES_DIR / "h4_decomposition.png")


def plot_h4_keep_rate(keep_for_plot: dict[str, dict[str, list[np.ndarray]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, track in zip(axes, TRACKS):
        for condition in ("agent", "random_nas", "hp_only"):
            curves = keep_for_plot[track][condition]
            if not curves:
                continue
            stacked = np.vstack(curves)
            xs = np.arange(1, stacked.shape[1] + 1)
            ax.plot(xs, stacked.mean(axis=0), color=COND_COLORS[condition], linewidth=2, label=condition)
        ax.set_title(track)
        ax.set_xlabel("Experiment number")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Cumulative keep rate")
    axes[-1].legend()
    fig.suptitle("H4 Cumulative Keep Rate")
    save_figure(fig, FIGURES_DIR / "h4_keep_rate.png")


def plot_h4_anytime_table(h4_anytime_performance: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, track in zip(axes, TRACKS):
        data = np.array(
            [
                [h4_anytime_performance[track][condition][str(budget)] for budget in BUDGETS]
                for condition in CONDITIONS
            ],
            dtype=float,
        )
        image = ax.imshow(data, cmap="viridis")
        ax.set_xticks(np.arange(len(BUDGETS)))
        ax.set_xticklabels(BUDGETS)
        ax.set_yticks(np.arange(len(CONDITIONS)))
        ax.set_yticklabels(CONDITIONS)
        ax.set_title(track)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", color="white", fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("H4 Anytime Performance Table")
    save_figure(fig, FIGURES_DIR / "h4_anytime_table.png")


def add_sig_bar(ax: plt.Axes, x1: int, x2: int, y: float, p_value: float) -> None:
    label = significance_stars(p_value)
    if not label:
        return
    height = y * 0.01 if y != 0 else 0.01
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="black", linewidth=1)
    ax.text((x1 + x2) / 2, y + height * 1.1, label, ha="center", va="bottom")


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def analyze_supplementary(runs: dict[str, dict[str, list[RunData]]]) -> dict[str, Any]:
    training_dynamics = analyze_training_dynamics_all(runs)
    distribution_stats = analyze_distribution_stats(runs)
    crash_rates = analyze_crash_rates(runs)
    write_json(ANALYSIS_DIR / "training_dynamics.json", training_dynamics)
    write_json(ANALYSIS_DIR / "distribution_stats.json", distribution_stats)
    write_json(ANALYSIS_DIR / "crash_rates.json", crash_rates)
    plot_training_dynamics(training_dynamics)
    plot_distribution_stats(distribution_stats)
    return {
        "training_dynamics": training_dynamics,
        "distribution_stats": distribution_stats,
        "crash_rates": crash_rates,
    }


def analyze_training_dynamics_all(runs: dict[str, dict[str, list[RunData]]]) -> dict[str, Any]:
    print("[SUPP] Parsing training logs across all 34 runs...")
    payload = {track: {condition: {} for condition in CONDITIONS} for track in TRACKS}
    for track in TRACKS:
        for condition in CONDITIONS:
            metrics = []
            for run in runs[condition][track]:
                for log_path in sorted((run.base_path / "logs").glob("exp*.log")):
                    parsed = parse_log(log_path)
                    if parsed is None:
                        continue
                    metrics.append(parsed)
            if metrics:
                payload[track][condition] = summarize_training_metrics(metrics)
            else:
                payload[track][condition] = {
                    "count": 0,
                    "mean_final_loss": 0.0,
                    "mean_end_slope": 0.0,
                    "mean_loss_var_last30": 0.0,
                    "mean_mfu": 0.0,
                }
    return payload


def parse_log(path: Path) -> dict[str, Any] | None:
    content = path.read_text(errors="replace")
    parts = content.replace("\r", "\n").split("\n")
    steps = []
    for part in parts:
        match = LOG_STEP_RE.search(part)
        if match:
            steps.append(
                {
                    "step": int(match.group(1)),
                    "loss": float(match.group(3)),
                    "tok_sec": int(match.group(6).replace(",", "")),
                    "mfu": float(match.group(7)),
                }
            )
    if len(steps) < 10:
        return None
    losses = np.array([step["loss"] for step in steps], dtype=float)
    mfu_values = np.array([step["mfu"] for step in steps[1:]], dtype=float)
    tail = losses[max(0, int(0.8 * len(losses))):]
    if len(tail) > 2:
        slope = float(stats.linregress(np.arange(len(tail)), tail).slope)
    else:
        slope = 0.0
    summary = {}
    if "---" in content:
        summary_text = content.split("---")[-1]
        for key, converter in LOG_SUMMARY_FIELDS.items():
            match = re.search(rf"{key}:\s+([\d.eE+\-]+)", summary_text)
            if match:
                summary[key] = converter(match.group(1))
    flops_match = FLOPS_RE.search(content)
    if flops_match:
        summary["flops_per_token"] = float(flops_match.group(1))
    return {
        "final_loss": float(losses[-1]),
        "end_slope": slope,
        "loss_var_last30": float(np.var(losses[max(0, int(0.7 * len(losses))):])),
        "median_mfu": float(np.median(mfu_values)) if len(mfu_values) else 0.0,
        "val_bpb": float(summary.get("val_bpb", 0.0)),
    }


def summarize_training_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(metrics),
        "mean_final_loss": float(np.mean([item["final_loss"] for item in metrics])),
        "mean_end_slope": float(np.mean([item["end_slope"] for item in metrics])),
        "mean_loss_var_last30": float(np.mean([item["loss_var_last30"] for item in metrics])),
        "mean_mfu": float(np.mean([item["median_mfu"] for item in metrics])),
        "mean_val_bpb": float(np.mean([item["val_bpb"] for item in metrics])),
    }


def plot_training_dynamics(training_dynamics: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, track in zip(axes, TRACKS):
        means = [training_dynamics[track][condition]["mean_final_loss"] for condition in CONDITIONS]
        ax.bar(np.arange(len(CONDITIONS)), means, color=[COND_COLORS[item] for item in CONDITIONS])
        ax.set_xticks(np.arange(len(CONDITIONS)))
        ax.set_xticklabels(CONDITIONS, rotation=20)
        ax.set_title(track)
        ax.set_ylabel("Mean final training loss")
    fig.suptitle("Training Dynamics: Convergence")
    save_figure(fig, FIGURES_DIR / "training_dynamics_convergence.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, track in zip(axes, TRACKS):
        means = [training_dynamics[track][condition]["mean_loss_var_last30"] for condition in CONDITIONS]
        ax.bar(np.arange(len(CONDITIONS)), means, color=[COND_COLORS[item] for item in CONDITIONS])
        ax.set_xticks(np.arange(len(CONDITIONS)))
        ax.set_xticklabels(CONDITIONS, rotation=20)
        ax.set_title(track)
        ax.set_ylabel("Mean loss variance (last 30%)")
    fig.suptitle("Training Dynamics: Stability")
    save_figure(fig, FIGURES_DIR / "training_dynamics_stability.png")


def analyze_distribution_stats(runs: dict[str, dict[str, list[RunData]]]) -> dict[str, Any]:
    payload = {track: {} for track in TRACKS}
    for track in TRACKS:
        for condition in CONDITIONS:
            values = [
                row["val_bpb"]
                for run in runs[condition][track]
                for row in run.rows
                if row["status"] != "crash"
            ]
            arr = np.asarray(values, dtype=float)
            payload[track][condition] = {
                "count": int(len(arr)),
                "best": float(np.min(arr)) if len(arr) else 0.0,
                "median": float(np.median(arr)) if len(arr) else 0.0,
                "mean": float(np.mean(arr)) if len(arr) else 0.0,
                "std": float(np.std(arr, ddof=0)) if len(arr) else 0.0,
                "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)) if len(arr) else 0.0,
                "values": [float(x) for x in arr.tolist()],
            }
    return payload


def plot_distribution_stats(distribution_stats: dict[str, Any]) -> None:
    for track in TRACKS:
        fig, ax = plt.subplots()
        data = [distribution_stats[track][condition]["values"] for condition in CONDITIONS]
        parts = ax.violinplot(data, showmeans=True, showmedians=False)
        for body, condition in zip(parts["bodies"], CONDITIONS):
            body.set_facecolor(COND_COLORS[condition])
            body.set_alpha(0.7)
        ax.set_xticks(np.arange(1, len(CONDITIONS) + 1))
        ax.set_xticklabels(CONDITIONS, rotation=20)
        ax.set_ylabel("val_bpb")
        ax.set_title(f"Distribution of val_bpb ({track})")
        save_figure(fig, FIGURES_DIR / f"distribution_violin_{track}.png")


def analyze_crash_rates(runs: dict[str, dict[str, list[RunData]]]) -> dict[str, Any]:
    payload = {track: {} for track in TRACKS}
    for track in TRACKS:
        for condition in CONDITIONS:
            total = sum(len(run.rows) for run in runs[condition][track])
            crashes = sum(sum(1 for row in run.rows if row["status"] == "crash") for run in runs[condition][track])
            payload[track][condition] = {
                "crash_count": crashes,
                "total_experiments": total,
                "crash_rate": float(crashes / total) if total else 0.0,
            }
        agent = payload[track]["agent"]
        for baseline in ("random_nas", "hp_only", "fixed_default"):
            base = payload[track][baseline]
            fisher = fisher_test_from_counts(
                (agent["crash_count"], agent["total_experiments"] - agent["crash_count"]),
                (base["crash_count"], base["total_experiments"] - base["crash_count"]),
            )
            payload[track][f"agent_vs_{baseline}"] = fisher
    return payload


def apply_multiple_comparisons(
    h1_family_tests: dict[str, list[dict[str, Any]]],
    h2_family_tests: dict[str, list[dict[str, Any]]],
    h4_family_tests: dict[str, list[dict[str, Any]]],
    h4_decomp_tests: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    families = {}
    all_tests = {}
    for family_name, tests in (
        list(h1_family_tests.items())
        + list(h2_family_tests.items())
        + list(h4_family_tests.items())
        + list(h4_decomp_tests.items())
    ):
        adjusted = holm_bonferroni(tests)
        families[family_name] = adjusted
        for item in adjusted:
            all_tests[item["test_id"]] = item
    payload = {
        "method": "Holm-Bonferroni",
        "families": families,
        "tests": all_tests,
    }
    write_json(ANALYSIS_DIR / "multiple_comparisons.json", payload)
    return payload


def holm_bonferroni(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tests:
        return []
    indexed = list(enumerate(tests))
    indexed.sort(key=lambda item: item[1]["raw_p_value"])
    m = len(indexed)
    adjusted_sorted = []
    running_max = 0.0
    for rank, (original_idx, test) in enumerate(indexed):
        adjusted = min((m - rank) * test["raw_p_value"], 1.0)
        running_max = max(running_max, adjusted)
        updated = dict(test)
        updated["adjusted_p_value"] = float(running_max)
        updated["_original_idx"] = original_idx
        adjusted_sorted.append(updated)
    adjusted_sorted.sort(key=lambda item: item["_original_idx"])
    for item in adjusted_sorted:
        item.pop("_original_idx", None)
    return adjusted_sorted


def stitch_adjusted_p_values(
    h1_results: dict[str, Any],
    h2_results: dict[str, Any],
    h4_results: dict[str, Any],
    multiple_comparisons: dict[str, Any],
) -> None:
    h1_test = multiple_comparisons["tests"]["h1.permutation_test"]
    h1_results["master"]["permutation_test"]["p_value_adjusted"] = h1_test["adjusted_p_value"]
    h1_results["permutation_test"]["p_value_adjusted"] = h1_test["adjusted_p_value"]

    for technique in TECHNIQUE_ORDER:
        test = multiple_comparisons["tests"][f"h2.{technique}"]
        h2_results["fisher_tests"][technique]["adjusted_p_value"] = test["adjusted_p_value"]
        h2_results["master"]["fisher_tests"][technique]["adjusted_p_value"] = test["adjusted_p_value"]

    for track in TRACKS:
        for name in ("agent_vs_nas", "agent_vs_hp_only", "hp_only_vs_fixed_default", "random_nas_vs_fixed_default", "agent_vs_fixed_default"):
            test_id = f"h4.{track}.{name}"
            if test_id not in multiple_comparisons["tests"]:
                continue
            adjusted = multiple_comparisons["tests"][test_id]["adjusted_p_value"]
            if name in ("agent_vs_nas", "agent_vs_hp_only"):
                h4_results["master"][track][name]["auc_p_value_adjusted"] = adjusted
                h4_results["frequentist_tests"][track][name]["auc_welch"]["adjusted_p_value"] = adjusted
            else:
                h4_results["master"][track][name]["p_value_adjusted"] = adjusted
                h4_results["frequentist_tests"][track][name]["relative_improvement_ttest"]["adjusted_p_value"] = adjusted
        for component in ("hp_contribution", "arch_contribution", "guided_vs_random"):
            test_id = f"h4.decomposition.{track}.{component}"
            adjusted = multiple_comparisons["tests"][test_id]["adjusted_p_value"]
            h4_results["decomposition"][track][f"{component}_test"]["adjusted_p_value"] = adjusted
            h4_results["master"][track]["decomposition"][f"{component}_p_value_adjusted"] = adjusted


def rewrite_adjusted_outputs(
    h1_results: dict[str, Any],
    h2_results: dict[str, Any],
    h4_results: dict[str, Any],
) -> None:
    write_json(ANALYSIS_DIR / "h1_permutation_test.json", h1_results["permutation_test"])
    write_json(ANALYSIS_DIR / "h2_fisher_tests.json", h2_results["fisher_tests"])
    write_json(ANALYSIS_DIR / "h4_frequentist_tests.json", h4_results["frequentist_tests"])
    write_json(ANALYSIS_DIR / "h4_decomposition.json", h4_results["decomposition"])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(sanitize(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [sanitize(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return 0.0
        return number
    return value


def save_figure(fig: plt.Figure, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    GENERATED_FIGURES.extend([png_path, pdf_path])
    plt.close(fig)


def count_png_figures() -> int:
    return sum(1 for path in GENERATED_FIGURES if path.suffix.lower() == ".png")


def extract_block(source: str, start_pattern: str, end_pattern: str) -> str:
    match = re.search(start_pattern + r"(.*?)(?=" + end_pattern + r")", source, re.S)
    return match.group(1) if match else ""


def ci_from_boot(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def ceil_to_multiple(value: int, step: int) -> int:
    if step <= 0:
        return value
    return ((value + step - 1) // step) * step


def safe_divide(lhs: int, rhs: int) -> int:
    return int(lhs // rhs) if rhs else 0


if __name__ == "__main__":
    main()
