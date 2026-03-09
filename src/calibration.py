"""
Calibration study for the 5-minute proxy on the SMILES track.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "calibration"
LOGS_DIR = RESULTS_DIR / "logs"
VARIANTS_PATH = RESULTS_DIR / "variants.json"
RESULTS_PATH = RESULTS_DIR / "results.json"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
DECISION_PATH = RESULTS_DIR / "decision.md"

WIDTH_CHOICES = [value for value in range(128, 513, 32)]
ACTIVATIONS = ["ReLU", "GELU", "SiLU", "ReluSquared"]
ATTENTION_CHOICES = ["full", "windowed"]

SUMMARY_PATTERN = re.compile(r"^(val_bpb|peak_vram_mb|num_params_M):\s+([0-9.]+)$", re.MULTILINE)


def sample_variants(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    variants = []
    while len(variants) < count:
        depth = rng.randint(3, 8)
        width = rng.choice(WIDTH_CHOICES)
        heads = rng.randint(2, 8)
        if width % heads != 0:
            continue
        head_dim = width // heads
        if not (16 <= head_dim <= 128):
            continue
        variants.append(
            {
                "id": f"variant_{len(variants):02d}",
                "depth": depth,
                "model_dim": width,
                "num_heads": heads,
                "head_dim": head_dim,
                "activation": rng.choice(ACTIVATIONS),
                "attention": rng.choice(ATTENTION_CHOICES),
            }
        )
    return variants


def run_variant(variant: dict, budget_seconds: int, force: bool = False) -> dict:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{variant['id']}_{budget_seconds}.log"
    if log_path.exists() and not force:
        return parse_log(log_path, variant, budget_seconds)

    env = os.environ.copy()
    env.update(
        {
            "RECURSIVE_MOL_TRACK": "smiles",
            "RECURSIVE_MOL_TIME_BUDGET": str(budget_seconds),
            "RECURSIVE_MOL_DEPTH": str(variant["depth"]),
            "RECURSIVE_MOL_MODEL_DIM": str(variant["model_dim"]),
            "RECURSIVE_MOL_NUM_HEADS": str(variant["num_heads"]),
            "RECURSIVE_MOL_HEAD_DIM": str(variant["head_dim"]),
            "RECURSIVE_MOL_ACTIVATION": variant["activation"],
            "RECURSIVE_MOL_ATTENTION": variant["attention"],
            "RECURSIVE_MOL_ENABLE_COMPILE": env.get("RECURSIVE_MOL_ENABLE_COMPILE", "1"),
        }
    )
    command = [sys.executable, "train.py"]
    with open(log_path, "w") as handle:
        subprocess.run(command, cwd=SRC_DIR, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return parse_log(log_path, variant, budget_seconds)


def parse_log(log_path: Path, variant: dict, budget_seconds: int) -> dict:
    text = log_path.read_text()
    metrics = {match.group(1): float(match.group(2)) for match in SUMMARY_PATTERN.finditer(text)}
    status = "ok" if "val_bpb" in metrics else "crash"
    return {
        "variant_id": variant["id"],
        "budget_seconds": budget_seconds,
        "status": status,
        "log_path": str(log_path),
        "config": variant,
        "val_bpb": metrics.get("val_bpb"),
        "peak_vram_mb": metrics.get("peak_vram_mb"),
        "num_params_M": metrics.get("num_params_M"),
    }


def decide(rho: float) -> str:
    if math.isnan(rho):
        return "Calibration failed: insufficient successful runs."
    if rho > 0.7:
        return "rho > 0.7: proceed."
    if rho >= 0.4:
        return "rho in [0.4, 0.7]: proceed with caution."
    return "rho < 0.4: increase TIME_BUDGET to 15-30 min."


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibration variants for the SMILES track.")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budgets", nargs="+", type=int, default=[300, 7200])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    variants = sample_variants(count=args.count, seed=args.seed)
    VARIANTS_PATH.write_text(json.dumps(variants, indent=2))

    results = []
    for budget in args.budgets:
        for variant in variants:
            results.append(run_variant(variant, budget, force=args.force))

    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    if len(args.budgets) < 2:
        rho, p_value, common_ids = float("nan"), float("nan"), []
        short_scores = {}
        long_scores = {}
    else:
        short_budget, long_budget = args.budgets[:2]
        short_scores = {
            item["variant_id"]: item["val_bpb"]
            for item in results
            if item["budget_seconds"] == short_budget and item["status"] == "ok"
        }
        long_scores = {
            item["variant_id"]: item["val_bpb"]
            for item in results
            if item["budget_seconds"] == long_budget and item["status"] == "ok"
        }
        common_ids = sorted(short_scores.keys() & long_scores.keys())
        short_vals = [short_scores[variant_id] for variant_id in common_ids]
        long_vals = [long_scores[variant_id] for variant_id in common_ids]
        rho, p_value = spearmanr(short_vals, long_vals) if common_ids else (float("nan"), float("nan"))

    summary = {
        "num_variants": len(variants),
        "budgets": args.budgets,
        "successful_pairs": len(common_ids),
        "spearman_rho": rho,
        "spearman_p_value": p_value,
        "decision": decide(rho),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    DECISION_PATH.write_text(summary["decision"] + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
