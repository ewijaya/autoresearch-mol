"""
Generate random architecture variants and optionally materialize train.py files.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

WIDTH_CHOICES = [value for value in range(128, 513, 32)]
ACTIVATIONS = ["ReLU", "GELU", "SiLU", "ReluSquared"]
ATTENTION_CHOICES = ["full", "windowed"]


def sample_configs(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    variants = []
    seen = set()
    while len(variants) < count:
        depth = rng.randint(3, 8)
        width = rng.choice(WIDTH_CHOICES)
        heads = rng.randint(2, 8)
        if width % heads != 0:
            continue
        signature = (depth, width, heads)
        if signature in seen:
            continue
        seen.add(signature)
        variants.append(
            {
                "id": f"arch_{len(variants):03d}",
                "depth": depth,
                "width": width,
                "heads": heads,
                "activation": rng.choice(ACTIVATIONS),
                "attention": rng.choice(ATTENTION_CHOICES),
            }
        )
    return variants


def _replace(text: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Unable to patch pattern: {pattern}")
    return updated


def render_train_variant(template_text: str, config: dict) -> str:
    updated = template_text
    updated = _replace(
        updated,
        r'^ATTENTION_VARIANT = os\.environ\.get\("RECURSIVE_MOL_ATTENTION", ".*?"\)\.lower\(\)$',
        f'ATTENTION_VARIANT = os.environ.get("RECURSIVE_MOL_ATTENTION", "{config["attention"]}").lower()',
    )
    updated = _replace(
        updated,
        r'^ACTIVATION = os\.environ\.get\("RECURSIVE_MOL_ACTIVATION", ".*?"\)$',
        f'ACTIVATION = os.environ.get("RECURSIVE_MOL_ACTIVATION", "{config["activation"]}")',
    )
    updated = _replace(
        updated,
        r'^DEPTH = env_int\("RECURSIVE_MOL_DEPTH", \d+\)$',
        f'DEPTH = env_int("RECURSIVE_MOL_DEPTH", {config["depth"]})',
    )
    updated = _replace(
        updated,
        r'^MODEL_DIM_OVERRIDE = env_int\("RECURSIVE_MOL_MODEL_DIM", \d+\)$',
        f'MODEL_DIM_OVERRIDE = env_int("RECURSIVE_MOL_MODEL_DIM", {config["width"]})',
    )
    updated = _replace(
        updated,
        r'^NUM_HEADS_OVERRIDE = env_int\("RECURSIVE_MOL_NUM_HEADS", \d+\)$',
        f'NUM_HEADS_OVERRIDE = env_int("RECURSIVE_MOL_NUM_HEADS", {config["heads"]})',
    )
    return updated


def materialize_variants(template_path: Path, output_dir: Path, count: int, seed: int) -> list[dict]:
    template_text = template_path.read_text()
    variants = sample_configs(count=count, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for variant in variants:
        variant_dir = output_dir / variant["id"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        train_path = variant_dir / "train.py"
        train_path.write_text(render_train_variant(template_text, variant))
        (variant_dir / "config.json").write_text(json.dumps(variant, indent=2, sort_keys=True))
        manifest.append(
            {
                **variant,
                "train_path": str(train_path.resolve()),
            }
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random architecture train.py variants.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).resolve().parent / "train.py",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = materialize_variants(
        template_path=args.template,
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
    )
    print(json.dumps({"count": len(manifest), "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
