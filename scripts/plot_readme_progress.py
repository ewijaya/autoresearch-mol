"""Generate a 3-panel best-so-far progress figure for README.md.

Style follows Karpathy's autoresearch progress.png:
- Title: track name with experiment count
- Y-axis: "Validation BPB (lower is better)"
- X-axis: "Experiment #"
- Stepped running-best lines per condition
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "results" / "analysis" / "h4_auc_values.json"
FIGURES_DIR = PROJECT_ROOT / "manuscript" / "figures"
OUTPUT_PNG = FIGURES_DIR / "readme_progress.png"
OUTPUT_PDF = FIGURES_DIR / "readme_progress.pdf"

COND_COLORS = {
    "agent": "#2196F3",
    "random_nas": "#FF9800",
    "hp_only": "#4CAF50",
    "fixed_default": "#9E9E9E",
}
COND_LABELS = {
    "agent": "Agent (arch + HP)",
    "random_nas": "Random NAS",
    "hp_only": "HP-only agent",
    "fixed_default": "Fixed default",
}
TRACK_TITLES = {
    "smiles": "SMILES (ZINC-250K)",
    "protein": "Protein (UniRef50)",
    "nlp": "NLP (FineWeb-Edu)",
}
CONDITIONS = ["agent", "random_nas", "hp_only", "fixed_default"]
TRACKS = ["smiles", "protein", "nlp"]


def main():
    data = json.loads(DATA_PATH.read_text())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, track in zip(axes, TRACKS):
        track_data = data[track]
        n_experiments = 0

        for condition in CONDITIONS:
            runs = track_data.get(condition, [])
            if not runs:
                continue

            curves = np.array([r["best_so_far_curve"] for r in runs])
            n_experiments += sum(len(r["best_so_far_curve"]) for r in runs)
            xs = np.arange(1, curves.shape[1] + 1)
            mean_curve = curves.mean(axis=0)

            ax.plot(
                xs,
                mean_curve,
                color=COND_COLORS[condition],
                linewidth=2.0,
                label=COND_LABELS[condition],
                drawstyle="steps-post",
            )
            if curves.shape[0] > 1:
                ax.fill_between(
                    xs,
                    curves.min(axis=0),
                    curves.max(axis=0),
                    color=COND_COLORS[condition],
                    alpha=0.10,
                    step="post",
                )

        ax.set_xlabel("Experiment #", fontsize=12)
        ax.set_ylabel("Validation BPB (lower is better)", fontsize=12)
        ax.set_title(f"{TRACK_TITLES[track]}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=10)

    fig.suptitle(
        "autoresearch-mol: 3,106 Experiments Across 4 Conditions",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
