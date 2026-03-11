"""
Helpers for bounded Phase 2 experiment sessions.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SUMMARY_PATTERN = re.compile(r"^(val_bpb|peak_vram_mb|num_params_M):\s+([0-9.]+)$", re.MULTILINE)


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).resolve()


SRC_DIR = Path(__file__).resolve().parent
RUN_DIR = _env_path("RECURSIVE_MOL_RUN_DIR", SRC_DIR.parent)
PYTHON_BIN = os.environ.get("RECURSIVE_MOL_PYTHON", sys.executable)
RESULTS_PATH = _env_path("RECURSIVE_MOL_RESULTS_TSV", RUN_DIR / "results.tsv")
LOGS_DIR = _env_path("RECURSIVE_MOL_LOGS_DIR", RUN_DIR / "logs")
DIFFS_DIR = _env_path("RECURSIVE_MOL_DIFFS_DIR", RUN_DIR / "diffs")
VERSIONS_DIR = _env_path("RECURSIVE_MOL_TRAIN_VERSIONS_DIR", RUN_DIR / "train_versions")
STATE_DIR = _env_path("RECURSIVE_MOL_STATE_DIR", RUN_DIR / ".state")
STATE_PATH = STATE_DIR / "session_state.json"
SUMMARY_PATH = RUN_DIR / "summary.json"
TRAIN_PATH = SRC_DIR / "train.py"


@dataclass
class SessionState:
    best_val_bpb: float | None = None
    best_snapshot: str | None = None
    best_experiment: int | None = None

    @classmethod
    def load(cls) -> "SessionState":
        if not STATE_PATH.exists():
            return cls()
        payload = json.loads(STATE_PATH.read_text())
        return cls(**payload)

    def save(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(self.__dict__, indent=2, sort_keys=True))


def ensure_layout(force: bool = False) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    DIFFS_DIR.mkdir(parents=True, exist_ok=True)
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if force and RESULTS_PATH.exists():
        RESULTS_PATH.unlink()
    if not RESULTS_PATH.exists():
        with open(RESULTS_PATH, "w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["commit", "val_bpb", "memory_gb", "status", "description"])


def load_rows() -> list[dict[str, str]]:
    if not RESULTS_PATH.exists():
        return []
    with open(RESULTS_PATH, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def next_experiment_id() -> int:
    return len(load_rows()) + 1


def append_row(commit: str, val_bpb: float, memory_gb: float, status: str, description: str) -> None:
    clean_description = " ".join(description.replace("\t", " ").split())
    with open(RESULTS_PATH, "a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([commit, f"{val_bpb:.6f}", f"{memory_gb:.1f}", status, clean_description])


def write_diff(previous_text: str, current_text: str, diff_path: Path, previous_label: str, current_label: str) -> None:
    diff = difflib.unified_diff(
        previous_text.splitlines(),
        current_text.splitlines(),
        fromfile=previous_label,
        tofile=current_label,
        lineterm="",
    )
    diff_path.write_text("\n".join(diff) + "\n")


def restore_best(state: SessionState) -> None:
    if state.best_snapshot is None:
        return
    best_path = Path(state.best_snapshot)
    if best_path.exists():
        shutil.copy2(best_path, TRAIN_PATH)


def finalize_summary() -> dict:
    rows = load_rows()
    keep_rows = [row for row in rows if row["status"] == "keep" and float(row["val_bpb"]) > 0.0]
    best_row = min(keep_rows, key=lambda row: float(row["val_bpb"])) if keep_rows else None
    summary = {
        "num_experiments": len(rows),
        "num_keep": len([row for row in rows if row["status"] == "keep"]),
        "num_discard": len([row for row in rows if row["status"] == "discard"]),
        "num_crash": len([row for row in rows if row["status"] == "crash"]),
        "best_val_bpb": None if best_row is None else float(best_row["val_bpb"]),
        "best_experiment": None if best_row is None else best_row["commit"],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run_experiment(description: str, timeout_buffer: int = 600) -> int:
    ensure_layout()
    state = SessionState.load()
    exp_id = next_experiment_id()
    exp_tag = f"exp{exp_id:03d}"
    candidate_text = TRAIN_PATH.read_text()
    candidate_snapshot = VERSIONS_DIR / f"{exp_tag}_candidate.py"
    candidate_snapshot.write_text(candidate_text)

    previous_text = candidate_text
    previous_label = "baseline"
    if state.best_snapshot is not None and Path(state.best_snapshot).exists():
        previous_text = Path(state.best_snapshot).read_text()
        previous_label = Path(state.best_snapshot).name

    diff_path = DIFFS_DIR / f"{exp_tag}.diff"
    write_diff(previous_text, candidate_text, diff_path, previous_label, candidate_snapshot.name)

    log_path = LOGS_DIR / f"{exp_tag}.log"
    env = os.environ.copy()
    timeout_seconds = int(env.get("RECURSIVE_MOL_TIME_BUDGET", "300")) + timeout_buffer

    try:
        with open(log_path, "w") as handle:
            subprocess.run(
                [PYTHON_BIN, "train.py"],
                cwd=SRC_DIR,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_seconds,
            )
    except subprocess.TimeoutExpired:
        with open(log_path, "a") as handle:
            handle.write(f"\nTimed out after {timeout_seconds} seconds.\n")

    metrics = {
        match.group(1): float(match.group(2))
        for match in SUMMARY_PATTERN.finditer(log_path.read_text())
    }
    val_bpb = metrics.get("val_bpb", 0.0)
    memory_gb = metrics.get("peak_vram_mb", 0.0) / 1024.0

    if "val_bpb" not in metrics:
        status = "crash"
    elif state.best_val_bpb is None or val_bpb < state.best_val_bpb:
        status = "keep"
    else:
        status = "discard"

    final_snapshot = VERSIONS_DIR / f"{exp_tag}_{status}.py"
    final_snapshot.write_text(candidate_text)
    append_row(exp_tag, val_bpb, memory_gb, status, description)

    if status == "keep":
        state.best_val_bpb = val_bpb
        state.best_snapshot = str(final_snapshot)
        state.best_experiment = exp_id
        state.save()
    else:
        restore_best(state)

    summary = finalize_summary()
    print(json.dumps({"experiment": exp_tag, "status": status, **summary}, indent=2, sort_keys=True))
    return 0 if status != "crash" else 1


def print_status() -> None:
    summary = finalize_summary()
    state = SessionState.load()
    payload = {
        **summary,
        "results_path": str(RESULTS_PATH),
        "best_snapshot": state.best_snapshot,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 session helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create run layout and results header.")
    init_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run one experiment with the current train.py.")
    run_parser.add_argument("--description", required=True)
    run_parser.add_argument("--timeout-buffer", type=int, default=600)

    subparsers.add_parser("status", help="Print run status.")

    args = parser.parse_args()
    if args.command == "init":
        ensure_layout(force=args.force)
        finalize_summary()
        return
    if args.command == "run":
        raise SystemExit(run_experiment(args.description, timeout_buffer=args.timeout_buffer))
    if args.command == "status":
        print_status()
        return


if __name__ == "__main__":
    main()
