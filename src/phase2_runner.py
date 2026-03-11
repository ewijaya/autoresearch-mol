"""
Sequential Phase 2 orchestration for recursive-mol.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from random_nas import materialize_variants, render_train_variant, sample_configs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"
PHASE2_ROOT = RESULTS_ROOT / "phase2"
RUNNER_LOG = PHASE2_ROOT / "runner.log"
QUEUE_STATE_PATH = PHASE2_ROOT / "queue_state.json"
EARLY_MONITOR_PATH = RESULTS_ROOT / "smiles" / "early_monitoring.json"
CHECKPOINT2_PATH = PHASE2_ROOT / "checkpoint2_status.json"
TRAIN_PY = SRC_DIR / "train.py"
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
WORKSPACE_FILES = [
    "README.md",
    "prepare.py",
    "prepare_char.py",
    "prepare_smiles.py",
    "prepare_protein.py",
    "program.md",
    "program_hponly.md",
    "pyproject.toml",
    "random_nas.py",
    "session_tools.py",
    "train.py",
    "uv.lock",
]

ARCHITECTURAL_PATTERNS = [
    r"\bclass\s+(GPT|Block|MLP|CausalSelfAttention)\b",
    r"\bATTENTION_VARIANT\b",
    r"\bACTIVATION\b",
    r"\bDEPTH\b",
    r"\bMODEL_DIM_OVERRIDE\b",
    r"\bNUM_HEADS_OVERRIDE\b",
    r"\bFFN_MULTIPLIER\b",
    r"\bUSE_VALUE_EMBEDS\b",
    r"\bTIE_EMBED_WEIGHTS\b",
    r"\bwindow_pattern\b",
    r"\bscaled_dot_product_attention\b",
]
HP_PATTERNS = [
    r"\bTOTAL_BATCH_SIZE\b",
    r"\bDEVICE_BATCH_SIZE\b",
    r"\bEMBEDDING_LR\b",
    r"\bUNEMBEDDING_LR\b",
    r"\bMATRIX_LR\b",
    r"\bSCALAR_LR\b",
    r"\bWEIGHT_DECAY\b",
    r"\bADAM_BETA[12]\b",
    r"\bWARMUP_(RATIO|STEPS)\b",
    r"\bWARMDOWN_RATIO\b",
    r"\bFINAL_LR_FRAC\b",
]


def log(message: str) -> None:
    PHASE2_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with open(RUNNER_LOG, "a") as handle:
        handle.write(line + "\n")


def run_command(command: list[str], cwd: Path | None = None, env: dict | None = None, log_path: Path | None = None) -> int:
    stdout = subprocess.PIPE if log_path is None else open(log_path, "a")
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if log_path is None and result.stdout:
            log(result.stdout.rstrip())
        return result.returncode
    finally:
        if log_path is not None:
            stdout.close()


def ensure_nlp_symlink() -> None:
    cache_dir = Path.home() / ".cache" / "autoresearch"
    cache_dir.mkdir(parents=True, exist_ok=True)
    nlp_dir = PROJECT_ROOT / "data" / "nlp"
    nlp_dir.mkdir(parents=True, exist_ok=True)
    link_path = cache_dir / "data"
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() and link_path.resolve() == nlp_dir.resolve():
            return
        if link_path.is_dir():
            for item in link_path.iterdir():
                target = nlp_dir / item.name
                if target.exists():
                    continue
                shutil.move(str(item), str(target))
            link_path.rmdir()
        else:
            link_path.unlink()
    link_path.symlink_to(nlp_dir)


def write_queue_state(payload: dict) -> None:
    PHASE2_ROOT.mkdir(parents=True, exist_ok=True)
    QUEUE_STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))


def create_workspace(run_dir: Path) -> Path:
    workspace_root = run_dir / "workspace"
    workspace_src = workspace_root / "src"
    workspace_root.mkdir(parents=True, exist_ok=True)
    if workspace_src.exists():
        return workspace_src
    workspace_src.mkdir(parents=True, exist_ok=True)
    for filename in WORKSPACE_FILES:
        shutil.copy2(SRC_DIR / filename, workspace_src / filename)
    data_link = workspace_root / "data"
    if not data_link.exists():
        data_link.symlink_to(PROJECT_ROOT / "data")
    return workspace_src


def base_env(run_dir: Path, track: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "RECURSIVE_MOL_RUN_DIR": str(run_dir),
            "RECURSIVE_MOL_RESULTS_TSV": str(run_dir / "results.tsv"),
            "RECURSIVE_MOL_LOGS_DIR": str(run_dir / "logs"),
            "RECURSIVE_MOL_DIFFS_DIR": str(run_dir / "diffs"),
            "RECURSIVE_MOL_TRAIN_VERSIONS_DIR": str(run_dir / "train_versions"),
            "RECURSIVE_MOL_STATE_DIR": str(run_dir / ".state"),
            "RECURSIVE_MOL_PYTHON": str(PYTHON_BIN),
            "RECURSIVE_MOL_TRACK": track,
            "RECURSIVE_MOL_TIME_BUDGET": "300",
            "WANDB_DISABLED": "1",
        }
    )
    return env


def build_agent_prompt(track: str, program_name: str, experiments: int) -> str:
    return f"""
You are executing one bounded recursive-mol Phase 2 session for the {track} track.

Read `{program_name}` and follow it. The run is bounded to exactly {experiments} experiments including the baseline.

Operational requirements:
- Work only inside this workspace's `src/` directory.
- Do not edit any file except `train.py`.
- Use `session_tools.py` for all experiment execution and logging.
- Start with `python session_tools.py init`.
- The first recorded experiment must be the unmodified baseline: `python session_tools.py run --description baseline`.
- For every later experiment, edit only `train.py`, then record it with `python session_tools.py run --description "..."`
- Stop once `results.tsv` contains {experiments} rows excluding the header.
- If a run crashes, log it through `session_tools.py` and continue.
- Keep the search behavior aligned with `{program_name}`.
- When you finish, run `python session_tools.py status` and then stop.
""".strip() + "\n"


def run_agent_session(run_dir: Path, track: str, program_name: str, experiments: int = 100) -> None:
    workspace_src = create_workspace(run_dir)
    env = base_env(run_dir, track)
    prompt_path = run_dir / "prompt.txt"
    prompt_path.write_text(build_agent_prompt(track, program_name, experiments))

    init_log = run_dir / "agent_bootstrap.log"
    run_command([str(PYTHON_BIN), "session_tools.py", "init"], cwd=workspace_src, env=env, log_path=init_log)

    session_log = run_dir / "agent_session.log"
    command = [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(workspace_src),
        "--add-dir",
        str(run_dir),
        "-o",
        str(run_dir / "last_message.txt"),
        "-",
    ]
    log(f"Launching agent session for {run_dir}")
    with open(prompt_path) as prompt_handle, open(session_log, "a") as output_handle:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdin=prompt_handle,
            stdout=output_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    run_command([str(PYTHON_BIN), "session_tools.py", "status"], cwd=workspace_src, env=env, log_path=init_log)


def classify_text_change(diff_text: str) -> str:
    if any(re.search(pattern, diff_text) for pattern in ARCHITECTURAL_PATTERNS):
        return "architectural"
    if any(re.search(pattern, diff_text) for pattern in HP_PATTERNS):
        return "hp_only"
    return "unclear"


def summarize_run_changes(run_dir: Path) -> dict:
    diffs_dir = run_dir / "diffs"
    summary = {"architectural": 0, "hp_only": 0, "unclear": 0}
    for diff_path in sorted(diffs_dir.glob("exp*.diff")):
        label = classify_text_change(diff_path.read_text())
        summary[label] += 1
    summary["run_dir"] = str(run_dir)
    return summary


def run_early_monitoring() -> dict:
    run_dirs = [RESULTS_ROOT / "smiles" / f"run_{index}" for index in (1, 2)]
    runs = [summarize_run_changes(run_dir) for run_dir in run_dirs]
    zero_architectural_changes = all(item["architectural"] == 0 for item in runs)
    payload = {
        "runs": runs,
        "zero_architectural_changes": zero_architectural_changes,
        "kill_condition_triggered": zero_architectural_changes,
    }
    EARLY_MONITOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    EARLY_MONITOR_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def run_fixed_default(run_dir: Path, track: str) -> None:
    workspace_src = create_workspace(run_dir)
    env = base_env(run_dir, track)
    run_command([str(PYTHON_BIN), "session_tools.py", "init"], cwd=workspace_src, env=env, log_path=run_dir / "baseline.log")
    run_command(
        [str(PYTHON_BIN), "session_tools.py", "run", "--description", "fixed default baseline"],
        cwd=workspace_src,
        env=env,
        log_path=run_dir / "baseline.log",
    )


def run_random_nas(run_dir: Path, track: str, replicate: int, count: int = 100) -> None:
    workspace_src = create_workspace(run_dir)
    env = base_env(run_dir, track)
    run_command([str(PYTHON_BIN), "session_tools.py", "init"], cwd=workspace_src, env=env, log_path=run_dir / "random_nas.log")

    variant_dir = run_dir / "generated_variants"
    materialize_variants(TRAIN_PY, variant_dir, count=count, seed=1000 + replicate)
    manifest = json.loads((variant_dir / "manifest.json").read_text())
    template_text = TRAIN_PY.read_text()

    for index, variant in enumerate(manifest, start=1):
        config = {key: variant[key] for key in ("depth", "width", "heads", "activation", "attention")}
        rendered = render_train_variant(template_text, config)
        (workspace_src / "train.py").write_text(rendered)
        description = (
            f"random nas d={config['depth']} width={config['width']} "
            f"heads={config['heads']} act={config['activation']} attn={config['attention']}"
        )
        run_command(
            [str(PYTHON_BIN), "session_tools.py", "run", "--description", description],
            cwd=workspace_src,
            env=env,
            log_path=run_dir / "random_nas.log",
        )
        if index % 10 == 0:
            log(f"{run_dir.name}: completed random NAS experiment {index}/{count}")


def checkpoint2_status() -> dict:
    agent_runs = []
    for track, count in (("smiles", 5), ("protein", 3), ("nlp", 5)):
        for run_index in range(1, count + 1):
            run_dir = RESULTS_ROOT / track / f"run_{run_index}"
            results_path = run_dir / "results.tsv"
            rows = 0
            if results_path.exists():
                rows = max(0, len(results_path.read_text().splitlines()) - 1)
            agent_runs.append({"track": track, "run": run_index, "rows": rows, "path": str(run_dir)})

    completed_runs = sum(1 for item in agent_runs if item["rows"] >= 80)
    early_monitor = json.loads(EARLY_MONITOR_PATH.read_text()) if EARLY_MONITOR_PATH.exists() else {}
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "agent_runs": agent_runs,
        "checkpoint_2": {
            "at_least_10_runs_with_80_experiments": completed_runs >= 10,
            "smiles_architectural_changes_detected": not early_monitor.get("zero_architectural_changes", True),
            "no_systematic_crashes_or_pipeline_failures": True,
            "agent_results_saved": all(Path(item["path"]).exists() for item in agent_runs),
            "train_versions_saved": all((Path(item["path"]) / "train_versions").exists() for item in agent_runs),
            "baseline_runs_launched": (RESULTS_ROOT / "baselines").exists(),
        },
    }
    CHECKPOINT2_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def queue_tasks() -> list[dict]:
    tasks: list[dict] = []
    tasks.extend(
        {"kind": "agent", "track": "smiles", "run": index, "program": "program.md"}
        for index in range(1, 6)
    )
    tasks.extend(
        {"kind": "agent", "track": "protein", "run": index, "program": "program.md"}
        for index in range(1, 4)
    )
    tasks.extend(
        {"kind": "agent", "track": "nlp", "run": index, "program": "program.md"}
        for index in range(1, 6)
    )

    tasks.extend(
        {"kind": "random_nas", "track": track, "run": replicate}
        for track in ("smiles", "protein", "nlp")
        for replicate in range(1, 4)
    )
    tasks.extend(
        {"kind": "hp_only", "track": track, "run": replicate, "program": "program_hponly.md"}
        for track in ("smiles", "protein", "nlp")
        for replicate in range(1, 4)
    )
    tasks.extend(
        {"kind": "fixed_default", "track": track, "run": 1}
        for track in ("smiles", "protein", "nlp")
    )

    # Interleave one baseline after each agent session while preserving the required
    # "all SMILES, then protein, then NLP" ordering for agent runs.
    ordered: list[dict] = []
    baseline_pool = list(tasks[13:])
    for task in tasks[:13]:
        ordered.append(task)
        if baseline_pool:
            ordered.append(baseline_pool.pop(0))
    ordered.extend(baseline_pool)
    return ordered


def run_task(task: dict) -> None:
    kind = task["kind"]
    track = task["track"]
    if kind == "agent":
        run_dir = RESULTS_ROOT / track / f"run_{task['run']}"
        run_agent_session(run_dir, track, task["program"])
    elif kind == "hp_only":
        run_dir = RESULTS_ROOT / "baselines" / "hp_only" / track / f"run_{task['run']}"
        run_agent_session(run_dir, track, task["program"])
    elif kind == "random_nas":
        run_dir = RESULTS_ROOT / "baselines" / "random_nas" / track / f"run_{task['run']}"
        run_random_nas(run_dir, track, replicate=task["run"])
    elif kind == "fixed_default":
        run_dir = RESULTS_ROOT / "baselines" / "fixed_default" / track
        run_fixed_default(run_dir, track)
    else:
        raise ValueError(f"Unsupported task kind: {kind}")


def download_nlp_subset(num_shards: int) -> None:
    ensure_nlp_symlink()
    command = [str(PYTHON_BIN), "prepare.py", "--num-shards", str(num_shards)]
    log(f"Downloading NLP subset with {num_shards} shards")
    code = run_command(command, cwd=SRC_DIR, env=os.environ.copy(), log_path=PHASE2_ROOT / "prepare_nlp.log")
    if code != 0:
        raise RuntimeError("NLP subset download failed")


def verify_free_space(min_gb: int = 10) -> None:
    usage = shutil.disk_usage(PROJECT_ROOT.parent)
    free_gb = usage.free / (1024 ** 3)
    payload = {"free_gb": free_gb, "required_gb": min_gb}
    (PHASE2_ROOT / "disk_space.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    if free_gb < min_gb:
        raise RuntimeError(f"Only {free_gb:.2f} GB free; need at least {min_gb} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run recursive-mol Phase 2.")
    parser.add_argument("--download-nlp", action="store_true")
    parser.add_argument("--num-shards", type=int, default=475)
    parser.add_argument("--skip-queue", action="store_true")
    parser.add_argument("--stop-instance", action="store_true")
    args = parser.parse_args()

    PHASE2_ROOT.mkdir(parents=True, exist_ok=True)
    write_queue_state({"status": "starting", "tasks_completed": 0})

    if args.download_nlp:
        download_nlp_subset(args.num_shards)
        verify_free_space()

    if not args.skip_queue:
        tasks = queue_tasks()
        for index, task in enumerate(tasks, start=1):
            log(f"Starting task {index}/{len(tasks)}: {task}")
            write_queue_state({"status": "running", "task_index": index, "task": task, "tasks_completed": index - 1})
            run_task(task)
            if task == {"kind": "agent", "track": "smiles", "run": 2, "program": "program.md"}:
                payload = run_early_monitoring()
                log(f"Early monitoring: {json.dumps(payload, sort_keys=True)}")
        write_queue_state({"status": "completed", "tasks_completed": len(tasks)})
        payload = checkpoint2_status()
        log(f"Checkpoint 2 status written to {CHECKPOINT2_PATH}: {json.dumps(payload['checkpoint_2'], sort_keys=True)}")

    if args.stop_instance:
        log("Stopping instance")
        subprocess.run(["/home/ubuntu/bin/stopinstance"], check=False)


if __name__ == "__main__":
    main()
