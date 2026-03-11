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
AGENT_STEP_TIMEOUT_SECONDS = 20 * 60
AGENT_STEP_MAX_RETRIES = 3
FIVE_HOUR_RESET_SECONDS = 5 * 60 * 60
WEEKLY_RESET_SECONDS = 7 * 24 * 60 * 60

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


class RateLimitPause(RuntimeError):
    def __init__(self, message: str, scope: str, retry_after_seconds: int, task: dict | None = None) -> None:
        super().__init__(message)
        self.scope = scope
        self.retry_after_seconds = retry_after_seconds
        self.task = task


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


def results_row_count(run_dir: Path) -> int:
    results_path = run_dir / "results.tsv"
    if not results_path.exists():
        return 0
    return max(0, len(results_path.read_text().splitlines()) - 1)


def task_completed(task: dict) -> bool:
    return results_row_count(run_dir_for_task(task)) >= expected_rows_for_task(task)


def is_codex_task(task: dict) -> bool:
    return task["kind"] in {"agent", "hp_only"}


def detect_rate_limit_pause(text: str, returncode: int) -> RateLimitPause | None:
    lower = text.lower()
    is_weekly = bool(re.search(r"weekly limit:.*0% left", lower))
    is_five_hour = bool(re.search(r"5h limit:.*0% left", lower))
    generic_limit = any(
        phrase in lower
        for phrase in (
            "rate limit",
            "usage limit",
            "limit reached",
            "too many requests",
            "credits exhausted",
            "credits depleted",
            "buy additional credits",
        )
    )
    if not (is_weekly or is_five_hour or generic_limit):
        return None
    if returncode == 0 and not (is_weekly or is_five_hour):
        return None

    if is_weekly or "weekly" in lower:
        return RateLimitPause(
            "Codex weekly usage limit reached",
            scope="weekly",
            retry_after_seconds=WEEKLY_RESET_SECONDS,
        )
    return RateLimitPause(
        "Codex 5-hour usage limit reached",
        scope="5h",
        retry_after_seconds=FIVE_HOUR_RESET_SECONDS,
    )


def iso_timestamp_from_now(seconds: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + seconds))


def maybe_run_early_monitoring(task: dict) -> None:
    if task == {"kind": "agent", "track": "smiles", "run": 2, "program": "program.md"}:
        payload = run_early_monitoring()
        log(f"Early monitoring: {json.dumps(payload, sort_keys=True)}")


def count_completed_tasks(tasks: list[dict]) -> int:
    return sum(1 for task in tasks if task_completed(task))


def expected_rows_for_task(task: dict) -> int:
    if task["kind"] in {"agent", "hp_only", "random_nas"}:
        return 100
    if task["kind"] == "fixed_default":
        return 1
    raise ValueError(f"Unsupported task kind: {task['kind']}")


def run_dir_for_task(task: dict) -> Path:
    kind = task["kind"]
    track = task["track"]
    if kind == "agent":
        return RESULTS_ROOT / track / f"run_{task['run']}"
    if kind == "hp_only":
        return RESULTS_ROOT / "baselines" / "hp_only" / track / f"run_{task['run']}"
    if kind == "random_nas":
        return RESULTS_ROOT / "baselines" / "random_nas" / track / f"run_{task['run']}"
    if kind == "fixed_default":
        return RESULTS_ROOT / "baselines" / "fixed_default" / track
    raise ValueError(f"Unsupported task kind: {kind}")


def verify_task_completion(task: dict) -> None:
    run_dir = run_dir_for_task(task)
    required_rows = expected_rows_for_task(task)
    actual_rows = results_row_count(run_dir)
    if actual_rows < required_rows:
        raise RuntimeError(
            f"Task incomplete for {run_dir}: expected {required_rows} rows, found {actual_rows}"
        )


def build_agent_prompt(track: str, program_name: str, experiments: int, existing_rows: int) -> str:
    next_experiment = existing_rows + 1
    if existing_rows >= experiments:
        return f"""
You are resuming a completed recursive-mol Phase 2 session for the {track} track.

The run already has {existing_rows} recorded experiments, which meets the target of {experiments}.
Do not make any further edits. Run `python session_tools.py status` and stop.
""".strip() + "\n"

    return f"""
You are executing experiment {next_experiment} of up to {experiments} for the recursive-mol Phase 2 {track} track.

Read `{program_name}` and follow it. This Codex session is responsible for exactly one additional experiment row.

Operational requirements:
- Work only inside this workspace's `src/` directory.
- Do not edit any file except `train.py`.
- Use `session_tools.py` for all experiment execution and logging.
- Start with `python session_tools.py init`.
- Run `python session_tools.py status` before deciding the next step.
- The shared run state lives in `../../results.tsv`, `../../summary.json`, `../../logs/`, and `../../train_versions/`.
- Existing completed experiments: {existing_rows}.
- Your goal is to leave the run with exactly one additional row in `../../results.tsv`.
- If the run has zero completed experiments, the next row must be the untouched baseline: `python session_tools.py run --description baseline`.
- Otherwise, inspect the current `train.py` and recent `../../results.tsv` rows, make one coherent change, then record it with `python session_tools.py run --description "..."`
- Stop after that single additional row is recorded.
- If a run crashes, log it through `session_tools.py` and continue.
- Keep the search behavior aligned with `{program_name}`.
- `session_tools.py run` is the source of truth. Treat it as a blocking command and wait for it to finish.
- Do not read full training logs unless a run crashes. For successful runs, use `python session_tools.py status` and `../../results.tsv`. If you need a crash log, read only the tail of the newest file in `../../logs/`.
- Do not spend time on long post-hoc analysis. Make one reasonable next-step decision and execute it.
- Avoid commands that print huge files. If you need log context, read only the tail.
- When the new row is present, run `python session_tools.py status` and stop immediately.
""".strip() + "\n"


def run_agent_session(run_dir: Path, track: str, program_name: str, experiments: int = 100) -> None:
    workspace_src = create_workspace(run_dir)
    env = base_env(run_dir, track)
    init_log = run_dir / "agent_bootstrap.log"
    run_command([str(PYTHON_BIN), "session_tools.py", "init"], cwd=workspace_src, env=env, log_path=init_log)
    session_log = run_dir / "agent_session.log"
    retries_without_progress = 0

    while True:
        existing_rows = results_row_count(run_dir)
        if existing_rows >= experiments:
            run_command([str(PYTHON_BIN), "session_tools.py", "status"], cwd=workspace_src, env=env, log_path=init_log)
            return

        prompt_path = run_dir / "prompt.txt"
        prompt_path.write_text(build_agent_prompt(track, program_name, experiments, existing_rows))
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
        log(f"Launching agent step for {run_dir} experiment {existing_rows + 1}/{experiments}")
        session_offset = session_log.stat().st_size if session_log.exists() else 0
        returncode = 0
        with open(prompt_path) as prompt_handle, open(session_log, "a") as output_handle:
            try:
                result = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdin=prompt_handle,
                    stdout=output_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=AGENT_STEP_TIMEOUT_SECONDS,
                )
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                output_handle.write(
                    f"\n[phase2_runner] agent step timed out after {AGENT_STEP_TIMEOUT_SECONDS} seconds\n"
                )
                returncode = 124

        appended_text = ""
        if session_log.exists():
            with open(session_log, "r") as handle:
                handle.seek(session_offset)
                appended_text = handle.read()

        run_command([str(PYTHON_BIN), "session_tools.py", "status"], cwd=workspace_src, env=env, log_path=init_log)
        updated_rows = results_row_count(run_dir)
        if updated_rows > existing_rows:
            retries_without_progress = 0
            log(f"{run_dir}: recorded experiment row {updated_rows}/{experiments}")
            continue

        pause = detect_rate_limit_pause(appended_text, returncode)
        if pause is not None:
            raise pause

        retries_without_progress += 1
        log(f"{run_dir}: no new row recorded after agent step (retry {retries_without_progress}/{AGENT_STEP_MAX_RETRIES})")
        if retries_without_progress >= AGENT_STEP_MAX_RETRIES:
            raise RuntimeError(f"Agent run stalled for {run_dir} after {AGENT_STEP_MAX_RETRIES} retries")


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
        run_dir = run_dir_for_task(task)
        run_agent_session(run_dir, track, task["program"])
    elif kind == "hp_only":
        run_dir = run_dir_for_task(task)
        run_agent_session(run_dir, track, task["program"])
    elif kind == "random_nas":
        run_dir = run_dir_for_task(task)
        run_random_nas(run_dir, track, replicate=task["run"])
    elif kind == "fixed_default":
        run_dir = run_dir_for_task(task)
        run_fixed_default(run_dir, track)
    else:
        raise ValueError(f"Unsupported task kind: {kind}")
    verify_task_completion(task)


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

    rate_limit_pause: RateLimitPause | None = None
    try:
        if not args.skip_queue:
            tasks = queue_tasks()
            skipped_codex_tasks: list[tuple[int, dict]] = []
            for index, task in enumerate(tasks, start=1):
                if task_completed(task):
                    maybe_run_early_monitoring(task)
                    continue

                if rate_limit_pause is not None and is_codex_task(task):
                    skipped_codex_tasks.append((index, task))
                    continue

                log(f"Starting task {index}/{len(tasks)}: {task}")
                write_queue_state(
                    {
                        "status": "running",
                        "task_index": index,
                        "task": task,
                        "tasks_completed": count_completed_tasks(tasks),
                    }
                )
                try:
                    run_task(task)
                    maybe_run_early_monitoring(task)
                except RateLimitPause as exc:
                    exc.task = task
                    rate_limit_pause = exc
                    skipped_codex_tasks.append((index, task))
                    next_retry_at = iso_timestamp_from_now(exc.retry_after_seconds)
                    write_queue_state(
                        {
                            "status": "paused_rate_limit",
                            "task_index": index,
                            "task": task,
                            "tasks_completed": count_completed_tasks(tasks),
                            "rate_limit_scope": exc.scope,
                            "next_retry_at": next_retry_at,
                            "pending_codex_tasks": len(
                                skipped_codex_tasks
                                + [
                                    (later_index, later_task)
                                    for later_index, later_task in enumerate(tasks[index:], start=index + 1)
                                    if is_codex_task(later_task) and not task_completed(later_task)
                                ]
                            ),
                        }
                    )
                    log(
                        f"Codex rate limit reached on task {index}/{len(tasks)}; "
                        f"continuing script-only tasks and pausing agent tasks until at least {next_retry_at}"
                    )

            if rate_limit_pause is not None:
                pending_codex = [
                    {"task_index": index, "task": task}
                    for index, task in skipped_codex_tasks
                    if not task_completed(task)
                ]
                next_retry_at = iso_timestamp_from_now(rate_limit_pause.retry_after_seconds)
                write_queue_state(
                    {
                        "status": "paused_rate_limit",
                        "tasks_completed": count_completed_tasks(tasks),
                        "rate_limit_scope": rate_limit_pause.scope,
                        "next_retry_at": next_retry_at,
                        "pending_codex_tasks": pending_codex,
                    }
                )
                log(
                    f"Queue paused on Codex {rate_limit_pause.scope} limit with "
                    f"{len(pending_codex)} Codex tasks remaining; resume after {next_retry_at}"
                )
                return

            write_queue_state({"status": "completed", "tasks_completed": len(tasks)})
            payload = checkpoint2_status()
            log(f"Checkpoint 2 status written to {CHECKPOINT2_PATH}: {json.dumps(payload['checkpoint_2'], sort_keys=True)}")
    except Exception as exc:
        write_queue_state(
            {
                "status": "failed",
                "error": str(exc),
            }
        )
        log(f"Phase 2 runner failed: {exc}")
        raise

    if args.stop_instance:
        log("Stopping instance")
        subprocess.run(["/home/ubuntu/bin/stopinstance"], check=False)


if __name__ == "__main__":
    main()
