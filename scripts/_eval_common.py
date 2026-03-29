from __future__ import annotations

import datetime as dt
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

TRACKS = ("smiles", "protein", "nlp")
TRACK_MAX_SEQ_LEN = {"smiles": 256, "protein": 512, "nlp": 2048}
TRACK_DEVICE_BATCH_CANDIDATES = {
    "smiles": [256, 128, 64, 32],
    "protein": [128, 64, 32, 16],
    "nlp": [32, 16, 8, 4],
}
REPLICATE_SEEDS = (42, 137, 2026)
DEFAULT_TIME_BUDGET = 300
DEFAULT_TOTAL_BATCH_SIZE = 65536
VAL_BPB_PATTERN = re.compile(r"val_bpb:\s*([0-9]+\.[0-9]+)")


@dataclass(frozen=True)
class ArchitectureInfo:
    track: str
    run: str
    experiment: str
    source_path: Path
    native_bpb: float

    @property
    def name(self) -> str:
        return f"agent_{self.track}_{self.run}_{self.experiment}"

    @property
    def native_seq_len(self) -> int:
        return TRACK_MAX_SEQ_LEN[self.track]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "track": self.track,
            "run": self.run,
            "experiment": self.experiment,
            "source": str(self.source_path),
            "native_bpb": self.native_bpb,
            "native_seq_len": self.native_seq_len,
        }


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def summarize_runs(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "runs": []}
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "runs": [round(v, 6) for v in values],
    }


def compute_pct_degradation(value: float, baseline: float) -> float:
    return round(100.0 * (value - baseline) / baseline, 6)


def _find_source_file(run_dir: Path, experiment: str) -> Path:
    candidates = sorted((run_dir / "train_versions").glob(f"{experiment}_*.py"))
    if not candidates:
        raise FileNotFoundError(f"Unable to find source file for {run_dir}/{experiment}")
    keep = [path for path in candidates if path.name.endswith("_keep.py")]
    return keep[0] if keep else candidates[0]


def list_architectures_for_track(track: str) -> list[ArchitectureInfo]:
    root = RESULTS_DIR / track
    items: list[ArchitectureInfo] = []
    for summary_path in sorted(root.glob("run_*/summary.json")):
        data = read_json(summary_path, {})
        experiment = data.get("best_experiment")
        native_bpb = data.get("best_val_bpb")
        if not experiment or native_bpb is None:
            continue
        items.append(
            ArchitectureInfo(
                track=track,
                run=summary_path.parent.name,
                experiment=experiment,
                source_path=_find_source_file(summary_path.parent, experiment),
                native_bpb=float(native_bpb),
            )
        )
    items.sort(key=lambda item: item.native_bpb)
    if not items:
        raise FileNotFoundError(f"No architecture summaries found for track={track}")
    return items


def best_architectures_by_track() -> dict[str, ArchitectureInfo]:
    return {track: list_architectures_for_track(track)[0] for track in TRACKS}


def top_smiles_architectures(limit: int = 3) -> list[ArchitectureInfo]:
    return list_architectures_for_track("smiles")[:limit]


def baseline_bpbs() -> dict[str, float]:
    payload: dict[str, float] = {}
    for track in TRACKS:
        summary = read_json(RESULTS_DIR / "baselines" / "fixed_default" / track / "summary.json", {})
        payload[track] = float(summary["best_val_bpb"])
    return payload


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise ValueError(f"Unable to patch {label}: marker not found")
    return text.replace(old, new, 1)


def patched_train_source(source_text: str) -> str:
    if "recursive-mol-eval patched" in source_text:
        return source_text

    helper_block = textwrap.dedent(
        """
        # recursive-mol-eval patched
        import json
        import pathlib as _recursive_pathlib

        _RECURSIVE_PROJECT_ROOT = _recursive_pathlib.Path(
            os.environ.get("RECURSIVE_MOL_PROJECT_ROOT", ".")
        ).resolve()
        if TRACK == "nlp":
            _recursive_nlp_data_dir = os.environ.get(
                "RECURSIVE_MOL_NLP_DATA_DIR",
                str(_RECURSIVE_PROJECT_ROOT / "data" / "nlp"),
            )
            _recursive_nlp_tokenizer_dir = os.environ.get(
                "RECURSIVE_MOL_NLP_TOKENIZER_DIR",
                str(_recursive_pathlib.Path.home() / ".cache" / "autoresearch" / "tokenizer"),
            )
            if hasattr(prepare_mod, "DATA_DIR"):
                prepare_mod.DATA_DIR = _recursive_nlp_data_dir
            if hasattr(prepare_mod, "TOKENIZER_DIR"):
                prepare_mod.TOKENIZER_DIR = _recursive_nlp_tokenizer_dir
            if hasattr(prepare_mod, "CACHE_DIR"):
                prepare_mod.CACHE_DIR = str(_recursive_pathlib.Path(_recursive_nlp_data_dir).parent)

        MAX_SEQ_LEN = int(os.environ.get("RECURSIVE_MOL_MAX_SEQ_LEN", prepare_mod.MAX_SEQ_LEN))
        prepare_mod.MAX_SEQ_LEN = MAX_SEQ_LEN
        BASE_TIME_BUDGET = prepare_mod.TIME_BUDGET
        Tokenizer = prepare_mod.Tokenizer
        make_dataloader = prepare_mod.make_dataloader
        evaluate_bpb = prepare_mod.evaluate_bpb

        def _recursive_load_matching_state(model, checkpoint_path, device):
            try:
                state = torch.load(checkpoint_path, map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(checkpoint_path, map_location=device)
            model_state = model.state_dict()
            loadable = {}
            skipped_shape = {}
            unexpected = []
            for key, value in state.items():
                normalized_key = key[10:] if key.startswith("_orig_mod.") else key
                target = model_state.get(normalized_key)
                if target is None:
                    unexpected.append(key)
                    continue
                if tuple(target.shape) != tuple(value.shape):
                    skipped_shape[normalized_key] = {"checkpoint": list(value.shape), "model": list(target.shape)}
                    continue
                loadable[normalized_key] = value
            result = model.load_state_dict(loadable, strict=False)
            print(f"checkpoint_loaded: {checkpoint_path}")
            print(
                "checkpoint_stats: "
                f"matched={len(loadable)} missing={len(result.missing_keys)} "
                f"unexpected={len(unexpected) + len(result.unexpected_keys)} "
                f"skipped_shape={len(skipped_shape)} strict=False"
            )
            return {
                "missing_keys": list(result.missing_keys),
                "unexpected_keys": unexpected + list(result.unexpected_keys),
                "skipped_shape": skipped_shape,
            }

        def _recursive_freeze_layers(model, freeze_n):
            for idx, block in enumerate(list(model.transformer.h)):
                if idx >= freeze_n:
                    break
                for param in block.parameters():
                    param.requires_grad = False
            print(f"frozen_layers: {freeze_n}")

        def _recursive_get_hidden(model, idx):
            previous = os.environ.get("RECURSIVE_MOL_RETURN_HIDDEN")
            os.environ["RECURSIVE_MOL_RETURN_HIDDEN"] = "1"
            try:
                return model(idx)
            finally:
                if previous is None:
                    os.environ.pop("RECURSIVE_MOL_RETURN_HIDDEN", None)
                else:
                    os.environ["RECURSIVE_MOL_RETURN_HIDDEN"] = previous

        def _recursive_feature_dump(model, tokenizer, device):
            input_path = os.environ["RECURSIVE_MOL_FEATURE_INPUT"]
            output_path = os.environ["RECURSIVE_MOL_FEATURE_OUTPUT"]
            with open(input_path) as handle:
                payload = json.load(handle)
            texts = payload["texts"]
            batch_size = int(payload.get("batch_size", 128))
            bos = tokenizer.get_bos_token_id() if hasattr(tokenizer, "get_bos_token_id") else None
            eos = tokenizer.get_eos_token_id() if hasattr(tokenizer, "get_eos_token_id") else None
            pad = tokenizer.get_pad_token_id() if hasattr(tokenizer, "get_pad_token_id") else (bos or 0)
            encoded = []
            lengths = []
            truncated = 0
            for text in texts:
                token_ids = tokenizer.encode(text)
                if bos is not None:
                    token_ids = [bos] + token_ids
                if eos is not None:
                    token_ids = token_ids + [eos]
                if len(token_ids) > MAX_SEQ_LEN:
                    token_ids = token_ids[:MAX_SEQ_LEN]
                    truncated += 1
                encoded.append(token_ids)
                lengths.append(len(token_ids))
            outputs = []
            model.eval()
            for start in range(0, len(encoded), batch_size):
                batch_ids = encoded[start:start + batch_size]
                max_len = max(len(item) for item in batch_ids)
                batch = torch.full((len(batch_ids), max_len), pad, dtype=torch.long, device=device)
                mask = torch.zeros((len(batch_ids), max_len), dtype=torch.bool, device=device)
                for row, token_ids in enumerate(batch_ids):
                    batch[row, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long, device=device)
                    mask[row, : len(token_ids)] = True
                with torch.no_grad():
                    with autocast_ctx:
                        hidden = _recursive_get_hidden(model, batch)
                    pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
                outputs.append(pooled.float().cpu())
            torch.save(
                {
                    "features": torch.cat(outputs, dim=0) if outputs else torch.empty((0, model.config.n_embd)),
                    "lengths": torch.tensor(lengths, dtype=torch.int32),
                    "truncated": truncated,
                },
                output_path,
            )
            print(f"feature_dump_saved: {output_path}")
            raise SystemExit(0)

        def _recursive_generate(model, tokenizer, device):
            output_path = os.environ["RECURSIVE_MOL_GENERATE_OUTPUT"]
            num_samples = int(os.environ.get("RECURSIVE_MOL_NUM_SAMPLES", "10000"))
            batch_size = int(os.environ.get("RECURSIVE_MOL_GENERATE_BATCH_SIZE", "256"))
            max_new_tokens = int(os.environ.get("RECURSIVE_MOL_GENERATE_MAX_NEW_TOKENS", str(MAX_SEQ_LEN - 1)))
            temperature = float(os.environ.get("RECURSIVE_MOL_TEMPERATURE", "1.0"))
            top_k = int(os.environ.get("RECURSIVE_MOL_TOP_K", "50"))
            bos = tokenizer.get_bos_token_id() if hasattr(tokenizer, "get_bos_token_id") else 0
            eos = tokenizer.get_eos_token_id() if hasattr(tokenizer, "get_eos_token_id") else None
            generated = []
            model.eval()
            while len(generated) < num_samples:
                current_batch = min(batch_size, num_samples - len(generated))
                idx = torch.full((current_batch, 1), bos, dtype=torch.long, device=device)
                finished = torch.zeros(current_batch, dtype=torch.bool, device=device)
                for _ in range(max_new_tokens):
                    with torch.no_grad():
                        with autocast_ctx:
                            logits = model(idx)[:, -1, :].float()
                    if temperature <= 0:
                        next_token = logits.argmax(dim=-1)
                    else:
                        logits = logits / max(temperature, 1e-6)
                        if 0 < top_k < logits.size(-1):
                            threshold = torch.topk(logits, top_k).values[:, [-1]]
                            logits = logits.masked_fill(logits < threshold, float("-inf"))
                        probs = F.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    if eos is not None:
                        next_token = torch.where(finished, torch.full_like(next_token, eos), next_token)
                        finished = finished | next_token.eq(eos)
                    idx = torch.cat([idx, next_token[:, None]], dim=1)
                    if eos is not None and bool(finished.all()):
                        break
                for row in idx.tolist():
                    if eos is not None and eos in row[1:]:
                        row = row[: row.index(eos) + 1]
                    generated.append(tokenizer.decode(row))
            torch.save(
                {
                    "generated": generated,
                    "num_samples": num_samples,
                    "temperature": temperature,
                    "top_k": top_k,
                },
                output_path,
            )
            print(f"generation_saved: {output_path}")
            raise SystemExit(0)
        """
    ).strip()

    source_text = _replace_once(
        source_text,
        textwrap.dedent(
            """
            MAX_SEQ_LEN = prepare_mod.MAX_SEQ_LEN
            BASE_TIME_BUDGET = prepare_mod.TIME_BUDGET
            Tokenizer = prepare_mod.Tokenizer
            make_dataloader = prepare_mod.make_dataloader
            evaluate_bpb = prepare_mod.evaluate_bpb
            """
        ).strip(),
        helper_block,
        "prepare block",
    )
    source_text = _replace_once(
        source_text,
        'tokenizer = Tokenizer.from_directory()',
        'tokenizer = Tokenizer.from_directory(getattr(prepare_mod, "TOKENIZER_DIR", getattr(prepare_mod, "DATA_DIR", None)))',
        "tokenizer path override",
    )
    source_text = _replace_once(
        source_text,
        "torch.manual_seed(42)",
        'SEED = env_int("RECURSIVE_MOL_SEED", 42)\ntorch.manual_seed(SEED)',
        "seed override",
    )
    source_text = _replace_once(
        source_text,
        "torch.cuda.manual_seed(42)",
        "torch.cuda.manual_seed(SEED)",
        "cuda seed override",
    )
    source_text = _replace_once(
        source_text,
        "if targets is not None:",
        'if os.environ.get("RECURSIVE_MOL_RETURN_HIDDEN", "") == "1":\n            return x\n        if targets is not None:',
        "hidden hook",
    )
    source_text = _replace_once(
        source_text,
        "model.init_weights()",
        textwrap.dedent(
            """
            model.init_weights()

            _recursive_runtime_mode = os.environ.get("RECURSIVE_MOL_MODE", "train").lower()
            _recursive_checkpoint_load = os.environ.get("RECURSIVE_MOL_CHECKPOINT_LOAD", "").strip()
            if _recursive_checkpoint_load:
                _recursive_load_matching_state(model, _recursive_checkpoint_load, device)
            _recursive_freeze_n = int(os.environ.get("RECURSIVE_MOL_FREEZE_LAYERS", "0"))
            if _recursive_freeze_n > 0:
                _recursive_freeze_layers(model, _recursive_freeze_n)
            if _recursive_runtime_mode == "features":
                _recursive_feature_dump(model, tokenizer, device)
            if _recursive_runtime_mode == "generate":
                _recursive_generate(model, tokenizer, device)
            """
        ).strip(),
        "post-init runtime block",
    )
    source_text = _replace_once(
        source_text,
        "t_end = time.time()",
        textwrap.dedent(
            """
            _recursive_checkpoint_save = os.environ.get("RECURSIVE_MOL_CHECKPOINT_SAVE", "").strip()
            if _recursive_checkpoint_save:
                torch.save(model.state_dict(), _recursive_checkpoint_save)
                print(f"checkpoint_saved: {_recursive_checkpoint_save}")

            t_end = time.time()
            """
        ).strip(),
        "checkpoint save hook",
    )
    return source_text


def write_patched_train_script(source_path: Path) -> Path:
    patched = patched_train_source(source_path.read_text())
    tmp_dir = Path(tempfile.mkdtemp(prefix="recursive_mol_eval_", dir=PROJECT_ROOT))
    tmp_path = tmp_dir / source_path.name
    tmp_path.write_text(patched)
    return tmp_path


def parse_val_bpb(output: str) -> float:
    match = VAL_BPB_PATTERN.search(output)
    if match is None:
        raise ValueError("Unable to parse val_bpb from subprocess output")
    return float(match.group(1))


def _subprocess_env(extra_env: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{SRC_DIR}:{pythonpath}" if pythonpath else str(SRC_DIR)
    env["WANDB_DISABLED"] = "1"
    env["RECURSIVE_MOL_PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["RECURSIVE_MOL_NLP_DATA_DIR"] = str(PROJECT_ROOT / "data" / "nlp")
    env["RECURSIVE_MOL_NLP_TOKENIZER_DIR"] = str(Path.home() / ".cache" / "autoresearch" / "tokenizer")
    env.update(extra_env)
    return env


def _is_oom(output: str) -> bool:
    return any(
        marker in output
        for marker in (
            "CUDA out of memory",
            "torch.OutOfMemoryError",
            "CUBLAS_STATUS_ALLOC_FAILED",
            "DefaultCPUAllocator: can't allocate memory",
        )
    )


def _batch_candidates(track: str, override: int | None = None) -> list[int]:
    candidates = [override] if override is not None else list(TRACK_DEVICE_BATCH_CANDIDATES[track])
    valid: list[int] = []
    for candidate in candidates:
        if candidate is None:
            continue
        tokens_per_step = candidate * TRACK_MAX_SEQ_LEN[track]
        if DEFAULT_TOTAL_BATCH_SIZE % tokens_per_step == 0:
            valid.append(candidate)
    if not valid:
        raise ValueError(f"No valid batch-size candidates for track={track}")
    return valid


def run_architecture_subprocess(
    source_path: Path,
    *,
    track: str,
    seed: int,
    time_budget: int = DEFAULT_TIME_BUDGET,
    mode: str = "train",
    checkpoint_save: Path | None = None,
    checkpoint_load: Path | None = None,
    freeze_layers: int = 0,
    seq_len_override: int | None = None,
    device_batch_size: int | None = None,
    feature_input: Path | None = None,
    feature_output: Path | None = None,
    generate_output: Path | None = None,
    generate_batch_size: int | None = None,
    num_samples: int | None = None,
    top_k: int | None = None,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    patched_path = write_patched_train_script(source_path)
    last_error: RuntimeError | None = None
    for batch_size in _batch_candidates(track, device_batch_size):
        env = _subprocess_env(
            {
                "RECURSIVE_MOL_TRACK": track,
                "RECURSIVE_MOL_SEED": str(seed),
                "RECURSIVE_MOL_TIME_BUDGET": str(time_budget),
                "RECURSIVE_MOL_DEVICE_BATCH_SIZE": str(batch_size),
                "RECURSIVE_MOL_MODE": mode,
            }
        )
        if seq_len_override is not None:
            env["RECURSIVE_MOL_MAX_SEQ_LEN"] = str(seq_len_override)
        if checkpoint_save is not None:
            env["RECURSIVE_MOL_CHECKPOINT_SAVE"] = str(checkpoint_save)
        if checkpoint_load is not None:
            env["RECURSIVE_MOL_CHECKPOINT_LOAD"] = str(checkpoint_load)
        if freeze_layers:
            env["RECURSIVE_MOL_FREEZE_LAYERS"] = str(freeze_layers)
        if feature_input is not None:
            env["RECURSIVE_MOL_FEATURE_INPUT"] = str(feature_input)
        if feature_output is not None:
            env["RECURSIVE_MOL_FEATURE_OUTPUT"] = str(feature_output)
        if generate_output is not None:
            env["RECURSIVE_MOL_GENERATE_OUTPUT"] = str(generate_output)
        if generate_batch_size is not None:
            env["RECURSIVE_MOL_GENERATE_BATCH_SIZE"] = str(generate_batch_size)
        if num_samples is not None:
            env["RECURSIVE_MOL_NUM_SAMPLES"] = str(num_samples)
        if top_k is not None:
            env["RECURSIVE_MOL_TOP_K"] = str(top_k)
        if temperature is not None:
            env["RECURSIVE_MOL_TEMPERATURE"] = str(temperature)
        if max_new_tokens is not None:
            env["RECURSIVE_MOL_GENERATE_MAX_NEW_TOKENS"] = str(max_new_tokens)
        if mode != "train":
            env["RECURSIVE_MOL_ENABLE_COMPILE"] = "0"
            env["RECURSIVE_MOL_COMPILE_OPTIMIZER"] = "0"

        t0 = time.time()
        try:
            completed = subprocess.run(
                [sys.executable, str(patched_path)],
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + "\n" + (exc.stderr or "")
            last_error = RuntimeError(f"Timed out after {timeout_seconds}s\n{output}")
            continue
        elapsed = time.time() - t0
        output = (completed.stdout or "") + "\n" + (completed.stderr or "")
        result = {
            "track": track,
            "mode": mode,
            "seed": seed,
            "time_budget": time_budget,
            "device_batch_size": batch_size,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": now_iso(),
        }
        if completed.returncode == 0:
            if mode == "train":
                result["val_bpb"] = parse_val_bpb(output)
            return result
        if _is_oom(output):
            last_error = RuntimeError(output)
            continue
        raise RuntimeError(f"Subprocess failed for {source_path} on track={track} mode={mode}\n{output}")
    if last_error is not None:
        raise last_error
    raise RuntimeError("Subprocess retries exhausted")


def save_run_record(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def load_torch_payload(path: Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu")
