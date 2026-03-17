#!/usr/bin/env python3
"""
Comprehensive analysis of training dynamics: Agent Search vs Random NAS.
Parses per-step training logs and final summaries to compare methods.
"""

import os
import re
import glob
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────────────────────
BASE = "/home/ubuntu/storage1/recursive-mol/results"
OUT = os.path.join(BASE, "analysis")
os.makedirs(OUT, exist_ok=True)

AGENT_RUNS = {
    f"run_{i}": os.path.join(BASE, "smiles", f"run_{i}", "logs")
    for i in range(1, 6)
}
NAS_RUNS = {
    f"run_{i}": os.path.join(BASE, "baselines", "random_nas", "smiles", f"run_{i}", "logs")
    for i in range(1, 4)
}

# Plot styling
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})
AGENT_COLOR = '#2196F3'
NAS_COLOR = '#FF9800'

# ── Parsing ──────────────────────────────────────────────────────────────────

STEP_RE = re.compile(
    r'step\s+(\d+)\s+\((\d+\.?\d*)%\)\s*\|'
    r'\s*loss:\s+([\d.]+)\s*\|'
    r'\s*lrm:\s+([\d.]+)\s*\|'
    r'\s*dt:\s+(\d+)ms\s*\|'
    r'\s*tok/sec:\s+([\d,]+)\s*\|'
    r'\s*mfu:\s+([\d.]+)%\s*\|'
    r'\s*epoch:\s+(\d+)\s*\|'
    r'\s*remaining:\s+(\d+)s'
)

SUMMARY_FIELDS = {
    'val_bpb': float,
    'training_seconds': float,
    'total_seconds': float,
    'peak_vram_mb': float,
    'mfu_percent': float,
    'total_tokens_M': float,
    'num_steps': int,
    'num_params_M': float,
    'depth': int,
}

FLOPS_RE = re.compile(r'Estimated FLOPs per token:\s+([\d.eE+\-]+)')


def parse_log(filepath):
    """Parse a single experiment log file."""
    with open(filepath, 'r', errors='replace') as f:
        content = f.read()

    # Split on \r to handle carriage-return separated step lines
    parts = content.replace('\r', '\n').split('\n')

    steps = []
    for part in parts:
        m = STEP_RE.search(part)
        if m:
            steps.append({
                'step': int(m.group(1)),
                'progress': float(m.group(2)),
                'loss': float(m.group(3)),
                'lrm': float(m.group(4)),
                'dt_ms': int(m.group(5)),
                'tok_sec': int(m.group(6).replace(',', '')),
                'mfu': float(m.group(7)),
                'epoch': int(m.group(8)),
            })

    # Parse summary after ---
    summary = {}
    if '---' in content:
        summary_text = content.split('---')[-1]
        for key, converter in SUMMARY_FIELDS.items():
            m = re.search(rf'{key}:\s+([\d.eE+\-]+)', summary_text)
            if m:
                summary[key] = converter(m.group(1))

    # Parse FLOPs per token from header
    m = FLOPS_RE.search(content)
    if m:
        summary['flops_per_token'] = float(m.group(1))

    return steps, summary


def load_all_experiments(runs_dict, label):
    """Load all experiments for a method."""
    experiments = []
    for run_name, log_dir in sorted(runs_dict.items()):
        if not os.path.isdir(log_dir):
            continue
        log_files = sorted(glob.glob(os.path.join(log_dir, "exp*.log")))
        for lf in log_files:
            exp_id = os.path.basename(lf).replace('.log', '')
            exp_num = int(exp_id.replace('exp', ''))
            steps, summary = parse_log(lf)
            if not steps or 'val_bpb' not in summary:
                continue
            experiments.append({
                'method': label,
                'run': run_name,
                'exp_id': exp_id,
                'exp_num': exp_num,
                'steps': steps,
                'summary': summary,
            })
    return experiments


# ── Derived metrics per experiment ───────────────────────────────────────────

def compute_experiment_metrics(exp):
    """Compute convergence/stability/efficiency metrics for one experiment."""
    steps = exp['steps']
    if len(steps) < 10:
        return None
    losses = np.array([s['loss'] for s in steps])
    n = len(losses)

    # Skip first step (compilation warmup) for dt/tok_sec/mfu
    dt_vals = np.array([s['dt_ms'] for s in steps[1:]])
    tok_vals = np.array([s['tok_sec'] for s in steps[1:]])
    mfu_vals = np.array([s['mfu'] for s in steps[1:]])

    # Convergence: loss at quartiles
    q_indices = [max(0, int(n * q) - 1) for q in [0.25, 0.50, 0.75, 1.0]]
    loss_at_quartiles = [float(losses[i]) for i in q_indices]

    # Rate of improvement: first 25% vs last 25%
    first_q = losses[:max(1, n // 4)]
    last_q = losses[max(0, 3 * n // 4):]
    rate_first = (first_q[0] - first_q[-1]) / max(1, len(first_q)) if len(first_q) > 1 else 0
    rate_last = (last_q[0] - last_q[-1]) / max(1, len(last_q)) if len(last_q) > 1 else 0

    # End slope: linear regression on last 20% of steps
    tail_start = max(0, int(0.8 * n))
    tail_losses = losses[tail_start:]
    if len(tail_losses) > 2:
        x = np.arange(len(tail_losses))
        slope, _, _, _, _ = stats.linregress(x, tail_losses)
    else:
        slope = 0.0

    # Stability: loss variance in last 30%
    stable_start = max(0, int(0.7 * n))
    stable_losses = losses[stable_start:]
    loss_var = float(np.var(stable_losses))

    # Loss spikes: rolling average window
    window = max(5, n // 50)
    spike_count = 0
    for i in range(window, n):
        rolling_avg = np.mean(losses[max(0, i - window):i])
        if losses[i] > 2.0 * rolling_avg:
            spike_count += 1

    # Compute cost proxy: total FLOPs
    total_tokens = exp['summary'].get('total_tokens_M', 0) * 1e6
    flops_per_token = exp['summary'].get('flops_per_token', 0)
    total_flops = total_tokens * flops_per_token * 6  # 6 = fwd+bwd multiplier

    return {
        'val_bpb': exp['summary']['val_bpb'],
        'loss_q25': loss_at_quartiles[0],
        'loss_q50': loss_at_quartiles[1],
        'loss_q75': loss_at_quartiles[2],
        'loss_final': loss_at_quartiles[3],
        'rate_first_q': rate_first,
        'rate_last_q': rate_last,
        'end_slope': slope,
        'loss_var_last30': loss_var,
        'spike_count': spike_count,
        'median_tok_sec': float(np.median(tok_vals)) if len(tok_vals) > 0 else 0,
        'median_mfu': float(np.median(mfu_vals)) if len(mfu_vals) > 0 else 0,
        'median_dt_ms': float(np.median(dt_vals)) if len(dt_vals) > 0 else 0,
        'num_steps': exp['summary'].get('num_steps', n),
        'total_tokens_M': exp['summary'].get('total_tokens_M', 0),
        'total_flops': total_flops,
        'training_seconds': exp['summary'].get('training_seconds', 0),
        'num_params_M': exp['summary'].get('num_params_M', 0),
    }


# ── Analysis functions ───────────────────────────────────────────────────────

def best_so_far_curves(agent_exps, nas_exps):
    """Plot best val_bpb seen so far as experiments progress."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Per-run curves
    ax = axes[0]
    agent_runs = defaultdict(list)
    nas_runs = defaultdict(list)
    for e in agent_exps:
        agent_runs[e['run']].append(e)
    for e in nas_exps:
        nas_runs[e['run']].append(e)

    convergence_points = {'Agent': [], 'NAS': []}

    for label, runs_dict, color, ls_set in [
        ('Agent', agent_runs, AGENT_COLOR, ['-', '--', '-.', ':']),
        ('NAS', nas_runs, NAS_COLOR, ['-', '--', '-.', ':']),
    ]:
        for idx, (run_name, exps) in enumerate(sorted(runs_dict.items())):
            exps_sorted = sorted(exps, key=lambda e: e['exp_num'])
            bpbs = [e['summary']['val_bpb'] for e in exps_sorted]
            best_so_far = np.minimum.accumulate(bpbs)
            xs = np.arange(1, len(best_so_far) + 1)
            ls = ls_set[idx % len(ls_set)]
            ax.plot(xs, best_so_far, color=color, linestyle=ls, alpha=0.7,
                    label=f"{label} {run_name}")

            # Convergence point: first time within 1% of final best
            final_best = best_so_far[-1]
            threshold = final_best * 1.01
            conv_point = np.argmax(best_so_far <= threshold) + 1
            convergence_points[label].append(conv_point)

    ax.set_xlabel('Experiment Number')
    ax.set_ylabel('Best val_bpb So Far')
    ax.set_title('Best-So-Far Curves (per run)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Averaged across runs
    ax = axes[1]
    for label, runs_dict, color in [
        ('Agent', agent_runs, AGENT_COLOR),
        ('NAS', nas_runs, NAS_COLOR),
    ]:
        all_curves = []
        for run_name, exps in sorted(runs_dict.items()):
            exps_sorted = sorted(exps, key=lambda e: e['exp_num'])
            bpbs = [e['summary']['val_bpb'] for e in exps_sorted]
            all_curves.append(np.minimum.accumulate(bpbs))
        min_len = min(len(c) for c in all_curves)
        trimmed = np.array([c[:min_len] for c in all_curves])
        mean = trimmed.mean(axis=0)
        std = trimmed.std(axis=0)
        xs = np.arange(1, min_len + 1)
        ax.plot(xs, mean, color=color, linewidth=2, label=f'{label} (mean)')
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel('Experiment Number')
    ax.set_ylabel('Best val_bpb So Far')
    ax.set_title('Best-So-Far Curves (mean +/- std)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'best_so_far_curves.png'))
    plt.close()

    return convergence_points


def convergence_analysis(agent_metrics, nas_metrics):
    """Compare convergence characteristics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_to_plot = [
        ('end_slope', 'End Slope (last 20% of steps)', 'Slope'),
        ('rate_last_q', 'Improvement Rate (last 25%)', 'Rate'),
        ('loss_final', 'Final Training Loss', 'Loss'),
        ('val_bpb', 'Validation BPB', 'BPB'),
    ]

    test_results = {}
    for ax, (key, title, xlabel) in zip(axes.flat, metrics_to_plot):
        a_vals = [m[key] for m in agent_metrics]
        n_vals = [m[key] for m in nas_metrics]
        ax.hist(a_vals, bins=30, alpha=0.6, color=AGENT_COLOR, label='Agent', density=True)
        ax.hist(n_vals, bins=30, alpha=0.6, color=NAS_COLOR, label='NAS', density=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        u_stat, u_p = stats.mannwhitneyu(a_vals, n_vals, alternative='two-sided')
        t_stat, t_p = stats.ttest_ind(a_vals, n_vals, equal_var=False)
        test_results[key] = {
            'agent_mean': np.mean(a_vals), 'agent_std': np.std(a_vals),
            'nas_mean': np.mean(n_vals), 'nas_std': np.std(n_vals),
            'mwu_p': u_p, 'ttest_p': t_p,
        }

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'convergence_analysis.png'))
    plt.close()
    return test_results


def stability_analysis(agent_metrics, nas_metrics):
    """Compare training stability."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    test_results = {}
    for ax, key, title, xlabel in [
        (axes[0], 'loss_var_last30', 'Loss Variance (last 30%)', 'Variance'),
        (axes[1], 'spike_count', 'Loss Spike Count', 'Count'),
    ]:
        a_vals = [m[key] for m in agent_metrics]
        n_vals = [m[key] for m in nas_metrics]

        ax.hist(a_vals, bins=30, alpha=0.6, color=AGENT_COLOR, label='Agent', density=True)
        ax.hist(n_vals, bins=30, alpha=0.6, color=NAS_COLOR, label='NAS', density=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        u_stat, u_p = stats.mannwhitneyu(a_vals, n_vals, alternative='two-sided')
        t_stat, t_p = stats.ttest_ind(a_vals, n_vals, equal_var=False)
        test_results[key] = {
            'agent_mean': np.mean(a_vals), 'agent_std': np.std(a_vals),
            'nas_mean': np.mean(n_vals), 'nas_std': np.std(n_vals),
            'mwu_p': u_p, 'ttest_p': t_p,
        }

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'stability_analysis.png'))
    plt.close()
    return test_results


def efficiency_analysis(agent_metrics, nas_metrics):
    """Compare compute efficiency and Pareto frontiers."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # tok/sec distribution
    ax = axes[0, 0]
    a_vals = [m['median_tok_sec'] for m in agent_metrics]
    n_vals = [m['median_tok_sec'] for m in nas_metrics]
    ax.hist(a_vals, bins=30, alpha=0.6, color=AGENT_COLOR, label='Agent', density=True)
    ax.hist(n_vals, bins=30, alpha=0.6, color=NAS_COLOR, label='NAS', density=True)
    ax.set_xlabel('Median tok/sec')
    ax.set_title('Throughput Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MFU distribution
    ax = axes[0, 1]
    a_vals = [m['median_mfu'] for m in agent_metrics]
    n_vals = [m['median_mfu'] for m in nas_metrics]
    ax.hist(a_vals, bins=30, alpha=0.6, color=AGENT_COLOR, label='Agent', density=True)
    ax.hist(n_vals, bins=30, alpha=0.6, color=NAS_COLOR, label='NAS', density=True)
    ax.set_xlabel('Median MFU (%)')
    ax.set_title('MFU Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pareto: val_bpb vs total_tokens
    ax = axes[1, 0]
    a_bpb = [m['val_bpb'] for m in agent_metrics]
    a_tok = [m['total_tokens_M'] for m in agent_metrics]
    n_bpb = [m['val_bpb'] for m in nas_metrics]
    n_tok = [m['total_tokens_M'] for m in nas_metrics]
    ax.scatter(a_tok, a_bpb, alpha=0.3, color=AGENT_COLOR, label='Agent', s=15)
    ax.scatter(n_tok, n_bpb, alpha=0.3, color=NAS_COLOR, label='NAS', s=15)

    # Compute and plot Pareto frontiers
    for tok_vals, bpb_vals, color, lbl in [
        (a_tok, a_bpb, AGENT_COLOR, 'Agent Pareto'),
        (n_tok, n_bpb, NAS_COLOR, 'NAS Pareto'),
    ]:
        pts = sorted(zip(tok_vals, bpb_vals))
        pareto = []
        best = float('inf')
        for t, b in pts:
            if b < best:
                pareto.append((t, b))
                best = b
        if pareto:
            px, py = zip(*pareto)
            ax.step(px, py, where='post', color=color, linewidth=2, linestyle='--',
                    label=lbl)

    ax.set_xlabel('Total Tokens (M)')
    ax.set_ylabel('val_bpb')
    ax.set_title('Pareto: val_bpb vs Tokens')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Pareto: val_bpb vs total_flops
    ax = axes[1, 1]
    a_flops = [m['total_flops'] / 1e12 for m in agent_metrics]
    n_flops = [m['total_flops'] / 1e12 for m in nas_metrics]
    ax.scatter(a_flops, a_bpb, alpha=0.3, color=AGENT_COLOR, label='Agent', s=15)
    ax.scatter(n_flops, n_bpb, alpha=0.3, color=NAS_COLOR, label='NAS', s=15)

    for flops_vals, bpb_vals, color, lbl in [
        (a_flops, a_bpb, AGENT_COLOR, 'Agent Pareto'),
        (n_flops, n_bpb, NAS_COLOR, 'NAS Pareto'),
    ]:
        pts = sorted(zip(flops_vals, bpb_vals))
        pareto = []
        best = float('inf')
        for f, b in pts:
            if b < best:
                pareto.append((f, b))
                best = b
        if pareto:
            px, py = zip(*pareto)
            ax.step(px, py, where='post', color=color, linewidth=2, linestyle='--',
                    label=lbl)

    ax.set_xlabel('Total FLOPs (TFLOPs)')
    ax.set_ylabel('val_bpb')
    ax.set_title('Pareto: val_bpb vs FLOPs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'efficiency_analysis.png'))
    plt.close()

    # Statistical tests for efficiency
    test_results = {}
    for key, name in [('median_tok_sec', 'tok/sec'), ('median_mfu', 'MFU')]:
        a = [m[key] for m in agent_metrics]
        n = [m[key] for m in nas_metrics]
        u_stat, u_p = stats.mannwhitneyu(a, n, alternative='two-sided')
        t_stat, t_p = stats.ttest_ind(a, n, equal_var=False)
        test_results[key] = {
            'agent_mean': np.mean(a), 'agent_std': np.std(a),
            'nas_mean': np.mean(n), 'nas_std': np.std(n),
            'mwu_p': u_p, 'ttest_p': t_p,
        }
    return test_results


def keep_rate_analysis(agent_exps, nas_exps):
    """Plot cumulative keep rate over experiment progression."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Per run
    ax = axes[0]
    agent_runs = defaultdict(list)
    nas_runs = defaultdict(list)
    for e in agent_exps:
        agent_runs[e['run']].append(e)
    for e in nas_exps:
        nas_runs[e['run']].append(e)

    for label, runs_dict, color, ls_set in [
        ('Agent', agent_runs, AGENT_COLOR, ['-', '--', '-.', ':']),
        ('NAS', nas_runs, NAS_COLOR, ['-', '--', '-.', ':']),
    ]:
        for idx, (run_name, exps) in enumerate(sorted(runs_dict.items())):
            exps_sorted = sorted(exps, key=lambda e: e['exp_num'])
            bpbs = [e['summary']['val_bpb'] for e in exps_sorted]
            # "Kept" = is this the new best so far?
            best = float('inf')
            kept = []
            for b in bpbs:
                if b < best:
                    kept.append(1)
                    best = b
                else:
                    kept.append(0)
            cum_keep = np.cumsum(kept) / np.arange(1, len(kept) + 1)
            xs = np.arange(1, len(cum_keep) + 1)
            ls = ls_set[idx % len(ls_set)]
            ax.plot(xs, cum_keep, color=color, linestyle=ls, alpha=0.7,
                    label=f"{label} {run_name}")

    ax.set_xlabel('Experiment Number')
    ax.set_ylabel('Cumulative Keep Rate')
    ax.set_title('Keep Rate Over Time (per run)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Averaged
    ax = axes[1]
    for label, runs_dict, color in [
        ('Agent', agent_runs, AGENT_COLOR),
        ('NAS', nas_runs, NAS_COLOR),
    ]:
        all_curves = []
        for run_name, exps in sorted(runs_dict.items()):
            exps_sorted = sorted(exps, key=lambda e: e['exp_num'])
            bpbs = [e['summary']['val_bpb'] for e in exps_sorted]
            best = float('inf')
            kept = []
            for b in bpbs:
                if b < best:
                    kept.append(1)
                    best = b
                else:
                    kept.append(0)
            all_curves.append(np.cumsum(kept) / np.arange(1, len(kept) + 1))
        min_len = min(len(c) for c in all_curves)
        trimmed = np.array([c[:min_len] for c in all_curves])
        mean = trimmed.mean(axis=0)
        std = trimmed.std(axis=0)
        xs = np.arange(1, min_len + 1)
        ax.plot(xs, mean, color=color, linewidth=2, label=f'{label} (mean)')
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel('Experiment Number')
    ax.set_ylabel('Cumulative Keep Rate')
    ax.set_title('Keep Rate Over Time (mean +/- std)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'keep_rate_analysis.png'))
    plt.close()


def loss_curves_sample(agent_exps, nas_exps):
    """Plot sample loss curves (best, median, worst per method) for visual comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, exps, label, color in [
        (axes[0], agent_exps, 'Agent', AGENT_COLOR),
        (axes[1], nas_exps, 'NAS', NAS_COLOR),
    ]:
        bpbs = [e['summary']['val_bpb'] for e in exps]
        sorted_idx = np.argsort(bpbs)
        picks = {
            'Best': sorted_idx[0],
            'Median': sorted_idx[len(sorted_idx) // 2],
            'Worst': sorted_idx[-1],
        }
        for pick_label, idx in picks.items():
            e = exps[idx]
            losses = [s['loss'] for s in e['steps']]
            n = len(losses)
            progress = np.linspace(0, 100, n)
            ax.plot(progress, losses, alpha=0.8,
                    label=f"{pick_label} ({e['exp_id']}, bpb={e['summary']['val_bpb']:.4f})")
        ax.set_xlabel('Training Progress (%)')
        ax.set_ylabel('Loss')
        ax.set_title(f'{label} Loss Curves (best/median/worst)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'loss_curves_sample.png'))
    plt.close()


def convergence_speed_plot(agent_exps, nas_exps):
    """For each experiment, plot loss reduction over normalized training progress."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for exps, label, color in [
        (agent_exps, 'Agent', AGENT_COLOR),
        (nas_exps, 'NAS', NAS_COLOR),
    ]:
        # Compute mean loss curve across all experiments, normalized to progress
        n_bins = 100
        all_binned = []
        for e in exps:
            losses = np.array([s['loss'] for s in e['steps']])
            n = len(losses)
            if n < 20:
                continue
            # Normalize: bin into 100 progress bins
            bin_indices = np.linspace(0, n - 1, n_bins).astype(int)
            binned = losses[bin_indices]
            # Normalize loss: start at 1.0
            binned = binned / binned[0]
            all_binned.append(binned)

        all_binned = np.array(all_binned)
        mean = all_binned.mean(axis=0)
        std = all_binned.std(axis=0)
        xs = np.linspace(0, 100, n_bins)
        ax.plot(xs, mean, color=color, linewidth=2, label=f'{label} (mean)')
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.12)

    ax.set_xlabel('Training Progress (%)')
    ax.set_ylabel('Normalized Loss (relative to step 0)')
    ax.set_title('Normalized Loss Curves (mean +/- std)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'normalized_loss_curves.png'))
    plt.close()


def val_bpb_distribution(agent_metrics, nas_metrics):
    """Violin/box plot of val_bpb distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    ax = axes[0]
    a_bpb = [m['val_bpb'] for m in agent_metrics]
    n_bpb = [m['val_bpb'] for m in nas_metrics]
    bp = ax.boxplot([a_bpb, n_bpb], labels=['Agent', 'NAS'], patch_artist=True,
                    widths=0.5)
    bp['boxes'][0].set_facecolor(AGENT_COLOR)
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(NAS_COLOR)
    bp['boxes'][1].set_alpha(0.6)
    ax.set_ylabel('val_bpb')
    ax.set_title('val_bpb Distribution')
    ax.grid(True, alpha=0.3)

    # CDF
    ax = axes[1]
    for vals, color, label in [(a_bpb, AGENT_COLOR, 'Agent'), (n_bpb, NAS_COLOR, 'NAS')]:
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, color=color, linewidth=2, label=label)
    ax.set_xlabel('val_bpb')
    ax.set_ylabel('CDF')
    ax.set_title('Cumulative Distribution of val_bpb')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'val_bpb_distribution.png'))
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("TRAINING DYNAMICS ANALYSIS: Agent Search vs Random NAS")
    print("=" * 80)

    # Load data
    print("\n--- Loading data ---")
    agent_exps = load_all_experiments(AGENT_RUNS, 'Agent')
    nas_exps = load_all_experiments(NAS_RUNS, 'NAS')
    print(f"Agent experiments loaded: {len(agent_exps)} (runs: {sorted(set(e['run'] for e in agent_exps))})")
    print(f"NAS experiments loaded:   {len(nas_exps)} (runs: {sorted(set(e['run'] for e in nas_exps))})")

    # Compute per-experiment metrics
    agent_metrics = [m for m in (compute_experiment_metrics(e) for e in agent_exps) if m]
    nas_metrics = [m for m in (compute_experiment_metrics(e) for e in nas_exps) if m]
    print(f"Agent metrics computed:   {len(agent_metrics)}")
    print(f"NAS metrics computed:     {len(nas_metrics)}")

    # ── 1. Best-so-far curves ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("1. BEST-SO-FAR CURVES (Sample Efficiency)")
    print("=" * 80)
    conv_points = best_so_far_curves(agent_exps, nas_exps)
    for method, pts in conv_points.items():
        print(f"  {method}: experiments to reach within 1% of final best: {pts} "
              f"(mean={np.mean(pts):.1f}, std={np.std(pts):.1f})")
    if conv_points['Agent'] and conv_points['NAS']:
        u_stat, u_p = stats.mannwhitneyu(conv_points['Agent'], conv_points['NAS'],
                                          alternative='two-sided')
        print(f"  Mann-Whitney U p-value for convergence speed: {u_p:.4f}")

    # ── 2. Convergence analysis ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("2. CONVERGENCE ANALYSIS")
    print("=" * 80)
    conv_results = convergence_analysis(agent_metrics, nas_metrics)
    for key, r in conv_results.items():
        print(f"\n  {key}:")
        print(f"    Agent: {r['agent_mean']:.6f} +/- {r['agent_std']:.6f}")
        print(f"    NAS:   {r['nas_mean']:.6f} +/- {r['nas_std']:.6f}")
        print(f"    Mann-Whitney U p={r['mwu_p']:.4e}, t-test p={r['ttest_p']:.4e}")
        sig = "***" if r['mwu_p'] < 0.001 else "**" if r['mwu_p'] < 0.01 else "*" if r['mwu_p'] < 0.05 else "n.s."
        print(f"    Significance: {sig}")

    # ── 3. Stability analysis ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("3. STABILITY ANALYSIS")
    print("=" * 80)
    stab_results = stability_analysis(agent_metrics, nas_metrics)
    for key, r in stab_results.items():
        print(f"\n  {key}:")
        print(f"    Agent: {r['agent_mean']:.6f} +/- {r['agent_std']:.6f}")
        print(f"    NAS:   {r['nas_mean']:.6f} +/- {r['nas_std']:.6f}")
        print(f"    Mann-Whitney U p={r['mwu_p']:.4e}, t-test p={r['ttest_p']:.4e}")
        sig = "***" if r['mwu_p'] < 0.001 else "**" if r['mwu_p'] < 0.01 else "*" if r['mwu_p'] < 0.05 else "n.s."
        print(f"    Significance: {sig}")

    # ── 4. Efficiency analysis ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("4. EFFICIENCY ANALYSIS")
    print("=" * 80)
    eff_results = efficiency_analysis(agent_metrics, nas_metrics)
    for key, r in eff_results.items():
        print(f"\n  {key}:")
        print(f"    Agent: {r['agent_mean']:.2f} +/- {r['agent_std']:.2f}")
        print(f"    NAS:   {r['nas_mean']:.2f} +/- {r['nas_std']:.2f}")
        print(f"    Mann-Whitney U p={r['mwu_p']:.4e}, t-test p={r['ttest_p']:.4e}")
        sig = "***" if r['mwu_p'] < 0.001 else "**" if r['mwu_p'] < 0.01 else "*" if r['mwu_p'] < 0.05 else "n.s."
        print(f"    Significance: {sig}")

    # Additional efficiency stats
    a_params = [m['num_params_M'] for m in agent_metrics]
    n_params = [m['num_params_M'] for m in nas_metrics]
    print(f"\n  Parameter count (M):")
    print(f"    Agent: {np.mean(a_params):.2f} +/- {np.std(a_params):.2f}")
    print(f"    NAS:   {np.mean(n_params):.2f} +/- {np.std(n_params):.2f}")

    a_steps = [m['num_steps'] for m in agent_metrics]
    n_steps = [m['num_steps'] for m in nas_metrics]
    print(f"\n  Number of training steps:")
    print(f"    Agent: {np.mean(a_steps):.0f} +/- {np.std(a_steps):.0f}")
    print(f"    NAS:   {np.mean(n_steps):.0f} +/- {np.std(n_steps):.0f}")

    a_tok = [m['total_tokens_M'] for m in agent_metrics]
    n_tok = [m['total_tokens_M'] for m in nas_metrics]
    print(f"\n  Total tokens (M):")
    print(f"    Agent: {np.mean(a_tok):.1f} +/- {np.std(a_tok):.1f}")
    print(f"    NAS:   {np.mean(n_tok):.1f} +/- {np.std(n_tok):.1f}")

    # ── 5. Keep rate analysis ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("5. KEEP RATE ANALYSIS")
    print("=" * 80)
    keep_rate_analysis(agent_exps, nas_exps)

    # Compute final keep rates per run
    for label, exps_list in [('Agent', agent_exps), ('NAS', nas_exps)]:
        runs = defaultdict(list)
        for e in exps_list:
            runs[e['run']].append(e)
        for run_name, exps in sorted(runs.items()):
            exps_sorted = sorted(exps, key=lambda e: e['exp_num'])
            bpbs = [e['summary']['val_bpb'] for e in exps_sorted]
            best = float('inf')
            n_kept = 0
            for b in bpbs:
                if b < best:
                    n_kept += 1
                    best = b
            print(f"  {label} {run_name}: {n_kept}/{len(bpbs)} kept "
                  f"({100*n_kept/len(bpbs):.1f}%), final best={best:.6f}")

    # ── 6. Additional plots ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("6. ADDITIONAL VISUALIZATIONS")
    print("=" * 80)
    loss_curves_sample(agent_exps, nas_exps)
    convergence_speed_plot(agent_exps, nas_exps)
    val_bpb_distribution(agent_metrics, nas_metrics)
    print("  Generated: loss_curves_sample.png, normalized_loss_curves.png, val_bpb_distribution.png")

    # ── 7. Summary statistics table ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("7. COMPREHENSIVE SUMMARY TABLE")
    print("=" * 80)

    a_bpb = [m['val_bpb'] for m in agent_metrics]
    n_bpb = [m['val_bpb'] for m in nas_metrics]

    print(f"\n  {'Metric':<30s} {'Agent':>15s} {'NAS':>15s} {'p-value':>12s} {'Sig':>5s}")
    print(f"  {'-'*77}")

    summary_metrics = [
        ('val_bpb (mean)', a_bpb, n_bpb),
        ('val_bpb (best)', [min(a_bpb)], [min(n_bpb)]),
        ('val_bpb (median)', [np.median(a_bpb)], [np.median(n_bpb)]),
        ('Final loss', [m['loss_final'] for m in agent_metrics],
                       [m['loss_final'] for m in nas_metrics]),
        ('End slope', [m['end_slope'] for m in agent_metrics],
                      [m['end_slope'] for m in nas_metrics]),
        ('Loss var (last 30%)', [m['loss_var_last30'] for m in agent_metrics],
                                [m['loss_var_last30'] for m in nas_metrics]),
        ('Spike count', [m['spike_count'] for m in agent_metrics],
                        [m['spike_count'] for m in nas_metrics]),
        ('Median tok/sec', [m['median_tok_sec'] for m in agent_metrics],
                           [m['median_tok_sec'] for m in nas_metrics]),
        ('Median MFU %', [m['median_mfu'] for m in agent_metrics],
                         [m['median_mfu'] for m in nas_metrics]),
        ('Num steps', [m['num_steps'] for m in agent_metrics],
                      [m['num_steps'] for m in nas_metrics]),
        ('Total tokens (M)', [m['total_tokens_M'] for m in agent_metrics],
                             [m['total_tokens_M'] for m in nas_metrics]),
        ('Params (M)', [m['num_params_M'] for m in agent_metrics],
                       [m['num_params_M'] for m in nas_metrics]),
    ]

    for name, a, n in summary_metrics:
        a_mean = np.mean(a)
        n_mean = np.mean(n)
        if len(a) > 1 and len(n) > 1:
            _, p = stats.mannwhitneyu(a, n, alternative='two-sided')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {name:<30s} {a_mean:>15.4f} {n_mean:>15.4f} {p:>12.4e} {sig:>5s}")
        else:
            print(f"  {name:<30s} {a_mean:>15.4f} {n_mean:>15.4f} {'N/A':>12s} {'':>5s}")

    # Effect sizes (Cohen's d)
    print(f"\n  Effect sizes (Cohen's d):")
    for name, a, n in summary_metrics:
        if len(a) > 1 and len(n) > 1:
            pooled_std = np.sqrt((np.var(a) + np.var(n)) / 2)
            if pooled_std > 0:
                d = (np.mean(a) - np.mean(n)) / pooled_std
                mag = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small" if abs(d) > 0.2 else "negligible"
                print(f"    {name:<30s} d={d:>+.4f} ({mag})")

    print(f"\n  Plots saved to: {OUT}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
