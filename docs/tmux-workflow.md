# Tmux Workflow for Running Phase Prompts

Run phase prompts from `docs/phase-prompts.md` on your AWS EC2 instance without keeping your local machine on.

---

## First Time Setup

```bash
# SSH into your instance
awsg3

# Install tmux (if not already installed)
sudo apt install tmux -y
```

---

## Running a Phase Prompt

```bash
# 1. SSH in
awsg3

# 2. Start a named tmux session
tmux new -s phase1

# 3. Navigate to the project
cd recursive-mol

# 4. Start Codex CLI
codex

# 5. Paste the Phase 1 prompt from docs/phase-prompts.md as-is.
#    It starts with:
#      "You are working on the recursive-mol project..."
#    and ends with:
#      "...report the status of each Checkpoint 1 criterion below."
#
#    Codex will start working autonomously.

# 6. Detach when you want to leave:
#    Press Ctrl+B, then D
#
# 7. Close your terminal / shut down your PC. Codex keeps running.
```

---

## Checking Progress Later

```bash
# SSH back in (from any machine)
awsg3

# Reattach to the running session
tmux attach -t phase1

# You'll see Codex's progress right where you left off
```

---

## Auto-Stop Between Phases

Each phase prompt ends with `stopinstance`, which stops the EC2 instance when the phase completes. To start the next phase:

```bash
# 1. Start the instance (from your local machine)
aws ec2 start-instances --instance-ids i-0620c2546bd7f9322

# 2. SSH in
awsg3

# 3. Start a new tmux session for the next phase
tmux new -s phase2
cd recursive-mol
codex
# Paste the next phase prompt
```

## Repeating for Later Phases

```bash
# For Phase 2, 3, etc. — just use a different session name
tmux new -s phase2
cd recursive-mol
codex
# Paste the Phase 2 prompt from docs/phase-prompts.md
```

---

## Tmux Cheat Sheet

| Action | Keys |
|---|---|
| Detach (leave without stopping) | `Ctrl+B`, then `D` |
| Reattach to session | `tmux attach -t phase1` |
| List all sessions | `tmux ls` |
| Scroll up to see output | `Ctrl+B`, then `[`, then arrow keys. `q` to exit |
| Kill session (when done) | `tmux kill-session -t phase1` |

---

## Troubleshooting

### SSH connection dropped before detaching
Tmux survives SSH disconnects. Just reconnect and reattach:
```bash
awsg3
tmux attach -t phase1
```

### Spot instance got reclaimed
Re-launch the instance and re-run. Check `results/` for any saved progress — Codex can pick up from where things left off.

### Codex is waiting for input
If Codex hit a decision gate and is waiting for your response, reattach with `tmux attach -t phase1` and respond directly.
