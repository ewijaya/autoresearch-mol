# Phase 2 Weekly Limit Recovery

Use this when `codex-usage` shows the **Weekly limit** is `0% left`.

## What will happen

The Phase 2 runner is configured to:

- auto-wait only for short `5h` limit pauses
- exit cleanly on `weekly` limit pauses
- save resumable state in `results/phase2/queue_state.json`

That means you should stop the EC2 instance and restart it after the weekly reset.

## 1. Confirm the runner paused on weekly limit

```bash
cd /home/ubuntu/storage1/recursive-mol
cat results/phase2/queue_state.json
```

You want to see something like:

```json
{
  "status": "paused_rate_limit",
  "rate_limit_scope": "weekly",
  "next_retry_at": "YYYY-MM-DD HH:MM:SS"
}
```

Optional checks:

```bash
cd /home/ubuntu/storage1/recursive-mol
codex-usage
ls -1t logs/phase2-resume-*.log | head -n 1
tail -n 40 "$(ls -1t logs/phase2-resume-*.log | head -n 1)"
```

## 2. Stop the EC2 instance to avoid idle GPU cost

```bash
/home/ubuntu/bin/stopinstance
```

## 3. After the weekly reset, start the instance again

Reconnect, then verify the repo state:

```bash
cd /home/ubuntu/storage1/recursive-mol
codex-usage
cat results/phase2/queue_state.json
```

If the weekly limit has reset, resume Phase 2.

## 4. Resume Phase 2

Launch the runner again in detached `tmux`:

```bash
cd /home/ubuntu/storage1/recursive-mol
tmux new-session -d -s phase2-resume 'cd /home/ubuntu/storage1/recursive-mol && .venv/bin/python src/phase2_runner.py > logs/phase2-resume-$(date +%Y%m%d_%H%M%S).log 2>&1; status=$?; if [ $status -eq 0 ]; then /home/ubuntu/bin/stopinstance; fi; exit $status'
```

## 5. Verify that it resumed

```bash
cd /home/ubuntu/storage1/recursive-mol
tmux ls
LATEST_LOG="$(ls -1t logs/phase2-resume-*.log | head -n 1)"
echo "$LATEST_LOG"
tail -n 40 "$LATEST_LOG"
cat results/phase2/queue_state.json
```

The runner should continue from the saved task and experiment count. It will not restart completed runs from scratch.

## One-command checklist

If you just want the minimum commands:

```bash
cd /home/ubuntu/storage1/recursive-mol
cat results/phase2/queue_state.json
/home/ubuntu/bin/stopinstance
```

Later, after the weekly reset:

```bash
cd /home/ubuntu/storage1/recursive-mol
codex-usage
tmux new-session -d -s phase2-resume 'cd /home/ubuntu/storage1/recursive-mol && .venv/bin/python src/phase2_runner.py > logs/phase2-resume-$(date +%Y%m%d_%H%M%S).log 2>&1; status=$?; if [ $status -eq 0 ]; then /home/ubuntu/bin/stopinstance; fi; exit $status'
```
