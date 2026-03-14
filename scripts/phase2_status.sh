#!/usr/bin/env bash
set -euo pipefail

SOURCE="${BASH_SOURCE[0]}"
while [[ -L "$SOURCE" ]]; do
  DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "== Codex Usage =="
if command -v codex-usage >/dev/null 2>&1; then
  if ! codex-usage; then
    echo "codex-usage unavailable"
  fi
else
  echo "codex-usage not found"
fi

echo
echo "== Queue State =="
if [[ -f results/phase2/queue_state.json ]]; then
  cat results/phase2/queue_state.json
else
  echo "results/phase2/queue_state.json not found"
fi

echo
echo "== Latest Phase 2 Log =="
latest_log="$(ls -1t logs/phase2-resume-*.log 2>/dev/null | head -n 1 || true)"
if [[ -n "${latest_log}" ]]; then
  echo "$latest_log"
  tail -n 20 "$latest_log"
else
  echo "No logs/phase2-resume-*.log found"
fi
