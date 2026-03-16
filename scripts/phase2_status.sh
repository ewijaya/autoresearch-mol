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

QUEUE="results/phase2/queue_state.json"

if [[ ! -f "$QUEUE" ]]; then
  echo "queue_state.json not found"
  exit 1
fi

STATUS=$(jq -r .status "$QUEUE")
TASK_INDEX=$(jq -r .task_index "$QUEUE")
TRACK=$(jq -r .task.track "$QUEUE")
KIND=$(jq -r .task.kind "$QUEUE")
RUN=$(jq -r '.task.run // empty' "$QUEUE")
TOTAL=34

# Build task label
if [[ "$KIND" == "agent" ]]; then
  TASK_LABEL="agent $TRACK run_$RUN"
elif [[ "$KIND" == "random_nas" ]]; then
  TASK_LABEL="random_nas $TRACK"
else
  TASK_LABEL="$KIND $TRACK"
fi

# Experiment count and best score from results.tsv for current track/run
if [[ -n "$RUN" ]]; then
  RESULTS_TSV="results/$TRACK/run_$RUN/results.tsv"
else
  RESULTS_TSV=""
fi

if [[ -n "$RESULTS_TSV" && -f "$RESULTS_TSV" ]]; then
  EXP_COUNT=$(( $(wc -l < "$RESULTS_TSV") - 1 ))
  BEST_LINE=$(awk 'NR>1 && $4=="keep" {print $1, $2}' "$RESULTS_TSV" | sort -k2 -n | head -1)
  BEST_EXP=$(echo "$BEST_LINE" | awk '{print $1}')
  BEST_VAL=$(echo "$BEST_LINE" | awk '{printf "%.4f", $2}')
  EXP_PART="| $EXP_COUNT/100 exp | best $BEST_VAL ($BEST_EXP)"
else
  EXP_PART=""
fi

echo "$TASK_INDEX/$TOTAL | $TASK_LABEL $EXP_PART"
