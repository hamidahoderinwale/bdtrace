#!/bin/bash
# Guarded smoke-3 orchestrator: serve -> wait for URL -> rollout 3 instances -> teardown.
# Cost guards: smoke-only, hard timeouts, ALWAYS kill the serve at the end.
set -u
cd /Users/hamidaho/learning-from-dev/bidirect-align-dev-traces
V=distillation_run/.venv/bin
LOG=/tmp/swe_smoke_orchestrator.log
: > "$LOG"
mkdir -p distillation_run/child_traj

teardown() { tmux kill-session -t sweserve 2>/dev/null; echo "[$(date +%T)] serve torn down" >> "$LOG"; }
trap teardown EXIT

echo "[$(date +%T)] launching modal serve (GPU A100)" >> "$LOG"
tmux kill-session -t sweserve 2>/dev/null
tmux new-session -d -s sweserve "$V/modal serve distillation_run/modal_serve.py > /tmp/swe_serve.log 2>&1"

# wait up to ~12.5 min for the endpoint URL (cold start + first image build)
URL=""; n=0
until [ -n "$URL" ] || [ $n -ge 150 ]; do
  URL=$(grep -oE 'https://[A-Za-z0-9._-]+\.modal\.run' /tmp/swe_serve.log 2>/dev/null | head -1)
  if [ -z "$URL" ] && grep -qiE "error|traceback|exception|failed" /tmp/swe_serve.log 2>/dev/null; then
    echo "[$(date +%T)] serve error before URL" >> "$LOG"; break
  fi
  n=$((n+1)); sleep 5
done
echo "[$(date +%T)] URL=${URL:-<none>} (waited $((n*5))s)" >> "$LOG"

if [ -n "$URL" ]; then
  echo "[$(date +%T)] smoke-3 rollout starting" >> "$LOG"
  # portable wall-clock cap (macOS has no `timeout`): background + watchdog (~60 min)
  "$V/python" distillation_run/rollout.py --smoke --endpoint "$URL" >> "$LOG" 2>&1 &
  RPID=$!; w=0
  until ! kill -0 "$RPID" 2>/dev/null || [ $w -ge 720 ]; do sleep 5; w=$((w+1)); done
  if kill -0 "$RPID" 2>/dev/null; then echo "[$(date +%T)] CAP hit -- killing rollout" >> "$LOG"; kill -9 "$RPID" 2>/dev/null; fi
  echo "[$(date +%T)] rollout finished (waited $((w*5))s)" >> "$LOG"
else
  echo "[$(date +%T)] NO URL -- serve log tail:" >> "$LOG"; tail -25 /tmp/swe_serve.log >> "$LOG" 2>&1
fi

echo "[$(date +%T)] captured trajectories:" >> "$LOG"
ls -la distillation_run/child_traj/ >> "$LOG" 2>&1
echo "SMOKE_ORCHESTRATOR_DONE" >> "$LOG"
