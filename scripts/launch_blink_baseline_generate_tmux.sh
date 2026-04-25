#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANSWER_BACKEND="${BLINK_ANSWER_BACKEND:-loss}"
LOSS_CHOICE_FORMAT="${BLINK_LOSS_CHOICE_FORMAT:-plain_text}"
RUN_ID="${1:-$(date -u +%Y%m%d_%H%M%S)_baseline_mllava_blink_multiview_full133_${ANSWER_BACKEND}}"
RUN_DIR="${RUN_DIR:-/mnt/jarvis/migration/Mantis/runs/autoprune/blink/${RUN_ID}}"
MANTIS_ROOT="${MANTIS_ROOT:-/mnt/jarvis/migration/Mantis}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$MANTIS_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$MANTIS_ROOT/.venv/bin/python"
  elif [[ -x /mnt/jarvis/migration/envs/Mantis.venv/bin/python ]]; then
    PYTHON_BIN="/mnt/jarvis/migration/envs/Mantis.venv/bin/python"
  else
    PYTHON_BIN="/home/liudongxu/Mantis/.venv/bin/python"
  fi
fi
GPU="${CUDA_VISIBLE_DEVICES:-0}"
SESSION="baseline_gen_${RUN_ID}_g${GPU}"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/results" "$RUN_DIR/pids" "$RUN_DIR/workers"

LOG_PATH="$RUN_DIR/logs/baseline_raw.log"
RESULT_PATH="$RUN_DIR/results/baseline_raw.json"
PID_PATH="$RUN_DIR/pids/baseline_raw.pid"
WORKER_PATH="$RUN_DIR/workers/baseline_raw.sh"

cat > "$RUN_DIR/launch_manifest.tsv" <<MANIFEST
name	gpu	session	log	result
baseline_raw	$GPU	$SESSION	$LOG_PATH	$RESULT_PATH
MANIFEST

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 1
fi

cat > "$WORKER_PATH" <<WORKER
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT_DIR"
export MANTIS_ROOT="$MANTIS_ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/jarvis/migration/.cache}"
export HF_HOME="${HF_HOME:-/mnt/jarvis/migration/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/mnt/jarvis/migration/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/jarvis/migration/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/jarvis/migration/.cache/huggingface/hub}"
export TMPDIR="${TMPDIR:-/mnt/jarvis/migration/tmp}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$MANTIS_ROOT:$MANTIS_ROOT/mantis:\${PYTHONPATH:-}"
unset MANTIS_BASELINE_PRUNE_ENABLED
unset MANTIS_BASELINE_PRUNING_ENABLED
echo \$\$ > "$PID_PATH"
echo "[launcher] session=$SESSION gpu=$GPU baseline=no_pruning result=$RESULT_PATH" > "$LOG_PATH"
echo "[launcher] started_utc=\$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_PATH"
exec "$PYTHON_BIN" "$MANTIS_ROOT/scripts/autoprune/run_blink_multiview.py" \\
  --model-name "${MANTIS_MODEL_NAME:-mantis-8b-siglip-llama3-ablation1}" \\
  --task-name "${BLINK_TASK_NAME:-Multi-view_Reasoning}" \\
  --split "${BLINK_SPLIT:-val}" \\
  --eval-mode accuracy \\
  --answer-backend "$ANSWER_BACKEND" \\
  --loss-choice-format "$LOSS_CHOICE_FORMAT" \\
  --sub-sample-size 133 \\
  --seed "${BLINK_SEED:-42}" \\
  --results-path "$RESULT_PATH" >> "$LOG_PATH" 2>&1
WORKER

chmod +x "$WORKER_PATH"
tmux new-session -d -s "$SESSION" "bash '$WORKER_PATH'"
echo "$RUN_DIR"
