#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANSWER_BACKEND="${BLINK_ANSWER_BACKEND:-loss}"
LOSS_CHOICE_FORMAT="${BLINK_LOSS_CHOICE_FORMAT:-plain_text}"
RUN_ID="${1:-$(date -u +%Y%m%d_%H%M%S)_fastv_mllava_blink_multiview_full133_${ANSWER_BACKEND}_rerun}"
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

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/results" "$RUN_DIR/pids"
mkdir -p "$RUN_DIR/workers"

KEEP_VALUES=(448 336 224 143 128 64 48 32)
GPU_VALUES=(0 1 2 3 4 5 6 7)

cat > "$RUN_DIR/launch_manifest.tsv" <<MANIFEST
keep	gpu	session	log	result
MANIFEST

for i in "${!KEEP_VALUES[@]}"; do
  keep="${KEEP_VALUES[$i]}"
  gpu="${GPU_VALUES[$i]}"
  session="fastv_gen_${RUN_ID}_k${keep}_g${gpu}"
  log_path="$RUN_DIR/logs/fastv${keep}.log"
  result_path="$RUN_DIR/results/fastv${keep}.json"
  pid_path="$RUN_DIR/pids/fastv${keep}.pid"
  worker_path="$RUN_DIR/workers/fastv${keep}.sh"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "session already exists: $session" >&2
    exit 1
  fi

  printf "%s\t%s\t%s\t%s\t%s\n" "$keep" "$gpu" "$session" "$log_path" "$result_path" >> "$RUN_DIR/launch_manifest.tsv"

  cat > "$worker_path" <<WORKER
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
export CUDA_VISIBLE_DEVICES="$gpu"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT_DIR:$MANTIS_ROOT:$MANTIS_ROOT/mantis:\${PYTHONPATH:-}"
export MANTIS_BASELINE_PRUNE_METHOD=fastv
export MANTIS_BASELINE_PRUNE_ENABLED=1
export MANTIS_BASELINE_PRUNE_TARGET_TOKEN_NUM="$keep"
export MANTIS_BASELINE_PRUNE_FASTV_PRUNING_LAYER="${MANTIS_BASELINE_PRUNE_FASTV_PRUNING_LAYER:-2}"
export MANTIS_BASELINE_PRUNE_FASTV_QUERY_MODE="${MANTIS_BASELINE_PRUNE_FASTV_QUERY_MODE:-all_nonvisual}"
export MANTIS_BASELINE_PRUNE_FASTV_VISUAL_POOL="${MANTIS_BASELINE_PRUNE_FASTV_VISUAL_POOL:-unified}"
export MANTIS_BASELINE_PRUNE_DEBUG="${MANTIS_BASELINE_PRUNE_DEBUG:-0}"
echo \$\$ > "$pid_path"
echo "[launcher] session=$session gpu=$gpu keep=$keep result=$result_path" > "$log_path"
echo "[launcher] started_utc=\$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log_path"
exec "$PYTHON_BIN" "$ROOT_DIR/scripts/run_blink_fastv_baseline.py" \\
  --model-name "${MANTIS_MODEL_NAME:-mantis-8b-siglip-llama3-ablation1}" \\
  --task-name "${BLINK_TASK_NAME:-Multi-view_Reasoning}" \\
  --split "${BLINK_SPLIT:-val}" \\
  --eval-mode accuracy \\
  --answer-backend "$ANSWER_BACKEND" \\
  --loss-choice-format "$LOSS_CHOICE_FORMAT" \\
  --sub-sample-size 133 \\
  --seed "${BLINK_SEED:-42}" \\
  --results-path "$result_path" >> "$log_path" 2>&1
WORKER
  chmod +x "$worker_path"
  tmux new-session -d -s "$session" "bash '$worker_path'"
done

echo "$RUN_DIR"
