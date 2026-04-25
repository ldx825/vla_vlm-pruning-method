#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MANTIS_ROOT="${MANTIS_ROOT:-/mnt/jarvis/migration/Mantis}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/jarvis/migration/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HUGGINGFACE_HUB_CACHE}"
export TMPDIR="${TMPDIR:-/mnt/jarvis/migration/tmp}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT_DIR:$MANTIS_ROOT:$MANTIS_ROOT/mantis:${PYTHONPATH:-}"

export MANTIS_BASELINE_PRUNE_METHOD="${MANTIS_BASELINE_PRUNE_METHOD:-${MANTIS_BASELINE_PRUNING_METHOD:-fastv}}"
export MANTIS_BASELINE_PRUNE_ENABLED="${MANTIS_BASELINE_PRUNE_ENABLED:-${MANTIS_BASELINE_PRUNING_ENABLED:-1}}"
export MANTIS_BASELINE_PRUNE_TARGET_TOKEN_NUM="${MANTIS_BASELINE_PRUNE_TARGET_TOKEN_NUM:-${MANTIS_BASELINE_PRUNING_TARGET_TOKEN_NUM:-32}}"
export MANTIS_BASELINE_PRUNE_FASTV_PRUNING_LAYER="${MANTIS_BASELINE_PRUNE_FASTV_PRUNING_LAYER:-${MANTIS_BASELINE_PRUNING_LAYER:-2}}"
export MANTIS_BASELINE_PRUNE_DEBUG="${MANTIS_BASELINE_PRUNE_DEBUG:-1}"

RUN_DIR="${FASTV_DEBUG_RUN_DIR:-$ROOT_DIR/results/mantis/blink_multiview/fastv/debug}"
mkdir -p "$RUN_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/liudongxu/Mantis/.venv/bin/python}"
DEBUG_HOST="${DEBUG_HOST:-0.0.0.0}"
DEBUG_PORT="${DEBUG_PORT:-5678}"
SAMPLE_SIZE="${BLINK_SUB_SAMPLE_SIZE:-4}"
ANSWER_BACKEND="${BLINK_ANSWER_BACKEND:-generate}"
RESULTS_PATH="${FASTV_DEBUG_RESULTS_PATH:-$RUN_DIR/fastv_keep${MANTIS_BASELINE_PRUNE_TARGET_TOKEN_NUM}_${ANSWER_BACKEND}_debug.json}"

echo "[debug-fastv-mantis] cwd=$ROOT_DIR"
echo "[debug-fastv-mantis] python=$PYTHON_BIN"
echo "[debug-fastv-mantis] cuda=$CUDA_VISIBLE_DEVICES"
echo "[debug-fastv-mantis] method=$MANTIS_BASELINE_PRUNE_METHOD keep=$MANTIS_BASELINE_PRUNE_TARGET_TOKEN_NUM layer=$MANTIS_BASELINE_PRUNE_FASTV_PRUNING_LAYER"
echo "[debug-fastv-mantis] backend=$ANSWER_BACKEND sample_size=$SAMPLE_SIZE"
echo "[debug-fastv-mantis] results=$RESULTS_PATH"
echo "[debug-fastv-mantis] waiting for debug client on $DEBUG_HOST:$DEBUG_PORT"

exec "$PYTHON_BIN" -m debugpy --listen "$DEBUG_HOST:$DEBUG_PORT" --wait-for-client \
  "$ROOT_DIR/scripts/run_blink_fastv_baseline.py" \
  --model-name "${MANTIS_MODEL_NAME:-mantis-8b-siglip-llama3-ablation1}" \
  --task-name "${BLINK_TASK_NAME:-Multi-view_Reasoning}" \
  --split "${BLINK_SPLIT:-val}" \
  --eval-mode "${BLINK_EVAL_MODE:-accuracy}" \
  --answer-backend "$ANSWER_BACKEND" \
  --sub-sample-size "$SAMPLE_SIZE" \
  --seed "${BLINK_SEED:-42}" \
  --results-path "$RESULTS_PATH"
