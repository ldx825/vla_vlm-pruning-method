#!/usr/bin/env bash
set -euo pipefail

# Start VLA-Pruner-OFT under debugpy, then attach from VSCode to localhost:5678.

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-${PWD}/.uvenv/bin/python}"
OPENVLA_OFT_SRC="${OPENVLA_OFT_SRC:-/home/liudongxu/vla_vlm-pruning-method/ldx/openvla-oft-main}"
LIBERO_SRC="${LIBERO_SRC:-${PWD}/LIBERO/LIBERO}"
VENV_SITE="${VENV_SITE:-${PWD}/.uvenv/lib/python3.10/site-packages}"
DEBUGPY_SITE="${DEBUGPY_SITE:-/home/liudongxu/venvs/dynamicvla-openvla-oft-copy/lib/python3.10/site-packages}"
export PYTHONPATH="${PWD}:${OPENVLA_OFT_SRC}:${LIBERO_SRC}:${VENV_SITE}:${DEBUGPY_SITE}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${PWD}/.hf_home_liudongxu}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${PWD}/.hf_modules_cache_liudongxu}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PWD}/.mplconfig_liudongxu}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${PWD}/.libero_config_liudongxu}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "${HF_HOME}" "${HF_MODULES_CACHE}" "${MPLCONFIGDIR}" "${LIBERO_CONFIG_PATH}"
cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
assets: ${PWD}/LIBERO/LIBERO/libero/libero/assets
bddl_files: ${PWD}/LIBERO/LIBERO/libero/libero/bddl_files
benchmark_root: ${PWD}/LIBERO/LIBERO/libero/libero
datasets: ${PWD}/LIBERO/LIBERO/libero/datasets
init_states: ${PWD}/LIBERO/LIBERO/libero/libero/init_files
EOF

GPU="${GPU:-7}"
DEBUG_PORT="${DEBUG_PORT:-5678}"
CHECKPOINT="${CHECKPOINT:-/mnt/jarvis/migration/home/jarvis/.cache/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-spatial/models--moojink--openvla-7b-oft-finetuned-libero-spatial/snapshots/6d0231af0e48c5985f1ff86908f4674b84bc049b}"
KEEP_VISUAL_TOKENS="${KEEP_VISUAL_TOKENS:-224}"
NUM_TRIALS="${NUM_TRIALS:-10}"
TASK_ID="${TASK_ID:-0}"
SEED="${SEED:-7}"
VISUAL_TOKENS="${VISUAL_TOKENS:-512}"
PRUNING_LAYER="${PRUNING_LAYER:-3}"
TEMPORAL_WINDOW="${TEMPORAL_WINDOW:-3}"
TEMPORAL_GAMMA="${TEMPORAL_GAMMA:-0.8}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
ACTION_ATTENTION_LAYER_START="${ACTION_ATTENTION_LAYER_START:-half}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-./experiments/logs/vlapruner_oft_debug_spatial_task${TASK_ID}_gpu${GPU}}"

mkdir -p "${LOCAL_LOG_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m debugpy --listen "${DEBUG_PORT}" --wait-for-client run_libero_eval.py \
  --pretrained_checkpoint "${CHECKPOINT}" \
  --task_suite_name libero_spatial \
  --task_id "${TASK_ID}" \
  --num_trials_per_task "${NUM_TRIALS}" \
  --seed "${SEED}" \
  --use_vlapruner_oft True \
  --visual_tokens "${VISUAL_TOKENS}" \
  --vlapruner_keep_visual_tokens "${KEEP_VISUAL_TOKENS}" \
  --vlapruner_pruning_layer "${PRUNING_LAYER}" \
  --vlapruner_temporal_window "${TEMPORAL_WINDOW}" \
  --vlapruner_temporal_gamma "${TEMPORAL_GAMMA}" \
  --vlapruner_warmup_steps "${WARMUP_STEPS}" \
  --vlapruner_action_attention_layer_start "${ACTION_ATTENTION_LAYER_START}" \
  --run_id_note "debug_vlapruner_oft_spatial_task${TASK_ID}_keep${KEEP_VISUAL_TOKENS}_K${PRUNING_LAYER}" \
  --local_log_dir "${LOCAL_LOG_DIR}"
