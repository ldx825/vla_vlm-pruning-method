#!/usr/bin/env bash
set -euo pipefail

# Debug VLA-ADP-OFT on LIBERO-Long task0.
# Attach from VS Code with a debugpy attach config on localhost:5678.

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-${PWD}/.uvenv/bin/python}"
OPENVLA_OFT_SRC="${OPENVLA_OFT_SRC:-/home/liudongxu/vla_vlm-pruning-method/ldx/openvla-oft-main}"
LIBERO_SRC="${LIBERO_SRC:-${PWD}/LIBERO/LIBERO}"
VENV_SITE="${VENV_SITE:-${PWD}/.uvenv/lib/python3.10/site-packages}"
EXTRA_SITE="${EXTRA_SITE:-/home/liudongxu/venvs/dynamicvla-openvla-oft-copy/lib/python3.10/site-packages}"
export PYTHONPATH="${PWD}:${OPENVLA_OFT_SRC}:${LIBERO_SRC}:${VENV_SITE}:${EXTRA_SITE}:${PYTHONPATH:-}"
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

GPU="${GPU:-0}"
DEBUG_PORT="${DEBUG_PORT:-5678}"
CHECKPOINT="${CHECKPOINT:-/mnt/jarvis/migration/.cache/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-10/snapshots/95220f9a3421a7ff12d4218e73d09ade830fa9a3}"
KEEP_VISUAL_TOKENS="${KEEP_VISUAL_TOKENS:-224}"
NUM_TRIALS="${NUM_TRIALS:-1}"
TASK_ID="${TASK_ID:-0}"
SEED="${SEED:-7}"
VISUAL_TOKENS="${VISUAL_TOKENS:-512}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-./experiments/logs/debug_vlaadp_oft_long_task${TASK_ID}_keep${KEEP_VISUAL_TOKENS}_gpu${GPU}}"
mkdir -p "${LOCAL_LOG_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m debugpy \
  --listen "${DEBUG_PORT}" \
  --wait-for-client \
  run_libero_eval.py \
  --pretrained_checkpoint "${CHECKPOINT}" \
  --task_suite_name libero_10 \
  --task_id "${TASK_ID}" \
  --num_trials_per_task "${NUM_TRIALS}" \
  --seed "${SEED}" \
  --num_images_in_input 2 \
  --use_vlaadp_oft True \
  --visual_tokens "${VISUAL_TOKENS}" \
  --vlaadp_keep_visual_tokens "${KEEP_VISUAL_TOKENS}" \
  --vlaadp_qk_layer 0 \
  --vlaadp_main_view_weight 0.4 \
  --vlaadp_wrist_view_weight 0.6 \
  --vlaadp_use_dynamic_visual_strategy True \
  --vlaadp_decision_method adjacent \
  --vlaadp_adjacent_extrema_window 3 \
  --vlaadp_adjacent_last_state True \
  --vlaadp_cold_start_windows 2 \
  --vlaadp_limit_max_consecutive_pruned 3 \
  --run_id_note "debug_vlaadp_oft_long_task${TASK_ID}_keep${KEEP_VISUAL_TOKENS}" \
  --local_log_dir "${LOCAL_LOG_DIR}"
