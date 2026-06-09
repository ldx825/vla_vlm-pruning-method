#!/usr/bin/env bash
set -euo pipefail

# Generic single-GPU VLA-Pruner-OFT run for one LIBERO suite/task/keep setting.
# Intended to be called by table queue scripts.

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
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_10}"
KEEP_VISUAL_TOKENS="${KEEP_VISUAL_TOKENS:-224}"
NUM_TRIALS="${NUM_TRIALS:-20}"
TASK_ID="${TASK_ID:-0}"
SEED="${SEED:-7}"
VISUAL_TOKENS="${VISUAL_TOKENS:-512}"
PRUNING_LAYER="${PRUNING_LAYER:-3}"
TEMPORAL_WINDOW="${TEMPORAL_WINDOW:-3}"
TEMPORAL_GAMMA="${TEMPORAL_GAMMA:-0.8}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
ACTION_ATTENTION_LAYER_START="${ACTION_ATTENTION_LAYER_START:-half}"

case "${TASK_SUITE_NAME}" in
  libero_spatial)
    DEFAULT_CHECKPOINT="/mnt/jarvis/migration/.cache/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-spatial/snapshots/6d0231af0e48c5985f1ff86908f4674b84bc049b"
    ;;
  libero_object)
    DEFAULT_CHECKPOINT="/mnt/jarvis/migration/home/jarvis/Auto_openvla-oft/work/hf/transformers/models--moojink--openvla-7b-oft-finetuned-libero-object/snapshots/4c89574e1c538b6c102f43f0526d60a9d3650148"
    ;;
  libero_goal)
    DEFAULT_CHECKPOINT="/mnt/jarvis/migration/.cache/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-goal/snapshots/c2d0f9fbbd82674683b397ff923168a12f6a307b"
    ;;
  libero_10|libero_long)
    TASK_SUITE_NAME="libero_10"
    DEFAULT_CHECKPOINT="/mnt/jarvis/migration/.cache/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-10/snapshots/95220f9a3421a7ff12d4218e73d09ade830fa9a3"
    ;;
  *)
    echo "Unsupported TASK_SUITE_NAME=${TASK_SUITE_NAME}" >&2
    exit 2
    ;;
esac

CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"
RUN_NAME="${RUN_NAME:-vlapruner_oft_${TASK_SUITE_NAME}_task${TASK_ID}_keep${KEEP_VISUAL_TOKENS}_${NUM_TRIALS}trial_gpu${GPU}_table}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-./experiments/logs/${RUN_NAME}}"
mkdir -p "${LOCAL_LOG_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" run_libero_eval.py \
  --pretrained_checkpoint "${CHECKPOINT}" \
  --task_suite_name "${TASK_SUITE_NAME}" \
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
  --run_id_note "vlapruner_oft_${TASK_SUITE_NAME}_task${TASK_ID}_keep${KEEP_VISUAL_TOKENS}_K${PRUNING_LAYER}" \
  --local_log_dir "${LOCAL_LOG_DIR}" \
  2>&1 | tee -a "${LOCAL_LOG_DIR}/run.log"
