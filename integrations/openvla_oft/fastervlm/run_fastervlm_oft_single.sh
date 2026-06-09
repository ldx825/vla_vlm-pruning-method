#!/usr/bin/env bash
set -euo pipefail

INTEGRATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${OPENVLA_OFT_RUNTIME_ROOT:-/home/liudongxu/Auto_openvla-oft}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/jarvis/migration/home/jarvis/Auto_openvla-oft_liudongxu/.uvenv/bin/python}"

GPU="${GPU:-1}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_spatial}"
TASK_ID="${TASK_ID:-0}"
KEEP_VISUAL_TOKENS="${KEEP_VISUAL_TOKENS:-128}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-7}"
# Use the runtime with the same verified OpenVLA-OFT inference implementation
# as the Object/Goal/Long checkpoints. The legacy spatial runtime contains an
# older locally modified modeling_prismatic.py.
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${ROOT}/checkpoints/fastervlm-runtime-spatial-official}"
RUN_TAG="${RUN_TAG:-fastervlm_${TASK_SUITE_NAME}_task${TASK_ID}_keep${KEEP_VISUAL_TOKENS}}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${ROOT}/experiments/logs/${RUN_TAG}}"

export HOME="/home/liudongxu"
# This host's EGL enumeration is shifted by one relative to nvidia-smi.
# Keep GPU as CUDA logical device 0, while allowing an explicit EGL mapping.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_OVERRIDE:-${GPU}}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID_OVERRIDE:-${GPU}}"
export PYOPENGL_PLATFORM="egl"
export MUJOCO_GL="egl"
export TOKENIZERS_PARALLELISM="false"
export USE_TF="0"
export USE_FLAX="0"
export TRANSFORMERS_NO_TF="1"
export TRANSFORMERS_NO_FLAX="1"
export HF_HOME="/home/liudongxu/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export LIBERO_CONFIG_PATH="${ROOT}/experiments/libero_config"
OPENVLA_OFT_SRC="${OPENVLA_OFT_SRC:-/home/liudongxu/vla_vlm-pruning-method/ldx/openvla-oft-main}"
VENV_SITE="${VENV_SITE:-/mnt/jarvis/migration/home/jarvis/Auto_openvla-oft_liudongxu/.uvenv/lib/python3.10/site-packages}"
EXTRA_SITE="${EXTRA_SITE:-/home/liudongxu/venvs/dynamicvla-openvla-oft-copy/lib/python3.10/site-packages}"
export LIBERO_DEPS_SITE=""
export PYTHONPATH="${INTEGRATION_ROOT}/fastervlm:${INTEGRATION_ROOT}/visionzip:${INTEGRATION_ROOT}/pyramiddrop:${INTEGRATION_ROOT}/baseline_libero:${ROOT}:${OPENVLA_OFT_SRC}:${ROOT}/LIBERO/LIBERO:${VENV_SITE}:${EXTRA_SITE}"

mkdir -p "${LOCAL_LOG_DIR}"
cd "${INTEGRATION_ROOT}/baseline_libero"

EXTRA_ARGS=()
if [[ "${USE_COMPARE_PRUNING:-1}" == "1" ]]; then
  EXTRA_ARGS+=(
    --use_compare_pruning True
    --compare_pruning_method fastervlm
    --compare_visual_token_num "${KEEP_VISUAL_TOKENS}"
  )
fi

exec "${PYTHON_BIN}" "${INTEGRATION_ROOT}/baseline_libero/run_libero_eval.py" \
  --model_family openvla \
  --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --task_id "${TASK_ID}" \
  --num_trials_per_task "${NUM_TRIALS}" \
  --num_open_loop_steps 8 \
  --center_crop True \
  --local_log_dir "${LOCAL_LOG_DIR}" \
  --run_id_note "${RUN_TAG}" \
  --seed "${SEED}" \
  --use_wandb False \
  "${EXTRA_ARGS[@]}"
