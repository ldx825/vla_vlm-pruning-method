#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${NORA_CUDA_VISIBLE_DEVICES:-7}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
set +u
source "$ROOT_DIR/activate_nora_libero.sh"
set -u

GPU_STACK_CHECK_MODE=${NORA_GPU_STACK_CHECK_MODE:-require}
if [[ "$GPU_STACK_CHECK_MODE" != "skip" ]]; then
    if ! "$ROOT_DIR/tools/check_gpu_stack.sh" require; then
        echo >&2
        echo "Refusing to launch FastV debug eval because the GPU stack is unhealthy." >&2
        echo "Set NORA_GPU_STACK_CHECK_MODE=skip to override, but this is not recommended." >&2
        exit 1
    fi
fi

DEBUG_PORT=${NORA_DEBUG_PORT:-5678}
MODEL_PATH=${NORA_MODEL_PATH:-declare-lab/nora-long-finetuned-libero-object}
OBJECT_CKPT_ROOT=${NORA_OBJECT_CKPT_ROOT:-$ROOT_DIR/checkpoints/object_448_best_12000}
LORA_ADAPTER_PATH=${NORA_LORA_ADAPTER:-$OBJECT_CKPT_ROOT/adapter}
FULL_MODEL_STATE_PATH=${NORA_FULL_MODEL_STATE_PATH:-$OBJECT_CKPT_ROOT/model.safetensors}

if [[ ! -d "$LORA_ADAPTER_PATH" ]]; then
    echo "Missing LoRA adapter path: $LORA_ADAPTER_PATH" >&2
    exit 1
fi

if [[ ! -f "$FULL_MODEL_STATE_PATH" ]]; then
    echo "Missing full model state path: $FULL_MODEL_STATE_PATH" >&2
    exit 1
fi

mkdir -p "$ROOT_DIR/experiments/logs"

echo "Debug Python: $(command -v python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Debug port: $DEBUG_PORT"
echo "Model path: $MODEL_PATH"
echo "LoRA adapter: $LORA_ADAPTER_PATH"
echo "Full model state: $FULL_MODEL_STATE_PATH"
echo "FastV keep visual tokens: ${NORA_COMPARE_VISUAL_TOKENS:-128}"
echo "FastV pruning layer: ${NORA_FASTV_PRUNING_LAYER:-2}"
echo "Starting debugpy and waiting for client..."

cd "$ROOT_DIR"

python -m debugpy --listen "$DEBUG_PORT" --wait-for-client \
    "$ROOT_DIR/experiments/libero/run_libero_eval_nora.py" \
    --model_path "$MODEL_PATH" \
    --lora_adapter_path "$LORA_ADAPTER_PATH" \
    --full_model_state_path "$FULL_MODEL_STATE_PATH" \
    --task_suite_name "${NORA_TASK_SUITE:-libero_object}" \
    --num_trials_per_task "${NORA_TRIALS:-10}" \
    --num_tasks_to_run "${NORA_NUM_TASKS_TO_RUN:-2}" \
    --trial_offset "${NORA_TRIAL_OFFSET:-0}" \
    --action_horizon "${NORA_ACTION_HORIZON:-5}" \
    --image_resize_resolution "${NORA_IMAGE_RESIZE:-448,448}" \
    --env_camera_resolution "${NORA_ENV_CAMERA_RESOLUTION:-448}" \
    --use_wrist_image "${NORA_USE_WRIST_IMAGE:-true}" \
    --dual_view_mode "${NORA_DUAL_VIEW_MODE:-separate}" \
    --use_anyres "${NORA_USE_ANYRES:-false}" \
    --force_eager_attention "${NORA_FORCE_EAGER_ATTENTION:-true}" \
    --enable_stage_profiling "${NORA_ENABLE_STAGE_PROFILING:-true}" \
    --enable_autoprune false \
    --enable_compare_pruning true \
    --compare_pruning_method fastv \
    --compare_visual_tokens "${NORA_COMPARE_VISUAL_TOKENS:-128}" \
    --compare_num_views 2 \
    --compare_fastv_pruning_layer "${NORA_FASTV_PRUNING_LAYER:-2}" \
    --compare_detailed_timing "${NORA_COMPARE_DETAILED_TIMING:-true}" \
    --compare_debug "${NORA_COMPARE_DEBUG:-true}" \
    --run_id_note "${NORA_RUN_ID_NOTE:-debug-fastv-object-t01-trials10}" \
    --local_log_dir "${NORA_LOCAL_LOG_DIR:-$ROOT_DIR/experiments/logs}" \
    --latency_csv_filename "${NORA_LATENCY_CSV_FILENAME:-debug_fastv_object_t01_trials10.csv}"
