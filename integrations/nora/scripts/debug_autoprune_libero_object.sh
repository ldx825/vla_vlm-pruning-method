#!/usr/bin/env bash

set -euo pipefail

# 新增：指定使用GPU 0
export CUDA_VISIBLE_DEVICES=7

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
set +u
source "$ROOT_DIR/activate_nora_libero.sh"
set -u

GPU_STACK_CHECK_MODE=${NORA_GPU_STACK_CHECK_MODE:-require}
if [[ "$GPU_STACK_CHECK_MODE" != "skip" ]]; then
    if ! "$ROOT_DIR/tools/check_gpu_stack.sh" require; then
        echo >&2
        echo "Refusing to launch debug eval because the GPU stack is unhealthy." >&2
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
echo "Debug port: $DEBUG_PORT"
echo "Model path: $MODEL_PATH"
echo "LoRA adapter: $LORA_ADAPTER_PATH"
echo "Full model state: $FULL_MODEL_STATE_PATH"
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
    --enable_autoprune "${NORA_ENABLE_AUTOPRUNE:-true}" \
    --autoprune_preset "${NORA_AUTOPRUNE_PRESET:-off}" \
    --autoprune_visual_tokens "${NORA_AUTOPRUNE_VISUAL_TOKENS:-512}" \
    --autoprune_start_layer "${NORA_AUTOPRUNE_START_LAYER:-2}" \
    --autoprune_num_views "${NORA_AUTOPRUNE_NUM_VIEWS:-2}" \
    --autoprune_fixed_total_budget "${NORA_AUTOPRUNE_FIXED_TOTAL_BUDGET:-128}" \
    --autoprune_target_token_num "${NORA_AUTOPRUNE_TARGET_TOKEN_NUM:-128}" \
    --autoprune_min_keep_per_view "${NORA_AUTOPRUNE_MIN_KEEP_PER_VIEW:-24}" \
    --autoprune_budget_mode "${NORA_AUTOPRUNE_BUDGET_MODE:-absolute}" \
    --autoprune_budget_semantics "${NORA_AUTOPRUNE_BUDGET_SEMANTICS:-final_cap}" \
    --autoprune_enable_adaptive_layer_policy "${NORA_AUTOPRUNE_ENABLE_ADAPTIVE_LAYER_POLICY:-true}" \
    --autoprune_enable_oneshot_fallback "${NORA_AUTOPRUNE_ENABLE_ONESHOT_FALLBACK:-true}" \
    --autoprune_monitor_only "${NORA_AUTOPRUNE_MONITOR_ONLY:-false}" \
    --autoprune_prefer_flash_attention "${NORA_AUTOPRUNE_PREFER_FLASH_ATTENTION:-false}" \
    --autoprune_manual_attention_probe "${NORA_AUTOPRUNE_MANUAL_ATTENTION_PROBE:-false}" \
    --autoprune_detailed_timing "${NORA_AUTOPRUNE_DETAILED_TIMING:-true}" \
    --autoprune_debug "${NORA_AUTOPRUNE_DEBUG:-true}" \
    --run_id_note "${NORA_RUN_ID_NOTE:-debug-autoprune-object-t01-trials10}" \
    --local_log_dir "${NORA_LOCAL_LOG_DIR:-$ROOT_DIR/experiments/logs}"
