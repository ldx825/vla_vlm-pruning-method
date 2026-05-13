#!/usr/bin/env bash

set -euo pipefail

# VLA-Pruner on dual-view NORA-LONG for LIBERO.
# Example:
#   TASK_SUITE=libero_spatial KEEP_VISUAL_TOKENS=224 NUM_TRIALS=10 TASK_START_ID=0 NUM_TASKS_TO_RUN=1 GPU=0 \
#     bash experiments/libero/run_vlapruner_nora_libero.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

GPU=${GPU:-0}
TASK_SUITE=${TASK_SUITE:-libero_spatial}
KEEP_VISUAL_TOKENS=${KEEP_VISUAL_TOKENS:-224}
NUM_TRIALS=${NUM_TRIALS:-10}
TRIAL_OFFSET=${TRIAL_OFFSET:-0}
TASK_START_ID=${TASK_START_ID:-0}
NUM_TASKS_TO_RUN=${NUM_TASKS_TO_RUN:-1}
SEED=${SEED:-7}

PYTHON_BIN=${NORA_PYTHON_BIN:-/mnt/jarvis/migration/envs/auto_nora.uvenv/bin/python}
LIBERO_ROOT=${NORA_LIBERO_ROOT:-/mnt/jarvis/migration/home/jarvis/Auto_openvla-oft/LIBERO/LIBERO/libero}
HF_HOME_ROOT=${NORA_HF_HOME:-/mnt/jarvis/migration/.cache/nora/.hf}
NORM_STATS_PATH=${NORA_NORM_STATS_PATH:-/mnt/jarvis/.hf/hub/models--declare-lab--nora/snapshots/3f373eff454d299bad4aa0d845a5c0db4fb0e0bc/norm_stats.json}
KNOWN_RUN_ROOT=${NORA_KNOWN_RUN_ROOT:-/home/liudongxu/vla_vlm-pruning-method}

SPATIAL_BASE=/mnt/jarvis/migration/.cache/nora/.hf/hub/models--declare-lab--nora-long-finetuned-libero-spatial/snapshots/ef6b673da0d0278d29b67658bf5764ef7d470029
OBJECT_BASE=/mnt/jarvis/migration/.cache/nora/.hf/hub/models--declare-lab--nora-long-finetuned-libero-object/snapshots/7c205efb04d628b4e9bf507f794d8d2187b5b67a
GOAL_BASE=/mnt/jarvis/migration/.cache/nora/.hf/hub/models--declare-lab--nora-long-finetuned-libero-goal/snapshots/19f25d50c850d91b8fc100c21bf044e35c53c476
LONG_BASE=/mnt/jarvis/.hf/hub/models--declare-lab--nora-long-finetuned-libero-10/snapshots/ddeffe36e738e407673371933b3fe7229e2287f1

SPATIAL_ADAPTER=/mnt/jarvis/migration/home/jarvis/auto_nora/checkpoints/448_dualview_best_12k
OBJECT_STATE=/mnt/jarvis/migration/mnt/jarvis/nora_runs/libero_object_dual_448_stage2/steps_12000/steps_12000/model.safetensors
OBJECT_ADAPTER=/mnt/jarvis/migration/mnt/jarvis/nora_runs/libero_object_dual_448_stage2/steps_12000/steps_12000/adapter
GOAL_STATE=/mnt/jarvis/migration/mnt/jarvis/nora_runs/libero_goal_dual_448_stage2/steps_10000/steps_10000/model.safetensors
GOAL_ADAPTER=/mnt/jarvis/migration/mnt/jarvis/nora_runs/libero_goal_dual_448_stage2/steps_10000/steps_10000/adapter
LONG_STATE=/mnt/jarvis/migration/project_runs/nora_runs/libero10_dual_448_stage2_from8000_20260429_044644/steps_12000/model.safetensors
LONG_ADAPTER=/mnt/jarvis/migration/project_runs/nora_runs/libero10_dual_448_stage2_from8000_20260429_044644/steps_12000/adapter

MODEL_PATH=${NORA_MODEL_PATH:-}
FULL_MODEL_STATE_PATH=${NORA_FULL_MODEL_STATE_PATH:-}
LORA_ADAPTER_PATH=${NORA_LORA_ADAPTER:-}
case "$TASK_SUITE" in
  libero_spatial)
    MODEL_PATH=${MODEL_PATH:-$SPATIAL_BASE}
    LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-$SPATIAL_ADAPTER}
    ;;
  libero_object)
    MODEL_PATH=${MODEL_PATH:-$OBJECT_BASE}
    FULL_MODEL_STATE_PATH=${FULL_MODEL_STATE_PATH:-$OBJECT_STATE}
    LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-$OBJECT_ADAPTER}
    ;;
  libero_goal)
    MODEL_PATH=${MODEL_PATH:-$GOAL_BASE}
    FULL_MODEL_STATE_PATH=${FULL_MODEL_STATE_PATH:-$GOAL_STATE}
    LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-$GOAL_ADAPTER}
    ;;
  libero_10|libero_long)
    TASK_SUITE=libero_10
    MODEL_PATH=${MODEL_PATH:-$LONG_BASE}
    FULL_MODEL_STATE_PATH=${FULL_MODEL_STATE_PATH:-$LONG_STATE}
    LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-$LONG_ADAPTER}
    ;;
  *)
    echo "Unsupported TASK_SUITE=$TASK_SUITE" >&2
    exit 1
    ;;
esac

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing model path: $MODEL_PATH" >&2
  exit 1
fi
if [[ -n "$FULL_MODEL_STATE_PATH" && ! -f "$FULL_MODEL_STATE_PATH" ]]; then
  echo "Missing full model state: $FULL_MODEL_STATE_PATH" >&2
  exit 1
fi
if [[ ! -f "$LORA_ADAPTER_PATH/adapter_model.safetensors" ]]; then
  echo "Missing LoRA adapter under: $LORA_ADAPTER_PATH" >&2
  exit 1
fi

RUN_NAME=${RUN_NAME:-vlapruner_nora_${TASK_SUITE}_task${TASK_START_ID}_keep${KEEP_VISUAL_TOKENS}_${NUM_TRIALS}trial_gpu${GPU}}
LOCAL_LOG_DIR=${LOCAL_LOG_DIR:-$ROOT_DIR/experiments/logs/$RUN_NAME}
LIBERO_CONFIG_DIR=${LIBERO_CONFIG_DIR:-$LOCAL_LOG_DIR/libero_config}
MPLCONFIGDIR=${MPLCONFIGDIR:-$LOCAL_LOG_DIR/matplotlib}
mkdir -p "$LOCAL_LOG_DIR" "$LIBERO_CONFIG_DIR" "$MPLCONFIGDIR"

cat > "$LIBERO_CONFIG_DIR/config.yaml" <<EOF
benchmark_root: $LIBERO_ROOT/libero
bddl_files: $LIBERO_ROOT/libero/bddl_files
init_states: $LIBERO_ROOT/libero/init_files
datasets: $LIBERO_ROOT/datasets
assets: $LIBERO_ROOT/libero/assets
EOF

export HOME=${HOME:-/home/liudongxu}
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_DIR"
export HF_HOME="$HF_HOME_ROOT"
export HUGGINGFACE_HUB_CACHE="$HF_HOME_ROOT/hub"
export TRANSFORMERS_CACHE="$HF_HOME_ROOT/hub"
export HF_MODULES_CACHE="$HF_HOME_ROOT/modules"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_DISABLE_XET=1
export USE_TF=0
export USE_FLAX=0
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export NORA_NORM_STATS_PATH="$NORM_STATS_PATH"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export MPLCONFIGDIR
export PYTHONPATH="/tmp/nora_transformers_nodeps:/tmp/nora_eval_deps:$SCRIPT_DIR:$ROOT_DIR:$KNOWN_RUN_ROOT/ldx/LIBERO:${PYTHONPATH:-}"

PY_CMD=("$PYTHON_BIN")
if [[ -n "${DEBUGPY_PORT:-}" ]]; then
  PY_CMD=("$PYTHON_BIN" -m debugpy --listen "${DEBUGPY_HOST:-0.0.0.0}:${DEBUGPY_PORT}")
  if [[ "${DEBUGPY_WAIT_FOR_CLIENT:-1}" == "1" ]]; then
    PY_CMD+=(--wait-for-client)
  fi
fi

CUDA_VISIBLE_DEVICES="$GPU" "${PY_CMD[@]}" "$SCRIPT_DIR/run_libero_eval_nora.py" \
  --model_family nora \
  --model_path "$MODEL_PATH" \
  --full_model_state_path "$FULL_MODEL_STATE_PATH" \
  --lora_adapter_path "$LORA_ADAPTER_PATH" \
  --task_suite_name "$TASK_SUITE" \
  --num_trials_per_task "$NUM_TRIALS" \
  --trial_offset "$TRIAL_OFFSET" \
  --task_start_id "$TASK_START_ID" \
  --num_tasks_to_run "$NUM_TASKS_TO_RUN" \
  --action_horizon 5 \
  --image_resize_resolution 448,448 \
  --env_camera_resolution 448 \
  --use_wrist_image true \
  --dual_view_mode separate \
  --use_anyres false \
  --force_eager_attention true \
  --enable_stage_profiling true \
  --stop_at_expected_action_length true \
  --enable_autoprune false \
  --enable_compare_pruning false \
  --enable_vlapruner true \
  --vlapruner_keep_visual_tokens "$KEEP_VISUAL_TOKENS" \
  --vlapruner_visual_tokens 512 \
  --vlapruner_num_views 2 \
  --vlapruner_pruning_layer "${VLAPRUNER_PRUNING_LAYER:-3}" \
  --vlapruner_temporal_window "${VLAPRUNER_TEMPORAL_WINDOW:-3}" \
  --vlapruner_temporal_gamma "${VLAPRUNER_TEMPORAL_GAMMA:-0.8}" \
  --vlapruner_warmup_steps "${VLAPRUNER_WARMUP_STEPS:-3}" \
  --vlapruner_action_attention_layer_start "${VLAPRUNER_ACTION_ATTENTION_LAYER_START:-half}" \
  --vlapruner_calibrate_average_keep "${VLAPRUNER_CALIBRATE_AVERAGE_KEEP:-true}" \
  --vlapruner_measured_skip_initial_calls "${VLAPRUNER_MEASURED_SKIP_INITIAL_CALLS:-1}" \
  --vlapruner_average_target_measured_steps "${VLAPRUNER_AVERAGE_TARGET_MEASURED_STEPS:-40}" \
  --run_id_note "$RUN_NAME" \
  --local_log_dir "$LOCAL_LOG_DIR" \
  --latency_csv_filename "${RUN_NAME}_latency.csv" \
  --use_wandb false \
  --seed "$SEED" \
  2>&1 | tee -a "$LOCAL_LOG_DIR/stdout.log"
