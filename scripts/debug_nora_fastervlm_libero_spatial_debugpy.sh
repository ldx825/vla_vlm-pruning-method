#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/liudongxu/vla_vlm-pruning-method"
PYTHON_BIN="/home/liudongxu/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10"

GPU="${GPU:-0}"
PORT="${PORT:-5678}"
SUITE="libero_spatial"
KEEP="${KEEP:-224}"
TRIALS="${TRIALS:-5}"
NUM_TASKS_TO_RUN="${NUM_TASKS_TO_RUN:-1}"
BASELINE_DEBUG="${BASELINE_DEBUG:-true}"
BASELINE_METHOD="fastervlm"
RUN_TAG="${RUN_TAG:-debug_fastervlm_nora_dual_${SUITE}_keep${KEEP}_trials${TRIALS}}"

cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-/tmp/nora_local_models_20260425/spatial}"
FULL_MODEL_STATE_PATH="${FULL_MODEL_STATE_PATH:-/mnt/jarvis/nora_runs/libero_spatial_896_resume_direct_20260323_0948/steps_4750/steps_4750/model.safetensors}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-/mnt/jarvis/nora_runs/libero_spatial_896_resume_direct_20260323_0948/steps_4750/steps_4750/adapter}"

RESULT_ROOT="$ROOT_DIR/results/nora/libero/$RUN_TAG"
LIBERO_BENCHMARK_ROOT="$ROOT_DIR/ldx/LIBERO/libero/libero"
LIBERO_CONFIG_DIR="$RESULT_ROOT/libero_config"
mkdir -p "$RESULT_ROOT/eval_logs" "$RESULT_ROOT/matplotlib" "$LIBERO_CONFIG_DIR"

cat > "$LIBERO_CONFIG_DIR/config.yaml" <<EOF
benchmark_root: $LIBERO_BENCHMARK_ROOT
bddl_files: $LIBERO_BENCHMARK_ROOT/bddl_files
init_states: $LIBERO_BENCHMARK_ROOT/init_files
datasets: $ROOT_DIR/ldx/LIBERO/libero/datasets
EOF

export HOME="/home/liudongxu"
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_DIR"
export HF_HOME="${HF_HOME:-/mnt/jarvis/.hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/jarvis/.hf/hub}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export MPLCONFIGDIR="$RESULT_ROOT/matplotlib"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_EGL_DEVICE_ID="$GPU"
export PYTHONPATH="/tmp/nora_transformers_nodeps:$ROOT_DIR/Mantis/.venv/lib/python3.10/site-packages:/tmp/nora_eval_deps:$ROOT_DIR/methods/auto_prune/nora/experiments/libero:$ROOT_DIR/models/nora/experiments/libero:$ROOT_DIR/ldx/LIBERO:${PYTHONPATH:-}"

echo "[debug-nora] method=$BASELINE_METHOD suite=$SUITE keep=$KEEP trials=$TRIALS gpu=$GPU port=$PORT"
echo "[debug-nora] model=$MODEL_PATH"
echo "[debug-nora] result_root=$RESULT_ROOT"

"$PYTHON_BIN" -m debugpy --listen "0.0.0.0:$PORT" --wait-for-client \
  methods/auto_prune/nora/experiments/libero/run_libero_eval_nora.py \
  --model_family nora \
  --model_path "$MODEL_PATH" \
  --full_model_state_path "$FULL_MODEL_STATE_PATH" \
  --lora_adapter_path "$LORA_ADAPTER_PATH" \
  --task_suite_name "$SUITE" \
  --num_trials_per_task "$TRIALS" \
  --num_tasks_to_run "$NUM_TASKS_TO_RUN" \
  --image_resize_resolution 448,448 \
  --env_camera_resolution 448 \
  --use_wrist_image true \
  --dual_view_mode separate \
  --action_horizon 5 \
  --enable_baseline_pruning true \
  --baseline_pruning_method "$BASELINE_METHOD" \
  --baseline_target_token_num "$KEEP" \
  --baseline_visual_tokens 512 \
  --baseline_start_layer 2 \
  --baseline_max_pruning_layers 6 \
  --baseline_debug "$BASELINE_DEBUG" \
  --run_id_note "$RUN_TAG" \
  --local_log_dir "$RESULT_ROOT/eval_logs" \
  --latency_csv_filename "${RUN_TAG}_latency.csv" \
  --use_wandb false \
  --seed 7
