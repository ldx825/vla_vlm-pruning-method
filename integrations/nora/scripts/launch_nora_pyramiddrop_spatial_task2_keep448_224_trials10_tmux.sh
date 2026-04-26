#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SESSION=${SESSION:-nora_pdrop_spatial_task2_t10_$(date +%Y%m%d_%H%M%S)}
PYTHON_BIN=${NORA_PYTHON_BIN:-/home/liudongxu/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10}
KNOWN_RUN_ROOT=${NORA_KNOWN_RUN_ROOT:-/home/liudongxu/vla_vlm-pruning-method}
RESULT_ROOT=${RESULT_ROOT:-$ROOT_DIR/experiments/logs/pyramiddrop_spatial_task2_trials10_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$RESULT_ROOT"

launch_one() {
    local keep=$1
    local gpu=$2
    local window="keep${keep}_gpu${gpu}"
    local run_tag="pyramiddrop_nora_dual_libero_spatial_task2_keep${keep}_trials10"
    local log_dir="$RESULT_ROOT/keep${keep}"
    local libero_config_dir="$log_dir/libero_config"
    local shell_cmd

    mkdir -p "$log_dir" "$libero_config_dir" "$log_dir/matplotlib"
    cat > "$libero_config_dir/config.yaml" <<EOF
benchmark_root: $KNOWN_RUN_ROOT/ldx/LIBERO/libero/libero
bddl_files: $KNOWN_RUN_ROOT/ldx/LIBERO/libero/libero/bddl_files
init_states: $KNOWN_RUN_ROOT/ldx/LIBERO/libero/libero/init_files
datasets: $KNOWN_RUN_ROOT/ldx/LIBERO/libero/datasets
assets: $KNOWN_RUN_ROOT/ldx/LIBERO/libero/libero/assets
EOF

    shell_cmd=$(printf '%q ' \
        env \
        HOME=/home/liudongxu \
        LIBERO_CONFIG_PATH="$libero_config_dir" \
        HF_HOME="${HF_HOME:-/mnt/jarvis/.hf}" \
        HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/mnt/jarvis/.hf/hub}" \
        TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/jarvis/.hf/hub}" \
        HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
        HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
        HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
        HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}" \
        MPLCONFIGDIR="$log_dir/matplotlib" \
        WANDB_MODE=disabled \
        TOKENIZERS_PARALLELISM=false \
        USE_TF=0 \
        USE_FLAX=0 \
        TRANSFORMERS_NO_TF=1 \
        TRANSFORMERS_NO_FLAX=1 \
        PYOPENGL_PLATFORM=egl \
        MUJOCO_GL=egl \
        CUDA_VISIBLE_DEVICES="$gpu" \
        MUJOCO_EGL_DEVICE_ID="$gpu" \
        PYTHONPATH="/tmp/nora_transformers_nodeps:/tmp/nora_eval_deps:$KNOWN_RUN_ROOT/Mantis/.venv/lib/python3.10/site-packages:$ROOT_DIR/experiments/libero:$ROOT_DIR:$KNOWN_RUN_ROOT/ldx/LIBERO:${PYTHONPATH:-}" \
        "$PYTHON_BIN" "$ROOT_DIR/experiments/libero/run_libero_eval_nora.py" \
        --model_family nora \
        --model_path /tmp/nora_local_models_20260425/spatial \
        --full_model_state_path /mnt/jarvis/nora_runs/libero_spatial_896_resume_direct_20260323_0948/steps_4750/steps_4750/model.safetensors \
        --lora_adapter_path /mnt/jarvis/nora_runs/libero_spatial_896_resume_direct_20260323_0948/steps_4750/steps_4750/adapter \
        --task_suite_name libero_spatial \
        --task_start_id 2 \
        --num_tasks_to_run 1 \
        --num_trials_per_task 10 \
        --trial_offset 0 \
        --action_horizon 5 \
        --image_resize_resolution 448,448 \
        --env_camera_resolution 448 \
        --use_wrist_image true \
        --dual_view_mode separate \
        --use_anyres false \
        --force_eager_attention true \
        --enable_stage_profiling true \
        --enable_autoprune false \
        --enable_compare_pruning true \
        --compare_pruning_method pyramiddrop \
        --compare_visual_tokens "$keep" \
        --compare_num_views 2 \
        --compare_debug true \
        --run_id_note "$run_tag" \
        --local_log_dir "$log_dir" \
        --latency_csv_filename "${run_tag}_latency.csv" \
        --use_wandb false \
        --seed 7)

    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux new-window -t "$SESSION" -n "$window" "cd '$ROOT_DIR' && $shell_cmd 2>&1 | tee '$log_dir/stdout.log'"
    else
        tmux new-session -d -s "$SESSION" -n "$window" "cd '$ROOT_DIR' && $shell_cmd 2>&1 | tee '$log_dir/stdout.log'"
    fi
}

launch_one 448 7
launch_one 224 6

echo "SESSION=$SESSION"
echo "RESULT_ROOT=$RESULT_ROOT"
tmux list-windows -t "$SESSION"
