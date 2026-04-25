#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/liudongxu/vla_vlm-pruning-method"
PYTHON_BIN="/home/liudongxu/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10"
RUN_TAG="${1:-20260425_pyramiddrop_nora_dual_libero4suite_keep448_336_224_143_trials3}"
SESSION="${SESSION:-nora_pdrop_t3_20260425}"

cd "$ROOT_DIR"

RESULT_ROOT="$ROOT_DIR/results/nora/libero/$RUN_TAG"
LOG_ROOT="$RESULT_ROOT/launcher_logs"
mkdir -p "$LOG_ROOT" "$RESULT_ROOT/eval_logs" "$RESULT_ROOT/hf_cache" "$RESULT_ROOT/matplotlib"

export HOME="/home/liudongxu"
export HF_HOME="${HF_HOME:-/mnt/jarvis/.hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/jarvis/.hf/hub}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export MPLCONFIGDIR="$RESULT_ROOT/matplotlib"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/tmp/nora_transformers_nodeps:$ROOT_DIR/Mantis/.venv/lib/python3.10/site-packages:/tmp/nora_eval_deps:$ROOT_DIR/methods/auto_prune/nora/experiments/libero:$ROOT_DIR/models/nora/experiments/libero:$ROOT_DIR/ldx/LIBERO:${PYTHONPATH:-}"

model_for_suite() {
  case "$1" in
    libero_object) echo "/tmp/nora_local_models_20260425/object" ;;
    libero_goal) echo "/tmp/nora_local_models_20260425/goal" ;;
    libero_spatial) echo "/tmp/nora_local_models_20260425/spatial" ;;
    libero_10) echo "declare-lab/nora-long-finetuned-libero-10" ;;
    *) echo "Unknown suite: $1" >&2; return 1 ;;
  esac
}

full_state_for_suite() {
  case "$1" in
    libero_object) echo "/mnt/jarvis/nora_runs/libero_object_dual_448_stage2/steps_12000/steps_12000/model.safetensors" ;;
    libero_goal) echo "/mnt/jarvis/nora_runs/libero_goal_dual_448_stage2/steps_10000/steps_10000/model.safetensors" ;;
    libero_spatial) echo "/mnt/jarvis/nora_runs/libero_spatial_896_resume_direct_20260323_0948/steps_4750/steps_4750/model.safetensors" ;;
    libero_10) echo "" ;;
    *) echo "Unknown suite: $1" >&2; return 1 ;;
  esac
}

adapter_for_suite() {
  case "$1" in
    libero_object) echo "/mnt/jarvis/nora_runs/libero_object_dual_448_stage2/steps_12000/steps_12000/adapter" ;;
    libero_goal) echo "/mnt/jarvis/nora_runs/libero_goal_dual_448_stage2/steps_10000/steps_10000/adapter" ;;
    libero_spatial) echo "/mnt/jarvis/nora_runs/libero_spatial_896_resume_direct_20260323_0948/steps_4750/steps_4750/adapter" ;;
    libero_10) echo "" ;;
    *) echo "Unknown suite: $1" >&2; return 1 ;;
  esac
}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

KEEPS=(448 336 224 143)
SUITES=(libero_object libero_goal libero_spatial libero_10)

tmux new-session -d -s "$SESSION" -n "launcher" "cd '$ROOT_DIR'; bash"

job_idx=0
for keep in "${KEEPS[@]}"; do
  for suite in "${SUITES[@]}"; do
    gpu=$((job_idx / 2))
    model_path="$(model_for_suite "$suite")"
    full_state_path="$(full_state_for_suite "$suite")"
    adapter_path="$(adapter_for_suite "$suite")"
    model_state_args=""
    if [ -n "$full_state_path" ]; then
      model_state_args="$model_state_args --full_model_state_path '$full_state_path'"
    fi
    if [ -n "$adapter_path" ]; then
      model_state_args="$model_state_args --lora_adapter_path '$adapter_path'"
    fi
    name="gpu${gpu}_keep${keep}_${suite}"
    stdout_log="$LOG_ROOT/${name}.out"
    latency_csv="${name}_latency.csv"
    run_note="pyramiddrop_nora_dual_keep${keep}_trials3_${suite}_${RUN_TAG}"

    cmd="cd '$ROOT_DIR'; \
export HOME='$HOME' HF_HOME='$HF_HOME' TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE' HF_ENDPOINT='$HF_ENDPOINT' HF_HUB_DOWNLOAD_TIMEOUT='$HF_HUB_DOWNLOAD_TIMEOUT' HF_HUB_ETAG_TIMEOUT='$HF_HUB_ETAG_TIMEOUT' MPLCONFIGDIR='$MPLCONFIGDIR' WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONPATH='$PYTHONPATH' CUDA_VISIBLE_DEVICES='$gpu' MUJOCO_EGL_DEVICE_ID='$gpu'; \
echo '[start]' \$(date -Is) '$name' 'gpu='$gpu 'model=$model_path' | tee '$stdout_log'; \
'$PYTHON_BIN' methods/auto_prune/nora/experiments/libero/run_libero_eval_nora.py \
  --model_family nora \
  --model_path '$model_path' \
  $model_state_args \
  --task_suite_name '$suite' \
  --num_trials_per_task 3 \
  --num_tasks_to_run 0 \
  --image_resize_resolution 448,448 \
  --env_camera_resolution 448 \
  --use_wrist_image true \
  --dual_view_mode separate \
  --action_horizon 5 \
  --enable_baseline_pruning true \
  --baseline_pruning_method pyramiddrop \
  --baseline_target_token_num '$keep' \
  --baseline_visual_tokens 512 \
  --baseline_start_layer 2 \
  --baseline_max_pruning_layers 6 \
  --run_id_note '$run_note' \
  --local_log_dir '$RESULT_ROOT/eval_logs' \
  --latency_csv_filename '$latency_csv' \
  --use_wandb false \
  --seed 7 >> '$stdout_log' 2>&1; \
status=\$?; echo '[exit]' \$(date -Is) '$name' 'status='\$status | tee -a '$stdout_log'; exit \$status"

    if [ "$job_idx" -eq 0 ]; then
      tmux rename-window -t "$SESSION:0" "$name"
      tmux send-keys -t "$SESSION:0" "$cmd" C-m
    else
      tmux new-window -t "$SESSION" -n "$name" "$cmd"
    fi
    job_idx=$((job_idx + 1))
  done
done

cat > "$RESULT_ROOT/RUN_INFO.txt" <<EOF
Session: $SESSION
Run tag: $RUN_TAG
Result root: $RESULT_ROOT
Eval logs: $RESULT_ROOT/eval_logs
Launcher logs: $LOG_ROOT
Attach: tmux attach -t $SESSION
Watch logs: tail -f $LOG_ROOT/gpu0_keep448_libero_object.out
Jobs: ${#KEEPS[@]} keeps x ${#SUITES[@]} suites = $job_idx
GPU assignment: job_idx / 2, so each GPU has two jobs.
EOF

tmux ls
tmux list-windows -t "$SESSION"
echo "Launched $job_idx jobs in tmux session: $SESSION"
echo "Results will be under: $RESULT_ROOT"
