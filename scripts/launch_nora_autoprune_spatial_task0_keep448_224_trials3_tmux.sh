#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/liudongxu/vla_vlm-pruning-method"
PYTHON_BIN="/home/liudongxu/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10"
RUN_TAG="${1:-20260425_autoprune_nora_dual_libero_spatial_task0_keep448_224_trials3}"
SESSION="${SESSION:-nora_ap_spatial_task0_t3_20260425}"

cd "$ROOT_DIR"

RESULT_ROOT="$ROOT_DIR/results/nora/libero/$RUN_TAG"
LOG_ROOT="$RESULT_ROOT/launcher_logs"
LIBERO_BENCHMARK_ROOT="$ROOT_DIR/ldx/LIBERO/libero/libero"
LIBERO_CONFIG_DIR="$RESULT_ROOT/libero_config"
mkdir -p "$LOG_ROOT" "$RESULT_ROOT/eval_logs" "$RESULT_ROOT/matplotlib" "$LIBERO_CONFIG_DIR"

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
export PYTHONPATH="/tmp/nora_transformers_nodeps:$ROOT_DIR/Mantis/.venv/lib/python3.10/site-packages:/tmp/nora_eval_deps:$ROOT_DIR/methods/auto_prune/nora/experiments/libero:$ROOT_DIR/models/nora/experiments/libero:$ROOT_DIR/ldx/LIBERO:${PYTHONPATH:-}"

SUITE="libero_spatial"
MODEL_PATH="/tmp/nora_local_models_20260425/spatial"
FULL_MODEL_STATE_PATH="/mnt/jarvis/nora_runs/libero_spatial_896_resume_direct_20260323_0948/steps_4750/steps_4750/model.safetensors"
LORA_ADAPTER_PATH="/mnt/jarvis/nora_runs/libero_spatial_896_resume_direct_20260323_0948/steps_4750/steps_4750/adapter"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

KEEPS=(448 224)
GPUS=(${GPUS:-0 1})

if [ "${#GPUS[@]}" -lt "${#KEEPS[@]}" ]; then
  echo "Need at least ${#KEEPS[@]} GPUs in GPUS, got: ${GPUS[*]}" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" -n "launcher" "cd '$ROOT_DIR'; bash"
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null

for i in "${!KEEPS[@]}"; do
  keep="${KEEPS[$i]}"
  gpu="${GPUS[$i]}"
  name="gpu${gpu}_autoprune_keep${keep}_${SUITE}_task0_t3"
  stdout_log="$LOG_ROOT/${name}.out"
  latency_csv="${name}_latency.csv"
  run_note="autoprune_nora_dual_keep${keep}_trials3_${SUITE}_task0_${RUN_TAG}"

  cmd="cd '$ROOT_DIR'; \
export HOME='$HOME' LIBERO_CONFIG_PATH='$LIBERO_CONFIG_PATH' HF_HOME='$HF_HOME' TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE' HF_ENDPOINT='$HF_ENDPOINT' HF_HUB_DOWNLOAD_TIMEOUT='$HF_HUB_DOWNLOAD_TIMEOUT' HF_HUB_ETAG_TIMEOUT='$HF_HUB_ETAG_TIMEOUT' MPLCONFIGDIR='$MPLCONFIGDIR' WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYOPENGL_PLATFORM=egl PYTHONPATH='$PYTHONPATH' CUDA_VISIBLE_DEVICES='$gpu' MUJOCO_EGL_DEVICE_ID='$gpu'; \
echo '[start]' \$(date -Is) '$name' 'gpu='$gpu 'keep='$keep 'task0 trials=3' | tee '$stdout_log'; \
'$PYTHON_BIN' methods/auto_prune/nora/experiments/libero/run_libero_eval_nora.py \
  --model_family nora \
  --model_path '$MODEL_PATH' \
  --full_model_state_path '$FULL_MODEL_STATE_PATH' \
  --lora_adapter_path '$LORA_ADAPTER_PATH' \
  --task_suite_name '$SUITE' \
  --num_trials_per_task 3 \
  --num_tasks_to_run 1 \
  --image_resize_resolution 448,448 \
  --env_camera_resolution 448 \
  --use_wrist_image true \
  --dual_view_mode separate \
  --action_horizon 5 \
  --enable_autoprune true \
  --autoprune_visual_tokens 512 \
  --autoprune_num_views 2 \
  --autoprune_budget_mode absolute \
  --autoprune_budget_semantics final_cap \
  --autoprune_fixed_total_budget '$keep' \
  --autoprune_target_token_num '$keep' \
  --run_id_note '$run_note' \
  --local_log_dir '$RESULT_ROOT/eval_logs' \
  --latency_csv_filename '$latency_csv' \
  --use_wandb false \
  --seed 7 >> '$stdout_log' 2>&1; \
status=\$?; echo '[exit]' \$(date -Is) '$name' 'status='\$status | tee -a '$stdout_log'; exit \$status"

  if [ "$i" -eq 0 ]; then
    tmux rename-window -t "$SESSION:0" "$name"
    tmux send-keys -t "$SESSION:0" "$cmd" C-m
  else
    tmux new-window -t "$SESSION" -n "$name" "$cmd"
  fi
done

cat > "$RESULT_ROOT/RUN_INFO.txt" <<EOF
Session: $SESSION
Run tag: $RUN_TAG
Result root: $RESULT_ROOT
Eval logs: $RESULT_ROOT/eval_logs
Launcher logs: $LOG_ROOT
Suite: $SUITE
Task: task0 only (--num_tasks_to_run 1)
Method: autoprune
Trials: 3
K: ${KEEPS[*]}
GPUs: ${GPUS[*]}
Model path: $MODEL_PATH
Full model state: $FULL_MODEL_STATE_PATH
LoRA adapter: $LORA_ADAPTER_PATH
LIBERO config: $LIBERO_CONFIG_DIR/config.yaml
Attach: tmux attach -t $SESSION
List panes: tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} #{pane_pid} #{pane_current_command} #{window_name}'
Watch keep448: tail -f $LOG_ROOT/gpu${GPUS[0]}_autoprune_keep448_${SUITE}_task0_t3.out
Watch keep224: tail -f $LOG_ROOT/gpu${GPUS[1]}_autoprune_keep224_${SUITE}_task0_t3.out
EOF

tmux ls
tmux list-windows -t "$SESSION"
echo "Launched ${#KEEPS[@]} jobs in tmux session: $SESSION"
echo "Results will be under: $RESULT_ROOT"
