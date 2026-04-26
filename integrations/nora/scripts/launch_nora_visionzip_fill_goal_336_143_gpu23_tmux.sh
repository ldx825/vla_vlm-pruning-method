#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=/mnt/jarvis/migration/home/jarvis/auto_nora
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
SESSION=${SESSION:-visionzip_nora_fill_goal_gpu23_${RUN_STAMP}}
RUN_ROOT=${RUN_ROOT:-$ROOT_DIR/overnight_launches/visionzip_fill_goal_336_143_${RUN_STAMP}}
WORKER=$ROOT_DIR/tools/run_visionzip_multisuite_worker.sh

mkdir -p "$RUN_ROOT/worker_logs"

cat > "$RUN_ROOT/README.txt" <<EOF
VisionZip-Nora table fill run
Created: $(date '+%F %T')

Purpose:
- Fill missing LIBERO-goal full-suite rows for prune keeps 336 and 143.
- Existing completed/running tasks are intentionally not duplicated.

Assignments:
- GPU 2: libero_goal:336
- GPU 3: libero_goal:143

Notes:
- Proxies are unset because the previous goal 336/143 run failed on a stale 127.0.0.1:7892 proxy.
- HF/Transformers offline mode is enabled; required Nora files are already cached locally.
EOF

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION"
    exit 1
fi

launch_gpu() {
    local gpu=$1
    local assignment=$2
    local window="gpu${gpu}_${assignment//:/_}"
    local logfile="$RUN_ROOT/worker_logs/gpu${gpu}.log"
    local cmd

    cmd=$(printf "%q " \
        env \
        -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
        HF_HOME=/mnt/jarvis/.hf \
        HUGGINGFACE_HUB_CACHE=/mnt/jarvis/.hf/hub \
        TRANSFORMERS_CACHE=/mnt/jarvis/.hf/hub \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        HF_HUB_DISABLE_XET=1 \
        bash "$WORKER" "$gpu" "$RUN_ROOT" "$assignment")

    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux new-window -d -t "$SESSION" -n "$window" "cd '$ROOT_DIR' && $cmd 2>&1 | tee '$logfile'"
    else
        tmux new-session -d -s "$SESSION" -n "$window" "cd '$ROOT_DIR' && $cmd 2>&1 | tee '$logfile'"
    fi
}

launch_gpu 2 libero_goal:336
launch_gpu 3 libero_goal:143

echo "Started tmux session: $SESSION"
echo "Run root: $RUN_ROOT"
echo "Logs:"
echo "  $RUN_ROOT/worker_logs/gpu2.log"
echo "  $RUN_ROOT/worker_logs/gpu3.log"
