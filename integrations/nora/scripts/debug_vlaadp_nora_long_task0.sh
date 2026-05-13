#!/usr/bin/env bash

set -euo pipefail

# Debug VLA-ADP on dual-view NORA-LONG / LIBERO-Long task0.
# Attach from VS Code with a debugpy attach config at localhost:5678.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export TASK_SUITE=${TASK_SUITE:-libero_10}
export KEEP_VISUAL_TOKENS=${KEEP_VISUAL_TOKENS:-224}
export NUM_TRIALS=${NUM_TRIALS:-1}
export TASK_START_ID=${TASK_START_ID:-0}
export NUM_TASKS_TO_RUN=${NUM_TASKS_TO_RUN:-1}
export GPU=${GPU:-0}
export RUN_NAME=${RUN_NAME:-debug_vlaadp_nora_long_task0_keep${KEEP_VISUAL_TOKENS}_gpu${GPU}}

export DEBUGPY_HOST=${DEBUGPY_HOST:-0.0.0.0}
export DEBUGPY_PORT=${DEBUGPY_PORT:-5678}
export DEBUGPY_WAIT_FOR_CLIENT=${DEBUGPY_WAIT_FOR_CLIENT:-1}
export VLAADP_DEBUG=${VLAADP_DEBUG:-true}

bash "$SCRIPT_DIR/run_vlaadp_nora_libero.sh"
