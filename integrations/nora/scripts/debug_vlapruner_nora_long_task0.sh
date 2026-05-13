#!/usr/bin/env bash

set -euo pipefail

# Attach with the existing VSCode configuration:
#   "Attach NORA VLA-Pruner :5678"
# This wrapper pauses before run_libero_eval_nora.py starts executing.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cd "$SCRIPT_DIR"

GPU=${GPU:-0} \
TASK_SUITE=${TASK_SUITE:-libero_10} \
KEEP_VISUAL_TOKENS=${KEEP_VISUAL_TOKENS:-224} \
NUM_TRIALS=${NUM_TRIALS:-1} \
TASK_START_ID=${TASK_START_ID:-0} \
NUM_TASKS_TO_RUN=${NUM_TASKS_TO_RUN:-1} \
SEED=${SEED:-7} \
RUN_NAME=${RUN_NAME:-debug_vlapruner_nora_libero_10_task0_keep${KEEP_VISUAL_TOKENS}_gpu${GPU}} \
DEBUGPY_HOST=${DEBUGPY_HOST:-0.0.0.0} \
DEBUGPY_PORT=${DEBUGPY_PORT:-5678} \
DEBUGPY_WAIT_FOR_CLIENT=${DEBUGPY_WAIT_FOR_CLIENT:-1} \
bash ./run_vlapruner_nora_libero.sh
