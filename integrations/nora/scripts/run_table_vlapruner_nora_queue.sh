#!/usr/bin/env bash
set -euo pipefail

# Queue runner for the VLA-Pruner-Nora table.
# One process owns one GPU and runs its assigned suites sequentially.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/../.."

GPU="${GPU:?Set GPU for this queue worker.}"
QUEUE_SUITES="${QUEUE_SUITES:-libero_10 libero_object}"
KEEP_LIST="${KEEP_LIST:-448 336 224 143}"
NUM_TRIALS="${NUM_TRIALS:-50}"
TASK_START_ID="${TASK_START_ID:-0}"
NUM_TASKS_TO_RUN="${NUM_TASKS_TO_RUN:-10}"
TRIAL_OFFSET="${TRIAL_OFFSET:-0}"
SEED="${SEED:-7}"
TABLE_TAG="${TABLE_TAG:-tables_20260513}"

echo "[queue] method=vlapruner_nora gpu=${GPU} suites=${QUEUE_SUITES} task_start=${TASK_START_ID} num_tasks=${NUM_TASKS_TO_RUN} keeps=${KEEP_LIST} trials=${NUM_TRIALS}"

for suite in ${QUEUE_SUITES}; do
  for keep in ${KEEP_LIST}; do
    run_name="table_${TABLE_TAG}_vlapruner_nora_${suite}_task${TASK_START_ID}_keep${keep}_${NUM_TRIALS}trial_gpu${GPU}"
    echo "[run-start] ${run_name}"
    GPU="${GPU}" \
    TASK_SUITE="${suite}" \
    KEEP_VISUAL_TOKENS="${keep}" \
    NUM_TRIALS="${NUM_TRIALS}" \
    TASK_START_ID="${TASK_START_ID}" \
    NUM_TASKS_TO_RUN="${NUM_TASKS_TO_RUN}" \
    TRIAL_OFFSET="${TRIAL_OFFSET}" \
    SEED="${SEED}" \
    RUN_NAME="${run_name}" \
    bash experiments/libero/run_vlapruner_nora_libero.sh
    echo "[run-done] ${run_name}"
  done
done

echo "[queue-done] method=vlapruner_nora gpu=${GPU}"
