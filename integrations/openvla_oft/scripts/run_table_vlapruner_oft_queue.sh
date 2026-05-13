#!/usr/bin/env bash
set -euo pipefail

# Queue runner for the VLA-Pruner-OpenVLA-OFT table.
# One process owns one GPU and runs its assigned suites sequentially.

cd "$(dirname "$0")"

GPU="${GPU:?Set GPU for this queue worker.}"
QUEUE_SUITES="${QUEUE_SUITES:-libero_10 libero_object}"
KEEP_LIST="${KEEP_LIST:-448 336 224 143}"
NUM_TRIALS="${NUM_TRIALS:-50}"
TASK_IDS="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
SEED="${SEED:-7}"
TABLE_TAG="${TABLE_TAG:-tables_20260513}"

echo "[queue] method=vlapruner_oft gpu=${GPU} suites=${QUEUE_SUITES} tasks=${TASK_IDS} keeps=${KEEP_LIST} trials=${NUM_TRIALS}"

for suite in ${QUEUE_SUITES}; do
  for task_id in ${TASK_IDS}; do
    for keep in ${KEEP_LIST}; do
      run_name="table_${TABLE_TAG}_vlapruner_oft_${suite}_task${task_id}_keep${keep}_${NUM_TRIALS}trial_gpu${GPU}"
      echo "[run-start] ${run_name}"
      GPU="${GPU}" \
      TASK_SUITE_NAME="${suite}" \
      KEEP_VISUAL_TOKENS="${keep}" \
      NUM_TRIALS="${NUM_TRIALS}" \
      TASK_ID="${task_id}" \
      SEED="${SEED}" \
      RUN_NAME="${run_name}" \
      bash ./run_vlapruner_oft_single.sh
      echo "[run-done] ${run_name}"
    done
  done
done

echo "[queue-done] method=vlapruner_oft gpu=${GPU}"
