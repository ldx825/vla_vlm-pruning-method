#!/usr/bin/env bash
set -euo pipefail

# Launch all four table-filling experiment queues in tmux.
# This script only starts queue workers; each worker runs its assigned suites/keeps sequentially.
#
# GPU allocation:
#   0,1 -> VLA-Pruner-OpenVLA-OFT
#   2,3 -> VLA-Pruner-Nora
#   4,5 -> VLA-ADP-OpenVLA-OFT
#   6,7 -> VLA-ADP-Nora
#
# Defaults are formal table settings: 50 trials and all 10 LIBERO tasks per suite.
# Override NUM_TRIALS/TABLE_TAG/KEEP_LIST/TASK_IDS before launching if needed.

if [[ "$(id -un)" != "liudongxu" ]]; then
  SCRIPT_PATH="$(readlink -f "$0")"
  cmd="cd /mnt/jarvis/migration/home/jarvis && NUM_TRIALS='${NUM_TRIALS:-50}' TASK_IDS='${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}' TASK_START_ID='${TASK_START_ID:-0}' NUM_TASKS_TO_RUN='${NUM_TASKS_TO_RUN:-10}' TRIAL_OFFSET='${TRIAL_OFFSET:-0}' SEED='${SEED:-7}' KEEP_LIST='${KEEP_LIST:-448 336 224 143}' TABLE_TAG='${TABLE_TAG:-tables_20260513}' bash '$SCRIPT_PATH'"
  echo "[reexec] switching to liudongxu for tmux ownership"
  exec su - liudongxu -c "$cmd"
fi

NUM_TRIALS="${NUM_TRIALS:-50}"
TASK_IDS="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
TASK_START_ID="${TASK_START_ID:-0}"
NUM_TASKS_TO_RUN="${NUM_TASKS_TO_RUN:-10}"
TRIAL_OFFSET="${TRIAL_OFFSET:-0}"
SEED="${SEED:-7}"
KEEP_LIST="${KEEP_LIST:-448 336 224 143}"
TABLE_TAG="${TABLE_TAG:-tables_20260513}"

OFT_ROOT="/mnt/jarvis/migration/home/jarvis/Auto_openvla-oft_liudongxu"
NORA_ROOT="/mnt/jarvis/migration/home/jarvis/auto_nora"
LOG_ROOT="/mnt/jarvis/migration/home/jarvis/pruning_table_tmux_logs/${TABLE_TAG}"
mkdir -p "$LOG_ROOT"

launch_session() {
  local session="$1"
  local workdir="$2"
  local command="$3"
  local log_file="$4"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[skip] tmux session already exists: $session"
    return 0
  fi

  tmux new-session -d -s "$session" "cd '$workdir' && $command 2>&1 | tee -a '$log_file'"
  sleep 1
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[launched] $session -> $log_file"
  else
    echo "[error] failed to launch tmux session: $session" >&2
    exit 1
  fi
}

COMMON_ENV="NUM_TRIALS='$NUM_TRIALS' KEEP_LIST='$KEEP_LIST' SEED='$SEED' TABLE_TAG='$TABLE_TAG'"
OFT_TASK_ENV="TASK_IDS='$TASK_IDS'"
NORA_TASK_ENV="TASK_START_ID='$TASK_START_ID' NUM_TASKS_TO_RUN='$NUM_TASKS_TO_RUN' TRIAL_OFFSET='$TRIAL_OFFSET'"

launch_session \
  "tbl_vlapruner_oft_g0_${TABLE_TAG}" \
  "$OFT_ROOT" \
  "$COMMON_ENV $OFT_TASK_ENV GPU='0' QUEUE_SUITES='libero_10 libero_object' bash ./run_table_vlapruner_oft_queue.sh" \
  "$LOG_ROOT/tbl_vlapruner_oft_g0.log"

launch_session \
  "tbl_vlapruner_oft_g1_${TABLE_TAG}" \
  "$OFT_ROOT" \
  "$COMMON_ENV $OFT_TASK_ENV GPU='1' QUEUE_SUITES='libero_goal libero_spatial' bash ./run_table_vlapruner_oft_queue.sh" \
  "$LOG_ROOT/tbl_vlapruner_oft_g1.log"

launch_session \
  "tbl_vlapruner_nora_g2_${TABLE_TAG}" \
  "$NORA_ROOT" \
  "$COMMON_ENV $NORA_TASK_ENV GPU='2' QUEUE_SUITES='libero_10 libero_object' bash experiments/libero/run_table_vlapruner_nora_queue.sh" \
  "$LOG_ROOT/tbl_vlapruner_nora_g2.log"

launch_session \
  "tbl_vlapruner_nora_g3_${TABLE_TAG}" \
  "$NORA_ROOT" \
  "$COMMON_ENV $NORA_TASK_ENV GPU='3' QUEUE_SUITES='libero_goal libero_spatial' bash experiments/libero/run_table_vlapruner_nora_queue.sh" \
  "$LOG_ROOT/tbl_vlapruner_nora_g3.log"

launch_session \
  "tbl_vlaadp_oft_g4_${TABLE_TAG}" \
  "$OFT_ROOT" \
  "$COMMON_ENV $OFT_TASK_ENV GPU='4' QUEUE_SUITES='libero_10 libero_object' bash ./run_table_vlaadp_oft_queue.sh" \
  "$LOG_ROOT/tbl_vlaadp_oft_g4.log"

launch_session \
  "tbl_vlaadp_oft_g5_${TABLE_TAG}" \
  "$OFT_ROOT" \
  "$COMMON_ENV $OFT_TASK_ENV GPU='5' QUEUE_SUITES='libero_goal libero_spatial' bash ./run_table_vlaadp_oft_queue.sh" \
  "$LOG_ROOT/tbl_vlaadp_oft_g5.log"

launch_session \
  "tbl_vlaadp_nora_g6_${TABLE_TAG}" \
  "$NORA_ROOT" \
  "$COMMON_ENV $NORA_TASK_ENV GPU='6' QUEUE_SUITES='libero_10 libero_object' bash experiments/libero/run_table_vlaadp_nora_queue.sh" \
  "$LOG_ROOT/tbl_vlaadp_nora_g6.log"

launch_session \
  "tbl_vlaadp_nora_g7_${TABLE_TAG}" \
  "$NORA_ROOT" \
  "$COMMON_ENV $NORA_TASK_ENV GPU='7' QUEUE_SUITES='libero_goal libero_spatial' bash experiments/libero/run_table_vlaadp_nora_queue.sh" \
  "$LOG_ROOT/tbl_vlaadp_nora_g7.log"

echo "[done] all table tmux queue sessions requested."
tmux ls | grep "tbl_.*_${TABLE_TAG}" || true
