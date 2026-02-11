#!/bin/bash

# SGLang-based evaluation (no vLLM).
# Aligns with the original veRL interface: <model_path> <save_name>

set -euo pipefail

MODEL_PATH="${1:-/root/model/R-Zero/solver_reward_hf}"
SAVE_NAME="${2:-current}"
IFS=',' read -r -a GPU_LIST <<< "${QUESTION_EVAL_GPUS}"
SHARDS="${QUESTION_EVAL_SHARDS}"
if (( SHARDS > ${#GPU_LIST[@]} )); then
  SHARDS="${#GPU_LIST[@]}"
fi

SOLVER_URL="${QUESTION_EVAL_SOLVER_URL:-http://${QUESTION_EVAL_HOST}:${QUESTION_EVAL_PORT}/generate}"
STAGE_TIMEOUT="${QUESTION_EVAL_STAGE_TIMEOUT}"

echo "[question_evaluate] save_name=${SAVE_NAME}"
echo "[question_evaluate] shards=${SHARDS}, gpus=${QUESTION_EVAL_GPUS}"
echo "[question_evaluate] solver_url=${SOLVER_URL}"
echo "[question_evaluate] num_samples=${QUESTION_EVAL_NUM_SAMPLES}, max_new_tokens=${QUESTION_EVAL_MAX_NEW_TOKENS}"
echo "[question_evaluate] request_timeout=${QUESTION_EVAL_REQUEST_TIMEOUT}s, retries=${QUESTION_EVAL_REQUEST_RETRIES}"
echo "[question_evaluate] stage_timeout=${STAGE_TIMEOUT}s"

pids=()
shard_ids=()

if (( SHARDS <= 0 )); then
  echo "[question_evaluate] No shards to run, skipping."
  exit 0
fi

for i in $(seq 0 $((SHARDS - 1))); do
  GPU_ID="${GPU_LIST[$i]}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 python3 /root/R-Zero/question_evaluate/evaluate.py \
    --model "${MODEL_PATH}" \
    --num_samples "${QUESTION_EVAL_NUM_SAMPLES}" \
    --max_new_tokens "${QUESTION_EVAL_MAX_NEW_TOKENS}" \
    --request-timeout "${QUESTION_EVAL_REQUEST_TIMEOUT}" \
    --request-retries "${QUESTION_EVAL_REQUEST_RETRIES}" \
    --progress-every "${QUESTION_EVAL_PROGRESS_EVERY}" \
    --suffix "${i}" \
    --save_name "${SAVE_NAME}" \
    --solver-url "${SOLVER_URL}" &
  pids+=("$!")
  shard_ids+=("${i}")
done

watchdog_pid=""
if [[ "${STAGE_TIMEOUT}" =~ ^[0-9]+$ ]] && (( STAGE_TIMEOUT > 0 )); then
  (
    sleep "${STAGE_TIMEOUT}"
    echo "[question_evaluate] Timeout reached (${STAGE_TIMEOUT}s). Stopping unfinished shards..."
    for pid in "${pids[@]}"; do
      kill -TERM "${pid}" >/dev/null 2>&1 || true
    done
    sleep 5
    for pid in "${pids[@]}"; do
      kill -9 "${pid}" >/dev/null 2>&1 || true
    done
  ) &
  watchdog_pid=$!
fi

failed=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  shard_id="${shard_ids[$idx]}"

  if wait "${pid}"; then
    echo "[question_evaluate] shard ${shard_id} finished"
  else
    status=$?
    failed=1
    echo "[question_evaluate] WARNING: shard ${shard_id} exited with status ${status}"
  fi
done

if [ -n "${watchdog_pid}" ] && kill -0 "${watchdog_pid}" >/dev/null 2>&1; then
  kill "${watchdog_pid}" >/dev/null 2>&1 || true
fi
wait "${watchdog_pid}" >/dev/null 2>&1 || true

if (( failed > 0 )); then
  echo "[question_evaluate] WARNING: one or more shards failed/timed out; continuing with available results."
fi

exit 0
