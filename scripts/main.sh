#!/bin/bash

set -euo pipefail

csv_count() {
  local csv="$1"
  awk -F',' '{print NF}' <<<"${csv}"
}

unique_gpu_csv() {
  printf "%s\n" "$@" | tr ',' '\n' | awk 'NF' | sort -nu | paste -sd, -
}

kill_gpu_processes() {
  local gpu_csv="$1"
  [ -z "${gpu_csv}" ] && return 0
  command -v nvidia-smi >/dev/null 2>&1 || return 0

  local -a pids=()
  local gpu_id
  IFS=',' read -r -a gpu_ids <<<"${gpu_csv}"
  for gpu_id in "${gpu_ids[@]}"; do
    while IFS= read -r raw_pid; do
      local pid
      pid="$(echo "${raw_pid}" | tr -d '[:space:]')"
      if [[ "${pid}" =~ ^[0-9]+$ ]]; then
        pids+=("${pid}")
      fi
    done < <(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
  done

  if [ ${#pids[@]} -eq 0 ]; then
    return 0
  fi

  mapfile -t unique_pids < <(printf "%s\n" "${pids[@]}" | sort -u)
  for pid in "${unique_pids[@]}"; do
    kill -9 "${pid}" >/dev/null 2>&1 || true
  done
}

kill_runtime_processes() {
  local reason="$1"
  local gpu_csv="${2:-}"

  echo "[main.sh] Cleanup (${reason})"
  set +e

  ray stop --force >/dev/null 2>&1 || true
  pkill -9 -f "sglang.launch_server" >/dev/null 2>&1 || true
  pkill -9 -f "question_generate/question_generate.py" >/dev/null 2>&1 || true
  pkill -9 -f "question_evaluate/evaluate.py" >/dev/null 2>&1 || true
  pkill -9 -f "/root/slime/train.py" >/dev/null 2>&1 || true
  pkill -9 -f "/root/slime/train_async.py" >/dev/null 2>&1 || true
  pkill -9 -f "scripts/questioner_train.sh" >/dev/null 2>&1 || true
  pkill -9 -f "scripts/solver_train.sh" >/dev/null 2>&1 || true
  pkill -9 sglang >/dev/null 2>&1 || true
  pkill -9 ray >/dev/null 2>&1 || true

  if [ -n "${gpu_csv}" ]; then
    kill_gpu_processes "${gpu_csv}"
  fi

  sleep 2
  set -e
}

RM_SERVER_PID=""
QUESTION_GEN_SERVER_PID=""
QUESTION_EVAL_SERVER_PID=""

start_questioner_rm_server() {
  local solver_model_path="$1"
  local rm_log="$2"

  stop_questioner_rm_server

  echo "[main.sh] Start solver RM server on GPUs ${QUESTIONER_RM_GPUS}"
  CUDA_VISIBLE_DEVICES="${QUESTIONER_RM_GPUS}" python3 -m sglang.launch_server \
    --model-path "${solver_model_path}" \
    --host "${QUESTIONER_RM_HOST}" \
    --port "${QUESTIONER_RM_PORT}" \
    --tp "${QUESTIONER_RM_TP}" \
    --dp "${QUESTIONER_RM_DP}" \
    --mem-fraction-static 0.8 \
    >"${rm_log}" 2>&1 &
  RM_SERVER_PID=$!

  until curl -sf "http://${QUESTIONER_RM_HOST}:${QUESTIONER_RM_PORT}/health_generate" >/dev/null; do
    if ! kill -0 "${RM_SERVER_PID}" 2>/dev/null; then
      echo "[main.sh] RM server exited unexpectedly"
      tail -n 20 "${rm_log}" || true
      return 1
    fi
    echo "[main.sh] Waiting for RM server..."
    tail -n 5 "${rm_log}" || true
    sleep 5
  done
}

stop_questioner_rm_server() {
  set +e
  if [ -n "${RM_SERVER_PID}" ] && kill -0 "${RM_SERVER_PID}" 2>/dev/null; then
    kill -9 "${RM_SERVER_PID}" >/dev/null 2>&1 || true
  fi
  RM_SERVER_PID=""
  set -e
}

start_question_generate_server() {
  local questioner_hf_path="$1"
  local server_log="$2"

  stop_question_generate_server

  echo "[main.sh] Start question_generate sglang on GPUs ${QUESTION_GEN_GPUS}"
  CUDA_VISIBLE_DEVICES="${QUESTION_GEN_GPUS}" python3 -m sglang.launch_server \
    --model-path "${questioner_hf_path}" \
    --host "${QUESTION_GEN_HOST}" \
    --port "${QUESTION_GEN_PORT}" \
    --tp "${QUESTION_GEN_TP}" \
    --dp "${QUESTION_GEN_DP}" \
    --mem-fraction-static 0.8 \
    >"${server_log}" 2>&1 &
  QUESTION_GEN_SERVER_PID=$!

  until curl -sf "http://${QUESTION_GEN_HOST}:${QUESTION_GEN_PORT}/health_generate" >/dev/null; do
    if ! kill -0 "${QUESTION_GEN_SERVER_PID}" 2>/dev/null; then
      echo "[main.sh] question_generate sglang server exited unexpectedly"
      tail -n 20 "${server_log}" || true
      return 1
    fi
    echo "[main.sh] Waiting for question_generate sglang server..."
    tail -n 5 "${server_log}" || true
    sleep 5
  done
}

stop_question_generate_server() {
  set +e
  if [ -n "${QUESTION_GEN_SERVER_PID}" ] && kill -0 "${QUESTION_GEN_SERVER_PID}" 2>/dev/null; then
    kill -9 "${QUESTION_GEN_SERVER_PID}" >/dev/null 2>&1 || true
  fi
  QUESTION_GEN_SERVER_PID=""
  set -e
}

start_question_evaluate_server() {
  local solver_model_path="$1"
  local server_log="$2"

  stop_question_evaluate_server

  echo "[main.sh] Start question_evaluate sglang on GPUs ${QUESTION_EVAL_GPUS}"
  CUDA_VISIBLE_DEVICES="${QUESTION_EVAL_GPUS}" python3 -m sglang.launch_server \
    --model-path "${solver_model_path}" \
    --host "${QUESTION_EVAL_HOST}" \
    --port "${QUESTION_EVAL_PORT}" \
    --tp "${QUESTION_EVAL_TP}" \
    --dp "${QUESTION_EVAL_DP}" \
    --mem-fraction-static 0.8 \
    >"${server_log}" 2>&1 &
  QUESTION_EVAL_SERVER_PID=$!

  until curl -sf "http://${QUESTION_EVAL_HOST}:${QUESTION_EVAL_PORT}/health_generate" >/dev/null; do
    if ! kill -0 "${QUESTION_EVAL_SERVER_PID}" 2>/dev/null; then
      echo "[main.sh] question_evaluate sglang server exited unexpectedly"
      tail -n 20 "${server_log}" || true
      return 1
    fi
    echo "[main.sh] Waiting for question_evaluate sglang server..."
    tail -n 5 "${server_log}" || true
    sleep 5
  done
}

stop_question_evaluate_server() {
  set +e
  if [ -n "${QUESTION_EVAL_SERVER_PID}" ] && kill -0 "${QUESTION_EVAL_SERVER_PID}" 2>/dev/null; then
    kill -9 "${QUESTION_EVAL_SERVER_PID}" >/dev/null 2>&1 || true
  fi
  QUESTION_EVAL_SERVER_PID=""
  set -e
}

run_questioner_stage() {
  local solver_model_path="$1"
  local questioner_model_path="$2"
  local save_name="$3"
  local iter_id="$4"
  local stage_gpus
  stage_gpus="$(unique_gpu_csv "${QUESTIONER_TRAIN_GPUS}" "${QUESTIONER_RM_GPUS}")"
  local rm_log="${MAIN_LOG_DIR}/questioner_rm_iter${iter_id}.log"
  local train_log="${MAIN_LOG_DIR}/questioner_train_iter${iter_id}.log"
  local rm_url="http://${QUESTIONER_RM_HOST}:${QUESTIONER_RM_PORT}/generate"

  kill_runtime_processes "before questioner_train iter${iter_id}" "${stage_gpus}"
  start_questioner_rm_server "${solver_model_path}" "${rm_log}"

  echo "[main.sh] Start questioner_train iter${iter_id} on GPUs ${QUESTIONER_TRAIN_GPUS}"
  SLIME_ROOT="${SLIME_ROOT}" \
  SGLANG_PYTHONPATH="${SGLANG_PYTHONPATH}" \
  RZERO_ROOT="${RZERO_ROOT}" \
  SAVE_ROOT="${SAVE_ROOT}" \
  QUESTIONER_TRAIN_GPUS="${QUESTIONER_TRAIN_GPUS}" \
  RZERO_SOLVER_RM_URL="${rm_url}" \
  bash "${RZERO_ROOT}/scripts/questioner_train.sh" \
    "${solver_model_path}" \
    "${questioner_model_path}" \
    "${save_name}" \
    2>&1 | tee "${train_log}"

  if ! grep -q "Questioner training finished:" "${train_log}"; then
    echo "[main.sh] Missing questioner completion marker in ${train_log}"
    return 1
  fi

  stop_questioner_rm_server
  kill_runtime_processes "after questioner_train iter${iter_id}" "${stage_gpus}"
}

run_generation_stage() {
  local questioner_hf_path="$1"
  local save_name="$2"
  local server_log="${MAIN_LOG_DIR}/question_generate_${save_name}.log"
  local sglang_url="http://${QUESTION_GEN_HOST}:${QUESTION_GEN_PORT}/generate"

  kill_runtime_processes "before question_generate ${save_name}" "${QUESTION_GEN_GPUS}"
  start_question_generate_server "${questioner_hf_path}" "${server_log}"

  echo "[main.sh] Start question_generate on GPUs ${QUESTION_GEN_GPUS}"
  QUESTION_GEN_GPUS="${QUESTION_GEN_GPUS}" \
  QUESTION_GEN_SHARDS="${QUESTION_GEN_SHARDS}" \
  SGLANG_URL="${sglang_url}" \
  bash "${RZERO_ROOT}/question_generate/question_generate.bash" \
    "${questioner_hf_path}" \
    "${QUESTION_GEN_SAMPLES}" \
    "${save_name}"

  stop_question_generate_server
  kill_runtime_processes "after question_generate ${save_name}" "${QUESTION_GEN_GPUS}"
}

run_evaluation_stage() {
  local solver_model_path="$1"
  local save_name="$2"
  local server_log="${MAIN_LOG_DIR}/question_evaluate_${save_name}.log"
  local solver_url="http://${QUESTION_EVAL_HOST}:${QUESTION_EVAL_PORT}/generate"

  kill_runtime_processes "before question_evaluate ${save_name}" "${QUESTION_EVAL_GPUS}"
  start_question_evaluate_server "${solver_model_path}" "${server_log}"

  echo "[main.sh] Start question_evaluate on GPUs ${QUESTION_EVAL_GPUS}"
  QUESTION_EVAL_GPUS="${QUESTION_EVAL_GPUS}" \
  QUESTION_EVAL_SHARDS="${QUESTION_EVAL_SHARDS}" \
  QUESTION_EVAL_NUM_SAMPLES="${QUESTION_EVAL_NUM_SAMPLES}" \
  QUESTION_EVAL_MAX_NEW_TOKENS="${QUESTION_EVAL_MAX_NEW_TOKENS}" \
  QUESTION_EVAL_REQUEST_TIMEOUT="${QUESTION_EVAL_REQUEST_TIMEOUT}" \
  QUESTION_EVAL_REQUEST_RETRIES="${QUESTION_EVAL_REQUEST_RETRIES}" \
  QUESTION_EVAL_PROGRESS_EVERY="${QUESTION_EVAL_PROGRESS_EVERY}" \
  QUESTION_EVAL_QUESTION_BATCH="${QUESTION_EVAL_QUESTION_BATCH}" \
  QUESTION_EVAL_STAGE_TIMEOUT="${QUESTION_EVAL_STAGE_TIMEOUT}" \
  QUESTION_EVAL_SOLVER_URL="${solver_url}" \
  bash "${RZERO_ROOT}/question_evaluate/evaluate.sh" \
    "${solver_model_path}" \
    "${save_name}"

  stop_question_evaluate_server
  kill_runtime_processes "after question_evaluate ${save_name}" "${QUESTION_EVAL_GPUS}"
}

build_solver_data() {
  local save_name="$1"

  echo "[main.sh] Build solver jsonl for ${save_name}"
  python3 "${RZERO_ROOT}/scripts/build_solver_jsonl.py" \
    --save_name "${save_name}" \
    --num_shards "${QUESTION_EVAL_SHARDS}" \
    --storage_path "${STORAGE_PATH}"

  local solver_data_file="${STORAGE_PATH}/solver_data/solver_current.jsonl"
  if [ ! -s "${solver_data_file}" ]; then
    echo "[main.sh] ERROR: solver data is empty: ${solver_data_file}"
    exit 1
  fi
}

run_solver_stage() {
  local solver_model_path="$1"
  local questioner_hf_path="$2"
  local save_name="$3"

  kill_runtime_processes "before solver_train ${save_name}" "${SOLVER_ALL_GPUS}"
  echo "[main.sh] Start solver_train on GPUs ${SOLVER_ALL_GPUS}"
  SLIME_ROOT="${SLIME_ROOT}" \
  SGLANG_PYTHONPATH="${SGLANG_PYTHONPATH}" \
  RZERO_ROOT="${RZERO_ROOT}" \
  SAVE_ROOT="${SAVE_ROOT}" \
  STORAGE_PATH="${STORAGE_PATH}" \
  SOLVER_ALL_GPUS="${SOLVER_ALL_GPUS}" \
  bash "${RZERO_ROOT}/scripts/solver_train.sh" \
    "${solver_model_path}" \
    "${questioner_hf_path}" \
    "${save_name}"
  kill_runtime_processes "after solver_train ${save_name}" "${SOLVER_ALL_GPUS}"
}

on_interrupt() {
  echo "[main.sh] Interrupt received, exiting"
  exit 130
}

on_exit() {
  local exit_code=$?
  trap - EXIT INT TERM
  set +e
  stop_questioner_rm_server
  stop_question_generate_server
  stop_question_evaluate_server
  kill_runtime_processes "script exit" "${ALL_STAGE_GPUS:-}"
  exit "${exit_code}"
}

trap on_interrupt INT TERM
trap on_exit EXIT

# Run identity and shared roots
BASE_MODEL="$1"
MODEL_ABBR="$2"
NUM_ITERS="${NUM_ITERS:-5}"

SLIME_ROOT="${SLIME_ROOT:-/root/slime}"
SGLANG_PYTHONPATH="${SGLANG_PYTHONPATH:-/sgl-workspace/sglang/python}"
RZERO_ROOT="${RZERO_ROOT:-/root/R-Zero}"
SAVE_ROOT="${SAVE_ROOT:-/root/model/R-Zero}"
STORAGE_PATH="${STORAGE_PATH:-/root/dataset/R-Zero}"

# Questioner training + RM
QUESTIONER_TRAIN_GPUS="${QUESTIONER_TRAIN_GPUS:-0,1,2,3}"
QUESTIONER_RM_GPUS="${QUESTIONER_RM_GPUS:-4,5,7}"
QUESTIONER_RM_HOST="${QUESTIONER_RM_HOST:-127.0.0.1}"
QUESTIONER_RM_PORT="${QUESTIONER_RM_PORT:-15200}"
QUESTIONER_RM_TP="${QUESTIONER_RM_TP:-1}"
QUESTIONER_RM_DP="${QUESTIONER_RM_DP:-$(csv_count "${QUESTIONER_RM_GPUS}")}"

# Question generation
QUESTION_GEN_GPUS="${QUESTION_GEN_GPUS:-0,1,2,3,4,5,7}"
QUESTION_GEN_HOST="${QUESTION_GEN_HOST:-127.0.0.1}"
QUESTION_GEN_PORT="${QUESTION_GEN_PORT:-15100}"
QUESTION_GEN_TP="${QUESTION_GEN_TP:-1}"
QUESTION_GEN_DP="${QUESTION_GEN_DP:-$(csv_count "${QUESTION_GEN_GPUS}")}"
QUESTION_GEN_SHARDS="${QUESTION_GEN_SHARDS:-$(csv_count "${QUESTION_GEN_GPUS}")}"
QUESTION_GEN_SAMPLES="${QUESTION_GEN_SAMPLES:-1000}"

# Question evaluation
QUESTION_EVAL_GPUS="${QUESTION_EVAL_GPUS:-0,1,2,3,4,5,7}"
QUESTION_EVAL_HOST="${QUESTION_EVAL_HOST:-127.0.0.1}"
QUESTION_EVAL_PORT="${QUESTION_EVAL_PORT:-15200}"
QUESTION_EVAL_TP="${QUESTION_EVAL_TP:-1}"
QUESTION_EVAL_DP="${QUESTION_EVAL_DP:-$(csv_count "${QUESTION_EVAL_GPUS}")}"
QUESTION_EVAL_SHARDS="${QUESTION_EVAL_SHARDS:-$(csv_count "${QUESTION_EVAL_GPUS}")}"
QUESTION_EVAL_NUM_SAMPLES="${QUESTION_EVAL_NUM_SAMPLES:-9}"
QUESTION_EVAL_MAX_NEW_TOKENS="${QUESTION_EVAL_MAX_NEW_TOKENS:-4096}"
QUESTION_EVAL_QUESTION_BATCH="${QUESTION_EVAL_QUESTION_BATCH:-16}"
QUESTION_EVAL_REQUEST_TIMEOUT="${QUESTION_EVAL_REQUEST_TIMEOUT:-600}"
QUESTION_EVAL_REQUEST_RETRIES="${QUESTION_EVAL_REQUEST_RETRIES:-2}"
QUESTION_EVAL_PROGRESS_EVERY="${QUESTION_EVAL_PROGRESS_EVERY:-20}"
QUESTION_EVAL_STAGE_TIMEOUT="${QUESTION_EVAL_STAGE_TIMEOUT:-0}"

# Solver training
SOLVER_ALL_GPUS="${SOLVER_ALL_GPUS:-0,1,2,3,4,5,7}"

MAIN_LOG_DIR="${MAIN_LOG_DIR:-/tmp/rzero_main}"
mkdir -p "${SAVE_ROOT}" "${STORAGE_PATH}/solver_data" "${MAIN_LOG_DIR}"

ALL_STAGE_GPUS="$(unique_gpu_csv "${QUESTIONER_TRAIN_GPUS}" "${QUESTIONER_RM_GPUS}" "${QUESTION_GEN_GPUS}" "${QUESTION_EVAL_GPUS}" "${SOLVER_ALL_GPUS}")"

echo "[main.sh] Base model: ${BASE_MODEL}"
echo "[main.sh] Model abbr: ${MODEL_ABBR}"
echo "[main.sh] Iterations: ${NUM_ITERS}"
echo "[main.sh] Questioner GPUs: ${QUESTIONER_TRAIN_GPUS}; RM GPUs: ${QUESTIONER_RM_GPUS} (tp=${QUESTIONER_RM_TP}, dp=${QUESTIONER_RM_DP})"
echo "[main.sh] Generate GPUs: ${QUESTION_GEN_GPUS} (tp=${QUESTION_GEN_TP}, dp=${QUESTION_GEN_DP}); Evaluate GPUs: ${QUESTION_EVAL_GPUS} (tp=${QUESTION_EVAL_TP}, dp=${QUESTION_EVAL_DP})"
echo "[main.sh] Eval params: num_samples=${QUESTION_EVAL_NUM_SAMPLES}, max_new_tokens=${QUESTION_EVAL_MAX_NEW_TOKENS}, question_batch=${QUESTION_EVAL_QUESTION_BATCH}, timeout=${QUESTION_EVAL_REQUEST_TIMEOUT}s"
echo "[main.sh] Eval stage timeout: ${QUESTION_EVAL_STAGE_TIMEOUT}s"
echo "[main.sh] Solver GPUs: ${SOLVER_ALL_GPUS}"

kill_runtime_processes "startup reset" "${ALL_STAGE_GPUS}"

for i in $(seq 1 "${NUM_ITERS}"); do
  export RZERO_ITER="${i}"
  prev=$((i - 1))

  if [ "${i}" -eq 1 ]; then
    solver_model_path="${BASE_MODEL}"
    questioner_model_path="${BASE_MODEL}"
  else
    solver_model_path="${SAVE_ROOT}/${MODEL_ABBR}_solver_v${prev}/hf/rollout_0"
    questioner_model_path="${SAVE_ROOT}/${MODEL_ABBR}_questioner_v${prev}/hf/rollout_0"
  fi

  questioner_save_name="${MODEL_ABBR}_questioner_v${i}"
  solver_save_name="${MODEL_ABBR}_solver_v${i}"
  questioner_hf_path="${SAVE_ROOT}/${questioner_save_name}/hf/rollout_0"

  echo "[main.sh] ===== Iteration ${i}/${NUM_ITERS} ====="
  run_questioner_stage "${solver_model_path}" "${questioner_model_path}" "${questioner_save_name}" "${i}"
  run_generation_stage "${questioner_hf_path}" "${solver_save_name}"
  run_evaluation_stage "${solver_model_path}" "${solver_save_name}"
  build_solver_data "${solver_save_name}"
  run_solver_stage "${solver_model_path}" "${questioner_hf_path}" "${solver_save_name}"
done

echo "[main.sh] Training loop finished"
