#!/bin/bash

# SGLang-based question generation (no vLLM).
# Aligns with the original veRL interface: <model_path> <num_samples> <save_name>

set -euo pipefail

MODEL_PATH="${1:-/root/model/R-Zero/questioner_current/hf/rollout_0}"
NUM_SAMPLES="${2:-1000}"
SAVE_NAME="${3:-current}"

IFS=',' read -r -a GPU_LIST <<< "${QUESTION_GEN_GPUS}"
SHARDS="${QUESTION_GEN_SHARDS}"
if (( SHARDS > ${#GPU_LIST[@]} )); then
  SHARDS="${#GPU_LIST[@]}"
fi
SGLANG_URL="${SGLANG_URL}"

pids=()
for i in $(seq 0 $((SHARDS - 1))); do
  GPU_ID="${GPU_LIST[$i]}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 /root/R-Zero/question_generate/question_generate.py \
    --model-path "${MODEL_PATH}" \
    --num_samples "${NUM_SAMPLES}" \
    --suffix "${i}" \
    --save_name "${SAVE_NAME}" \
    --sglang-url "${SGLANG_URL}" &
  pids[$i]=$!
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done
