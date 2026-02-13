#!/bin/bash

set -euo pipefail

SOLVER_MODEL_PATH="$1"
QUESTIONER_MODEL_PATH="$2"
SAVE_NAME="$3"

SLIME_ROOT="${SLIME_ROOT:-/root/slime}"
SGLANG_PYTHONPATH="${SGLANG_PYTHONPATH:-/sgl-workspace/sglang/python}"

SOLVER_REF_LOAD="${SOLVER_REF_LOAD:-/root/model/Qwen3-4B-Instruct-2507_torch_dist}"

SOLVER_SAVE_DIR="${SAVE_ROOT}/${SAVE_NAME}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

NUM_SOLVER_GPUS=$(echo "${SOLVER_ALL_GPUS}" | awk -F',' '{print NF}')

echo "Start solver training: ${SOLVER_MODEL_PATH} + ${QUESTIONER_MODEL_PATH} -> ${SOLVER_SAVE_DIR}"

set -x

SOLVER_DATA_FILE="${STORAGE_PATH}/solver_data/solver_current.jsonl"
if [ ! -s "${SOLVER_DATA_FILE}" ]; then
  echo "ERROR: solver training data is empty: ${SOLVER_DATA_FILE}"
  echo "Hint: generated questions/evaluations likely failed or all scores were filtered out."
  exit 1
fi

export PYTHONPATH="${SGLANG_PYTHONPATH}:/root/Megatron-LM:${RZERO_ROOT}:${SLIME_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONBUFFERED=16

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SGLANG_PYTHONPATH}:/root/Megatron-LM:${RZERO_ROOT}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"

CKPT_ARGS=(
  --hf-checkpoint "${SOLVER_MODEL_PATH}"
  --ref-load "${SOLVER_REF_LOAD}"
  --save "${SOLVER_SAVE_DIR}"
  --save-interval 1
  --save-hf "${SOLVER_SAVE_DIR}/hf/rollout_{rollout_id}"
  --rotary-base 5000000
)

ROLLOUT_ARGS=(
  --prompt-data "${SOLVER_DATA_FILE}"
  --input-key prompt
  --label-key answer
  --apply-chat-template
  --rollout-shuffle
  --num-rollout 1
  --num-steps-per-rollout 20
  --rollout-batch-size 504
  --n-samples-per-prompt 5
  --global-batch-size 126
  --rollout-max-response-len 4096
  --rollout-max-prompt-len 2048
  --balance-data
)

EVAL_ARGS=(
  --eval-interval 1
  --eval-prompt-data aime  /root/dataset/aime-2024/aime-2024.jsonl
  --eval-input-key prompt
  --eval-label-key label
  --n-samples-per-eval-prompt 8
  --eval-max-response-len 8192
  --eval-top-p 1
  --skip-eval-before-train
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.01
  --kl-loss-type low_var_kl
  --use-rollout-logprobs
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.98
)

PERF_ARGS=(
  --tensor-model-parallel-size 1
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 9216
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

CUSTOM_ARGS=(
  --custom-rm-path customization.reward_model.solver_rm
)

WANDB_PROJECT="${WANDB_PROJECT:-rzero}"
WANDB_GROUP="${WANDB_GROUP:-${SAVE_NAME}}"
WANDB_KEY="${WANDB_KEY:-}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ARGS=(
  --use-wandb
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-key "${WANDB_KEY}"
  --wandb-mode "${WANDB_MODE}"
)

CUDA_VISIBLE_DEVICES="${SOLVER_ALL_GPUS}" ray start --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${NUM_SOLVER_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port "${RAY_DASHBOARD_PORT}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${SLIME_ROOT}/train.py" \
  --train-backend megatron \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_SOLVER_GPUS}" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}"

echo "Solver training finished: ${SOLVER_SAVE_DIR}"
