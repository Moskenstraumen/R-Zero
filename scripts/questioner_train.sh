#!/bin/bash

set -euo pipefail

SOLVER_MODEL_PATH="$1"
QUESTIONER_MODEL_PATH="$2"
SAVE_NAME="$3"

SLIME_ROOT="${SLIME_ROOT:-/root/slime}"
SGLANG_PYTHONPATH="${SGLANG_PYTHONPATH:-/sgl-workspace/sglang/python}"
QUESTIONER_REF_LOAD="${QUESTIONER_REF_LOAD:-/root/model/Qwen3-4B-Instruct-2507_torch_dist}"

QUESTIONER_SAVE_DIR="${SAVE_ROOT}/${SAVE_NAME}"

RZERO_SOLVER_RM_URL="${RZERO_SOLVER_RM_URL:-}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

NUM_GPUS=$(echo "${QUESTIONER_TRAIN_GPUS}" | awk -F',' '{print NF}')

echo "Start questioner training: ${QUESTIONER_MODEL_PATH} -> ${QUESTIONER_SAVE_DIR}"

set -x
export PYTHONBUFFERED=16
RZERO_SOLVER_HF_CHECKPOINT="${SOLVER_MODEL_PATH}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SGLANG_PYTHONPATH}:/root/Megatron-LM:${RZERO_ROOT}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"RZERO_SOLVER_RM_URL\": \"${RZERO_SOLVER_RM_URL}\",
    \"RZERO_SOLVER_HF_CHECKPOINT\": \"${RZERO_SOLVER_HF_CHECKPOINT}\"
  }
}"

source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"

CKPT_ARGS=(
  --hf-checkpoint "${QUESTIONER_MODEL_PATH}"
  --ref-load "${QUESTIONER_REF_LOAD}"
  --save "${QUESTIONER_SAVE_DIR}"
  --save-interval 1
  --save-hf "${QUESTIONER_SAVE_DIR}/hf/rollout_{rollout_id}"
  --rotary-base 5000000
)

ROLLOUT_ARGS=(
  --data-source-path customization.rzero_hooks.QuestionerDataSource
  --num-rollout 1
  --num-steps-per-rollout 6
  --rollout-batch-size 24
  --n-samples-per-prompt 4
  --global-batch-size 16
  --rollout-max-response-len 4096
  --rollout-max-context-len 4096
  --rollout-max-prompt-len 2048
  --balance-data
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.01
  --kl-loss-type low_var_kl
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
  --tensor-model-parallel-size 2
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
  --rollout-function-path customization.rzero_rollout.generate_rollout
  --custom-rm-path customization.rzero_hooks.questioner_rm_group
  --group-rm
  --rm-url "${RZERO_SOLVER_RM_URL}"
)

WANDB_PROJECT="${WANDB_PROJECT:-rzero}"
WANDB_GROUP="${WANDB_GROUP:-${SAVE_NAME}}"
WANDB_KEY="${WANDB_KEY:-${WANDB_KEY:-}}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ARGS=(
  --use-wandb
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-key "${WANDB_KEY}"
  --wandb-mode "${WANDB_MODE}"
)

CUDA_VISIBLE_DEVICES="${QUESTIONER_TRAIN_GPUS}" ray start --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port "${RAY_DASHBOARD_PORT}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${SLIME_ROOT}/train.py" \
  --train-backend megatron \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --rollout-num-gpus "${NUM_GPUS}" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}"

LATEST_ITER_RAW=$(cat "${QUESTIONER_SAVE_DIR}/latest_checkpointed_iteration.txt")
if [[ ! "${LATEST_ITER_RAW}" =~ ^[0-9]+$ ]]; then
  echo "Unexpected latest checkpoint iteration: ${LATEST_ITER_RAW}"
  exit 1
fi
LATEST_ITER_DIR=$(printf "%s/iter_%07d" "${QUESTIONER_SAVE_DIR}" "${LATEST_ITER_RAW}")
QUESTIONER_VOCAB_SIZE=$(python3 -c 'import sys, torch; print(torch.load(sys.argv[1], weights_only=False)["args"].vocab_size)' "${LATEST_ITER_DIR}/common.pt")

echo "Convert questioner torch_dist -> HF: ${LATEST_ITER_DIR} -> ${QUESTIONER_SAVE_DIR}/hf/rollout_0 (vocab_size=${QUESTIONER_VOCAB_SIZE})"
PYTHONPATH="/root/Megatron-LM:${SLIME_ROOT}:${PYTHONPATH:-}" python3 "${SLIME_ROOT}/tools/convert_torch_dist_to_hf.py" \
  --input-dir "${LATEST_ITER_DIR}" \
  --output-dir "${QUESTIONER_SAVE_DIR}/hf/rollout_0" \
  --origin-hf-dir "${QUESTIONER_MODEL_PATH}" \
  --vocab-size "${QUESTIONER_VOCAB_SIZE}" \
  --force

ln -sfn "${QUESTIONER_SAVE_DIR}" "${SAVE_ROOT}/questioner_current"

echo "Questioner training finished: ${QUESTIONER_SAVE_DIR}"
