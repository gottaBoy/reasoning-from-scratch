#!/bin/sh
set -eu

DEVICE="${DEVICE:-cpu}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
DATASET_SIZE="${DATASET_SIZE:-1}"
RUN_TRAINING="${RUN_TRAINING:-0}"

run() {
    printf '\n>>>'
    printf ' %s' "$@"
    printf '\n'
    "$@"
}

echo "Smoke run for reasoning-from-scratch"
echo "DEVICE=${DEVICE} MAX_NEW_TOKENS=${MAX_NEW_TOKENS} DATASET_SIZE=${DATASET_SIZE}"

run python -c '
import torch
import reasoning_from_scratch
print("package:", reasoning_from_scratch.__version__)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
'

run python ch02/05_use_model/generate_simple.py \
    --device "$DEVICE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --prompt "Explain attention in one sentence."

run python ch03/02_math500-verifier-scripts/evaluate_math500.py \
    --device "$DEVICE" \
    --which_model base \
    --dataset_size "$DATASET_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS"

run python ch04/02_math500-inference-scaling-scripts/cot_prompting_math500.py \
    --device "$DEVICE" \
    --which_model base \
    --dataset_size "$DATASET_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS"

run python ch04/02_math500-inference-scaling-scripts/self_consistency_math500.py \
    --device "$DEVICE" \
    --which_model base \
    --dataset_size "$DATASET_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_samples 2 \
    --temperature 0.9 \
    --top_p 0.9

run python ch05/02_math500-more-inference-scaling-scripts/self_refinement_math500.py \
    --device "$DEVICE" \
    --which_model base \
    --dataset_size "$DATASET_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --critique_max_new_tokens 16 \
    --iterations 1 \
    --scoring none

run python ch05/02_math500-more-inference-scaling-scripts/best_of_n_math500.py \
    --device "$DEVICE" \
    --which_model base \
    --dataset_size "$DATASET_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_samples 2 \
    --scoring heuristic

run python ch08/04_train_with_distillation/distill.py \
    --data_path ch08/02_generate_distillation_data/sample_openrouter_outputs.json \
    --dataset_size 5 \
    --validation_size 1 \
    --epochs 1 \
    --log_every 1

if [ "$RUN_TRAINING" = "1" ]; then
    run python ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl.py \
        --steps 1 \
        --num_rollouts 2 \
        --max_new_tokens "$MAX_NEW_TOKENS"
fi

run sh -lc 'SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 pytest -q tests'

echo
echo "Smoke examples completed."
