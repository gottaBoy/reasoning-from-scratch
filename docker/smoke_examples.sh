#!/bin/sh
set -eu

DEVICE="${DEVICE:-cpu}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
DATASET_SIZE="${DATASET_SIZE:-1}"
RUN_TRAINING="${RUN_TRAINING:-0}"
RUN_ADVANCED_TRAINING="${RUN_ADVANCED_TRAINING:-0}"
RUN_TESTS="${RUN_TESTS:-1}"
LOG_DIR="${SMOKE_LOG_DIR:-/workspace/logs/smoke-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
mkdir -p "$LOG_DIR"
printf 'stage\texit_code\tlog\n' > "$LOG_DIR/summary.tsv"

run() {
    stage="$1"
    shift
    logfile="$LOG_DIR/$stage.log"
    printf '\n>>> %s:' "$stage"
    printf ' %s' "$@"
    printf '\n'
    if "$@" > "$logfile" 2>&1; then
        cat "$logfile"
        printf '%s\t0\t%s\n' "$stage" "$logfile" >> "$LOG_DIR/summary.tsv"
    else
        code="$?"
        cat "$logfile"
        printf '%s\t%s\t%s\n' "$stage" "$code" "$logfile" >> "$LOG_DIR/summary.tsv"
        printf 'Failed: %s (exit %s). Logs: %s\n' "$stage" "$code" "$LOG_DIR" >&2
        exit "$code"
    fi
}

echo "Smoke run for reasoning-from-scratch"
echo "DEVICE=${DEVICE} MAX_NEW_TOKENS=${MAX_NEW_TOKENS} DATASET_SIZE=${DATASET_SIZE}"
echo "Logs: $LOG_DIR"

run environment python docker/check_environment.py \
    --device "$DEVICE" --output "$LOG_DIR/environment.json"

run ch02_generation python ch02/05_use_model/generate_simple.py \
    --device "$DEVICE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --prompt "Explain attention in one sentence."

run ch03_evaluation python ch03/02_math500-verifier-scripts/evaluate_math500.py \
    --device "$DEVICE" \
    --which_model base \
    --dataset_size "$DATASET_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS"

run ch04_cot python ch04/02_math500-inference-scaling-scripts/cot_prompting_math500.py \
    --device "$DEVICE" \
    --which_model base \
    --dataset_size "$DATASET_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS"

run ch04_self_consistency python ch04/02_math500-inference-scaling-scripts/self_consistency_math500.py \
    --device "$DEVICE" \
    --which_model base \
    --dataset_size "$DATASET_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_samples 2 \
    --temperature 0.9 \
    --top_p 0.9

run ch05_self_refinement python ch05/02_math500-more-inference-scaling-scripts/self_refinement_math500.py \
    --device "$DEVICE" \
    --which_model base \
    --dataset_size "$DATASET_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --critique_max_new_tokens 16 \
    --iterations 1 \
    --scoring none

run ch05_best_of_n python ch05/02_math500-more-inference-scaling-scripts/best_of_n_math500.py \
    --device "$DEVICE" \
    --which_model base \
    --dataset_size "$DATASET_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_samples 2 \
    --scoring heuristic

run ch08_distillation python ch08/04_train_with_distillation/distill.py \
    --data_path ch08/02_generate_distillation_data/sample_openrouter_outputs.json \
    --dataset_size 5 \
    --validation_size 1 \
    --epochs 1 \
    --log_every 1

if [ "$RUN_TRAINING" = "1" ]; then
    run ch06_grpo python ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl.py \
        --steps 1 \
        --num_rollouts 2 \
        --max_new_tokens "$MAX_NEW_TOKENS"
fi

if [ "$RUN_ADVANCED_TRAINING" = "1" ]; then
    run ch07_format_reward python ch07/03_rlvr_grpo_scripts_advanced/7_6_plus_format_reward.py \
        --steps 1 \
        --num_rollouts 2 \
        --max_new_tokens "$MAX_NEW_TOKENS"
fi

if [ "$RUN_TESTS" = "1" ]; then
    run tests env SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 python -m pytest -q tests
fi

echo
echo "Smoke examples completed. Results: $LOG_DIR/summary.tsv"
