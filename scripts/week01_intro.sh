#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
OUT="$RESULTS/week01_intro"

section "Week 1: Introduction to Robust and Secure Learning"

section "Target model: a clean CNN that performs well on MNIST"
ensure "$CKPT"/best_cnn_adam.pth --model cnn --optimizer adam
run --mode test --model cnn --attack none --num_samples "$NUM" --results_dir "$OUT"

section "Why AI systems fail under attack: tiny inference-time perturbations flip predictions"
run --mode test --model cnn --attack fgsm pgd_linf --steps "$STEPS" --num_samples "$NUM" --results_dir "$OUT"
run --mode visualize --model cnn --steps "$STEPS" --results_dir "$OUT"

section "Threat model: white-box (the attacker knows architecture and weights)"
run --mode test --model cnn --attack pgd_linf --steps "$STEPS" --num_samples "$NUM" --results_dir "$OUT"

section "Threat model: grey-box (same architecture, independently trained weights)"
ensure "$CKPT"/best_cnn_momentum.pth --model cnn --optimizer momentum --lr 0.01
run --mode test --model cnn --attack fgsm pgd_linf --steps "$STEPS" --num_samples "$NUM" \
    --surrogate_model cnn --surrogate_optimizer momentum --results_dir "$OUT"

section "Threat model: black-box (transfer from a different architecture, and query-only access)"
ensure "$CKPT"/best_mlp_adam.pth --model mlp --optimizer adam
run --mode test --model cnn --attack fgsm pgd_linf --steps "$STEPS" --num_samples "$NUM" \
    --surrogate_model mlp --results_dir "$OUT"
run --mode test --model cnn --attack square --square_queries "$QUERIES" --num_samples "$AA_NUM" --results_dir "$OUT"

section "Training-time attack: poisoning the training data implants a backdoor"
run --mode both "${TRAIN[@]}" --model cnn --backdoor badnets --attack none --num_samples "$NUM" --results_dir "$OUT"

echo "Week 1 results: $OUT"
