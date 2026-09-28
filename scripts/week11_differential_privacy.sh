#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
OUT="$RESULTS/week11_differential_privacy"
SMALL=(--validation_size 59000 --epochs "$MIA_EPOCHS" --optimizer momentum --lr 0.05 --log_interval 1000
    --checkpoint_dir "$CKPT/week11_small_training_set" --attack none --mia --mia_samples 1000)

section "Week 11: Differential Privacy"

section "Motivation: a model trained on few images memorizes them (membership inference)"
run --mode both "${SMALL[@]}" --results_dir "$OUT/non_private"

section "DP as a defense: the same training with DP-SGD leaks far less membership"
run --mode both "${SMALL[@]}" --dp --dp_noise 1.0 --dp_clip 1.0 --results_dir "$OUT/dp_sgd"

section "Definition, basic mechanisms and properties: Laplace, Gaussian, randomized response,"
section "post-processing, sequential composition and group privacy on MNIST counting queries"
run --mode dp_mechanisms --dp_trials "$DP_TRIALS" --results_dir "$OUT/mechanisms"

echo "Week 11 results: $OUT"
