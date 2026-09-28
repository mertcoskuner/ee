#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
OUT="$RESULTS/week12_dp_machine_learning"
DP=(--epochs "$DP_EPOCHS" --optimizer sgd --lr 0.5 --log_interval 1000 --attack none)

section "Week 12: DP in Machine Learning"

section "DP with composition and sub-sampling: epsilon of DP-SGD under four accountants"
run --mode dp_accounting --epochs "$DP_EPOCHS" --dp_noise 1.0 --results_dir "$OUT/accounting"
for batch in 25 250; do
    run --mode dp_accounting --epochs "$DP_EPOCHS" --dp_noise 1.0 --batch_size "$batch" \
        --results_dir "$OUT/accounting_batch_$batch"
done

section "Building a DP machine learning algorithm: DP-SGD, the privacy-utility trade-off"
run --mode both "${DP[@]}" --results_dir "$OUT/non_private"
for noise in 0.5 1.0 2.0; do
    run --mode both "${DP[@]}" --dp --dp_noise "$noise" --dp_clip 1.0 --results_dir "$OUT/dp_noise_$noise"
done

echo "Week 12 results: $OUT"
