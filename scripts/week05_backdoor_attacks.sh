#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
OUT="$RESULTS/week05_backdoor_attacks"

section "Week 5: Backdoor Trojan Attacks"

section "Backdoor attacks on vision tasks: BadNets patch, Blend, dynamic triggers"
run --mode both "${TRAIN[@]}" --model cnn --backdoor none badnets blend dynamic --attack none \
    --num_samples "$NUM" --results_dir "$OUT/triggers"

section "Data poisoning methods: effect of the poison rate"
for rate in 0.01 0.05 0.1; do
    run --mode both "${TRAIN[@]}" --model cnn --backdoor badnets --poison_rate "$rate" --attack none \
        --num_samples "$NUM" --results_dir "$OUT/poison_rate_$rate"
done
ensure "$CKPT"/best_cnn_adam_bd-badnets.pth --model cnn --backdoor badnets

section "Dynamic trigger training: random pattern and location in every epoch"
run --mode both "${TRAIN[@]}" --model cnn --backdoor dynamic --trigger_size 5 --attack none \
    --num_samples "$NUM" --results_dir "$OUT/dynamic"

echo "Week 5 results: $OUT"
