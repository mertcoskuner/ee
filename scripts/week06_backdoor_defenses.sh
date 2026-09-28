#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
OUT="$RESULTS/week06_backdoor_defenses"
DEFENSE=(--nc_steps "$NC_STEPS" --ls_samples "$LS_SAMPLES" --num_samples "$NUM")

section "Week 6: Defense Against Backdoor Attacks"
for trigger in badnets blend dynamic; do
    ensure "best_cnn_adam_bd-$trigger.pth" --model cnn --backdoor "$trigger"
done
ensure best_cnn_adam.pth --model cnn --optimizer adam

section "Neural Cleanse: reverse-engineered triggers and anomaly index"
run --mode defense --model cnn --backdoor none badnets blend dynamic --defense neural_cleanse "${DEFENSE[@]}" \
    --results_dir "$OUT/neural_cleanse"

section "Latent separability analysis of penultimate representations"
run --mode defense --model cnn --backdoor none badnets blend dynamic --defense latent_separability "${DEFENSE[@]}" \
    --results_dir "$OUT/latent_separability"

section "Fine-pruning: prune dormant units, fine-tune, and re-measure the attack success rate"
run --mode defense --model cnn --backdoor badnets blend dynamic --defense fine_pruning "${DEFENSE[@]}" \
    --results_dir "$OUT/fine_pruning"

echo "Week 6 results: $OUT"
