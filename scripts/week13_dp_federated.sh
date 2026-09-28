#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
OUT="$RESULTS/week13_dp_federated"
FL=(--mode federated --fl_rounds "$ROUNDS" --fl_clients "$FL_CLIENTS" --fl_eval_every 5 --fl_dp_clip 1.0)

section "Week 13: Differential Privacy in Federated Learning"

section "Collaboration under DP: no DP vs. central DP (DP-FedAvg) vs. local DP"
run "${FL[@]}" --fl_dp none central local --fl_dp_noise 0.01 --results_dir "$OUT/dp_modes"

section "DP in the FL framework: central DP noise vs. accuracy and epsilon"
for noise in 0.001 0.01 0.1; do
    run "${FL[@]}" --fl_dp central --fl_dp_noise "$noise" --results_dir "$OUT/central_noise_$noise"
done

section "Local DP: every client privatizes its own update"
for noise in 0.001 0.01; do
    run "${FL[@]}" --fl_dp local --fl_dp_noise "$noise" --results_dir "$OUT/local_noise_$noise"
done

section "Client sampling amplifies central DP"
run "${FL[@]}" --fl_dp central --fl_dp_noise 0.01 --fl_participation 0.5 --results_dir "$OUT/central_sampling"

echo "Week 13 results: $OUT"
