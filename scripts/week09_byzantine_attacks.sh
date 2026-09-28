#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
OUT="$RESULTS/week09_byzantine_attacks"
FL=(--mode federated --fl_rounds "$ROUNDS" --fl_clients "$FL_CLIENTS" --fl_eval_every 5)

section "Week 9: Byzantine Attacks in Federated Learning"

section "Byzantine vulnerability of FL: FedAvg under every attack with 20% Byzantine clients"
run "${FL[@]}" --fl_aggregator fedavg --fl_attack all --fl_byzantine_ratio 0.2 --results_dir "$OUT/fedavg"

section "Byzantine frameworks: label flip, ALIE, IPM against a robust rule"
run "${FL[@]}" --fl_aggregator trimmed_mean --fl_attack none label_flip alie ipm --fl_byzantine_ratio 0.2 \
    --results_dir "$OUT/trimmed_mean"

section "Attack strength: fraction of Byzantine clients"
for ratio in 0.1 0.2 0.3; do
    run "${FL[@]}" --fl_aggregator median --fl_attack alie ipm --fl_byzantine_ratio "$ratio" \
        --results_dir "$OUT/ratio_$ratio"
done

section "Attack parameters: IPM epsilon"
for eps in 0.1 0.5 2.0; do
    run "${FL[@]}" --fl_aggregator fedavg median --fl_attack ipm --fl_ipm_epsilon "$eps" \
        --fl_byzantine_ratio 0.2 --results_dir "$OUT/ipm_eps_$eps"
done

echo "Week 9 results: $OUT"
