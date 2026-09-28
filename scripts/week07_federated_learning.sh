#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
OUT="$RESULTS/week07_federated_learning"
FL=(--mode federated --fl_rounds "$ROUNDS" --fl_clients "$FL_CLIENTS" --fl_eval_every 5)

section "Week 7: Federated Learning"

section "Federated learning framework: FedAvg with full and partial participation"
run "${FL[@]}" --results_dir "$OUT/fedavg"
run "${FL[@]}" --fl_participation 0.5 --results_dir "$OUT/partial_participation"

section "Aggregation methods on honest clients"
run "${FL[@]}" --fl_aggregator fedavg median trimmed_mean geometric_median --fl_assumed_byzantine 2 \
    --results_dir "$OUT/aggregators"

section "Accelerated FL: server momentum (FedAvgM), Nesterov, FedAdam, local momentum"
run "${FL[@]}" --fl_server_opt sgd momentum nesterov --results_dir "$OUT/server_momentum"
run "${FL[@]}" --fl_server_opt adam --fl_server_lr 0.01 --results_dir "$OUT/fedadam"
run "${FL[@]}" --fl_local_momentum 0.9 --fl_local_lr 0.01 --results_dir "$OUT/local_momentum"

echo "Week 7 results: $OUT"
