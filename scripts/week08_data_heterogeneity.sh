#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
OUT="$RESULTS/week08_data_heterogeneity"
FL=(--mode federated --fl_rounds "$ROUNDS" --fl_clients "$FL_CLIENTS" --fl_eval_every 5)
NONIID=(--fl_partition dirichlet --fl_alpha 0.1)

section "Week 8: Data Heterogeneity in Federated Learning"

section "Federated learning with non-IID data: IID vs. Dirichlet vs. label shards"
run "${FL[@]}" --fl_partition iid dirichlet shards --fl_alpha 0.1 --results_dir "$OUT/partitions"

section "Local drift control: SCAFFOLD vs. FedAvg"
run "${FL[@]}" "${NONIID[@]}" --fl_local plain scaffold --fl_local_lr 0.02 --results_dir "$OUT/scaffold"

section "Federated learning with regularization: FedProx"
for mu in 0.01 0.1 1.0; do
    run "${FL[@]}" "${NONIID[@]}" --fl_local fedprox --fl_mu "$mu" --results_dir "$OUT/fedprox_mu_$mu"
done

section "Two-layer aggregation: group-wise FedAvg, then a second aggregation across groups"
run "${FL[@]}" "${NONIID[@]}" --fl_group_size 2 --fl_inner_aggregator fedavg --fl_aggregator fedavg median \
    --results_dir "$OUT/two_layer"

section "Knowledge distillation for regularization"
run "${FL[@]}" "${NONIID[@]}" --fl_local plain kd --fl_kd_beta 1.0 --fl_kd_temperature 3.0 \
    --results_dir "$OUT/distillation"

echo "Week 8 results: $OUT"
