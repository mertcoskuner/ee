#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
OUT="$RESULTS/week10_byzantine_defenses"
FL=(--mode federated --fl_rounds "$ROUNDS" --fl_clients "$FL_CLIENTS" --fl_eval_every 5 --fl_byzantine_ratio 0.2)
ATTACKS=(--fl_attack none label_flip alie ipm sign_flip gaussian)

section "Week 10: Defenses Against Byzantine Attacks"

section "Outlier detection and elimination: Krum, Multi-Krum, Bulyan, MAD outlier removal"
run "${FL[@]}" "${ATTACKS[@]}" --fl_aggregator krum multi_krum bulyan outlier_removal \
    --results_dir "$OUT/outlier_elimination"

section "Gradient / model update sanitization: median, trimmed mean, centered clipping, geometric median, norm clipping"
run "${FL[@]}" "${ATTACKS[@]}" \
    --fl_aggregator median trimmed_mean centered_clipping geometric_median norm_clipping \
    --results_dir "$OUT/sanitization"

section "Robust rules under non-IID data, with and without two-layer bucketing"
run "${FL[@]}" --fl_partition dirichlet --fl_alpha 0.5 --fl_attack alie ipm \
    --fl_aggregator trimmed_mean centered_clipping --results_dir "$OUT/non_iid"
run "${FL[@]}" --fl_partition dirichlet --fl_alpha 0.5 --fl_attack alie ipm --fl_group_size 2 \
    --fl_assumed_byzantine 2 --fl_aggregator trimmed_mean centered_clipping --results_dir "$OUT/non_iid_bucketing"

echo "Week 10 results: $OUT"
