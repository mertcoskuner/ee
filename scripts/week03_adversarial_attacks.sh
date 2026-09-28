#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
OUT="$RESULTS/week03_adversarial_attacks"

section "Week 3: Adversarial Attacks"
ensure "$CKPT"/best_cnn_adam.pth --model cnn --optimizer adam

section "Generating adversarial samples: FGSM, PGD (L-inf and L2), L-BFGS"
run --mode test --model cnn --attack fgsm pgd_linf pgd_l2 lbfgs --steps "$STEPS" --num_samples "$NUM" \
    --results_dir "$OUT"

section "Budget sweep: accuracy under FGSM and PGD as epsilon grows"
for eps in 0.05 0.1 0.2 0.3; do
    run --mode test --model cnn --attack fgsm pgd_linf --eps_linf "$eps" --steps "$STEPS" \
        --num_samples "$NUM" --results_dir "$OUT/eps_$eps"
done

section "Adversarial examples and perturbations"
run --mode visualize --model cnn --steps "$STEPS" --results_dir "$OUT"

section "Geometry of adversarial perturbations: gradient vs. random directions, decision regions"
run --mode geometry --model cnn --steps "$STEPS" --num_samples "$NUM" --results_dir "$OUT"
run --mode tsne --model cnn --steps "$STEPS" --results_dir "$OUT"

echo "Week 3 results: $OUT"
