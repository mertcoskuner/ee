#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
OUT="$RESULTS/week04_advanced_attacks_defenses"

section "Week 4: Advanced Adversarial Attacks and Defense Methods"
ensure "$CKPT"/best_cnn_adam.pth --model cnn --optimizer adam
ensure "$CKPT"/best_mlp_adam.pth --model mlp --optimizer adam
ensure "$CKPT"/best_transformer_adam.pth --model transformer --optimizer adam

section "Carlini-Wagner L2 attack"
run --mode test --model cnn --attack cw --cw_steps "$CW_STEPS" --num_samples "$AA_NUM" --results_dir "$OUT/cw"

section "Transferability: adversarial examples crafted on other models"
for source in mlp transformer; do
    run --mode test --model cnn --attack fgsm pgd_linf --steps "$STEPS" --num_samples "$NUM" \
        --surrogate_model "$source" --results_dir "$OUT/transfer_from_$source"
done

section "Black-box attacks: query-based Square Attack and transfer"
run --mode test --model cnn --attack square --square_queries "$QUERIES" --num_samples "$AA_NUM" \
    --results_dir "$OUT/black_box"

section "Adversarial training with PGD, tracking robust test accuracy (adversarial overfitting)"
run --mode both --model cnn --optimizer adam --lr 1e-3 --epochs "$ADV_EPOCHS" --train_attack pgd_linf \
    --track_test --track_samples "$NUM" --attack fgsm pgd_linf --steps "$STEPS" --num_samples "$NUM" \
    --log_interval 1000 --results_dir "$OUT/adversarial_training"

section "AutoAttack: standard vs. adversarially trained model"
run --mode test --model cnn --train_attack none pgd_linf --attack autoattack --num_samples "$AA_NUM" \
    --results_dir "$OUT/autoattack"

echo "Week 4 results: $OUT"
