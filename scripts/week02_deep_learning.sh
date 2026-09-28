#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
OUT="$RESULTS/week02_deep_learning"

section "Week 2: Deep Learning and PyTorch Overview"

section "Architectures: multi-layer perceptron, convolutional network, Vision Transformer"
for model in mlp cnn transformer; do
    ensure "$CKPT/best_${model}_adam.pth" --model "$model" --optimizer adam
done
run --mode test --model mlp cnn transformer --optimizer adam --attack none --results_dir "$OUT"

section "Optimizers: SGD, SGD with momentum, Adam, AdamW"
run --mode both "${SCHEDULE[@]}" --lr 0.01 --model cnn --optimizer sgd momentum --attack none \
    --checkpoint_dir "$CKPT/week02_sgd" --results_dir "$OUT/sgd"
run --mode both "${TRAIN[@]}" --model cnn --optimizer adam adamw --weight_decay 0.01 --attack none \
    --checkpoint_dir "$CKPT/week02_adaptive" --results_dir "$OUT/adaptive"

section "Regularization: L2 weight decay, L1 penalty, dropout, early stopping"
run --mode both "${SCHEDULE[@]}" --model mlp --optimizer momentum --lr 0.01 --weight_decay 5e-4 --attack none \
    --checkpoint_dir "$CKPT/week02_l2" --results_dir "$OUT/l2"
run --mode both "${TRAIN[@]}" --model mlp --optimizer adamw --l1 1e-6 --attack none \
    --checkpoint_dir "$CKPT/week02_l1" --results_dir "$OUT/l1"
run --mode both "${TRAIN[@]}" --model mlp --optimizer adam --dropout 0.5 --patience 2 --attack none \
    --checkpoint_dir "$CKPT/week02_dropout_early_stopping" --results_dir "$OUT/dropout_early_stopping"

echo "Week 2 results: $OUT"
