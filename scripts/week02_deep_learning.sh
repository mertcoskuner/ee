#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
OUT="$RESULTS/week02_deep_learning"

section "Week 2: Deep Learning and PyTorch Overview"

section "Architectures: multi-layer perceptron, convolutional network, Vision Transformer"
run --mode both "${TRAIN[@]}" --model mlp cnn transformer --optimizer adam --attack none --results_dir "$OUT"

section "Optimizers: SGD, SGD with momentum, Adam, AdamW"
run --mode both "${TRAIN[@]}" --lr 0.01 --model cnn --optimizer sgd momentum --attack none --results_dir "$OUT/sgd"
run --mode both "${TRAIN[@]}" --model cnn --optimizer adam adamw --weight_decay 0.01 --attack none --results_dir "$OUT/adaptive"

section "Regularization: L2 weight decay, L1 penalty, dropout, early stopping"
run --mode both "${TRAIN[@]}" --model mlp --optimizer momentum --lr 0.01 --weight_decay 5e-4 --attack none \
    --results_dir "$OUT/l2"
run --mode both "${TRAIN[@]}" --model mlp --optimizer adamw --l1 1e-6 --attack none --results_dir "$OUT/l1"
run --mode both "${TRAIN[@]}" --model mlp --optimizer adam --dropout 0.5 --patience 2 --attack none \
    --results_dir "$OUT/dropout_early_stopping"

echo "Week 2 results: $OUT"
