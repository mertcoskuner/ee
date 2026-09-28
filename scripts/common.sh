#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-auto}
QUICK=${QUICK:-0}
RESULTS=${RESULTS:-results}
if [ "$QUICK" = 1 ]; then CKPT=${CKPT:-checkpoints_quick}; else CKPT=${CKPT:-checkpoints}; fi

if [ "$QUICK" = 1 ]; then
    EPOCHS=1; NUM=200; STEPS=10; CW_STEPS=50; QUERIES=200; AA_NUM=50
    ADV_EPOCHS=2; NC_STEPS=100; ROUNDS=3; FL_CLIENTS=10; LS_SAMPLES=5000
    MIA_EPOCHS=10; DP_EPOCHS=1; DP_TRIALS=50
else
    EPOCHS=5; NUM=1000; STEPS=40; CW_STEPS=1000; QUERIES=1000; AA_NUM=500
    ADV_EPOCHS=20; NC_STEPS=1000; ROUNDS=30; FL_CLIENTS=20; LS_SAMPLES=5000
    MIA_EPOCHS=40; DP_EPOCHS=5; DP_TRIALS=500
fi

COMMON=(--device "$DEVICE" --num_workers 0 --checkpoint_dir "$CKPT")
SCHEDULE=(--epochs "$EPOCHS" --log_interval 1000)
TRAIN=("${SCHEDULE[@]}" --lr 1e-3)

section() {
    printf '\n\033[1m=== %s ===\033[0m\n' "$*"
}

run() {
    echo "+ $PYTHON main.py $*"
    "$PYTHON" main.py "${COMMON[@]}" "$@"
}

ensure() {
    local checkpoint=$1
    shift
    if [ -f "$checkpoint" ]; then
        echo "reusing $checkpoint"
    else
        if [[ " $* " == *" --lr "* ]]; then
            run --mode train "${SCHEDULE[@]}" "$@"
        else
            run --mode train "${TRAIN[@]}" "$@"
        fi
    fi
}
