#!/usr/bin/env bash

set -euo pipefail

DATASET_ROOT="cache/analysis_dataset"
BASE_OUTPUT="outputs/ml/nn_multiseed_sweep"

ARCHITECTURES=("baseline" "wide" "very_deep")
SEEDS=(42 43 44 45 46)

MAX_EVENTS=100000
EPOCHS=60
BATCH_SIZE=4096
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001
PATIENCE=8
DEVICE="cuda"

mkdir -p "${BASE_OUTPUT}"

echo "============================================================"
echo "NN multi-seed architecture sweep"
echo "============================================================"
echo "Architectures: ${ARCHITECTURES[*]}"
echo "Seeds:         ${SEEDS[*]}"
echo "Output:        ${BASE_OUTPUT}"
echo "============================================================"

for ARCH in "${ARCHITECTURES[@]}"; do

    for SEED in "${SEEDS[@]}"; do

        OUTPUT_DIR="${BASE_OUTPUT}/${ARCH}/seed_${SEED}"

        echo
        echo "============================================================"
        echo "Architecture: ${ARCH}"
        echo "Seed:         ${SEED}"
        echo "Output:       ${OUTPUT_DIR}"
        echo "============================================================"

        python -m scripts.ML.train_event_nn_architecture_sweep \
            --dataset-root "${DATASET_ROOT}" \
            --output-dir "${OUTPUT_DIR}" \
            --architectures "${ARCH}" \
            --max-events-per-class "${MAX_EVENTS}" \
            --epochs "${EPOCHS}" \
            --batch-size "${BATCH_SIZE}" \
            --learning-rate "${LEARNING_RATE}" \
            --weight-decay "${WEIGHT_DECAY}" \
            --patience "${PATIENCE}" \
            --random-seed "${SEED}" \
            --device "${DEVICE}"

    done
done

echo
echo "============================================================"
echo "All training runs finished."
echo "Now combining results..."
echo "============================================================"

python -m scripts.ML.summarise_nn_multiseed \
    --input-root "${BASE_OUTPUT}" \
    --output-dir "${BASE_OUTPUT}/summary"

echo
echo "Finished."