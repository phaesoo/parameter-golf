#!/bin/bash
# Batch runner for all experiments.
# Runs each experiment for a short comparison run (3000 steps on 1xGPU).
# Results are logged to experiments/results/<experiment_name>.log

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

# Configurable defaults
ITERATIONS="${ITERATIONS:-3000}"
NPROC="${NPROC:-1}"
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"

echo "=== Parameter Golf Experiment Batch Runner ==="
echo "ITERATIONS=${ITERATIONS}, NPROC=${NPROC}"
echo "DATA_PATH=${DATA_PATH}"
echo "Results: ${RESULTS_DIR}/"
echo ""

# Collect experiment directories (sorted)
EXPERIMENTS=$(find "${SCRIPT_DIR}" -maxdepth 1 -type d -name 'e*' | sort)

if [ -z "${EXPERIMENTS}" ]; then
    echo "No experiment directories found."
    exit 1
fi

# Also run baseline for comparison
echo "--- baseline ---"
BASELINE_LOG="${RESULTS_DIR}/baseline.log"
RUN_ID=baseline \
DATA_PATH="${DATA_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
ITERATIONS="${ITERATIONS}" \
MAX_WALLCLOCK_SECONDS=0 \
VAL_LOSS_EVERY="${ITERATIONS}" \
TRAIN_LOG_EVERY=500 \
torchrun --standalone --nproc_per_node="${NPROC}" \
    "${SCRIPT_DIR}/../train_gpt.py" 2>&1 | tee "${BASELINE_LOG}"
echo ""

# Run each experiment
for EXP_DIR in ${EXPERIMENTS}; do
    EXP_NAME=$(basename "${EXP_DIR}")
    TRAIN_SCRIPT="${EXP_DIR}/train_gpt.py"

    if [ ! -f "${TRAIN_SCRIPT}" ]; then
        echo "--- ${EXP_NAME}: SKIPPED (no train_gpt.py) ---"
        continue
    fi

    echo "--- ${EXP_NAME} ---"
    LOG_FILE="${RESULTS_DIR}/${EXP_NAME}.log"

    RUN_ID="${EXP_NAME}" \
    DATA_PATH="${DATA_PATH}" \
    TOKENIZER_PATH="${TOKENIZER_PATH}" \
    ITERATIONS="${ITERATIONS}" \
    MAX_WALLCLOCK_SECONDS=0 \
    VAL_LOSS_EVERY="${ITERATIONS}" \
    TRAIN_LOG_EVERY=500 \
    torchrun --standalone --nproc_per_node="${NPROC}" \
        "${TRAIN_SCRIPT}" 2>&1 | tee "${LOG_FILE}"
    echo ""
done

echo "=== All experiments complete ==="
echo "Run: python3 ${SCRIPT_DIR}/compare.py to see results table"
