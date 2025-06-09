# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# Check if enough arguments are provided
if [ $# -lt 2 ]; then
    echo "Error: PREFIX_DIR and MODEL_PATH are required as the first and second arguments respectively."
    exit 1
fi

PREFIX_DIR=$1
MODEL_PATH=$2
LOG_PATH=$3
if [ ! -d "$LOG_PATH" ]; then
    mkdir -p "$LOG_PATH"
fi
shift 3
ARGS=("$@")
export MASTER_PORT=10042

FULL_MODEL_PATH="$PREFIX_DIR/$MODEL_PATH"

IFS=' ' read -r -a DATASETS <<< "$DATASETS_STR"

for DATASET in "${DATASETS[@]}"; do
    bash eval/vlm/evaluate.sh \
        "$FULL_MODEL_PATH" \
        "$DATASET" \
        --out-dir "$LOG_PATH/$DATASET" \
        "${ARGS[@]}"
done