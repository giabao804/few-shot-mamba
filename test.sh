#!/bin/bash

usage() {
    echo "Usage: $0 <mode> <dataset>"
    echo "  <mode>      Testing mode: 1 for 1-shot or 5 for 5-shot"
    echo "  <dataset>   Dataset selection: 'CWRU' or 'HUST'"
    echo
    echo "Examples:"
    echo "  bash $0 1 CWRU    # Runs 1-shot testing on CWRU dataset"
    echo "  bash $0 5 HUST    # Runs 5-shot testing on HUST dataset"
    exit 1
}

if [ "$#" -ne 2 ]; then
    echo "Error: Invalid number of arguments."
    usage
fi

MODE=$1
DATASET=$2
MODEL_NAME="Net"

# Validate dataset selection
if [[ "$DATASET" != "CWRU" && "$DATASET" != "HUST" ]]; then
    echo "Error: Dataset must be 'CWRU' or 'HUST'."
    usage
fi


BEST_WEIGHT_CWRU_1SHOT="/path/to/CWRU_1shot_best_weight.pth"
BEST_WEIGHT_CWRU_5SHOT="/path/to/CWRU_5shot_best_weight.pth"
BEST_WEIGHT_HUST_1SHOT="/path/to/HUST_1shot_best_weight.pth"
BEST_WEIGHT_HUST_5SHOT="/path/to/HUST_5shot_best_weight.pth"

# Select the correct weight file based on mode and dataset
if [ "$MODE" -eq 1 ]; then
    TEST_SCRIPT="test_1shot.py"
    if [ "$DATASET" == "CWRU" ]; then
        BEST_WEIGHT="$BEST_WEIGHT_CWRU_1SHOT"
    else
        BEST_WEIGHT="$BEST_WEIGHT_HUST_1SHOT"
    fi
elif [ "$MODE" -eq 5 ]; then
    TEST_SCRIPT="test_5shot.py"
    if [ "$DATASET" == "CWRU" ]; then
        BEST_WEIGHT="$BEST_WEIGHT_CWRU_5SHOT"
    else
        BEST_WEIGHT="$BEST_WEIGHT_HUST_5SHOT"
    fi
else
    echo "Error: Mode must be either 1 (1-shot) or 5 (5-shot)."
    usage
fi

# Check if test script exists
if [ ! -f "$TEST_SCRIPT" ]; then
    echo "Error: Testing script '$TEST_SCRIPT' not found in the current directory."
    exit 1
fi


if [ ! -f "$BEST_WEIGHT" ]; then
    echo "Error: Best weight file '$BEST_WEIGHT' not found."
    echo "Please update the BEST_WEIGHT path in the script."
    exit 1
fi

# Display configuration
echo "========================================="
echo "Testing Configuration:"
echo "Mode             : ${MODE}-shot"
echo "Dataset          : $DATASET"
echo "Model Name       : $MODEL_NAME"
echo "Testing Script   : $TEST_SCRIPT"
echo "Best Weight Path : $BEST_WEIGHT"
echo "========================================="
echo

# Run the test
echo "Starting ${MODE}-shot testing on ${DATASET} dataset..."
python3 "$TEST_SCRIPT" --dataset "$DATASET" --best_weight "$BEST_WEIGHT" --model_name "$MODEL_NAME"
echo "Testing completed successfully for ${MODE}-shot learning on ${DATASET} dataset."
