#!/bin/bash

usage() {
    echo "Usage: $0 <mode>"
    echo "  <mode>    Testing mode: 1 for 1-shot or 5 for 5-shot"
    echo
    echo "Examples:"
    echo "  bash $0 1    # Runs 1-shot testing"
    echo "  bash $0 5    # Runs 5-shot testing"
    exit 1
}

if [ "$#" -ne 1 ]; then
    echo "Error: Invalid number of arguments."
    usage
fi

MODE=$1
DATASET="CWRU"
MODEL_NAME="Net"


# 🛠️ **IMPORTANT:** Update these paths to point to actual best weight files
BEST_WEIGHT_1SHOT="/path/to/1shot_best_weight.pth"
BEST_WEIGHT_5SHOT="/path/to/5shot_best_weight.pth"


if [ "$MODE" -eq 1 ]; then
    TEST_SCRIPT="test_1shot.py"
    BEST_WEIGHT="$BEST_WEIGHT_1SHOT"
elif [ "$MODE" -eq 5 ]; then
    TEST_SCRIPT="test_5shot.py"
    BEST_WEIGHT="$BEST_WEIGHT_5SHOT"
else
    echo "Error: Mode must be either 1 (1-shot) or 5 (5-shot)."
    usage
fi


if [ ! -f "$TEST_SCRIPT" ]; then
    echo "Error: Testing script '$TEST_SCRIPT' not found in the current directory."
    exit 1
fi


if [ ! -f "$BEST_WEIGHT" ]; then
    echo "Error: Best weight file '$BEST_WEIGHT' not found."
    echo "Please update the BEST_WEIGHT path in the script."
    exit 1
fi

echo "========================================="
echo "Testing Configuration:"
echo "Mode             : ${MODE}-shot"
echo "Dataset          : $DATASET"
echo "Model Name       : $MODEL_NAME"
echo "Testing Script   : $TEST_SCRIPT"
echo "Best Weight Path : $BEST_WEIGHT"
echo "========================================="
echo


echo "Starting ${MODE}-shot testing..."
python3 "$TEST_SCRIPT" --dataset "$DATASET" --best_weight "$BEST_WEIGHT" --model_name "$MODEL_NAME"
echo "Testing completed successfully for ${MODE}-shot learning."
