#!/bin/bash

usage() {
    echo "Usage: $0 <mode> <dataset>"
    echo "  <mode>      Training mode: 1 for 1-shot or 5 for 5-shot"
    echo "  <dataset>   Dataset selection: 'CWRU' or 'HUST'"
    echo
    echo "Examples:"
    echo "  bash $0 1 CWRU    # Runs 1-shot training on CWRU dataset"
    echo "  bash $0 5 HUST    # Runs 5-shot training on HUST dataset"
    exit 1
}

if [ "$#" -ne 2 ]; then
    echo "Error: Invalid number of arguments."
    usage
fi

MODE=$1
DATASET=$2

# Validate dataset selection
if [[ "$DATASET" != "CWRU" && "$DATASET" != "HUST" ]]; then
    echo "Error: Dataset must be 'CWRU' or 'HUST'."
    usage
fi

# Config parameters
MODEL_NAME="few-shot-structural-rep"

# Define training samples for each dataset
TRAINING_SAMPLES_CWRU="60"
TRAINING_SAMPLES_HUST="80"  # Change if needed

# Select training samples based on dataset
if [ "$DATASET" == "CWRU" ]; then
    TRAINING_SAMPLES="$TRAINING_SAMPLES_CWRU"
    TRAINING_SAMPLES_OPTION="--training_samples_CWRU $TRAINING_SAMPLES"
else
    TRAINING_SAMPLES="$TRAINING_SAMPLES_HUST"
    TRAINING_SAMPLES_OPTION="--training_samples_HUST $TRAINING_SAMPLES"
fi

# Choose the appropriate training script
if [ "$MODE" -eq 1 ]; then
    TRAIN_SCRIPT="train_1shot.py"
elif [ "$MODE" -eq 5 ]; then
    TRAIN_SCRIPT="train_5shot.py"
else
    echo "Error: Mode must be either 1 (1-shot) or 5 (5-shot)."
    usage
fi

# Check if training script exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script '$TRAIN_SCRIPT' not found in the current directory."
    exit 1
fi

# Display configuration
echo "========================================="
echo "Training Configuration:"
echo "Mode             : ${MODE}-shot"
echo "Dataset          : $DATASET"
echo "Training Samples : $TRAINING_SAMPLES"
echo "Model Name       : $MODEL_NAME"
echo "Training Script  : $TRAIN_SCRIPT"
echo "========================================="
echo

# Run the training
echo "Starting ${MODE}-shot training on ${DATASET} dataset..."
python3 "$TRAIN_SCRIPT" --dataset "$DATASET" $TRAINING_SAMPLES_OPTION --model_name "$MODEL_NAME"
echo "Training completed successfully for ${TRAINING_SAMPLES} training samples with ${MODE}-shot learning on ${DATASET} dataset."
