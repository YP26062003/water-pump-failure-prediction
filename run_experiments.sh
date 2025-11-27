#!/bin/bash

# Define constant input filenames
TRAIN_INPUT="training-input.csv"
TRAIN_LABELS="training-labels.csv"
TEST_INPUT="test-input.csv"

# Define preprocessing options
NUMERICAL_OPTIONS=("None" "StandardScaler")
CATEGORICAL_OPTIONS=("OneHotEncoder" "OrdinalEncoder" "TargetEncoder")
MODELS=("LogisticRegression" "RandomForestClassifier" "GradientBoostingClassifier" "HistGradientBoostingClassifier" "MLPClassifier")

# Create an output directory
#OUTPUT_DIR="outputs"
#mkdir -p "$OUTPUT_DIR"

# Loop over all combinations
for NUM_PRE in "${NUMERICAL_OPTIONS[@]}"; do
    for CAT_PRE in "${CATEGORICAL_OPTIONS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            OUTPUT_FILE="pred_${NUM_PRE}_${CAT_PRE}_${MODEL}.csv"
            echo "Running: NumPre=$NUM_PRE, CatPre=$CAT_PRE, Model=$MODEL"
            python train-and-predict.py "$TRAIN_INPUT" "$TRAIN_LABELS" "$TEST_INPUT" "$NUM_PRE" "$CAT_PRE" "$MODEL" "$OUTPUT_FILE"
        done
    done
done

echo "All experiments completed. Results saved in outputs/"
