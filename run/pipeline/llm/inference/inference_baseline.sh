#!/usr/bin/env bash

# Function: Print current timestamp (seconds)
get_time() {
    date +%s
}

# Function: Calculate and output duration
print_duration() {
    local start=$1
    local end=$2
    local step_name=$3
    local duration=$((end - start))
    echo "[$step_name] Duration: $duration seconds"
}

export CUDA_VISIBLE_DEVICES=4,5,6,7

declare -A SUBJECTS_BY_STATE
SUBJECTS_BY_STATE[speaking]="S1 S5 S6 S7 S9 S11 S12"
SUBJECTS_BY_STATE[listening]="S4 S5 S6 S7 S11 S12"

PROJECT_DIR='Speech_Decoding'
SUBJECTS=("S1" "S2" "S3" "S4" "S5" "S6" "S7" "S8" "S9" "S10" "S11" "S12")
STATE=("speaking" "listening")
SEEDS=(42 43 44 45 46 47 48 49 50 51)

# Initialize total start time
total_start=$(get_time)

# Loop through all subjects
for subject in "${SUBJECTS[@]}"
do
    echo "########## Processing Subject: ${subject} ##########"
    # Loop through all states
    for state in "${STATE[@]}"
    do
      for seed in "${SEEDS[@]}"
      do
        echo "========== Subject: ${subject}, state ${state}, seed ${seed} =========="
        if [[ "${SUBJECTS_BY_STATE[$state]}" == *"$subject"* ]]; then
          python ${PROJECT_DIR}/run/pipeline/llm/inference/baseline.py \
              dataset=speech_decoding_${subject} \
              test.state=${state} \
              test.seed=${seed}
        fi
      done
    done
    echo "########## Completed Subject: ${subject} ##########"
done

# Total duration
total_end=$(get_time)
print_duration $total_start $total_end "Total Execution"

echo "All steps completed."