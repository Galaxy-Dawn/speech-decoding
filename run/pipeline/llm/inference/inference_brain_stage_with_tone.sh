#!/usr/bin/env bash

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

PROJECT_DIR='Speech_Decoding'
TOTAL_GPUS=4  # Number of available GPUs
MAX_JOBS_PER_GPU=1

SUBJECTS=("S1" "S2" "S3" "S4" "S5" "S6" "S7" "S8" "S9" "S10" "S11" "S12")
STATE=("speaking" "listening")
MODEL=("NeuroSketch" "ModernTCN" "MedFormer" "MultiResGRU")
SEEDS=(42 43 44 45 46 47 48 49 50 51)

# Initialize total start time
total_start=$(get_time)

param_combinations=()

for state in "${STATE[@]}"; do
    for subject in "${SUBJECTS[@]}"; do
        for model in "${MODEL[@]}"; do
          for seed in "${SEEDS[@]}"; do
            param_combinations+=("$subject $state $model $seed")
          done
        done
    done
done

# Set training resources: number of GPUs, tasks, and batch size
total_jobs=${#param_combinations[@]}  # Total number of jobs
if [ "$total_jobs" -eq 0 ]; then
    echo "No valid parameter combinations found, skipping"
    exit 1
fi
batch_size=$((TOTAL_GPUS * MAX_JOBS_PER_GPU))
# Calculate total number of batches
current_job=0
total_batches=$(( (total_jobs + batch_size - 1) / batch_size ))

# Execute tasks in batches
for ((batch=0; batch<total_batches; batch++)); do
    start=$((batch * batch_size))
    end=$((start + batch_size))
    # Generate GPU commands for the current batch
    commands=()
    gpu_ids=()
    for ((i=start; i<end && i<total_jobs; i++)); do
        IFS=' ' read -r subject state model seed <<< "${param_combinations[$i]}"
        gpu_id=$((i % TOTAL_GPUS+4))  # Assign GPU

        # Check if subject is S11 or S12
        if [[ "$subject" == "S11" || "$subject" == "S12" ]]; then
            extra_lexicon_flag="True"
        else
            extra_lexicon_flag="False"
        fi

        commands+=("CUDA_VISIBLE_DEVICES=$gpu_id python ${PROJECT_DIR}/run/pipeline/llm/inference/Brain_stage.py \\
                    test.is_extra_lexicon=${extra_lexicon_flag} \\
                    test.decode_with_tone=True \\
                    test.state=${state} \\
                    test.seed=${seed} \\
                    model=${model} \\
                    dataset=speech_decoding_${subject}")
        gpu_ids+=($gpu_id)  # Save GPU ID
    done

    # Output current batch information
    echo "===================================================================="
    echo "Executing batch $((batch+1))/$total_batches | Remaining tasks $((total_jobs - current_job)) | Subject $id"
    echo "GPUs used: ${gpu_ids[*]}"
    echo "===================================================================="
    # Start training tasks for current batch
    for cmd in "${commands[@]}"; do
        eval "$cmd &"  # Execute each task in background
        ((current_job++))  # Update current job count
    done
    # Wait for current batch to complete
    wait
    echo "Batch $((batch+1)) completed | Total completed $current_job/$total_jobs"
    echo ""
done
echo "All tasks completed! Total executed combinations: $total_jobs"

# Total duration
total_end=$(get_time)
print_duration $total_start $total_end "Total Execution"

echo "All steps completed."