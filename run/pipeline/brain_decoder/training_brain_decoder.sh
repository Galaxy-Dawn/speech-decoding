#!/bin/bash

PROJECT_DIR='Speech_Decoding'
DATA_DIR='' #Raw data directory, the same as /run/conf/dir/local.yaml data_dir
SUBJECT_ID=("S1" "S2" "S3" "S4" "S5" "S6" "S7" "S8" "S9" "S10" "S11" "S12")
MODELS=("NeuroSketch" "ModernTCN" "MedFormer" "MultiResGRU")  # Models used
EPOCH=500  # Number of training epochs
SWA_START=250
AGG_START_EPOCH=250
AGG_FREQ=1
TASKS=("speaking_initial" "speaking_final" "listening_initial" "listening_final")
total_gpus=4  # Number of available GPUs
MAX_JOBS_PER_GPU=2


for arg in "$@"; do
    key=$(echo "$arg" | cut -d'=' -f1)
    value=$(echo "$arg" | cut -d'=' -f2)
    if [ "$key" == "subject_id" ]; then
        IFS=',' read -r -a SUBJECT_ID <<< "$value"
        SUBJECT_ID=("${SUBJECT_ID[@]/#/}")
        SUBJECT_ID=("${SUBJECT_ID[@]/%/}")
    elif [ "$key" == "agg_start_epoch" ]; then  # Add handling for agg_start_epoch parameter
        AGG_START_EPOCH=$value
    fi
done

param_combinations=()

# Iterate over each subject ID
for id in "${SUBJECT_ID[@]}"; do
  for task in "${TASKS[@]}"; do
    output_results_file="$DATA_DIR/$id/${task}_channel_selection_results.txt"
    if [ ! -f "$output_results_file" ]; then
      echo "Result file not found for subject $id: $output_results_file"
      continue
    fi
    line=$(grep "^$task:" "$output_results_file")
    if [ -z "$line" ]; then
        echo "Channel list not found for task $task, subject $id"
        continue
    fi
    channels_str=${line#*: }  # Remove "task: "
    channels_str=${channels_str//\[/}  # Remove [
    channels_str=${channels_str//\]/}  # Remove ]
    channels_str=${channels_str// /}  # Remove all spaces
    channels_str=${channels_str//,/-}  # Replace commas with hyphens
    for model in "${MODELS[@]}"; do
        param_combinations+=("$id $task $channels_str $model")
    done
  done
done

# Set training resources: number of GPUs, number of tasks, and batch size
total_jobs=${#param_combinations[@]}  # Total number of tasks
if [ "$total_jobs" -eq 0 ]; then
    echo "No valid parameter combinations, skipping"
    exit 1
fi
batch_size=$((total_gpus * MAX_JOBS_PER_GPU))
# Calculate total number of batches
current_job=0
total_batches=$(( (total_jobs + batch_size - 1) / batch_size ))

# Execute tasks in batches
for ((batch=0; batch<total_batches; batch++)); do
    start=$((batch * batch_size))
    end=$((start + batch_size))

    # Generate GPU commands for current batch
    commands=()
    gpu_ids=()
    for ((i=start; i<end && i<total_jobs; i++)); do
        IFS=' ' read -r id task channels_str model<<< "${param_combinations[$i]}"
        gpu_id=$((i % total_gpus))  # Assign GPU
        commands+=("CUDA_VISIBLE_DEVICES=$gpu_id python $PROJECT_DIR/run/train.py \\
            training.overwrite_output_dir=True \\
            training.do_train=True \\
            training.do_eval=False \\
            training.do_predict=True \\
            training.gpu_id=$gpu_id \\
            training.num_train_epochs=$EPOCH \\
            training.swa_start=$SWA_START \\
            training.save_start_epoch=$AGG_START_EPOCH \\
            training.save_end_epoch=500 \\
            training.save_total_limit=100000 \\
            training.agg_start_epoch=$AGG_START_EPOCH \\
            training.agg_end_epoch=500 \\
            training.agg_freq=$AGG_FREQ \\
            training.evaluation_strategy=no \\
            training.save_strategy=no \\
            model=$model \\
            dataset=speech_decoding_${id} \\
            dataset.id=$id \\
            dataset.task=$task \\
            dataset.split_method=none \\
            dataset.channel_index_name=\"$channels_str\" \\
            swanlab.exp_name=\"${model}_${id}_${task}\"")
        gpu_ids+=($gpu_id)  # Save GPU ID
    done

    # Output current batch information
    echo "===================================================================="
    echo "Executing batch $((batch+1))/$total_batches | Remaining tasks $((total_jobs - current_job)) | "
    echo "Using GPUs: ${gpu_ids[*]}"
    echo "===================================================================="
    # Start training tasks for current batch
    for cmd in "${commands[@]}"; do
        eval "$cmd &"  # Execute each task in background
        ((current_job++))  # Update current task count
    done
    # Wait for current batch to complete
    wait
    echo "Batch $((batch+1)) completed | Total completed $current_job/$total_jobs"
    echo ""
done
echo "All tasks completed! Total combinations executed: $total_jobs"