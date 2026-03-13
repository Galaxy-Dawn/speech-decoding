#!/bin/bash
# Configuration section: Set tasks and parameters
PROJECT_DIR='Speech_Decoding'
SUBJECT_ID=("S1" "S2" "S3" "S4" "S5" "S6" "S7" "S8" "S9" "S10" "S11" "S12")
MODEL="NeuroSketch"  # Model to use
EPOCH=100  # Number of training epochs
TASKS=("speaking_initial" "speaking_final" "listening_initial" "listening_final")
TOTAL_GPUS=8
MAX_JOBS_PER_GPU=2

function get_yaml_value() {
    local file=$1
    local key=$2
    python3 -c "import yaml; print(yaml.safe_load(open('$file'))['$key'])"
}

for arg in "$@"; do
    key=$(echo "$arg" | cut -d'=' -f1)
    value=$(echo "$arg" | cut -d'=' -f2)
    if [ "$key" == "subject_id" ]; then
        IFS=',' read -r -a SUBJECT_ID <<< "$value"
        SUBJECT_ID=("${SUBJECT_ID[@]/#/}")
        SUBJECT_ID=("${SUBJECT_ID[@]/%/}")
    fi
done

for subject in "${SUBJECT_ID[@]}"; do
  yaml_file="$PROJECT_DIR/run/conf/dataset/speech_decoding_$subject.yaml"
  input_channels=$(get_yaml_value "$yaml_file" "input_channels")
  channels_array=($(seq 0 $((input_channels - 1))))
  for task in "${TASKS[@]}"; do
    for channel in "${channels_array[@]}"; do
      param_combinations+=("$subject $task $channel")
    done
  done
done

total_gpus=$TOTAL_GPUS  # Number of available GPUs
total_jobs=${#param_combinations[@]}  # Total number of tasks
if [ "$total_jobs" -eq 0 ]; then
    echo "No valid parameter combinations, skipping"
    exit 1
fi
batch_size=$((total_gpus * MAX_JOBS_PER_GPU))
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
        IFS=' ' read -r subject task channel <<< "${param_combinations[$i]}"
        gpu_id=$(( (i % total_gpus)))
        commands+=("CUDA_VISIBLE_DEVICES=$gpu_id python ${PROJECT_DIR}/run/train.py \\
            training.do_train=True \\
            training.do_eval=True \\
            training.do_predict=False \\
            training.gpu_id=$gpu_id \\
            training.overwrite_output_dir=True \\
            training.num_train_epochs=$EPOCH \\
            model=$MODEL \\
            dataset=speech_decoding_${subject} \\
            dataset.id=$subject \\
            dataset.split_method=simple \\
            dataset.task=$task \\
            dataset.channel_index_name=\"$channel\" \\
            swanlab.exp_name=\"${subject}_channel_contribution_${task}_$(echo $channel | tr '-' '_')\"")
        gpu_ids+=($gpu_id)  # Save GPU ID
    done

    # Output current batch information
    echo "===================================================================="
    echo "Executing batch $((batch+1))/$total_batches | Remaining tasks $((total_jobs - current_job)) | Subject $subject"
    echo "Occupied GPUs: ${gpu_ids[*]}"
    echo "===================================================================="

    # Launch training tasks for the current batch
    for cmd in "${commands[@]}"; do
        eval "$cmd &"  # Execute each task in the background
        ((current_job++))  # Update current task count
    done

    # Wait for the current batch to complete
    wait
    echo "Batch $((batch+1)) completed | Total completed $current_job/$total_jobs | Subject $subject"
    echo ""
done

echo "All tasks for all subjects have been completed!"
