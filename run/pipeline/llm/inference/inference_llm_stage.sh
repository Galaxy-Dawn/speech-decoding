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
INFERENCE_MODULE="all_in_two"
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
            if [[ "${INFERENCE_MODULE}" == "all_in_one" ]]; then
                echo "Skip 1 Step Listwise"
            else
                # Step 1: Prepare Listwise data
                echo "Subject ${subject}, state ${state}: Running Listwise_stage.py..."
                step1_start=$(get_time)
                python ${PROJECT_DIR}/run/pipeline/llm/inference/Listwise_stage.py \
                    inference_module=${INFERENCE_MODULE} \
                    dataset=speech_decoding_${subject} \
                    test.state=${state} \
                    test.seed=${seed}
                step1_end=$(get_time)
                print_duration $step1_start $step1_end "Listwise_stage.py (Subject ${subject}, state ${state})"

                # Step 2: Listwise_infer.sh
                echo "Subject ${subject}, state ${state}: Running Listwise_infer.sh..."
                step2_start=$(get_time)
                source "${PROJECT_DIR}/run/pipeline/llm/inference/Listwise_infer.sh"
                run_listwise_inference "${PROJECT_DIR}/run/conf/inference_module/${INFERENCE_MODULE}.yaml" ${subject} ${state}
                step2_end=$(get_time)
                print_duration $step2_start $step2_end "Listwise_infer.sh (Subject ${subject}, state ${state})"
            fi

             Step 3: Prepare Correction data
            echo "Subject ${subject}, state ${state}: Running Correction_stage.py..."
            step3_start=$(get_time)
            python ${PROJECT_DIR}/run/pipeline/llm/inference/Correction_stage.py \
                inference_module=${INFERENCE_MODULE} \
                dataset=speech_decoding_${subject} \
                test.state=${state} \
                test.seed=${seed}
            step3_end=$(get_time)
            print_duration $step3_start $step3_end "Correction_stage.py (Subject ${subject}, state ${state})"

            # Step 4: Correction_infer.sh
            echo "Subject ${subject}, state ${state}: Running Correction_infer.sh..."
            step4_start=$(get_time)
            source "${PROJECT_DIR}/run/pipeline/llm/inference/Correction_infer.sh"
            run_correction_inference "${PROJECT_DIR}/run/conf/inference_module/${INFERENCE_MODULE}.yaml" ${subject} ${state}
            step4_end=$(get_time)
            print_duration $step4_start $step4_end "Correction_infer.sh (Subject ${subject}, state ${state})"

            # Step 5: Summarize Results
            echo "Subject ${subject}, state ${state}: Summarizing Results..."
            step5_start=$(get_time)
            python ${PROJECT_DIR}/run/pipelne/llm/inference/check_results.py \
                inference_module=${INFERENCE_MODULE} \
                dataset=speech_decoding_${subject} \
                test.state=${state} \
                test.seed=${seed}
            step5_end=$(get_time)
            print_duration $step5_start $step5_end "Summarize Results (Subject ${subject}, state ${state})"

            echo "=================================================="
        else
            echo "Skipping subject ${subject} because it is not in the list of subjects for state ${state}."
        fi

      done
    done
    echo "########## Completed Subject: ${subject} ##########"
done

# Total duration
total_end=$(get_time)
print_duration $total_start $total_end "Total Execution"

echo "All steps completed."