#!/bin/bash
PROJECT_DIR='Speech_Decoding'
SUBJECT_ID=('S1' 'S2' 'S3' 'S4' 'S5' 'S6' 'S7' 'S8' 'S9' 'S10' 'S11' 'S12')  # Subject ID list

for arg in "$@"; do
    key=$(echo "$arg" | cut -d'=' -f1)
    value=$(echo "$arg" | cut -d'=' -f2)
    if [ "$key" == "subject_id" ]; then
        IFS=',' read -r -a SUBJECT_ID <<< "$value"
        SUBJECT_ID=("${SUBJECT_ID[@]/#/}")
        SUBJECT_ID=("${SUBJECT_ID[@]/%/}")
    fi
done

# Iterate over each subject ID
for id in "${SUBJECT_ID[@]}"; do
    # Generate data for each subject
    echo "Start generating data for subject ${id}..."
    python ${PROJECT_DIR}/run/prepare_data/generate_yaml.py subject_id=${id}
    python ${PROJECT_DIR}/run/prepare_data/prepare_data.py -m \
          dataset=speech_decoding_${id} \
          dataset.id=${id} \
          dataset.task=speaking_initial,speaking_final,speaking_tone,listening_initial,listening_final,listening_tone \
          dataset.split_method=simple
    echo "Data generation for subject ${id} completed successfully"
done