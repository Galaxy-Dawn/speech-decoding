#!/usr/bin/env bash

PROJECT_DIR="Speech_Decoding"

python ${PROJECT_DIR}/run/pipeline/llm/training/data_preprocessing.py \

python ${PROJECT_DIR}/run/pipeline/llm/training/get_translation_and_listwise_training_data.py \

bash ${PROJECT_DIR}/run/pipeline/llm/training/training_translation.sh \

bash ${PROJECT_DIR}/run/pipeline/llm/training/training_listwise.sh \

python ${PROJECT_DIR}/run/pipeline/llm/training/get_correction_training_data.py \

bash ${PROJECT_DIR}/run/pipeline/llm/training/training_correction.sh
