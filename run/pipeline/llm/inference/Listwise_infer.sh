#!/usr/bin/env bash

LLM_DATA_DIR="" #LLM training data directory, the same as /run/conf/dir/local.yaml llm_data_dir

run_listwise_inference() {
    # Parameter explanation:
    # $1: Config file path (default uses original path)
    local CONFIG_FILE=$1
    local SUBJECT=$2
    local STATE=$3

    echo "Using config file: $CONFIG_FILE"
    echo "Using subject: $SUBJECT"
    echo "Using state: $STATE"

    MODEL_NAME=$(yq e '.listwise.model_name' "$CONFIG_FILE")
    DATASET_NAME=$(yq e '.listwise.dataset_name' "$CONFIG_FILE")
    MODEL_PATH=$(yq e '.listwise.model_path' "$CONFIG_FILE")
    OUTPUT_PATH=$(yq e '.listwise.output_path' "$CONFIG_FILE")

    local MODEL_PATH=${MODEL_PATH/\$\{inference_module.listwise.model_name\}/$MODEL_NAME}
    local DATASET_NAME=${DATASET_NAME/\$\{test.subject\}/$SUBJECT}
    local DATASET_NAME=${DATASET_NAME/\$\{test.state\}/$STATE}
    local OUTPUT_PATH=${OUTPUT_PATH/\$\{test.subject\}/$SUBJECT}
    local OUTPUT_PATH=${OUTPUT_PATH/\$\{test.state\}/$STATE}
    local OUTPUT_PATH=${OUTPUT_PATH/\$\{inference_module.listwise.model_name\}/$MODEL_NAME}

    # Parameter validation
    if [[ -z "$MODEL_PATH" ]]; then
        echo "ERROR: model_path is empty in config file" >&2
        return 1
    fi

    echo "Starting listwise inference with:"
    echo "  - Dataset: $DATASET_NAME"
    echo "  - Model: $MODEL_PATH"
    echo "  - Output: $OUTPUT_PATH"

    # Execute training command
    llamafactory-cli train \
        --stage sft \
        --model_name_or_path "${MODEL_PATH}" \
        --preprocessing_num_workers 16 \
        --finetuning_type lora \
        --quantization_method awq \
        --template qwen \
        --flash_attn auto \
        --dataset_dir "${LLM_DATA_DIR}" \
        --eval_dataset "${DATASET_NAME}" \
        --cutoff_len 2048 \
        --per_device_eval_batch_size 1 \
        --predict_with_generate true \
        --max_new_tokens 3 \
        --do_sample false \
        --top_p 0.0 \
        --temperature 0.0 \
        --output_dir "${OUTPUT_PATH}" \
        --do_train false \
        --do_predict true \
        --seed 2025 \
        --report_to none

    local ret=$?
    if [[ $ret -ne 0 ]]; then
        echo "Inference failed with code $ret" >&2
        return $ret
    fi

    echo "Listwise inference completed successfully"
    return 0
}