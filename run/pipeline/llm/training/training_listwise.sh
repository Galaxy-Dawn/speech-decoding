#!/usr/bin/env bash

# Define variables
PROJECT_DIR='Speech_Decoding'
DATASET_DIR="" #LLM training data directory, the same as /run/conf/dir/local.yaml llm_data_dir
LLM_SAVE_DIR="" #LLM save directory, the same as /run/conf/dir/local.yaml llm_save_dir
DATASET_NAME="syllable_direct_rerank_uni"
MODEL_PATH="${LLM_SAVE_DIR}/Qwen7B_translation/lora"
OUTPUT_PATH="${LLM_SAVE_DIR}/Qwen7B_rerank"


CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --main_process_port=29501 \
    ${PROJECT_DIR}/src/llm/LLaMA-Factory/src/train.py \
    --stage sft \
    --do_train True \
    --model_name_or_path ${MODEL_PATH} \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen \
    --flash_attn fa2 \
    --enable_liger_kernel True \
    --use_unsloth False \
    --dataset_dir ${DATASET_DIR} \
    --dataset ${DATASET_NAME}_train \
    --val_size 0.025 \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 100 \
    --warmup_ratio 0.05 \
    --optim adamw_torch \
    --packing True \
    --neat_packing True \
    --report_to none \
    --output_dir ${OUTPUT_PATH} \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target all \
    --resize_vocab True \
    --additional_target embedding,lm_head

wait

llamafactory-cli export \
    --model_name_or_path ${MODEL_PATH} \
    --adapter_name_or_path ${OUTPUT_PATH} \
    --template qwen \
    --finetuning_type lora \
    --export_dir ${OUTPUT_PATH}/lora \
    --export_legacy_format True \
    --resize_vocab True \

wait

CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train \
    --stage sft \
    --model_name_or_path ${OUTPUT_PATH}/lora \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method awq \
    --template qwen \
    --flash_attn fa2 \
    --dataset_dir ${DATASET_DIR} \
    --eval_dataset ${DATASET_NAME}_eval \
    --cutoff_len 2048 \
    --max_samples 10000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --max_new_tokens 3 \
    --top_p 0.7 \
    --temperature 0.5 \
    --output_dir ${OUTPUT_PATH}/eval \
    --do_predict True \
    --report_to none \
    --seed 2025

wait


CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train \
    --stage sft \
    --model_name_or_path ${OUTPUT_PATH}/lora \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method awq \
    --template qwen \
    --flash_attn fa2 \
    --dataset_dir ${DATASET_DIR} \
    --eval_dataset ${DATASET_NAME}_train \
    --cutoff_len 2048 \
    --max_samples 1000000000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --max_new_tokens 3 \
    --top_p 0.7 \
    --temperature 0.5 \
    --output_dir ${OUTPUT_PATH}/train \
    --do_predict True \
    --report_to none \
    --seed 2025