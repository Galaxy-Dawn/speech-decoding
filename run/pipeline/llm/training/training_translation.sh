#!/usr/bin/env bash

# Define variables
PROJECT_DIR='Speech_Decoding'
DATASET_VERSION="uni"
DATASET_DIR="" #LLM training data directory, the same as /run/conf/dir/local.yaml llm_data_dir
LLM_SAVE_DIR="" # LLM save directory, the same as /run/conf/dir/local.yaml llm_save_dir
MODEL_PATH="" # Path to the Qwen2.5 7B model checkpoint
OUTPUT_PATH="${LLM_SAVE_DIR}/Qwen7B_translation"

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
    --dataset syllable_translation_${DATASET_VERSION} \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 1000000 \
    --per_device_train_batch_size 16 \
    --val_size 0.025 \
    --per_device_eval_batch_size 16 \
    --eval_strategy steps \
    --eval_steps 100 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
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
    --add_tokens beng,biao,cang,ceng,chao,chong,chou,chuai,chuan,chuang,chui,chun,chuo,cong,cuan,cui,cun,cuo,dai,dang,dei,deng,dian,diao,diu,dou,duan,dui,dun,duo,fei,feng,fou,gai,gao,gei,geng,gong,gou,guai,guan,guang,guo,hao,heng,hong,huai,huan,huang,hui,hun,huo,jia,jiao,jiong,jiu,jue,kai,kang,kao,keng,kong,kou,kou,kua,kuai,kuan,kuang,kui,kun,kuo,lai,lao,leng,liang,liao,liu,lou,luan,lue,lun,luo,lve,mang,mao,mei,meng,mian,miao,miu,mou,nai,nang,nao,nei,neng,nian,niang,niao,niu,nong,nuan,nue,nuo,nve,pang,pao,pian,piao,pou,qia,qian,qiang,qiao,qie,qin,qiong,qiu,qun,qve,rao,reng,ruan,rui,ruo,sai,sang,sao,seng,shai,shang,shao,shei,shen,sheng,shou,shu,shua,shuai,shuan,shuang,shui,shun,shuo,sou,suan,sui,suo,tai,tang,tao,teng,tian,tiao,tong,tou,tuan,tui,tun,tuo,wai,weng,xian,xiang,xiao,xie,xiong,xiu,xu,xuan,xue,xun,yao,yin,yong,yuan,yue,yve,zai,zang,zao,zei,zeng,zha,zhai,zhan,zhang,zhao,zhe,zhen,zheng,zhi,zhong,zhu,zhua,zhuai,zhuan,zhuang,zhui,zhun,zhuo,zong,zou,zuan,zui,zun,zuo,biang,biu,dare,duang,giao,hia,jio,jiou,jou,kiang,kio,kiong,kira,kiu,mea,miang,mio,mua,nou,pia,rua,suai,wry,wua,Ġbai,Ġbao,Ġbeng,Ġbiao,Ġbie,Ġbing,Ġcai,Ġcang,Ġcao,Ġceng,Ġchao,Ġchen,Ġcheng,Ġchong,Ġchou,Ġchuai,Ġchuan,Ġchuang,Ġchui,Ġchun,Ġchuo,Ġcuan,Ġcun,Ġcuo,Ġdeng,Ġdian,Ġdiao,Ġdiu,Ġduan,Ġdui,Ġfang,Ġfei,Ġfeng,Ġgai,Ġgao,Ġgei,Ġgeng,Ġgong,Ġgua,Ġguai,Ġguan,Ġguang,Ġguo,Ġhao,Ġheng,Ġhong,Ġhou,Ġhua,Ġhuai,Ġhuan,Ġhuang,Ġhui,Ġhuo,Ġjia,Ġjian,Ġjiang,Ġjiao,Ġjie,Ġjin,Ġjing,Ġjiong,Ġjiu,Ġjuan,Ġkai,Ġkeng,Ġkong,Ġkou,Ġkou,Ġkua,Ġkuai,Ġkuan,Ġkuang,Ġkui,Ġkuo,Ġlai,Ġlao,Ġlian,Ġliang,Ġliao,Ġliu,Ġluan,Ġlue,Ġluo,Ġlve,Ġmao,Ġmei,Ġmian,Ġmiao,Ġmiu,Ġnai,Ġnang,Ġneng,Ġnian,Ġniang,Ġniao,Ġniu,Ġnong,Ġnuan,Ġnuo,Ġnve,Ġpao,Ġpei,Ġpiao,Ġqia,Ġqian,Ġqiang,Ġqiao,Ġqie,Ġqin,Ġqing,Ġqiong,Ġqiu,Ġquan,Ġqun,Ġqve,Ġrao,Ġreng,Ġrong,Ġruan,Ġrui,Ġruo,Ġsao,Ġseng,Ġshai,Ġshan,Ġshang,Ġshao,Ġshei,Ġshen,Ġsheng,Ġshi,Ġshou,Ġshu,Ġshua,Ġshuai,Ġshuan,Ġshuang,Ġshui,Ġshun,Ġshuo,Ġsuan,Ġtao,Ġtian,Ġtiao,Ġtuan,Ġtui,Ġwai,Ġweng,Ġwu,Ġxia,Ġxian,Ġxiang,Ġxiao,Ġxie,Ġxin,Ġxing,Ġxiong,Ġxiu,Ġxuan,Ġxue,Ġxun,Ġyao,Ġyin,Ġying,Ġyong,Ġyu,Ġyue,Ġyun,Ġyve,Ġzai,Ġzan,Ġzang,Ġzao,Ġzei,Ġzeng,Ġzha,Ġzhai,Ġzhan,Ġzhang,Ġzhao,Ġzhe,Ġzhen,Ġzheng,Ġzhi,Ġzhong,Ġzhou,Ġzhu,Ġzhua,Ġzhuai,Ġzhuan,Ġzhuang,Ġzhui,Ġzhun,Ġzhuo,Ġzong,Ġzuan,Ġzui,Ġzuo,Ġbiang,Ġbiu,Ġduang,Ġgiao,Ġhia,Ġjio,Ġjiou,Ġkiang,Ġkio,Ġkiong,Ġkira,Ġkiu,Ġmea,Ġmiang,Ġmua,Ġpia,Ġsuai,Ġwry,Ġwua \
    --additional_target embed_tokens,embedding,lm_head

wait

llamafactory-cli export \
    --model_name_or_path ${MODEL_PATH} \
    --adapter_name_or_path ${OUTPUT_PATH} \
    --template qwen \
    --finetuning_type lora \
    --export_dir ${OUTPUT_PATH}/lora \
    --export_legacy_format True \
    --resize_vocab True \
    --add_tokens beng,biao,cang,ceng,chao,chong,chou,chuai,chuan,chuang,chui,chun,chuo,cong,cuan,cui,cun,cuo,dai,dang,dei,deng,dian,diao,diu,dou,duan,dui,dun,duo,fei,feng,fou,gai,gao,gei,geng,gong,gou,guai,guan,guang,guo,hao,heng,hong,huai,huan,huang,hui,hun,huo,jia,jiao,jiong,jiu,jue,kai,kang,kao,keng,kong,kou,kou,kua,kuai,kuan,kuang,kui,kun,kuo,lai,lao,leng,liang,liao,liu,lou,luan,lue,lun,luo,lve,mang,mao,mei,meng,mian,miao,miu,mou,nai,nang,nao,nei,neng,nian,niang,niao,niu,nong,nuan,nue,nuo,nve,pang,pao,pian,piao,pou,qia,qian,qiang,qiao,qie,qin,qiong,qiu,qun,qve,rao,reng,ruan,rui,ruo,sai,sang,sao,seng,shai,shang,shao,shei,shen,sheng,shou,shu,shua,shuai,shuan,shuang,shui,shun,shuo,sou,suan,sui,suo,tai,tang,tao,teng,tian,tiao,tong,tou,tuan,tui,tun,tuo,wai,weng,xian,xiang,xiao,xie,xiong,xiu,xu,xuan,xue,xun,yao,yin,yong,yuan,yue,yve,zai,zang,zao,zei,zeng,zha,zhai,zhan,zhang,zhao,zhe,zhen,zheng,zhi,zhong,zhu,zhua,zhuai,zhuan,zhuang,zhui,zhun,zhuo,zong,zou,zuan,zui,zun,zuo,biang,biu,dare,duang,giao,hia,jio,jiou,jou,kiang,kio,kiong,kira,kiu,mea,miang,mio,mua,nou,pia,rua,suai,wry,wua,Ġbai,Ġbao,Ġbeng,Ġbiao,Ġbie,Ġbing,Ġcai,Ġcang,Ġcao,Ġceng,Ġchao,Ġchen,Ġcheng,Ġchong,Ġchou,Ġchuai,Ġchuan,Ġchuang,Ġchui,Ġchun,Ġchuo,Ġcuan,Ġcun,Ġcuo,Ġdeng,Ġdian,Ġdiao,Ġdiu,Ġduan,Ġdui,Ġfang,Ġfei,Ġfeng,Ġgai,Ġgao,Ġgei,Ġgeng,Ġgong,Ġgua,Ġguai,Ġguan,Ġguang,Ġguo,Ġhao,Ġheng,Ġhong,Ġhou,Ġhua,Ġhuai,Ġhuan,Ġhuang,Ġhui,Ġhuo,Ġjia,Ġjian,Ġjiang,Ġjiao,Ġjie,Ġjin,Ġjing,Ġjiong,Ġjiu,Ġjuan,Ġkai,Ġkeng,Ġkong,Ġkou,Ġkou,Ġkua,Ġkuai,Ġkuan,Ġkuang,Ġkui,Ġkuo,Ġlai,Ġlao,Ġlian,Ġliang,Ġliao,Ġliu,Ġluan,Ġlue,Ġluo,Ġlve,Ġmao,Ġmei,Ġmian,Ġmiao,Ġmiu,Ġnai,Ġnang,Ġneng,Ġnian,Ġniang,Ġniao,Ġniu,Ġnong,Ġnuan,Ġnuo,Ġnve,Ġpao,Ġpei,Ġpiao,Ġqia,Ġqian,Ġqiang,Ġqiao,Ġqie,Ġqin,Ġqing,Ġqiong,Ġqiu,Ġquan,Ġqun,Ġqve,Ġrao,Ġreng,Ġrong,Ġruan,Ġrui,Ġruo,Ġsao,Ġseng,Ġshai,Ġshan,Ġshang,Ġshao,Ġshei,Ġshen,Ġsheng,Ġshi,Ġshou,Ġshu,Ġshua,Ġshuai,Ġshuan,Ġshuang,Ġshui,Ġshun,Ġshuo,Ġsuan,Ġtao,Ġtian,Ġtiao,Ġtuan,Ġtui,Ġwai,Ġweng,Ġwu,Ġxia,Ġxian,Ġxiang,Ġxiao,Ġxie,Ġxin,Ġxing,Ġxiong,Ġxiu,Ġxuan,Ġxue,Ġxun,Ġyao,Ġyin,Ġying,Ġyong,Ġyu,Ġyue,Ġyun,Ġyve,Ġzai,Ġzan,Ġzang,Ġzao,Ġzei,Ġzeng,Ġzha,Ġzhai,Ġzhan,Ġzhang,Ġzhao,Ġzhe,Ġzhen,Ġzheng,Ġzhi,Ġzhong,Ġzhou,Ġzhu,Ġzhua,Ġzhuai,Ġzhuan,Ġzhuang,Ġzhui,Ġzhun,Ġzhuo,Ġzong,Ġzuan,Ġzui,Ġzuo,Ġbiang,Ġbiu,Ġduang,Ġgiao,Ġhia,Ġjio,Ġjiou,Ġkiang,Ġkio,Ġkiong,Ġkira,Ġkiu,Ġmea,Ġmiang,Ġmua,Ġpia,Ġsuai,Ġwry,Ġwua