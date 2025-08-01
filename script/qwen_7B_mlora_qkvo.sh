SCRIPT_PATH=mlora_finetune_copy.py
# DATA_PATH=""  # .../commonsense_170k_taskid.json
# CACHE_DIR=""  # Cache directory is not used in this script
# DEEPSPEED_CONFIG=config/ds2.json
# OUTPUT_PATH=""  # Output directory is not used in this script
DATA_PATH="/home/ly/med-moe-mtl/data/train.jsonl"
VAL_DATA_PATH="/home/ly/med-moe-mtl/data/dev.jsonl"
DEEPSPEED_CONFIG="/home/ly/med-moe-mtl/config/ds2.json"
OUTPUT_DIR="/home/ly/med-moe-mtl/results/mlora_r16_n2"

lora_r=16
lora_alpha=32
lora_dropout=0.1

wandb_run_name="mlora_r16_n2"
wandb_project="mlora_r16_n2"

# export MASTER_PORT=29501  # 更改为未被占用的端口


CUDA_VISIBLE_DEVICES=0,1,3 deepspeed --master_port 29520 $SCRIPT_PATH \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --data_path $DATA_PATH \
    --val_data_path $VAL_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size 4 \
    --num_epochs 2 \
    --learning_rate 2e-4 \
    --cutoff_len 1024 \
    --save_step 1000 \
    --adapter_name mlora \
     --wandb_project $wandb_project \
    --wandb_run_name $wandb_run_name \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --use_gradient_checkpointing \
    --lambda_num 16 \
    --num_B 3 \
    --temperature 0.01 \
    --deepspeed $DEEPSPEED_CONFIG \