SCRIPT_PATH=lora_finetune.py
DATA_PATH=""
VAL_DATA_PATH=""
DEEPSPEED_CONFIG=""
OUTPUT_DIR=""
lora_r=16
lora_alpha=32
lora_dropout=0.1

wandb_run_name=""
wandb_project=""


CUDA_VISIBLE_DEVICES=1,2,3 deepspeed --master_port 29501 $SCRIPT_PATH \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --data_path $DATA_PATH \
    --val_data_path $VAL_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 2e-4 --cutoff_len 1024 \
    --wandb_project $wandb_project \
    --wandb_run_name $wandb_run_name \
    --save_step 100 \
    --adapter_name lora \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --use_gradient_checkpointing \
    --deepspeed $DEEPSPEED_CONFIG \