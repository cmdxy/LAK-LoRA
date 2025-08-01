CUDA_VISIBLE_DEVICES=3 python mlora_evaluate.py \
    --model Qwen2.5-7B-Instruct \
    --adapter laklora \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --task_num 16 \
    --te_dim 64 \
    --epsilon 4.0 \
    --lora_weights '' \