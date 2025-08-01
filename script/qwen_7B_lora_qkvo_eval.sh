CUDA_VISIBLE_DEVICES=1 python lora_evaluate.py \
    --model Qwen2.5-7B-Instruct \
    --adapter LoRA \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --batch_size 1 \
    --lora_weights '' \
