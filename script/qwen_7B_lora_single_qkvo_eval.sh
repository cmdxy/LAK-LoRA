for task_id in {0..15}
do
    echo "Starting inference for model with task_id=${task_id}..."

    CUDA_VISIBLE_DEVICES=2 python lora_evaluate_single.py \
    --model Qwen2.5-7B-Instruct \
    --adapter LoRA \
    --base_model 'Qwen/Qwen2.5-7B-Instruct' \
    --batch_size 1 \
    --lora_weights '' \
    --task_id $task_id \

    if [ $? -ne 0 ]; then
        echo "Inference failed for model with task_id=${task_id}. Stopping subsequent inferences."
        exit 1
    fi
    
    echo "Inference complete for model with task_id=${task_id}."
done

echo "All task inferences have been completed."