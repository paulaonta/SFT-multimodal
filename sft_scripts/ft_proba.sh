
export PYTHONPATH="$PYTHONPATH:/home/pontalvillla/multimodal/multimodal/SFTvsRL/sft/src"
source /scratch/pontalvillla/multimodal/qwen_env/bin/activate

MODEL_NAME="unsloth/Llama-3.2-11B-Vision-Instruct"
DATA_JSON_TRAIN="/home/pontalvillla/multimodal/multimodal/SFTvsRL/data/train_50/train.json"
DATA_JSON_DEV="/home/pontalvillla/multimodal/multimodal/SFTvsRL/data/dev_4/dev.json"
IMAGE_FOLDER="/"
OUTPUT_FOLDER="../train_ckpt/sft_proba_100"

LR=1e-5
EPOCH=100

accelerate launch  --main_process_port 29516 src/training/train.py \
    --deepspeed sft_scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path $DATA_JSON_TRAIN \
    --image_folder $IMAGE_FOLDER \
    --disable_flash_attn2 True \
    --lora_enable False \
    --tune_img_projector True \
    --freeze_vision_tower False \
    --freeze_llm False \
    --bf16 True \
    --output_dir $OUTPUT_FOLDER \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate ${LR} \
    --projector_lr ${LR} \
    --vision_lr ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "no" \
    --save_total_limit 4 \
    --dataloader_num_workers 4 \
    --save_only_model True 