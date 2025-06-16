#!/bin/bash
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --job-name=ft_llama_proba8
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=100GB
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=paula.ontalvilla@ehu.eus 
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100-sxm4
#SBATCH --output=../results/.slurm/sft_llama_proba_8.out
#SBATCH --error=../results/.slurm/sft_llama_proba_8.err

export PYTHONPATH="$PYTHONPATH:/home/pontalvillla/multimodal/multimodal/SFTvsRL/sft/src"
source /scratch/pontalvillla/multimodal/qwen_env/bin/activate

MODEL_NAME="unsloth/Llama-3.2-11B-Vision-Instruct"
DATA_JSON_TRAIN="/home/pontalvillla/multimodal/multimodal/SFTvsRL/data/train_4/train.json"
DATA_JSON_DEV="/home/pontalvillla/multimodal/multimodal/SFTvsRL/data/dev_4/dev.json"
IMAGE_FOLDER="/"
OUTPUT_FOLDER="../train_ckpt/sft_proba_8"

LR=1e-5
EPOCH=100

accelerate launch  src/training/train.py \
    --deepspeed sft_scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path $DATA_JSON_TRAIN \
    --data_path_eval $DATA_JSON_DEV \
    --image_folder $IMAGE_FOLDER \
    --disable_flash_attn2 True \
    --lora_enable False \
    --eval_strategy "steps" \
    --eval_steps 10 \
    --tune_img_projector True \
    --freeze_vision_tower False \
    --freeze_llm False \
    --bf16 True \
    --output_dir $OUTPUT_FOLDER \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size 4 \
    --learning_rate ${LR} \
    --projector_lr ${LR} \
    --vision_lr ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "epoch" \
    --save_total_limit 4 \
    --dataloader_num_workers 4 \
    --save_only_model True 
