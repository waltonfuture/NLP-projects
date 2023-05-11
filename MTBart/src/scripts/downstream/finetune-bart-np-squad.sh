#!/bin/bash
#SBATCH --job-name=bart-squad
#SBATCH --partition=a10
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem 20G
#SBATCH -o /mnt/lustre/sjtu/home/lw023/remote/bart/results/bartnopretrain/log/squad/bart.log

OUTPUT_DIR="/mnt/lustre/sjtu/home/lw023/remote/bart/results/bartnopretrain/predict/squad"

export TOKENIZERS_PARALLELISM=false

python ../../downstream/run/finetune_qa_bert.py \
    --dataset_path "/mnt/lustre/sjtu/home/lw023/datasets-wl/SQuAD/squad" \
    --max_seq_length 512 \
    --do_train true \
    --do_eval true \
    --num_train_epochs 3 \
    --logging_steps 100 \
    --warmup_steps 500 \
    --config_name "facebook/bart-base" \
    --model_name_or_path "facebook/bart-base" \
    --evaluation_strategy "epoch" \
    --save_steps 2742 \
    --doc_stride 128 \
    --dataloader_num_workers 8 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 8 \
    --eval_accumulation_steps 1 \
    --learning_rate 3e-2 \
    --output_dir $OUTPUT_DIR \
    --ddp_find_unused_parameters false \
    --overwrite_output_dir true \
    --use_bart true \
