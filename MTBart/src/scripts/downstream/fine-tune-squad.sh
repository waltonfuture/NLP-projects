#!/bin/bash
#SBATCH --job-name=bart
#SBATCH --partition=a10
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem 20G
#SBATCH -o ../log/downstream/bart-large.log

OUTPUT_DIR="../ckpts/downstream/MT-bart/squad-v1"
mkdir -p $OUTPUT_DIR
wait

export TOKENIZERS_PARALLELISM=false

python -u utils/launch.py \
    --nproc_per_node 2 \
    downstream/run/finetune_squad.py \
    --dataset_path "/mnt/lustre/sjtu/home/dm311/remote/pretrain/data/downstream/qa/squad-v1" \
    --max_seq_length 512 \
    --do_train true \
    --do_eval true \
    --num_train_epochs 3 \
    --logging_steps 100 \
    --warmup_steps 500 \
    --config_name "facebook/bart-large" \
    --model_name_or_path "/mnt/lustre/sjtu/home/dm311/remote/pretrain/ckpts/pretrain/MTL-large/checkpoint-45000" \
    --evaluation_strategy "epoch" \
    --label_smoothing_factor 0.1 \
    --save_steps 2742 \
    --dataloader_num_workers 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --output_dir $OUTPUT_DIR \
    --ddp_find_unused_parameters false \
    --overwrite_output_dir true \
    --use_bart false \
