#!/bin/bash
#SBATCH --job-name=cola
#SBATCH --partition=a10
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem 20G
#SBATCH -o ../log/downstream/glue/mtl-cola.log
mkdir ../ckpts/downstream/glue/mtl-cola
rm -rf ../ckpts/downstream/glue/mtl-cola/file_share_cache
wait

python -u utils/launch.py \
    --nproc_per_node 2 \
    downstream/run/finetune_seq_classification.py \
    --task_name "cola" \
    --data_path "/mnt/lustre/sjtu/home/dm311/remote/pretrain/data/downstream/glue" \
    --max_seq_length 512 \
    --do_train true \
    --do_eval true \
    --num_train_epochs 10 \
    --warmup_steps 200 \
    --tokenizer_name "facebook/bart-base" \
    --model_name_or_path "../ckpts/pretrain/checkpoint-140000" \
    --evaluation_strategy "epoch" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --output_dir "../ckpts/downstream/glue/mtl-cola/" \
    --ddp_find_unused_parameters true \
    --overwrite_output_dir true \

