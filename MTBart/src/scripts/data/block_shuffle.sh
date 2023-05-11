#!/bin/bash
#SBATCH -a 384-511
#SBATCH -o ../log/test/pretrain-data-%j.out

python -u data/shuffle.py \
    --data_file "/mnt/lustre/sjtu/home/dm311/remote/pretrain/data/pretrain/train.large.txt" \
    --dir "/mnt/lustre/sjtu/home/dm311/remote/pretrain/data/pretrain/tmp" \
    --block $SLURM_ARRAY_TASK_ID
