#!/bin/bash
#SBATCH --job-name wiki
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH -o ../log/data/wiki-merge.log

python -u data/merge_blocks.py \
    --src_dir "/mnt/lustre/sjtu/home/dm311/remote/pretrain/data/buffer/wiki" \
    --dst_dir "/mnt/lustre/sjtu/home/dm311/remote/pretrain/data/pretrain" \
    --file_mode "a"
