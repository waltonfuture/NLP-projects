#!/bin/bash
#SBATCH --job-name index
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH -a 0
#SBATCH -o ../log/data/index.log

DATA="wiki" # book or wiki

BASE_PRETRAIN="/mnt/lustre/sjtu/home/dm311/remote/PLUTO/data/pretrain"
BASE_BUFFER="/mnt/lustre/sjtu/home/dm311/remote/pretrain/data/buffer/${DATA}"
mkdir -p $BASE_BUFFER

# -------------------- BookCorpus Config ------------------- #
if [ $DATA = "book" ]; then
    DATA_DIR="${BASE_PRETRAIN}/bookcorpus"
# -------------------- Wiki Config -------------------- #
else
    DATA_DIR="${BASE_PRETRAIN}/wiki"
fi

TOTAL=$(ls ${DATA_DIR} | wc -l)
NODE_NUM=64

# get all files
c=0
for file in `ls ${DATA_DIR}`
do
  files[$c]=$file
  ((c++))
done

# get block files
remainder=$(($TOTAL % $NODE_NUM))
if [ $SLURM_ARRAY_TASK_ID -lt $remainder ]; then
    start=$((($TOTAL / $NODE_NUM + 1) * $SLURM_ARRAY_TASK_ID))
    end=$(($start + $TOTAL / $NODE_NUM + 1))
else
    start=$((($TOTAL / $NODE_NUM + 1) * $remainder + $TOTAL / $NODE_NUM * ($SLURM_ARRAY_TASK_ID - $remainder)))
    end=$(($start + $TOTAL / $NODE_NUM))
fi

echo "------ INFO ${SLURM_ARRAY_TASK_ID} ------"
echo "start: ${start}"
echo "end: ${end}"

blocks=""
for i in $(seq $start $(($end - 1))); do
    blocks="${blocks} ${files[$i]}"
done
blocks=${blocks: 1}

OUTPUT_FILE="${BASE_BUFFER}/block-${SLURM_ARRAY_TASK_ID}.out"

# --build_data indicates build noised pre-training data
# --build_index indicates build noised pre-training data index

echo $blocks
OUTPUT_FILE="/mnt/lustre/sjtu/home/dm311/remote/pretrain/data/pretrain/train.large.txt"

python -u data/build_pretrain_data.py \
    --blocks_dir "${DATA_DIR}" \
    --block_files ${blocks} \
    --poisson_lambda 3.5 \
    --output_file "${OUTPUT_FILE}" \
    --max_subword_len 512 \
    --build_index