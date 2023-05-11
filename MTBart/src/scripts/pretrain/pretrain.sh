#!/bin/bash
#SBATCH --job-name=MT-pretrain
#SBATCH --partition=a10
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem 10G
#SBATCH -a 0-7
#SBATCH --output=../log/pretrain/MTL-base-epoch-1-%A-%a.log

# ----------------------------------------- CONFIG ------------------------------------- #
DATA="/mnt/lustre/sjtu/home/dm311/remote/pretrain/data/pretrain" # data
OUTPUT_DIR="../ckpts/pretrain/MTL-base"                                   # output dir
TRAIN_FILE="train.large.txt"                                     # train file
INDEX_FILE="train.large.txt.index"
MODEL_NAME="Bart-base"                                           # Transformer, Tud, or Bart
MAX_SENT_LENGTH=512                                              # Maximum sentence length
MODEL_NAME_OR_PATH="None"                                        # MODEL CKPT, initialization
WITH_BART_INITIALIZE="None"                        # The part of parameters to initialize
PER_DEVICE_TRAIN_BATCH_SIZE=4                                    # per device train batch size
GRADIENT_ACCUMULATION_STEPS=8                                    # gradient accumulation
LR=3e-4                                                          # learning rate
NUM_TRAIN_EPOCHS=1
MAX_STEPS=-1          # max steps
WARMUP_STEPS=10000         # warm up steps
SAVE_STEPS=1000            # save steps
LABEL_SMOOTHING_FACTOR=0 # label smoothing factor
DECODERS="L2R,R2L,BERT"    # decoders
LOGGING_STEPS=1000          # logging steps
DATALOADER_NUM_WORKERS=4
CONFIG_NAME="facebook/bart-base"

# -------------------------------------------------- #
# launch
echo "${SLURM_ARRAY_TASK_ID}"

#mkdir $OUTPUT_DIR
#rm -rf "${OUTPUT_DIR}/file_share_cache"
#wait
python -u utils/launch.py \
    --nproc_per_node 1 \
    --nnodes 8 \
    --node_rank "${SLURM_ARRAY_TASK_ID}" \
    pretrain.py \
    --train_file "${DATA}/${TRAIN_FILE}" \
    --index_file "${DATA}/${INDEX_FILE}" \
    --max_sent_length $MAX_SENT_LENGTH \
    --do_train true \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --with_bart_initialize $WITH_BART_INITIALIZE \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LR \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --warmup_steps $WARMUP_STEPS \
    --save_steps $SAVE_STEPS \
    --label_smoothing_factor $LABEL_SMOOTHING_FACTOR \
    --max_steps $MAX_STEPS \
    --decoders $DECODERS \
    --logging_steps $LOGGING_STEPS \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --dataloader_drop_last false \
    --ddp_find_unused_parameters false \
    --overwrite_output_dir true \
    --max_grad_norm 0.1 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --save_total_limit 30 \
    --config_name ${CONFIG_NAME}
