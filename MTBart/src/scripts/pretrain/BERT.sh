#!/bin/bash
#SBATCH --job-name=BERT
#SBATCH --partition=2080ti
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem 20G
#SBATCH -a 0-31
#SBATCH -x gqxx-01-118
#SBATCH --output=../log/pretrain/bert-%A-%a.log
# ----------------------------------------- CONFIG ------------------------------------- #
DATA="/mnt/lustre/sjtu/home/dm311/remote/pretrain/data/pretrain" # data
OUTPUT_DIR="../ckpts/pretrain/bert-base"                                   # output dir
TRAIN_FILE="train.large.txt"                                     # train file
INDEX_FILE="train.large.txt.index"
MODEL_NAME="facebook/bart-base"                                           # Transformer, Tud, or Bart
MAX_SENT_LENGTH=512                                              # Maximum sentence length
MODEL_NAME_OR_PATH="None"                                        # MODEL CKPT, initialization
WITH_BART_INITIALIZE="facebook/bart-base"                        # The part of parameters to initialize
PER_DEVICE_TRAIN_BATCH_SIZE=1                                    # per device train batch size
GRADIENT_ACCUMULATION_STEPS=8``                             ``                                    # gradient accumulation
LR=1e-5                                                          # learning rate
NUM_TRAIN_EPOCHS=1
MAX_STEPS=200000          # max steps
WARMUP_STEPS=10000         # warm up steps
SAVE_STEPS=5000            # save steps
LABEL_SMOOTHING_FACTOR=0 # label smoothing factor
DECODERS="L2R,R2L,BERT"    # decoders
LOGGING_STEPS=500          # logging steps
DATALOADER_NUM_WORKERS=4

# -------------------------------------------------- #
# launch
echo "${SLURM_ARRAY_TASK_ID}"

mkdir $OUTPUT_DIR
#rm -rf "${OUTPUT_DIR}/file_share_cache"
wait
python -u utils/launch.py \
    --nproc_per_node 1 \
    --nnodes 32 \
    --node_rank "${SLURM_ARRAY_TASK_ID}" \
    pretrain.py \
    --train_file "${DATA}/${TRAIN_FILE}" \
    --index_file "${DATA}/${INDEX_FILE}" \
    --max_sent_length $MAX_SENT_LENGTH \
    --with_span true \
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
    --weight_decay 0.01
    --disable_tqdm true
