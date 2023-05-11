#!/bin/bash
#SBATCH --job-name=pred
#SBATCH --partition=2080ti,gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=../log/inference/%j.out
#SBATCH -a 0-70

# ----------------------------------------- CONFIG ------------------------------------- #
CHECKPOINT=$((5000 * ($SLURM_ARRAY_TASK_ID + 1)))

TASK="summarization"                                           # summarization
DATA="gigaword"                                                # data
DO_PRED=true                                                   # true or false
PRED_FILE="test.txt"                                           # prediction file
LOWERCASE=false                                                # true or false, whether to lower case the data
SAVE_DIR="ALL_output/checkpoint-${CHECKPOINT}"                     # output prediction to this directory
MAX_SENT_LENGTH=512                                            # Maximum sentence length
MODEL_NAME_OR_PATH="../ckpts/step1-ALL/checkpoint-${CHECKPOINT}" # MODEL CKPT, initialization
PER_DEVICE_EVAL_BATCH_SIZE=8                                   # per device train batch size
BEAM_SIZE=5                                                    # beam size
BEAM_MIN_LENGTH=10                                             # min length of beam search
BEAM_MAX_LENGTH=64                                             # max length of beam search
BEAM_LENGTH_PENALTY=1.0                                        # length penalty of beam search
NO_REPEAT_NGRAM_SIZE=3                                         # no repeat ngram size of beam search
REVERSE_DIRECTION=false                                         # reverse direction
# -------------------------------------------------------------------------------------- #

# ------------------ DATA CONFIG ------------------- #
DATA_PREFIX="../data/downstream/${TASK}/${DATA}"
PRED_FILEPATH="${DATA_PREFIX}/${PRED_FILE}"
SAVE_PATH="../data/downstream/${TASK}/${DATA}/${SAVE_DIR}"
# -------------------------------------------------- #
python -u run.py \
    --pred_file $PRED_FILEPATH \
    --save_path $SAVE_PATH \
    --lowercase $LOWERCASE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --max_sent_length $MAX_SENT_LENGTH \
    --do_pred $DO_PRED \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --beam_size $BEAM_SIZE \
    --beam_min_length $BEAM_MIN_LENGTH \
    --beam_max_length $BEAM_MAX_LENGTH \
    --beam_length_penalty $BEAM_LENGTH_PENALTY \
    --no_repeat_ngram_size "${NO_REPEAT_NGRAM_SIZE}" \
    --reverse_direction $REVERSE_DIRECTION \
    --output_dir "test"
