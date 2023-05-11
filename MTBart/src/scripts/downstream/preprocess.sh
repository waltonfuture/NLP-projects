#!/bin/bash

# --------------------------------------------------- CONFIG -------------------------------------------------- #
# task, support now:
#   * summarization
TASK="summarization"

DATA="../../PLUTO/data/downstream/summarization/gigaword" # data path, absolute or relative path
SAVE="../data/downstream/summarization/gigaword"          # save path, absolute or relative path
# ------------------------------------------------------------------------------------------------------------- #

python -u utils/preprocessing.py \
    --task $TASK \
    --data $DATA \
    --save $SAVE
