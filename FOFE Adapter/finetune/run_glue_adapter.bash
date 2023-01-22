#!/bin/bash

TASK_NAME="CoLA"

MODEL_DIR="bert-base-cased"


TRAIN_FILE="../${TASK_NAME}/train.csv"

VALIDATION_FILE="..${TASK_NAME}/dev.csv"

TEST_FILE="../${TASK_NAME}/test.csv"

OUTPUT_DIR="../output/${TASK_NAME}/"

EPOCHS=3

D0_PREDICT=False

python run_glue_ours.py \
  --model_name_or_path ${MODEL_DIR} \
  --task_name ${TASK_NAME} \
  --do_train \
  --do_eval \
  --do_predict ${D0_PREDICT}\
  --train_file ${TRAIN_FILE} \
  --validation_file ${VALIDATION_FILE} \
  --test_file ${TEST_FILE} \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs ${EPOCHS} \
  --output_dir ${OUTPUT_DIR} \
  --weight_decay 0.01 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --warmup_ratio  0.1 \
  --logging_steps 100 \
  --save_steps 100 \




  