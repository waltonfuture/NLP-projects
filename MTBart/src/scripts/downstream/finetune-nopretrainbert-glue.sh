#!/bin/bash
#SBATCH -N 1
#SBATCH -p 2080ti,gpu
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH -J mnli
#SBATCH --output=/mnt/lustre/sjtu/home/lw023/remote/bart/results/bertnopretrain/log/MNLI/%A-%a.out
#SBATCH --error=/mnt/lustre/sjtu/home/lw023/remote/bart/results/bertnopretrain/log/MNLI/%A-%a.err


TASK_NAME="MNLI"

MODEL_DIR="bert-base-cased"
#MODEL_DIR="distilbert-base-cased"

TRAIN_FILE="/mnt/lustre/sjtu/home/lw023/remote/GLUE-wl/glue/${TASK_NAME}/train.csv"

VALIDATION_FILE="/mnt/lustre/sjtu/home/lw023/remote/GLUE-wl/glue/${TASK_NAME}/dev.csv"

TEST_FILE="/mnt/lustre/sjtu/home/lw023/remote/GLUE-wl/glue/${TASK_NAME}/test.csv"

DEV_MM="/mnt/lustre/sjtu/home/lw023/datasets-wl/glue/MNLI/dev_mismatched.csv"

TEST_MM="/mnt/lustre/sjtu/home/lw023/datasets-wl/glue/MNLI/test_mismatched.csv"

OUTPUT_DIR="/mnt/lustre/sjtu/home/lw023/remote/bart/results/bertnopretrain/predict/${TASK_NAME}/"

EPOCHS=3


python ../../downstream/run/finetune_seq_classification_bert_nopre.py \
  --model_name_or_path ${MODEL_DIR} \
  --task_name ${TASK_NAME} \
  --data_path "/mnt/lustre/sjtu/home/lw023/datasets-wl/GLUE" \
  --do_train true \
  --do_eval true \
  --do_predict true \
  --train_file ${TRAIN_FILE} \
  --validation_file ${VALIDATION_FILE} \
  --test_file ${TEST_FILE} \
  --eval_mnli_m ${VALIDATION_FILE} \
  --eval_mnli_mm ${DEV_MM} \
  --test_mnli_m ${TEST_FILE} \
  --test_mnli_mm ${TEST_MM} \
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
