#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
from os.path import exists, join
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch

import datasets
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
#from model.plm import PlmForSequenceClassification
#from model.config import PlmConfig

import os.path

from transformers import AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from utils.glue_utils import seed_everything, ProgressBar, compute_metrics

class Trainer(object):
    global_step = 0
    best_step = None
    best_score = .0
    cnt_patience = 0

    def __init__(
            self,
            args,
            model,
            logger,
            tokenizer,
            metric,
            train_dataloader=None,
            eval_dataloader=None,
    ):

        self.args = args
        self.model = model
        self.metric=metric
        self.tokenizer=tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_optimizer_and_scheduler(self, num_training_steps: int):
        args = self.args
        args.weight_decay = 0.01
        no_decay = ['bias', 'LayerNorm.weight']
        '''
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]
        '''
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters()],
             'weight_decay': args.weight_decay},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,betas=(0.9,0.98))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=num_training_steps,
                                                    num_warmup_steps=num_training_steps * args.warmup_ratio)
        return optimizer, scheduler

    def train(self):
        args = self.args
        logger = self.logger
        num_training_steps = len(self.train_dataloader) * args.num_train_epochs
        optimizer, scheduler = self._get_optimizer_and_scheduler(num_training_steps)
        seed_everything(args.seed)
        self.model.zero_grad()

        logger.info("***** Running training *****")
        logger.info("Num samples %d", len(self.train_dataloader.dataset))
        logger.info("Num epochs %d", args.num_train_epochs)
        logger.info("Num training steps %d", num_training_steps)
        logger.info("Num warmup steps %d", num_training_steps * args.warmup_ratio)

        self.global_step = 0
        self.best_step = None
        self.best_score = .0
        self.cnt_patience = 0

        for i in range(int(args.num_train_epochs)):
            self.train_epoch(self.train_dataloader, optimizer, scheduler)

        logger.info("Training Stop! The best step %s: %s", self.best_step, self.best_score)
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        self._save_best_checkpoint(best_step=self.best_step)
        return self.global_step, self.best_step

    def train_epoch(self, train_dataloader, optimizer, scheduler):
        args = self.args
        model = self.model.to(self.device)
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')

        for step, item in enumerate(train_dataloader):
            loss = self.training_step(model, item)
            pbar(step, {'loss': loss.item()})
            if (step + 1) % getattr(args, "accumulate_steps", 1) == 0:
                if args.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            self.global_step += 1

            if args.logging_steps > 0 and self.global_step % args.logging_steps == 0:
                print("")
                score = self.evaluate(model)
                if score > self.best_score:
                    self.best_score = score
                    self.best_step = self.global_step
                    self.cnt_patience = 0
                    self._save_checkpoint(model, self.global_step)


        return 0

    def training_step(self, model, item):
        raise NotImplementedError

    def evaluate(self, model):
        raise NotImplementedError

    def _save_checkpoint(self, model, step):
        raise NotImplementedError

    def _save_best_checkpoint(self, best_step):
        raise NotImplementedError


class GLUETrainer(Trainer):
    def __init__(
        self,
        args,
        model,
        logger,
        metric,
        tokenizer,
        train_dataloader=None,
        eval_dataloader=None,
    ):
        super(GLUETrainer, self).__init__(
            args=args,
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            logger=logger,
            metric=metric,
            tokenizer=tokenizer,
        )

    def training_step(self, model, item):
        model.train()
        inputs = {k: v.to(self.device) for k, v in item.items()}
        outputs = model(**inputs)
        loss = outputs[0]
        loss.backward()
        return loss.detach()

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        num_examples = len(self.eval_dataloader.dataset)
        preds = None
        eval_labels = None
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        for step, item in enumerate(self.eval_dataloader):
            model.eval()
            inputs = {k: v.to(self.device) for k, v in item.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                loss, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                eval_labels = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                eval_labels = np.append(
                    eval_labels, inputs['labels'].detach().cpu().numpy(), axis=0
                )
        print("preds!!!")
        print(preds)
        task_name = args.output_dir.split('/')[-2]
        preds =  np.squeeze(preds) if task_name == 'STSB' else np.argmax(preds,axis=1)
        metrics = compute_metrics(preds, eval_labels,self.metric)
        logger.info("%s-%s acc: %s", task_name, metrics)

        if task_name == 'CoLA':
            print("scores!!!")
            print(metrics['matthews_correlation'])
            return metrics['matthews_correlation']
        if task_name == 'STSB':
            return metrics['combined_score']
        if task_name == 'MRPC' or task_name == 'QQP':
            return metrics['f1']
        return metrics['accuracy']

    def predict(self, test_dataloader, model):
        args = self.args
        logger = self.logger
        num_examples = len(test_dataloader)
        model.to(self.device)
        preds = None
        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc="Prediction")
        for step, item in enumerate(test_dataloader):
            model.eval()
            inputs = {k: v.to(self.device) for k, v in item.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs[0].detach()
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            pbar(step=step, info="")

        task_name = args.output_dir.split('/')[-2]
        preds = np.squeeze(preds) if task_name == 'STSB' else np.argmax(preds, axis=1)
        return preds


    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, "checkpoint-{}".format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        self.tokenizer.save_vocabulary(save_directory=output_dir)
        self.logger.info("Saving models checkpoint to %s", output_dir)

    def _save_best_checkpoint(self, best_step):
        model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(self.args.output_dir, f"checkpoint-{best_step}"),
        )
        model.save_pretrained(self.args.output_dir)
        torch.save(
            self.args, os.path.join(self.args.output_dir, "training_args.bin")
        )
        self.tokenizer.save_vocabulary(save_directory=self.args.output_dir)
        self.logger.info("Saving models checkpoint to %s", self.args.output_dir)





task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "The name of the task to train on: " +
            ", ".join(task_to_keys.keys())
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "The name of the dataset to use (via the datasets library)."
        })
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "The configuration name of the dataset to use (via the datasets library)."
        })
    max_seq_length: int = field(
        default=128,
        metadata={
            "help":
            ("The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.")
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the cached preprocessed datasets or not."
        })
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help":
            ("Whether to pad all samples to `max_seq_length`. "
             "If False, will pad the samples dynamically when batching to the maximum length in the batch."
             )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
            ("For debugging purposes or quicker training, truncate the number of training examples to this "
             "value if set.")
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
            ("For debugging purposes or quicker training, truncate the number of evaluation examples to this "
             "value if set.")
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
            ("For debugging purposes or quicker training, truncate the number of prediction examples to this "
             "value if set.")
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "A csv or a json file containing the training data."
        })

    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "A csv or a json file containing the validation data."
        })

    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " +
                                 ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError(
                "Need either a GLUE task, a training/validation file or a dataset name."
            )
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv", "json"
            ], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help":
            "Path to pretrained model or model identifier from huggingface.co/models"
        })
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained config name or path if not the same as model_name"
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained tokenizer name or path if not the same as model_name"
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help":
            "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help":
            "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help":
            ("Will use the token generated when running `huggingface-cli login` (necessary to use this script "
             "with private models).")
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help":
            "Will enable to load a pretrained model whose head dimensions are different."
        },
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        )

    send_example_telemetry("run_glue", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        +
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None

    training_args.overwrite_output_dir = True
    if os.path.isdir(
            training_args.output_dir
    ) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(
                training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {
        "train": data_args.train_file,
        "validation": data_args.validation_file,
        "test": data_args.test_file
    }

    train_dataset = load_dataset(
        "csv",
        data_files=data_files['train'],
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    dev_dataset = load_dataset(
        "csv",
        data_files=data_files['validation'],
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    test_dataset = load_dataset(
        "csv",
        data_files=data_files['test'],
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    raw_datasets = datasets.DatasetDict({
        'train': train_dataset['train'],
        'validation': dev_dataset['train'],
        'test': test_dataset['train'],
    })

    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:

        is_regression = raw_datasets["train"].features["label"].dtype in [
            "float32", "float64"
        ]
        if is_regression:
            num_labels = 1
        else:

            label_list = raw_datasets["train"].unique("label")
            label_list.sort()
            num_labels = len(label_list)


    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=True,
    )

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        non_label_column_names = [
            name for name in raw_datasets["train"].column_names
            if name != "label"
        ]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (model.config.label2id !=
            PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None and not is_regression):
        # Some have all caps in their config, some don't.
        label_name_to_id = {
            k.lower(): v
            for k, v in model.config.label2id.items()
        }
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]])
                for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {
            id: label
            for label, id in config.label2id.items()
        }
    elif data_args.task_name is not None and not is_regression:
        label_to_id = {l: i for i, l in enumerate(label_list)}
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {
            id: label
            for label, id in config.label2id.items()
        }

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        args = ((examples[sentence1_key], ) if sentence2_key is None else
                (examples[sentence1_key], examples[sentence2_key]))

        result = tokenizer(*args,
                           padding=padding,
                           max_length=max_seq_length,
                           truncation=True)

        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1)
                               for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset),
                                    data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset),
                                   data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset),
                                      data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(
                range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name)
    else:
        metric = evaluate.load("accuracy")

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer,
                                                pad_to_multiple_of=8)
    else:
        data_collator = None

    train_dataloader = DataLoader(train_dataset,shuffle=True, batch_size=training_args.per_device_train_batch_size,
                                  collate_fn=data_collator)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=training_args.per_device_eval_batch_size,
                                  collate_fn=data_collator)
    test_dataloader = DataLoader(predict_dataset, shuffle=False, batch_size=training_args.per_device_eval_batch_size,
                                 collate_fn=data_collator)

    trainer = GLUETrainer(
        model=model,
        args=training_args,
        logger=logger,
        metric=metric,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    # Training
    if training_args.do_train:
        '''
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        '''
        _global_step, _best_step  = trainer.train()

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions = trainer.predict(test_dataloader, model=model)
        output_predict_file = os.path.join(training_args.output_dir,
                                               f"predict_results_{data_args.task_name}.txt")

        with open(output_predict_file, "w") as writer:
            logger.info(f"***** Predict results {data_args.task_name} *****")
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                if is_regression:
                    writer.write(f"{index}\t{item:3.3f}\n")
                else:
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
