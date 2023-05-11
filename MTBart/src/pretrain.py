import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
from utils.trainer import MTGenerationTrainer
from dataclasses import dataclass, field
from transformers import HfArgumentParser, set_seed, BartTokenizer
from utils.training_args import FileTrainingArguments
from transformers.trainer_utils import is_main_process, get_last_checkpoint
import os
from typing import Optional
from data.dataset import EncoderDecoderDataset
from model.model import MTForConditionalGeneration
from model.config import MTConfig
from data.utils import batchify
import json
import socket
import logging

logger = logging.getLogger(__name__)


# ============================= Arguments Definition =============================== #
@dataclass
class DataArguments:
    train_file: str = field(default=None, metadata={"help": "The train data file path."})
    index_file: str = field(default=None, metadata={"help": "The index file of train data."})
    train_num: int = field(default=None, metadata={"help": "train sample number"})
    max_sent_length: Optional[int] = field(default=1024, metadata={"help": "Maximum token number of a sentence."})
    with_span: Optional[bool] = field(default=False, metadata={"help": "pretraining step."})


@dataclass
class ModelArguments:
    decoders: Optional[str] = field(default="L2R", metadata={"help": "decoders. Comma seperated."})
    model_name_or_path: Optional[str] = field(default=None,
                                              metadata={"help": "The model checkpoint for weights initialization."})
    with_bart_initialize: Optional[str] = field(default=None, metadata={
        "help": "Use Bart-base or Bart-large to initialize the part of parameters."})
    config_name: Optional[str] = field(default=None, metadata={"help": "config name"})


# ======================================================================================== #

def is_none(argument):
    return argument is None or argument == 'None'


# run function
def run():
    # =========== parse arguments =========== #
    parser = HfArgumentParser((FileTrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()

    # =========== Detecting last checkpoint =========== #
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",

        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    logger.info(f"hostname: {socket.gethostname()}")

    # log on each process the small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # log parameters
    # training arguments
    logger.info("\n====== Training/evaluation parameters ======\n%s", training_args.to_json_string())
    # model arguments
    logger.info("\n====== model parameters ======\n%s", json.dumps(vars(model_args), indent=4))
    # data arguments
    logger.info("\n====== data parameters ======\n%s", json.dumps(vars(data_args), indent=4))

    # ========== set seed at first ========== #
    set_seed(training_args.seed)

    # ========== prepare for training/evaluation =========== #
    # tokenizer
    if not is_none(model_args.model_name_or_path):
        logger.info(f"Build tokenizer from {model_args.model_name_or_path}.")
        tokenizer = BartTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        # use facebook bart tokenizer default
        logger.info("Use default Bart tokenizer.")
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        tokenizer.add_special_tokens({'additional_special_tokens': ['<nonesense>','<MaskL2R>', '<MaskR2L>', '<MaskBert>']})

    # data collator
    def data_collator(data):
        return batchify(data, tokenizer, data_args.max_sent_length, data_args.with_span) # unsolved

    config = MTConfig.from_pretrained(model_args.config_name)
    config.set_decoders(model_args.decoders)
    logger.info(f"decoders: {config.decoders}")
    assert config.L2RMaskIdx == tokenizer.convert_tokens_to_ids('<MaskL2R>'),"oops!"
    #config.R2LMaskIdx = tokenizer.convert_tokens_to_ids('<MaskR2L>')
    #config.BertMaskIdx = tokenizer.convert_tokens_to_ids('<MaskBert>')
    assert len(tokenizer) == 50269,"oops!"
    # train
    if training_args.do_train:
        logger.info("Start to prepare for training.")

        # dataset
        logger.info("Loading index file... Please wait for a moment (several miniutes)")
        train_set = EncoderDecoderDataset(data_args.train_file, data_args.index_file)

        # model
        if last_checkpoint is None:
            if not is_none(model_args.model_name_or_path):
                logger.info(f"Initializing model from {model_args.model_name_or_path}.")
                model = MTForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
            else:
                logger.info("Initializing model from scratch...")
                model = MTForConditionalGeneration(config)
        else:
            model = MTForConditionalGeneration(config)
            logger.info("Resume from the previous checkpoint...")

        #last_checkpoint = "../ckpts/pretrain/MTL-base/checkpoint-22000"
        last_checkpoint = None
        model.resize_token_embeddings(len(tokenizer))

        # unsolved trainer
        trainer = MTGenerationTrainer(model=model, args=training_args, train_dataset=train_set,
                                      tokenizer=tokenizer,
                                      data_collator=data_collator, decoders=config.decoders)
        logger.info("Start to train...")
        logger.info("******* train *******")
        logger.info(f"last checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)

        logger.info("Finish training!")


if __name__ == '__main__':
    run()
