import sys
sys.path.append(".")
from utils.trainer import TudGenerationTrainer
from dataclasses import dataclass, field
from transformers import HfArgumentParser, set_seed, BartTokenizer
from utils.training_args import FileTrainingArguments
import logging
from transformers.trainer_utils import is_main_process
import os
from typing import Optional
from model.tokenizer import TudTokenizer
from data.dataset import SummarizationDataset
from model.model import TudForConditionalGeneration, TudModel
from model.config import TudConfig
import torch
from tqdm import tqdm
from data.utils import batchify, move_to_cuda
import json

logger = logging.getLogger(__name__)


# ============================= Arguments Definition =============================== #
@dataclass
class DataArguments:
    train_file: Optional[str] = field(default=None, metadata={"help": "The train data file path."})
    pred_file: Optional[str] = field(default=None, metadata={"help": "The prediction data file path."})
    save_path: Optional[str] = field(default=None, metadata={"help": "The save path of predictions."})
    lowercase: Optional[bool] = field(default=False, metadata={"help": "Whether to lowercase the data."})
    max_sent_length: Optional[int] = field(default=1024, metadata={"help": "Maximum token number of a sentence."})
    output_to_single_file: Optional[bool] = field(default=False,
                                                  metadata={"help": "Whether to output to a single file."})


@dataclass
class ModelArguments:
    resume_checkpoint: Optional[str] = field(default=None, metadata={"help": "resume from this checkpoint."})
    decoders: Optional[str] = field(default="L2R", metadata={"help": "decoders. Comma seperated."})
    model_name_or_path: Optional[str] = field(default=None,
                                              metadata={"help": "The model checkpoint for weights initialization."})
    with_bart_initialize: Optional[str] = field(default=None, metadata={
        "help": "Use Bart-base or Bart-large to initialize the part of parameters."})
    d_model: Optional[int] = field(default=512, metadata={"help": "Dimensionality of the layers."})
    max_pos_length: Optional[int] = field(default=1024, metadata={
        "help": "The maximum sequence length that this model might ever be used with. Typically set this to something"
                " large just in case (e.g., 512 or 1024 or 2048)."})
    encoder_layers: Optional[int] = field(default=12, metadata={"help": "Number of encoder layers."})
    decoder_layers: Optional[int] = field(default=12, metadata={"help": "Number of decoder layers."})
    encoder_attention_heads: Optional[int] = field(default=16, metadata={
        "help": "Number of attention heads for each attention layer in the Transformer encoder."})
    decoder_attention_heads: Optional[int] = field(default=16, metadata={
        "help": "Number of attention heads for each attention layer in the Transformer decoder."})
    encoder_ffn_dim: Optional[int] = field(default=4096, metadata={
        "help": "Dimensionality of the 'intermediate' (often named feed-forward) layer in encoder."})
    decoder_ffn_dim: Optional[int] = field(default=4096, metadata={
        "help": "Dimensionality of the 'intermediate' (often named feed-forward) layer in decoder."})
    activation_function: Optional[str] = field(default="gelu", metadata={
        "help": 'The non-linear activation function (function or string) in the encoder and pooler. If string, '
                ':obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.'})
    dropout: Optional[float] = field(default=0.0, metadata={
        "help": "The dropout probability for all fully connected layers in the embeddings, encoder."})
    attention_dropout: Optional[float] = field(default=0.0,
                                               metadata={"help": "The dropout ratio for the attention probabilities."})
    activation_dropout: Optional[float] = field(default=0.0, metadata={
        "help": "The dropout ratio for activations inside the fully connected layer."})
    classifier_dropout: Optional[float] = field(default=0.0, metadata={"help": "The dropout ratio for classifier."})
    init_std: Optional[float] = field(default=0.02, metadata={
        "help": "The standard deviation of the truncated_normal_initializer for initializing all weight matrices."})
    scale_embedding: Optional[bool] = field(default=False,
                                            metadata={"help": "Scale embeddings by diving by sqrt(d_model)."})
    beam_size: Optional[int] = field(default=1,
                                     metadata={"help": "beam size for beam search. 1 indicates greedy search."})
    beam_min_length: Optional[int] = field(default=2, metadata={"help": "Minimum beam length."})
    beam_max_length: Optional[int] = field(default=100, metadata={"help": "Maximum beam length."})
    beam_length_penalty: Optional[float] = field(default=1.0, metadata={"help": "Length penalty in beam search."})
    no_repeat_ngram_size: Optional[int] = field(default=3,
                                                metadata={"help": "The ngram size not permitted in beam search."})
    reverse_direction: Optional[bool] = field(default=False,
                                              metadata={"help": "whether to predict at reverse direction"})


# ======================================================================================== #

def is_none(argument):
    return argument is None or argument == 'None'


# run function
def run():
    # =========== parse arguments =========== #
    parser = HfArgumentParser((FileTrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()

    # # =========== Detecting last checkpoint =========== #
    # last_checkpoint = None
    # if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #     if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
    #         raise ValueError(
    #             f"Output directory ({training_args.output_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #     elif last_checkpoint is not None:
    #         logger.info(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )

    # =========== setup logger =========== #
    logging.basicConfig(format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s', datefmt="%m/%d/%Y %H:%M:%S")
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

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
        tokenizer = TudTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        # use facebook bart tokenizer default
        logger.info("Use default Bart tokenizer.")
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # data collator
    def data_collator(batch):
        return batchify(batch, tokenizer, data_args.max_sent_length)

    decoders = list(map(lambda x: x.strip(), model_args.decoders.split(',')))

    config = TudConfig(vocab_size=len(tokenizer), max_position_embeddings=model_args.max_pos_length,
                       encoder_layers=model_args.encoder_layers, decoder_layers=model_args.decoder_layers,
                       d_model=model_args.d_model, encoder_attention_heads=model_args.encoder_attention_heads,
                       decoder_attention_heads=model_args.decoder_attention_heads,
                       encoder_ffn_dim=model_args.encoder_ffn_dim, decoder_ffn_dim=model_args.decoder_ffn_dim,
                       activation_function=model_args.activation_function, dropout=model_args.dropout,
                       attention_dropout=model_args.attention_dropout,
                       activation_dropout=model_args.activation_dropout,
                       classifier_dropout=model_args.classifier_dropout, init_std=model_args.init_std,
                       scale_embedding=model_args.scale_embedding, bos_token_id=tokenizer.bos_token_id,
                       eos_token_id=tokenizer.eos_token_id, decoder_start_token_id=tokenizer.eos_token_id,
                       pad_token_id=tokenizer.pad_token_id,
                       decoders=decoders)

    # train
    if training_args.do_train:
        logger.info("Start to prepare for training.")
        if data_args.lowercase:
            info = f"Build training dataset from {data_args.train_file} with lowercase."
        else:
            info = f"Build training dataset from {data_args.train_file} without lowercase."
        logger.info(info)
        # dataset
        train_set = SummarizationDataset(data_args.train_file, data_args.lowercase)

        # model
        if not is_none(model_args.resume_checkpoint):
            logger.info(f"Continue to train with checkpoint: {model_args.resume_checkpoint}")
            model = TudForConditionalGeneration(config)
        elif not is_none(model_args.model_name_or_path):
            logger.info(f"Initializing model from {model_args.model_name_or_path}.")
            model = TudForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        else:
            logger.info("Initializing model from scratch...")
            model = TudForConditionalGeneration(config)
            if not is_none(model_args.with_bart_initialize):
                logger.info(f"Reinitializing part parameters with {model_args.with_bart_initialize}...")
                model.model = TudModel.from_pretrained(model_args.with_bart_initialize, config=config)

        trainer = TudGenerationTrainer(model=model, args=training_args, train_dataset=train_set, tokenizer=tokenizer,
                                       data_collator=data_collator)
        logger.info("Start to train...")
        logger.info("******* train *******")

        trainer.train(resume_from_checkpoint=model_args.resume_checkpoint)

        logger.info("Finish training!")

    # prediction
    if training_args.do_predict:
        assert not is_none(data_args.pred_file) and not is_none(data_args.save_path)

        if not data_args.output_to_single_file and not os.path.exists(data_args.save_path):
            logger.info(f"Create directory: {data_args.save_path}")
            os.makedirs(data_args.save_path)

        elif data_args.output_to_single_file:
            dirname = os.path.dirname(data_args.save_path)
            if not os.path.exists(dirname):
                logger.info(f"Create directory: {dirname}")
                os.makedirs(dirname)
            logger.info(f"Output to the single file: {data_args.save_path} .")

        logger.info(f"Build dataset from {data_args.pred_file}.")
        pred_set = SummarizationDataset(data_args.pred_file, data_args.lowercase)

        if not is_none(model_args.model_name_or_path):
            logger.info(f"Load model from {model_args.model_name_or_path}.")
            model = TudForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        else:
            logger.info("Use the model initialized from scratch.")
            model = TudForConditionalGeneration(config)

        # reverse direction
        if model_args.reverse_direction:
            model.config.bos_token_id = tokenizer.eos_token_id
            model.config.eos_token_id = tokenizer.bos_token_id
            model.config.decoder_start_token_id = tokenizer.bos_token_id

        model = model.cuda()

        logger.info("Start to predict...")
        logger.info("******* predict *******")

        dataloader = torch.utils.data.DataLoader(pred_set, batch_size=training_args.per_device_eval_batch_size,
                                                 shuffle=False, collate_fn=data_collator)

        if model_args.reverse_direction:
            decoder_mode = 'R2L'
        else:
            decoder_mode = 'L2R'

        if data_args.output_to_single_file:
            outf = open(data_args.save_path, 'w')

        with torch.no_grad():
            order = 0
            for batch in tqdm(dataloader):
                batch = move_to_cuda(batch)
                batch_sent_ids = model.generate(batch['input_ids'], attention_mask=batch['attention_mask'],
                                                num_beams=model_args.beam_size, min_length=model_args.beam_min_length,
                                                max_length=model_args.beam_max_length,
                                                length_penalty=model_args.beam_length_penalty,
                                                no_repeat_ngram_size=model_args.no_repeat_ngram_size,
                                                decoder_mode=decoder_mode)

                for sent_id in batch_sent_ids:
                    filename = f"{order}_decode.txt"
                    if model_args.reverse_direction:
                        sent_id = torch.flip(sent_id, dims=[0])
                    sent_text = tokenizer.decode(sent_id, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)

                    if not data_args.output_to_single_file:
                        with open(os.path.join(data_args.save_path, filename), 'w') as outf:
                            outf.write(f"{sent_text}\n")
                    else:
                        outf.write(f"{sent_text}\n")
                        outf.flush()
                    order += 1


if __name__ == '__main__':
    run()
