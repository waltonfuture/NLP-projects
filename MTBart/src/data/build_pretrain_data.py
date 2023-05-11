"""
This script is to build pretrain data. Specifically, in the following format:
each line is composed of two sequences seperated by [PRETRAIN_FILE_SEPERATOR]

<Encoder Seq> [Seperator] <Decoder Seq>

### NOTE:
    [TODO] UPDATE 1: add extra mask info.
        * span mask: add extra spans: [format: TODO]
        * sentence mask: introduce extra sentence concepts: [format: TODO]
"""
import sys
sys.path.append(".")

import argparse
import os
import logging
import json
from transformers import BartTokenizer
from data.utils import load_sents, build_sequences, process_line, get_sents
from datetime import datetime
from data.const import PRETRAIN_FILE_SEPERATOR
from transformers.trainer_utils import set_seed
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--blocks_dir', type=str, help="blocks directory")
    parser.add_argument('--block_files', type=str, nargs='+', help="block files list")
    parser.add_argument('--output_file', type=str, help="the output file to the processed pretrain data")
    parser.add_argument('--max_subword_len', type=int, default=512, help="maximum subword sequence length")
    parser.add_argument('--sent_perm_ratio', type=float, default=1.0, help="sentence permutation ratio")
    parser.add_argument('--mask_random_ratio', type=float, default=0.2,
                        help="ratio of replacing the mask span to a random token")
    parser.add_argument('--mask_span_ratio', type=float, default=0.3, help="the mask span ratio")
    parser.add_argument('--build_data', action='store_true', help="whether to build pretrain data")
    parser.add_argument('--build_index', action='store_true', help="whether to build the index")
    parser.add_argument('--poisson_lambda', type=float, default=3.0,
                        help="the lambda of poisson distribution of masked span length")
    parser.add_argument('--span_poisson_lambda', type=float, default=3.0,
                        help="the lambda of poisson distribution of masked span length")
    parser.add_argument('--tokenizer', type=str, help="pretrained tokenizer name")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--log_steps', type=int, default=50000, help="logging steps")

    return parser.parse_args()


def process_data(args):
    """
    """
    if args.build_data:
        # check output file
        if not os.path.exists(args.output_file):
            dirname = os.path.dirname(args.output_file)
            os.makedirs(dirname, exist_ok=True)

        logging.info("Building tokenizer...")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

        logging.info("Start to process blocks...")
        with open(args.output_file, 'w') as outf:
            for block_file in args.block_files:
                block_filename = os.path.join(args.blocks_dir, block_file)
                logging.info(f"Start processing block: \033[1;32;40m{block_filename}\033[0m.")

                start = datetime.now()
                logging.info(f"Loading and tokenizing sentences...")
                doc_tokenized_sents = load_sents(block_filename, tokenizer, args.max_subword_len)

                cnt = 0
                for tokenized_sents in doc_tokenized_sents:
                    tokenized_seqs = build_sequences(tokenized_sents, args.max_subword_len) #让每段sequence长度为max_subword_len

                    for tokenized_seq in tokenized_seqs:
                        noised_seq = process_line(tokenized_seq, args.sent_perm_ratio, args.mask_span_ratio, tokenizer,
                                                  args.poisson_lambda, args.mask_random_ratio)
                        origin_seq = get_sents(tokenized_seq, tokenizer, clean=True)

                        # print(origin_seq)
                        while not noised_seq:
                            logging.info(f"Empty noised sequence and recreate!")
                            noised_seq = process_line(tokenized_seq, args.sent_perm_ratio, args.mask_span_ratio, tokenizer,
                                                      args.poisson_lambda, args.mask_random_ratio)

                        noised_seq = noised_seq.strip()
                        origin_seq = origin_seq.strip()
                        total_len = min(args.max_subword_len, len(tokenizer.tokenize(origin_seq)) + 2)

                        segments = []
                        sum = 0

                        while sum < total_len:
                            next_len = np.random.poisson(args.span_poisson_lambda, 1).item()
                            if next_len == 0:
                                continue
                            if sum + next_len > total_len:
                                next_len = total_len - sum
                            segments.append(next_len)
                            sum += next_len
                        # segments??
                        example = {
                            'src': noised_seq,
                            'tgt': origin_seq,
                            'spans': segments
                        }
                        line = json.dumps(example)
                        outf.write(f"{line}\n")

                        cnt += 1

                        if not cnt % args.log_steps:
                            logging.info(f"processed {cnt} samples.")
                    outf.flush()
                end = datetime.now()
                logging.info(
                    f"\n ======= Summary ======\n \033[1;34;40m time: {(end - start).seconds} seconds \n  number: {cnt}\033[0m")

    if args.build_index:
        # build index file
        logging.info("Start to build index file...")
        dirname = os.path.dirname(args.output_file)
        index_file = os.path.join(dirname, f"{os.path.basename(args.output_file)}.index")

        cnt = 0
        pre = 0
        with open(index_file, 'w') as outf:
            with open(args.output_file, 'r') as inf:
                while True:
                    line = inf.readline()
                    if not line:
                        break
                    outf.write(f"{pre}\n")
                    pre = inf.tell()
                    cnt += 1
                    if cnt % 10000 == 0:
                        logging.info(f"processed {cnt}")

        # with open(index_file, 'r') as inf:
        #     offsets = inf.read().strip().split('\n')[:-1]
        #
        # with open(index_file, 'w') as outf:
        #     outf.write('\n'.join(offsets))


if __name__ == '__main__':
    args = parse()
    logging.info(f"\n ====== CONFIG ====== \n{json.dumps(vars(args), indent=4)}")

    set_seed(args.seed)

    logging.info(f" Start to get pretrain data...")
    start = datetime.now()
    process_data(args)
    end = datetime.now()

    logging.info(f"Finished! With time: \033[1;34;40m{(end - start)}\033[0m")
