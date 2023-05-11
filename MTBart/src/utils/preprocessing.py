import argparse
import logging
from data import const
from datasets import load_from_disk
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s')


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, help="task to preprocess")
    parser.add_argument('--data', type=str, help="data path for the task")
    parser.add_argument('--save', type=str, help="path to save processed data")

    return parser.parse_args()


def process_summarization(data, save):
    """
    Now, all summarization datasets are from huggingface datasets.
    """
    if not os.path.exists(save):
        os.makedirs(save)

    dataset = load_from_disk(data)

    for split in ['train', 'validation', 'test']:
        logging.info(f"processing split: {split}")
        split_set = dataset[split]
        filename = os.path.join(save, f"{split}.txt")

        with open(filename, 'w') as outf:
            # CNN/Daily Mail
            if "cnn" in data.lower():
                for sample in tqdm(split_set):
                    doc = sample['article']
                    # remove (CNN)
                    if doc[:5] == '(CNN)':
                        doc = doc[5:]
                    outf.write(f"{doc}{const.SUMMARIZATION_FILE_SEPERATOR}{sample['highlights']}\n")

            # XSum && Gigaword
            elif "xsum" in data.lower() or "gigaword" in data.lower():
                for sample in tqdm(split_set):
                    outf.write(f"{sample['document']}{const.SUMMARIZATION_FILE_SEPERATOR}{sample['summary']}\n")

            else:
                logging.info(f"ERROR: {data} not found!")


if __name__ == '__main__':
    args = parse()

    assert args.task in ['summarization']

    logging.info(f"preprocessing {args.task}...")
    logging.info(f"origin data path: {args.data}")
    logging.info(f"to save: {args.save}")

    # summarization
    if args.task == "summarization":
        process_summarization(args.data, args.save)

    logging.info("preprocessing is finished!")
