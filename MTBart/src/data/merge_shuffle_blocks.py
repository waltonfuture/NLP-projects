import argparse
import os
from tqdm import tqdm


def parse():
    parser = argparse.ArgumentParser('merge')

    parser.add_argument('--dir', type=str)
    parser.add_argument('--block_num', type=int)
    parser.add_argument('--output', type=str)

    return parser.parse_args()

args = parse()

with open(args.output, 'w') as outf:
    for block in tqdm(range(args.block_num)):
        filepath = os.path.join(args.dir, f"block_{block}.idx")
        with open(filepath, 'r') as inf:
            while True:
                line = inf.readline()

                if not line.strip():
                    break
                outf.write(line)
