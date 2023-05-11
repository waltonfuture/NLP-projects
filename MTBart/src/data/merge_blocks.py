import os
from tqdm import tqdm
import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_dir', type=str, help="directory to merge")
    parser.add_argument('--dst_dir', type=str, help="destination directory")
    parser.add_argument('--file_mode', type=str, help="file mode")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    files = sorted(os.listdir(args.src_dir))

    with open(os.path.join(args.dst_dir, "train.large.txt"), args.file_mode) as outf:
        for filename in tqdm(files):
            filename = os.path.join(args.src_dir, filename)
            with open(filename, 'r') as inf:
                for line in inf:
                    outf.write(line)
