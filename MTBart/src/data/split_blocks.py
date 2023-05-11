import os
import shutil
import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_dir', type=str, help="src directory")
    parser.add_argument('--dst_dir', type=str, help="dst directory")
    parser.add_argument('--block_size', type=int, default=20, help="block size")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    files = sorted(os.listdir(args.src_dir))
    total = len(files)

    start = 0
    block_idx = 0
    while start < total:
        end = min(total, start + args.block_size)

        subdir = os.path.join(args.dst_dir, str(block_idx))
        if not os.path.exists(subdir):
            os.makedirs(subdir)

        for i in range(start, end):
            oldfile = os.path.join(args.src_dir, files[i])
            newfile = os.path.join(subdir, files[i])
            shutil.copyfile(oldfile, newfile)

        block_idx += 1
        start = end
