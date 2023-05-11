import sys

sys.path.append(".")

import argparse
import os
from tqdm import  tqdm

def parse():
    parser = argparse.ArgumentParser('shuffle parser')

    parser.add_argument('--data_file', type=str)
    parser.add_argument('--dir', type=str)
    parser.add_argument('--block', type=int)

    return parser.parse_args()


args = parse()

idx_file = os.path.join(args.dir, f"block_{args.block}.idx")
out_file = os.path.join(args.dir, f"block_{args.block}.shuffle")

with open(idx_file, 'r') as inf:
    ids = inf.read().strip().split('\n')
ids = list(map(int, ids))
"""
打开 idx_file 文件，并读取其中的内容为一个字符串，再将其按行分割成一个列表 ids，每个元素为一个字符串。
然后将 ids 中的每个字符串都转换为整数类型，并存储在 ids 列表中。
"""

with open(args.data_file, 'r') as inf:
    with open(out_file, 'w') as outf:
        for idx in tqdm(ids):
            inf.seek(idx, os.SEEK_SET)
            outf.write(inf.readline())

"""
打开 args.data_file 文件，并循环遍历 ids 列表中的元素。
在每次循环中，根据 idx 从文件中读取一行，并将其写入 out_file 文件中。
"""
print(f"block {args.block} is finished!")
