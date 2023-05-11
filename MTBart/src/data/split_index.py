import sys
sys.path.append(".")

import os
from tqdm import tqdm
from data.dataset import ShuffledEncoderDecoderDataset
from data.const import PRETRAIN_FILE_SEPERATOR
import torch
import json
import random

prefix = "/mnt/lustre/sjtu/home/dm311/remote/pretrain/data/pretrain"
index_file = f"{prefix}/train.large.txt.index"
# data_file = f"{prefix}/test.txt"
# index_file = f"{prefix}/test.txt.index"
BLOCK = 512
random.seed(42)
tmp = f"{prefix}/tmp"

with open(index_file, 'r') as inf:
    ids = inf.read().strip().split('\n')

random.shuffle(ids)
block_num = len(ids) // BLOCK
left = len(ids) % BLOCK

start = 0
end = 0
for block in range(BLOCK):
    filename = os.path.join(tmp, f"block_{block}.idx")
    start = end
    if block < left:
        end = min(len(ids), start + block_num + 1)
    else:
        end = min(len(ids), start + block_num)

    with open(filename, 'w') as outf:
        outf.write('\n'.join(ids[start: end]) + "\n")

"""
这段代码首先导入了一些必要的包和模块，包括了sys，os，tqdm，torch等等。其中，data.dataset和data.const模块的导入语句暗示了它们是一个自定义的模块和常量模块。

接下来，定义了一些常量和变量。prefix是一个目录前缀，index_file是一个索引文件的路径，BLOCK是一个块的大小，tmp是一个临时目录。此外，random.seed(42)是一个随机数种子，用于保证随机化的可重复性。

然后，打开了一个索引文件，将其中的行读入一个ids列表中。random.shuffle(ids)函数随机化了这个列表，使得每个块中的行都被随机打乱。block_num表示ids中整块的个数，left表示剩下不足一块的部分的大小。

接下来，遍历所有的块。在每个块中，根据块的编号，定义了一个文件名，然后取出这个块在ids中的起始和结束位置。如果这个块是不完整的（即block小于left），则需要多取一个样本；否则不用多取。然后，将这个块对应的样本id写入一个文件中。
"""