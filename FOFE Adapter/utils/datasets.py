import os
from collections import deque
import torch
from torch.utils.data import Dataset, DataLoader
import datasets
import nltk
import numpy as np
from datasets import load_dataset


class CoLADataset(Dataset):

    def __init__(self,
                 data_args,
                 model_args,
                 mode='train'):  # mode:train,validation,test
        data_files = {
            "train": data_args.train_file,
            "validation": data_args.validation_file,
            "test": data_args.test_file
        }

        dataset = load_dataset(
            "csv",
            data_files=data_files[mode],
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        self.raw_dataset = dataset['train']

    def __len__(self):
        return len(self.raw_datasets)

    def __getitem__(self, idx):

        return self.raw_datasets[idx]
