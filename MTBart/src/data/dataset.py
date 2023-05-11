from torch.utils.data import Dataset
from transformers.utils import logging
from data.const import SUMMARIZATION_FILE_SEPERATOR, PRETRAIN_FILE_SEPERATOR
import os
from tqdm import tqdm
import json
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch
logger = logging.get_logger()


class ShuffledEncoderDecoderDataset(Dataset):
    def __init__(self, data_file, idx_file):
        self.file_pointer = open(data_file, 'r')

        self.ids = []
        self.skip = 0
        with open(idx_file, 'r') as inf:
            for line in tqdm(inf):
                self.ids.append(int(line.strip()))

    def __getitem__(self, idx):
        if idx < 0:
            return None
        offset = self.ids[idx]
        self.file_pointer.seek(offset, os.SEEK_SET)
        line = self.file_pointer.readline().strip("\n")
        # print(line)
        # print(json.loads(line))
        # print('-' * 50)
        example = json.loads(line).split(PRETRAIN_FILE_SEPERATOR)
        return tuple(example)

    def __len__(self):
        return len(self.ids)

    def setSkip(self, skip):
        self.skip = skip

    def cancelSkip(self):
        self.skip = 0


class SkipDataset(Dataset):
    """
    NOTE: When skipping data, return None to escape the processing of collator function.
    """

    def __init__(self, skipNum: int = 0) -> None:
        super().__init__()
        self.skipNum = skipNum

    def setSkipNum(self, skipNum: int):
        self.skipNum = skipNum

    def clearSkipNum(self):
        self.skipNum = 0


class EncoderDecoderDataset(SkipDataset):
    def __init__(self, data_file, idx_file, skipNum=0):
        super().__init__(skipNum)

        self.data_file = data_file
        self.ids = []
        with open(idx_file, 'r') as inf:
            for line in tqdm(inf):
                if line.strip():
                    self.ids.append(int(line.strip()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self.skipNum > 0:
            self.skipNum -= 1
            return None
        with open(self.data_file, 'r') as file_pointer:
            file_pointer.seek(self.ids[idx], os.SEEK_SET)
            line = file_pointer.readline().strip("\n")
        example = json.loads(line)
        return example['src'], example['tgt'], example['spans']


# ------------------------------- Downstream datasets ------------------------------- #
class SummarizationDataset(Dataset):
    """
    Summarization Datadset class.
    """

    def __init__(self, filename, lowercase=False, fake_L2R_file=None, fake_R2L_file=None) -> None:
        """
        Get a dataset in filename. The data in following format:
            <document>\t\t\t<summary>
        """
        super(SummarizationDataset, self).__init__()

        fake_L2R = None
        fake_R2L = None

        if fake_L2R_file is not None:
            fake_L2R = open(fake_L2R_file, 'r')

        if fake_R2L_file is not None:
            fake_R2L = open(fake_R2L_file, 'r')

        data = []
        with open(filename, 'r') as inf:
            for line in inf.readlines():
                if lowercase:
                    line = line.strip().lower()
                else:
                    line = line.strip()

                example = line.split(SUMMARIZATION_FILE_SEPERATOR)

                if fake_L2R is not None:
                    example.append(fake_L2R.readline().strip())

                if fake_R2L is not None:
                    example.append(fake_R2L.readline().strip())

                data.append(tuple(example))

        if fake_L2R_file is not None:
            fake_L2R.close()

        if fake_R2L_file is not None:
            fake_R2L.close()

        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class SQuADDataset(Dataset):
    def __init__(self, filename):
        super(SQuADDataset, self).__init__()
        with open(filename, 'r') as inf:
            data = json.load(inf)['data']

        self.raw_datasets = []
        for topic in tqdm(data):
            for paragraph in topic['paragraphs']:
                context = paragraph['context']
                for sample in paragraph['qas']:
                    example = {
                        'context': context,
                        'id': sample['id'],
                        'question': sample['question'],
                        'answers': {
                            'answer_start': list(map(lambda x: x['answer_start'], sample['answers'])),
                            'text': list(map(lambda x: x['text'], sample['answers']))
                        }
                    }
                    self.raw_datasets.append(example)

    def __getitem__(self, idx):
        # FIXME：debug
        # x = self.raw_datasets[idx]
        # x["context"] = "I am a boy"
        return self.raw_datasets[idx]
        return x

    def __len__(self):
        return len(self.raw_datasets)


class SkipDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True,
                 seed: int = 0, drop_last: bool = False, skip=0):
        super(SkipDistributedSampler, self).__init__(dataset, num_replicas, rank, shuffle)
        self.skip = skip
        self.seed = seed
        self.drop_last = drop_last

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        indices = [-1] * self.skip + indices[self.skip:]
        return iter(indices)
