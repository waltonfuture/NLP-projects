""" TUD model configuration, refer to BART configuration from huggingface transformers """

from transformers import BartConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class MTConfig(BartConfig):
    def __init__(self, decoders=['L2R', 'R2L', 'BERT'], Lambda=0.1, L2RMaskIdx=50266, R2LMaskIdx=50267, BertMaskIdx=50268, use_bertmask=True,l2r_mask = False, r2l_mask = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoders = decoders
        self.Lambda = Lambda
        self.L2RMaskIdx = L2RMaskIdx
        self.R2LMaskIdx = R2LMaskIdx
        self.BertMaskIdx = BertMaskIdx
        self.use_bertmask = use_bertmask
        self.l2r_mask = l2r_mask
        self.r2l_mask = r2l_mask
    def set_decoders(self, decoders):
        self.decoders = list(map(lambda x: x.strip(), decoders.strip().split(',')))

    def set_lambda(self, Lambda):
        self.Lambda = Lambda
