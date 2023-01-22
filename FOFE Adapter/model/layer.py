#from unicodedata import bidirectional

#import pandas as pd
import torch.nn as nn
import torch
import math
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification
import random
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class FOFELayerWindow(nn.Module):
    def __init__(self,
                 alpha=0.5,
                 max_len=128,
                 mode="concat",
                 dim=768,
                 fofe_dim = 128,
                 is_conv1d = True,
                 window_size = None,
                 ):
        super().__init__()
        self.window_size = window_size
        self.is_conv1d = is_conv1d
        self.max_len = max_len
        temp_dim = fofe_dim
        self.reduction = nn.Linear(dim, temp_dim)

        if self.window_size == 0 or self.window_size == 1:
            self.is_conv1d = False
        if self.window_size != None and self.is_conv1d == True:
            self.conv1d = nn.Conv1d(in_channels=fofe_dim,out_channels=fofe_dim,kernel_size=self.window_size,padding=self.window_size//2)

        if mode == "concat":
            self.concat = True
            self.expand = nn.Linear(temp_dim * 2 , dim)
        else:
            self.concat = False
            self.expand = nn.Linear(temp_dim , dim)

        self.alpha = alpha
        self.use_grad = False
        self.base_fofe_l2r = self._build_base_fofe_matrix(max_len)
        if torch.cuda.is_available():
            self.base_fofe_l2r = self.base_fofe_l2r.cuda()

        #self.reduction.apply(self.init_bert_weights)
        #self.expand.apply(self.init_bert_weights)


    def _build_base_fofe_matrix(self, max_len):

        weigths = [[1.0],[0.5,1.0,0.5],[0.2,0.5,1.0,0.5,0.2],[0.1,0.2,0.5,1.0,0.5,0.2,0.1]]
        weigths = weigths[(self.window_size-1) // 2]
        base_fofe_vec = [0.0 for i in range(max_len)]
        for i in range(self.window_size):
            base_fofe_vec[i] = weigths[i]

        base_fofe_vec = torch.FloatTensor(base_fofe_vec).view(-1, 1)  # 变成列向量
        base_fofe_vecs = []
        for i in range(max_len):
            base_fofe_vecs.append(
                torch.clone(torch.roll(base_fofe_vec,(i-(self.window_size-1) // 2),0)))

        base_fofe_matrix = torch.cat(base_fofe_vecs, dim=1)
        base_fofe_matrix[max_len - 1][0] = base_fofe_matrix[max_len - 2][0] = base_fofe_matrix[max_len - 1][1] = 0.0
        base_fofe_matrix[0][max_len - 1] = base_fofe_matrix[0][max_len - 2] = base_fofe_matrix[1][max_len - 1] = 0.0

        return base_fofe_matrix

    def forward(self,
                char_inputs,
                word_left_bound,
                word_right_bound,
                seg_bos=None,
                seg_eos=None):
        # [bsz, seq_len, dim]
        bsz, seq_len, dim = char_inputs.size()
        device = char_inputs.device
        x = torch.relu(self.reduction(char_inputs))

        if self.is_conv1d == True:
            fofe_out = self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1)
            return self.expand(fofe_out)

        else:
            fofe_window = self.base_fofe_l2r[:seq_len, :seq_len].repeat(bsz, 1, 1)
            fofe_out = fofe_window.bmm(x)
            if self.window_size == 0:
                return self.expand(fofe_out) * 0.0
            else:
                return self.expand(fofe_out)

    @staticmethod
    def init_bert_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



class FOFELayer(nn.Module):

    def __init__(self,
                 alpha=0.5,
                 max_len=128,
                 mode="concat",
                 dim=768,
                 fofe_dim = 64,
                 ):  # dim_scale = num_attention_heads
        super().__init__()

        self.max_len = max_len
        temp_dim = fofe_dim
        self.reduction = nn.Linear(dim, temp_dim)
        self.dropout = nn.Dropout(0.1)

        if mode == "concat":
            self.concat = True
            self.expand = nn.Linear(temp_dim * 2 , dim)
        else:
            self.concat = False
            self.expand = nn.Linear(temp_dim , dim)

        if alpha is None:
            self.alpha = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
            self.use_grad = True
            self.base_fofe_l2r = self._build_base_fofe_matrix(max_len)
            if torch.cuda.is_available():
                self.base_fofe_l2r = self.base_fofe_l2r.cuda()

        else:
            self.alpha = alpha
            self.use_grad = False
            self.base_fofe_l2r = self._build_base_fofe_matrix(max_len)
            if torch.cuda.is_available():
                self.base_fofe_l2r = self.base_fofe_l2r.cuda()

        self.reduction.apply(self.init_bert_weights)
        self.expand.apply(self.init_bert_weights)


    def _build_base_fofe_matrix(self, max_len):

        base_fofe_vec = range(max_len)
        base_fofe_vec = torch.FloatTensor(base_fofe_vec).view(-1, 1)  # 变成列向量
        base_fofe_vecs = []

        for _ in range(max_len):
            base_fofe_vecs.append(
                torch.clone(base_fofe_vec))  # max_len应该是指文本长度把，torch.clone
            base_fofe_vec = torch.roll(base_fofe_vec, 1, 0)  # 每个行都向下移一位

        base_fofe_matrix = torch.cat(base_fofe_vecs, dim=1)

        return base_fofe_matrix

    def forward(self,
                char_inputs,
                word_left_bound,
                word_right_bound,
                seg_bos=None,
                seg_eos=None):
        # [bsz, seq_len, dim]
        bsz, seq_len, dim = char_inputs.size()
        device = char_inputs.device
        short_cut = char_inputs
        x = torch.relu(self.reduction(char_inputs))
        # calculate mask
        # cut according to segments
        # word_left_bound: [bsz, seq_len] ===> [bsz, seq_len, 1]

        word_left_bound = word_left_bound.unsqueeze(-1)
        word_right_bound = word_right_bound.unsqueeze(-1)  # 感觉right有点多余

        # [1, 1, seq_len]
        mask = (torch.arange(seq_len, device=device).view(
            1, seq_len).unsqueeze(0))
        # [bsz, seq_len, seq_len]
        mask = torch.logical_and(
            mask.ge(word_left_bound),  # mask中大于等于，和小于，做逻辑与
            mask.lt(word_right_bound))

        #mask = mask.ge(word_left_bound)[0]
        mask = ~mask
        # cut base matrix first: [bsz, seq_len, seq_len]
        fofe_l2r = self.base_fofe_l2r[:seq_len, :seq_len].repeat(bsz, 1, 1)
        fofe_l2r = torch.pow(self.alpha, fofe_l2r)

        temp_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(device)
        temp_mask = torch.gt(temp_mask, 0)
        fofe_l2r = fofe_l2r.masked_fill(temp_mask, 0)
        fofe_l2r = fofe_l2r.masked_fill(mask, 0)  # mask中为0的保留原值，为1则设为0

        fofe_r2l = torch.transpose(fofe_l2r, 1, 2)  # r2l转置一下就可以了

        # [bsz, seq_len, dim]
        fofe_l2r_out = fofe_l2r.bmm(x)
        fofe_r2l_out = fofe_r2l.bmm(x)

        # calculate bias
        fofe_bias = fofe_l2r.masked_fill(fofe_l2r == 0, 100)

        # [bsz, seq_len, 1]
        l2r_bos = torch.min(fofe_bias, dim=2)[0].unsqueeze(-1)
        r2l_bos = torch.min(fofe_bias, dim=1)[0].unsqueeze(-1)

        if seg_bos is not None and seg_eos is not None:
            # [1, 1, dim]
            seg_bos = seg_bos.view(1, 1, -1)
            seg_eos = seg_eos.view(1, 1, -1)

            # [bsz, seq_len, dim]
            fofe_l2r_out = fofe_l2r_out + torch.matmul(l2r_bos,
                                                       self.reduction(seg_bos))  # 为什么要乘l2r_bos
            fofe_r2l_out = fofe_r2l_out + torch.matmul(r2l_bos, self.reduction(seg_eos))

        if self.concat:
            fofe_out = torch.cat([fofe_l2r_out, fofe_r2l_out], dim=-1)
            return self.expand(fofe_out)
        else:
            fofe_out = fofe_l2r_out + fofe_r2l_out
            return self.expand(fofe_out)


    @staticmethod
    def init_bert_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


"""
Refer to https://github.com/luozhouyang/TPLinker/blob/3cacf48901f73a4d4e90ed51d8d5bbf8aecb5a02/tplinker/layers_torch.py
"""


class DistanceEmbedding(nn.Module):

    def __init__(self, max_positions=512, embedding_size=768, **kwargs):
        super().__init__()
        self.max_positions = max_positions
        self.embedding_size = embedding_size
        self.dist_embedding = self._init_embedding_table()
        self.register_parameter("distance_embedding", self.dist_embedding)

    def _init_embedding_table(self):
        matrix = np.zeros([self.max_positions, self.embedding_size])
        for d in range(self.max_positions):
            for i in range(self.embedding_size):
                if i % 2 == 0:
                    matrix[d][i] = math.sin(d /
                                            10000**(i / self.embedding_size))
                else:
                    matrix[d][i] = math.cos(
                        d / 10000**((i - 1) / self.embedding_size))
        embedding_table = nn.Parameter(data=torch.tensor(matrix,
                                                         requires_grad=False),
                                       requires_grad=False)
        return embedding_table

    def forward(self, inputs, **kwargs):
        """Distance embedding.

        Args:
            inputs: Tensor, shape (batch_size, seq_len, hidden_size)

        Returns:
            embedding: Tensor, shape (batch_size, 1+2+...+seq_len, embedding_size)
        """
        batch_size, seq_len = inputs.size()[0], inputs.size()[1]
        segs = []
        for index in range(seq_len, 0, -1):
            segs.append(self.dist_embedding[:index, :])
        segs = torch.cat(segs, dim=0)
        embedding = segs[None, :, :].repeat(batch_size, 1, 1)
        return embedding


class TaggingProjector(nn.Module):

    def __init__(self, hidden_size, num_relations, name="proj", **kwargs):
        super().__init__()
        self.name = name
        self.fc_layers = [
            nn.Linear(hidden_size, 3) for _ in range(num_relations)
        ]
        for index, fc in enumerate(self.fc_layers):
            self.register_parameter("{}_weights_{}".format(self.name, index),
                                    fc.weight)
            self.register_parameter("{}_bias_{}".format(self.name, index),
                                    fc.bias)

    def forward(self, hidden, **kwargs):
        """Project hiddens to tags for each relation.

        Args:
            hidden: Tensor, shape (batch_size, 1+2+...+seq_len, hidden_size)

        Returns:
            outputs: Tensor, shape (batch_size, num_relations, 1+2+...+seq_len, num_tags=3)
        """
        outputs = []
        for fc in self.fc_layers:
            outputs.append(fc(hidden))
        outputs = torch.stack(outputs, dim=1)
        # outputs = torch.softmax(outputs, dim=-1)
        return outputs


class ConcatHandshaking(nn.Module):

    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden, **kwargs):
        """Handshaking.

        Args:
            hidden: Tensor, shape (batch_size, seq_len, hidden_size)

        Returns:
            handshaking_hiddens: Tensor, shape (batch_size, 1+2+...+seq_len, hidden_size)
        """
        seq_len = hidden.size()[1]
        handshaking_hiddens = []
        for i in range(seq_len):
            _h = hidden[:, i, :]
            repeat_hidden = _h[:, None, :].repeat(1, seq_len - i, 1)
            visibl_hidden = hidden[:, i:, :]
            shaking_hidden = torch.cat([repeat_hidden, visibl_hidden], dim=-1)
            shaking_hidden = self.fc(shaking_hidden)
            shaking_hidden = torch.tanh(shaking_hidden)
            handshaking_hiddens.append(shaking_hidden)
        handshaking_hiddens = torch.cat(handshaking_hiddens, dim=1)
        return handshaking_hiddens


class ConcatLSTMHandshaking(nn.Module):

    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        self.fc = nn.Linear(hidden_size * 3, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )

    def forward(self, hidden, **kwargs):
        """Handshaking.

        Args:
            hidden: Tensor, shape (batch_size, seq_len, hidden_size)

        Returns:
            handshaking_hiddens: Tensor, shape (batch_size, 1+2+...+seq_len, hidden_size)
        """
        seq_len = hidden.size()[1]
        handshaking_hiddens = []
        for i in range(seq_len):
            _h = hidden[:, i, :]
            repeat_hidden = _h[:, None, :].repeat(1, seq_len - i, 1)
            visibl_hidden = hidden[:, i:, :]
            context, _ = self.lstm(visibl_hidden)
            shaking_hidden = torch.cat([repeat_hidden, visibl_hidden, context],
                                       dim=-1)
            shaking_hidden = self.fc(shaking_hidden)
            shaking_hidden = torch.tanh(shaking_hidden)
            handshaking_hiddens.append(shaking_hidden)
        handshaking_hiddens = torch.cat(handshaking_hiddens, dim=1)
        return handshaking_hiddens


class CDEETPLinker(nn.Module):

    def __init__(self,
                 hidden_size,
                 num_relations,
                 max_positions=512,
                 add_distance_embedding=False,
                 shaking_mode="cat",
                 **kwargs):
        super().__init__()
        if shaking_mode == 'cat':
            self.handshaking = ConcatHandshaking(hidden_size)
        elif shaking_mode == "cat_lstm":
            self.handshaking = ConcatLSTMHandshaking(hidden_size)
        self.h2t_proj = nn.Linear(hidden_size, 2)
        self.h2h_proj = TaggingProjector(hidden_size,
                                         num_relations,
                                         name="h2hproj")
        self.t2t_proj = TaggingProjector(hidden_size,
                                         num_relations,
                                         name="t2tproj")
        self.tendency_proj = nn.Linear(hidden_size, 4)
        self.add_distance_embedding = add_distance_embedding
        if self.add_distance_embedding:
            self.distance_embedding = DistanceEmbedding(
                max_positions, embedding_size=hidden_size)

    def forward(self, hidden, **kwargs):
        """TPLinker model forward pass.

        Args:
            hidden: Tensor, output of BERT or BiLSTM, shape (batch_size, seq_len, hidden_size)

        Returns:
            h2t_hidden: Tensor, shape (batch_size, 1+2+...+seq_len, 2),
                logits for entity recognization
            h2h_hidden: Tensor, shape (batch_size, num_relations, 1+2+...+seq_len, 3),
                logits for relation recognization
            t2t_hidden: Tensor, shape (batch_size, num_relations, 1+2+...+seq_len, 3),
                logits for relation recognization
        """
        handshaking_hidden = self.handshaking(hidden)
        h2t_hidden, rel_hidden = handshaking_hidden, handshaking_hidden
        if self.add_distance_embedding:
            h2t_hidden += self.distance_embedding(hidden)
            rel_hidden += self.distance_embedding(hidden)
        tendency_logits = self.tendency_proj(h2t_hidden)
        h2t_hidden = self.h2t_proj(h2t_hidden)
        h2h_hidden = self.h2h_proj(rel_hidden)
        t2t_hidden = self.t2t_proj(rel_hidden)
        return h2t_hidden, h2h_hidden, t2t_hidden, tendency_logits


if __name__ == "__main__":

    seed_everything(233)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layer = FOFELayer(mode="add",alpha=0.5,dim=10).to(device)
    inputs = torch.linspace(1, 150, 150).view(3, 5, 10)

    left = torch.LongTensor([[0, 0, 2, 2, 2], [0, 1, 1, 1, 1],
                             [0, 0, 0, 3, 3]])  ## longtenor?
    right = torch.LongTensor([[2, 2, 5, 5, 5], [1, 5, 5, 5, 5],
                              [3, 3, 3, 5, 5]])
    bos = torch.rand(10)

    outputs = layer(inputs.to(device), left.to(device), right.to(device),
                    bos.to(device), bos.to(device))

    print(layer)
    print(outputs)
    print(outputs.shape)

    '''
    loss = outputs * 2
    loss.requires_grad_(True)
    loss.sum().backward()
    for n, p in layer.named_parameters():
        if 'alpha' in n:
            print(n, p.grad)
    '''
