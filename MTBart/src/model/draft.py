import torch
import torch.nn.functional as F
import random
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
def draft_test():

        first = torch.tensor(tokenizer('I have a dream',padding='max_length',max_length=6)['input_ids'])
        second = torch.tensor(tokenizer('I do',padding='max_length',max_length=6)['input_ids'])
        third = torch.tensor(tokenizer('You and me', padding='max_length', max_length=6)['input_ids'])
        input_ids = torch.stack((first, second, third), dim=0)
        L2R_decoder_inputs = input_ids
        #L2R_decoder_inputs = torch.where(L2R_decoder_inputs == tokenizer.sep_token_id, tokenizer.pad_token_id, L2R_decoder_inputs)
        R2L_decoder_inputs = torch.full_like(L2R_decoder_inputs, tokenizer.pad_token_id)

        lens = input_ids.ne(tokenizer.pad_token_id).sum(dim=1)  # [:lens[i]] 不是pad_token_id的部分
        for i in range(lens.size(0)):
            R2L_decoder_inputs[i][:lens[i]] = input_ids[i][:lens[i]].flip(dims=[0])

        # R2L_decoder_inputs = shift_tokens_right(R2L_decoder_inputs, tokenizer.pad_token_id)
        # get L2R and R2L lengths
        bsz, tgt_len = input_ids.size()
        L2R_len = torch.full((bsz, tgt_len), 0, dtype=torch.long)
        R2L_len = L2R_len.clone()
        for i in range(lens.size(0)):
            prefix = torch.arange(1,lens[i]+1)
            L2R_len[i][:lens[i]] = prefix
            R2L_len[i][:lens[i]] = torch.flip(prefix, [0])
        L2RInputIds = L2R_decoder_inputs[:,:-1]
        print(f"L2R_ids:{L2RInputIds}")
        R2LInputIds = R2L_decoder_inputs[:,:-1]
        print(f"R2L_ids:{R2LInputIds}")
        L2R_positions = L2R_len[:,:-1].to(L2RInputIds.device)
        print(f"L2R_pos:{L2R_positions}")
        Bert_positions = L2R_len.to(L2RInputIds.device)
        print(f"Bert_pos:{Bert_positions}")
        R2L_positions = R2L_len[:,:-1].to(L2RInputIds.device)
        print(f"R2L_pos:{R2L_positions}")
        #print(torch.arange(R2LInputIds.size(1), 0, -1))
        # PredictionStream
        # prepare attention mask [bsz, 1, tgt_seq_len, src_seq_len]
        inputShape = L2RInputIds.size()
        bsz, seq_len = inputShape
        # [bsz, seq_len, seq_len]
        diagMask = torch.eye(seq_len, dtype=torch.bool, device=L2RInputIds.device).repeat(bsz, 1, 1) # [bsz, seq_len, seq_len] 单位矩阵
        mask_cond = torch.arange(seq_len, device=L2RInputIds.device)
        causalMask = (mask_cond <= mask_cond.view(-1, 1)).repeat(bsz, 1, 1) #bsz个下三角矩阵(下三角和对角线为true)
        #print(causalMask)
        L2RPredictionStreamAttentionMask = torch.cat([causalMask, diagMask], dim=-1).unsqueeze(1) #[bsz,1,seq_len,2*seq_len] 下三角拼接对角矩阵
        #print(L2RPredictionStreamAttentionMask)


        R2LPredictionStreamAttentionMask = torch.cat([causalMask, diagMask], dim=-1).unsqueeze(1)

        mask_cond = torch.arange(seq_len + 1, device=L2RInputIds.device)
        L2RCausalMask = (mask_cond <= mask_cond.view(-1, 1)).repeat(bsz, 1, 1)
        #print(L2RCausalMask.shape)
        nodiagCausalMask = (mask_cond < mask_cond.view(-1, 1)).repeat(bsz, 1, 1) #bsz个下三角为true（不包括对角线）的矩阵

        ##
        decoderPaddingMask = torch.eq(input_ids,
                                      tokenizer.pad_token_id)  # self.padding_idx: The id of the _padding_ token
        decoderPaddingMask = ~decoderPaddingMask
        pad_num = torch.sum(~decoderPaddingMask, dim=1, keepdim=True).view(bsz,)
        #print(pad_num)
        offsets = torch.sum(decoderPaddingMask, dim=1, keepdim=True).unsqueeze(-1)
        baseCausalNum = mask_cond.repeat(bsz, seq_len + 1, 1)
        upperBound = torch.arange(seq_len, -1, -1).view(-1, 1)
        #print(upperBound)
        causalMask = (mask_cond <= upperBound).repeat(bsz, 1, 1)
        #causalMask[:, 0] = 0

        #paddingCausalMask = (baseCausalNum < offsets)
        r2l = torch.clone(causalMask)
        r2la = torch.clone(causalMask)
        #print(pad_num[0],pad_num[1])
        for i in range(bsz):
            m = r2la[i]
            n = pad_num[i]
            r2la[i] = torch.cat([m[n:], m[:n]], dim=0)
        #print(r2la[:, :, 1:])
        indices = torch.arange(seq_len+1).unsqueeze(0).repeat(bsz, 1)
        shifts = pad_num.unsqueeze(1)
        shifted_indices = (indices + shifts) % (seq_len+1)
        r2l = r2l.gather(1, shifted_indices.view(bsz,seq_len+1,1).repeat(1,1,seq_len+1))

        #R2LCausalMask = torch.logical_and(causalMask, paddingCausalMask)
        R2LCausalMask = r2l
        #print(R2LCausalMask)
        ##

        '''decoderPaddingMask = torch.eq(L2R_decoder_inputs,tokenizer.pad_token_id)  # self.padding_idx: The id of the _padding_ token
        decoderPaddingMask = ~decoderPaddingMask
        offsets = torch.sum(decoderPaddingMask, dim=1, keepdim=True).unsqueeze(-1)
        baseCausalNum = mask_cond.repeat(bsz, seq_len + 1, 1)
        paddingCausalMask = (baseCausalNum < offsets)
        R2LCausalMask = torch.logical_and(~nodiagCausalMask, paddingCausalMask)'''
        diagMask = torch.eye(seq_len + 1, dtype=torch.bool, device=L2RInputIds.device).repeat(bsz, 1, 1)
        #print(R2LCausalMask)
        #print(L2RCausalMask[:,:,1:])
        print(R2LCausalMask[:,:,1:])
        BertMask = torch.cat([L2RCausalMask[:,:,1:], R2LCausalMask[:,:,:1:], diagMask], dim=-1).unsqueeze(1)
        #print(BertMask)
        inputShape = L2R_decoder_inputs.size()
        specialtoken_mask = input_ids[:, 1:].eq(2)
        #print(L2RInputIds)
        #print(specialtoken_mask)
        bert_sep_mask = input_ids.eq(2)
        bert_cls_mask = input_ids.eq(0)
        #print(bert_sep_mask)
        bert_token_mask = ~torch.bitwise_or(bert_cls_mask, bert_sep_mask)
        #print(bert_token_mask)


def data_collator(batch):
        '''
        data = {'attention_mask': [],
                'input_ids': [],
                'start_positions': [],
                'end_positions':[]
                }
        for item in batch:
            for key in item:
                if key in data:
                    data[key].append(item[key])

        '''
        keys = batch[0].keys()
        data = {}
        for key in keys:
            data[key] = []

        for item in batch:
            for key in item:
                data[key].append(item[key])

        for key in data:
            data[key] = torch.LongTensor(data[key])

        data['decoder_input_ids'] = {}
        #L2R_decoder_inputs = shift_tokens_right(data['input_ids'], tokenizer.pad_token_id)
        L2R_decoder_inputs = data['input_ids']

        #specialtoken_mask = L2R_decoder_inputs[:, 1:]
        #indexes = torch.nonzero(specialtoken_mask == 2).tolist()
        #print(indexes)

        R2L_decoder_inputs = torch.full_like(L2R_decoder_inputs, tokenizer.pad_token_id)
        lens = data['input_ids'].ne(tokenizer.pad_token_id).sum(dim=1)  # [:lens[i]] 不是pad_token_id的部分
        for i in range(lens.size(0)):
            R2L_decoder_inputs[i][:lens[i]] = data['input_ids'][i][:lens[i]].flip(dims=[0])
        #R2L_decoder_inputs = shift_tokens_right(R2L_decoder_inputs, tokenizer.pad_token_id)
        # get L2R and R2L lengths
        bsz, tgt_len = data['input_ids'].size()
        L2R_len = torch.full((bsz, tgt_len), 0, dtype=torch.long)
        R2L_len = L2R_len.clone()
        for i in range(lens.size(0)):
            prefix = torch.arange(1,1+lens[i])
            L2R_len[i][:lens[i]] = prefix
            R2L_len[i][:lens[i]] = torch.flip(prefix, [0])
        data['decoder_input_ids'] = {
            'L2R': L2R_decoder_inputs,
            'R2L': R2L_decoder_inputs,
            'BERT': data['input_ids'],
            'L2R_len': L2R_len,
            'R2L_len': R2L_len,
        }

        return data


draft_test()


def _build_base_fofe_matrix(window_size, max_len):
    weigths = [[1.0], [0.5, 1.0, 0.5], [0.2, 0.5, 1.0, 0.5, 0.2], [0.1, 0.2, 0.5, 1.0, 0.5, 0.2, 0.1]]
    weigths = weigths[(window_size - 1) // 2]
    base_fofe_vec = [0.0 for i in range(max_len)]
    for i in range(window_size):
        base_fofe_vec[i] = weigths[i]

    base_fofe_vec = torch.FloatTensor(base_fofe_vec).view(-1, 1)  # 变成列向量
    base_fofe_vecs = []
    for i in range(max_len):
        base_fofe_vecs.append(
            torch.clone(torch.roll(base_fofe_vec, (i - (window_size - 1) // 2), 0)))

    base_fofe_matrix = torch.cat(base_fofe_vecs, dim=1)
    base_fofe_matrix[max_len - 1][0] = base_fofe_matrix[max_len - 2][0] = base_fofe_matrix[max_len - 1][1] = 0.0
    base_fofe_matrix[0][max_len - 1] = base_fofe_matrix[0][max_len - 2] = base_fofe_matrix[1][max_len - 1] = 0.0

    return base_fofe_matrix


#base_fofe_l2r = _build_base_fofe_matrix(3,8)
#print(base_fofe_l2r)

#from datasets import load_dataset

#dataset = load_dataset('squad')
def k():
    output_data = []
    for article in raw_datasets:
        for p in article['paragraphs']:
            for qas in p['qas']:
                answers = {
                    "text": [],
                    "answer_start": []
                }
                for ans in qas['answers']:
                    answers['text'].append(ans['text'])
                    answers['answer_start'].append(ans['answer_start'])

                output_data.append({
                    "id": qas['id'],
                    "context": p['context'],
                    "question": qas['question'],
                    "answers": answers
                })