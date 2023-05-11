import torch
import math
import numpy as np
from tqdm import tqdm


# This function is copied from modeling_bart.py
def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def move_to_cuda(data, ignore_keys={}):
    if isinstance(data, torch.Tensor):
        return data.cuda()
    elif isinstance(data, dict):
        for key in data:
            if key in ignore_keys:
                continue
            data[key] = move_to_cuda(data[key])
        return data
    else:
        return data


def load_sents(filename, tokenizer, max_subword_len=512):
    """Load sentence-format data from filename and return a
    tokenized List[List]. each element is a [tok1, tok2, tok3...]
    NOTE: update to consider removing samples accross documents of Wikipedia.
    return List[List[List]]
    """
    tokenized_sents = []
    doc_tokenized_sents = []
    pre_line = None

    with open(filename, 'r') as inf:
        for line in tqdm(inf.readlines()):
            line = line.strip()
            if pre_line == '' and line == '':
                continue

            # fixed: do not ignore empty lines
            if line:
                toks = tokenizer.tokenize(line)
                toks = toks[:min(max_subword_len, len(toks))]
                tokenized_sents.append(toks)
            else:
                doc_tokenized_sents.append(tokenized_sents)
                tokenized_sents = []

            pre_line = line

        if tokenized_sents:
            doc_tokenized_sents.append(tokenized_sents)

    return doc_tokenized_sents


# ------------- noise functions -------------- #
def permutate_sent(sents, sent_perm_ratio):
    """Permutate sentences in a sequence referring to <sent_perm_ratio>
    """
    if sent_perm_ratio <= 0:
        return sents[:]
    else:
        perm_sent_num = math.ceil(len(sents) * sent_perm_ratio)
        substitutions = np.random.permutation(len(sents))[:perm_sent_num]
        ordering = np.arange(len(sents))
        ordering[substitutions] = substitutions[np.random.permutation(
            perm_sent_num)]

        results = sents[:]
        results[0][0] = 'Ġ' + results[0][0] #??

        for idx, order in enumerate(ordering):
            results[idx] = sents[order]

        results[0][0] = results[0][0].strip('Ġ')
        return results


def insert_mask(tokens, p, mask_random_ratio, mask_token_id, vocab_size):
    """Insert mask or random words referring to <p> for 0-len mask span.
    """
    #merged_idx_sent = insert_mask(
        #        merged_idx_sent, num_inserts / merged_idx_sent.shape[0],
          #      mask_random_ratio, tokenizer.mask_token_id, len(tokenizer))
    if p <= 0:
        return tokens

    num_tokens = len(tokens)
    n = int(math.ceil(num_tokens * p))

    noise_indices = np.random.permutation(num_tokens + n)[:n]
    noise_mask = np.zeros((num_tokens + n,), dtype=np.bool)
    noise_mask[noise_indices] = 1
    result = np.full((n + num_tokens,), -1, dtype=np.int)

    num_random = int(math.ceil(n * mask_random_ratio))
    result[noise_indices[num_random:]] = mask_token_id
    result[noise_indices[:num_random]] = np.random.randint(
        4, vocab_size - 2, num_random
    )  # remove specical tokens [<s>, </s>, <pad>, <unk>, <mask>, [SEP]]

    result[~noise_mask] = tokens

    assert (result >= 0).all()
    return result


def get_word_mask(x):
    """For BPE Tokenizer, get the whole word mask.
    """
    mask = [1]
    for item in x[1:]:
        if item[0] == 'Ġ':
            mask.append(1)
        else:
            mask.append(0)
    return mask


def mask_seq(sents, mask_ratio, tokenizer, poisson_lambda, mask_random_ratio):
    """mask source sequence.
    params:
    -------
        sents: List[List] tokenized seq
    Return:
    -------
        masked merged sequence id: List
    """
    merged_sent = []
    word_mask = []
    for sent in sents:
        word_mask += get_word_mask(sent)
        merged_sent += sent

    if mask_ratio <= 0:
        return tokenizer.convert_tokens_to_ids(merged_sent)
    else:
        word_mask = np.array(word_mask)
        num_to_mask = int(
            math.ceil(word_mask.astype(np.float).sum() * mask_ratio))
        # 统计Ġ出现次数
        if num_to_mask == 0:
            return tokenizer.convert_tokens_to_ids(merged_sent)

        # convert tokens to ids
        merged_idx_sent = np.array(
            tokenizer.convert_tokens_to_ids(merged_sent))

        # construct the length list of masked spans
        cur_len = 0
        lengths = []
        while cur_len < num_to_mask:
            next_len = np.random.poisson(poisson_lambda, 1).item()
            if cur_len + next_len > num_to_mask:
                next_len = num_to_mask - cur_len
            lengths.append(next_len)
            cur_len += next_len

        # Handle 0-length mask (inserts) separately
        num_to_mask = len(lengths)
        lengths = np.array(lengths)
        lengths = lengths[lengths > 0]
        num_inserts = num_to_mask - len(lengths)
        num_to_mask -= num_inserts

        # find where to mask and where to insert, mask first
        word_starts = np.nonzero(word_mask)[0] # [0]只是实际操作需要
        indices = word_starts[np.random.permutation(
            word_starts.shape[0])][:num_to_mask] # mask元素下标
        mask_random = np.random.uniform(size=num_to_mask) < mask_random_ratio
        to_keep = np.ones(merged_idx_sent.shape[0], dtype=np.bool)
        mask_idx = tokenizer.mask_token_id

        # mask start of span
        merged_idx_sent[indices] = mask_idx # mask
        merged_idx_sent[indices[mask_random]] = np.random.randint(
            4,
            len(tokenizer) - 2, mask_random.sum()) # 对于其中随机选取的一部分位置，采用随机的方式替换成其他的单词。

        # mask other tokens in span ??
        word_mask = np.append(word_mask, 0)
        word_mask[-1] = 255
        lengths -= 1
        while indices.shape[0] > 0:
            assert lengths.shape[0] == indices.shape[0]
            lengths -= word_mask[indices + 1].astype(np.int)
            uncompleted = lengths >= 0
            indices = indices[uncompleted] + 1
            lengths = lengths[uncompleted]
            to_keep[indices] = 0

        merged_idx_sent = merged_idx_sent[to_keep]

        if num_inserts > 0:
            merged_idx_sent = insert_mask(
                merged_idx_sent, num_inserts / merged_idx_sent.shape[0],
                mask_random_ratio, tokenizer.mask_token_id, len(tokenizer))

        return merged_idx_sent


def process_line(sents, sent_perm_ratio, mask_ratio, tokenizer, poisson_lambda,
                 mask_random_ratio):
    """
    NOTE: This function is the most important one for pretrained data. We can control the
    noising format in this function.
    """
    per_sents = permutate_sent(sents, sent_perm_ratio)

    merged_masked_sent = mask_seq(per_sents, mask_ratio, tokenizer,
                                  poisson_lambda, mask_random_ratio)

    return tokenizer.decode(merged_masked_sent, )


# ------------------------------------------- #
def merge_sents(src, tgt, tokenizer):
    seq = []
    for sent in src:
        seq += sent
    seq += ['</s>']

    for sent in tgt:
        seq += sent
    return tokenizer.convert_tokens_to_string(seq)


def get_sents(src, tokenizer, clean=False):
    seq = []
    for sent in src:
        seq += sent
    text = tokenizer.convert_tokens_to_string(seq)
    if clean:
        text = tokenizer.clean_up_tokenization(text)
    return text


def build_sequences(tokenized_sents, max_subword_len):
    """
    NOTE: refined version. Now we distribute construted sequences to each process and then each
    process the sequences distinguishly.
    ------------------- (UPDATE 1 --- 3.15 2021) --------------------
    EXCLUDE condition of <a1, a2,...,a_n | b1, b2,...,b_m>
    and <a2,...,a_n | b1, b2,...,b_m>
    ------------------- (UPDATE 2 --- 3.19 2021) --------------------
    SAVE short documents. For those short documents with length less
    than MAXIMUM LENGTH, construct according to the rate also.
    params:
    -------
    tokenized_sents: List[List]
    Return:
        List[Tuple]
    """
    seqs = []

    # merge consecutive sentences into a sequence
    data = tokenized_sents
    data_num = len(data)
    src_start, src_end = 0, 0
    src = []
    src_len = 0

    while src_end < data_num:
        while src_end < data_num and src_len + len(data[src_end]) <= max_subword_len:
            sent = data[src_end][:]
            if src:
                sent[0] = "Ġ" + sent[0].strip('Ġ')
            src.append(sent)
            src_len += len(data[src_end])
            src_end += 1
        if src:
            src[0][0] = src[0][0].strip('Ġ')
            seqs.append(src)
            src = []
            src_len = 0

    """
    UPDATE: DO NOT APPLY OVERLAP

    while src_end < data_num:
        while src_end < data_num and src_len + len(data[src_end]) <= max_subword_len:
            sent = data[src_end][:]
            if src:
                sent[0] = "Ġ" + sent[0].strip('Ġ')
            src.append(sent)
            src_len += len(data[src_end])
            src_end += 1

        if src:
            src[0][0] = src[0][0].strip('Ġ')
            seqs.append(src)

        while src_end < data_num and src_len + len(data[src_end]) > max_subword_len:
            src_start += 1
            src_len -= len(src[0])
            src = src[1:]
    """

    # add smaller sequences
    # while src_len:
    #     src_len -= len(src[0])
    #     if src_len <= max_subword_len:
    #         seqs.append(src)
    #     src = src[1:]

    return seqs


def batchify(batch, tokenizer, max_length, use_span=False):
    """Transform a batch data into the batch format of model with the help of tokenizer.
    params:
    -------
        batch: List[Tuple] each element is (source, target)
        step:
            In pretraining, there are two steps. "first" controls to get the data for first step
            and "second" means the second step.
    """

    batch_src, batch_tgt, batch_spans = zip(*batch)
    inputs = tokenizer.prepare_seq2seq_batch(src_texts=batch_src,
                                             tgt_texts=batch_tgt,
                                             max_length=max_length,
                                             return_tensors='pt')


    data = {}
    data['input_ids'] = inputs['input_ids']
    data['attention_mask'] = inputs['attention_mask']
    data['decoder_input_ids'] = {}
    L2R_decoder_inputs = data['input_ids']

    R2L_decoder_inputs = torch.full_like(L2R_decoder_inputs, tokenizer.pad_token_id)
    lens = data['input_ids'].ne(tokenizer.pad_token_id).sum(dim=1)  # [:lens[i]] 不是pad_token_id的部分
    for i in range(lens.size(0)):
        R2L_decoder_inputs[i][:lens[i]] = data['input_ids'][i][:lens[i]].flip(dims=[0])
    # get L2R and R2L lengths
    bsz, tgt_len = data['input_ids'].size()
    L2R_len = torch.full((bsz, tgt_len), 0, dtype=torch.long)
    R2L_len = L2R_len.clone()
    for i in range(lens.size(0)):
        prefix = torch.arange(1, 1 + lens[i])
        L2R_len[i][:lens[i]] = prefix
        R2L_len[i][:lens[i]] = torch.flip(prefix, [0])
    data['decoder_input_ids'] = {
        'L2R': L2R_decoder_inputs,
        'R2L': R2L_decoder_inputs,
        'BERT': data['input_ids'],
        'L2R_len': L2R_len,
        'R2L_len': R2L_len,
    }
    data['labels'] = {}
    labels = inputs.pop('labels')
    L2R_labels = labels[:, 1:]
    R2L_labels = torch.full_like(L2R_decoder_inputs, tokenizer.pad_token_id)
    lens = labels.ne(tokenizer.pad_token_id).sum(dim=1)
    for i in range(lens.size(0)):
        R2L_labels[i][:lens[i]] = labels[i][:lens[i]].flip(dims=[0])
    R2L_labels = R2L_labels[:, 1:]
    BERT_labels = torch.clone(labels)

    # NOTE: transform pad token id to -100 to ignore the padding loss
    L2R_labels = L2R_labels.masked_fill(L2R_labels == tokenizer.pad_token_id, -100)
    R2L_labels = L2R_labels.masked_fill(R2L_labels == tokenizer.pad_token_id, -100)
    #BERT_labels[:, 0] = -100 #??
    BERT_labels = BERT_labels.masked_fill(BERT_labels == tokenizer.pad_token_id, -100)
    data['labels'] = {
        'L2R': L2R_labels,
        'R2L': R2L_labels,
        'BERT': BERT_labels
    }
    # # reverse L2R to R2L
    # R2L_labels = torch.full_like(L2R_labels, tokenizer.pad_token_id)
    # lens = inputs['labels'].ne(tokenizer.pad_token_id).sum(dim=1)
    # for i in range(lens.size(0)):
    #     R2L_labels[i][:lens[i]] = inputs['labels'][i][:lens[i]].flip(dims=[0])
    # R2L_decoder_inputs = shift_tokens_right(R2L_labels, tokenizer.pad_token_id)
    # R2L_labels = R2L_labels.masked_fill(R2L_labels == tokenizer.pad_token_id, -100)
    #
    # mask_inputs = inputs['labels'].masked_fill(inputs['labels'] != tokenizer.pad_token_id,
    #                                            tokenizer.mask_token_id)
    #
    # if not use_span:
    #     # get L2R and R2L lengths
    #     bsz, tgt_len = mask_inputs.size()
    #     L2R_len = torch.full((bsz, tgt_len), -100, dtype=torch.long)
    #     R2L_len = L2R_len.clone()
    #     for i in range(lens.size(0)):
    #         prefix = torch.arange(lens[i])
    #         L2R_len[i][:lens[i]] = prefix
    #         R2L_len[i][:lens[i]] = torch.flip(prefix, [0])
    #     self_right = None
    # else:
    #     # get L2R and R2L lengths
    #     bsz, tgt_len = mask_inputs.size()
    #     L2R_len = torch.full((bsz, tgt_len), -100, dtype=torch.long)
    #     R2L_len = L2R_len.clone()
    #     self_right = torch.full((bsz, tgt_len), tgt_len, dtype=torch.long)
    #     for i in range(bsz):
    #         spans = batch_spans[i]
    #         total = sum(spans)
    #         prefix = 0
    #
    #         for span in spans:
    #             L2R_len[i][prefix: prefix + span] = prefix
    #             R2L_len[i][prefix: prefix + span] = total - prefix - span
    #             self_right[i][prefix: prefix + span] = prefix + span
    #             prefix += span
    #
    # batch = {
    #     'input_ids': inputs['input_ids'],
    #     'attention_mask': inputs['attention_mask'],
    #     'decoder_input_ids': {
    #         'L2R': L2R_decoder_inputs,
    #         'R2L': R2L_decoder_inputs,
    #         'BERT': mask_inputs,
    #         'L2R_len': L2R_len,
    #         'R2L_len': R2L_len,
    #         'self_right': self_right
    #     },
    #     'labels': {
    #         'L2R': L2R_labels,
    #         'R2L': R2L_labels,
    #         'BERT': L2R_labels
    #     }
    # }

    return data
