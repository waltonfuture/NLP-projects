from model.config import MTConfig
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import BartPretrainedModel, BartEncoder, BartClassificationHead, BartModel
from model.decoder import MTDecoder, MTModelLearnedPositionalEmbedding
import torch.nn as nn
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.bert.modeling_bert import BertForQuestionAnswering
from typing import Optional
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqSequenceClassifierOutput, Seq2SeqQuestionAnsweringModelOutput, QuestionAnsweringModelOutput
from utils.custom_logger import getLogger
import logging

logger = getLogger(__name__, logging.DEBUG)
from transformers import BartForQuestionAnswering

class MTEncoder(BartEncoder):
    def __init__(self, config: MTConfig, embed_tokens: Optional[nn.Embedding] = None):

        super().__init__(config, embed_tokens)

        self.embed_positions = MTModelLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )

        self.init_weights()


class MTModel(BartPretrainedModel):
    def __init__(self, config: MTConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = MTEncoder(config, self.shared)
        self.decoder = MTDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                cross_attn_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                output_attentions=None,
                use_cache=None,
                output_hidden_states=None):
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids, # ??
            #input_ids=input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        outputs = {
            "encoder": encoder_outputs,
            "decoder": decoder_outputs
        }

        return outputs


class MTForSequenceClassification(BartPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: MTConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = MTModel(config)
        self.classification_head1 = BartClassificationHead(
            2 * config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        #self.classification_head2 = BartClassificationHead(
            #3 * config.d_model,
            #config.d_model,
            #config.num_labels,
            #config.classifier_dropout,
        #)
        #self.model._init_weights(self.classification_head.dense)
        #self.model._init_weights(self.classification_head.out_proj)
        self.model._init_weights(self.classification_head1.dense)
        self.model._init_weights(self.classification_head1.out_proj)
        #self.model._init_weights(self.classification_head2.dense)
        #self.model._init_weights(self.classification_head2.out_proj)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) :
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        outputs["logits"] = {}
        outputs["final_logit"] = 0
        real_part, cls_part, sep_part = [], [], []
        specialtoken_mask = input_ids[:, 1:].eq(self.config.eos_token_id)
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        for decoder_type in outputs["decoder"]:
            if decoder_type == 'L2R':
                l2r_sep = outputs["decoder"][decoder_type].last_hidden_state
                l2r_sep = l2r_sep[specialtoken_mask, :].view(l2r_sep.size(0), -1, l2r_sep.size(-1))[:, -1, :]
                sep_part.append(l2r_sep)
            elif decoder_type == 'R2L':
                r2l_cls = outputs["decoder"][decoder_type].last_hidden_state
                r2l_cls = r2l_cls[specialtoken_mask, :].view(r2l_cls.size(0), -1, r2l_cls.size(-1))[:, -1, :]
                cls_part.append(r2l_cls)
            else:
                cls_part.append(outputs["decoder"][decoder_type].last_hidden_state[:, 0, :])
                bert_sep = outputs["decoder"][decoder_type].last_hidden_state
                bert_sep = bert_sep[eos_mask, :].view(bert_sep.size(0), -1, bert_sep.size(-1))[:, -1, :]
                sep_part.append(bert_sep)

        cls_part_hidden = torch.cat(tuple([i for i in cls_part]), dim=-1)
        sep_part_hidden = torch.cat(tuple([i for i in sep_part]), dim=-1) # [bsz,2*hidden_dim]
        cls_part_hidden = cls_part_hidden.view(cls_part_hidden.size(0),-1,cls_part_hidden.size(1))
        sep_part_hidden = sep_part_hidden.view(sep_part_hidden.size(0),-1,sep_part_hidden.size(1)) # [bsz, 1, 2*hidden_dim]
        #eos_mask = input_ids[:, 1:].eq(self.config.eos_token_id) # [bsz,127]
        #pooled_output = pooled_output[eos_mask, :]

        #pooled_output = pooled_output.view(pooled_output.size(0), -1, pooled_output.size(-1))[:, -1, :]
        #pooled_output = pooled_output[eos_mask, :].view(pooled_output.size(0), -1,pooled_output.size(-1))[:, -1, :]
        cls_logit = self.classification_head1(cls_part_hidden)
        sep_logit = self.classification_head1(sep_part_hidden)
        #real_part_logit = self.classification_head2(real_part_hidden)
        logit = torch.cat((cls_logit, sep_logit),dim=1)
        logit = torch.mean(logit, dim=1)
        logit = torch.squeeze(logit, dim=1)
        outputs["final_logit"] = logit


        return outputs


class MTForSequenceClassificationBert(BartPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: MTConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = MTModel(config)
        '''self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )'''

        self.classification_head1 = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        #self.model._init_weights(self.classification_head.dense)
        #self.model._init_weights(self.classification_head.out_proj)
        self.model._init_weights(self.classification_head1.dense)
        self.model._init_weights(self.classification_head1.out_proj)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) :
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        outputs["logits"] = {}
        outputs["final_logit"] = 0

        pooled_output = outputs["decoder"]["BERT"].last_hidden_state
        eos_mask = input_ids.eq(self.config.eos_token_id)
        bos_mask = input_ids.eq(self.config.bos_token_id)
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        #pooled_output = pooled_output[eos_mask, :]
        #pooled_output = pooled_output.view(pooled_output.size(0), -1, pooled_output.size(-1))[:, -1, :]
        #pooled_output = pooled_output[eos_mask, :].view(pooled_output.size(0), -1,pooled_output.size(-1))[:, -1, :]
        eos_output = pooled_output[eos_mask, :].view(pooled_output.size(0), -1,pooled_output.size(-1))
        bos_output = pooled_output[bos_mask, :].view(pooled_output.size(0), -1,pooled_output.size(-1))[:, -1, :].view(pooled_output.size(0), -1,pooled_output.size(-1))
        pooled_output = torch.cat((eos_output, bos_output),dim=1)
        pooled_output = torch.mean(pooled_output,dim=1)
        pooled_output = torch.squeeze(pooled_output,dim=1)
        # classifier of course has to be 3 * hidden_dim, because we concat 3 layers
        logit = self.classification_head1(pooled_output)
        outputs["final_logit"] = logit

        '''for decoder_type in outputs["decoder"]:
            hidden_states = outputs["decoder"][decoder_type].last_hidden_state  # last hidden state
            eos_mask = input_ids[:, :-1].eq(self.config.eos_token_id)
            sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                                      hidden_states.size(-1))[:, -1, :]
            logits = self.classification_head(sentence_representation)
            outputs["logits"][decoder_type] = logits'''


        return outputs


class MTForSequenceClassificationBertAllTokens(BartPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: MTConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = MTModel(config)
        '''self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )'''

        self.classification_head1 = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        #self.model._init_weights(self.classification_head.dense)
        #self.model._init_weights(self.classification_head.out_proj)
        self.model._init_weights(self.classification_head1.dense)
        self.model._init_weights(self.classification_head1.out_proj)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) :
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        outputs["logits"] = {}
        outputs["final_logit"] = 0

        pooled_output = outputs["decoder"]["BERT"].last_hidden_state
        pad_mask = input_ids.eq(self.config.pad_token_id)
        eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")

        # classifier of course has to be 3 * hidden_dim, because we concat 3 layers
        logits = self.classification_head1(pooled_output)

        logits[pad_mask, :] = 0
        logit = torch.sum(logits, dim=1) / torch.sum(logits != 0, dim=1)
        outputs["final_logit"] = logit
        '''for decoder_type in outputs["decoder"]:
            hidden_states = outputs["decoder"][decoder_type].last_hidden_state  # last hidden state
            eos_mask = input_ids[:, :-1].eq(self.config.eos_token_id)
            sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                                      hidden_states.size(-1))[:, -1, :]
            logits = self.classification_head(sentence_representation)
            outputs["logits"][decoder_type] = logits'''


        return outputs


class MTForSequenceClassification1(BartPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: MTConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = MTModel(config)
        self.classification_head1 = BartClassificationHead(
            2 * config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        self.model._init_weights(self.classification_head1.dense)
        self.model._init_weights(self.classification_head1.out_proj)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) :
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        outputs["logits"] = {}
        outputs["final_logit"] = 0
        cls_part, sep_part = [], []
        specialtoken_mask = input_ids[:, 1:].eq(self.config.eos_token_id)
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")

        for decoder_type in outputs["decoder"]:
            if decoder_type == 'L2R':
                l2r_sep = outputs["decoder"][decoder_type].last_hidden_state
                l2r_sep = l2r_sep[specialtoken_mask,:].view(l2r_sep.size(0), -1, l2r_sep.size(-1))[:, -1, :]
                sep_part.append(l2r_sep)
            elif decoder_type == 'R2L':
                r2l_cls = outputs["decoder"][decoder_type].last_hidden_state
                r2l_cls = r2l_cls[specialtoken_mask,:].view(r2l_cls.size(0), -1, r2l_cls.size(-1))[:, -1, :]
                cls_part.append(r2l_cls)
            else:
                cls_part.append(outputs["decoder"][decoder_type].last_hidden_state[:, 0, :])
                bert_sep = outputs["decoder"][decoder_type].last_hidden_state
                bert_sep = bert_sep[eos_mask,:].view(bert_sep.size(0), -1, bert_sep.size(-1))[:, -1, :]
                sep_part.append(bert_sep)

        cls_part_hidden = torch.cat(tuple([i for i in cls_part]), dim=-1)
        sep_part_hidden = torch.cat(tuple([i for i in sep_part]), dim=-1) # [bsz,2*hidden_dim]
        cls_part_hidden = cls_part_hidden.view(cls_part_hidden.size(0),-1,cls_part_hidden.size(1))
        sep_part_hidden = sep_part_hidden.view(sep_part_hidden.size(0),-1,sep_part_hidden.size(1)) # [bsz, 1, 2*hidden_dim]
        pooled = torch.cat((cls_part_hidden, sep_part_hidden), dim=1) # [bsz, 2, 2*hidden_dim]
        pooled = torch.mean(pooled, dim=1) # [bsz, 1, 2*hidden_dim]
        pooled = torch.squeeze(pooled, dim=1) # [bsz, 2*hidden_dim]
        logit = self.classification_head1(pooled)

        outputs["final_logit"] = logit


        return outputs


class MTForSequenceClassification_all_token_average(BartPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: MTConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = MTModel(config)
        self.classification_head1 = BartClassificationHead(
            3 * config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        self.model._init_weights(self.classification_head1.dense)
        self.model._init_weights(self.classification_head1.out_proj)
        #self.model._init_weights(self.classification_head2.dense)
        #self.model._init_weights(self.classification_head2.out_proj)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) :
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        outputs["logits"] = {}
        outputs["final_logit"] = 0
        real_part = []

        decoder_output = input_ids[:, 1:]
        lens = decoder_output.ne(self.config.pad_token_id).sum(dim=1)


        pad_mask = input_ids[:, 1:].eq(self.config.pad_token_id)
        pad_mask_bert = input_ids.eq(self.config.pad_token_id)

        for decoder_type in outputs["decoder"]:
            if decoder_type == 'BERT':
                bert_hidden = outputs["decoder"][decoder_type].last_hidden_state
                bert_hidden[pad_mask_bert, :] = 0
                bert_hidden = torch.sum(bert_hidden,dim=1) / torch.sum(bert_hidden != 0,dim=1) #[bsz,hidden_size]
                real_part.append(bert_hidden)
            elif decoder_type == 'L2R':
                LR_hidden = outputs["decoder"][decoder_type].last_hidden_state
                LR_hidden[pad_mask, :] = 0
                LR_hidden = torch.sum(LR_hidden, dim=1) / torch.sum(LR_hidden != 0, dim=1)  # [bsz,hidden_size]
                real_part.append(LR_hidden)
            else:
                LR_hidden = outputs["decoder"][decoder_type].last_hidden_state
                for i in range(lens.size(0)):
                    LR_hidden[i, :lens[i], :] = LR_hidden[i, :lens[i], :].flip(dims=[0])
                LR_hidden[pad_mask, :] = 0
                LR_hidden = torch.sum(LR_hidden, dim=1) / torch.sum(LR_hidden != 0, dim=1)  # [bsz,hidden_size]
                real_part.append(LR_hidden)


        real_part_hidden = torch.cat(tuple([i for i in real_part]), dim=-1)# [bsz,3*hidden_dim]

        logit = self.classification_head1(real_part_hidden)
        outputs["final_logit"] = logit

        return outputs

class MTForSequenceClassification_all_tokens(BartPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: MTConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = MTModel(config)
        self.classification_head1 = BartClassificationHead(
            3 * config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        self.classification_head = BartClassificationHead(
            2 * config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        self.model._init_weights(self.classification_head1.dense)
        self.model._init_weights(self.classification_head1.out_proj)
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) :
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        outputs["logits"] = {}
        outputs["final_logit"] = 0
        bert_output = outputs["decoder"]["BERT"].last_hidden_state
        l2r_output = outputs["decoder"]["L2R"].last_hidden_state
        r2l_output = outputs["decoder"]["R2L"].last_hidden_state

        decoder_output = input_ids[:, 1:]
        lens = decoder_output.ne(self.config.pad_token_id).sum(dim=1)
        for i in range(lens.size(0)):
            r2l_output[i, :lens[i], :] = r2l_output[i, :lens[i], :].flip(dims=[0])
        real_part, cls_part, sep_part = [], [], []
        specialtoken_mask = torch.zeros(decoder_output.shape, dtype=torch.bool).to(input_ids.device)
        for i in range(decoder_output.size(0)):
            specialtoken_mask[i, torch.where(decoder_output[i] == self.config.eos_token_id)[0][-1]] = True

        bert_sep_mask = torch.zeros(input_ids.shape, dtype=torch.bool).to(input_ids.device)
        for i in range(input_ids.size(0)):
            bert_sep_mask[i, torch.where(input_ids[i] == self.config.eos_token_id)[0][-1]] = True

        bert_cls_mask = input_ids.eq(self.config.bos_token_id)
        bert_token_mask = ~torch.bitwise_or(bert_cls_mask, bert_sep_mask)
        for decoder_type in outputs["decoder"]:
            if decoder_type == 'L2R':
                sep_part.append(l2r_output[specialtoken_mask, :].view(l2r_output.size(0), -1, l2r_output.size(-1)))
                real_part.append(l2r_output[~specialtoken_mask, :].view(l2r_output.size(0), -1, l2r_output.size(-1)))
            elif decoder_type == 'R2L':
                cls_part.append(r2l_output[:, 0, :].view(r2l_output.size(0), -1, r2l_output.size(-1)))
                real_part.append(r2l_output[:, 1:, :].view(r2l_output.size(0), -1, r2l_output.size(-1)))
            else:
                cls_part.append(bert_output[bert_cls_mask, :].view(bert_output.size(0), -1, bert_output.size(-1)))
                sep_part.append(bert_output[bert_sep_mask, :].view(bert_output.size(0), -1, bert_output.size(-1)))
                real_part.append(bert_output[bert_token_mask, :].view(bert_output.size(0), -1, bert_output.size(-1)))

        real_part_hidden = torch.cat(tuple([i for i in real_part]), dim=-1)  # [bsz,seq_length - 2, 3*hidden_dim]
        cls_part_hidden = torch.cat(tuple([i for i in cls_part]), dim=-1)  # [bsz, 1, 2*hidden_dim]
        sep_part_hidden = torch.cat(tuple([i for i in sep_part]), dim=-1)  # [bsz, 1, 2*hidden_dim]
        logits_real = self.classification_head1(real_part_hidden)
        logits_cls = self.classification_head(cls_part_hidden)
        logits_sep = self.classification_head(sep_part_hidden)
        logits = torch.cat((logits_cls, logits_real, logits_sep), dim=1)
        pad_mask_bert = input_ids.eq(self.config.pad_token_id)
        logits[pad_mask_bert, :] = 0
        logit = torch.sum(logits, dim=1) / torch.sum(logits != 0, dim=1)
        outputs["final_logit"] = logit

        return outputs


class MTForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: MTConfig):
        super().__init__(config)

        self.model = MTModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.model.shared.weight = self.lm_head.weight
        self.init_weights()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
        Returns:
        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        outputs["logits"] = {}

        for decoder_type in outputs["decoder"]:
            outputs["logits"][decoder_type] = self.lm_head(
                outputs["decoder"][decoder_type].last_hidden_state) + self.final_logits_bias

        return outputs


class MTForQuestionAnswering(BartPretrainedModel):
    def __init__(self, config:MTConfig):
        super().__init__(config)
        config.num_labels = 2
        self.num_labels = config.num_labels

        self.model = MTModel(config)
        self.qa_outputs = nn.Linear(3*config.hidden_size, config.num_labels)
        self.qa_outputs1 = nn.Linear(2*config.hidden_size, config.num_labels)
        self.model._init_weights(self.qa_outputs)
        self.model._init_weights(self.qa_outputs1)


    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs= None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        bert_output = outputs["decoder"]["BERT"].last_hidden_state
        l2r_output = outputs["decoder"]["L2R"].last_hidden_state
        r2l_output = outputs["decoder"]["R2L"].last_hidden_state
        decoder_output = input_ids[:, 1:]
        lens = decoder_output.ne(self.config.pad_token_id).sum(dim=1)
        for i in range(lens.size(0)):
            r2l_output[i, :lens[i], :] = r2l_output[i, :lens[i], :].flip(dims=[0])
        real_part, cls_part, sep_part = [], [], []
        specialtoken_mask = torch.zeros(decoder_output.shape, dtype=torch.bool).to(input_ids.device)
        for i in range(decoder_output.size(0)):
            specialtoken_mask[i, torch.where(decoder_output[i] == self.config.eos_token_id)[0][-1]] = True

        bert_sep_mask = torch.zeros(input_ids.shape, dtype=torch.bool).to(input_ids.device)
        for i in range(input_ids.size(0)):
            bert_sep_mask[i, torch.where(input_ids[i] == self.config.eos_token_id)[0][-1]] = True

        bert_cls_mask = input_ids.eq(self.config.bos_token_id)
        bert_token_mask = ~torch.bitwise_or(bert_cls_mask, bert_sep_mask)
        for decoder_type in outputs["decoder"]:
            if decoder_type == 'L2R':
                sep_part.append(l2r_output[specialtoken_mask, :].view(l2r_output.size(0), -1, l2r_output.size(-1)))
                real_part.append(l2r_output[~specialtoken_mask, :].view(l2r_output.size(0), -1, l2r_output.size(-1)))
            elif decoder_type == 'R2L':
                cls_part.append(r2l_output[:, 0, :].view(r2l_output.size(0), -1, r2l_output.size(-1)))
                real_part.append(r2l_output[:, 1:, :].view(r2l_output.size(0), -1, r2l_output.size(-1)))
            else:
                cls_part.append(bert_output[bert_cls_mask, :].view(bert_output.size(0), -1, bert_output.size(-1)))
                sep_part.append(bert_output[bert_sep_mask, :].view(bert_output.size(0), -1, bert_output.size(-1)))
                real_part.append(bert_output[bert_token_mask, :].view(bert_output.size(0), -1, bert_output.size(-1)))

        real_part_hidden = torch.cat(tuple([i for i in real_part]), dim=-1)  # [bsz,seq_length - 2, 3*hidden_dim]
        cls_part_hidden = torch.cat(tuple([i for i in cls_part]), dim=-1)  # [bsz, 1, 2*hidden_dim]
        sep_part_hidden = torch.cat(tuple([i for i in sep_part]), dim=-1)  # [bsz, 1, 2*hidden_dim]
        logits_real = self.qa_outputs(real_part_hidden)
        logits_cls = self.qa_outputs1(cls_part_hidden)
        logits_sep = self.qa_outputs1(sep_part_hidden)
        logits = torch.cat((logits_cls, logits_real, logits_sep), dim=1)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            #print(f"start_positions:{start_positions.shape}")
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=None,
            attentions=None,
        )


class MTForQuestionAnsweringBert(BartPretrainedModel):
    def __init__(self, config:MTConfig):
        super().__init__(config)
        config.num_labels = 2
        self.num_labels = config.num_labels

        self.model = MTModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.model._init_weights(self.qa_outputs)


    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs= None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs["decoder"]["BERT"].last_hidden_state
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=None,
            attentions=None,
        )


class BartForQuestionAnswering1(BartPretrainedModel):
    def __init__(self, config):
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.model = BartModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.model._init_weights(self.qa_outputs)


    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if start_positions is not None and end_positions is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.decoder_hidden_states,
            attentions=outputs.decoder_attentions,
        )

        #Bert
    '''
    if not return_dict:
        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output

    return QuestionAnsweringModelOutput(
        loss=total_loss,
        start_logits=start_logits,
        end_logits=end_logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    '''


class BartForSequenceClassification1(BartPretrainedModel):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # last hidden state
        pad_mask = input_ids.eq(self.config.pad_token_id)
        eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]
        logits = self.classification_head(hidden_states)
        logits[pad_mask, :] = 0
        logits = torch.sum(logits, dim=1) / torch.sum(logits != 0, dim=1)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )