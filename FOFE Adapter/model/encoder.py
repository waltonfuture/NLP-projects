"""
Author: Da Ma
Date: 2022.01.06
"""
# [FIXME: INTRODUCE MORE ENCODERS HERE!]

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
    BertPooler,
    BertLayer,
    BertAttention,
    BertIntermediate,
    BertOutput,
)
from model.layer import FOFELayer,FOFELayerWindow
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.activations import ACT2FN

#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt


@dataclass
class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    kl_loss: Optional[torch.FloatTensor] = None

class BertOutputWithFOFE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, fofe_dict, fofe_layer, weights):
        if self.config.parallel == False:
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            input_tensor = hidden_states
            hidden_states = fofe_layer(hidden_states,
                                       fofe_dict['word_left_bound'],
                                       fofe_dict['word_right_bound'],
                                       fofe_dict['seg_bos_embedding'],
                                       fofe_dict['seg_eos_embedding'],
                                       )
            hidden_states = self.LayerNorm(hidden_states + input_tensor)

        else:
            weights = F.gumbel_softmax(weights,tau=self.config.gumbel_tau,hard=True,dim=-1) #??
            #weights = F.softmax(weights, dim=-1)
            hidden_states_fofe = 0
            for i , layer_module in enumerate(fofe_layer):
                hidden_states_fofe = hidden_states_fofe + weights[0][i] * layer_module(input_tensor,
                                            fofe_dict['word_left_bound'],
                                            fofe_dict['word_right_bound'],
                                            fofe_dict['seg_bos_embedding'],
                                            fofe_dict['seg_eos_embedding'],)
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + hidden_states_fofe + input_tensor)

        return hidden_states

class BertLayerWithFOFE(nn.Module):
    def __init__(self, layer_idx, config):
        super().__init__()

        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention

        fofe_layer_list = range(12)
        if layer_idx not in fofe_layer_list:
            windows_sizes = [0, 0, 0, 0, 0]
        else:
            windows_sizes = [0, 1, 3, 5, 7] #??

        self.fofe_layer = nn.ModuleList(
            [
                FOFELayerWindow(
                    config.alpha,
                    config.max_position_embeddings,
                    config.fofe_mode,
                    config.hidden_size,
                    config.fofe_dim,
                    config.is_conv1d,
                    window_size,
                ) for window_size in windows_sizes
            ]
        )

        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutputWithFOFE(config)

    def forward(
        self,
        hidden_states,
        word_left_bound=None,
        word_right_bound=None,
        seg_bos_embedding=None,
        seg_eos_embedding=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        window_weights=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple

            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        self.fofe_dict = {'word_left_bound':word_left_bound,
                          'word_right_bound':word_right_bound,
                          'seg_bos_embedding':seg_bos_embedding,
                          'seg_eos_embedding':seg_eos_embedding
                          }

        self.weights = torch.clone(window_weights)

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk_WithFOFE, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        outputs = (layer_output,) + outputs

        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk_WithFOFE(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output,self.fofe_dict,self.fofe_layer,self.weights)
        return layer_output


class BertEncoderWithEveryFOFE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fofe_layers = range(12)
        if config.use_fofe == True and config.every_layer == True:
            self.layer = nn.ModuleList(
                [BertLayerWithFOFE(i,config) if i in self.fofe_layers else BertLayer(config) for i in range(config.num_hidden_layers)]
            )
        else:
            self.layer = nn.ModuleList(
                [BertLayer(config) for _ in range(config.num_hidden_layers)]
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        word_left_bound=None,
        word_right_bound=None,
        seg_bos_embedding=None,
        seg_eos_embedding=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        window_weights=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                if isinstance(layer_module,BertLayerWithFOFE) == True:
                    layer_outputs = layer_module(
                        hidden_states,
                        word_left_bound,
                        word_right_bound,
                        seg_bos_embedding,
                        seg_eos_embedding,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                        window_weights[i],
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertModelWithFOFE(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)

        self.config = config

        self.embeddings = BertEmbeddings(config)

        if self.config.use_fofe:
            if not self.config.every_layer:
                self.fofe_layer = FOFELayer(
                    config.alpha,
                    config.max_position_embeddings,
                    config.fofe_mode,
                    config.hidden_size,
                )
                self.encoder = BertEncoder(config)
            else:
                # TODO
                self.encoder = BertEncoderWithEveryFOFE(config)
        else:
            self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.window_weights = nn.ParameterList(
            [
                nn.Parameter(torch.ones((1,5))/5.0,requires_grad=True) for _ in range(config.num_hidden_layers)
            ]
        )

        self.init_weights()

    def get_seg_bos_embedding(self):
        inputs = torch.LongTensor([self.config.seg_bos_id])
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        return self.embeddings.word_embeddings(inputs)

    def get_seg_eos_embedding(self):
        inputs = torch.LongTensor([self.config.seg_eos_id])
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        return self.embeddings.word_embeddings(inputs)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        word_left_bound=None,
        word_right_bound=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # NOTE: fofe layer
        if self.config.use_fofe and not self.config.every_layer:
            embedding_output = self.fofe_layer(
                embedding_output,
                word_left_bound,
                word_right_bound,
                self.get_seg_bos_embedding(),
                self.get_seg_eos_embedding(),
            )

        if self.config.use_fofe and self.config.every_layer:
            encoder_outputs = self.encoder(
                embedding_output,
                word_left_bound=word_left_bound,
                word_right_bound=word_right_bound,
                seg_bos_embedding=self.get_seg_bos_embedding(),
                seg_eos_embedding=self.get_seg_eos_embedding(),
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                window_weights=self.window_weights,
            )
        else:
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        weights = torch.cat([param for param in self.window_weights],axis=0)
        logits = F.gumbel_softmax(weights, tau=1, hard=False, dim=-1)
        index = torch.LongTensor(np.zeros((weights.shape))).to(weights.device)
        labels = torch.zeros_like(weights, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)

        #是否需要加参数限制一下
        kl_loss = F.kl_div(torch.log(logits),labels,reduction='mean') * self.config.kl_beta
        #??

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            kl_loss=kl_loss,
        )
