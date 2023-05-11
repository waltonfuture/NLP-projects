from transformers.models.bart.modeling_bart import BaseModelOutputWithPastAndCrossAttentions, \
    _expand_mask, BartDecoder, BartDecoderLayer
import torch
import torch.nn.functional as F
import random
import torch.nn as nn
from utils.custom_logger import getLogger
from model.config import MTConfig
from typing import Optional, Tuple

# logger
logger = getLogger(__name__)


class MTModelLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        assert padding_idx is not None, "`padding_idx` should not be None, but of type int"
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0, positions=None):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        if positions is None:
            if len(input_ids_shape)<2:
                print(f"input_ids_shape:{input_ids_shape}")
                seq_len = len(input_ids_shape[:1])
            else:
                bsz, seq_len = input_ids_shape[:2]
            #input_ids_shape may be inputs or inputs_shape
            seq_len = len(seq_len) if type(seq_len)==torch.Tensor and len(seq_len)>1 else seq_len

            positions = torch.arange(
                past_key_values_length + 1, past_key_values_length + seq_len + 1, dtype=torch.long, device=self.weight.device
            )
        return super().forward(positions)


class MTDecoderLayer(BartDecoderLayer):
    def additionalMemoryForward(self,
                                hidden_states: torch.Tensor,
                                attention_mask: Optional[torch.Tensor] = None,
                                encoder_hidden_states: Optional[torch.Tensor] = None,
                                encoder_attention_mask: Optional[torch.Tensor] = None,
                                layer_head_mask: Optional[torch.Tensor] = None,
                                encoder_layer_head_mask: Optional[torch.Tensor] = None,
                                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                                output_attentions: Optional[bool] = False,
                                use_cache: Optional[bool] = True,
                                memory=None):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            encoder_layer_head_mask (:obj:`torch.FloatTensor`): mask for encoder attention heads in a given layer of
                size `(config.encoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Attention to memory
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            key_value_states=memory,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MTDecoder(BartDecoder):
    def __init__(self, config: MTConfig, embed_tokens: Optional[nn.Embedding] = None):
        # judge decoder combinations
        self.decoders = config.decoders
        assert len(self.decoders) > 0, "there must be a decoder!!!"
        self.L2RMaskIdx = config.L2RMaskIdx
        self.R2LMaskIdx = config.R2LMaskIdx
        self.BertMaskIdx = config.BertMaskIdx
        self.clsIdx = config.bos_token_id
        self.use_bertmask = config.use_bertmask
        self.l2r_mask = config.l2r_mask
        self.r2l_mask = config.r2l_mask

        super().__init__(config, embed_tokens)

        self.embed_positions = MTModelLearnedPositionalEmbedding(config.max_position_embeddings,
                                                                 config.d_model,
                                                                 self.padding_idx)
        #usage: self.embed_positions(input_shape, past_key_values_length, pos)
        self.layers = nn.ModuleList([MTDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.init_weights()

    def oneStreamForward(self,
                         input_ids=None,
                         attention_mask=None,
                         encoder_hidden_states=None,
                         encoder_attention_mask=None,
                         head_mask=None,
                         encoder_head_mask=None,
                         past_key_values=None,
                         inputs_embeds=None,
                         use_cache=None,
                         output_attentions=None,
                         output_hidden_states=None,
                         return_dict=None,
                         memory=None,
                         pos=None):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            encoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        #print(input_shape)
        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        #print(f"input_ids_size:{input_ids.size()}") #(32,127)
        #print(f"min:{input_ids.min()}")
        #print(f"max:{input_ids.max()}")
        #print(f"embed_token_size:{self.embed_tokens}") #Embedding(50265, 768, padding_idx=1)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        #print(input_shape)
        positions = self.embed_positions(input_shape,past_key_values_length,pos)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            # why design memory as past-kv??
            curLayerMemory = torch.cat([memory[idx], hidden_states], dim=1) #here is different: memory contains previous decoder's hidden_state
            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    encoder_head_mask[idx] if encoder_head_mask is not None else None,
                    None,
                    curLayerMemory,
                )
            else:

                layer_outputs = decoder_layer.additionalMemoryForward(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    memory=curLayerMemory
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

    def mainStreamForward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pos=None
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            encoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        #print(input_shape)
        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length,pos) #pos is different: 把序列信息也放入embed层

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    encoder_head_mask[idx] if encoder_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    # encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

    def predictionStreamMaskForward(self, encoder_hidden_states=None, encoder_attention_mask=None, L2RMemory=None,
                                    R2LMemory=None, attention_mask=None, prediction='L2R',positions=None,
                                    inputShape=None, device=None, inputid=None):
        if prediction == 'L2R':
            input_ids = torch.full(inputShape, self.L2RMaskIdx, dtype=torch.long, device=device)
            memory = L2RMemory
            #positions = torch.arange(1, inputShape[1] + 1, 1, dtype=torch.long, device=device)#pro
        elif prediction == 'R2L':
            input_ids = torch.full(inputShape, self.R2LMaskIdx, dtype=torch.long, device=device)
            memory = R2LMemory
            #positions = torch.arange(inputShape[1] - 1, -1, -1, dtype=torch.long, device=device)#pro
        else:
            if self.use_bertmask:
                input_ids = torch.full(inputShape, self.BertMaskIdx, dtype=torch.long, device=device)
            else:
                input_ids = inputid
                input_ids.to(device, dtype=torch.long)

            #positions = None #pro
            # NOTE: the first token is cls
            input_ids[:, 0] = self.clsIdx
            memory = []
            for idx, L2RLayerMemory in enumerate(L2RMemory):
                memory.append(torch.cat([L2RLayerMemory, R2LMemory[idx]], dim=1))
            #print(f"input_id_size:{input_ids.shape}")
            #print(f"attention_mask_size:{attention_mask.shape}")
        attention_mask = torch.full_like(attention_mask, float('-inf'), dtype=torch.float).masked_fill(attention_mask,0)
        return self.oneStreamForward(input_ids=input_ids, attention_mask=attention_mask,
                                     encoder_hidden_states=encoder_hidden_states,
                                     encoder_attention_mask=encoder_attention_mask, memory=memory, pos=positions)

    def forward(self, input_ids=None,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                head_mask=None,
                encoder_head_mask=None,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        outputs = {}

        # main stream
        L2RInputIds = input_ids['L2R'][:,:-1]
        #L2RInputIds = input_ids[:, :-1] # torch.LongTensor of shape (batch_size, target_sequence_length)
        R2LInputIds = input_ids['R2L'][:,:-1]
        #R2LInputIds = torch.flip(input_ids[:, 1:], dims=[1]) # 将target_sequence_length维度的向量翻转
        # prepare R2L decoder padding mask
        #decoderPaddingMask = torch.eq(R2LInputIds, self.padding_idx) #self.padding_idx: The id of the _padding_ token
        #decoderPaddingMask = ~decoderPaddingMask #与padding id相同的为false，不同则为true
        L2R_positions = input_ids['L2R_len'][:,:-1].to(L2RInputIds.device)
        Bert_positions = input_ids['L2R_len'].to(L2RInputIds.device)
        L2RMainStreamMemory = self.mainStreamForward(input_ids=L2RInputIds, encoder_hidden_states=encoder_hidden_states,
                                                     encoder_attention_mask=encoder_attention_mask,
                                                     attention_mask=None,pos=L2R_positions,
                                                     output_hidden_states=True).hidden_states
        R2L_positions = input_ids['R2L_len'][:,:-1].to(L2RInputIds.device)
        #positions = torch.arange(R2LInputIds.size(1), 0, -1, dtype=torch.long, device=L2RInputIds.device) # [4,3,2,1,0]
        R2LMainStreamMemory = self.mainStreamForward(input_ids=R2LInputIds, encoder_hidden_states=encoder_hidden_states,
                                                     encoder_attention_mask=encoder_attention_mask,
                                                     attention_mask=None, pos=R2L_positions,
                                                     output_hidden_states=True).hidden_states

        # PredictionStream
        # prepare attention mask [bsz, 1, tgt_seq_len, src_seq_len]
        inputShape = L2RInputIds.size()
        bsz, seq_len = inputShape
        # [bsz, seq_len, seq_len]
        diagMask = torch.eye(seq_len, dtype=torch.bool, device=L2RInputIds.device).repeat(bsz, 1, 1) # [bsz, seq_len, seq_len] 单位矩阵
        mask_cond = torch.arange(seq_len, device=L2RInputIds.device)
        causalMask = (mask_cond <= mask_cond.view(-1, 1)).repeat(bsz, 1, 1) #bsz个下三角矩阵(下三角和对角线为true)
        if "L2R" in self.decoders:
            # L2R
            if self.l2r_mask:
                L2RPredictionStreamAttentionMask = torch.cat([causalMask, causalMask], dim=-1).unsqueeze(
                    1)  # [bsz,1,seq_len,2*seq_len] 下三角拼接对角矩阵
            else:
                L2RPredictionStreamAttentionMask = torch.cat([causalMask, diagMask], dim=-1).unsqueeze(
                    1)  # [bsz,1,seq_len,2*seq_len] 下三角拼接对角矩阵

            outputs['L2R'] = self.predictionStreamMaskForward(encoder_hidden_states=encoder_hidden_states,
                                                              encoder_attention_mask=encoder_attention_mask,
                                                              L2RMemory=L2RMainStreamMemory,positions=L2R_positions,
                                                              attention_mask=L2RPredictionStreamAttentionMask,
                                                              prediction='L2R', inputShape=inputShape,
                                                              device=L2RInputIds.device)

        #offsets = torch.sum(~decoderPaddingMask, dim=1, keepdim=True).unsqueeze(-1)
        #baseCausalNum = mask_cond.repeat(bsz, seq_len, 1)
        #paddingCausalMask = (baseCausalNum >= offsets)  # [bsz,seq_len,seq_len] padding部分为true
        if "R2L" in self.decoders:
            # R2L

            #R2LCausalMask = torch.logical_and(causalMask, paddingCausalMask)
            #R2LPredictionStreamAttentionMask = torch.cat([R2LCausalMask, diagMask], dim=-1).unsqueeze(1)
            if self.r2l_mask:
                R2LPredictionStreamAttentionMask = torch.cat([causalMask, causalMask], dim=-1).unsqueeze(1)
            else:
                R2LPredictionStreamAttentionMask = torch.cat([causalMask, diagMask], dim=-1).unsqueeze(1)
            outputs['R2L'] = self.predictionStreamMaskForward(encoder_hidden_states=encoder_hidden_states,
                                                              encoder_attention_mask=encoder_attention_mask,
                                                              R2LMemory=R2LMainStreamMemory,positions=R2L_positions,
                                                              attention_mask=R2LPredictionStreamAttentionMask,
                                                              prediction='R2L', inputShape=inputShape,
                                                              device=L2RInputIds.device)

        if "BERT" in self.decoders:
            # BERT
            # CLS && causal
            mask_cond = torch.arange(seq_len + 1, device=L2RInputIds.device)
            L2RCausalMask = (mask_cond <= mask_cond.view(-1, 1)).repeat(bsz, 1, 1)
            #nodiagCausalMask = (mask_cond < mask_cond.view(-1, 1)).repeat(bsz, 1, 1) #bsz个下三角为true（不包括对角线）的矩阵

            # R2L causal
            #upperBound = torch.arange(seq_len - 1, -1, -1, device=L2RInputIds.device).view(-1, 1)
            #causalMask = (mask_cond <= upperBound).repeat(bsz, 1, 1)  # bsz个左上三角矩阵（左上为true）
            #causalMask[:, 0] = 0

            # remove padding
            decoderPaddingMask = torch.eq(input_ids['BERT'],self.padding_idx)  # self.padding_idx: The id of the _padding_ token
            decoderPaddingMask = ~decoderPaddingMask
            #offsets = torch.sum(decoderPaddingMask, dim=1, keepdim=True).unsqueeze(-1)
            pad_num = torch.sum(~decoderPaddingMask, dim=1, keepdim=True).view(bsz,).to(L2RInputIds.device)
            #baseCausalNum = mask_cond.repeat(bsz, seq_len + 1, 1)
            upperBound = torch.arange(seq_len, -1, -1, device=L2RInputIds.device).view(-1, 1)
            causalMask = (mask_cond <= upperBound).repeat(bsz, 1, 1).to(L2RInputIds.device)
            # causalMask[:, 0] = 0
            R2LCausalMask = torch.clone(causalMask)
            indices = torch.arange(seq_len + 1, device=L2RInputIds.device).unsqueeze(0).repeat(bsz, 1)
            shifts = pad_num.unsqueeze(1)
            shifted_indices = (indices + shifts) % (seq_len + 1)
            R2LCausalMask = R2LCausalMask.gather(1, shifted_indices.view(bsz, seq_len + 1, 1).repeat(1, 1, seq_len + 1))
            #paddingCausalMask = (baseCausalNum < offsets)
            #R2LCausalMask = torch.logical_and(causalMask, paddingCausalMask)
            '''decoderPaddingMask = ~decoderPaddingMask
            offsets = torch.sum(decoderPaddingMask, dim=1, keepdim=True).unsqueeze(-1)
            baseCausalNum = mask_cond.repeat(bsz, seq_len + 1, 1)
            paddingCausalMask = (baseCausalNum < offsets)
            R2LCausalMask = torch.logical_and(~nodiagCausalMask, paddingCausalMask)'''
            diagMask = torch.eye(seq_len + 1, dtype=torch.bool, device=L2RInputIds.device).repeat(bsz, 1, 1)

            BertMask = torch.cat([L2RCausalMask[:,:,1:], R2LCausalMask[:,:,1:], diagMask], dim=-1).unsqueeze(1) #区分是否加入diagmask

            inputShape = input_ids['BERT'].size()
            outputs['BERT'] = self.predictionStreamMaskForward(encoder_hidden_states=encoder_hidden_states,
                                                               encoder_attention_mask=encoder_attention_mask,
                                                               L2RMemory=L2RMainStreamMemory,
                                                               R2LMemory=R2LMainStreamMemory, attention_mask=BertMask,
                                                               prediction='BERT', positions=Bert_positions,
                                                               inputShape=inputShape, device=L2RInputIds.device, inputid=input_ids['BERT'])

        return outputs

