"""
Author: Da Ma
Date: 2022.01.05
Description: the pre-training model file. Refer to https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/bert/modeling_bert.py
"""
from dataclasses import dataclass
from typing import Optional, Tuple
from model.encoder import BertModelWithFOFE
from model.layer import CDEETPLinker


import torch
import torch.nn as nn
from transformers import BertPreTrainedModel
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.models.bert.modeling_bert import BertPreTrainingHeads

#from cblue.models.model import TwoHeadTokenClassifierOutput
from model.config import PlmConfig


@dataclass
class PlmModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    kl_loss: Optional[torch.FloatTensor] = None

    # [FIXME: INTRODUCE MORE PLM Model OUTPUTS HERE]


@dataclass
class PlmForPreTrainingOutput:
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None


class PlmPreTrainedModel(BertPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PlmConfig
    base_model_prefix = "plm"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _reorder_cache(self, past, beam_idx):
        pass


class PlmModel(PlmPreTrainedModel):
    """
    NOTE:
        1. [2022.01.06]
            Support MLM solely.
    """

    def __init__(self, config: PlmConfig, add_pooling_layer=True):
        super().__init__(config)

        # self.bertEncoder = BertModel(config)
        self.bertEncoder = BertModelWithFOFE(config, add_pooling_layer)

        # [FIXME: INTRODUCE MORE MODULES HERE!]

        self.init_weights()

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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        """
        Include two parts:
            1. dataflow
            2. output
        """
        # =================== DATAFLOW =================== #
        bertEncoderOutput = self.bertEncoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_left_bound=word_left_bound,
            word_right_bound=word_right_bound,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # [FIXME: INTRODUCE MORE FORWARD DATAFLOW HERE!]

        # =================== OUTPUT =================== #
        # add required output to the [PlmOutput] class above and
        # extract corresponding results from the dataflow.

        # [FIXME: INTRODUCE MORE OUTPUTS HERE!]
        if not return_dict:
            return bertEncoderOutput
        else:
            return PlmModelOutput(
                last_hidden_state=bertEncoderOutput.last_hidden_state,
                pooler_output=bertEncoderOutput.pooler_output,
                hidden_states=bertEncoderOutput.hidden_states,
                attentions=bertEncoderOutput.attentions,
                cross_attentions=bertEncoderOutput.cross_attentions,
                kl_loss=bertEncoderOutput.kl_loss,
            )

    # NOTE: These two following methods are important! They are related
    # to the embedding tying.
    def get_input_embeddings(self):
        return self.bertEncoder.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.bertEncoder.embeddings.word_embeddings = value


class PlmForPreTraining(PlmPreTrainedModel):
    def __init__(self, config: PlmConfig):
        super().__init__(config)

        self.plm = PlmModel(config)

        # ================== SOME OUTPUT LAYERS ================== #
        self.mlm = BertPreTrainingHeads(config)
        self.clm = ...

        # [FIXME: INTRODUCE OUTPUT LAYERS FOR DIFFERENT TASK HERE!]

        self.init_weights()

    # NOTE: These two following methods are important! They are related
    # to the embedding tying.
    def get_output_embeddings(self):
        return self.mlm.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.mlm.predictions.decoder = new_embeddings

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
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # =================== DATAFLOW =================== #
        plmOutputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_left_bound=word_left_bound,
            word_right_bound=word_right_bound,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        sequence_output, pooled_output = plmOutputs[:2]
        prediction_scores, seq_relationship_score = self.mlm(
            sequence_output, pooled_output
        )

        # [FIXME: INTRODUCE OUTPUT LAYER DATA FLOW HERE!]

        # =================== OUTPUT =================== #
        # [FIXME: INTRODUCE MORE OUTPUT LOGITS HERE!]
        if not return_dict:
            return prediction_scores, seq_relationship_score
        else:
            return PlmForPreTrainingOutput(
                prediction_logits=prediction_scores,
                seq_relationship_logits=seq_relationship_score,
            )


class PlmForSequenceClassification(PlmPreTrainedModel):
    def __init__(self, config: PlmConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.plm = PlmModel(config)

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

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
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # =================== DATAFLOW =================== #
        plmOutputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_left_bound=word_left_bound,
            word_right_bound=word_right_bound,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = plmOutputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # ================== LOSS ================== #
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
                #loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

            #if self.config.multiwindow == True:
            loss += plmOutputs.kl_loss

        if not return_dict:
            output = (logits,) + plmOutputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=plmOutputs.hidden_states,
            attentions=plmOutputs.attentions,
        )


class PlmForTokenClassification(PlmPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: PlmConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.plm = PlmModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

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
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        plmOutputs = self.plm(
            input_ids,
            attention_mask=attention_mask,
            word_left_bound=word_left_bound,
            word_right_bound=word_right_bound,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = plmOutputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # if self.config.multiwindow == True:
            loss += plmOutputs.kl_loss

        if not return_dict:
            output = (logits,) + plmOutputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=plmOutputs[3] if output_hidden_states else None,
            attentions=plmOutputs[4] if output_attentions else None,
        )


# class PlmForCDEE(PlmPreTrainedModel):
#     _keys_to_ignore_on_load_unexpected = [r"pooler"]

#     def __init__(self, config: PlmConfig):
#         super().__init__(config)

#         self.plm = PlmModel(config)
#         self.dropout1 = torch.nn.Dropout(config.hidden_dropout_prob)
#         self.dropout2 = torch.nn.Dropout(config.hidden_dropout_prob)
#         self.dropout3 = torch.nn.Dropout(config.hidden_dropout_prob)
#         self.tendency_classifier = torch.nn.Linear(config.hidden_size, 3)
#         self.character_classifier = torch.nn.Linear(config.hidden_size, 3)
#         self.anatomy_classifier = torch.nn.Linear(config.hidden_size, 3)
#         self.num_labels = 3

#         self.init_weights()

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         word_left_bound=None,
#         word_right_bound=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         tendency_labels=None,
#         character_labels=None,
#         anatomy_labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         return_dict = (
#             return_dict if return_dict is not None else self.config.use_return_dict
#         )

#         plmOutputs = self.plm(
#             input_ids,
#             attention_mask=attention_mask,
#             word_left_bound=word_left_bound,
#             word_right_bound=word_right_bound,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=False,
#         )

#         pooled_output = plmOutputs[1]
#         tendency_logits = self.tendency_classifier(self.dropout1(pooled_output))

#         sequence_output = plmOutputs[0]

#         character_logits = self.character_classifier(self.dropout2(sequence_output))
#         anatomy_logits = self.anatomy_classifier(self.dropout3(sequence_output))

#         loss = None
#         if tendency_labels is not None:
#             loss_fct = nn.CrossEntropyLoss()

#             tendency_loss = loss_fct(tendency_logits, tendency_labels)

#             # Only keep active parts of the loss
#             if attention_mask is not None:
#                 active_loss = attention_mask.view(-1) == 1
#                 active_character_logits = character_logits.view(-1, self.num_labels)
#                 active_character_labels = torch.where(
#                     active_loss,
#                     character_labels.view(-1),
#                     torch.tensor(loss_fct.ignore_index).type_as(character_labels),
#                 )
#                 character_loss = loss_fct(
#                     active_character_logits, active_character_labels
#                 )

#                 active_anatomy_logits = anatomy_logits.view(-1, self.num_labels)
#                 active_anatomy_labels = torch.where(
#                     active_loss,
#                     anatomy_labels.view(-1),
#                     torch.tensor(loss_fct.ignore_index).type_as(anatomy_labels),
#                 )
#                 anatomy_loss = loss_fct(active_anatomy_logits, active_anatomy_labels)

#                 loss = tendency_loss + character_loss + anatomy_loss

#         output = (tendency_logits, character_logits, anatomy_logits) + plmOutputs[2:]
#         return ((loss,) + output) if loss is not None else output


class PlmForTwoHeadTokenClassification(PlmPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: PlmConfig, num_labels2: int):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_labels2 = num_labels2

        self.plm = PlmModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier2 = nn.Linear(config.hidden_size, num_labels2)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels2=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        plmOutputs = self.plm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        sequence_output = plmOutputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        logits2 = self.classifier2(sequence_output)

        loss = None
        if (labels is not None) and (labels2 is not None):
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss1 = loss_fct(active_logits, active_labels)

                active_logits2 = logits2.view(-1, self.num_labels2)
                active_labels2 = torch.where(
                    active_loss,
                    labels2.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels2),
                )
                loss2 = loss_fct(active_logits2, active_labels2)
            else:
                loss1 = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss2 = loss_fct(logits2.view(-1, self.num_labels), labels2.view(-1))
            loss = loss1 + loss2

        if not return_dict:
            output = (logits, logits2) + plmOutputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TwoHeadTokenClassifierOutput(
            loss=loss,
            logits=logits,
            logits2=logits2,
            hidden_states=plmOutputs[3] if output_hidden_states else None,
            attentions=plmOutputs[4] if output_attentions else None,
        )


class PlmForCDEEWithTPLinker(PlmPreTrainedModel):
    def __init__(self, config: PlmConfig):
        super().__init__(config)

        self.plm = PlmModel(config, add_pooling_layer=False)
        self.tplinker = CDEETPLinker(config.hidden_size, 2, shaking_mode="cat")

        self.init_weights()

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
        tendency_labels=None,
        h2t_labels=None,
        h2h_labels=None,
        t2t_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        plmOutputs = self.plm(
            input_ids,
            attention_mask=attention_mask,
            word_left_bound=word_left_bound,
            word_right_bound=word_right_bound,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        sequence_output = plmOutputs[0]

        # [bsz, seq_len * (seq_len + 1) / 2, 2]
        h2t_logits, h2h_logits, t2t_logits, tendency_logits = self.tplinker(
            sequence_output
        )

        loss = None
        if tendency_labels is not None:
            # compute loss
            loss_fn = nn.CrossEntropyLoss()
            h2t_loss = loss_fn(
                h2t_logits.view(-1, h2t_logits.size(-1)),
                h2t_labels.view(-1),
            )

            h2h_loss = loss_fn(
                h2h_logits.view(-1, h2h_logits.size(-1)),
                h2h_labels.view(-1),
            )

            t2t_loss = loss_fn(
                t2t_logits.view(-1, t2t_logits.size(-1)),
                t2t_labels.view(-1),
            )

            tendency_loss = loss_fn(
                tendency_logits.view(-1, tendency_logits.size(-1)),
                tendency_labels.view(-1),
            )

            loss = h2t_loss + h2h_loss + t2t_loss + tendency_loss

        output = (h2t_logits, h2h_logits, t2t_logits, tendency_logits) + plmOutputs[2:]
        return ((loss,) + output) if loss is not None else output


class PlmForMedDGEntityDetection(PlmPreTrainedModel):
    def __init__(self, config: PlmConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.plm = PlmModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

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
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        plmOutputs = self.plm(
            input_ids,
            attention_mask=attention_mask,
            word_left_bound=word_left_bound,
            word_right_bound=word_right_bound,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        pooled_output = plmOutputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = torch.sigmoid(self.classifier(pooled_output))

        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss(reduction='sum')
            loss = loss_fct(logits.view(-1), labels.view(-1))

        output = (logits,) + plmOutputs[2:]
        return ((loss,) + output) if loss is not None else output
