"""
Author: Da Ma
Date: 2022.01.06
Description: config for plm.
"""
from transformers import BertConfig


class PlmConfig(BertConfig):
    """
    Config of our plm.
    NOTE:
        1. [2022.01.06]
            Support MLM solely. Different from Bert, the vocabulary size changes.
    """

    def __init__(
            self,
            vocab_size=20000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            gradient_checkpointing=False,
            position_embedding_type="absolute",
            use_cache=True,
            use_fofe=True,
            parallel=True,
            fofe_dim=256,
            window_size=7,
            is_conv1d=True,
            alpha=0.5,
            fofe_mode="add",
            every_layer=True,
            gumbel_tau=1.0,
            kl_beta=0.0,
            seg_bos_id=5,
            seg_eos_id=6,
            freeze=False,
            **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            position_embedding_type=position_embedding_type,
            use_cache=use_cache,
            **kwargs
        )

        # [FIXME: INTRODUCE MORE CONFIGURATION HERE!]

        self.use_fofe = use_fofe
        self.alpha = alpha
        self.fofe_mode = fofe_mode
        self.every_layer = every_layer
        self.parallel = parallel
        self.window_size = window_size
        self.fofe_dim = fofe_dim
        self.is_conv1d = is_conv1d
        self.gumbel_tau = gumbel_tau
        self.kl_beta = kl_beta
        self.seg_bos_id = seg_bos_id
        self.seg_eos_id = seg_eos_id
        self.freeze = freeze
