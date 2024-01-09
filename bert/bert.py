from typing import Optional, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

from torch import Tensor

from .embedding import BertEmbedding


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout,
        output_attention,
    ) -> None:
        super().__init__()
        self.output_attention = output_attention
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)

    def forward(
        self,
        x: Tensor,
        # attention_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        attn_output, attn_output_weights = self.multi_head_attention(
            q, k, v, key_padding_mask=key_padding_mask, average_attn_weights=False
        )
        return attn_output, attn_output_weights if self.output_attention else None


class BertOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        hidden_dropout_prob,
        output_attentions,
    ) -> None:
        super().__init__()
        self.attn = SelfAttention(
            hidden_size, num_attention_heads, hidden_dropout_prob, output_attentions
        )
        self.output = BertOutput(hidden_size, hidden_dropout_prob)

    def forward(
        self,
        hidden_states: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        self_outputs, attention_weights = self.attn(
            hidden_states, key_padding_mask=key_padding_mask
        )
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output, attention_weights


class BertIntermediate(nn.Module):
    def __init__(
        self, hidden_size, intermediate_size=3072, intermediate_act_fn=F.gelu
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = intermediate_act_fn

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertEncoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        hidden_dropout_prob,
        output_attentions,
    ) -> None:
        super().__init__()
        self.output_attentions = output_attentions
        self.attn = BertAttention(
            hidden_size, num_attention_heads, hidden_dropout_prob, output_attentions
        )
        self.attn_output = BertOutput(hidden_size, hidden_dropout_prob)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.intermediate_dense = nn.Linear(intermediate_size, hidden_size)
        self.intermediate_output = BertOutput(hidden_size, hidden_dropout_prob)

    def forward(
        self,
        hidden_states: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        attn, attention_weights = self.attn(hidden_states, key_padding_mask)
        hidden_states = self.attn_output(attn, hidden_states)
        intermediate = self.intermediate(hidden_states)
        intermediate = self.intermediate_dense(intermediate)
        hidden_states = self.intermediate_output(intermediate, hidden_states)
        return hidden_states, attention_weights


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        """
        :param hidden_states:  [src_len, batch_size, hidden_size]
        :return: [batch_size, hidden_size]
        """

        token_tensor = hidden_states[0, :]  # [batch_size, hidden_size]
        pooled_output = self.dense(token_tensor)  # [batch_size, hidden_size]
        pooled_output = self.activation(pooled_output)  # [batch_size, hidden_size]
        return pooled_output


class BertLayers(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        hidden_dropout_prob,
        num_hidden_layers,
        output_attentions,
    ) -> None:
        super().__init__()
        self.output_attentions = output_attentions
        self.layers = nn.ModuleList(
            [
                BertEncoder(
                    hidden_size,
                    num_attention_heads,
                    intermediate_size,
                    hidden_dropout_prob,
                    output_attentions,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        attentions = []
        for layer in self.layers:
            hidden_states, attention = layer(hidden_states, attention_mask)
            if self.output_attentions:
                attentions.append(attention)
        return (hidden_states, attentions if self.output_attentions else None)


class BertModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        intermediate_size,
        max_position_embeddings,
        num_attention_heads,
        hidden_dropout_prob,
        num_hidden_layers,
        output_attentions,
        need_pooled: bool = True,
    ):
        super().__init__()
        self.need_pooled = need_pooled
        self.hidden_size = d_model
        self.output_attentions = output_attentions
        self.bert_embedding = BertEmbedding(
            vocab_size, d_model, 0, max_position_embeddings, 1, hidden_dropout_prob
        )
        self.bert_layers = BertLayers(
            d_model,
            num_attention_heads,
            intermediate_size,
            hidden_dropout_prob,
            num_hidden_layers,
            output_attentions,
        )
        if self.need_pooled:
            self.bert_pooler = BertPooler(hidden_size=d_model)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ):
        """
        :param input_ids:  [src_len, batch_size]
        :param attention_mask: [batch_size, src_len]
        :param token_type_ids: [src_len, batch_size]
        :param position_ids: [1, src_len]
        """
        embedding_output = self.bert_embedding(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        # embedding_output: [src_len, batch_size, hidden_size]
        sequence_output, attention_weights = self.bert_layers(
            embedding_output, attention_mask=attention_mask
        )
        # sequence_output: [src_len, batch_size, hidden_size]
        # 默认是最后一层的first token 即[cls]位置经dense + tanh 后的结果
        # pooled_output: [batch_size, hidden_size]
        return (
            self.bert_pooler(sequence_output) if self.need_pooled else sequence_output,
            attention_weights,
        )
