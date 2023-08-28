from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

from torch import Tensor


class BertEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : normal embedding matrix
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        pad_token_id,
        max_position_embeddings,
        type_vocab_size,
        hidden_dropout_prob,
    ):
        super().__init__()
        self.token_embeddings = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            pad_token_id=pad_token_id,
        )
        # return shape [src_len, batch_size, hidden_size]

        self.position_embeddings = PositionalEmbedding(
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
        )
        # return shape [src_len, 1, hidden_size]

        self.token_type_embeddings = SegmentEmbedding(
            type_vocab_size=type_vocab_size,
            hidden_size=hidden_size,
        )
        # return shape  [src_len, batch_size, hidden_size]

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings).expand((1, -1))
        )
        # shape: [1, max_position_embeddings]

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ):
        src_len = input_ids.size(0)
        token_embedding: Tensor = self.token_embeddings(input_ids)
        # shape:[src_len, batch_size, hidden_size]

        if position_ids is None:
            position_ids = self.position_ids[:, :src_len]  # [1,src_len]
        positional_embedding: Tensor = self.position_embeddings(position_ids)
        # [src_len, 1, hidden_size]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(
                input_ids, device=self.position_ids.device
            )  # [src_len, batch_size]
        segment_embedding: Tensor = self.token_type_embeddings(token_type_ids)
        # [src_len, batch_size, hidden_size]

        embeddings = token_embedding + positional_embedding + segment_embedding
        # [src_len, batch_size, hidden_size] + [src_len, 1, hidden_size] + [src_len, batch_size, hidden_size]
        embeddings = self.layer_norm(embeddings)  # [src_len, batch_size, hidden_size]
        embeddings = self.dropout(embeddings)
        return embeddings


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token_id=0):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)

    def forward(self, input_ids):
        """
        :param input_ids: shape : [input_ids_len, batch_size]
        :return: shape: [input_ids_len, batch_size, hidden_size]
        """
        return self.embedding(input_ids)


class PositionalEmbedding(nn.Module):
    """
    Bert中的位置编码完全不同于Transformer中的位置编码，
    前者本质上也是一个普通的Embedding层，而后者是通过公式计算得到，
    而这也是为什么Bert只能接受长度为512字符的原因，因为位置编码的最大size为512
    """

    def __init__(self, hidden_size, max_position_embeddings):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_position_embeddings, hidden_size)

    def forward(self, position_ids):
        """
        :param position_ids: [1, position_ids_len]
        :return: [position_ids_len, 1, hidden_size]
        """
        return self.embedding(position_ids).transpose(0, 1)


class SegmentEmbedding(nn.Module):
    def __init__(self, type_vocab_size, hidden_size):
        super(SegmentEmbedding, self).__init__()
        self.embedding = nn.Embedding(type_vocab_size, hidden_size)

    def forward(self, token_type_ids):
        """
        :param token_type_ids:  shape: [token_type_ids_len, batch_size]
        :return: shape: [token_type_ids_len, batch_size, hidden_size]
        """
        return self.embedding(token_type_ids)
