import torch

from torch import nn
from torch.nn import functional as F

from .bert import BertModel


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size, hidden_act=F.gelu, layer_norm_eps=1e-5):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = hidden_act
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, hidden_act=F.gelu, layer_norm_eps=1e-5):
        super().__init__()
        self.transform = BertPredictionHeadTransform(
            hidden_size, hidden_act, layer_norm_eps
        )

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.predictions = BertLMPredictionHead(**kwargs)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertForMaskedLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        intermediate_size,
        max_position_embeddings,
        num_attention_heads,
        hidden_dropout_prob,
        num_hidden_layers,
        output_attentions=False,
    ):
        super().__init__()
        self.output_attentions = output_attentions
        self.vocab_size = vocab_size
        self.bert = BertModel(
            vocab_size,
            d_model,
            intermediate_size,
            max_position_embeddings,
            num_attention_heads,
            hidden_dropout_prob,
            num_hidden_layers,
            output_attentions,
            need_pooled=False,
        )
        self.classifier = BertOnlyMLMHead(vocab_size=vocab_size, hidden_size=d_model)

    def forward(self, input_ids, labels, attention_mask=None, token_type_ids=None):
        attention_mask = attention_mask.to(dtype=torch.float32)
        # attention_mask.transpose_(0, 1)
        input_ids.transpose_(0, 1)
        bert_output, attention_weights = self.bert(
            input_ids, attention_mask, token_type_ids
        )  # [batch_size, hidden_size]
        logits = self.classifier(bert_output)  # [batch_size, vocab_size]
        loss = nn.CrossEntropyLoss()(logits.view(-1, self.vocab_size), labels.view(-1))
        return loss, logits, attention_weights
