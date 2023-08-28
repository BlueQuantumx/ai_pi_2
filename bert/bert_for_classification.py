from torch import nn

from .bert import BertModel


class BertForClassification(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        intermediate_size,
        max_position_embeddings,
        num_attention_heads,
        hidden_dropout_prob,
        num_hidden_layers,
        num_labels: int,
    ):
        super().__init__()
        self.bert = BertModel(
            vocab_size,
            d_model,
            intermediate_size,
            max_position_embeddings,
            num_attention_heads,
            hidden_dropout_prob,
            num_hidden_layers,
        )
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        bert_output = self.bert(
            input_ids, attention_mask, token_type_ids
        )  # [batch_size, hidden_size]
        logits = self.fc(bert_output)  # [batch_size, num_labels]
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss, logits
