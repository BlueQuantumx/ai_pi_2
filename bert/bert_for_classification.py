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
        output_attentions=False,
    ):
        super().__init__()
        self.output_attentions = output_attentions
        self.bert = BertModel(
            vocab_size,
            d_model,
            intermediate_size,
            max_position_embeddings,
            num_attention_heads,
            hidden_dropout_prob,
            num_hidden_layers,
            output_attentions,
        )
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, labels, attention_mask=None, token_type_ids=None):
        bert_output = self.bert(
            input_ids, attention_mask, token_type_ids
        )  # [batch_size, hidden_size]
        logits = self.classifier(bert_output[0])  # [batch_size, num_labels]
        loss = nn.CrossEntropyLoss()(logits, labels)
        if self.output_attentions:
            return loss, logits, bert_output[-1]
        return loss, logits
