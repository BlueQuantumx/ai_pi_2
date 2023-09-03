import torch

from torch import nn, Tensor, functional as F

from .lstm import LSTM


class LstmForClassification(nn.Module):
    def __init__(self, num_labels, hidden_size, num_layers, vocab_size, pad_token_id):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        # self.lstm = nn.LSTM(
        #     input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers
        # )
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.lstm = LSTM(hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_labels)

    def forward(self, x: Tensor, labels: Tensor):
        x = self.embedding(x)
        _, (hidden_state, _) = self.lstm(x)
        logits = self.fc(hidden_state)  # [batch_size, num_labels]
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss, logits
