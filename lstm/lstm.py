from typing import Optional

import torch

from torch import nn, Tensor


class LSTMCell(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        # i_t
        self.i = nn.Linear(hidden_size * 2, hidden_size)

        # f_t
        self.f = nn.Linear(hidden_size * 2, hidden_size)

        # c_t
        self.c = nn.Linear(hidden_size * 2, hidden_size)

        # o_t
        self.o = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x: Tensor, init_states: Optional[tuple[Tensor, Tensor]] = None):
        src_len, batch_size, hidden_size = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(batch_size, hidden_size).to(x.device),
                torch.zeros(batch_size, hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states

        for t in range(src_len):
            x_t = x[t, :, :]

            i_t = torch.sigmoid(self.i(torch.cat([x_t, h_t], dim=-1)))
            f_t = torch.sigmoid(self.f(torch.cat([x_t, h_t], dim=-1)))
            g_t = torch.tanh(self.c(torch.cat([x_t, h_t], dim=-1)))
            o_t = torch.sigmoid(self.o(torch.cat([x_t, h_t], dim=-1)))
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t)
        hidden_seq = torch.stack(hidden_seq, dim=0)
        # hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, vocab_size, pad_token_id):
        super().__init__()
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList([LSTMCell(hidden_size) for _ in range(num_layers)])

    def forward(self, x: Tensor, init_states: Optional[tuple[Tensor, Tensor]] = None):
        _, batch_size, _ = x.size()
        if init_states is None:
            h_t, c_t = (
                torch.zeros(batch_size, self.hidden_size).to(x.device),
                torch.zeros(batch_size, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states
        for cell in self.cells:
            x, (h_t, c_t) = cell(x, (h_t, c_t))
        return x, (h_t, c_t)

    def init_hidden(self, batch_size):
        return (
            torch.zeros(batch_size, self.hidden_size),
            torch.zeros(batch_size, self.hidden_size),
        )
