import torch
import torch.nn as nn
import math
import sys

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_length, hidden_size, nhead, num_encoder_layers, seq_len, dropout=0.1,
                 regression_size=256):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.seq_len = seq_len

        self.positional_encodings = PositionalEncoding(embedding_length, dropout)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_length)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_length, nhead=self.nhead, dim_feedforward=self.hidden_size)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, self.num_encoder_layers)

        self.regression_head = RegressionHead(embedding_length, hidden_size=regression_size)
        self.linear = nn.Linear(seq_len*embedding_length, 1)
        #self.linear = nn.Linear(1, embedding_length)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.embedding_length)
        src = self.positional_encodings(src)

        encoder_output = self.encoder(src, src_mask)
        output = self.linear(encoder_output.view(-1, self.embedding_length*self.seq_len))

        #print(output)

        return output

def generate_square_subsequent_mask(size):
    return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class RegressionHead(nn.Module):
    def __init__(self, d_model, hidden_size=512):
        super().__init__()

        self.d_model = d_model
        self.hidden_size = hidden_size

        self.lin1 = nn.Linear(d_model, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.lin1.bias.data.zero_()
        self.lin1.weight.data.uniform_(-initrange, initrange)

    def forward(self, hidden_vector):
        return self.lin1(hidden_vector)
