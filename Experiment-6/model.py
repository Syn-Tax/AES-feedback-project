import torch
import torch.nn as nn
import math
import sys

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_length, hidden_size, nhead, num_encoder_layers, seq_len, dropout=0.1,
                 regression_size=256, num_regression_layers=1):
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

        self.regression_head = RegressionHead(embedding_length, seq_len, hidden_size=regression_size, num_hidden_layers=num_regression_layers)

        #self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.embedding_length)
        src = self.positional_encodings(src)

        encoder_output = self.encoder(src, src_mask)
        output = self.regression_head(encoder_output.view(-1, self.embedding_length*self.seq_len))

        #print(output)

        return output

def generate_square_subsequent_mask(size):
    return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_length, dropout, max_len=512):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_length, 2) * (-math.log(10000.0) / embedding_length))
        pe = torch.zeros(max_len, 1, embedding_length)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class RegressionHead(nn.Module):
    def __init__(self, embedding_length, seq_len, hidden_size=512, num_hidden_layers=0):
        super().__init__()

        self.embedding_length = embedding_length
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.input_fc = nn.Linear(embedding_length*seq_len, hidden_size)
        self.hidden_fcs = [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)]
        self.output_fc = nn.Linear(hidden_size, 1)

        #self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_fc.bias.data.zero_()
        self.input_fc.weight.data.uniform_(-initrange, initrange)
        self.output_fc.bias.data.zero_()
        self.output_fc.weight.data.uniform_(-initrange, initrange)

        for fc in self.hidden_fcs:
            fc.bias.data.zero_()
            fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, hidden_matrix):
        output = self.input_fc(hidden_matrix.view(-1, self.embedding_length*self.seq_len))

        for fc in self.hidden_fcs:
            output = fc(output)

        output = nn.Sigmoid()(self.output_fc(output))

        return output
