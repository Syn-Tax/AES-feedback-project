import torch
import torch.nn as nn
import sys

class Model(nn.Module):
    def __init__(self, output_size, d_model=512, nhead=8, num_encoder_layers=6):
        super().__init__()

		self.output_size = output_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers

        self.encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, self.num_encoder_layers)

    def forward(self, input_sequences):
        encoder_output = self.encoder(input_sequences)

        print(encoder_output)
        sys.exit(0)
