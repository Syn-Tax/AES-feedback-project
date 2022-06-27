import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, output_size, vocab_size, embedding_length):
        super().__init__()

		self.output_size = output_size
		self.vocab_size = vocab_size
        self.embedding_length = embedding_length
