import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class Model(nn.Module):
	def __init__(self, vocab_size, embedding_length, hidden_size):
		super().__init__()

		"""
		Arguments
		---------
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension
		hidden_size : Size of the hidden_state of the LSTM

		--------

		"""

		self.output_size = 1
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length

		self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_length)
		self.dropout = 0.8
		self.bilstm = nn.LSTM(self.embedding_length, self.hidden_size, dropout=self.dropout, bidirectional=True)
		self.W_s1 = nn.Linear(2*hidden_size, 350)
		self.W_s2 = nn.Linear(350, 30)

		self.label = nn.Linear(30*2*hidden_size, self.output_size)

	def attention_net(self, lstm_output):

		"""
		Arguments
		---------
		lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
		---------
		Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
				  attention to different parts of the input sentence.
		Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
					  attn_weight_matrix.size() = (batch_size, 30, num_seq)
		"""
		attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
		attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
		attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

		return attn_weight_matrix

	def forward(self, input_sentences, batch_size):

		"""
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size: The batch size

		Returns
		-------
		Output of the linear layer

		"""


		input = self.word_embeddings(input_sentences)
		input = input.permute(1, 0, 2)
		print(input)
		h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
		c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())

		output, (h_n, c_n) = self.bilstm(input, (h_0, c_0))
		output = output.permute(1, 0, 2)
		print(output)

		attn_weight_matrix = self.attention_net(output)
		hidden_matrix = torch.bmm(attn_weight_matrix, output)

		logits = torch.sigmoid(self.label(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2])))

		return logits
