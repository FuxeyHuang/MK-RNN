import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
# from utils.masked_cross_entropy import *
from utils.config import *
import random
import numpy as np
import datetime
# from utils.measures import wer, moses_multi_bleu
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn  as sns
import nltk
import os
import logging
from sklearn.metrics import f1_score


class Lstm(nn.Module):
	def __init__(self, hidden_size, max_len, max_p, lang, path, lr, n_layers, dropout, relation_size):
		super(Lstm, self).__init__()
		self.name = "Lstm"
		self.input_size = lang.n_words
		self.output_size = lang.n_words
		self.hidden_size = hidden_size
		self.max_len = max_len  ## max input
		self.max_p = max_p  ## max article len
		self.lang = lang
		self.relation_size = relation_size
		self.lr = lr
		self.n_layers = n_layers
		self.dropout = dropout
		
		self.decoder = Decoder(max_p, max_len, hidden_size, dropout, lang.n_words, embedding_dim = hidden_size, relation_size = relation_size)
		self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
		
		self.criterion = nn.MSELoss()
		self.loss = 0
		# Move models to GPU
		if USE_CUDA:
			self.encoder.cuda()


	def train_batch(self, input_batches, input_lengths, target_batches, reset):
		if reset:
			self.loss = 0
			self.loss_gate = 0
			self.loss_ptr = 0
			self.loss_vac = 0
			self.print_every = 1

		self.batch_size = 1
		# Zero gradients of both optimizers
#		self.encoder_optimizer.zero_grad()
		self.decoder_optimizer.zero_grad()
		self.decoder.clean_data()
				
		h = Variable(torch.zeros(1, 1, self.hidden_size))
		c = Variable(torch.zeros(1, 1, self.hidden_size))
		if USE_CUDA:
			c = c.cuda()
		
		sofa = 0
		for i in range(input_batches.size()[1]):
			h, c = self.decoder(input_batches[:,i].long(), h, c)
			
		heads = [a[0] for a in target_batches]
		tails = [a[1] for a in target_batches]
		relas = Variable(torch.from_numpy(np.array([a[2] for a in target_batches])).long())
		loss = self.decoder.extract_ralation(heads, tails, relas)
		print(loss)
		loss.backward()
		self.encoder_optimizer.step()
		self.decoder_optimizer.step()
		
	
			


class Decoder(nn.Module):
	def __init__(self, max_p, max_len, hidden_size, dropout, vocab, embedding_dim, relation_size):
		super(Decoder, self).__init__()
		self.max_p = max_p
		self.max_len = max_len
		self.hidden_size = hidden_size
		self.dropout = dropout
		self.num_vocab = vocab
		self.embedding_dim = embedding_dim
		self.relation_size = relation_size
		
		self.embed = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
		
#		self.pointer = 0
#		self.A = Variable(torch.zeros(1, hidden_size))

		self.output_pointer = 0
		self.outputs = Variable(torch.zeros(max_len, hidden_size))
		if USE_CUDA:
			self.outputs = self.outputs.cuda()
#			self.A = self.A.cuda()

		self.Lstm = nn.LSTM(input_size = self.hidden_size, hidden_size = self.hidden_size, dropout = self.dropout, batch_first=True)
		self.bili = torch.nn.Bilinear(self.hidden_size, self.hidden_size, self.relation_size)
		self.softmax = nn.Softmax(dim=1)
		self.criterion = nn.CrossEntropyLoss()
		
	def clean_data(self):
#		self.pointer = 0
#		self.A = Variable(torch.zeros(1, self.hidden_size))
		self.output_pointer = 0
		self.outputs = Variable(torch.zeros(self.max_len, self.hidden_size))
		if USE_CUDA:
#			self.A = self.A.cuda()
			self.outputs = self.outputs.cuda()
		
	
	def forward(self, sent, h, c):
		sent = self.embed(sent)   # one sentence a time 1*word_num*hidden_size
				
		for i in range(sent.size()[1]):
			o, (h, c) = self.Lstm(sent[:, i:i+1, :], (h, c))
			self.outputs[self.output_pointer] = o[0][0]
			self.output_pointer += 1
		
		return h, c
	
	def extract_ralation(self, heads, tails, relas, training=True):
		h = Variable(torch.zeros(len(heads), self.hidden_size))
		t = Variable(torch.zeros(len(tails), self.hidden_size))
		for i in range(len(heads)):
			h[i] = self.outputs[heads[i]]
			t[i] = self.outputs[tails[i]]
		
		r = self.bili(h, t)
#        r = self.softmax(r)
		
		if training:
			return self.criterion(r, relas)
		else:
			return torch.argmax(r, dim = 1)
		
		
		


class AttrProxy(object):
	"""
	Translates index lookups into attribute lookups.
	To implement some trick which able to use list of nn.Module in a nn.Module
	see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
	"""

	def __init__(self, module, prefix):
		self.module = module
		self.prefix = prefix

	def __getitem__(self, i):
		return getattr(self.module, self.prefix + str(i))
