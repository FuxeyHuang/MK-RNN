import json
import torch
import torch.utils.data as data
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import logging
from nltk.tokenize import word_tokenize
from utils.config import *

UNK_token = 1
PAD_token = 2
EOS_token = 3
SOS_token = 4


def hasNumbers(inputString):
		return any(char.isdigit() for char in inputString)

class Lang:
		def __init__(self):
				self.word2index = {}
				self.word2count = {}
				self.index2word = {UNK_token: 'UNK'}
				self.n_words = 2  # Count default tokens

		def index_words(self, story):
				for word in story:
						self.index_word(word)

		def index_word(self, word):
				if word not in self.word2index:
						self.word2index[word] = self.n_words
						self.word2count[word] = 1
						self.index2word[self.n_words] = word
						self.n_words += 1
				else:
						self.word2count[word] += 1
				return self.word2index[word]


class Dataset(data.Dataset):
		"""Custom data.Dataset compatible with data.DataLoader."""

		def __init__(self, train_data, word2index, rela2index, max_len, max_p):
				"""Reads source and target sequences from txt files."""
				self.train_data = train_data
				self.word2index = word2index
				self.rela2index = rela2index
				self.num_total_data = len(self.train_data)
				self.max_len = max_len
				self.max_p = max_p

		def __getitem__(self, index):

				"""Returns one data pair (source and target)."""
				sents = self.preprocess(self.train_data[index]['sentences'], self.word2index, self.max_len[index], trg=False)
#        relas = self.preprocess(self.train_data[index]['relations'], self.rela2index)
				# index_s = self.preprocess_index(self.index_seqs[index], src_seq)
				return sents, None, self.max_len[index], self.train_data[index]['sentences'], self.train_data[index]['relations']

		def __len__(self):
				return self.num_total_data

		def preprocess(self, sequence, word2id, max_len=0, trg=True):
				"""Converts words to ids."""
				if trg:
						entity_index = [[word[0], word[1]] for word in sequence]
						story = [word2id[word[2]] if word[2] in word2id else UNK_token for word in sequence]
						story = torch.Tensor(story)

						if USE_CUDA:
								story = story.cuda()
						return story, entity_index
				else:

						story = []
						# print(sequence)
						for s in sequence:
								now = [0 for i in range(max_len)]
								for index, j in enumerate(s):
										now[index] = word2id[j] if j in word2id else UNK_token
								story.append(now)


						story = torch.Tensor(story)

						if USE_CUDA:
								story = story.cuda()

						return story



def read_langs(file_name, max_line=None):
		logging.info(("Reading lines from {}".format(file_name)))
		data = []
		max_len = 0
		max_len_p = 0

		with open(file_name, 'r', encoding='utf-8') as fin:
				raw_json = json.load(fin)
				for d in raw_json:
						data_now = {}
						data_now['sentences'] = []
						data_now['relations'] = []
						vertexid2pos = {}
						max_len_p = max(max_len_p, len(d['sents_with_ent']))

#            max_len_s = 0
#            len_sofa = 0
						for s in d['sents_with_ent']:
								data_now['sentences'].append(s['tokens'])
								max_len = max(max_len, len(s['tokens']))

								for v in s['vertexSet']:

										vertexid2pos[v['vertex_id']] = (v['sent_id'], v['pos'][0][-1], v['pos'][0][0])
								
#                len_sofa += len(s['tokens'])
						
						
						for e in d['edgeSet']:
								data_now['relations'].append([vertexid2pos[e['h']], vertexid2pos[e['t']], e['r']])
								
							
						add_0_relation = True
						if add_0_relation:
							res = []
							mark = {}
							tot  = {}
							for j in data_now['relations']:
								tot[j[0]] = 0
								tot[j[1]] = 0
								mark[(j[0], j[1])] = j[2]
								mark[(j[1], j[0])] = j[2]
							
							for k1 in tot:
								for k2 in tot:
									if k1 == k2:
										continue
									
									if (k1, k2) in mark:
										res.append([k1, k2, mark[(k1, k2)]])
									else:
										res.append([k1, k2, -1])
							data_now['relations'] = res              

						data.append(data_now)

		# data format: number_of_article * ( number_of_sentence * number_of_words + relation_set )
		return data, max_len*max_len_p, max_len_p

def collate_fn(data):
		mx_word = 0
		mx_sent = 0
		for i in range(len(data)):
				mx_word = max(mx_word, data[i][0].size()[1])
				mx_sent = max(mx_sent, data[i][0].size()[0])
		
		o_data4 = []
		o_data0 = torch.zeros((len(data), mx_sent, mx_word))
		for i in range(len(data)):
				o_data4.append(data[i][4])
#        print(o_data0[i, 0:data[i][0].size(0), 0:data[i][0].size(1)].size())
#        print(data[i][0].size()[0], data[i][0].size()[1])
				o_data0[i, 0:data[i][0].size(0), 0:data[i][0].size(1)] = data[i][0]
		
		return o_data0, None, None, None, o_data4 # We igore the data that is temporily useless
		

def shrink_lang(a, lim=30000):
	l = []
	for i in a.word2count:
		l.append((-a.word2count[i], i))
	l.sort()
	
	l = l[:lim]
	a.n_words = 2
	a.word2index = {}
	a.word2count = {}
	a.index2word = {UNK_token: 'UNK'}
	for i in l:
		a.index_word(i[1])
		a.word2count[i[1]] = -i[0]
	
	return a    

def get_seq(train_data, lang_word, lang_rela, batch_size, type, max_len, max_p, training=True):

		# create proper dictionary
		
		re_count_max_len = []
		for d in train_data:
				max_len_now = 0
				for s in d['sentences']:
						max_len_now = max(max_len_now, len(s))
						if training:
							for k in s:
									lang_word.index_word(k)
				
				re_count_max_len.append(max_len_now)
				for i in range(len(d['relations'])):
						d['relations'][i][2] = (lang_rela.index_word(d['relations'][i][2]) if d['relations'][i][2] != -1 else UNK_token)

#    print(train_data[0]['relations'])
		shrink_lang(lang_word)
		dataset = Dataset(train_data, lang_word.word2index, lang_rela.word2index, re_count_max_len, max_p)
		data_loader = torch.utils.data.DataLoader(dataset=dataset,
																							batch_size=batch_size,
																							shuffle=type,
																							collate_fn = collate_fn)
		return data_loader
		
		
def prepare_data_seq(batch_size=1, shuffle=True, Debug="False", training=True, lw = None, lr = None):
		
		# determine which dataset to use
		if training:
				if Debug == 'True':
						file_train = './utils/distant_10w_train_data_debug.json'
				else:
						file_train = './utils/distant_10w_train_data.json'
		else:
				if Debug == 'True':
						file_train = './utils/distant_10w_test_data_debug.json'
				else:
						file_train = './utils/distant_10w_test_data.json'

		train_data, max_len, max_p = read_langs(file_train, max_line=None)
		
		# print(max_len, max_p)
		if training:
			lang_word = Lang()
			lang_rela = Lang()
		else:
			lang_word = lw
			lang_rela = lr
		
		train = get_seq(train_data, lang_word, lang_rela, batch_size, True, max_len, max_p, training=training)

		return train, lang_word, lang_rela, max_len, max_p

train, _, _, _, _ = prepare_data_seq()