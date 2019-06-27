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
import math

glove_path = "/data/disk1/private/hlx/glove/glove.6B.50d.txt"

def data_manager(input_batches, target_batches, hidden_size, max_entity=100):
    heads = []
    tails = []
    relas = []
    i_size = input_batches.size()
    pos3 = (torch.zeros((i_size[0], i_size[1], i_size[2]))).long()
    input_batches = input_batches.data.cpu().numpy()
    for index, i in enumerate(target_batches):

        h = []
        t = []
        r = []
        mark = {}
        tot  = {}
        igr  = {}
        for j in i:
            tot[j[0]] = 0
            tot[j[1]] = 0
            mark[(j[0], j[1])] = j[2]
            if j[3] == 1:
                igr[(j[0], j[1])] = 1
#            mark[(j[1], j[0])] = j[2]
        
        pos_cnt = 0
        for k1 in tot:
            pos_cnt += 1
#            if k1[2] != k1[1]:
#                pos3[index][k1[0]][k1[2]] = 1
#                pos3[index][k1[0]][k1[1]] = 2
#            else:
#                pos3[index][k1[0]][k1[1]] = 3
            for k2 in range(k1[2], k1[1]+1):
                pos3[index][k1[0]][k2] = pos_cnt
        
        for k1 in tot:
            for k2 in tot:
                if k1 == k2:
                    continue
                
#                if (k1, k2) in igr:
#                    if random.randint(1, 100) > 4:
#                        continue
                
                if (k1, k2) in mark:
                    h.append(k1)
                    t.append(k2)
                    r.append(mark[(k1, k2)])
                else:
                    h.append(k1)
                    t.append(k2)
                    r.append(0)
        heads.append(h)
        tails.append(t)
        relas.append(r)
        
    mx = 0
    for i in range(len(heads)):
        mx = max(mx, len(heads[i]))

    igored_index = -100 #by pytorch
    for i in range(len(relas)):
        for j in range(mx - len(relas[i])):
            relas[i].append(igored_index)
            
    h_ = np.zeros((len(heads)*mx, hidden_size))
    t_ = np.zeros((len(heads)*mx, hidden_size))
    cnt = -1
    for i in range(len(heads)):
        for j in range(mx):
            cnt += 1
            
            if j >= len(heads[i]):
                continue
            
            for k in range(hidden_size):
                h_[cnt][k] = (i*i_size[1]*i_size[2]+heads[i][j][0]*i_size[2]+heads[i][j][1])
                t_[cnt][k] = (i*i_size[1]*i_size[2]+tails[i][j][0]*i_size[2]+tails[i][j][1])
            
            
    
    relas = Variable(torch.from_numpy(np.array(relas)).long())
    h_ = Variable(torch.from_numpy(np.array(h_)).long())
    t_ = Variable(torch.from_numpy(np.array(t_)).long())

    pos = (torch.zeros((i_size[0], i_size[1], max_entity, i_size[2]))).float() # max_entity = 100
    pos2= (torch.zeros((i_size[0], i_size[1], i_size[2], hidden_size))).long()
    def compare(b, s1, l1, r1, s2, l2, r2):
        if r2-l2 != r1-l1:
            return False
        
        for i in range(r1-l1+1):
            if input_batches[b][s1][l1+i] != input_batches[b][s2][l2+i]:
                return False;
        return True
    
    mx_entity = 0

    cnt = 0
    cnt2= 0
    for index, i in enumerate(target_batches):
        
        tot  = {}
        entity_dic = {}
        for j in i:
            tot[j[0]] = 0
            tot[j[1]] = 0
                    
        for k1 in tot:
            same = False
            who = -1
            for k2 in entity_dic:
                if compare(index, k1[0], k1[2], k1[1], k2[0], k2[2], k2[1]):
                    same = True
                    who = entity_dic[k2]
                    break
            
            if same:
                entity_dic[k1] = who
                continue
            
            entity_dic[k1] = len(entity_dic) + 1
            for j in range(i_size[1]):
                for k in range(i_size[2]):
                    
                    if k + k1[1] - k1[2] < i_size[2]:
#                        cnt += k1[1] - k1[2]
                        if compare(index, k1[0], k1[2], k1[1], j, k, k+k1[1]-k1[2]):
                            for l in range(k1[1]-k1[2] + 1):
                                pos[index][j][entity_dic[k1]][k+l] = 1
                                pos2[index,j,k+l,:] = entity_dic[k1]
                for k in range(len(entity_dic) + 1):
                    s = torch.sum(pos[index, j, k, :])
                    if s > 0:
                        pos[index, j, k, :] /= s*1.0
                        
                    
                            
        mx_entity = max(mx_entity, len(entity_dic)+1)
        if mx_entity > max_entity:
            print("To much entity in an article")
            assert(False)
            

    pos = Variable(pos)
    pos2= Variable(pos2)
    pos3= Variable(pos3)

    if USE_CUDA:
        relas = relas.cuda()
        h_ = h_.cuda()
        t_ = t_.cuda()
        pos = pos.cuda()
        pos2 = pos2.cuda()
        pos3 = pos3.cuda()

    return h_, t_, relas, pos, mx, pos2, pos3

            
        

class CNN(nn.Module):
    def __init__(self, hidden_size, max_len, max_p, lang, path, lr, n_layers, dropout, relation_size, batch_size, window_size, model_name, glove=True):
        super(CNN, self).__init__()
#        glove = (embedding_dim == 50)
        self.name = model_name
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
        self.batch_size = batch_size
        self.window_size = window_size
        self.final_dic = {}
        self.final_dic['cnn1'] = hidden_size*4
        self.final_dic['cnn2'] = hidden_size*4
        self.final_dic['cnn3'] = hidden_size*4
        
        self.cnn = _CNN(max_p, max_len, hidden_size, dropout, lang.n_words, embedding_dim = hidden_size, relation_size = relation_size, batch_size = batch_size, window_size=window_size, final_channels=self.final_dic[self.name], model_name=self.name)
        self.cnn_optimizer = optim.Adam(self.cnn.parameters(), lr=lr, weight_decay=1e-5)
        
        self.criterion = nn.MSELoss()
        self.loss = 0
        # Move models to GPU
        if USE_CUDA:
            self.cnn.cuda()
        
        if glove:
            load_embed(self.cnn, glove_path, self.lang)


    def train_batch(self, input_batches, input_lengths, target_batches, reset):
        if reset:
            self.loss = 0
            self.loss_gate = 0
            self.loss_ptr = 0
            self.loss_vac = 0
            self.print_every = 1

        # Zero gradients of both optimizers
        self.cnn_optimizer.zero_grad()
        self.cnn.clean_data()
        
        from time import time
        heads, tails, relas, _, mx,_ , pos = data_manager(input_batches, target_batches, self.hidden_size)
        

        if USE_CUDA:
            input_batches = input_batches.cuda()
            pos = pos.cuda()
        
        
        i_size = input_batches.size()
        self.cnn(input_batches.view(i_size[0], i_size[1]*i_size[2]).long(), pos.view(i_size[0], i_size[1]*i_size[2]).long())
        loss, r = self.cnn.extract_relation(heads, tails, relas, mx)
        if self.training:
            loss.backward()
            self.cnn_optimizer.step()
        return loss, r


class Memnet(nn.Module):
    def __init__(self, hidden_size, max_len, max_p, lang, path, lr, n_layers, dropout, relation_size, batch_size, model_name, embedding_dim, glove=True, use_transformer=False, transformer_layer = 1, use_biLSTM=False, use_rmc=False, use_ca=False):
        super(Memnet, self).__init__()
        glove = (embedding_dim == 50)
        self.name = model_name
        self.use_transformer = use_transformer
        self.use_biLSTM = use_biLSTM
        self.use_rmc = use_rmc
        self.use_ca = use_ca
        self.transformer_layer = transformer_layer
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.embedding_dim= embedding_dim
        self.max_len = max_len  ## max input
        self.max_p = max_p  ## max article len
        self.lang = lang
        self.relation_size = relation_size
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_layer = 2
        self.num_direc = (2 if use_biLSTM else 1)
        self.decoder = Decoder(max_p, max_len, hidden_size, dropout, lang.n_words, embedding_dim = self.embedding_dim, relation_size = relation_size, batch_size = batch_size, num_layer=self.num_layer)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr, weight_decay=1e-5)
        
        self.criterion = nn.MSELoss()
        self.loss = 0
        # Move models to GPU
        if USE_CUDA:
            self.decoder.cuda()
        if glove:
            load_embed(self.decoder, glove_path, self.lang)
            print("Successfuly loading Glove 50D\n")

    def train_batch(self, input_batches, input_lengths, target_batches, reset, count_loss=True, last_loss=None): # input_batches: 
        if reset:
            self.loss = 0
            self.loss_gate = 0
            self.loss_ptr = 0
            self.loss_vac = 0
            self.print_every = 1
        
        # Zero gradients of both optimizers
        self.decoder_optimizer.zero_grad()
        self.decoder.clean_data()
        
        heads, tails, relas, pos, mx, pos2, pos3 = data_manager(input_batches, target_batches, self.hidden_size)
        if USE_CUDA:
            input_batches = input_batches.cuda()
            pos = pos.cuda()
            
        h = Variable(torch.zeros(self.num_layer*self.num_direc, self.batch_size, int(self.hidden_size/self.num_direc)))
        c = Variable(torch.zeros(self.num_layer*self.num_direc, self.batch_size, int(self.hidden_size/self.num_direc)))
        
        if USE_CUDA:
            c = c.cuda()
            h = h.cuda()
            
        
        s = input_batches.size()
        
        assert self.name in ['lstm', 'mem', 'g', 'mem_g', 'g_gcn', 'mem_g_gcn', 'after_gcn']
        model_dic = {}
        model_dic['lstm'] = [False, False, False]
        model_dic['mem'] = [True, False, False]
        model_dic['g'] = [False, True, False]
        model_dic['mem_g'] = [True, True, False]
        model_dic['g_gcn'] = [False, True, False]
        model_dic['mem_g_gcn'] = [True, True, True]
        now = model_dic[self.name]
        
        for i in range(s[1]):
            h, c = self.decoder(input_batches[:,i].long(), pos[:,i], pos2[:,i].long(), pos3[:, i].long(), h, c, now[0], i, s[1]*s[2], s[1], now[1], now[2], 
                self.use_transformer, self.transformer_layer, self.use_biLSTM, self.use_rmc)
        self.decoder.transpose_outputs(s)

        loss, r = self.decoder.extract_relation(heads, tails, relas, mx, use_ca = self.use_ca)
        if self.training:
            
            if count_loss == True:
                if not last_loss is None:
                    loss += last_loss
                loss.backward()
                
#            print("w_t")
#            print(self.decoder.w_t.weight)
#            print("transformer1")
#            for d in self.decoder.transformer1.parameters():
#                print(d)
            
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.25)
                self.decoder_optimizer.step()
            
        return loss, r
    

            


def load_embed(model, path, lang):
    dic = {}
    with open(path, 'r', encoding='utf-8') as f:
        for l in f:
            l = l.split()
            dic[l[0]] = np.array([float(i) for i in l[1:]])
    if USE_CUDA:
        pre_dic = model.embed.weight.cpu().data.numpy()
    else:
        pre_dic = model.embed.weight.data.numpy()
        
    for w in lang.word2index:
        if w.lower() in dic:
            pre_dic[lang.word2index[w]] = dic[w.lower()]
    
    model.embed.weight.data.copy_(torch.from_numpy(pre_dic))
class Decoder(nn.Module):
    def __init__(self, max_p, max_len, hidden_size, dropout, vocab, embedding_dim, relation_size, batch_size, num_layer):
        super(Decoder, self).__init__()
        self.max_p = max_p
        self.max_len = max_len*20
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_vocab = vocab
        self.embedding_dim = embedding_dim
        self.relation_size = relation_size
        self.batch_size = batch_size
        self.num_layer = num_layer
        self.memory_slot = 4
        self.rmc_memory_size=int(self.hidden_size/self.memory_slot)
        self.embed2rmc_mem = nn.Linear(self.embedding_dim, self.rmc_memory_size)
        self.rmc_layer = Transformer(d_model = self.rmc_memory_size, d_inner=self.rmc_memory_size*4, n_head=1, d_k = self.rmc_memory_size, d_v = self.rmc_memory_size, dropout=0.1)
        self.layer_norm_rmc = nn.LayerNorm(self.rmc_memory_size)
        self.rmc_fc1 = nn.Linear(self.rmc_memory_size, self.rmc_memory_size)
        self.rmc_fc2 = nn.Linear(self.rmc_memory_size, self.rmc_memory_size)
        self.EW_rmc = nn.Linear(self.rmc_memory_size, self.rmc_memory_size)
        self.EU_rmc = nn.Linear(self.rmc_memory_size, self.rmc_memory_size)

        self.embed = nn.Embedding(self.num_vocab, int(embedding_dim), padding_idx=PAD_token)
        self.embed_pos = nn.Embedding(self.max_len, int(embedding_dim), padding_idx=PAD_token) # 4 different tokens
#        self.memory_attention = MultiHeadAttention(n_head=1, d_model=self.hidden_size, d_k=self.hidden_size, d_v=self.embedding_dim)
        self.query = nn.Linear(self.embedding_dim, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.embedding_dim)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        
        self.EW = nn.Linear(self.hidden_size, self.hidden_size)
        self.EU = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.EW2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.EU2 = nn.Linear(self.hidden_size, self.hidden_size)

#        self.embed_pos = nn.Embedding(self.max_len, int(embedding_dim), padding_idx=PAD_token)
        self.max_entity = 100
        self.output_pointer = 0

        if USE_CUDA:

            self.outputs = Variable(torch.zeros(self.max_len*self.batch_size, self.hidden_size)).cuda()
            self.A = Variable(torch.zeros(1, self.hidden_size)).cuda()
            self.E = Variable(torch.zeros(self.batch_size, self.max_entity, self.hidden_size)).cuda()
            self.MS = Variable(torch.zeros(self.batch_size, self.memory_slot, self.rmc_memory_size)).cuda()
            self.last_hidden = Variable(torch.zeros(self.batch_size, 1, self.hidden_size)).cuda()
            for i in range(self.memory_slot):
                self.MS[:, i, i] = 1

        else:
            self.outputs = Variable(torch.zeros(self.max_len*self.batch_size, self.hidden_size))
            self.A = Variable(torch.zeros(1, self.hidden_size))
            self.E = Variable(torch.zeros(self.batch_size, self.max_entity, self.hidden_size))
            self.MS = Variable(torch.zeros(self.batch_size, self.memory_slot, self.rmc_memory_size))
            self.last_hidden = Variable(torch.zeros(self.batch_size, 1, self.hidden_size))
            for i in range(self.memory_slot):
                self.MS[:, i, i] = 1
        
        self.lstm = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.hidden_size, num_layers=self.num_layer, dropout = self.dropout, batch_first=True)
        self.b_lstm = nn.LSTM(input_size = self.embedding_dim, hidden_size = int(self.hidden_size/2), num_layers=self.num_layer, dropout = self.dropout, batch_first=True, bidirectional = True)
        
        self.bili = torch.nn.Bilinear(self.hidden_size, self.hidden_size, self.relation_size)
        self.softmax = nn.Softmax(dim=1)
        self.Dropout = nn.Dropout(p=self.dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.line = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.line_g=torch.nn.Linear(self.hidden_size, self.embedding_dim)
        self.nonlinear = nn.Sigmoid()
        
        ## Transformer module
        self.w_t = torch.nn.Linear(self.embedding_dim, self.hidden_size)
        self.num_heads = 8
        self.transformer1 = Transformer(d_model = self.hidden_size, d_inner=self.hidden_size*4, n_head=self.num_heads, d_k = int(self.hidden_size/self.num_heads), d_v = int(self.hidden_size/self.num_heads), dropout=0.1)
        self.transformer2 = Transformer(d_model = self.hidden_size, d_inner=self.hidden_size*4, n_head=self.num_heads, d_k = int(self.hidden_size/self.num_heads), d_v = int(self.hidden_size/self.num_heads), dropout=0.1)
        self.transformer3 = Transformer(d_model = self.hidden_size, d_inner=self.hidden_size*4, n_head=self.num_heads, d_k = int(self.hidden_size/self.num_heads), d_v = int(self.hidden_size/self.num_heads), dropout=0.1)

        # context aware structure
        self.ca = Transformer(d_model = self.hidden_size, d_inner=self.hidden_size*4, n_head=1, d_k = self.hidden_size, d_v = self.hidden_size, dropout=0.1)
        self.bili_ca = nn.Bilinear(self.hidden_size, self.hidden_size, self.hidden_size)
        self.h2r = nn.Linear(self.hidden_size, self.relation_size)
        
    def clean_data(self):
        self.output_pointer = 0
        del self.outputs
        del self.A
        
        if USE_CUDA:
            self.outputs = Variable(torch.zeros(self.max_len*self.batch_size, self.hidden_size)).cuda()
            self.A = Variable(torch.zeros(self.batch_size, 1, self.hidden_size)).cuda()
            self.E = Variable(torch.zeros(self.batch_size, self.max_entity, self.hidden_size)).cuda()
            self.MS = Variable(torch.zeros(self.batch_size, self.memory_slot, self.rmc_memory_size)).cuda()
            self.last_hidden = Variable(torch.zeros(self.batch_size, 1, self.hidden_size)).cuda()
            for i in range(self.memory_slot):
                self.MS[:, i, i] = 1

        else:
            self.outputs = Variable(torch.zeros(self.max_len*self.batch_size, self.hidden_size))
            self.A = Variable(torch.zeros(self.batch_size, 1, self.hidden_size))
            self.E = Variable(torch.zeros(self.batch_size, self.max_entity, self.hidden_size))
            self.MS = Variable(torch.zeros(self.batch_size, self.memory_slot, self.rmc_memory_size))
            self.last_hidden = Variable(torch.zeros(self.batch_size, 1, self.hidden_size))
            for i in range(self.memory_slot):
                self.MS[:, i, i] = 1
    
    def regenerate(self):
        new_E = self.nonlinear(self.line(torch.sum(self.E, dim=1,keepdim=True)/self.E.size()[1])-self.line(self.E))
        z = self.nonlinear(self.EW(self.E) + self.EU(new_E))
        self.E = self.E * (1-z) + new_E*z
        
#        self.E = self.nonlinear(self.line(torch.sum(self.E, dim=1,keepdim=True)/self.E.size()[1])*2-self.line(self.E))

    def transpose_outputs(self, s):
        self.outputs = self.outputs[:s[0]*s[2]*s[1]].contiguous().view(s[1], s[0], s[2], self.hidden_size)
        self.outputs = self.outputs.transpose(0,1)
        self.outputs = self.outputs.contiguous().view(-1, self.hidden_size)
        
    def entity_mask(self, v):
        return v.ne(0).type(torch.float).unsqueeze(2)
    
    def forward(self, sent, pos, pos2, pos3, h, c, use_att=False, now=-1, l1=-1, l2=-1, use_graph=False, generate=False, use_transformer=False, transformer_layer=1, use_biLSTM=False, use_rmc=False):
#        print(sent.size())
        
#        sent = torch.cat((self.embed(sent), self.embed_pos(pos)), -1)   # one sentence a time 1*word_num*hidden_size
        mask_data = torch.cat((torch.ones(sent.size()[0], 1, dtype=torch.float).cuda(), sent.data.type(torch.float)), 1)
        non_mask = get_non_pad_mask(mask_data) 
        att_mask = get_attn_key_pad_mask(mask_data, mask_data)
        non_mask2 = get_non_pad_mask(sent.data) 
        att_mask2 = get_attn_key_pad_mask(sent.data, sent.data)
#        att_mask = None
        
        sent = self.embed(sent) + self.embed_pos(pos3)
        residual = sent
        
#        if use_graph:
#            eg = self.line_g(torch.gather(self.E, 1, pos2)) * self.entity_mask(pos3)  # one sentence a time 1*word_num*hidden_size
#            eg = self.Dropout(eg)
#        
#        if use_att:
#            sent = sent + self.memory_attention(q=self.query(residual), k=self.A, v=self.A, mask=None)[0]
#            sent_q = self.query(residual)
#            att = self.softmax(torch.bmm(self.key(self.A), sent_q.permute(0,2,1))/math.sqrt(self.hidden_size * 1.0))
#            att = torch.bmm(att.permute(0,2,1), self.value(self.A))
#            att = self.layer_norm(att)
#        sent = torch.cat((residual, eg, att), 2)
        
        if use_graph:
            sent = sent + self.line_g(torch.gather(self.E, 1, pos2)) * self.entity_mask(pos3)  # one sentence a time 1*word_num*hidden_size
            sent = self.Dropout(sent)
        
        if use_att:
#            sent = sent + self.memory_attention(q=self.query(residual), k=self.A, v=self.A, mask=None)[0]
            sent_q = self.query(residual)
            att = self.softmax(torch.bmm(self.key(self.A), sent_q.permute(0,2,1))/math.sqrt(self.hidden_size * 1.0))
            att = torch.bmm(att.permute(0,2,1), self.value(self.A))
            sent = self.layer_norm(sent + att)
            del att
        
        
        # add when using transformer while concat when lstm
        if use_transformer:
            # we assume that we use 3 layers of transformer at the most
            o, _ = self.transformer1(torch.cat((self.last_hidden, self.w_t(sent)), 1), non_mask, att_mask)
            if transformer_layer > 1:
                o, _ = self.transformer2(o, non_mask, att_mask)
            if transformer_layer > 2:
                o, _ = self.transformer3(o, non_mask, att_mask)
                
            o = o[:,1:,:]
            h, c = torch.max(o, dim=1, keepdim=True)
            self.last_hidden = h
            h = h.transpose(0,1)
            
        else:
            if use_rmc:
                o = None
                for i in range(sent.size()[1]):
                    n_mem = self.embed2rmc_mem(sent[:,i:i+1,:])
                    
                    MS_bef = self.MS
                    self.MS = torch.cat((self.MS, n_mem), 1)
                    MS_residual = self.rmc_layer(self.MS)[0] + self.MS
                    
                    self.MS = MS_residual
                    self.MS = self.layer_norm_rmc(self.MS)
                    self.MS = self.rmc_fc1(self.MS)
                    self.MS = self.rmc_fc2(self.MS)
                    self.MS = self.MS + MS_residual
                    
                    self.MS = self.MS[:, -self.memory_slot:, :]
                    
                    z = self.nonlinear(self.EW_rmc(MS_bef) + self.EU_rmc(self.MS))
                    self.MS = self.MS * (1-z) + MS_bef*z
                    
                    output_now = self.MS.contiguous().view(-1, 1, self.hidden_size)
                    if o is None:
                        o = output_now
                    else:
                        o = torch.cat((o, output_now), 1)
                h = o[:, -1, :]
                    
            elif use_biLSTM:
                o, (h,c) = self.b_lstm(sent, (h, c))
            else:
                o, (h,c) = self.lstm(sent, (h, c))
            
        
        
        if use_graph:
            new_E = torch.bmm(pos, o)
            z = self.nonlinear(self.EW2(self.E) + self.EU2(new_E))
            self.E = self.E * (1-z) + new_E*z
#            self.E = (self.E + torch.bmm(pos, o))/2.0
            if generate:
                self.regenerate()

        o = o.contiguous().view(-1, self.hidden_size);#print(self.outputs.size(), o.size(), self.output_pointer)                
        self.outputs[self.output_pointer:self.output_pointer+o.size()[0]] = o
        self.output_pointer += o.size()[0]
        del o
        
        if use_att:
            if use_biLSTM:
                 self.A = torch.cat((self.A, h.contiguous().view(self.num_layer, -1, self.batch_size, self.hidden_size).permute(2,0,1,3).contiguous().view(self.num_layer, self.batch_size, -1).transpose(0,1)), 1)
            else:
                self.A = torch.cat((self.A, h.transpose(0,1)), 1)
            
        return h, c
            
            
    
    def extract_relation(self, heads, tails, relas, mx, training=True, use_ca=False):
        h = torch.gather(self.outputs, 0, heads) # (mx*b) * hidden
        t = torch.gather(self.outputs, 0, tails) # (mx*b) * hidden
        
        if USE_CUDA:
            h = h.cuda()
            t = t.cuda()
            
        
        non_pad_mask = get_non_pad_mask(heads[:,0].contiguous().view(-1, mx))
        att_mask = get_attn_key_pad_mask(heads[:,0].contiguous().view(-1, mx), heads[:,0].contiguous().view(-1, mx))
        
        if use_ca:
            r = self.bili_ca(h.view(-1, self.hidden_size), t.view(-1, self.hidden_size)).view(-1 , mx,  self.hidden_size)
            r, _ = self.ca(r, non_pad_mask, att_mask)
            r = self.h2r(r)
        else:
            r = self.bili(h.view(-1, self.hidden_size), t.view(-1, self.hidden_size)).view(-1 , mx,  self.relation_size) # batch* mx_query * relation_size
        if training:
            return self.criterion(r.view(-1, self.relation_size), relas.view(-1)), r
        else:
            return torch.argmax(r, dim = 1)
    
        
class _CNN(nn.Module):
    def __init__(self, max_p, max_len, hidden_size, dropout, vocab, embedding_dim, relation_size, batch_size, window_size, final_channels, model_name):
        super(_CNN, self).__init__()
        self.name = model_name
        self.max_p = max_p
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_vocab = vocab
        self.embedding_dim = embedding_dim
        self.relation_size = relation_size
        self.batch_size = batch_size
        self.embed = nn.Embedding(self.num_vocab, int(embedding_dim), padding_idx=PAD_token)
#        self.embed_pos = nn.Embedding(4, int(embedding_dim), padding_idx=PAD_token) # 4 different tokens
        self.embed_pos = nn.Embedding(self.max_len, int(embedding_dim), padding_idx=PAD_token)

        self.kernel_size = window_size
        self.final_channels = final_channels
        self.out_channels = hidden_size
        self.in_channels  = hidden_size
        self.stride = 1
        self.padding = int((window_size-1)/2)
        self.cnn_1 = nn.Conv1d(self.in_channels, self.out_channels * 4, self.kernel_size, self.stride, self.padding)
        self.cnn_2 = nn.Conv1d(self.out_channels*4, self.out_channels*4, self.kernel_size, self.stride, self.padding)
        self.cnn_3 = nn.Conv1d(self.out_channels*4, self.out_channels*4, self.kernel_size, self.stride, self.padding)
        self.max_pooling = nn.MaxPool1d(self.kernel_size, stride=self.stride, padding=self.padding)
        self.relu = nn.ReLU()
#        self.maxPool = nn.MaxPool1d(window_size, 1, int((window_size-1)/2))
        self.line = nn.Linear(self.final_channels, self.hidden_size)
        self.bili = torch.nn.Bilinear(self.hidden_size, self.hidden_size, self.relation_size)
        self.criterion = nn.CrossEntropyLoss()
        self.outputs = None
        
    def clean_data(self):
        self.outputs = None
    
    def forward(self, sent, pos):
        s = sent.size()
        
        sent = (self.embed(sent) + self.embed_pos(pos)).permute(0,2,1)
        # batch * embedding_size * max_len 
        x = self.cnn_1(sent)
        x = self.max_pooling(x)
        x = self.relu(x)
        if self.name in ['cnn2', 'cnn3']:
            x = self.cnn_2(x)
            x = self.max_pooling(x)
            x = self.relu(x)

        if self.name in ['cnn3']:
            x = self.cnn_3(x)
            x = self.max_pooling(x)
            x = self.relu(x)

        
        # batch * final_channels * max_len
        x = self.line(x.permute(0,2,1))
        x = self.relu(x)
        
        self.outputs = x.contiguous().view(-1, self.hidden_size)
        
    def extract_relation(self, heads, tails, relas, mx, training=True, use_ca=False):
        h = torch.gather(self.outputs, 0, heads) # (mx*b) * hidden
        t = torch.gather(self.outputs, 0, tails) # (mx*b) * hidden
        
        if USE_CUDA:
            h = h.cuda()
            t = t.cuda()
            
                
        r = self.bili(h.view(-1, self.hidden_size), t.view(-1, self.hidden_size)).view(-1 , mx,  self.relation_size) # batch* mx_query * relation_size
                
        if training:
            return self.criterion(r.view(-1, self.relation_size), relas.view(-1)), r
        else:
            return torch.argmax(r, dim = 1)


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

class Transformer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(Transformer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
#        print(enc_input.size(), non_pad_mask.size(), slf_attn_mask.size())
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        if not non_pad_mask is None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
#        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
        
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
        
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -10000)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

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
