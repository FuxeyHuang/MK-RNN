#!/usr/bin/python
import json
import ast
import random

fp = open("washed_data_r.json", 'r')
raw_json = json.load(fp)
fp.close()

#sent_size = len(raw_json)
#validation_size = sent_size/10

#random.shuffle(raw_json)
#ft = open('mini_test_length_250_350_top100_relation.json', 'w')
#json.dump(raw_json[:1000], ft)
#ft.close()
#
#ft = open('test_length_250_350_top100_relation.json', 'w')
#json.dump(raw_json[validation_size:], ft)
#ft.close()


#rl = {}
#useful_ent_min = 10000
#useful_ent_max = 0
#length_min = 1000000
#length_max = 0
#
#new_json = []
#cnt30 = 0
#cnt50 = 0
#cnt20 = 0
#cnt10 = 0
#cntmx = 0
#cntmn = 10000
#
#lcnt30 = 0
#lcnt40 = 0
#lcnt50 = 0
#lcnt60 = 0
#lcnt70 = 0
#lcnt80 = 0
#lmax = 0
#
#cnt = 0
#for d in raw_json:
#    n_e = []
#    useful_ent = {}
#
#    for e in d['edgeSet']:
#        rl[e['r']] = 0
#        useful_ent[e['h']] = 0
#        useful_ent[e['t']] = 0
#    useful_ent_min = min(useful_ent_min, len(useful_ent))
#    useful_ent_max = max(useful_ent_max, len(useful_ent))
#    if(len(useful_ent) > 10):
#        continue
#        
#    if len(d['edgeSet']) == 0:
#        assert(False)
#    
#    length = 0
#    l = len(d['sents_with_ent'])
#    if l < 10:
#        cnt10 += 1
#    if l < 20:
#        cnt20 += 1
#    if l < 30:
#        cnt30 += 1
#    cntmx = max(cntmx, l)
#    cntmn = min(cntmn, l)
#    
#    lmx = 0
#    for s in d['sents_with_ent']:
##        length += len(s['tokens'])
#        lmx = max(lmx, len(s['tokens']))
##    length_min = min(length_min, length)
##    length_max = max(length_max, length)
#    if lmx < 30:
#        lcnt30 += 1
#    if lmx < 40:
#        lcnt40 += 1
#    if lmx < 50:
#        lcnt50 += 1
#    if lmx < 60:
#        lcnt60 += 1
#    if lmx < 70:
#        lcnt70 += 1
#    if lmx < 80:
#        lcnt80 += 1
#    lmax = max(lmx, lmax)
#    
#    if lmx < 70 and l < 20:
#        cnt += 1
#        new_json.append(d)
#    
#
#print("For training data: sentence size from {} to {}".format(cntmn, cntmx))
#print("<=10: {}".format(cnt10))
#print("<=20: {}".format(cnt20))
#print("<=30: {}\n".format(cnt30))
#print("For training data: words size up to {}".format(lmax))
#print("<=30: {}".format(lcnt30))
#print("<=40: {}".format(lcnt40))
#print("<=50: {}".format(lcnt50))
#print("<=60: {}".format(lcnt60))
#print("<=70: {}".format(lcnt70))
#print("<=80: {}".format(lcnt80))
#
#print("\n\nTotal get {}".format(cnt))
#

#print("Total rel {}".format(len(rl)))
fp = open('distant_10w_view.json', 'r')
#fp2 = open('test_length_250_350_top100_relation_s20_w70_r_seperate.json', 'r')
raw_json = json.load(fp)
#raw_json2 = json.load(fp2)
fp.close()
print(raw_json[0].keys())
#fp2.close()
#import random
#random.shuffle(raw_json)
#random.shuffle(raw_json2)
#train = raw_json[:15000]
#test = raw_json2[:1500]

#pairs = set()
#for d in raw_json:
#    dic = {}
#    for s in d['sents_with_ent']:
#        for v in s['vertexSet']:
#            dic[v['vertex_id']] = v['Q']
#            
#    for e in d['edgeSet']:
#        pairs.add((dic[e['h']], dic[e['t']]))
#        
#for d in raw_json2:
#    dic = {}
#    for s in d['sents_with_ent']:
#        for v in s['vertexSet']:
#            dic[v['vertex_id']] = v['Q']
#            
#    for e in d['edgeSet']:
#        pairs.add((dic[e['h']], dic[e['t']]))
#
#pairs = list(pairs)
#test_pair = set(pairs[:int(len(pairs)/10)])
#
#train = []
#test = []
#
#
#
#for d in raw_json:
#    dic = {}
#    for s in d['sents_with_ent']:
#        for v in s['vertexSet']:
#            dic[v['vertex_id']] = v['Q']
#    
#    test_edges = []
#    train_edges = []
#    
#    for e in d['edgeSet']:
#        now = (dic[e['h']], dic[e['t']])
#        if now in test_pair:
#            test_edges.append(e)
#        else:
#            train_edges.append(e)
#
#
#    if len(test_edges)>3:
#        now = d
#        now['edgeSet'] = test_edges
#        test.append(now)
#        
#    if len(train_edges)>0:
#        now = d
#        now['edgeSet'] = train_edges
#        train.append(now)
#        
#for d in raw_json2:
#    dic = {}
#    for s in d['sents_with_ent']:
#        for v in s['vertexSet']:
#            dic[v['vertex_id']] = v['Q']
#    
#    test_edges = []
#    train_edges = []
#    
#    for e in d['edgeSet']:
#        now = (dic[e['h']], dic[e['t']])
#        if now in test_pair:
#            test_edges.append(e)
#        else:
#            train_edges.append(e)
#
#
#    if len(test_edges)>3:
#        now = d
#        now['edgeSet'] = test_edges
#        test.append(now)
#        
#    if len(train_edges)>0:
#        now = d
#        now['edgeSet'] = train_edges
#        train.append(now)
    
#fp = open('train_length_250_350_top100_relation_s20_w70_r_seperate_mini.json', 'w')
#fp2 = open('test_length_250_350_top100_relation_s20_w70_r_seperate_mini.json', 'w')
#
#
##val_size = int(cnt / 10)
#
#json.dump(train, fp)
#json.dump(test, fp2)
#fp.close()
#fp2.close()
#print("train_size: {}".format(len(train)))
#print("test_size: {}".format(len(test)))
#
#print("For this data:")
#print("There are {} articles".format(len(raw_json)))
#print("article length is from {} to {}".format(length_min, length_max))
#print("entity size is from {} to {}".format(useful_ent_min, useful_ent_max))



     