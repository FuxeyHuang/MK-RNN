#!/usr/bin/python
from nltk.tokenize import sent_tokenize, word_tokenize

def clean_data(s):
	res = ""
	
	i = 0
	while(i < len(s)):
		if i + 2 < len(s) and s[i:i+3] == '@@ ':
			i = i+3
		res += s[i]
		i += 1
	return res
		
relation_type = ["chem_disease_marker/mechanism"   ,
"chem_disease_therapeutic"        ,
"chem_gene_increases_expression"  ,
"gene_disease_marker/mechanism"   ,
"chem_gene_increases_metabolic_processing"        ,
"chem_gene_decreases_expression"  ,
"chem_gene_increases_activity"    ,
"chem_gene_affects_response_to_substance" ,
"chem_gene_decreases_activity"    ,
"chem_gene_affects_transport"     ,
"chem_gene_increases_reaction"    ,
"chem_gene_decreases_reaction"    ,
"chem_gene_decreases_metabolic_processing"        ,
"gene_disease_therapeutic"        ]

import ast
f_e = open("entities_CTD", 'r')
ets = set(ast.literal_eval(f_e.read()))
b_s = set()

#mx = 0
#ccc = [0 for _ in range(15)]
for i in ets:
#	mx = max(mx, len(i.split()))
#	ccc[len(i.split())] += 1
	b_s.add(i.split()[0])
#print(ccc)

def l2s(a):
	res = a[0]
	for i in a[1:]:
		res += " " + i
	return res

all_data = []
ents_size = [0 for _ in range(100)]

for ff in range(12):
	file_name = "./shards/positive_50000_vocab_train" + ( "0" if ff < 10 else "") + str(ff)
	f = open(file_name, 'r')
	
	for i in f.readlines():
		now = clean_data(i)
		ln = now.split()
		
		e1 = [ln[2]]
		e2 = []
		es = 0
		for j in range(3, len(ln)):
			if ln[j][0].isdigit() or ln[j] in ["Gene", "Chemical", "Disease"]:
				for k in range(j, len(ln)):
					if ln[k] in ["Gene", "Chemical", "Disease"]:
						e2 = [ln[k+1]]
						for p in range(k+2, len(ln)):
							if ln[p][0].isdigit() or ln[p] in relation_type:
								es = p
								break
							e2.append(ln[p])
						break
				break
			e1.append(ln[j])
		e1 = l2s(e1)
		e2 = l2s(e2)
		
		
		para_st = -1
		for j in range(len(ln)):
			if ln[j] in relation_type:
				para_st = j+1
		
		if para_st == -1:
			continue
		
		para = l2s(ln[para_st:len(ln)])
		ls = sent_tokenize(para)
		
		article = {}
		article['sents_with_ent'] = []
		article['edgeSet'] = []
		
		ents = {}
		vertex_cnt = 1
		for s in ls:
			now = {}
			now['tokens'] = word_tokenize(s)
			now['tokens'][0] = now['tokens'][0].lower()
			now['vertexSet'] = []
			
			for j in range(len(now['tokens'])):
				if now['tokens'][j] in b_s:
					for k in range(4):
						ll = l2s(now['tokens'][j:j+k+1])
						if j+k+1 <= len(now['tokens']) and ll in ets:
							idn = vertex_cnt
							vertex_cnt += 1
							now['vertexSet'].append({
								'sent_id' : len(article['sents_with_ent']),
								'Q' : ll,
								'pos' : [[j, j+k]],
								'name': ll,
								'text_name': ll,
								'vertex_id': idn
							})
							ents[ll] = idn
							break
			article['sents_with_ent'].append(now)
		
		if (not e1 in ents) or (not e2 in ents):
			continue
		ents_size[len(ents)] += 1
		
		article['edgeSet'] = [{'h':ents[e1], 't':ents[e2], 'r':ln[para_st-1], 'matters':1}]
		for i in ents:
			article['edgeSet'].append([{'h':ents[i], 't':ents[e2], 'r':" "}])
		all_data.append(article)

import json
f.close()
fp = open('ctd_all.json', 'w')
json.dump(all_data, fp)
fp.close()

cnt = 0
for i in range(1, len(ents_size)):
	if(i > 30 and ents_size[i] == 0):
		break
	cnt += ents_size[i]
	print(i, cnt)
	
		
## finding entites
#for ff in range(12):
#	file_name = "./shards/positive_50000_vocab_train" + ( "0" if ff < 10 else "") + str(ff)
#	f = open(file_name, 'r')
#	
#	for i in f.readlines():
#		now = clean_data(i)
#		ln = now.split()
#		e1 = [ln[2]]
#		e2 = []
#		es = 0
#		for j in range(3, len(ln)):
#			if ln[j][0].isdigit() or ln[j] in ["Gene", "Chemical", "Disease"]:
#				for k in range(j, len(ln)):
#					if ln[k] in ["Gene", "Chemical", "Disease"]:
#						e2 = [ln[k+1]]
#						for p in range(k+2, len(ln)):
#							if ln[p][0].isdigit() or ln[p] in relation_type:
#								es = p
#								break
#							e2.append(ln[p])
#						break
#				break
#			e1.append(ln[j])
#		
#		t1 = False
#		t2 = False
#		for j in range(es, len(ln)):
#			if j + len(e1) <= len(ln):
#				if e1 == ln[j : j+len(e1)]:
#					t1 = True
#			if j + len(e2) <= len(ln):
#				if e2 == ln[j : j+len(e2)]:
#					t2 = True
#		e11 = e1[0]
#		e22 = e2[0]
#		for l in range(1, len(e1)):
#			e11 += " " + e1[l]
#		for l in range(1, len(e2)):
#			e22 += " " + e2[l]
#		
#		if t1 and t2:
#			print(e11, e22)
#			ets.add(e11)
#			ets.add(e22)
#			
#print("{} Entities got!".format(len(list(ets))))
#f_e.write(str(list(ets)))
		
	