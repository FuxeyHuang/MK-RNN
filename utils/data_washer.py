#!/usr/bin/python
import json

#file_name = './test_human_annotated_reformat.json'
ralas = {}
	
def rlaspp(a):
	if a in ralas:
		ralas[a] += 1
	else:
		ralas[a]  = 1


l = []
ee = []

wanted = []

#for i in range(0, 32):
#	a = str(i)
#	if i < 10:
#		a = '0' + a
#	
file_name = 'dev__.json' 
#	file_name = './test.json'
print("Washing " + file_name)

with open(file_name, 'r', encoding='utf-8') as fin:
	raw_json = json.load(fin)

	for d in raw_json:
		entity_sum = 0
		
		length = 0
		for s in d['sents_with_ent']:
			length += len(s['tokens'])

			for v in s['vertexSet']:
				entity_sum = max(entity_sum, v['vertex_id'])
			
		
		
		for e in d['edgeSet']:
			rlaspp(e['r'])
		
		l.append(length)
		ee.append(entity_sum)
		
#			if(length >= 250 and length <= 350 and entity_sum >= 8 and entity_sum <= 12):
#				
	
	def count(l, lim):
		res = 0
		for i in l:
			if i <= lim:
				res += 1
		
		return res
		
	
l_lim = [200, 250, 300, 350, 400, 500]
for i in l_lim:
	print("Articles below {}".format(i))
	print(count(l, i))

e_lim = [6,7,8,9,10,11,12,15,20,]
for i in e_lim:
	print("entities below {}".format(i))
	print(count(ee, i))

r = []
for i in ralas:
	r.append([i, ralas[i]])

def takeSecond(e):
	return -e[1]
	
r.sort(key=takeSecond)

print("the 10 most common relations:")
for i in r[:10]:
	print(i)
		
fp = open("len", 'w')
fp.write(str(l))
fp.close()

fp = open("ent", 'w')
fp.write(str(ee))
fp.close()

fp = open("rel", 'w')
fp.write(str(r))
fp.close()





wanted_relations = [a[0] for a in r[:100]]			
wanted = []

#print(wanted_relations)

#for i in range(0, 32):
#	a = str(i)
#	if i < 10:
#		a = '0' + a
#	

entity_pair = set()
entity_pair_dic = {}
useful_relation_data = []
cnt = 0
with open(file_name, 'r', encoding='utf-8') as fin:
	raw_json = json.load(fin)
			
	for d in raw_json:
		entity_sum = 0
		wanted_rel = 0
		
		length = 0
		sent_len_max =-100000
		entity_dic = {}
		for s in d['sents_with_ent']:
			sent_len_max = max(sent_len_max, len(s['tokens']))
			length += len(s['tokens'])
			for v in s['vertexSet']:
#				if not 'Q' in v:
#					continue
				entity_dic[v['vertex_id']] = v['real_name']
				entity_sum = max(entity_sum, v['vertex_id'])
			
		
		wr = 0
		useful_ent = {}
		new_edgeSet = []
		for e in d['edgeSet']:
			if e['r'] in wanted_relations:
				new_edgeSet.append(e)
				useful_ent[e['h']] = 0
				useful_ent[e['t']] = 0
				if not (isinstance(e['t'], int) and isinstance(e['h'], int)):
					print(e['t'], e['h'])
				if not (entity_dic[e['t']], entity_dic[e['h']]) in entity_dic:
					entity_pair.add((entity_dic[e['h']], entity_dic[e['t']]))
				entity_pair_dic[(entity_dic[e['h']], entity_dic[e['t']])] = e['r']
				wr += 1
		if len(useful_ent) >= 8:
			d['edgeSet'] = new_edgeSet
			useful_relation_data.append(d)
#		if(length >= 250 and length <= 350 and len(useful_ent) >= 8 and len(useful_ent) <= 12 and wr >= 2):
#			wanted.append(d)

entity_pair_l = list(entity_pair)
import random
random.shuffle(entity_pair_l)
print("Total entity pair : {}".format(len(entity_pair_l)))
test_pair = set(entity_pair_l[:int(len(entity_pair_l)/5)])
train_pair = set(entity_pair_l[int(len(entity_pair_l)/5):])

test_lim = 1500
test_data = []
train_data = []
random.shuffle(useful_relation_data)
c1 = 0
c2 = 0
c3 = 0

for d in useful_relation_data:
	entity_sum = 0
	wanted_rel = 0
	
	sent_len_max =-100000
	
	entity_dic = {}
	for s in d['sents_with_ent']:
		sent_len_max = max(sent_len_max, len(s['tokens']))
	

		for v in s['vertexSet']:
			entity_dic[v['vertex_id']] = v['real_name']
			entity_sum = max(entity_sum, v['vertex_id'])
		
	
	wr = 0
	useful_ent_train = {}
	useful_ent_test = {}
	test_edge = []
	train_edge = []
	for e in d['edgeSet']:
		rr = e['r'] + '-1'
		if (entity_dic[e['t']], entity_dic[e['h']]) in entity_pair_dic:
			rr = entity_pair_dic[(entity_dic[e['t']], entity_dic[e['h']])]
		re = {}
		re['h'] = e['t']
		re['t'] = e['h']
		re['r'] = rr
		
		if (entity_dic[e['h']], entity_dic[e['t']]) in test_pair or (entity_dic[e['t']], entity_dic[e['h']]) in test_pair:
			test_edge.append(e)
			test_edge.append(re)
			useful_ent_test[e['h']] = 0
			useful_ent_test[e['t']] = 0
			
			
		else:
			train_edge.append(e)
			train_edge.append(re)
			useful_ent_train[e['h']] = 0
			useful_ent_train[e['t']] = 0

		wr += 1
		#len(train_data) < train_lim and 
	if len(test_data) < test_lim and len(useful_ent_test) >= 8 and len(useful_ent_test) <= 12 and sent_len_max < 70 and len(d['sents_with_ent']) < 20:
		d['edgeSet'] = test_edge
		for e in train_edge:
			e['ignore'] = 0
			d['edgeSet'].append(e)
		test_data.append(d)
		c2 += 1
	elif len(useful_ent_train) >= 8 and len(useful_ent_train) <= 12 and sent_len_max < 70 and len(d['sents_with_ent']) < 20:
		d['edgeSet'] = train_edge
		train_data.append(d)
		c1 += 1
	c3 += 1

	

train_lim = 10000000000
if len(train_data) > train_lim:
	train_data = train_data[:train_lim]
test_lim = 1500
if len(test_data) > test_lim:
	test_data = test_data[:test_lim]

print("TOTAL available train data: {}\n".format(len(train_data)))
print("TOTAL available test data: {}\n".format(len(test_data)))


print(c1, c2, c3)
fp = open("dataXS_train.json", 'w')
fp1 = open("dataXS_test.json", 'w')
json.dump(train_data, fp)
fp.close()
json.dump(test_data, fp1)
fp1.close()

for d in train_data:
	sent_len_max =-100000
	
	for s in d['sents_with_ent']:
		sent_len_max = max(sent_len_max, len(s['tokens']))
	assert(sent_len_max < 70)

	
			
		

