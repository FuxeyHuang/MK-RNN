#!/usr/bin/python

import json
fp = open('ctd_all.json', 'r')
raw_data = json.load(fp)

e_min = 8
e_max = 16

f_d = [d for d in raw_data if len(d['edgeSet']) >= e_min and len(d['edgeSet']) <= e_max]
#res = []
pair_counter = {}

for d in f_d:
	n_e = [d['edgeSet'][0]]
	now = (d['edgeSet'][0]['h'], d['edgeSet'][0]['t'])
	
	if now in pair_counter:
		pair_counter[now] += 1
	else:
		pair_counter[now] = 1
		
	
#	for i in range(1, len(d['edgeSet'])):
#		if not( (d['edgeSet'][i]['h'], d['edgeSet'][i]['t']) == (d['edgeSet'][0]['h'], d['edgeSet'][0]['t'])):
#			n_e.append(d['edgeSet'][i])
#	d['edgeSet'] = n_e
#	res.append(d)

num_total_pair = len(f_d)
num_test_pair = int(num_total_pair/10)
test_pair = set()
for i in pair_counter:
	if num_test_pair - pair_counter[i] > 0:
		test_pair.add(i)
		num_test_pair -= pair_counter[i]

test_set = []
train_set = []

for d in f_d:
	n_e = [d['edgeSet'][0]]
	now = (d['edgeSet'][0]['h'], d['edgeSet'][0]['t'])	
	
	for i in range(1, len(d['edgeSet'])):
		d['edgeSet'][i] = d['edgeSet'][i][0]
		if not ( (d['edgeSet'][i]['h'], d['edgeSet'][i]['t']) == now):
			n_e.append(d['edgeSet'][i])
	d['edgeSet'] = n_e
	
	if now in test_pair:
		test_set.append(d)
	else:
		train_set.append(d)

print("{}/{} train & test data collected.".format(len(train_set), len(test_set)))

f1 = open("ctd_train_1.json", 'w')
f2 = open('ctd_test_1.json', 'w')
json.dump(train_set, f2)
json.dump(test_set, f1)
f1.close()
f2.close()

			
			
		