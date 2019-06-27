#!/usr/bin/python
import json
import ast

fp = open("washed_data.json", 'r')
raw_json = json.load(fp)
fp.close()

fp = open("rel", 'r')
rel = ast.literal_eval(fp.read())
fp.close()

wanted_relations = [a[0] for a in rel[:100]]

rl = {}
for d in raw_json:
	n_e = []
	for e in d['edgeSet']:
		if e['r'] in wanted_relations:
			n_e.append(e)
	d['edgeSet'] = n_e
fp = open('washed_data_r.json', 'w')
json.dump(raw_json, fp)
fp.close()
#print(len(rl))