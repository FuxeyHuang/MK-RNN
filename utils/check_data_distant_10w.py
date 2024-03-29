#!/usr/bin/python
import json
import ast
import random


fp = open('distant_10w.json', 'r')
raw_json = json.load(fp)
fp.close()

out = []
for data in raw_json:
	now = {}
	sents_with_ent = []
	para_start = data['paras_start']
	edgeSet = []
	
	for s in data['sents']:
		hi = {}
		hi['tokens'] = s
		hi['vertexSet'] = []
		
		sents_with_ent.append(hi)
	
	lastVertex = {}
	cnt = 0
	for index, v in enumerate(data['vertexSet']):
		last = -1
		last_sent = -1
		last_pos  = -1
		
		for k in v:
			hi = k
			hi['text_name'] = k['name']
			hi['vertex_id'] = cnt
			
			
			sents_with_ent[k['sent_id']]['vertexSet'].append(hi)
			if last == -1 or (k['sent_id'] < last_sent) or (k['sent_id'] == last_sent and k['pos'][-1] < last_pos):
				last = cnt
				last_sent = k['sent_id']
				last_pos = k['pos'][-1]
			
			cnt += 1
		lastVertex[index] = last
	
	for e in data['edgeSet']:
		if len(e['r']) == 1:
			hi = {}
			hi['h'] = lastVertex[e['h']]
			hi['t'] = lastVertex[e['t']]
			hi['r'] = e['r'][0]
			
			edgeSet.append(hi)

	now['sents_with_ent'] = sents_with_ent
	now['para_start'] = para_start
	now['edgeSet'] = edgeSet
	out.append(now)

fp = open('distant_10w_reformat.json', 'w')
json.dump(out, fp)
fp.close()

		
		
	



     