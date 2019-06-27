import json
import types
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--files', nargs='+', help = 'files you wanna deal with')
parser.add_argument('--lim', type=int, help = 'limit you want to use')
args = parser.parse_args()

for file_name in args.files:
#file_name = 'data2_test.json'
	fp = open(file_name, 'r')
	#fp1 = open("distant_10w_test_data_debug.json", 'r')

	raw_json = json.load(fp)

	fp.close()
		
	res1 = []
	res2 = []
	
	appears = [0 for _ in range(10000)]
	lim = args.lim
	cnt = 0
	
	na_rat = 0.0
	ignore_rat = 0.0
	
	for i,d in enumerate(raw_json):
#		n_e = [d['edgeSet'][0]]
		id2word = {}
		word2id = {}
		
		for k in d['sents_with_ent']:
			for l in k['vertexSet']:
				id2word[l['vertex_id']] = l['text_name']
				word2id[l['text_name']] = l['vertex_id']
		ents = {}

		ig_num = 0
		for j in d['edgeSet']:
			ents[id2word[j['h']]] = 0
			ents[id2word[j['t']]] = 0
			if 'ignore' in j:
				ig_num += 1
		print(len(d['edgeSet'])/(len(ents)*(len(ents)-1)))
		
		if len(ents) > lim:
			cnt += 1
			na_rat += 1 - len(d['edgeSet'])/(len(ents)*(len(ents)-1))
			ignore_rat += 1 - (len(d['edgeSet']) - ig_num)/(len(ents)*(len(ents)-1))
			
			res1.append(d)
		else:
			res2.append(d)
#			print(d['edgeSet'][i])
#			if not( (d['edgeSet'][i]['h'], d['edgeSet'][i]['t']) == (d['edgeSet'][0]['h'], d['edgeSet'][0]['t'])):
#				n_e.append(d['edgeSet'][i])
#			else:
#				print("FOund")
		
		
#		for j in ents:
#			for k in d['sents_with_ent']:
#				for l in k['vertexSet']:
#					if l['text_name'] == j:
#						ents[j] += 1
#		
#		tot = 0
#		singles = 0
#		for j in d['edgeSet']:
#			if not 'ignore' in j:
#				tot += 1
#				if ents[id2word[j['h']]] == 1 and ents[id2word[j['t']]] == 1:
#					singles += 1
##		print(singles/tot)
#		for j in ents:
#			appears[ents[j]] += 1
#		
#		d1 = {}
#		d2 = {}
#		d1['sents_with_ent'] = d['sents_with_ent']
#		d2['sents_with_ent'] = d['sents_with_ent']
#		d1['edgeSet'] = []
#		d2['edgeSet'] = []
#		
#		u1 = 0
#		u2 = 0
#		
#		for j in (d['edgeSet']):
##			d1['edgeSet'].append(j.copy())
##			d2['edgeSet'].append(j.copy())
#			if not 'ignore' in j:
#				if  ents[id2word[j['h']]] != 1 or ents[id2word[j['t']]] != 1:
#
#					d1['edgeSet'].append(j.copy())
#					u1 = 1
#
#					assert not 'ignore' in d1['edgeSet'][-1]
#				else:
#					d2['edgeSet'].append(j.copy())
#					u2 = 1
#				
#		
##		for jj, j in enumerate(d1['edgeSet']):
##			if not 'ignore' in j:
###				print(j)
##				if ents[id2word[j['h']]] == 1 and ents[id2word[j['t']]] == 1:
##					d1['edgeSet'][jj]['ignore'] = 0
##				else:
##					u1 = 1
##		print("\n")
#		
#		
##		print("\n\n")
#		if u1 == 1:
##			for j in (d1['edgeSet']):
##				if not 'ignore' in j:
##					print(j)
#			res1.append(d1)
#		
#		if u2 == 1:
#			res2.append(d2)
#		d['edgeSet'] = n_e
#		res.append(d)
#	raw_json = res
	print("{} counted under limit {}".format(cnt, lim))	
	print(cnt/1500, na_rat/cnt, ignore_rat/cnt)
	
	random.shuffle(res1)
	fp = open("difficult" + file_name, 'w')
	json.dump(res1[:50], fp)
	fp.close()
	print("{} difficult data got".format(len(res1)))
	fp = open("easy" + file_name, 'w')
	json.dump(res2, fp)
	fp.close()
	print("{} easy data got".format(len(res2)))
