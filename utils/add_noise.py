import json
import types

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--files', nargs='+', help = 'files you wanna deal with')
args = parser.parse_args()

for file_name in args.files:
#file_name = 'data2_test.json'
	fp = open(file_name, 'r')
	raw_json = json.load(fp)
	fp.close()
			
	res = []
	num_d = int(len(raw_json)/2)-1
	
	for i,d in enumerate(raw_json[:num_d]):
		len_s1 = len(d['sents_with_ent'])
		len_s2 = len(raw_json[num_d+i]['sents_with_ent'])
		
		add_s = int(len_s1/2)
		for j in range(add_s, len_s1):
			for k in range(len(d['sents_with_ent'][j]['vertexSet'])):
				d['sents_with_ent'][j]['vertexSet'][k]['sent_id'] += len_s2
		
		added_sents = []
		for j in raw_json[num_d+i]['sents_with_ent']:
			j['vertexSet'] = []
			added_sents.append(j)
			
		d['sents_with_ent'] = d['sents_with_ent'][:add_s] + added_sents + d['sents_with_ent'][add_s:]

		res.append(d)
	raw_json = res
#        dic = {}
#        for j,s in enumerate(d['edgeSet']):
#            if 'matters' in s:
#                del raw_json[i]['edgeSet'][j]['matters']
#            else:
#                if type(s) == type([]):
#                    s = s[0]
#                s['ignore'] = 1
#                raw_json[i]['edgeSet'][j] = s
				
#        for j,s in enumerate(d['sents_with_ent']):
#            for index,v in enumerate(s['vertexSet']):
#                while(checkDepth(raw_json[i]['sents_with_ent'][j]['vertexSet'][index]['pos']) > 2):
#                    raw_json[i]['sents_with_ent'][j]['vertexSet'][index]['pos'] = raw_json[i]['sents_with_ent'][j]['vertexSet'][index]['pos'][0]
#                while(checkDepth(raw_json[i]['sents_with_ent'][j]['vertexSet'][index]['pos']) < 2):
#                    raw_json[i]['sents_with_ent'][j]['vertexSet'][index]['pos'] = [raw_json[i]['sents_with_ent'][j]['vertexSet'][index]['pos']]
#                raw_json[i]['sents_with_ent'][j]['vertexSet'][index]['pos'][0][-1] -= 1
#
#                
#                if args.hide == 'True':
#                    idx = -1
#                    if v['name'] in dic:
#                        idx = dic[v['name']]
#                    else:
#                        idx = len(dic) + 1
#                        dic[v['name']] = idx
#                    
#                    for k in range(v['pos'][0][0], v['pos'][0][1]+1):
#                        raw_json[i]['sents_with_ent'][j]['tokens'][k] = 'entity#' + str(idx)
			
			

	fp = open("N_"+file_name, 'w')
	json.dump(raw_json, fp)
	fp.close()
