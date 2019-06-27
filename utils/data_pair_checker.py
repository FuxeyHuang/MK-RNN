import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--files', nargs='+', help = 'files you wanna deal with')
args = parser.parse_args()

for file_name in args.files:
#file_name = 'data2_test.json'
	fp = open(file_name, 'r')
	#fp1 = open("distant_10w_test_data_debug.json", 'r')

	raw_json = json.load(fp)

	fp.close()
	def checkDepth(a):
		if isinstance(a,list):
			return checkDepth(a[0]) + 1
		return 0;
		
	dic = {}
	edge_all = 0
	edge_con = 0
	for i,d in enumerate(raw_json):
		word_dic = {}
		for j,s in enumerate(d['sents_with_ent']):
			for index,v in enumerate(s['vertexSet']):
				word_dic[v['vertex_id']] = v['name']
		for e in d['edgeSet']:
			edge_all += 1
			h = word_dic[e['h']]
			t = word_dic[e['t']]
			g = (h, t)
			
			if g in dic:
				
				if e['r'] != dic[g]:
					edge_con += 1
			dic[g] = e['r']
		
	print(edge_con, edge_all)
			
		
