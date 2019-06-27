import json
import types

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--files', nargs='+', help = 'files you wanna deal with')
parser.add_argument('--hide', type=str, default='True', help = 'whether you want to hide your entity')
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
        
#    res = []
    for i,d in enumerate(raw_json):
#        n_e = [d['edgeSet'][0]]
#        for i in range(1, len(d['edgeSet'])):
#            print(d['edgeSet'][i])
#            if not( (d['edgeSet'][i]['h'], d['edgeSet'][i]['t']) == (d['edgeSet'][0]['h'], d['edgeSet'][0]['t'])):
#                n_e.append(d['edgeSet'][i])
#            else:
#                print("FOund")
#        d['edgeSet'] = n_e
#        res.append(d)
#    raw_json = res
    
    
#        dic = {}
#        for j,s in enumerate(d['edgeSet']):
#            if 'matters' in s:
#                del raw_json[i]['edgeSet'][j]['matters']
#            else:
#                if type(s) == type([]):
#                    s = s[0]
#                s['ignore'] = 1
#                raw_json[i]['edgeSet'][j] = s
                
                
        for j,s in enumerate(d['sents_with_ent']):
            for index,v in enumerate(s['vertexSet']):
                while(checkDepth(raw_json[i]['sents_with_ent'][j]['vertexSet'][index]['pos']) > 2):
                    raw_json[i]['sents_with_ent'][j]['vertexSet'][index]['pos'] = raw_json[i]['sents_with_ent'][j]['vertexSet'][index]['pos'][0]
                while(checkDepth(raw_json[i]['sents_with_ent'][j]['vertexSet'][index]['pos']) < 2):
                    raw_json[i]['sents_with_ent'][j]['vertexSet'][index]['pos'] = [raw_json[i]['sents_with_ent'][j]['vertexSet'][index]['pos']]
                raw_json[i]['sents_with_ent'][j]['vertexSet'][index]['pos'][0][-1] -= 1

                
                if args.hide == 'True':
                    idx = -1
                    if v['name'] in dic:
                        idx = dic[v['name']]
                    else:
                        idx = len(dic) + 1
                        dic[v['name']] = idx
                    
                    for k in range(v['pos'][0][0], v['pos'][0][1]+1):
                        raw_json[i]['sents_with_ent'][j]['tokens'][k] = 'entity#' + str(idx)
            
            

    fp = open(file_name, 'w')
    json.dump(raw_json, fp)
    fp.close()
