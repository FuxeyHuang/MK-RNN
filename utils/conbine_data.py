import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_files', nargs='+', help = 'files you wanna input')
parser.add_argument('--out_files', nargs='+', help = 'files you wanna output')
parser.add_argument('--sent_lim', type=int, default=64, help = 'sent limit you wanna set')
args = parser.parse_args()

in_files = []
for i in args.in_files:
    in_files.append('data'+i+'_test.json')
    in_files.append('data'+i+'_train.json')
output_files = []
for i in args.out_files:
    output_files.append('data'+i+'_test.json')
    output_files.append('data'+i+'_train.json')

for index, file_name in enumerate(in_files):
    fp = open(file_name, 'r')
    raw_json = json.load(fp)
    fp.close()
    
        
    new_json = []
    for i,d in enumerate(raw_json):
        new_d = {}
        new_d['sents_with_ent'] = []
        new_d['para_start'] = d['para_start']
        new_d['edgeSet'] = d['edgeSet']
        
        last_stop = -1
        acu = 0
        mx_sent = 0
        for j,s in enumerate(d['sents_with_ent']):
            acu += len(d['sents_with_ent'][j]['tokens'])
            mx_sent = max(mx_sent, acu)
            if j == len(d['sents_with_ent']) -1 or acu + len(d['sents_with_ent'][j+1]['tokens']) > args.sent_lim:
                acu = 0
                now = {}
                
                now['tokens'] = []
                now['vertexSet'] = []
                for k in range(last_stop+1, j+1):
                    hi = d['sents_with_ent'][k]['vertexSet']
                    for p in range(len(hi)):
                        hi[p]['sent_id'] = len(new_d['sents_with_ent'])
                        hi[p]['pos'][0][0] += len(now['tokens'])
                        hi[p]['pos'][0][1] += len(now['tokens'])
                    now['vertexSet'] += hi
                    now['tokens'] += d['sents_with_ent'][k]['tokens']
#                    for p in range(len(now['vertexSet'])):
#                        
#                        print(now['tokens'][now['vertexSet'][p]['pos'][0][0]: now['vertexSet'][p]['pos'][0][1]+1])

                new_d['sents_with_ent'].append(now)
                
                
                last_stop = j
        new_json.append(new_d)
        
        print("number of sent:{}, number of words:{}".format(len(new_d['sents_with_ent']), mx_sent))
        

            
#                idx = -1
#                if v['name'] in dic:
#                    idx = dic[v['name']]
#                else:
#                    idx = len(dic) + 1
#                    dic[v['name']] = idx
#                
#                for k in range(v['pos'][0][0], v['pos'][0][1]+1):
#                    raw_json[i]['sents_with_ent'][j]['tokens'][k] = 'entity#' + str(idx)
            
            

    fp = open(output_files[index], 'w')
    json.dump(new_json, fp)
    fp.close()
