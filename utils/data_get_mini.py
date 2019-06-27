import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--files', nargs='+', help = 'files you wanna deal with')
args = parser.parse_args()

files = []
for f in args.files:
    files.append(f + '_train')
    files.append(f + '_test')



for file_name in files:
#file_name = 'data2_test.json'
    fp = open(file_name+ '.json', 'r')
    raw_json = json.load(fp)
    fp.close()
    raw_json = raw_json[:int(len(raw_json)/10)]
    fp = open(file_name+'_mini.json', 'w')
    json.dump(raw_json, fp)
    fp.close()
