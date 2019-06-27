import json

fp = open("distant_10w_train_data.json", 'r')
fp1 = open("distant_10w_test_data.json", 'r')

raw_json = json.load(fp)
raw_json2= json.load(fp1)
fp.close()
fp1.close()

import random
random.shuffle(raw_json)
random.shuffle(raw_json2)
raw_json = raw_json[:1000]
raw_json2= raw_json2[:100]

fp = open("distant_10w_train_data_debug.json", 'w')
fp1= open("distant_10w_test_data_debug.json", 'w')
json.dump(raw_json, fp)
fp.close()
json.dump(raw_json2, fp1)
fp1.close()

