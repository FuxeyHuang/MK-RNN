from utils.config import *
from models.mem2seq import CNN, Memnet
import torch
import numpy as np
import logging
from tqdm import tqdm
from torch.autograd import Variable
from utils.data_loader import prepare_data_seq # change this to change data source
import argparse
import sklearn.metrics
import random

print("Checking CUDA usage")
print(USE_CUDA)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'lstm', help = 'name of the model')
parser.add_argument('--use_transformer', type = str, default = 'False', help = 'whether use transformer as your decoder')
parser.add_argument('--use_bilstm', type = str, default = 'False', help = 'whether use bilstm as your decoder')
parser.add_argument('--use_rmc', type = str, default = 'False', help = 'whether use rmc as your decoder')
parser.add_argument('--use_ca', type = str, default = 'False', help = 'whether use context-aware as your extractor')
parser.add_argument('--transformer_layer', type = int, default = 1, help = 'number of layers of transformer')
parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate')
parser.add_argument('--debug', type = str, default = 'False', help = 'using debugging data')
parser.add_argument('--project_id', type = str, default = 'lab', help = 'codename for dumping and retrieving')
parser.add_argument('--batch_size', type = int, default = 32, help = 'batch_size for training')
parser.add_argument('--batch_count', type = int, default = 1, help = 'how many batches counted before loss back_propegate')
parser.add_argument('--cuda_id', type = int, default = 0, help = 'cuda device that you want to use')
parser.add_argument('--data_id', type = str, default = "data1", help = 'data type you want to run')
parser.add_argument('--extra_name', type = str, default = "", help = 'suffix for you model')
parser.add_argument('--pick_last', type = str, default = "False", help = 'if you want to pick the trained model')

args = parser.parse_args()
b_c = args.batch_count

transformer_suffix = ""
if args.use_transformer == "True":
	transformer_suffix = "_t" + str(args.transformer_layer)
	
blstm_suffix = ""
if args.use_bilstm == "True":
	blstm_suffix = "_b"

rmc_suffix = ""
if args.use_rmc == "True":
	rmc_suffix = "_r"
	
ca_suffix = ""
if args.use_ca == "True":
	ca_suffix = "_c"

torch.cuda.set_device(args.cuda_id)
batch_size = args.batch_size
filter_NA = False

#print("STAGE 1")
print(args.model_name)
print(args.debug)
train, lang_word, lang_rela, max_len, max_p = prepare_data_seq(batch_size = batch_size, Debug = args.debug, data_type=args.data_id)
test, lang_word_, lang_rela_, max_len_, max_p_ = prepare_data_seq(batch_size = batch_size, Debug = args.debug, training=False, lw=lang_word, lr = lang_rela, data_type=args.data_id)


if args.model_name in ['cnn1', 'cnn2', 'cnn3']:
	model = CNN(hidden_size= 50, max_len= max_len,
				max_p= max_p, lang=lang_word,
				path="",lr=args.lr, n_layers=3, dropout=0.5, relation_size = lang_rela.n_words, batch_size = batch_size, window_size = 3, model_name=args.model_name)
elif args.model_name in ['lstm', 'mem', 'g', 'mem_g', 'g_gcn', 'mem_g_gcn', 'after_gcn']:
	
	#
	use_transformer = False
	if args.use_transformer == 'True':
		use_transformer = True
	
	use_biLSTM = False
	if args.use_bilstm == 'True':
		use_biLSTM = True
	
	use_rmc = False
	if args.use_rmc == 'True':
		use_rmc = True
	
	use_ca = False
	if args.use_ca == 'True':
		use_ca = True

	
	model = Memnet(hidden_size= 256, max_len= max_len,
									max_p= max_p, lang=lang_word,
									path="",lr=args.lr, n_layers=3, dropout=0.5, relation_size = lang_rela.n_words, batch_size = batch_size, model_name=args.model_name, use_transformer=use_transformer, embedding_dim=50, transformer_layer=args.transformer_layer, use_biLSTM = use_biLSTM, use_rmc=use_rmc, use_ca=use_ca)

else:
	raise SystemExit('It failed with an unknown model name!')


def get_xy(overall, total_recall):
#	if total_recall == 0:
#		return [], []
	overall.sort(reverse = True)
	pr_y = []
	pr_x = []
	correct = 0
	
	for i, item in enumerate(overall):
		correct += item[1]
		pr_y.append(float(correct) / (i + 1))
		pr_x.append(float(correct) / total_recall)
	return pr_x, pr_y

class_NA = 1
def get_results(pred, target, count_NA=True, testing=False, filter_NA=False):
	
	p = pred.data.cpu().numpy()
	t = [a[2] for a in target]
	ig= [a[3] for a in target]
	
	overall = []
	total_recall = 0
	sum_NA = 4
	for i in range(p.shape[0]):    
		if filter_NA and t[i] == class_NA and sum_NA > 0:
			sum_NA -= 1
		
		for j in range(p.shape[1]):
			if j == class_NA:
				continue
			
			if j == t[i]:
				if ig[i] == 0:
					total_recall += 1
					overall.append([p[i][j], 1])
			else:
				overall.append([p[i][j], 0])
	
	pr_x, pr_y = get_xy(overall, total_recall)
	
	if len(pr_x) == 0:
		return 0,0,[],0
	auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
	
	possi = 100
	if testing:
		possi=1000
	if(random.randint(1, possi) == 7):
		print((np.argmax(p, axis = 1)))
		print(t)
		print("\n\n")
	
	if count_NA:
		cnt = 0
		cor = 0
		for i in range(len(t)):
			if ig[i] == 0 or t[i] == class_NA:
				cnt += 1
				cor += (np.argmax(p[i]) == t[i])
	else:
		cnt = 0
		cor = 0
		for i in range(len(t)):
			if ig[i] == 0:
				cnt += 1
				cor += (np.argmax(p[i]) == t[i])
	accuracy = 1.0*cor/cnt;
		
	return auc, accuracy, overall, total_recall, cor

def get_batch_results(pred, target, count_NA=True, testing=False, filter_NA=False):
	tot = len(target)
	auc = 0
	acc = 0
	overall = []
	total_recall = 0
	cor_all = 0
	for i in range(tot):
		u, c, o_now, r_now, cor_now = get_results(pred[i, :len(target[i])], target[i], count_NA, testing, filter_NA)
		auc += u
		acc += c
		cor_all += cor_now
		
		overall += o_now
		total_recall += r_now
	
	auc /= tot
	acc /= tot
	
	
	return auc, acc, overall, total_recall, cor_all
		

dump_path = './dumped_projects/{}_{}'.format(args.project_id, args.data_id)
model_save_path = dump_path + '/{}{}{}{}{}{}'.format(args.model_name, args.extra_name, transformer_suffix, blstm_suffix, rmc_suffix, ca_suffix)
log_f = open(model_save_path + "/test_log.txt", 'w')

assert os.path.exists(model_save_path)
if args.pick_last == "True":
	print("Trying to load previous model")
	if os.path.exists(model_save_path + '/best_auc'):
		model.load_state_dict(torch.load(model_save_path + '/best_auc'), strict=False)
		print(model_save_path + '/best_auc' + " Successfully loaded")


print("Starting validation:")
model.eval()
avg_auc = 0
avg_acc = 0
Navg_acc = 0
cnt = 0
o_all = []
r_all = 0
corret = 0
for i, data in enumerate(test):    
	if len(data[4]) != batch_size:
		continue
	loss, r = model.train_batch(input_batches=Variable(data[0]),
										input_lengths=Variable(data[2]),
										target_batches=data[4],
										reset=(i==0))
	auc,acc,o_now,r_now,cor_now = get_batch_results(r, data[4], True, True, filter_NA=filter_NA)
	_,Nacc,_,_,_ = get_batch_results(r, data[4], False, True, filter_NA=filter_NA)
	
	o_all += o_now
	r_all += r_now
	avg_auc += auc
	avg_acc += acc
	Navg_acc += Nacc
	cnt += 1
	corret += cor_now

avg_auc /= 1.0*cnt
avg_acc /= 1.0*cnt
Navg_acc /= 1.0*cnt

print("Test average auc: {}".format(avg_auc))
print("Test Correct datas : {}\n".format(corret))
print("Test average & Non-NA acc: {} {}\n\n".format(avg_acc, Navg_acc))
log_f.write("Test average : {}\n".format(avg_auc))
log_f.write("Test Correct datas : {}\n".format(corret))
log_f.write("Test average & Non-NA acc: {} {}\n\n\n".format(avg_acc, Navg_acc))
log_f.flush() 

x, y = get_xy(o_all, r_all)
np.save(model_save_path+'/best_auc_x.npy', np.array(x))
np.save(model_save_path+'/best_auc_y.npy', np.array(y))
 