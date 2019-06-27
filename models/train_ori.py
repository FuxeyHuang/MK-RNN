from utils.config import *
from models.mem2seq import Mem2Seq, Lstm

import numpy as np
import logging
from tqdm import tqdm
from torch.autograd import Variable
from utils.data_loader import prepare_data_seq
import argparse
import sklearn.metrics


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'lstm', help = 'name of the model')
parser.add_argument('--debug', type = str, default = 'False', help = 'using debugging data')
parser.add_argument('--codename', type = str, default = 'lab', help = 'codename for dumping and retrieving')
args = parser.parse_args()

train, lang_word, lang_rela, max_len, max_p = prepare_data_seq(batch_size = 1, Debug = args.debug)
test, lang_word_, lang_rela_, max_len_, max_p_ = prepare_data_seq(batch_size = 1, Debug = args.debug, training=False)

if args.model_name == 'mem2seq':
	model = Mem2Seq(hidden_size= 100, max_len= max_len,
									max_p= max_p, lang=lang_word,
									path="",lr=0.001, n_layers=3, dropout=0.0, relation_size = lang_rela.n_words)
elif args.model_name == 'lstm':
	model = Lstm(hidden_size= 100, max_len= max_len,
				max_p= max_p, lang=lang_word,
				path="",lr=0.001, n_layers=3, dropout=0.0, relation_size = lang_rela.n_words)

test_epoch = 1
train_batch = 10

def get_results(pred, target):
	p = pred.data.cpu().numpy()
	t = [a[2].cpu().numpy()[0] for a in target]
	
	overall = []
	total_recall = 0
	for i in range(p.shape[0]):
		if(t[i] != 0):
			total_recall += 1
			
		for j in range(p.shape[1]):
			if j == t[i]:
				overall.append([p[i][j], 1])
			else:
				overall.append([p[i][j], 0])
	
	overall.sort(reverse = True)
	correct = 0
	pr_y = []
	pr_x = []
	
	for i, item in enumerate(overall):
		correct += item[1]
		pr_y.append(float(correct) / (i + 1))
		pr_x.append(float(correct) / total_recall)
	auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
	
	accuracy = float((np.argmax(p, axis = 1) == t).sum())/p.shape[0]
	
	return auc, accuracy


	
	
	return 0

avg_best = 0
dump_path = ""
if args.codename != 'lab':
	dump_path = './dumped_models/{}'.format(args.codename)
	os.mkdir(dump_path)
		

for epoch in range(300):
		logging.info("Epoch:{}".format(epoch))
		# Run the train function
		# pbar = tqdm(enumerate(train),total=len(train))
		avg_auc = 0
		avg_acc = 0
		cnt = 0
		for i, data in enumerate(train):
				model.train()
				loss, r = model.train_batch(input_batches=Variable(data[0]),
													input_lengths=Variable(data[2]),
													target_batches=data[4],
													reset=(i==0))
				print("Batch {} finished: loss = {}".format(i, loss))
				auc,acc = get_results(r, data[4])
				avg_auc += auc
				avg_acc += acc
				print("auc: {}, acc: {}\n".format(auc, acc))
				cnt += 1
		
		avg_auc /= 1.0*cnt
		avg_acc /= 1.0*cnt
		print("Train auc after {} epoch".format(avg_auc))
		print("Train acc after {} epoch\n\n".format(avg_acc))
		
		
		if((i+1) % test_epoch):
			print("Starting validation:")
			model.eval()
			avg_auc = 0
			avg_acc = 0
			cnt = 0
			for i, data in enumerate(test):          
				loss, r = model.train_batch(input_batches=Variable(data[0]),
													input_lengths=Variable(data[2]),
													target_batches=data[4],
													reset=(i==0))
				auc,acc = get_results(r, data[4])
				avg_auc += auc
				avg_acc += acc
				cnt += 1
			avg_auc /= 1.0*cnt
			avg_acc /= 1.0*cnt
			print("Test auc after {} epoch\n".format(avg_auc))
			print("Test acc after {} epoch\n\n".format(avg_acc))
			
			if(avg_auc > avg_best):
				avg_best = avg_auc
				if dump_path != "":
					torch.save(model.state_dict(), dump_path+'/best_auc')
			
			

				# pbar.set_description(model.print_loss())

		# if((epoch+1) % 1 == 0):
		#     bleu = model.evaluate(train,avg_best)
		#     model.scheduler.step(bleu)
		#     if(bleu >= avg_best):
		#         avg_best = bleu
		#         cnt=0
		#     else:
		#         cnt+=1
		#
		#     if(cnt == 5): break
