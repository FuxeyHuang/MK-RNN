#!/usr/bin/python



import torch
import numpy as np
import logging
from tqdm import tqdm
import os
import argparse
import random
import threading
from multiprocessing.dummy import Pool as ThreadPool
import time
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--project_id', type=str, help = 'project id')
parser.add_argument('--data_id', type=str, help = 'data id')

args = parser.parse_args()
max_epoch = 300

def Round(a, n):
	a = float(a)
	a = round(a, n)
	return str(a)
	

file_header = ['Project: {} data: {} Best'.format(args.project_id, args.data_id), 
#    'Train acuracy (with NA)', 'Train acuracy (without NA)', 'Test acuracy (with NA)', 'Test acuracy (without NA)', 'Train AUC', 'Test AUC']
'F1']

path = './dumped_projects/{}_{}/'.format(args.project_id, args.data_id)
dump_path = "./csvfiles/{}_{}/".format(args.project_id, args.data_id)
if not os.path.exists(dump_path):
	os.mkdir(dump_path)
csvFile = open(dump_path + "f1.csv", "w")
writer = csv.writer(csvFile)
writer.writerow(file_header)

files = os.listdir(path)
model_order = ['cnn1', 'cnn2', 'cnn3', 'lstm', 'mem', 'g', 'mem_g', 'g_gcn', 'mem_g_gcn']
model_order += [a+'_t1' for a in model_order]
model_order += [a+'_t2' for a in model_order]
model_order += [a+'_t3' for a in model_order]
model_order += [a+'_b' for a in model_order]

def get_f1(x, y):
	res = 0
	for i in range(len(x)):
		if not x[i] + y[i] == 0:
			res = max(res, 2*(x[i] * y[i]) / (x[i] + y[i]))
	return res;


for model in model_order:
	
	fnames = model + ".txt"
	if not fnames in files:
		continue
	
	x = np.load(path + model + '/f1_x.npy')
	y = np.load(path + model + '/f1_y.npy')
	
	d = [fnames[:-4]]
	d.append(get_f1(x, y))
	writer.writerow(d)
	
csvFile.close()
	