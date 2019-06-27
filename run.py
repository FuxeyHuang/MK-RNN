#!/usr/bin/python



import torch
import numpy as np
import logging
from tqdm import tqdm
import os
import argparse
import random
from pynvml import *
import threading
from multiprocessing.dummy import Pool as ThreadPool
import time

parser = argparse.ArgumentParser()
parser.add_argument('--data_id', type = str, help = 'data type')
parser.add_argument('--model_names', nargs='+', help = 'list of models')
parser.add_argument('--project_id', type=str, help = 'project id')
parser.add_argument('--cuda_id', type=int, default=-1, help = '-1 will enable check cuda')
parser.add_argument('--batch_size', type=int, default=-1, help = '-1 will not set batch_size')
parser.add_argument('--use_transformer', type=str, default="False", help = 'if you want to use transformer')
parser.add_argument('--use_bilstm', type=str, default="False", help = 'if you want to use bilstm')
parser.add_argument('--transformer_layer', type=int, default=1, help = 'number of layers of transformer')
parser.add_argument('--batch_count', type = int, default = 1, help = 'how many batches counted before loss back_propegate')

args = parser.parse_args()
used = set()

def check_cuda():
	nvmlInit()
	deviceCount = nvmlDeviceGetCount()
	
	min_mem = -1
	who = -1
	
	keep_one_free=False
	if len(used) == deviceCount - 1:
		keep_one_free = True
	
	for i in range(deviceCount):
		if keep_one_free and (not i in used):
			continue
		
		handle = nvmlDeviceGetHandleByIndex(i)
		info = nvmlDeviceGetMemoryInfo(handle)
		if who == -1 or min_mem > info.used:
			min_mem = info.used
			who = i
	used.add(who)
	return who

def run_model(m):
#	for m in args.model_names:
	cuda_use = args.cuda_id
	if cuda_use == -1:
		time.sleep(120*m[0])
		cuda_use = check_cuda()
	
	if args.batch_size == -1:
		print("Start running " + "python ./train.py --model_name={} --cuda_id={} --project_id={} --data_id={} --use_transformer={} --transformer_layer={} --batch_count={} --use_bilstm={}".format(m[1], cuda_use, args.project_id, args.data_id, args.use_transformer, args.transformer_layer, args.batch_count, args.use_bilstm))
		os.system("python ./train.py --model_name={} --cuda_id={} --project_id={} --data_id={} --use_transformer={} --transformer_layer={} --batch_count={} --use_bilstm={}".format(m[1], cuda_use, args.project_id, args.data_id, args.use_transformer, args.transformer_layer, args.batch_count, args.use_bilstm))
	else:
		print("Start running " + "python ./train.py --model_name={} --cuda_id={} --project_id={} --data_id={} --batch_size={} --use_transformer={} --transformer_layer={} --batch_count={} --use_bilstm={}".format(m[1], cuda_use, args.project_id, args.data_id, args.batch_size, args.use_transformer, args.transformer_layer, args.batch_count, args.use_bilstm))
		os.system("python ./train.py --model_name={} --cuda_id={} --project_id={} --data_id={} --batch_size={} --use_transformer={} --transformer_layer={} --batch_count={} --use_bilstm={}".format(m[1], cuda_use, args.project_id, args.data_id, args.batch_size, args.use_transformer, args.transformer_layer, args.batch_count, args.use_bilstm))


waiting_list = enumerate(args.model_names)
pool = ThreadPool()
pool.map(run_model, waiting_list)
pool.close()
pool.join()
