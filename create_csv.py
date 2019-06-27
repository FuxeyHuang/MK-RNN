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
parser.add_argument('--epochs', nargs="+", type = int, help = 'epochs you want to forcus')
parser.add_argument('--project_id', type=str, help = 'project id')
parser.add_argument('--data_id', type=str, help = 'data id')
parser.add_argument('--choose_best', type=str, default="False", help = 'when set to True means we will find the best performence model')

args = parser.parse_args()
max_epoch = 300

def Round(a, n):
    a = float(a)
    a = round(a, n)
    return str(a)
    

if args.choose_best == "True":
    file_header = ['Project: {} data: {} Best'.format(args.project_id, args.data_id), 
#    'Train acuracy (with NA)', 'Train acuracy (without NA)', 'Test acuracy (with NA)', 'Test acuracy (without NA)', 'Train AUC', 'Test AUC']
    'Train AUC', 'Test AUC']
    
    path = './dumped_projects/{}_{}/'.format(args.project_id, args.data_id)
    dump_path = "./csvfiles/{}_{}/".format(args.project_id, args.data_id)
    if not os.path.exists(dump_path):
        os.mkdir(dump_path)
    csvFile = open(dump_path + "best.csv", "w")
    writer = csv.writer(csvFile)
    writer.writerow(file_header)
    
    files = os.listdir(path)
    model_order = ['cnn1', 'cnn2', 'cnn3', 'lstm', 'mem', 'g', 'mem_g', 'g_gcn', 'mem_g_gcn']
    model_order += [a+'_t1' for a in model_order]
    model_order += [a+'_t2' for a in model_order]
    model_order += [a+'_t3' for a in model_order]
    model_order += [a+'_b' for a in model_order]
    for model in model_order:
        
        fnames = model + ".txt"
        if not fnames in files:
            continue
        
        f = open(path+fnames, 'r')
        logs = f.readlines()
        f.close()
        
        if(len(logs) < 7):
            continue
        
        print("start creating {}".format(fnames[:-4]))
        
        e = -1
        mx = -1.0
        for i in range(max_epoch):
            if i*10+6 >= len(logs):
                break;
            if mx < float(logs[i*10+6].split()[-1]):
                mx = float(logs[i*10+6].split()[-1])
                e = i
        
        d = [fnames[:-4]]
#        d.append(Round(logs[e*10+2].split()[-2], 5))
#        d.append(Round(logs[e*10+2].split()[-1], 5))
#        
#        d.append(Round(logs[e*10+7].split()[-2], 5))
#        d.append(Round(logs[e*10+7].split()[-1], 5))
        
        d.append(Round(logs[e*10+1].split()[-1], 5))
        d.append(Round(logs[e*10+6].split()[-1], 5))
        d.append("From epoch {}".format(e))
        writer.writerow(d)
    csvFile.close()
        
else:
    for e in args.epochs:
        file_header = ['Porject: {} data: {} Epoch: {}'.format(args.project_id, args.data_id, e), 
#        'Train acuracy (with NA)', 'Train acuracy (without NA)', 'Test acuracy (with NA)', 'Test acuracy (without NA)', 'Train AUC', 'Test AUC']
        'Train AUC', 'Test AUC']
        
        path = './dumped_projects/{}_{}/'.format(args.project_id, args.data_id)
        dump_path = "./csvfiles/{}_{}/".format(args.project_id, args.data_id)
        if not os.path.exists(dump_path):
            os.mkdir(dump_path)
        csvFile = open(dump_path + "epoch_{}.csv".format(e), "w")
        writer = csv.writer(csvFile)
        writer.writerow(file_header)
        
        files = os.listdir(path)
        model_order = ['cnn1', 'cnn2', 'cnn3', 'lstm', 'mem', 'g', 'mem_g', 'g_gcn', 'mem_g_gcn']
        model_order += [a+'_t1' for a in model_order]
        model_order += [a+'_t2' for a in model_order]
        model_order += [a+'_t3' for a in model_order]

        for model in model_order:
            fnames = model + ".txt"
            if not fnames in files:
                continue
            
            f = open(path+fnames, 'r')
            logs = f.readlines()
            f.close()
            
            if(e*10+8 >= len(logs)):
                continue
            
            print("start creating {}".format(fnames[:-4]))
            d = [fnames[:-4]]
#            d.append(Round(logs[e*10+2].split()[-2], 5))
#            d.append(Round(logs[e*10+2].split()[-1], 5))
#            
#            d.append(Round(logs[e*10+7].split()[-2], 5))
#            d.append(Round(logs[e*10+7].split()[-1], 5))
            
            d.append(Round(logs[e*10+1].split()[-1], 5))
            d.append(Round(logs[e*10+6].split()[-1], 5))

            
            writer.writerow(d)
        csvFile.close()
