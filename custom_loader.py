import pandas as pd
import numpy as np
from collections import OrderedDict
import re

from sklearn.utils import shuffle
import pickle
import os
import gensim
import string

from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import time

#
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def deTokenize(tokens, start, end):
    string = []
    vocabulary = list(bert_tokenizer.vocab.keys())
    ## take only the tokens between start and end
    #print(tokens)
    idx_start = 0
    idx_end = np.where(tokens==end)[0][0]
    #print(idx_end)
    tokens = tokens[ int(idx_start+1) : idx_end]
    for tok in tokens:
        word = vocabulary[int(tok)]
        #print(word)
        string.append(word)
    return(string)

def bert_tokens(s, no_wordpiece=0, pad=30):

    if no_wordpiece:
        words = s.split(" ")
        words = [word for word in words if word in bert_tokenizer.vocab.keys()]
        s = " ".join(words)
    inputs = bert_tokenizer(s, return_tensors='tf', max_length=512)
    inputs = inputs['input_ids']
    tokens = np.zeros(pad)
    token_count = 0
    inputs = inputs.numpy()[0]
    if len(inputs) >=pad:
        tokens = inputs[:pad]
        tokens[-1] = 102
        token_count = pad
    else:
        tokens[:len(inputs)] = inputs
        token_count = len(inputs)
    return(tokens,token_count)


def clean(s):
    # s = re.sub(r'(\d+\.){3}\d+(:\d+)?', " ", s)
    # s = re.sub(r'(\/.*?\.[\S:]+)', ' ', s)
    s = re.sub('\]|\[|\)|\(|\=|\,|\;', ' ', s)
    s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])
    s = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s))
    s = " ".join([word for word in s.split() if not bool(re.search(r'\d', word))])
    trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
    content = s.translate(trantab)
    s = " ".join([word.lower().strip() for word in content.strip().split()])
    return s

def timestamp(log):
    log = log[log.find(" ") + 1:]
    t = log[:log.find(" ")]
    return float(t)



def load_logfile(log_file, windows_size=20, windows_wide=20, step_size=0, NoWordPiece=0):
    print("Loading", log_file)

    with open(log_file, mode="r", encoding='utf8') as f:
        logs = f.readlines()
        logs = [x.strip() for x in logs]

    encoder = bert_tokens
    print("Loaded log file. Start preprocessing")

    #####
    # Create the log Windows and labels
    # x_tr = log windows 
    # y_tr_window = labels of log windows, depending all log lines in a window
    # y_tr_log = labels of log windows, depending only on the last log line in a window
    #####
    x_tr, y_tr_window, y_tr_log = [], [], []
    entries = {} # save uniqe entries -> TODO: dictonary can be very slow
    i = 0

    ## iterate over all from the beginning
    n_train = int(len(logs))# * train_ratio)
    c = 0
    t0 = time.time()
    while i < (n_train - windows_size):
        c += 1
        if c % 1000 == 0:
            print("Loading {0:.2f} - {1} unique logs".format(i * 100 / n_train, len(entries.keys())))
        seq = []
        label_window = 0
        label_log = 0
        for j in range(i, i + windows_size):
            ###
            # Go through the log data
            # If the log line appears the first time -> tokenize
            # or use the already tokenized log line
            # build on this way the log windows 
            ###
            ## set the complete window to "anomaly" if there is a single anomaly (or more) (only on train-set)
            if logs[j][0] != "-":
                label_window = 1 # anomaly
            content = logs[j]
            # remove label from log messages
            content = content[content.find(' ') + 1:]
            content = clean(content.lower()) # lower case
            if content not in entries.keys():
                try:
                    entries[content], _ = encoder(content, NoWordPiece, windows_wide)
                except:
                    print(content)
            seq.append(entries[content])
        if logs[j][0] != "-": # if only the last log line is an anomaly
            label_log = 1            
        x_tr.append(seq.copy())
        y_tr_window.append(label_window)
        y_tr_log.append(label_log)
        #i = i + windows_size # NOTE: step_size ?!
        i = i + step_size



    print("last train index:", i)


    num_train = len(x_tr)
    num_train_pos = sum(y_tr_window)
    print('# of windows in total: %i , numbers of anomalies: %i' %(num_train, num_train_pos))


    return (x_tr, y_tr_window, y_tr_log)
