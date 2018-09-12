#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 06:23:53 2018

@author: limingfan
"""

def load_from_file_raw(file_raw):
    
    with open(file_raw, 'r', encoding = 'utf-8') as fp:
        lines = fp.readlines()
    
    labels_raw = []
    texts_raw = []
    for line in lines:
        if line.strip() != '':

            label = line[0:2]
            body = line[3:].strip()
            
            labels_raw.append(label)
            texts_raw.append(body)
    #
    return texts_raw, labels_raw

def write_to_file(texts_and_labels, file_path, stride = 1):
    
    texts, labels = texts_and_labels
    
    num_end = len(texts)
    
    fp = open(file_path, 'w', encoding = 'utf-8')
    for idx in range(num_end):
        if idx % stride == 0:
            fp.write('%s  %s\n' % (labels[idx], texts[idx]))
    fp.close()
    #
    
#
file_path = './data_raw/cnews.train.ori.txt'
file_des = file_path.replace('.ori', '')

data = load_from_file_raw(file_path)
write_to_file(data, file_des, 5)

#
file_path = './data_raw/cnews.test.ori.txt'
file_des = file_path.replace('.ori', '')

data = load_from_file_raw(file_path)
write_to_file(data, file_des)

#
file_path = './data_raw/cnews.val.ori.txt'
file_des = file_path.replace('.ori', '')

data = load_from_file_raw(file_path)
write_to_file(data, file_des)


#

