#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 05:49:37 2018

@author: limingfan
"""

import pickle
import jieba

# import xlrd

    
def segment_sentences(text, delimiters = None):
    """
    """
    if delimiters is None:
        delimiters = ['?', '!', ';', '？', '！', '。', '；', '…', '\n']
    #
    # as for " ", and “ ”
    #
    text = text.replace('...', '。。。').replace('..', '。。') \
               .replace('"', '').replace('“', '').replace('”', '')
    #
    len_text = len(text)
    
    sep_posi = []
    for item in delimiters:
        posi_start = 0
        while posi_start < len_text:
            try:
                posi = posi_start + text[posi_start:].index(item)
                sep_posi.append(posi)
                posi_start = posi + 1               
            except BaseException:
                break # while
        #
    #
    sep_posi.sort()
    num_sep = len(sep_posi)
    #
    
    #
    list_sent = []
    #
    if num_sep == 0: return [ text ]
    #
    posi_last = 0
    for idx in range(0, num_sep - 1):
        posi_curr = sep_posi[idx] + 1
        posi_next = sep_posi[idx + 1]
        if posi_next > posi_curr:
            list_sent.append( text[posi_last:posi_curr] )
            posi_last = posi_curr
    #
    posi_curr = sep_posi[-1] + 1
    if posi_curr == len_text:
        list_sent.append( text[posi_last:] )
    else:
        list_sent.extend( [text[posi_last:posi_curr], text[posi_curr:]] )
    #
    return list_sent

#
# task-related
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

    
    """
    text_raw = []
    with open(file_raw, 'r', encoding = 'utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            if line.strip() != '':
                text_raw.append(line)
    #
    return text_raw
    
    #
    work_book = xlrd.open_workbook(file_raw)
    data_sheet = work_book.sheets()[0]
    queries = data_sheet.col_values(0)
    labels = data_sheet.col_values(2)
    return queries, labels
    """

# 
def clean_and_seg_data_raw(data_raw):
    """ data_raw: (D,)
        data_seg: (D, S, T)
    """
    data_seg = []
    for text in data_raw:
        text = text.strip()
        sent_list = segment_sentences(text)
        #
        text_seg = []
        for sent in sent_list:
            seg_list = jieba.cut(sent, cut_all = False)
            tokens = list(seg_list)
            #
            #print(text)
            #tokens = list(text)   # cut chars
            #
            text_seg.append( tokens )
        #
        data_seg.append(text_seg)
    return data_seg

def convert_data_seg_to_idx(vocab, data_seg):
    data_converted = []
    for text in data_seg:
        text_idx = []
        for sent in text:
            ids = vocab.convert_tokens_to_ids(sent)
            text_idx.append(ids)
        #
        data_converted.append(text_idx)
    return data_converted
    
def convert_labels_to_idx(labels_map, labels):
    
    labels_idx = []
    for item in labels:
        if item in labels_map:
            idx = labels_map[item]
        else:
            idx = len(labels_map)
            labels_map[item] = idx
        labels_idx.append(idx)

    return labels_map, labels_idx


# task-independent
def save_data_to_pkl(data, file_path):
    with open(file_path, 'wb') as fp:
        pickle.dump(data, fp)
        
def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as fp:
        return pickle.load(fp)
    
    
#
if __name__ == '__main__':
    
    text = '“噢，我大概猜不出来。”大兔子说。'
    text = '孩子们太兴奋了，他们不停地喊：“有蛋糕和冰淇淋吗？”“有装饰物吗？”“有礼物吗？”'
    text = '这是最常遇到的一种情况，也是最简单的：只要在“说话人说”的后面加上冒号，然后用引号将他说的话引起来就可以了。'
    
    print(segment_sentences(text))


