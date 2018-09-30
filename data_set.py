# -*- coding: utf-8 -*-
"""
@author: limingfan

"""

# import xlrd

import os
import numpy as np

# import random
# random.shuffle(list_ori, random.seed(10))

import data_utils

from vocab import Vocab


"""
multi-sentences-level nlp tasks

for prediction,
for vocab,
for data_preprocessed,

for split,
for batching, standardizing,


"""


class Dataset():
    
    def __init__(self, vocab = None):
        
        # vocab
        self.vocab = vocab
        
        # directories for saving results, auto-mk,
        self.dir_vocab = './vocab'
        self.dir_data_converted = './data_converted'
        
        self.vocab_filter_cnt = 5
        self.emb_dim = 300
        self.pretrained_emb_file = None
             
        # train and valid
        self.file_train = "./data_raw/cnews.train.txt"
        self.file_valid = "./data_raw/cnews.val.txt"
        self.file_test = "./data_raw/cnews.test.txt"
        #
        self.labels_raw_train = []
        self.labels_raw_valid = []
        self.labels_raw_test = []

        self.labels_idx_train = []
        self.labels_idx_valid = []
        self.labels_idx_test = []

        self.labels_map = {}        

        #        
        self.data_raw_train = []
        self.data_raw_valid = []
        self.data_raw_test = []
        
        self.data_seg_train = []
        self.data_seg_valid = []
        self.data_seg_test = []
        
        self.data_idx_train = []
        self.data_idx_valid = []
        self.data_idx_test = []
        #

    #
    # interface function, preprocess, all are staticmethod
    @staticmethod
    def preprocess_for_prediction(data_raw, settings):
        """ data_raw: list of texts (documents, multi-sentences)        
            returning: data for deep-model input
        """
        vocab = settings.vocab
        
        data_seg = data_utils.clean_and_seg_data_raw(data_raw)
        data_converted = data_utils.convert_data_seg_to_idx(vocab, data_seg)
        data_s, num_s = Dataset.do_standardizing_examples(data_converted, settings)              
                
        return (data_s, num_s)
    
    #
    # vocab
    def build_vocab_tokens_and_emb(self):
        """ from dataset data_seg
        """
        print('build vocab tokens and emb ...')
        self.vocab = Vocab()
        
        # data_seg, task-related
        for text in self.data_seg_train:
            self.vocab.load_tokens_from_corpus(text)  # data_seg
        for text in self.data_seg_valid:
            self.vocab.load_tokens_from_corpus(text)
        for text in self.data_seg_test:
            self.vocab.load_tokens_from_corpus(text)
            
        # build, task-independent
        self.vocab.filter_tokens_by_cnt(self.vocab_filter_cnt)
        #
        if self.pretrained_emb_file:
            self.vocab.load_pretrained_embeddings(self.pretrained_emb_file)             
        else:
            self.vocab.randomly_init_embeddings(self.emb_dim)
        
        # save, task-independent
        self.save_vocab_tokens_and_emb()
        #
        
    # task-independent
    def load_vocab_tokens_and_emb(self, file_tokens = None, file_emb = None):
        """
        """
        print('load vocab tokens and emb ...')
        #        
        if file_tokens is None:
            file_tokens = os.path.join(self.dir_vocab, 'vocab_tokens.txt')
        if file_emb is None:
            file_emb = os.path.join(self.dir_vocab, 'vocab_emb.txt')

        self.vocab = Vocab()
        self.vocab.load_tokens_from_file(file_tokens)
        self.vocab.load_pretrained_embeddings(file_emb)
        
    def save_vocab_tokens_and_emb(self, file_tokens = None, file_emb = None):
        """
        """
        print('save vocab tokens and emb ...')
        #        
        if file_tokens is None:
            file_tokens = os.path.join(self.dir_vocab, 'vocab_tokens.txt')
        if file_emb is None:
            file_emb = os.path.join(self.dir_vocab, 'vocab_emb.txt')
        # 
        if not os.path.exists(self.dir_vocab): os.mkdir(self.dir_vocab)
        #
        self.vocab.save_tokens_to_file(os.path.join(self.dir_vocab, 'vocab_tokens.txt'))
        self.vocab.save_embeddings_to_file(os.path.join(self.dir_vocab, 'vocab_emb.txt'))
    
    #
    # data, task-related,
    #
    def load_data_raw(self):
        
        print('load data_raw ...')
        self.data_raw_train, self.labels_raw_train = data_utils.load_from_file_raw(self.file_train)
        self.data_raw_valid, self.labels_raw_valid = data_utils.load_from_file_raw(self.file_valid)
        self.data_raw_test, self.labels_raw_test = data_utils.load_from_file_raw(self.file_test)
    
    #
    def prepare_preprocessed_data(self, load_vocab):
        """ prepare data to train and test
        """
        
        # load
        self.load_data_raw()
        
        # cleanse and seg
        print('cleanse and seg ...')        
        self.data_seg_train = data_utils.clean_and_seg_data_raw(self.data_raw_train)
        self.data_seg_valid = data_utils.clean_and_seg_data_raw(self.data_raw_valid)
        self.data_seg_test = data_utils.clean_and_seg_data_raw(self.data_raw_test)

        # vocab
        print('load or build vocab ...')
        if load_vocab:
            self.load_vocab_tokens_and_emb()
        else:
            self.build_vocab_tokens_and_emb()
        print('num_tokens in vocab: %d' % self.vocab.size() )
        
        # convert
        print('convert to ids ...')
        self.data_idx_train = data_utils.convert_data_seg_to_idx(self.vocab, self.data_seg_train)
        self.data_idx_valid = data_utils.convert_data_seg_to_idx(self.vocab, self.data_seg_valid)
        self.data_idx_test = data_utils.convert_data_seg_to_idx(self.vocab, self.data_seg_test)
        
        l_map = {}
        l_map, self.labels_idx_train = data_utils.convert_labels_to_idx(l_map, self.labels_raw_train)
        l_map, self.labels_idx_valid = data_utils.convert_labels_to_idx(l_map, self.labels_raw_valid)
        l_map, self.labels_idx_test = data_utils.convert_labels_to_idx(l_map, self.labels_raw_test)
        
        for key in l_map:
            self.labels_map[l_map[key]] = key

        print(l_map)
        print(self.labels_map)
        
        #
        print('save data converted ...')
        if not os.path.exists(self.dir_data_converted): os.mkdir(self.dir_data_converted)
        
        data = (self.data_idx_train, self.labels_idx_train)
        file_path = os.path.join(self.dir_data_converted, 'data_train.pkl')
        data_utils.save_data_to_pkl(data, file_path)
        
        data = (self.data_idx_valid, self.labels_idx_valid)
        file_path = os.path.join(self.dir_data_converted, 'data_valid.pkl')
        data_utils.save_data_to_pkl(data, file_path)
            
        data = (self.data_idx_test, self.labels_idx_test)
        file_path = os.path.join(self.dir_data_converted, 'data_test.pkl')
        data_utils.save_data_to_pkl(data, file_path)
        
        import json
        file_path = os.path.join(self.dir_data_converted, 'labels_map.json')
        with open(file_path, 'w') as fp:
            json.dump(self.labels_map, fp, ensure_ascii = False)
        
        print('preparation done.')
        
    def load_preprocessed_data(self):
        """
        """
        self.load_vocab_tokens_and_emb()
        #
        print('load preprocessed data ...')
        #
        file_path = os.path.join(self.dir_data_converted, 'data_train.pkl')
        self.data_idx_train, self.labels_idx_train = data_utils.load_data_from_pkl(file_path)
            
        file_path = os.path.join(self.dir_data_converted, 'data_valid.pkl')
        self.data_idx_valid, self.labels_idx_valid = data_utils.load_data_from_pkl(file_path)
            
        file_path = os.path.join(self.dir_data_converted, 'data_test.pkl')
        self.data_idx_test, self.labels_idx_test = data_utils.load_data_from_pkl(file_path)
            
        import json
        file_path = os.path.join(self.dir_data_converted, 'labels_map.json')
        with open(file_path, 'r') as fp:
            self.labels_map = json.load(fp)
    
    #
    # not required for this task
    def split_train_and_test(self, ratio_train = 0.8, shuffle = False):
        
        num_1 = len(self.data_converted_1)
        num_2 = len(self.data_converted_2)
        
        print('split train and test ...')
        print('num_1: %d' % num_1)
        print('num_2: %d' % num_2)
        
        if shuffle:
            indices = np.random.permutation(np.arange(num_1))
            data_1 = np.array(self.data_converted_1)[indices].tolist()
            
            indices = np.random.permutation(np.arange(num_2))
            data_2 = np.array(self.data_converted_2)[indices].tolist()
        else:
            data_1 = self.data_converted_1
            data_2 = self.data_converted_2
        
        num_train_1 = int(num_1 * ratio_train)
        num_train_2 = int(num_2 * ratio_train)
        
        train_data = (data_1[0:num_train_1], data_2[0:num_train_2])
        test_data = (data_1[num_train_1:], data_2[num_train_2:])
        
        return train_data, test_data
        
    #
    # batching
    @staticmethod
    def do_batching_data(data, batch_size):
        """ data: (texts, labels)
            texts: (D, S, T)
            
            batches: [ (texts, labels) , ...]
        """
        
        texts, labels = data
        
        num_all = len(labels)
        
        indices = np.random.permutation(np.arange(num_all))
        texts = np.array(texts)[indices].tolist()
        labels = np.array(labels)[indices].tolist()
        
        num_batches = num_all // batch_size
        
        batches = []
        start_id = 0
        end_id = batch_size
        for i in range(num_batches):            
            batches.append( (texts[start_id:end_id], labels[start_id:end_id]) )
            start_id = end_id
            end_id += batch_size
        
        if num_batches * batch_size < num_all:
            num_batches += 1
            texts_rem = texts[end_id:]
            labels_rem = labels[end_id:]
            #
            d = batch_size - len(labels_rem)
            texts_rem.extend( texts[0:d] )
            labels_rem.extend( labels[0:d] )
            #
            batches.append( (texts_rem, labels_rem) )
        
        return batches 
    
    @staticmethod
    def do_standardizing_batches(data_batches, settings):
        """ data_batches: [ (texts, labels), ...]
            texts: (D, S, T)
            
            returning: [ (x, num_sent, y), ...]
        """
        batches_normed = []
        for batch in data_batches:
            x, y = batch        
            x_padded, num_sent = Dataset.do_standardizing_examples(x, settings)
            batches_normed.append( (x_padded, num_sent, y) )
            
        return batches_normed
        
    @staticmethod
    def do_standardizing_examples(x, settings = None):
        """ x: (D, S, T)
        
            returning: x_std, num_sent
            shape: (DS', T'), (D,)
        """        
        max_num_sent = 30
        min_seq_len = 5
        max_seq_len = 100
        if settings is not None:
            max_num_sent = settings.max_num_sent
            min_seq_len = settings.min_seq_len
            max_seq_len = settings.max_seq_len
        #
        x_trimmed = []
        num_sent = []
        for text in x:
            num_sent_curr = len(text)
            if num_sent_curr > max_num_sent:
                x_trimmed.append( text[0:max_num_sent] )
                num_sent.append( max_num_sent )
            else:
                x_trimmed.append( text )
                num_sent.append( num_sent_curr )
        #
        max_len = 0
        for text in x_trimmed:
            max_len_curr = max([len(sent) for sent in text])
            if max_len_curr > max_len:
                max_len = max_len_curr
        max_len = max(min_seq_len, min(max_seq_len, max_len) )
        #       
        x_padded = []
        for text in x_trimmed:
            text_padded = []
            for sent in text:
                d = max_len - len(sent)
                sent_n = sent.copy()
                if d > 0:
                    sent_n.extend([0] * d)  # pad_id, 0
                elif d < 0:
                    sent_n = sent_n[0:max_len]
                text_padded.append(sent_n)
            #
            x_padded.extend( text_padded )
            
        return x_padded, num_sent
    
    
if __name__ == '__main__':   
    
    pretrained_emb_file = None
    

    # prepare
    dataset = Dataset()
    
    dataset.pretrained_emb_file = pretrained_emb_file
    dataset.vocab_filter_cnt = 5
    dataset.emb_dim = 64
  
    dataset.prepare_preprocessed_data(load_vocab = False)
    
    print('prepared')
    
    #
    vocab = dataset.vocab
    data_raw = dataset.data_raw_valid
    
    from collections import namedtuple
    Settings = namedtuple('Settings', ['vocab', 'max_num_sent',
                                       'min_seq_len', 'max_seq_len'])
    settings = Settings(vocab, 10, 5, 100)
    
    batch_p = Dataset.preprocess_for_prediction(data_raw, settings)
    
    print('preprocessed')
    
   
    
    # load
    dataset = Dataset()
    dataset.load_preprocessed_data()
    
    print('loaded')

    #
    # data_train, data_valid = dataset.split_train_and_test()

    #
    data = dataset.data_idx_train, dataset.labels_idx_train
    train_batches = Dataset.do_batching_data(data, 32)
    
    data = dataset.data_idx_valid, dataset.labels_idx_valid
    valid_batches = Dataset.do_batching_data(data, 32)
    
    train_batches_padded = Dataset.do_standardizing_batches(train_batches, settings)
    valid_batches_padded = Dataset.do_standardizing_batches(valid_batches, settings)
    
    print('finished')

    