# -*- coding: utf-8 -*-

import os

from data_set import Dataset

from model_settings import ModelSettings
from model_wrapper import ModelWrapper


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#
model_tag = 'han'
#

if model_tag == 'cnn':
    from model_graph_cnn import build_graph
elif model_tag == 'csm':
    from model_graph_csm import build_graph
elif model_tag == 'rnn':
    from model_graph_rnn import build_graph
elif model_tag == 'han':
    from model_graph_han import build_graph
elif model_tag == 'rnf':
    from model_graph_rnf import build_graph
    
# 
# data
dataset = Dataset()
#

#
flag_load_data = True
# data
if flag_load_data:
    dataset.load_processed_data()
else:
    dataset.pretrained_emb_file = None
    dataset.prepare_processed_data(load_vocab = False)
#
data_train = dataset.data_idx_train, dataset.labels_idx_train
data_valid = dataset.data_idx_valid, dataset.labels_idx_valid
#

#
config = ModelSettings()
config.vocab = dataset.vocab
config.model_tag = model_tag
config.model_graph = build_graph
config.is_train = True
config.check_settings()

#
model = ModelWrapper(config)
model.prepare_for_train_and_valid()
#
model.train_and_valid(data_train, data_valid)
#
