# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 15:33:42 2018

@author: limingfan

"""

import tensorflow as tf

from zoo_layers import rnn_layer
# from zoo_layers import att_pool_layer
from zoo_layers import gather_and_pad_layer


def build_graph(config):
    
    input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
    input_n = tf.placeholder(tf.int32, [None], name='input_n')
    input_y = tf.placeholder(tf.int64, [None], name='input_y')

    with tf.device('/cpu:0'):
        emb_mat = tf.get_variable('embedding',
                                  [config.vocab.size(), config.vocab.emb_dim],
                                  initializer=tf.constant_initializer(config.vocab.embeddings),
                                  trainable = config.emb_tune)
        seq_emb = tf.nn.embedding_lookup(emb_mat, input_x)
        
        seq_mask = tf.cast(tf.cast(input_x, dtype = tf.bool), dtype = tf.int32)
        seq_len = tf.reduce_sum(seq_mask, 1)

    with tf.name_scope("rnn"):
        
        seq_e = rnn_layer(seq_emb, seq_len, config.hidden_units, config.keep_prob,
                          activation = tf.nn.relu, concat = True, scope = 'bi-lstm-1')        
        feat_s = seq_e[:,-1,:]
        
        #
        feat_g, mask_s = gather_and_pad_layer(feat_s, input_n)
        
        #
        seq_e = rnn_layer(feat_g, input_n, config.hidden_units, config.keep_prob,
                          activation = tf.nn.relu, concat = True, scope = 'bi-lstm-2')        
        feat = seq_e[:,-1,:]
        

    with tf.name_scope("score"):
        #
        fc = tf.contrib.layers.dropout(feat, config.keep_prob)
        fc = tf.layers.dense(fc, config.hidden_units, name='fc1')            
        fc = tf.nn.relu(fc)
        
        fc = tf.contrib.layers.dropout(fc, config.keep_prob)
        logits = tf.layers.dense(fc, config.num_classes, name='fc2')
        # logits = tf.nn.sigmoid(fc)
        
        normed_logits = tf.nn.softmax(logits, name='logits')          
        y_pred_cls = tf.argmax(logits, 1, name='pred_cls')
        
    with tf.name_scope("loss"):
        #
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,
                                                                       labels = input_y)
        loss = tf.reduce_mean(cross_entropy, name = 'loss')

    with tf.name_scope("accuracy"):
        #
        correct_pred = tf.equal(input_y, y_pred_cls)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = 'metric')
    
    #
    print(normed_logits)
    print(acc)
    print(loss)
    print()
    #

