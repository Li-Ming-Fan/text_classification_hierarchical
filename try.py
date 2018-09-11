# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 13:00:01 2018

@author: limingfan

"""

list_a = [0, 1, 3, 2]

print(list_a.index(max(list_a)))


#
import tensorflow as tf

tf.reset_default_graph()


a = tf.get_variable('a', shape = (2,4,5),  initializer = tf.random_normal_initializer() )
b = tf.get_variable('b', shape = (2,3,5), initializer = tf.random_normal_initializer() )

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(a))
print(sess.run(b))

c = tf.matmul(a, tf.transpose(b, [0,2,1]))
print(c)


print()
print(sess.run(tf.cast([0, 1.2, -1.0], dtype = tf.bool)) )



text = 'aaaa。子了;aaaabbb。'
try:
    print(text.index('!'))
except BaseException:
    print('None')
        
a = [5,7,6,3,4,1,2]
a.sort()
print(a)

import data_set
import jieba

text = '偏股型基金股票仓位大幅上升本报讯(记者窦红梅) ... 近期A股屡次创出反弹新高，基金增仓动作明显。德圣基金研究中心统计显示，越来越多的基金开始转向牛市思维，提高股票仓位的同时在结构调整上也更加积极。截至昨天，发布一季报的141只基金中，98只可比积极投资偏股型基金股票仓位为77.32%，与2008年四季度的67.84%相比大幅上升。各类以股票为主要投资方向的基金平均仓位均有明显上升。“德圣”测算数据显示，上周扣除被动增仓效应后，各偏股类型基金主动增仓幅度约在2%。基金仓位继续上升，已经接近牛市下的高点。天相投顾统计显示，在整体仓位大幅上升的情况下，不同基金公司增减仓有所分歧。金融保险、机械设备仪表和房地产为基金持有的前三大行业。'
sent_list = data_set.segment_sentences(text)
print(text)
print(sent_list)

sent = sent_list[0]

seg_list = jieba.cut(sent, cut_all = False)

print(seg_list)
print(list(seg_list))


l_map = {0: '体育', 1: '娱乐', 2: '家居', 3: '房产', 4: '教育', 5: '时尚', 6: '时政', 7: '游戏', 8: '科技', 9: '财经'}

import os
import json
file_path = os.path.join('./data_converted', 'labels_map.json')
with open(file_path, 'w') as fp:
    json.dump(l_map, fp, ensure_ascii = False)
            

with open(file_path, 'r') as fp:
    labels_map = json.load(fp)
    
print(labels_map)

#
def gather_and_pad_layer(x, num_items):
    """ x: (BS', D)
        num_items : (B,)
        
        returning: (B, S, D)
    """
    B = tf.shape(num_items)[0]
    T = tf.reduce_max(num_items)
    
    pad_item = tf.zeros(shape = tf.shape(x[0:1,:]) )
    one_int32 = tf.ones(shape = (1,), dtype = tf.int32)
    zero_int32 = tf.zeros(shape = (1,), dtype = tf.int32)
    
    bsd_ta = tf.TensorArray(size = B, dtype = tf.float32)
    mask_ta = tf.TensorArray(size = B, dtype = tf.int32)
    time = tf.constant(0)
    posi = tf.constant(0)
    
    def condition(time, posi_s, bsd_s, mask_s):
        return tf.less(time, B)
    
    def body(time, posi_s, bsd_s, mask_s):        
        posi_e = posi_s + num_items[time]        
        chunk = x[posi_s:posi_e, :]
        #
        mask_c = tf.tile(one_int32, [ num_items[time] ] )
        #
        d = T - num_items[time]
        chunk, mask_c = tf.cond(d > 0,
                                lambda: (tf.concat([chunk, tf.tile(pad_item, [d, 1])], 0),
                                         tf.concat([mask_c, tf.tile(zero_int32, [d])], 0) ),
                                lambda: (chunk, mask_c) )
        #
        bsd_s = bsd_s.write(time, chunk)
        mask_s = mask_s.write(time, mask_c)
        return (time + 1, posi_e, bsd_s, mask_s)
        
    t, p, bsd_w, mask_w = tf.while_loop(cond = condition, body = body,
                                        loop_vars = (time, posi, bsd_ta, mask_ta) )
    bsd = bsd_w.stack()
    mask = mask_w.stack()
    
    return bsd, mask
    
    
x = tf.get_variable('x', shape = (10,5),  initializer = tf.random_normal_initializer() )
n = tf.constant([1,2,3,4])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(x))
print(sess.run(n))

print(sess.run(x[0,:]))
bsd, mask = gather_and_pad_layer(x, n)
print(sess.run(bsd))
print(sess.run(mask))

#

one_int32 = tf.ones(shape = (1,), dtype = tf.int32)
zero_int32 = tf.zeros(shape = (1,), dtype = tf.int32)
    
mask_c = tf.tile(one_int32, [ 10 ] )

print(sess.run(mask_c))