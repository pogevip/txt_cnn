#encoding:utf-8

import load_data
from config import TextConfig as config

#参数
filename = config.filename
vocab_dir =  config.vocab_dir
vocab_size = config.vocab_size

#读取，分词
contents,labels = load_data.read_file_seg(filename)
#建立词典
load_data.build_vocab(filename,vocab_dir,vocab_size)
#词ID，类别ID
_,word_to_id = load_data.read_vocab(vocab_dir)
_,cat_to_id = load_data.read_category()
#生成向量
x,y = load_data.process_file(filename,word_to_id,cat_to_id,max_length=600)
a = load_data.batch_iter(x,y,10)
#load_data函数里面所有包含filename为参数的函数，都应该优化
#实现，运行load_data生成词典，词向量，引用直接加载训练好的模型就可以