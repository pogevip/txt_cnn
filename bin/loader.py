#encoding:utf-8

import pandas as pd
import numpy as np
import jieba_fast as jieba
import re
import time
from collections import Counter
import codecs
import tensorflow.contrib.keras as kr
import logging
from gensim.models import word2vec
from config import TextConfig as config
from itertools import chain

def read_file(filename):
    start_time=time.time()
    train=pd.read_csv(filename)
    for i in train.columns[1:]:
        train[i] = train[i].astype(str)
    text,labels = train.content.tolist(),train.price_discount.tolist()
    end_import=time.time()
    print("读取数据的时间为：",end_import-start_time,"s")
    return text,labels 

def read_file_seg(filename):
    contents = []
    text, labels  = read_file(filename)
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)") 
    k = 1
    for one in text:
        word = []
        blocks = re_han.split(one)
        for blk in blocks:
            if re_han.match(blk):
                word.extend(jieba.lcut(blk))
        contents.append(word)
        if k/100 == 0:
            print(k)
        k= k+1
    return contents,labels

def bulid_vocab(contents,vocab_size=20000):
    start_time=time.time()
    all_vocab = list(chain(*contents))
    counter=Counter(all_vocab)
    count_pairs=counter.most_common(vocab_size-1)
    words,_=list(zip(*count_pairs))
    words=['<PAD>']+list(words)
    word_to_id=dict(zip(words,range(len(words))))
    end_time=time.time()
    print('bulid_vocab cost',end_time-start_time,'s')
    return words,word_to_id

def read_category(labels):
    categories = list(set(labels))    #['-2', '-1', '0', '1']
    cat_to_id=dict(zip(categories,range(len(categories))))
    return categories,cat_to_id

def process_file(contents,labels,word_to_id,cat_to_id,max_length=600):
    data_id,label_id=[],[]
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id]) #把词转换成对应ID，形成列表（大列表套小列表）
        label_id.append(cat_to_id[labels[i]])   #把类别名转换成对应ID，列表
    max_length = max([len(i) for i in data_id])
    x_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length,padding='post', truncating='post')
    y_pad=kr.utils.to_categorical(label_id)
    return x_pad,y_pad

def batch_iter(x,y,batch_size=256):
    data_len=len(x)
    num_batch=int((data_len-1)/batch_size)+1
    indices=np.random.permutation(np.arange(data_len))
    x_shuffle=x[indices]
    y_shuffle=y[indices]
    for i in range(num_batch):
        start_id=i*batch_size
        end_id=min((i+1)*batch_size,data_len)
        yield x_shuffle[start_id:end_id],y_shuffle[start_id:end_id]





def train_word2vec(contents):
    t1 = time.time()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(contents, size=100, window=5, min_count=1, workers=6)
    model.wv.save_word2vec_format(config.vector_word_filename, binary=False)
    print('-------------------------------------------')
    print("Training word2vec model cost %.3f seconds...\n" % (time.time() - t1))

def export_word2vec_vectors(vocab_dir, vector_word_filename,vector_word_npz):#vector_word_npz='./data/vector_word.npz' 
    file_r = codecs.open(vector_word_filename, 'r', encoding='utf-8')
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(' '))
    _,vocab= read_vocab(vocab_dir)
    embeddings = np.zeros([len(vocab), vec_dim])
    line = file_r.readline()
    while line:
        try:
            items = line.split(' ')
            word = items[0]
            vec = np.asarray(items[1:], dtype='float32')
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(vec)
        except:
            pass
        line = file_r.readline()
    np.savez_compressed(vector_word_npz, embeddings=embeddings)
    return embeddings

def get_training_word2vec_vectors(vector_word_npz):
    with np.load(vector_word_npz) as data:
        return data["embeddings"]
        
if __name__ == '__main__':
    config=TextConfig()
    filenames=[config.train_filename,config.test_filename,config.val_filename]
    train_word2vec(filenames)
