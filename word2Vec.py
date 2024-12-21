from gensim.models import Word2Vec

from tools import getData

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import tqdm
import os
import time
import re
import pandas as pd
import string
import gensim
import time
import random
import snowballstemmer  # 词干提取算法的最佳实践的选择 【 nltk库：Porter、Snowball、Lancaster 】
import collections
from collections import Counter
from nltk.corpus import stopwords
from itertools import chain
from sklearn.metrics import accuracy_score

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from gensim.scripts.glove2word2vec import glove2word2vec

def word2Vec():
    train_loader,val_loader,no_label_loader,vocab,train_tokenized,idx_to_word,word_to_idx = getData()
    my_glove_100 = Word2Vec(train_tokenized,
                            vector_size= 100,# 提供给类的语料库  # 每个标记的密集向量的长度
                        min_count=1,  # 训练特定模型时可以考虑的最小单词数
                        window=5,
                        sg=1,
                        workers=4)  # 训练模型时所需的线程数

    my_glove_100.train(train_tokenized, total_examples=len(train_tokenized), epochs=1)
    path = r'./glove_my.100d.txt'
    my_glove_100.wv.save_word2vec_format(path)


if __name__ == "__main__":
    #glove_file_100 = datapath('./data/glove.6B.100d/vec.txt')
    # tmp_file_100 = get_tmpfile('./data/glove.6B.100d/wv.txt')
    # glove2word2vec('./data/glove.6B.100d/vec.txt','./data/glove.6B.100d/wv.txt')
    # wvmodel_100 = KeyedVectors.load_word2vec_format(tmp_file_100)
    word2Vec()