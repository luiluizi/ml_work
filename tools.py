# coding: utf-8
import os
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from itertools import chain
from torch.utils.data import DataLoader, Dataset
from collections import Counter

class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text_ids = torch.tensor(self.texts[idx])
        if self.labels is not None:
            return text_ids, torch.tensor(self.labels[idx])
        return text_ids

def ensure_dir(cache_dir):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

def readData(path):
    trainFile = open(path ,'r', encoding='utf-8')
    listOfLines = trainFile.readlines()
    data = []
    for lines in listOfLines:
        if lines != "":
            label = lines.split("+++$+++")[0]
            text = lines.split("+++$+++")[1]
            data.append([text,int(label)])
    return  data


def tokenizer(text):
    tokens = [tok.lower() for tok in text.split(' ') if tok.strip() and tok != '\n']
    return tokens


def encode_samples(tokenized_samples, vocab,word_to_idx):
    features = []  # 存储映射后的特征数字列表
    for sample in tokenized_samples:  # 遍历所有文本样本的分词列表
        feature = []
        for token in sample:  # 遍历某个样本的分词列表
            if token in word_to_idx:  # 如果该单词在词汇表，匹配相对应的数字映射
                feature.append(word_to_idx[token])
            else:
                feature.append(0)  # 不存在则为0

        features.append(feature)
    return features


def pad_samples(features, maxlen=100, PAD=0):
    padded_features = []  # 存储填充后的数据
    max = 0
    for feature in features:  # 遍历每个样本的特征数字列表，进行操作
        if len(feature) >= maxlen:  # 超出最大长度则截取最大长度前面的
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature  # 不足则填充到最大长度
            while (len(padded_feature) < maxlen):
                padded_feature.append(PAD)
        padded_features.append(padded_feature)
    return padded_features


def getData(valid_size = 0.1,batch_size = 64,random_seed = 20373222):
    ####  param
    # valid_size 采样比例
    # batch_size  批大小
    # random_seed 采样种子 用于复现

    # 做所有预处理等操作 传出dataset对象
    train_data = readData('./newTrain.txt')
    print("load data end")

    test = open('./newTest.txt', 'r', encoding='utf-8')
    listOfLines = test.readlines()
    test_data = []
    for lines in listOfLines:
        if lines != "":
            text = lines.split("+++$+++")[1]
            test_data.append(text)

    nolabel = open('./newNolabel.txt', 'r', encoding='utf-8')
    listOfLines = nolabel.readlines()
    nolabel_data = []
    for lines in listOfLines:
        if lines != "":
            text = lines.split("+++$+++")[1]
            nolabel_data.append(text)

    train_tokenized = []  # 存储训练数据的文本分词
    test_tokenized = []
    nolabel_tokenized = []

    train_data = train_data
    test_data = test_data
    nolabel_data = nolabel_data

    # 遍历训练数据集文本，执行分词操作
    for review, score in train_data:
        train_tokenized.append(tokenizer(review))
    for text in test_data:
        test_tokenized.append(tokenizer(text))
    for text in nolabel_data:
        nolabel_tokenized.append(tokenizer(text))

    # 统计词频
    all_tokens = list(chain(*train_tokenized, *test_tokenized, *nolabel_tokenized))
    token_counts = Counter(all_tokens)
    min_freq = 5
    vocab = {word for word, count in token_counts.items() if count >= min_freq}

    vocab = set(vocab)
    vocab_size = len(vocab)
    print('词汇量: ', vocab_size)
    # 为词表vocab中的每个单词映射一个数字标签
    word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
    word_to_idx['<unk>'] = 0
    # 为每个数字映射一个单词
    idx_to_word = {i + 1: word for i, word in enumerate(vocab)}
    idx_to_word[0] = '<unk>'

    features = encode_samples(train_tokenized, vocab, word_to_idx)
    train_features = torch.tensor(pad_samples(features))
    train_labels = torch.tensor([score for _, score in train_data])

    no_label_features = torch.tensor(pad_samples(encode_samples(nolabel_tokenized, vocab, word_to_idx)))
    test_features = torch.tensor(pad_samples(encode_samples(test_tokenized, vocab, word_to_idx)))

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)

    no_label_set = TextDataset(no_label_features)

    test_set = TextDataset(test_features)

    num_train = len(train_set)
    num_val = np.floor(valid_size * num_train)
    split = int(num_val)
    indices = list(range(num_train))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    val_idx = indices[:split]
    train_idx = indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # 设置数据加载器
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_set,
                                             batch_size=batch_size,
                                             sampler=val_sampler)
    no_label_loader = torch.utils.data.DataLoader(no_label_set,
                                             batch_size=batch_size*4,
                                             shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                             batch_size=batch_size,
                                             shuffle=False)
    print("数据读取结束")

    res_tokenized = train_tokenized + test_tokenized + nolabel_tokenized
    return train_loader,val_loader,no_label_loader, test_loader,vocab,res_tokenized,idx_to_word,word_to_idx

if __name__ == "__main__":
    getData()