# coding: utf-8
import torch
from torch import nn
from d2l import torch as d2l
import os
from tools import  getData
from net import BiRNN
from tools import  tokenizer, TextDataset
from trainer import Trainer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

gpu_id = 1
device = torch.device("cuda:%d" % gpu_id if torch.cuda.is_available() else "cpu")

def predict_sentiment(net, word_to_idx ,sequence):
    sequence = torch.tensor([word_to_idx[i] for i in tokenizer(sequence)], device=device)
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return label

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])

def load_glove_embeddings(glove_file, word_to_idx, embedding_dim=100):
    """加载 GloVe 词向量并返回一个张量，映射到 vocab 中的词"""
    embeddings = np.random.randn(len(word_to_idx), embedding_dim)  # 用随机初始化填充
    i = 0
    # 加载 GloVe 词向量
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            if word in word_to_idx:  # 只考虑 vocab 中的词
                embeddings[word_to_idx[word]] = vector
                i+=1

    print(len(word_to_idx), i)
    # 转换为 PyTorch tensor
    return torch.tensor(embeddings)

if __name__ == "__main__":
    batch_size = 512
    train_loader, val_loader, no_label_loader, vocab, train_tokenized,idx_to_word,word_to_idx = getData(random_seed=2406332)

    embed_size, num_hiddens, num_layers = 100, 100, 3
    net = BiRNN(len(vocab)+1, embed_size, num_hiddens, num_layers)

    net.apply(init_weights)

    glove_embedding = load_glove_embeddings('../data/glove.6B.100d/vec.txt', word_to_idx, 100)
    net.embedding.weight.data.copy_(glove_embedding)
    net.embedding.weight.requires_grad = False

    lr, num_epochs = 0.001, 20
    loss = nn.CrossEntropyLoss(reduction="none")
    # d2l.train_ch13(net, train_loader, val_loader, loss, trainer, num_epochs,devices)


    trainer = Trainer(model=net, device=device, criterion=loss, lr=lr)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # 训练基本模型
        loss = trainer.train_epoch(train_loader)
        
        # 生成伪标签
        if epoch >= 10:  # 等模型稳定后再使用伪标签
            # trainer.update_lr(lr/2)
            pseudo_texts, pseudo_labels = trainer.generate_pseudo_labels(no_label_loader)
            if pseudo_texts:
                pseudo_dataset = TextDataset(pseudo_texts, pseudo_labels)
                pseudo_loader = DataLoader(pseudo_dataset, batch_size=no_label_loader.batch_size*4, shuffle=True)
                
                # 使用伪标签继续训练
                loss = trainer.train_epoch(train_loader, pseudo_loader)
        
        print(f"Loss: {loss:.4f}")
        
        # 在验证集上评估（如果有的话）
        val_acc = trainer.evaluate(val_loader)
        print(f"Validation Accuracy: {val_acc:.4f}")



    # @save
    trainFile = open('./newTest.txt', 'r', encoding='utf-8')
    listOfLines = trainFile.readlines()

    with open('submission_{}.csv'.f(gpu_id), "wb") as f:
        f.write(b"index,label\n")
        for i in range(len(listOfLines)):
            line = listOfLines[i]
            if len(line) >0 and line[0] != "":
                label = predict_sentiment(net,word_to_idx,line.split("+++$+++")[1]).item()
                f.write((str(i) + ',' + str( label) + '\n').encode())
    print("finish")

