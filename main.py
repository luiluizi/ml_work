# coding: utf-8
import torch
from torch import nn
import os
import tools
from tools import  getData
from net import BiRNN
from tools import  tokenizer, TextDataset
from trainer import Trainer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--embed_size', type=int, default=100)
parser.add_argument('--num_hiddens', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--pseudo_threshold', type=float, default=0.95) # 伪标签置信度

args = parser.parse_args()
gpu_id = args.gpu_id
num_epochs = args.num_epochs
lr = args.lr
batch_size = args.batch_size
embed_size = args.embed_size
num_hiddens = args.num_hiddens
num_layers = args.num_layers
pseudo_threshold = args.pseudo_threshold

device = torch.device("cuda:%d" % gpu_id if torch.cuda.is_available() else "cpu")

def predict_sentiment(net, word_to_idx, vocab ,sequence):
    sequence = torch.tensor([word_to_idx[i] for i in tokenizer(sequence)], device=device, dtype=torch.long)
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
    glove_vectors = []
    # 加载 GloVe 词向量
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            if word in word_to_idx:  # 只考虑 vocab 中的词
                embeddings[word_to_idx[word]] = vector
                glove_vectors.append(vector)
                i+=1

    print(len(word_to_idx), i)

    # 计算平均向量
    glove_mean_vector = np.mean(glove_vectors, axis=0)

    # 对未命中的词汇进行处理，使用平均向量
    for word, idx in word_to_idx.items():
        if np.all(embeddings[idx] == np.random.randn(embedding_dim)):  # 如果是随机初始化的向量
            embeddings[idx] = glove_mean_vector
    # 转换为 PyTorch tensor
    return torch.tensor(embeddings)


def load_model(epoch):
    model_path = './cache/model_{}.tar'.format(epoch)
    print(model_path)
    assert os.path.exists(model_path)
    checkpoint = torch.load(model_path, map_location='cpu')
    return checkpoint['model_state_dict']

def save_best_model_with_epoch(best_config, epoch):
    cache_name = f'./cache/'
    tools.ensure_dir(cache_name)
    model_path = cache_name + '/' + 'model_{}.tar'.format(epoch)
    torch.save(best_config, model_path)

if __name__ == "__main__":
    train_loader, val_loader, no_label_loader, test_loader, vocab, train_tokenized,idx_to_word,word_to_idx = getData(random_seed=2406332)

    net = BiRNN(len(vocab)+1, embed_size, num_hiddens, num_layers)

    net.apply(init_weights)

    # 加载glove_embedding
    glove_embedding = load_glove_embeddings('../data/glove.6B.100d/glove.twitter.27B.100d.txt', word_to_idx, 100)
    net.embedding.weight.data.copy_(glove_embedding)
    net.embedding.weight.requires_grad = False

    loss = nn.CrossEntropyLoss(reduction="none")

    trainer = Trainer(model=net, device=device, criterion=loss, lr=lr)

    min_val_loss = 0.0
    best_config = dict()
    best_eopch = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # 训练基本模型
        loss = trainer.train_epoch(train_loader)
        
        # 生成伪标签
        # if epoch >= 20:  # 等模型稳定后再使用伪标签
        #     # trainer.update_lr(lr/2)
        #     pseudo_threshold = 0.9 + 0.05*(epoch-5)
        #     pseudo_texts, pseudo_labels = trainer.generate_pseudo_labels(no_label_loader, threshold=pseudo_threshold, max_samples_per_class=len(train_loader.dataset))
        #     # pseudo_length = min(int(0.1 * len(pseudo_texts)), len(train_loader.dataset))
        #     # pseudo_texts, pseudo_labels = pseudo_texts[:pseudo_length], pseudo_labels[:pseudo_length]
        #     if pseudo_texts:
        #         pseudo_dataset = TextDataset(pseudo_texts, pseudo_labels)
        #         pseudo_loader = DataLoader(pseudo_dataset, batch_size=no_label_loader.batch_size, shuffle=True)
        #         # 使用伪标签继续训练
        #         loss = trainer.train_epoch(train_loader, pseudo_loader)
            # trainer.update_lr(lr)
        
        print(f"Loss: {loss:.4f}")
        
        # 在验证集上评估（如果有的话）
        val_loss = trainer.evaluate(val_loader)
        if val_loss > min_val_loss:
            best_config['model_state_dict'] = trainer.model.state_dict()
            save_best_model_with_epoch(best_config, epoch)
            best_eopch = epoch + 1
            min_val_loss = val_loss
    # @save
    trainFile = open('./newTest.txt', 'r', encoding='utf-8')
    listOfLines = trainFile.readlines()

    trainer.test(test_loader=test_loader, output_file='submission.csv')
