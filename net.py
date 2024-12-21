import torch.nn as nn
import torch
import torch.nn.functional as F

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 将bidirectional设置为True以获取双向循环神经网络
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                               bidirectional=True)
        # self.decoder = nn.Linear(4 * num_hiddens, 2)

        self.fc1 = nn.Linear(4 * num_hiddens, 2 * num_hiddens)
        self.fc2 = nn.Linear(2 * num_hiddens, 2)

        self.batch_norm = nn.BatchNorm1d(4 * num_hiddens)

        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs):
        # inputs的形状是（批量大小，时间步数）
        # 因为长短期记忆网络要求其输入的第一个维度是时间维，
        # 所以在获得词元表示之前，输入会被转置。
        # 输出形状为（时间步数，批量大小，词向量维度）
        embeddings = self.embedding(inputs.T)

        embeddings = self.dropout(embeddings)

        self.encoder.flatten_parameters()
        # 返回上一个隐藏层在不同时间步的隐状态，
        # outputs的形状是（时间步数，批量大小，2*隐藏单元数）
        outputs, _ = self.encoder(embeddings)
        # 连结初始和最终时间步的隐状态，作为全连接层的输入，
        # 其形状为（批量大小，4*隐藏单元数）
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)

        encoding = self.batch_norm(encoding)
        encoding = self.dropout(F.relu(self.fc1(encoding)))
        outs = self.fc2(encoding)
        
        return outs
