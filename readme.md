

## 当前的工作

请在你看到这个readme文件所在目录创建工程！由于一些包写的很烂导致它读数据是在上一层目录data读入的，不过现在版本预训练模型已经被去掉了。

#### 环境安装

注意torch要自己看显卡安

```shell
pip install -r .\requirements.txt
```

#### 文件说明

dataProcess.py  数据预处理，主要工作是还原英语单词，去除停用词等,会生成newTest.txt  和newTrain.txt文件，第一次用会下载一个词典包，开梯子等一会。

main.py 主文件，一键运行训练模型，如果想看过程中loss变化过程，请进入train_ch13包内函数改一下，这个包写的真的很烂

net.py 模型文件，想要换模型可以在这里改

tools.py 工具文件，核心函数是getData，用去读取数据并且整理数据格式 并且给出映射表

word2Vec 词向量训练文件，因为某些垃圾包莫名其妙的路径导致暂时用不了，用于提前训练自己的词向量改进模型，不过没有这一步不耽误模型使用。

## 目前可以的改进方向

1 目前是全监督，还没有用半监督的数据，期待群友的改进



2 目前这版本数据我发现严重的过拟合，训练准确率要比测试高很多，



3 并没有进行预先的词向量训练，尝试过调预训练模型，不知何种原因会导致收敛到0.5（结果全输出1/0，摆烂）

相关代码在main 注释部分，目前预训练数据在data文件夹里

``` python
#glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
#embeds = glove_embedding[idx_to_word]
#net.embedding.weight.data.copy_(embeds)
#net.embedding.weight.requires_grad = False
```



## 参考资料

[自然语言处理实战——Pytorch实现基于LSTM的情感分析(LMDB)——详细_自然语言处理情感分析代码csdn-CSDN博客](https://blog.csdn.net/m0_53328738/article/details/128367345)