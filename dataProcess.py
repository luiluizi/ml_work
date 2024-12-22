# coding: utf-8
import torch
import re
import snowballstemmer  # 词干提取算法的最佳实践的选择 【 nltk库：Porter、Snowball、Lancaster 】
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
# cpu or gpu
print("downloading  stopword")
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
print("downloading  end")
def clean_text(text):
    ## 将文本转为小写并拆分单词， 【列表】
    text = text.lower().split()

    # 停用词 【'a','about','above','after','again','against','ain','all','am','an','and','any','are','aren'...】
    stops = set(stopwords.words("english"))
    # 将停用词从文本列表终移除
    text = [w for w in text if not w in stops and len(w) >= 3]

    # 重新连成文本
    text = " ".join(text)

    # 文本特殊清洗，将标点符号和一些特殊符 替换 为空格
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    ## 再次将文本切分成列表，然后进行词干提取
    text = text.split()
    stemmer = snowballstemmer.stemmer('english')  # 定义词干提取为 snowball，语言为英文
    stemmed_words = [lemmatizer.lemmatize(word) for word in text] # 单词还原
    # stemmed_words = [stemmer.stemWord(word) for word in text]  # 遍历每个单词执行词干提取操作
    
    # 重新连成文本
    text = " ".join(stemmed_words)
    return text

def process():
    trainPath = './train.txt'
    newTrainPath = './newTrain.txt'
    testPath = './test.txt'
    newTestPath = './newTest.txt'
    trainFile = open(trainPath,'r',encoding='utf-8')
    listOfLines = trainFile.readlines()
    with open(newTrainPath,'w',encoding='utf-8') as f:

        for lines in listOfLines:
            label = lines.split("+++$+++")[0]
            text = clean_text(lines.split("+++$+++")[1])
            f.writelines(label + " +++$+++ " + text + '\n')
    print("train end")
    testFile = open(testPath, 'r', encoding='utf-8')
    listOfLines = testFile.readlines()

    with open(newTestPath, 'w', encoding='utf-8') as f:

        for lines in listOfLines:
            label = lines.split(",")[0]
            text = clean_text("".join(lines.split(",")[1:]))
            f.writelines(label + " +++$+++ " + text + '\n')
    


# 数据处理
if __name__ == '__main__':
    process()
    nolabelPath = './nolabel.txt'
    newnolabelPath = './newNolabel.txt'

    testFile = open(nolabelPath, 'r', encoding='utf-8')
    listOfLines = testFile.readlines()

    with open(newnolabelPath, 'w', encoding='utf-8') as f:

        for lines in listOfLines:
            label = '1'
            text = clean_text(lines)
            f.writelines(label + " +++$+++ " + text + '\n')
    







