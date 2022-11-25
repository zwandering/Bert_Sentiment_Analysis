import torch
import time
import numpy as np
import pandas as pd
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element


tree = ET.parse('data/sample.positive - 副本.xml')
root = tree.getroot()
# for child in root:
#     print(child.text)
   #  print(child.tag)
data_worse = pd.read_xml('data/sample.negative - 副本.xml')
print(data_worse)
data_worse['label'] = -1
# data_bad = pd.read_csv('data/2.csv')
# data_bad['label'] = 1
# data_normal = pd.read_csv('data/3.csv')
# data_normal['label'] = 2
# data_good = pd.read_csv('data/4.csv')
# data_good['label'] = 3
data_better = pd.read_xml('data/sample.positive - 副本.xml')
data_better['label'] = 1


"""
# 以下注释都是来自于训练集等量的情况
# 来自不同电影的5星影评,大部分归为3星
data_test_better = pd.read_csv('data/test5.csv')
data_test_better['label'] = 4
# 来自不同电影的4星影评,大部分归为3星
data_test_good = pd.read_csv('data/test4.csv')
data_test_good['label'] = 3
# 来自不同电影的3星影评，大部分归为1星,看了看确实大部分都感觉是差评...
data_test_normal = pd.read_csv('data/test3.csv')
data_test_normal['label'] = 2
# 来自不同电影的1星影评,发现一个神奇的事情，大部分归到3星去了
data_test_worse = pd.read_csv('data/test1.csv')
data_test_worse['label'] = 0
# 来自不同电影的2星影评
data_test_bad = pd.read_csv('data/test2.csv')
data_test_bad['label'] = 1
"""


# 连接每个数据集作为训练集
# data = pd.concat([data_worse[:1000], data_bad[:1000], data_normal[:1000], data_good[:1000], data_better[:1000]], axis=0).reset_index(drop=True)
data = pd.concat([data_worse[:1000], data_better[:1000]], axis=0).reset_index(drop=True)
print(data.size)


"""
将随机将整个训练数据分为两组：一组包含90%的数据当作训练集和一组包含10%的数据当作测试集。
也就是9000-1000
"""
X = data.review.values  # comment
y = data.label.values  # label自己给的0 1 2

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.1)
print(len(X), len(X_train))

"""
# 这里就是把test n，来自不同电影的n星影评当作测试集来测试一下准确率，纯label n的准确率
X_train = data.comment.values
y_train = data.label.values
X_test = data_test_normal[1000:2000].comment.values
y_test = data_test_normal[1000:2000].label.values
# X_test = np.concatenate([X_test, data_test_normal[:1000].comment.values], axis=0)
"""

"""小数据集训练，效果很差
X_train = np.concatenate([X_train, data_good.comment.values[0:10]], axis=0)
y_train = np.concatenate([y_train, data_good.label.values[0:10]], axis=0)
X_test = np.concatenate([X_test, data_good.comment.values[10:]], axis=0)
y_test = np.concatenate([y_test, data_good.label.values[10:]], axis=0)
"""
# 看一下测试集的comment 和 label
# print(X_test)
# print(y_test)
"""
用GPU训练
这里要注意一个大问题，要用GPU跑，我之前用的CPU跑一个batch 32 跑了5分钟，GPU才10来秒，这里torch和cuda的版本要对应才行
"""

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

"""BERT Tokenizer
为了应用预先训练好的BERT，我们必须使用库提供的标记器。
这是因为 （1）模型有一个特定的、固定的词汇表，
        （2）Bert标记器有一种处理词汇表外词汇的特殊方法

此外，我们需要在每个句子的开头和结尾添加特殊标记，将所有句子填充并截断为一个固定长度，
并明确指定使用“注意掩码”填充标记的内容。

# encode_plus 作用:
# (1) 标记句子
# (2) 添加[CLS]开头和[SEP]结尾
# (3) 将句子填充或截断到最大长度
# (4) 把token映射到ID上
# (5) 创建 attention mask
# (6) 返回输出字典
"""

# 加载bert的tokenize方法
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)


# 进行token,预处理
def preprocessing_for_bert(data):
    # 空列表来储存信息
    input_ids = []
    attention_masks = []

    # 每个句子循环一次
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # 预处理语句
            add_special_tokens=True,  # 加 [CLS] 和 [SEP]
            max_length=MAX_LEN,  # 截断或者填充的最大长度
            padding='max_length',  # 填充为最大长度，这里的padding在之间可以直接用pad_to_max但是版本更新之后弃用了，老版本什么都没有，可以尝试用extend方法
            return_attention_mask=True  # 返回 attention mask
        )

        # 把输出加到列表里面
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # 把list转换为tensor
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


"""
在tokenize之前，我们需要指定句子的最大长度。
"""

# Encode 我们连接的数据
encoded_comment = [tokenizer.encode(sent, add_special_tokens=True) for sent in data.review.values]

# 找到最大长度
# max_len = max([len(sent) for sent in encoded_comment])
# print('Max length: ', max_len)  68

"""
现在我们开始tokenize数据
"""
# 文本最大长度
MAX_LEN = max([len(sent) for sent in encoded_comment])
print(MAX_LEN)
# MAX_LEN = 40

# 在train，test上运行 preprocessing_for_bert 转化为指定输入格式
train_inputs, train_masks = preprocessing_for_bert(X_train)
test_inputs, test_masks = preprocessing_for_bert(X_test)

"""Create PyTorch DataLoader

我们将使用torch DataLoader类为数据集创建一个迭代器。这将有助于在训练期间节省内存并提高训练速度。

"""
# 转化为tensor类型

train_labels = torch.tensor(y_train)
test_labels = torch.tensor(y_test)


# 用于BERT微调, batch size 16 or 32较好.
batch_size = 32

# 给训练集创建 DataLoader
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
# print(len(train_data))
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
# print(train_dataloader)

# 给验证集创建 DataLoader
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
