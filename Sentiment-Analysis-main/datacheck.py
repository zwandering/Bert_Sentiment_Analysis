import torch
import time
import argparse
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
import transformers.optimization
from matplotlib import pyplot as plt
from transformers import BertModel
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='bert-base-multilingual-cased', type=str, help='bert model')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
parser.add_argument('--reinit_layers', default=0, type=int, help='reinit layers')
parser.add_argument('--weight_decay', default=False, type=bool, help='weight decay')
parser.add_argument('--reinit_pooler', default=False, type=bool, help='weight decay')
parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
parser.add_argument('--pooling', default='cls', type=str, help='pooling layer type')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d:%(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# tree = ET.parse('data/sample.positive - 副本.xml')
# root = tree.getroot()
# for child in root:
#     print(child.text)
   #  print(child.tag)
data_worse = pd.read_xml('data/sample.negative.xml')
# print(data_worse)
data_worse['label'] = 0

for i,sent in enumerate(data_worse.review.values):
    if len(sent) > 100:
        data_worse.drop([i],inplace = True)

data_worse_en = pd.read_xml('data/en_sample_data/sample.negative.xml')
# print(data_worse)
data_worse_en['label'] = 0


# data_bad = pd.read_csv('data/2.csv')
# data_bad['label'] = 1
# data_normal = pd.read_csv('data/3.csv')
# data_normal['label'] = 2
# data_good = pd.read_csv('data/4.csv')
# data_good['label'] = 3
data_better = pd.read_xml('data/sample.positive.xml')
data_better['label'] = 1


data_better_en = pd.read_xml('data/en_sample_data/sample.positive.xml')
data_better_en['label'] = 1


print(len(data_worse),len(data_better),len(data_worse_en),len(data_better_en))
# 连接每个数据集作为训练集
# data = pd.concat([data_worse[:1000], data_bad[:1000], data_normal[:1000], data_good[:1000], data_better[:1000]], axis=0).reset_index(drop=True)
data = pd.concat([data_worse[:-1],data_worse_en[:-1], data_better[:-1], data_better_en[:-1]], axis=0).reset_index(drop=True)

# for i,sent in enumerate(data.review.values):
#     if len(sent) > 1870:
#         data.drop([i],inplace = True)
ALL_LEN = ([len(sent) for sent in data.review.values])
print(np.mean(np.array(ALL_LEN) < 1870))
# plt.hist(ALL_LEN, bins=30)
# plt.show()

MAX_LEN = max([len(sent) for sent in data.review.values])
print(MAX_LEN)