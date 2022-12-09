import torch
import os
import time
import argparse
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
import transformers.optimization
from transformers import BertModel
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='bert-base-multilingual-cased', type=str, help='bert model')
parser.add_argument('--batch_size', default=24, type=int, help='batch size')
parser.add_argument('--dropout', default=0.47, type=float, help='dropout probability')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
parser.add_argument('--reinit_layers', default=4, type=int, help='reinit layers')
parser.add_argument('--weight_decay', default=True, type=bool, help='weight decay')
parser.add_argument('--reinit_pooler', default=False, type=bool, help='reinit_pooler')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--pooling', default='last-avg', type=str, help='pooling layer type')
parser.add_argument('--activation', default='relu', type=str, help='activation layer type')
parser.add_argument('--out_dir', default='./model', type=str, help='output direction of the model to be saved')
parser.add_argument('--output', default='./output.txt', type=str, help='the file that stdout be redirected to')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d:%(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# savedStdout = sys.stdout
# f = open(args.output,'w+')
# sys.stdout = f


data_worse = pd.read_xml('data/sample.negative.xml')
# print(data_worse)
data_worse['label'] = 0

# for i,sent in enumerate(data_worse.review.values):
#     if len(sent) > 100:
#         data_worse.drop([i],inplace = True)

data_worse_en = pd.read_xml('data/en_sample_data/sample.negative.xml')
# print(data_worse)
data_worse_en['label'] = 0

# for i,sent in enumerate(data_worse_en.review.values):
#     if len(sent) > 2000:
#         data_worse_en.drop([i],inplace = True)
# data_bad = pd.read_csv('data/2.csv')
# data_bad['label'] = 1
# data_normal = pd.read_csv('data/3.csv')
# data_normal['label'] = 2
# data_good = pd.read_csv('data/4.csv')
# data_good['label'] = 3
data_better = pd.read_xml('data/sample.positive.xml')
data_better['label'] = 1

# for i,sent in enumerate(data_better.review.values):
#     if len(sent) > 100:
#         data_better.drop([i],inplace = True)

data_better_en = pd.read_xml('data/en_sample_data/sample.positive.xml')
data_better_en['label'] = 1

# for i,sent in enumerate(data_better_en.review.values):
#     if len(sent) > 2000:
#         data_better_en.drop([i],inplace = True)

print(len(data_worse),len(data_better),len(data_worse_en),len(data_better_en))
# 连接每个数据集作为训练集
# data = pd.concat([data_worse[:1000], data_bad[:1000], data_normal[:1000], data_good[:1000], data_better[:1000]], axis=0).reset_index(drop=True)
data = pd.concat([data_worse[:-1],data_worse_en[:-1], data_better[:-1], data_better_en[:-1]], axis=0).reset_index(drop=True)

for i,sent in enumerate(data.review.values):
    data.loc[i,'review'] = data.loc[i,'review'].replace('\n', ' ')

for i,sent in enumerate(data.review.values):
    if len(sent) > 1870:
        data.drop([i],inplace = True)


    # if i>5000 and i<5050: print(data.loc[i].review)
MAX_LEN = max([len(sent) for sent in data.review.values])
print(MAX_LEN)
# print(data.size)


"""
将随机将整个训练数据分为两组：一组包含90%的数据当作训练集和一组包含10%的数据当作测试集。
也就是9000-1000
"""
X = data.review.values  # comment
y = data.label.values  # label自己给的0 1 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
# print(len(X), len(X_train))
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
tokenizer = BertTokenizer.from_pretrained(args.model)
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


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
            truncation= True,
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
encoded_comment = [tokenizer.encode(sent, add_special_tokens=True, truncation=True, max_length=512) for sent in data.review.values]

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
batch_size = args.batch_size

# 给训练集创建 DataLoader
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
# print(len(train_data))
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, drop_last=True)
# print(train_dataloader)

# 给验证集创建 DataLoader
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


"""关键点来了，开始训练！
Create BertClassifier

BERT base由12个transformer层组成，每个transformer层接收一系列token embedding
           注:token embedding就是文本的转化，embedding，transformer用的是随机生成然后训练

并在输出上生成相同数量的具有相同hidden size（或dim）的embedding。
[CLS] token最后transformer层的输出用作序列的特征来feed classifier

注意文本是512的限制，所以不能太大，我就是通过筛选排列得到的数据集
"""


# 自己定义的Bert分类器的类，微调Bert模型
# class BertClassifier(nn.Module):
#     def __init__(self, ):
#         """
#         freeze_bert (bool): 设置是否进行微调，0就是不，1就是调
#         """
#         super(BertClassifier, self).__init__()
#         # 输入维度(hidden size of Bert)默认768，分类器隐藏维度，输出维度(label)
#         D_in, H, D_out = 768, 100, 2
#
#         # 实体化Bert模型
#         self.bert = BertModel.from_pretrained(args.model)
#         # self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
#         # self.bert = BertModel.from_pretrained('bert-base-chinese')
#
#         # 实体化一个单层前馈分类器，说白了就是最后要输出的时候搞个全连接层
#         self.classifier = nn.Sequential(
#             nn.Linear(D_in, H),  # 全连接
#             nn.ReLU(),  # 激活函数
#             nn.Dropout(p=0.3),
#             nn.Linear(H, D_out)  # 全连接
#         )
#
#     def forward(self, input_ids, attention_mask):
#         # 开始搭建整个网络了
#         # 输入
#         outputs = self.bert(input_ids=input_ids,
#                             attention_mask=attention_mask)
#         # 为分类任务提取标记[CLS]的最后隐藏状态，因为要连接传到全连接层去
#         last_hidden_state_cls = outputs[0][:, 0, :]
#         # 全连接，计算，输出label
#         logits = self.classifier(last_hidden_state_cls)
#
#         return logits

class BertClassifier(nn.Module):
    def __init__(self, ):
        """
        freeze_bert (bool): 设置是否进行微调，0就是不，1就是调
        """
        super(BertClassifier, self).__init__()
        # 输入维度(hidden size of Bert)默认768，分类器隐藏维度，输出维度(label)
        D_in, H, D_out = 768, 300, 2

        # 实体化Bert模型
        self.bert = BertModel.from_pretrained(args.model)
        # self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        # self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dense = nn.Linear(D_in * 2, D_in)
        # 实体化一个单层前馈分类器，说白了就是最后要输出的时候搞个全连接层
        self.activation = None
        if args.activation == 'relu': self.activation = nn.ReLU()
        if args.activation == 'sigmoid': self.activation = nn.Sigmoid()
        if args.activation == 'elu': self.activation = nn.ELU()
        self.classifier = nn.Sequential(
            nn.Dropout(p=args.dropout),  # dropot的概率可以再调调
            nn.LayerNorm(D_in), # dropout 和 Layernorm顺序需要再确定
            nn.Linear(D_in,D_out),
            self.activation,
            # nn.Linear(H, D_out),  # 全连接
            # self.activation
        )

    def forward(self, input_ids, attention_mask):
        # 开始搭建整个网络了
        # 输入
        out = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        # 为分类任务提取标记[CLS]的最后隐藏状态，因为要连接传到全连接层去
        #print(outputs.shape)
        # last_hidden_state_cls = outputs[0][:, 0, :]

        if args.pooling == 'cls':
            out = out.last_hidden_state[:, 0, :]  # [batch, 768]
        elif args.pooling == 'pooler':
            out = out.pooler_output  # [batch, 768]
        elif args.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            out = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        elif args.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            out = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]
        elif args.pooling == 'mean-max-avg':
            sequence_output = out.hidden_states[-1]
            # # sequence_output的维度是[batch_size, seq_len, embed_dim]
            avg_pooled = sequence_output.mean(1)
            max_pooled = torch.max(sequence_output, dim=1)
            pooled = torch.cat((avg_pooled, max_pooled[0]), dim=1)
            out = self.dense(pooled)
        # print(last_hidden_state_cls.shape)
        # print(outputs[1].shape)
        # print(last_hidden_state_cls==outputs[1])
        # _, pooled_output = self.bert(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     return_dict = True
        # )
        # print(pooled_output.shape)
        # print(last_hidden_state_cls==pooled_output)
        # 全连接，计算，输出label
        logits = self.classifier(out)

        return logits


# 注意这个地方的logits是全连接的返回， 两个output就是01二分类，我们这里用的是ouput为3，就是老师所需要的三分类问题


"""
然后就是深度学习的老一套定义优化器还有学习率等
"""


def initialize_model(epochs=2):
    """
    初始化我们的bert，优化器还有学习率，epochs就是训练次数
    """
    # 初始化我们的Bert分类器
    bert_classifier = BertClassifier()
    # 用GPU运算
    bert_classifier.to(device)
    # 训练的总步数
    total_steps = len(train_dataloader) * epochs
    # 创建优化器
    if args.weight_decay:
        optimizer = AdamW(bert_classifier.parameters(),
                          lr=args.lr,  # 默认学习率
                          eps=1e-8,  # 默认精度
                          correct_bias=True,
                          weight_decay=0.01
                          )
    else:
        optimizer = AdamW(bert_classifier.parameters(),
                          lr=args.lr,  # 默认学习率
                          eps=1e-8,  # 默认精度
                          correct_bias=True,
                          )
    # 学习率预热
    #lambda1 = lambda epoch: 0.65 ** epoch
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=0,  # Default value
    #                                             num_training_steps=total_steps)
    scheduler = transformers.optimization.get_polynomial_decay_schedule_with_warmup(optimizer,
                                                                        num_warmup_steps=0,
                                                                        num_training_steps=total_steps,
                                                                        power=4) # 12.9 0:35 power 3->4
    return bert_classifier, optimizer, scheduler


# 实体化loss function
loss_fn = nn.CrossEntropyLoss()  # 交叉熵


# 训练模型
def train(model, train_dataloader, test_dataloader=None, epochs=2, evaluation=False):
    # 开始训练循环

    # reinit pooler-layer
    if args.reinit_pooler:
        logger.info(f"reinit pooler layer of {args.model}")
        encoder_temp = model.bert
        encoder_temp.pooler.dense.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
        encoder_temp.pooler.dense.bias.data.zero_()
        for p in encoder_temp.pooler.parameters():
            p.requires_grad = True

    # reinit encoder layers
    if args.reinit_layers > 0:
        # assert config.reinit_pooler
        logger.info(f"reinit  layers count of {str(args.reinit_layers)}")

        encoder_temp = model.bert
        for layer in encoder_temp.encoder.layer[-args.reinit_layers:]:
            for module in layer.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
    best_accuracy = 0.
    for epoch_i in range(epochs):
        # if epoch_i <4:
        #     optimizer = AdamW(bert_classifier.parameters(),
        #                       lr=2e-5,  # 默认学习率
        #                       eps=1e-8  # 默认精度
        #                       )
        # elif epoch_i <10:
        #     optimizer = AdamW(bert_classifier.parameters(),
        #                       lr=1e-5,  # 默认学习率
        #                       eps=1e-8  # 默认精度
        #                       )
        # =======================================
        #               Training
        # =======================================
        # 表头
        print(
            f"{'Epoch':^7} | {'每40个Batch':^9} | {'训练集 Loss':^12} | {'测试集 Loss':^10} | {'测试集准确率':^9} | {'时间':^9}")
        print("-" * 80)

        # 测量每个epoch经过的时间
        t0_epoch, t0_batch = time.time(), time.time()

        # 在每个epoch开始时重置跟踪变量
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # 把model放到训练模式
        model.train()

        # 分batch训练
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # 把batch加载到GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            # print(b_labels.shape)
            # 归零导数
            model.zero_grad()
            # 真正的训练
            logits = model(b_input_ids, b_attn_mask)
            # print(logits.shape)
            # 计算loss并且累加
            # _, preds = torch.max(logits, dim=1)
            # print(preds)
            # print(logits)
            # print(b_labels)
            loss = loss_fn(logits, b_labels)

            batch_loss += loss.item()
            total_loss += loss.item()
            # 反向传播
            loss.backward()
            # 归一化，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 更新参数和学习率
            optimizer.step()
            scheduler.step()

            # Print每40个batch的loss和time
            if (step % 40 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # 计算40个batch的时间
                time_elapsed = time.time() - t0_batch

                # Print训练结果
                print(
                    f"{epoch_i + 1:^7} | {step:^10} | {batch_loss / batch_counts:^14.6f} | {'-':^12} | {'-':^13} | {time_elapsed:^9.2f}")

                # 重置batch参数
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # 计算平均loss 这个是训练集的loss
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 80)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation:  # 这个evalution是我们自己给的，用来判断是否需要我们汇总评估
            # 每个epoch之后评估一下性能
            # 在我们的验证集/测试集上.
            validate_loss, validate_accuracy = evaluate(model, test_dataloader)

            # 保存当前性能最好的模型
            if validate_accuracy > best_accuracy:
                best_accuracy = validate_accuracy
                output_dir = os.path.join(args.out_dir,args.model,str(args.pooling))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_name = 'AUC'+str(validate_accuracy)+'_lr_' + str(args.lr) + '_pooling_' + str(args.pooling) + '_batch_' + str(args.batch_size)+'.pth'
                output_path = os.path.join(output_dir, output_name)
                torch.save(model, output_path)
                print("Model Saved as :"+output_path)

            # Print 整个训练集的耗时
            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^10} | {avg_train_loss:^14.6f} | {validate_loss:^12.6f} | {validate_accuracy:^12.2f}% | {time_elapsed:^9.2f}")
            print("-" * 80)
        print("\n")


# 在测试集上面来看看我们的训练效果
def evaluate(model, test_dataloader):
    """
    在每个epoch后验证集上评估model性能
    """
    # model放入评估模式
    model.eval()

    # 准确率和误差
    test_accuracy = []
    test_loss = []

    # 验证集上的每个batch
    for batch in test_dataloader:
        # 放到GPU上
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # 计算结果，不计算梯度
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)  # 放到model里面去跑，返回验证集的ouput就是一行三列的
            # label向量可能性，这个时候还没有归一化所以还不能说是可能性，反正归一化之后最大的就是了

        # 计算误差
        loss = loss_fn(logits, b_labels.long())
        test_loss.append(loss.item())

        # get预测结果，这里就是求每行最大的索引咯，然后用flatten打平成一维
        preds = torch.argmax(logits, dim=1).flatten()  # 返回一行中最大值的序号

        # 计算准确率，这个就是俩比较，返回相同的个数, .cpu().numpy()就是把tensor从显卡上取出来然后转化为numpy类型的举证好用方法
        # 最后mean因为直接bool形了，也就是如果预测和label一样那就返回1，正好是正确的个数，求平均就是准确率了
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        test_accuracy.append(accuracy)

    # 计算整体的平均正确率和loss
    val_loss = np.mean(test_loss)
    val_accuracy = np.mean(test_accuracy)

    return val_loss, val_accuracy


for k,v in sorted(vars(args).items()):
    print(k,'=',v)

bert_classifier, optimizer, scheduler = initialize_model(epochs=args.epochs)
# print("Start training and validation:\n")
print("Start training and testing:\n")
train(bert_classifier, train_dataloader, test_dataloader, epochs=args.epochs, evaluation=True)  # 这个是有评估的

net = BertClassifier()
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))


# Do testing on test dataset and generates a xml-formatted output file
# test = Test(model_path='./model.pth',input_file='input_file',output_file='output_file')
# test.test()


f.close()
sys.stdout = savedStdout


# CUDA_VISIBLE_DEVICES=0 python sentiment.py --model 'bert-base-multilingual-cased' --batch_size 16 --lr 1e-5 --weight_decay True --reinit_layers 2 power 2.7

# CUDA_VISIBLE_DEVICES=0 python sentiment.py --model 'bert-base-multilingual-cased' --batch_size 16 --lr 1e-5 --weight_decay True --reinit_layers 2 --epochs 5
# 85.1%

# CUDA_VISIBLE_DEVICES=1 python sentiment.py --model 'bert-base-multilingual-cased' --batch_size 16 --lr 1e-5 --weight_decay True --reinit_layers 2 --epochs 5
# 85.46

#CUDA_VISIBLE_DEVICES=1 python sentiment.py --model 'bert-base-multilingual-cased' --batch_size 20 --lr 1e-5 --weight_decay True --reinit_layers 2 --epochs 10 --pooling 'last-avg'
# 85.89

# CUDA_VISIBLE_DEVICES=1 python sentiment.py --model 'bert-base-multilingual-cased' --batch_size 20 --lr 1e-5 --weight_decay True --reinit_layers 4 --epochs 10 --pooling 'last-avg'
# 86.31