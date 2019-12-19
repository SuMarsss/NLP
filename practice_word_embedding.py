# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch.nn.parameter import Parameter
import time
from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
print('starting ',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
USE_CUDA = torch.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)

# %%
# 设定一些超参数

K = 100  # number of negative samples
C = 3  # nearby words threshold
NUM_EPOCHS = 2  # The number of epochs of training
MAX_VOCAB_SIZE = 30000  # the vocabulary size
BATCH_SIZE = 8192  # the batch size
LEARNING_RATE = 0.2  # the initial learning rate
EMBEDDING_SIZE = 100

LOG_FILE = "word-embedding.log"


# tokenize函数，把一篇文本转化成一个个单词
def word_tokenize(text):
    return text.split()


with open("../text8/text8.train.txt", "r") as fin:
    text = fin.read()

# %%
# 文本text构建词汇表vocab <unk>
# 由于单词数量可能太大，我们只选取最常见的MAX_VOCAB_SIZE个单词
# 我们添加一个UNK单词表示所有不常见的单词
 # 增加功能 idx_to_word word_to_idx word_counts
 # word_freqs # 用来做 negative sampling
text = [w for w in word_tokenize(text.lower())]
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE-1))
vocab['<unk>'] = len(text) - np.sum(list(vocab.values()))
idx2word = [w for w in vocab.keys()]
word2idx = dict({word:idx for idx,word in enumerate(idx2word)})
word_count = np.sum(list(vocab.values()))
word_freqs = np.array([(count/word_count)**(3./4.) for count in vocab.values()])
z_freqs = np.sum(word_freqs)
word_freqs = word_freqs/z_freqs

'''实现Dataloader
一个dataloader需要以下内容：

把所有text编码成数字，然后用subsampling预处理这些文字。
保存vocabulary，单词count，normalized word frequency
每个iteration sample一个中心词
根据当前的中心词返回context单词
根据中心词sample一些negative单词
返回单词的counts
这里有一个好的tutorial介绍如何使用PyTorch dataloader. 为了使用dataloader，我们需要定义以下两个function:
# __len__ function需要返回整个数据集中有多少个item
# __get__ 根据给定的index返回一个item
# 有了dataloader之后，我们可以轻松随机打乱整个数据集，拿到一个batch的数据等等。'''
# 构造dataset 返回center_word_label context_words_labels neg_words_labels
# 重写__getitem__ 文本text用来训练skip gram
# neg_words_labels使用multinomial
#%%
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, idx2word, word2idx, word_freqs):
        ''' text: a list of words, all text from the training dataset
            word_to_idx: the dictionary from word to idx
            idx_to_word: idx to word mapping
            word_freq: the frequency of each word
            word_counts: the word counts
        '''
        super().__init__()
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.word_freqs = torch.tensor(word_freqs)
        # self.text_idx2vocab_idx = #传入text
        self.text_idx2vocab_idx = [self.word2idx.get(w, MAX_VOCAB_SIZE-1) for w in text]
        self.text_idx2vocab_idx = torch.tensor(self.text_idx2vocab_idx).long()

    def __len__(self):
        ''' 返回整个数据集（所有单词）的长度
        '''
        return len(text)

    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的(positive)单词
            - 随机采样的K个单词作为negative sample
            - idx 表示文字在文本text中的位置
            - center_word, pos_words, neg_words为vocab词汇表中的索引
        '''
        center_word_label = self.text_idx2vocab_idx[idx] # BZ
        context_words_text_idx = list(range(idx-C,idx))+list(range(idx+1,idx+C+1))
        context_words_text_idx = [i % len(text) for i in context_words_text_idx]
        context_words_labels = self.text_idx2vocab_idx[context_words_text_idx] # BZ *  2C
        neg_words_labels = torch.multinomial(self.word_freqs,num_samples=K * context_words_labels.shape[0], replacement=True)
        # BZ * (K*2C)
        if USE_CUDA:
            center_word_label = center_word_label.cuda()
            context_words_labels = context_words_labels.cuda()
            neg_words_labels = neg_words_labels.cuda()
        return center_word_label, context_words_labels, neg_words_labels

# 实例化WordEmbeddingDataset，然后传入DataLoader
dataset =  WordEmbeddingDataset(text, idx2word, word2idx, word_freqs)
dataloader = tud.DataLoader(dataset, BATCH_SIZE, shuffle=True)

# %% class EmbeddingModel(nn.Module):
# 构建Embedding layer 注意初始化 ***本项目核心模型 ***
# forward返回loss，因为归一化因子需要负采样，所以forward不能直接返回softmax的结果
# 需要手动计算

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        ''' 初始化输出和输出embedding
        '''
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.input_embedding = nn.Embedding(vocab_size, embed_size, sparse=False) # vocab_size * embed_size
        self.output_embedding = nn.Embedding(vocab_size, embed_size, sparse=False)
        # self.embed_weight_init()

    def embed_weight_init(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, center_word_label, context_words_labels, neg_words_labels):
        '''
        center_word_label: 中心词, [batch_size]
        context_words_labels: 中心词周围 context window 出现过的单词 [batch_size * (window_size * 2)]
        neg_words_labels: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (window_size * 2 * K)]

        return: loss, [batch_size]
        '''
        center_word_embeds = self.input_embedding(center_word_label) #  BZ * embed_size
        context_words_embeds = self.output_embedding(context_words_labels) # BZ*2C*embed_size
        neg_words_embeds =  self.output_embedding(neg_words_labels) # BZ *2KC * embed_size

        context_prob = torch.bmm(context_words_embeds, center_word_embeds.unsqueeze(2))
        # BZ * 2C * 1
        neg_prob = torch.bmm(neg_words_embeds, center_word_embeds.unsqueeze(2))
        # BZ * 2KC *1
        context_prob = context_prob.sum(1).squeeze(1) # 1 dim BZ
        neg_prob = neg_prob.sum(1).squeeze(1) # 1 dim

        context_prob = F.logsigmoid(context_prob)
        neg_prob = F.logsigmoid(- neg_prob)
        loss = context_prob + neg_prob # BZ
        return  -loss

    def input_embeddings(self):
        return self.input_embedding.weight.cpu().numpy()

model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()

fout = open(LOG_FILE, 'a')
#train
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
for i, (center_word_label, context_words_labels, neg_words_labels) in enumerate(dataloader):
    loss = model(center_word_label.cuda(), context_words_labels.cuda(),neg_words_labels.cuda() )
    loss = loss.mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    fout.write("iter: {} loss: {} \n".format(i, loss.item()))
    print('iter: ',i, 'loss: ',loss.item())
    if i > 50:
        break
fout.close()


# model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
# if USE_CUDA:
#     model = model.cuda()
#
# def evaluate(filename, embedding_weights):
#     if filename.endswith(".csv"):
#         data = pd.read_csv(filename, sep=",")
#     else:
#         data = pd.read_csv(filename, sep="\t")
#     human_similarity = []
#     model_similarity = []
#     for i in data.iloc[:, 0:2].index:
#         word1, word2 = data.iloc[i, 0], data.iloc[i, 1]
#         if word1 not in word_to_idx or word2 not in word_to_idx:
#             continue
#         else:
#             word1_idx, word2_idx = word_to_idx[word1], word_to_idx[word2]
#             word1_embed, word2_embed = embedding_weights[[word1_idx]], embedding_weights[[word2_idx]]
#             model_similarity.append(float(sklearn.metrics.pairwise.cosine_similarity(word1_embed, word2_embed)))
#             human_similarity.append(float(data.iloc[i, 2]))
#
#     return scipy.stats.spearmanr(human_similarity, model_similarity)# , model_similarity
#
# def find_nearest(word):
#     index = word_to_idx[word]
#     embedding = embedding_weights[index]
#     cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
#     return [idx_to_word[i] for i in cos_dis.argsort()[:10]]
#
#
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
# for e in range(NUM_EPOCHS):
#     for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
#
#         # TODO
#         input_labels = input_labels.long()
#         pos_labels = pos_labels.long()
#         neg_labels = neg_labels.long()
#         if USE_CUDA:
#             input_labels = input_labels.cuda()
#             pos_labels = pos_labels.cuda()
#             neg_labels = neg_labels.cuda()
#
#         optimizer.zero_grad()
#         loss = model(input_labels, pos_labels, neg_labels).mean()
#         loss.backward()
#         optimizer.step()
#
#         if i % 100 == 0:
#             with open(LOG_FILE, "a") as fout:
#                 fout.write("epoch: {}, iter: {}, loss: {}\n".format(e, i, loss.item()))
#                 print("epoch: {}, iter: {}, loss: {}".format(e, i, loss.item()))
#
#         if i % 2000 == 0:
#             embedding_weights = model.input_embeddings()
#             sim_simlex = evaluate("simlex-999.txt", embedding_weights)
#             sim_men = evaluate("men.txt", embedding_weights)
#             sim_353 = evaluate("wordsim353.csv", embedding_weights)
#             with open(LOG_FILE, "a") as fout:
#                 print("epoch: {}, iteration: {}, simlex-999: {}, men: {}, sim353: {}, nearest to monster: {}\n".format(
#                     e, i, sim_simlex, sim_men, sim_353, find_nearest("monster")))
#                 fout.write(
#                     "epoch: {}, iteration: {}, simlex-999: {}, men: {}, sim353: {}, nearest to monster: {}\n".format(
#                         e, i, sim_simlex, sim_men, sim_353, find_nearest("monster")))
#
#     embedding_weights = model.input_embeddings()
#     np.save("embedding-{}".format(EMBEDDING_SIZE), embedding_weights)
#     torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))
#
#     model.load_state_dict(torch.load("embedding-{}.th".format(EMBEDDING_SIZE)))
#
#     embedding_weights = model.input_embeddings()
#     print("simlex-999", evaluate("simlex-999.txt", embedding_weights))
#     print("men", evaluate("men.txt", embedding_weights))
#     print("wordsim353", evaluate("wordsim353.csv", embedding_weights))
#
#     for word in ["good", "fresh", "monster", "green", "like", "america", "chicago", "work", "computer", "language"]:
#         print(word, find_nearest(word))
#
#     man_idx = word_to_idx["man"]
#     king_idx = word_to_idx["king"]
#     woman_idx = word_to_idx["woman"]
#     embedding = embedding_weights[woman_idx] - embedding_weights[man_idx] + embedding_weights[king_idx]
#     cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
#     for i in cos_dis.argsort()[:20]:
#         print(idx_to_word[i])
#     print('finished ',time.strftime( '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))