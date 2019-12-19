#%% import
import torchtext
from torchtext.vocab import Vectors
import torch
import numpy as np
import random

#%% fix random seed
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(531113)

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
 torch.cuda.manual_seed(531113)

#%% init constant
BATCH_SIZE = 128
EMBEDDING = 650
MAX_VOCAB_SIZE = 50000
EMBEDDING_SIZE = 650
NUM_LAYERS = 1
TEXT = torchtext.data.Field(lower=True)

train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
 path='../text8', train='text8.train.txt', validation='text8.dev.txt', test='text8.test.txt', text_field=TEXT
)

TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
print("vocabulary size: {}".format(len(TEXT.vocab)))
#%%
VOCAB_SIZE = len(TEXT.vocab)
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), batch_size=BATCH_SIZE, device=-1, bptt_len=32, repeat=False, shuffle=True)


import torch
import torch.nn as nn

#%%
class RNNModel(nn.Module):
    """ 一个简单的循环神经网络"""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5):
        ''' 该模型包含以下几层:
            - 词嵌入层
            - 一个循环神经网络层(RNN, LSTM, GRU)
            - 一个线性层，从hidden state到输出单词表
            - 一个dropout层，用来做regularization
        '''
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.input_embed = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, batch_first=True)
        self.out_layer = nn.Linear(nhid, ntoken)
        self.nlayers = nlayers
        self.nhid = nhid

    # def init_weights(self):



    def forward(self, input, hidden):
        ''' Forward pass:
            - word embedding
            - 输入循环神经网络
            - 一个线性层从hidden state转化为输出单词表
        '''
        # input BZ * seq_len
        emb = self.input_embed(input) # emb BZ * seq_len * ninp
        #emb = self.drop(emb)

        out_put, hidden = self.rnn(emb, hidden)# out_put BZ * seq_len * nhid
        # hidden ( batch * nlayer * nhid, batch * nlayer * nhid)
        out_put = self.drop(out_put)
        out_emb = self.out_layer(out_put.reshape(-1, out_put.size(2))) # out_put (BZ * seq_len )* ntoken
        out_emb = out_emb.reshape(out_put.size(0), out_put.size(1), out_emb.size(1))
        return out_emb, hidden


    def init_hidden(self, requires_grad=True):
        weight = next(self.parameters())
        weight = weight.new_zeros(( self.nlayers, BATCH_SIZE, self.nhid), requires_grad=requires_grad)
        return (weight, weight)


model = RNNModel('LSTM', VOCAB_SIZE, EMBEDDING_SIZE, EMBEDDING_SIZE,NUM_LAYERS)
if USE_CUDA:
    model = model.cuda()
# Remove this part
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
#
#%%
def evaluate(model, val_iter):
    model.eval()
    tot_loss = 0
    tot_count = 0
    hidden = model.init_hidden(requires_grad=False)
    with torch.no_grad():
        for i, batch in enumerate(val_iter):
            data, target = batch.text.T, batch.target.T
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            # data BZ * seq_len   target BZ *seq_len
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)  # ouput BZ * seq_len * ntoken
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            tot_loss += loss.item() * np.multiply(*data.shape)
            tot_count += np.multiply(*data.shape)
    loss_mean = tot_loss / tot_count
    model.train()
    return loss_mean



GRAD_CLIP = 1
EPOCH = 1
loss_fn = nn.CrossEntropyLoss()
model.train()
hidden = model.init_hidden()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
val_losses = []
for epoch in range(EPOCH):
    for i, batch in enumerate(train_iter):
        data, target = batch.text.T, batch.target.T
        if USE_CUDA:
            data, target =data.cuda(), target.cuda()
        # data BZ * seq_len   target BZ *seq_len
        hidden = repackage_hidden(hidden)
        output,hidden = model(data,hidden) # ouput BZ * seq_len * ntoken
        model.zero_grad()
        loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
        #loss.backward(retain_graph=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if i % 1000 == 0:
            print('iter: {}, loss: {}, lr :{}'.format(i, loss, scheduler.get_lr()))

        if i % 4000 == 0:
            val_loss = evaluate(model, val_iter) # val_loss float
            if val_losses == [] or val_loss < min(val_losses):
                val_losses.append(val_loss)
                torch.save(model, 'LM-best-model.pkl')
                print('save best model iter: {}, loss: {:.5f}'.format(i, loss))
            else:
                scheduler.step()
                print('scheduler worked lr: {}'.format(scheduler.get_lr()))

#
#
# def evaluate(model, data):
#  model.eval()
#  total_loss = 0.
#  it = iter(data)
#  total_count = 0.
#  with torch.no_grad():
#   hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
#   for i, batch in enumerate(it):
#    data, target = batch.text, batch.target
#    if USE_CUDA:
#     data, target = data.cuda(), target.cuda()
#    hidden = repackage_hidden(hidden)
#    with torch.no_grad():
#     output, hidden = model(data, hidden)
#    loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
#    total_count += np.multiply(*data.size())
#    total_loss += loss.item() * np.multiply(*data.size())
#
#  loss = total_loss / total_count
#  model.train()
#  return loss
#
#


# loss_fn = nn.CrossEntropyLoss()
# learning_rate = 0.001
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
#
# import copy
#
# GRAD_CLIP = 1.
# NUM_EPOCHS = 2
#
# val_losses = []
# for epoch in range(NUM_EPOCHS):
#  model.train()
#  it = iter(train_iter)
#  hidden = model.init_hidden(BATCH_SIZE)
#  for i, batch in enumerate(it):
#   data, target = batch.text, batch.target
#   if USE_CUDA:
#    data, target = data.cuda(), target.cuda()
#   hidden = repackage_hidden(hidden)
#   model.zero_grad()
#   output, hidden = model(data, hidden)
#   loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
#   loss.backward()
#   torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
#   optimizer.step()
#   if i % 1000 == 0:
#    print("epoch", epoch, "iter", i, "loss", loss.item())
#
#   if i % 10000 == 0:
#    val_loss = evaluate(model, val_iter)
#
#    if len(val_losses) == 0 or val_loss < min(val_losses):
#     print("best model, val loss: ", val_loss)
#     torch.save(model.state_dict(), "lm-best.th")
#    else:
#     scheduler.step()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#    val_losses.append(val_loss)
#
# best_model = RNNModel("LSTM", VOCAB_SIZE, EMBEDDING_SIZE, EMBEDDING_SIZE, 2, dropout=0.5)
# if USE_CUDA:
#     best_model = best_model.cuda()
# best_model.load_state_dict(torch.load("lm-best.th"))
#
#
# val_loss = evaluate(best_model, val_iter)
# print("perplexity: ", np.exp(val_loss))
#
#
# test_loss = evaluate(best_model, test_iter)
# print("perplexity: ", np.exp(test_loss))
#
# hidden = best_model.init_hidden(1)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input = torch.randint(VOCAB_SIZE, (1, 1), dtype=torch.long).to(device)
# words = []
# for i in range(100):
#     output, hidden = best_model(input, hidden)
#     word_weights = output.squeeze().exp().cpu()
#     word_idx = torch.multinomial(word_weights, 1)[0]
#     input.fill_(word_idx)
#     word = TEXT.vocab.itos[word_idx]
#     words.append(word)
# print(" ".join(words))



