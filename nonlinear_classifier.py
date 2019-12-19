import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
# 手动生成一个随机的平面点分布，并画出来
np.random.seed(0)
X, y = make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show()


# 定义一个一个函数来画决策边界
def plot_decision_boundary(pred_fn, X, y ):
    # 设定最大最小值，附加一点点边缘填充
    if isinstance(X, torch.Tensor):
        X = X.data.cpu().numpy()
        y = y.data.cpu().numpy()
        pred_fn = pred_fn.cpu()
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 用预测函数预测一下
    input = np.c_[xx.ravel(), yy.ravel()]
    input = torch.Tensor(input)
    Z = pred_fn(input)# BZ * out_size
    Z = torch.max(Z, 1)[1].cpu() .data.numpy()
    Z = Z.reshape(xx.shape)

    # 然后画出图
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

    # 画出每个隐藏层单元的线性部分
    # 画出输出层的线性部分
    # 画出隐藏层空间


# from sklearn.linear_model import LogisticRegressionCV
#
# # 咱们先来瞄一眼逻辑斯特回归对于它的分类效果
# clf = LogisticRegressionCV()
# clf.fit(X, y)
#
# # 画一下决策边界
# plot_decision_boundary(lambda x: clf.predict(x))
# plt.title("Logistic Regression")
# plt.show()

import torch
from torch import nn
USE_CUDA = torch.cuda.is_available()
LR = 0.3
Epoch = 300
X = torch.tensor(X).float()
y = torch.tensor(y).long()
if USE_CUDA:
    X = X.cuda()
    y = y.cuda()
class non_linear_classifier(nn.Module):
    def __init__(self, input_size, out_size, num_hid_units):
        super().__init__()
        # input_size BZ * 2
        # output_size BZ*1
        self.input_size = input_size
        self.out_size = out_size
        self.num_hid_units = num_hid_units

        self.hid_layer_1 = nn.Linear(self.input_size, self.num_hid_units)
        self.sigmoid = nn.Sigmoid()
        self.out_layer_1 = nn.Linear(self.num_hid_units, self.out_size)

    def forward(self, input):
        out_hid = self.sigmoid(self.hid_layer_1(input)) # BZ *  num_hid_units
        out_put = self.out_layer_1(out_hid) # BZ * out_size
        return out_put

model = non_linear_classifier(input_size=X.shape[1], out_size=2, num_hid_units=3)
if USE_CUDA:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()
for i in range(Epoch):
    output = model(X) # BZ* out_size
    loss = loss_fn(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("iter : {}, loss: {} \n".format(i, loss.item()))

plot_decision_boundary(model, X, y )