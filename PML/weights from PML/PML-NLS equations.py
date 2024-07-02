# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import torch
import torch.autograd as autograd  # 梯度计算包
from torch import Tensor
import torch.nn as nn
import torch.optim as optim  # optimizers e.g. gradient descent, ADAM, etc.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
import numpy as np
import time

import scipy.io
import math
# Set default dtype to float32
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
import scipy.io as sio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
torch.manual_seed(3017)  # 设置pytorch随机种子

# Random number generators in other libraries
np.random.seed(3017
               )  # 设置numpy的随机种子

# Device configuration
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')  # 声明本次计算使用的devic。首先检索有没有gpu，如果有的话就调用，否则就在cpu上计算，此处‘cuda’相当于‘cuda：0‘，即gpu默认编号从0开始，可以跟据需要修改

print(device)
torch.set_default_tensor_type(torch.DoubleTensor)
if device == 'cuda':
    print(torch.cuda.get_device_name())

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.loss_function = nn.MSELoss()
        self.conv = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 2))


    def forward(self, x,para):
        Loss = []

        u_pred = self.conv(x)[:, [0]]
        u_x_t = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_x = u_x_t[:, [0]]
        u_t = u_x_t[:, [1]]
        u_tt = torch.autograd.grad(u_t, x, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, [1]]
        v_pred = self.conv(x)[:, [1]]
        v_x_t = torch.autograd.grad(v_pred, x, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
        v_x = v_x_t[:, [0]]
        v_t = v_x_t[:, [1]]
        v_tt = torch.autograd.grad(v_t, x, grad_outputs=torch.ones_like(v_t), create_graph=True)[0][:, [1]]
        Q_mod = u_pred.pow(2) + v_pred.pow(2)

        for i in para:

            a=i[0]
            b=i[1]
            c=i[2]
            f_1 = -1 * v_x + b * u_tt + a * Q_mod * u_pred
            f_2 = u_x + b * v_tt + a * Q_mod * v_pred

            f_11 = torch.zeros_like(f_1)
            f_22 = torch.zeros_like(f_2)
            loss = self.loss_function(f_1, f_11) + self.loss_function(f_2, f_22)
            Loss.append(loss)

        return Loss

w = torch.empty(2, 5000)
c=torch.empty(1,5000)
a_b=torch.nn.init.trunc_normal_(w,0.5,1,0,2).T
c=torch.nn.init.trunc_normal_(c,0.005,0.0005,0.0055).T
para=torch.cat((a_b,c),dim=1)

class ParaSet(Dataset):
    def __init__(self, para):
        self.para=para
    def __getitem__(self, idx):
        return self.para[idx]
    def __len__(self):
        return len(self.para)
Pset=ParaSet(para)
PPset=DataLoader(Pset,batch_size=250, shuffle=False)

mate=Cnn().to(device)

betensor = lambda x: torch.from_numpy(x).requires_grad_()
x=np.linspace(0,5,500)
t=np.linspace(-2.5,2.5,500)
X,T=np.meshgrid(x, t)
X_T=np.hstack((X.flatten()[:,None],T.flatten()[:,None]))
X_T = betensor(X_T).to(device)
for epoch in range(750):
    Loss=[]
    for p in PPset:
        loss=sum(mate(X_T,p))/len(p)
        optimzer = torch.optim.Adam(mate.parameters(), lr=0.008)
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        Loss.append(loss.item())
    print(sum(Loss)/len(Loss))
    if epoch % 100 == 0:
        torch.save(mate.state_dict(), "mate_nls_2_inver_3_50.pth")