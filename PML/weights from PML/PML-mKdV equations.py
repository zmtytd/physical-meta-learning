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
# PyTorch random number generator
torch.manual_seed(1234)  # 设置pytorch随机种子

# Random number generators in other libraries
np.random.seed(1234)  # 设置numpy的随机种子

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
            nn.Linear(50, 1))

    def forward(self, x, para):
        Loss = []

        u_pred = self.conv(x)
        u_t = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, [1]]
        u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, [0]]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, [0]]
        u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx), create_graph=True)[0][:, [0]]

        for i in para:
            a = i[0]
            b = i[1]
            c = i[2]
            ff = u_t + a * u_pred * u_x + b * u_xxx + c * u_x

            f_11 = torch.zeros_like(ff)

            loss = self.loss_function(ff, f_11)
            Loss.append(loss)

        return Loss

w = torch.empty(2, 5000)
c=torch.empty(1, 5000)
a_b=torch.nn.init.trunc_normal_(w,mean=0.5,std=1,a=0,b=4).T
c=torch.nn.init.trunc_normal_(c,mean=0.5,std=1,a=0,b=1).T
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
X_T=betensor(np.load('/kaggle/input/mkdv-data/X_T.npy')).to(device)
ch_u=betensor(np.load('/kaggle/input/mkdv-data/choieu.npy').reshape(-1,1)[0:20000,[0]]).to(device)
ch_X_T=betensor(np.load('/kaggle/input/mkdv-data/choieX_T.npy').reshape(-1,2)[0:50000,:]).to(device)

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
        torch.save(mate.state_dict(), 'mate_kdv_3_inver_3_50.pth')