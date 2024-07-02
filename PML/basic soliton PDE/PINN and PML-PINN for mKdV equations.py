import torch
import torch.autograd as autograd
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
torch.manual_seed(1234)  #

# Random number generators in other libraries
np.random.seed(1234)  #
# Device configuration
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')

print(device)
torch.set_default_tensor_type(torch.DoubleTensor)
if device == 'cuda':
    print(torch.cuda.get_device_name())

class resnet(nn.Module):
    def __init__(self,input,output):
        super(resnet, self).__init__()
        self.input=input
        self.output=output
        self.linear=nn.Linear(self.input,self.output)

    def forward(self,x):
        x=self.linear(x)+x
        return nn.Tanh()(x)


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.loss_function = nn.MSELoss()
        self.conv = nn.Sequential(
            nn.Linear(2,50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

        self.a = torch.nn.Parameter(torch.ones(1).requires_grad_()*3.0)
        self.b=torch.nn.Parameter(torch.ones(1).requires_grad_()*0.5)

    def forward(self, x):
        out = self.conv(x)
        return out

    def LOSS_1(self, x):
        u_pred = self.forward(x)
        u_t = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, [1]]
        u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, [0]]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, [0]]
        u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx), create_graph=True)[0][:, [0]]
        ff = u_t -  self.a*u_pred * u_x + self.b*u_xxx
        f = torch.zeros_like(ff)

        return self.loss_function(ff, f), self.a,self.b


    def LOSS_2(self, u, x):
        u_pred = self.forward(x)
        return self.loss_function(u_pred, u)

    def predict(self, x, u):
        with torch.no_grad():
            u_pred = self.forward(x)
            loss = self.loss_function(u_pred, u)
        return (u_pred-u).detach().cpu().numpy(), loss

model=PINN().to(device)

pretrained_dict = torch.load('mate_mKdV_2_inver_3_50.pth')
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
betensor = lambda x: torch.from_numpy(x).requires_grad_()


data = scipy.io.loadmat('mKdV-1.mat')

t = np.real(data['t'].flatten()[:, None])
x = np.real(data['x'].flatten()[:, None])
t_test = np.real(data['t_test'].flatten()[:, None])
x_test = np.real(data['x_test'].flatten()[:, None])
u = np.real(data['usol']).T
u_test = np.real(data['usol_test']).T
X,T=np.meshgrid(x, t)
X_test,T_test=np.meshgrid(x_test,t_test)
X_T=np.hstack((X.flatten()[:,None],T.flatten()[:,None]))
ch_X_T =betensor(X_T).to(device)
X_T_test=betensor(np.hstack((X_test.flatten()[:,None],T_test.flatten()[:,None]))).to(device)
u=u + 0.01*(np.std(u)*np.random.randn(u.shape[0],u.shape[1]))#######the 1% noise environments
#u=u + 0.05*(np.std(u)*np.random.randn(u.shape[0],u.shape[1]))  #######the 5% noise environments
u=u.flatten()[:,None]
ch_u=betensor(u).to(device)
u_test=u_test.flatten()[:,None]
U_test=betensor(u_test).to(device)
params_dict=[{'params': model.conv.parameters(), 'lr': 0.000055},

            {'params':model.a, 'lr': 0.0008},
            {'params':model.b, 'lr': 0.0008}]

optimzer = torch.optim.Adam(params_dict)
Time = []
A = []
B = []
Loss = []
Loss_acu = []
for i in range(40000):
    start = time.time()
    optimzer.zero_grad()
    loss1, para_a, para_b = model.LOSS_1(ch_X_T)
    loss2 = model.LOSS_2(ch_u, ch_X_T)
    loss = loss1 + loss2

    loss.backward()
    optimzer.step()
    end = time.time()
    a = para_a.clone().detach().data
    b = para_b.clone().detach().data
    Loss.append(loss.data)
    A.append(a)
    B.append(b)
    Time.append(end - start)

    if i % 100 == 0:
        print(para_a.data, para_b.data, i, loss.data)

    if i % 1000 == 0:
        u_pred, loss_acu = model.predict(X_T_test, U_test)
        Loss_acu.append(loss_acu)
        np.save('/working//u_predict_%s' % i, u_pred)
        print("error for the test set is：", loss_acu.item())

Loss = torch.tensor(Loss).detach().cpu().numpy()
Loss_acu = torch.tensor(Loss_acu).detach().cpu().numpy()
A = torch.tensor(A)
A.detach().cpu().numpy()
B = torch.tensor(B)
B.detach().cpu().numpy()

np.save('/working//Loss_acu', Loss_acu)
torch.save(model.state_dict(), '/working/kdv_2_inver_50.pth')

np.save('/working/A', A)
np.save('/working/B', B)
np.save('/working/Loss', Loss)