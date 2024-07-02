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
import scipy.io as sio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')

print(device)
torch.set_default_tensor_type(torch.DoubleTensor)
if device == 'cuda':
    print(torch.cuda.get_device_name())


class PML_params(object):
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev
        self.distribute = lambda x: np.random.normal(loc=self.mean, scale=self.stddev, size=x)

    def sample(self, lower_bound, upper_bound, *shape):
        if len(shape) == 1:
            length = shape[0]
        elif len(shape) == 2:
            length = shape[0] * shape[1]
        else:
            length = shape[0] * shape[1] * shape[2]
        Smpled = np.array([])

        while len(Smpled) - length <= 0:
            samples = self.distribute((length,))
            truncated_samples = samples[(samples >= lower_bound) & (samples <= upper_bound)]
            Smpled = np.concatenate((Smpled, truncated_samples))
        Smpled = Smpled[0:length].reshape(shape)
        return Smpled

    def to_torch(self, x):
        x = torch.from_numpy(x).to(device)
        return x


A = PML_params(0.5,1)
para = A.sample(0,100,2,5000)
para = A.to_torch(para)
class ParaSet(Dataset):
    def __init__(self, para):
        self.para=para
    def __getitem__(self, idx):
        return self.para[idx]
    def __len__(self):
        return len(self.para)
Pset=ParaSet(para)
PPset=DataLoader(Pset,batch_size=250, shuffle=False)


class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.Tanh()


        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))


        self.layers.append(nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class PI_DeepONet(nn.Module):
    def __init__(self, branch_layers_real, branch_layers_imag, trunk_layers):
        super(PI_DeepONet, self).__init__()

        self.branch_network_real = MLP(branch_layers_real)
        self.branch_network_imag = MLP(branch_layers_imag)
        self.trunk_network = MLP(trunk_layers)
        self.lossfunc = nn.MSELoss()

        self.loss_log = []
        self.loss_operator_log = []
        self.opt = optim.Adam([{'params': self.branch_network_real.parameters(), 'lr': 0.0001},
                               {'params': self.branch_network_imag.parameters(), 'lr': 0.0001},
                               {'params': self.trunk_network.parameters(), 'lr': 0.0001}])


    def forward(self, u_real, u_imag, x, t):
        y = torch.stack((x, t), dim=1)
        B_real = self.branch_network_real(u_real)
        B_imag = self.branch_network_imag(u_imag)
        T = self.trunk_network(y)
        outputs_real = torch.sum(B_real * T, dim=1)
        outputs_imag = torch.sum(B_imag * T, dim=1)
        return outputs_real, outputs_imag

    def loss_PML(self, batch, para):
        Loss = []
        inputs, outputs = batch
        u_real, u_imag, y = inputs
        x = y[:, 0]
        t = y[:, 1]
        s_real, s_imag = self.forward(u_real, u_imag, x, t)
        s_real_x = torch.autograd.grad(s_real, x, torch.ones_like(s_real), create_graph=True)[0]
        s_real_xx = torch.autograd.grad(s_real_x, x, torch.ones_like(s_real_x), create_graph=True)[0]
        s_imag_x = torch.autograd.grad(s_imag, x, torch.ones_like(s_imag), create_graph=True)[0]
        s_imag_xx = torch.autograd.grad(s_imag_x, x, torch.ones_like(s_imag_x), create_graph=True)[0]
        s_real_t = torch.autograd.grad(s_real, t, torch.ones_like(s_real), create_graph=True)[0]
        s_imag_t = torch.autograd.grad(s_imag, t, torch.ones_like(s_imag), create_graph=True)[0]
        Q_mod = s_real.pow(2) + s_imag.pow(2)
        for i in para:
            a = i[0]
            b = i[1]
            f_1 = -1 * s_imag_t + a * s_real_xx + b * Q_mod * s_real
            f_2 = s_real_t + a * s_imag_xx + b * Q_mod * s_imag
            loss = self.lossfunc(f_1, torch.zeros_like(f_1)) + self.lossfunc(f_2, torch.zeros_like(f_2))
            Loss.append(loss)
        return Loss

    ##     def residual_net(self,u,x,t):
    #         s=self.forward(u,x,t)
    #         s_t=torch.autograd.grad(s,x,torch.ones_like(s),create_graph=True)[0]
    #         s_x=torch.autograd.grad(s,t,torch.ones_like(s),create_graph=True)[0]
    #         s_xx=torch.autograd.grad(s_x,x,torch.ones_like(s_x),create_graph=True)[0]
    #         res=s_t - 0.01 * s_xx - 0.01 * s**2
    #         return res

    #     def loss_res(self, batch):
    #         inputs, outputs = batch # 输入和输出
    #         u, y = inputs
    #         x=y[:,0]
    #         t=y[:,1]
    #         pred=self.residual_net(u,x,t)
    #         loss = self.lossfunc(outputs.flatten(), pred.flatten())
    #         return loss

    def train(self, operator_dataset, para_set, nIter=750):
        # operator_dataset = iter(operator_dataset)
        # operator_dataset_test = iter(operator_dataset_test)
        # operator_dataset = iter(operator_dataset)
        for i in range(nIter):
            for para in para_set:
                batch = operator_dataset
                self.opt.zero_grad()
                loss = self.loss_PML(batch, para)
                loss = sum(loss) / len(loss)
                loss.backward()
                self.opt.step()
                self.loss_log.append(loss.item())

            if i % 50 == 0:
                # loss_test = self.test_deeponet(next(operator_dataset_test))
                print("---------------------------------" + "epoch is" + str(i))
                print(loss.item())

                # print("the test loss is"+"{:.5f}".format(loss_test.item()))
m=100
branch_layers_real = [m, 50, 50, 50, 50, 50]
branch_layers_imag = [m, 50, 50, 50, 50, 50]
trunk_layers =  [2, 50, 50, 50, 50, 50]
model=PI_DeepONet(branch_layers_real, branch_layers_imag,trunk_layers).to(device)

batch_ = 750
model.train(batch_,PPset)
torch.save(model.state_dict(), "/working/mate_deeponet_nlse.pth")
Loss_Log = model.loss_log
torch.save(Loss_Log,'/working/loss_log')