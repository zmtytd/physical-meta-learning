import torch
import torch.autograd as autograd
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

torch.manual_seed(3017)  #

# Random number generators in other libraries
np.random.seed(3017
               )
# Device configuration
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_default_tensor_type(torch.DoubleTensor)
if device == 'cuda':
    print(torch.cuda.get_device_name())


class PML(nn.Module):
    def __init__(self):
        super(PML, self).__init__()
        self.loss_function = nn.MSELoss()
        self.conv = nn.Sequential(
            nn.Linear(3, 50),
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

    def forward(self, x, para):
        Loss = []

        u_pred = self.conv(x)[:, [0]]
        u_x_y_t = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_x = u_x_y_t[:, [0]]
        u_y = u_x_y_t[:, [1]]
        u_t = u_x_y_t[:, [2]]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, [0]]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, [1]]

        v_pred = self.conv(x)[:, [1]]
        v_x_y_t = torch.autograd.grad(v_pred, x, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
        v_x = v_x_y_t[:, [0]]
        v_y = v_x_y_t[:, [1]]
        v_t = v_x_y_t[:, [2]]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, [0]]
        v_yy = torch.autograd.grad(v_y, x, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, [1]]
        Q_mod = u_pred.pow(2) + v_pred.pow(2)

        for i in para:
            a = i[0]
            b = i[1]
            c = i[2]
            f_2 = u_t + 0.5 * v_xx + 0.5 * a * v_yy - 0.5 * u_yy + Q_mod * v_pred - Q_mod * c * u_pred + u_pred
            f_1 = v_t - 0.5 * u_xx - 0.5 * a * u_yy - 0.5 * v_yy - Q_mod * u_pred - Q_mod * c * v_pred + v_pred

            f_11 = torch.zeros_like(f_1)
            f_22 = torch.zeros_like(f_2)
            loss = self.loss_function(f_1, f_11) + self.loss_function(f_2, f_22)
            Loss.append(loss)

        return Loss


w = torch.empty(2, 5000)
c1 = torch.empty(1, 5000)

a_b = torch.nn.init.normal_(w, -2, 1).T
c = torch.nn.init.normal_(c1, 0.5, 1).T
para = torch.cat((a_b, c), dim=1)


class ParaSet(Dataset):
    def __init__(self, para):
        self.para = para

    def __getitem__(self, idx):
        return self.para[idx]

    def __len__(self):
        return len(self.para)


Pset = ParaSet(para)
PPset = DataLoader(Pset, batch_size=250, shuffle=False)

mate = PML().to(device)

betensor = lambda x: torch.from_numpy(x).requires_grad_()
x = np.linspace(0, 3, 50)
y = np.linspace(0, 3, 50)
t = np.linspace(-1, 1, 20)
X, Y, T = np.meshgrid(x, y, t)
X_T = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], T.flatten()[:, None]))
X_T = betensor(X_T).to(device)
optimzer = torch.optim.Adam(mate.parameters(), lr=0.008)
for epoch in range(750):
    Loss = []
    for p in PPset:
        loss = sum(mate(X_T, p)) / len(p)

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        Loss.append(loss.item())
    print(sum(Loss) / len(Loss))
    if epoch % 100 == 0:
        torch.save(mate.state_dict(), "mate_nls_2_inver_3_50_less_%s.pth" % epoch)