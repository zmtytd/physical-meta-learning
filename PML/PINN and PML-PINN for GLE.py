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
torch.manual_seed(1234)

# Random number generators in other libraries
np.random.seed(1234)
# Device configuration
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')

print(device)
#torch.set_default_tensor_type(torch.DoubleTensor)
if device == 'cuda':
    print(torch.cuda.get_device_name())
torch.set_default_dtype(torch.float32)

data = scipy.io.loadmat('GLE.mat')
betensor = lambda x: torch.from_numpy(x).requires_grad_()
T = np.real(data['t'].flatten()[:, None])
X = np.real(data['x'].flatten()[:, None])
Y = np.real(data['y'].flatten()[:, None])
T_test = np.real(data['t_test'].flatten()[:, None])
X_test = np.real(data['x_test'].flatten()[:, None])
Y_test = np.real(data['y_test'].flatten()[:, None])
u = np.real(data['usol']).T
u_test = np.real(data['usol_test']).T

x,y,t = np.meshgrid(X,Y,T)
x_test,y_test,t_test = np.meshgrid(X_test,Y_test,T_test)
inputs=np.hstack((x.flatten()[:,None],y.flatten()[:,None],t.flatten()[:,None]))
u = np.hstack((u.real.flatten()[:,None], u.imag.flatten()[:,None]))

inputs_test=np.hstack((x_test.flatten()[:,None],y_test.flatten()[:,None],t_test.flatten()[:,None]))
u_test = np.hstack((u_test.real.flatten()[:,None], u_test.imag.flatten()[:,None]))
outputs_test =betensor(u_test).to(device).to(device)
inputs_test = betensor(inputs_test).to(device)


outputs =betensor(u).to(device)
inputs = betensor(inputs).to(device)





class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
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

        self.beta_0 = torch.nn.Parameter(torch.ones(1).requires_grad_() * -2.0)
        self.beta_1 = torch.nn.Parameter(torch.ones(1).requires_grad_() * 0.5)
        self.gama_0 = torch.nn.Parameter(torch.ones(1).requires_grad_() * 1.5)

    def forward(self, x):
        out = self.conv(x)
        return out

    def LOSS_1(self, x):
        u_pred = self.forward(x)[:, [0]]
        u_x_y_t = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_x = u_x_y_t[:, [0]]
        u_y = u_x_y_t[:, [1]]
        u_t = u_x_y_t[:, [2]]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, [0]]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, [1]]

        v_pred = self.forward(x)[:, [1]]
        v_x_y_t = torch.autograd.grad(v_pred, x, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
        v_x = v_x_y_t[:, [0]]
        v_y = v_x_y_t[:, [1]]
        v_t = v_x_y_t[:, [2]]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, [0]]
        v_yy = torch.autograd.grad(v_y, x, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, [1]]
        Q_mod = u_pred.pow(2) + v_pred.pow(2)
        f_2 = u_t + 0.5 * v_xx + 0.5 * self.beta_0 * v_yy - 0.5 * u_yy + Q_mod * v_pred - Q_mod * self.gama_0 * u_pred + u_pred
        f_1 = v_t - 0.5 * u_xx - 0.5 * self.beta_0 * u_yy - 0.5 * v_yy - Q_mod * u_pred - Q_mod * self.gama_0 * v_pred + v_pred

        f_11 = torch.zeros_like(f_1)
        f_22 = torch.zeros_like(f_2)

        #      v_xxx=torch.autograd.grad(v_xx, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, [0]]
        #     v_xxxx=torch.autograd.grad(v_xxx, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, [0]]
        # u_mod = (u_pred.pow(2) + v_pred.pow(2)).pow(0.5)
        #      u_x_mod=(u_x.pow(2)+v_x.poe(2)).pow(0.5)
        #        u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx), create_graph=True)[0][:, [0]]
        #  #       u_xxxx=torch.autograd.grad(u_xxx, x, grad_outputs=torch.ones_like(u_xx), create_graph=True)[0][:, [0]]

        return self.loss_function(f_1, f_11) + self.loss_function(f_2, f_22), self.beta_0, self.beta_1, self.gama_0

    def LOSS_2(self, x, u):
        u_pred = self.forward(x)
        return self.loss_function(u_pred, u)

    def predict(self, x, u):
        with torch.no_grad():
            u_pred = self.forward(x)
            loss = self.loss_function(u_pred, u)
        return u_pred.detach().cpu().numpy(), loss



model = PINN().to(device)
params_dict=[{'params': model.conv.parameters(), 'lr': 0.0008},

            {'params':model.beta_0, 'lr': 0.008},
            {'params':model.beta_1, 'lr': 0.0001},
            {'params':model.gama_0, 'lr': 0.008}]

optimzer = torch.optim.Adam(params_dict)

# pretrained_dict = torch.load('mate_nls_2_inver_3_50_less_600.pth')
# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)
BETA_0 = []
BETA_1=[]
GAMA_0 = []
Loss=[]
Loss_acu=[]

for i in range(40000):

    optimzer.zero_grad()
    loss1,para_beta_0,para_beta_1,para_gama_0 = model.LOSS_1(inputs)
    loss2=model.LOSS_2(inputs, outputs)

    loss = loss1+15*loss2

    loss.backward()
    optimzer.step()

    a = para_beta_0.clone().detach().data
    b = para_beta_1.clone().detach().data
    c = para_gama_0.clone().detach().data

    Loss.append(loss.data)
    BETA_0.append(a)
    BETA_1.append(b)
    GAMA_0.append(c)





    if i%100==0:
         print(i, a,b,c,loss.data)


    if i % 500 == 0:
        u_pred, loss_acu = model.predict(inputs_test, outputs_test)
        Loss_acu.append(loss_acu)
        np.save('/working/u_predict_%s'%i,u_pred)
        torch.save(model.state_dict(), '/working/GLE_2_inver_%s.pth'%i)
        print("error of the test set is：",loss_acu.item())



Loss=torch.tensor(Loss).detach().cpu().numpy()
Loss_acu=torch.tensor(Loss_acu).detach().cpu().numpy()
BETA_0 = torch.tensor(BETA_0)
BETA_0.detach().cpu().numpy()
BETA_1=torch.tensor(BETA_1)
BETA_1.detach().cpu().numpy()

GAMA_0 = torch.tensor(GAMA_0)
GAMA_0.detach().cpu().numpy()



np.save('/working/Loss_acu', Loss_acu)
np.save('/working/BETA_0', BETA_0)
np.save('/working/BETA_1', BETA_1)
np.save('/working/GAMA_0', GAMA_0)
np.save('/working/Loss', Loss)

