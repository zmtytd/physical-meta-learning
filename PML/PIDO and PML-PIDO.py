import numpy as np
import torch

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')

print(device)
torch.set_default_tensor_type(torch.DoubleTensor)
if device == 'cuda':
    print(torch.cuda.get_device_name())

import torch.nn as nn
import torch.optim as optim

data_init = np.load('soliton_fam_inti.npz')
data_u = np.load('soliton_fam_u.npz')

def soliton_exact(ni):
    init = data_init[list(data_init.keys())[ni]]
    s = data_u[list(data_u.keys())[ni]]
    return init, s


def data_generate_one(x, t, ni):
    X, T = np.meshgrid(x, t)
    X = X.flatten()
    T = T.flatten()
    train_u, train_s = soliton_exact(ni)
    train_x = np.hstack((X[:, None], T[:, None]))
    train_u = np.tile(train_u, (train_x.shape[0], 1))
    assert train_u.shape[0] == train_s.shape[0] == train_x.shape[0]
    return train_u, train_x, train_s


def data_generate(N, x=np.linspace(-2, 2, 20), t=np.linspace(0, 2, 20)):
    ni = np.linspace(1, 1.5, N)
    train_U = []
    train_X = []
    train_S = []
    for i in ni:
        train_u, train_x, train_s = data_generate_one(x, t, i)
        train_U.append(train_u)
        train_X.append(train_x)
        train_S.append(train_s)
    train_U = np.array(train_U).reshape(-1, 100)
    train_X = np.array(train_X).reshape(-1, 2)
    train_S = np.array(train_S).reshape(-1, 1)
    return train_U, train_X, train_S


import torch
from torch.utils.data import Dataset


class DataGenerator_pytorch(Dataset):
    def __init__(self, u, y, s, batch_size=64):

        self.u_real = u.real
        self.u_imag = u.imag
        self.y = y
        self.s_real = s.real
        self.s_imag = s.imag
        self.N = int(u.shape[0])
        self.N_batch = int(u.shape[0] / batch_size)
        self.batch_size = batch_size


        self.u_real = torch.from_numpy(self.u_real).to(device).double().requires_grad_()
        self.u_imag = torch.from_numpy(self.u_imag).to(device).double().requires_grad_()
        self.y = torch.from_numpy(self.y).to(device).double().requires_grad_()
        self.s_real = torch.from_numpy(self.s_real).to(device).double().requires_grad_()
        self.s_imag = torch.from_numpy(self.s_imag).to(device).double().requires_grad_()

    def __len__(self):
        '返回数据集的大小'
        return self.N / batch_size

    #     def __getitem__(self, index):

    #         u_real_batch = self.u_real[index]
    #         u_imag_batch = self.u_imag[index]
    #         y_batch = self.y[index]
    #         s_real_batch = self.s_real[index]
    #         s_imag_batch = self.s_imag[index]

    #         return u_real_batch, u_imag_batch, y_batch,s_real_batch,s_imag_batch

    def __getitem__(self, index):


        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.N)


        u_real_batch = self.u_real[start_idx:end_idx]
        u_imag_batch = self.u_imag[start_idx:end_idx]
        y_batch = self.y[start_idx:end_idx]
        s_real_batch = self.s_real[start_idx:end_idx]
        s_imag_batch = self.s_imag[start_idx:end_idx]


        inputs = (u_real_batch, u_imag_batch, y_batch)
        outputs = (s_real_batch, s_imag_batch)

        return inputs, outputs



train_u,train_y,train_s=data_generate(80,x=np.linspace(-2,2,20),t=np.linspace(0,2,20))
test_u,test_y,test_s=data_generate(10,x=np.linspace(-2,2,20),t=np.linspace(0,2,20))
Data_train=DataGenerator_pytorch(train_u,train_y,train_s,batch_size=16000)
Data_test = DataGenerator_pytorch(test_u,test_y,test_s,batch_size=800)

class aa:
    def __init__(self, u, y, s, batch_size=64):
        self.u_real = u.real
        self.u_imag = u.imag
        self.y = y
        self.s_real = s.real
        self.s_imag = s.imag
        self.N = int(u.shape[0])
        self.N_batch = int(u.shape[0]/batch_size)
        self.batch_size = batch_size


        self.u_real = torch.from_numpy(self.u_real).to(device).double().requires_grad_()
        self.u_imag = torch.from_numpy(self.u_imag).to(device).double().requires_grad_()
        self.y = torch.from_numpy(self.y).to(device).double().requires_grad_()
        self.s_real = torch.from_numpy(self.s_real).to(device).double().requires_grad_()
        self.s_imag = torch.from_numpy(self.s_imag).to(device).double().requires_grad_()

AA=aa(train_u,train_y,train_s)
AA_ = aa(test_u,test_y,test_s)
batch_ = ((AA.u_real,AA.u_imag,AA.y),(AA.s_real,AA.s_imag))
batch_test = ((AA_.u_real,AA_.u_imag,AA_.y),(AA_.s_real,AA_.s_imag))



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

        self.A_log = []
        self.B_log = []


        self.loss_log = []
        self.loss_test_log = []
        self.a = torch.nn.Parameter(torch.ones(1).requires_grad_() * 1.0)
        self.b = torch.nn.Parameter(torch.ones(1).requires_grad_() * 0.5)

        self.opt = optim.Adam([{'params': self.branch_network_real.parameters(), 'lr': 0.001},
                               {'params': self.branch_network_imag.parameters(), 'lr': 0.001},
                               {'params': self.trunk_network.parameters(), 'lr': 0.001},
                               {'params': self.b, 'lr': 0.0008},
                               {'params': self.a, 'lr': 0.0008}])


    def forward(self, u_real, u_imag, x, t):
        y = torch.stack((x, t), dim=1)
        B_real = self.branch_network_real(u_real)
        B_imag = self.branch_network_imag(u_imag)
        T = self.trunk_network(y)
        outputs_real = torch.sum(B_real * T, dim=1)
        outputs_imag = torch.sum(B_imag * T, dim=1)
        return outputs_real, outputs_imag

    def loss_bcs(self, batch):

        inputs, outputs = batch
        out_real, out_imag = outputs
        u_real, u_imag, y = inputs
        x = y[:, 0]
        t = y[:, 1]

        pred_real, pred_imag = self.forward(u_real, u_imag, x, t)


        loss = self.lossfunc(out_real.flatten(), pred_real.flatten()) + self.lossfunc(out_imag.flatten(),
                                                                                      pred_imag.flatten())
        return loss

    def loss_squre(self, batch):
        inputs, outputs = batch
        out_real, out_imag = outputs
        u_real, u_imag, y = inputs
        x = y[:, 0]
        t = y[:, 1]

        pred_real, pred_imag = self.forward(u_real, u_imag, x, t)
        U_pred = pred_real.flatten() ** 2 + pred_imag.flatten() ** 2
        U_true = out_real.flatten() ** 2 + out_imag.flatten() ** 2

        loss = self.lossfunc(U_pred ** 0.5, U_true ** 0.5)
        return loss

    def std(self, batch):
        inputs, outputs = batch
        out_real, out_imag = outputs
        u_real, u_imag, y = inputs
        x = y[:, 0]
        t = y[:, 1]

        pred_real, pred_imag = self.forward(u_real, u_imag, x, t)
        U_pred = (pred_real.flatten() ** 2 + pred_imag.flatten() ** 2) ** 0.5
        U_true = (out_real.flatten() ** 2 + out_imag.flatten() ** 2) ** 0.5

        std = torch.std(U_pred - U_true)
        std_zheng = torch.std(torch.absolute(U_pred - U_true))
        return std, std_zheng

    def loss_pde(self, batch):
        inputs, outputs = batch  #
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
        f_1 = -1 * s_imag_t + self.a * s_real_xx + self.b * Q_mod * s_real
        f_2 = s_real_t + self.a * s_imag_xx + self.b * Q_mod * s_imag

        loss = self.lossfunc(f_1, torch.zeros_like(f_1)) + self.lossfunc(f_2, torch.zeros_like(f_2))
        return loss

    ##     def residual_net(self,u,x,t):
    #         s=self.forward(u,x,t)
    #         s_t=torch.autograd.grad(s,x,torch.ones_like(s),create_graph=True)[0]
    #         s_x=torch.autograd.grad(s,t,torch.ones_like(s),create_graph=True)[0]
    #         s_xx=torch.autograd.grad(s_x,x,torch.ones_like(s_x),create_graph=True)[0]
    #         res=s_t - 0.01 * s_xx - 0.01 * s**2
    #         return res

    #     def loss_res(self, batch):
    #         inputs, outputs = batch #
    #         u, y = inputs
    #         x=y[:,0]
    #         t=y[:,1]
    #         pred=self.residual_net(u,x,t)
    #         loss = self.lossfunc(outputs.flatten(), pred.flatten())
    #         return loss

    def test_deeponet(self, batch_test):
        with torch.no_grad():
            loss_test = self.loss_squre(batch_test)

            return loss_test

    def train(self, operator_dataset, operator_dataset_test, nIter=40000):
        # operator_dataset = iter(operator_dataset)
        # operator_dataset_test = iter(operator_dataset_test)
        # operator_dataset = iter(operator_dataset)

        for i in range(nIter):

            batch = operator_dataset
            self.opt.zero_grad()
            loss = self.loss_bcs(batch) + self.loss_squre(batch) + self.loss_pde(batch)
            loss.backward()
            self.opt.step()
            self.A_log.append(self.a)
            self.B_log.append(self.b)
            self.loss_log.append(loss.item())

            if i % 100 == 0:
                # loss_test = self.test_deeponet(next(operator_dataset_test))
                loss_test = self.test_deeponet(operator_dataset_test)
                std, std_zheng = self.std(operator_dataset_test)
                dict_ = {'branch_network_real': self.branch_network_real.state_dict(),
                         'branch_network_imag': self.branch_network_imag.state_dict(),
                         'trunk_network': self.trunk_network.state_dict(), 'a': self.a, 'b': self.b}
                torch.save(dict_, '/working/dict_%s' % i)
                self.loss_test_log.append(loss_test.item())
                print("---------------------------------" + "epoch is" + str(i))
                print("loss_test is " + "{:.7f}".format(
                    loss_test.item()) + " " + "the std is" + "%.6f" % std.item() + " " + "the std_zheng is " + "%.6f " % std_zheng.item())
                print("the loss is" + "{:.7f}".format(loss.item()))
                print(self.a, self.b)
                # print("the test loss is"+"{:.5f}".format(loss_test.item()))


m=100
branch_layers_real = [m, 50, 50, 50, 50, 50]
branch_layers_imag = [m, 50, 50, 50, 50, 50]
trunk_layers =  [2, 50, 50, 50, 50, 50]
model=PI_DeepONet(branch_layers_real, branch_layers_imag,trunk_layers).to(device)
pretrained_dict = torch.load('mate_deeponet_nlse.pth')
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)


model.train(batch_,batch_test)
torch.save(model.state_dict(), "/working/deeponet_nlse_nopretrain.pth")
torch.save(model.A_log,'/working/A')
torch.save(model.B_log,'/working/B')
torch.save(model.loss_log,'/working/loss_log')
torch.save(model.loss_test_log,'/working/loss_test_log')