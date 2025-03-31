#from typing import Tuple
import itertools

from fbm import FBM
import numpy as np
from pandas.io.stata import stata_epoch
#from numpy.core.fromnumeric import put
import torch
#from torch._C import T
import torch.nn as nn
from torch.nn import init
from torch.nn.modules.activation import ELU
from torch.nn.modules.linear import Linear

import math


bool_xavier_normal = True
init_gain_sde = 1
init_gain_fsde = 1 #1.5 
init_gain_armasde = 1
batch_dim, state_dim, bm_dim = 100, 1, 1
nhidden_rnn, nhidden_sde, nhidden_fsde = 40, 20, 20
nhidden_armasde = 20
latent_dim = 1


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20, gain=1):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden) # fully connected
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        if bool_xavier_normal:
            nn.init.xavier_normal_(self.fc1.weight, gain)
            nn.init.xavier_normal_(self.fc2.weight, gain)
            nn.init.xavier_normal_(self.fc3.weight, gain)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class GeneratorRNN(nn.Module):
    """
      h: hidden-layer variable
      x: observed variable
    """
    def __init__(self, obs_dim=state_dim, nhidden=nhidden_rnn):
        super(GeneratorRNN, self).__init__()
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, obs_dim)
        self.h2h_out = nn.Linear(nhidden, obs_dim)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        h_out = self.h2h_out(h)
        x_out = self.h2o(h)
        return x_out, h_out # both has state_dim

    #def initHidden(self):
    #    return torch.zeros(self.nbatch, self.nhidden)


class RecognitionRNN(nn.Module):
    """
      h: hidden-layer variable
      x: observed variable
    """
    def __init__(self, latent_dim=4, obs_dim=1, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=1, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Hyperperameters for SDE- and fSDE-Net
#batch_dim, latent_dim, bm_dim = 3, 2, 1

class LatentSDEfunc(nn.Module):
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self, nhidden=nhidden_sde, state_dim=state_dim, gain=init_gain_sde):
        super().__init__()
        #self.nhidden = nhidden
        #self.latent_dim = latent_dim
        #self.bm_dim = bm_dim
        #self.batch_dim = batch_dim

        self.drift_fc1 = nn.Linear(state_dim, nhidden)
        self.drift_fc2 = nn.Linear(nhidden, nhidden)
        self.drift_fc3 = nn.Linear(nhidden, state_dim)
        
        self.diff_fc1 = nn.Linear(state_dim, nhidden)
        self.diff_fc2 = nn.Linear(nhidden, nhidden)
        self.diff_fc3 = nn.Linear(nhidden, state_dim) # * bm_dim)
        
        self.act = nn.Tanh() #(inplace=True)

        if bool_xavier_normal:
            nn.init.xavier_normal_(self.drift_fc1.weight, gain)
            nn.init.xavier_normal_(self.drift_fc2.weight, gain)
            nn.init.xavier_normal_(self.drift_fc3.weight, gain)
            nn.init.xavier_normal_(self.diff_fc1.weight, gain)
            nn.init.xavier_normal_(self.diff_fc2.weight, gain)
            nn.init.xavier_normal_(self.diff_fc3.weight, gain)
        
    # Drift
    def f(self, t, y):
        out = self.drift_fc1(y)
        out = self.act(out)
        out = self.drift_fc2(out)
        out = self.act(out)
        out = self.drift_fc3(out)
        #out = self.act(out)
        return out  # shape (batch_size, state_size)

    # Diffusion
    def g(self, t, y):
        out = self.diff_fc1(y)
        out = self.act(out)
        out = self.diff_fc2(out)
        out = self.act(out)
        out = self.diff_fc3(out)
        #out = self.act(out)
        return out.view(batch_dim, state_dim, bm_dim) #.view(self.batch_dim, self.latent_dim, self.bm_dim)


class LatentFSDEfunc(nn.Module):

    def __init__(self, nhidden=nhidden_fsde, state_dim=state_dim, gain=init_gain_fsde):
        super(LatentFSDEfunc, self).__init__()
        self.drift_fc1 = nn.Linear(state_dim, nhidden)
        self.drift_fc2 = nn.Linear(nhidden, nhidden)
        self.drift_fc3 = nn.Linear(nhidden, state_dim)
        #self.drift_act = nn.Tanh() #(inplace=True)
        #self.act = nn.Tanh() #(inplace=True)
        
        self.diff_fc1 = nn.Linear(state_dim, nhidden)
        self.diff_fc2 = nn.Linear(nhidden, nhidden)
        self.diff_fc3 = nn.Linear(nhidden, state_dim)
        #self.diff_act = nn.Tanh() #(inplace=True)
        
        self.act = nn.Tanh() #(inplace=True)

        if bool_xavier_normal:
            nn.init.xavier_normal_(self.drift_fc1.weight, gain)
            nn.init.xavier_normal_(self.drift_fc2.weight, gain)
            nn.init.xavier_normal_(self.drift_fc3.weight, gain)
            nn.init.xavier_normal_(self.diff_fc1.weight, gain)
            nn.init.xavier_normal_(self.diff_fc2.weight, gain)
            nn.init.xavier_normal_(self.diff_fc3.weight, gain)
        
    def drift(self, y):
        out = self.drift_fc1(y)
        out = self.act(out)
        out = self.drift_fc2(out)
        out = self.act(out)
        out = self.drift_fc3(out)
        return out #.reshape(batch_dim, latent_dim)  

    def diffusion(self, y):
        out = self.diff_fc1(y)
        out = self.act(out)
        out = self.diff_fc2(out)
        out = self.act(out)
        out = self.diff_fc3(out)
        return out #.reshape(batch_dim, latent_dim)  




"""
# Following will not be used. 
class FSDENet(nn.Module):

    def __init__(self, nhidden=2, state_size=latent_dim, gain=init_gain):
        super(FSDENet, self).__init__()
        self.drift_fc1 = nn.Linear(state_size, nhidden)
        self.drift = nn.Sequential(

            nn.Linear(state_size, nhidden),
            nn.Tanh(),
            nn.Linear(nhidden, state_size),
        )
        self.diffusion = nn.Sequential(
            nn.Linear(state_size, nhidden),
            nn.Tanh(),
            nn.Linear(nhidden, state_size)
        )
        if boole_xavier_normal:
            for param in FSDENet.parameters():
                nn.init.xavier_normal_(param, gain)
                #nn.init.xavier_normal_(self.diffusion.weight, gain)
         
    def forward(self, hurst, x0, ts):
        batch_size = x0.size(0)
        state_size = x0.size(1)
        t_start = float(ts[0])
        t_end = float(ts[-1])
        nsteps = 5000
        dt = (t_end - t_start) / nsteps
        
        #y = torch.tile(x0, (nsteps + 1,)).reshape(batch_size, nsteps + 1, state_size).clone()
        #y = torch.zeros(batch_size, nsteps + 1, state_size)
        #print(y.shape)
        y = []
        #dB = []
        #for i, k in itertools.product(range(batch_size), range(nsteps+1)): 
        for i in range(batch_size):
            y.append(x0[i])
            B = FBM(n=nsteps, hurst=hurst, length=t_end-t_start).fbm()
            dB = np.diff(B) if i==0 else np.append(dB, np.diff(B)) 
            for k in range(1, nsteps+1):    
                index = k + i * (nsteps + 1)
                y_next = y[index-1] + self.drift(y[index-1]) * dt \
                    + self.diffusion(y[index-1]) * dB[index-1-(k-1)]
                y.append(y_next)
                #y[i,k] = y[i,k-1] + dB[k-1] + self.drift(y[i,k-1]) * dt \
                    #+ self.diffusion(y[i,k-1]) * dB[k-1] # "dB[k]=B[k+1]-B[k]"    
        #print(torch.stack(y))
        y = torch.stack(y, axis=0).reshape(batch_size, -1, state_size) 
        print(y)
 
        ts_panel = torch.tile(ts, (batch_size, state_size)).reshape(batch_size, state_size, -1).permute(0, 2, 1) 
        num = (ts_panel - t_start) / (t_end -t_start) * nsteps 
        n_floor = torch.floor(num).to(torch.int64) 
        n_ceil = torch.ceil(num).to(torch.int64)
        ts_floor = t_start + n_floor * dt
        solution = y[:,n_floor[0,:,0]] + (ts_panel - ts_floor) * (y[:,n_ceil[0,:,0]] - y[:,n_floor[0,:,0]]) / dt 
        #print(solution[:,:3])
        return solution
"""

class LatentArmaSDEfunc(nn.Module):

    def __init__(self, nhidden=nhidden_armasde, state_dim=state_dim, gain=init_gain_armasde):
        super(LatentArmaSDEfunc, self).__init__()
        self.drift_fc1 = nn.Linear(state_dim, nhidden)
        self.drift_fc2 = nn.Linear(nhidden, nhidden)
        self.drift_fc3 = nn.Linear(nhidden, state_dim)
        #self.drift_act = nn.Tanh() #(inplace=True)
        #self.act = nn.Tanh() #(inplace=True)
        
        self.diff_fc1 = nn.Linear(state_dim, nhidden)
        self.diff_fc2 = nn.Linear(nhidden, nhidden)
        self.diff_fc3 = nn.Linear(nhidden, state_dim)
        #self.diff_act = nn.Tanh() #(inplace=True)
        
        self.act = nn.Tanh() #(inplace=True)

        self.p = nn.Parameter(torch.tensor(0.0758 + 0.0696, dtype=torch.float32))
        self.theta = nn.Parameter(torch.tensor(0.0696, dtype=torch.float32))

        if bool_xavier_normal:
            nn.init.xavier_normal_(self.drift_fc1.weight, gain)
            nn.init.xavier_normal_(self.drift_fc2.weight, gain)
            nn.init.xavier_normal_(self.drift_fc3.weight, gain)
            nn.init.xavier_normal_(self.diff_fc1.weight, gain)
            nn.init.xavier_normal_(self.diff_fc2.weight, gain)
            nn.init.xavier_normal_(self.diff_fc3.weight, gain)
        
    def drift(self, y):
        out = self.drift_fc1(y)
        out = self.act(out)
        out = self.drift_fc2(out)
        out = self.act(out)
        out = self.drift_fc3(out)
        return out #.reshape(batch_dim, latent_dim)  

    def diffusion(self, y):
        out = self.diff_fc1(y)
        out = self.act(out)
        out = self.diff_fc2(out)
        out = self.act(out)
        out = self.diff_fc3(out)
        return out #.reshape(batch_dim, latent_dim)  

    def l_func(self, u):
        """ torch.Tensor を維持しながら計算 """
        numerator = 2 * self.theta * (self.p - self.theta)
        denominator = (2 * self.p - self.theta) ** 2 * torch.exp(2 * (self.p - self.theta) * u) - self.theta ** 2
        return self.theta * torch.exp(self.p * u) * (1 - numerator / denominator)

    def noise_drift(self, X, t):
        """ torch の計算を維持しつつ、numpy に渡す前に detach """
        p_val = self.p.detach().numpy()  # detach() して計算グラフを切り離す
        dX1 = - np.exp(-p_val * t) * X[1]
        dX2 = 0.0  
        return np.array([dX1, dX2])

    def noise_diffusion(self, X, t):
        """ l_func の計算結果も detach して numpy に変換 """
        return np.array([
            [1.0, 0.0], 
            [0.0, self.l_func(torch.tensor(t, dtype=torch.float32)).detach().numpy()]
        ])



class ArmaSDE(torch.nn.Module):
    noise_type = 'general'  # ノイズタイプ
    sde_type = 'ito'  # SDEのタイプ

    def __init__(self, nhidden=nhidden_armasde, state_dim=state_dim, gain=init_gain_armasde):
        super(ArmaSDE, self).__init__()
        self.p = nn.Parameter(torch.tensor(0.0758 + 0.0696, dtype=torch.float32))
        self.theta = nn.Parameter(torch.tensor(0.0696, dtype=torch.float32))

        self.drift_fc1 = nn.Linear(state_dim, nhidden)
        self.drift_fc2 = nn.Linear(nhidden, nhidden)
        self.drift_fc3 = nn.Linear(nhidden, state_dim)

        self.diff_fc1 = nn.Linear(state_dim, nhidden)
        self.diff_fc2 = nn.Linear(nhidden, nhidden)
        self.diff_fc3 = nn.Linear(nhidden, state_dim)

        self.act = nn.Tanh()

        if bool_xavier_normal:
            nn.init.xavier_normal_(self.drift_fc1.weight, gain)
            nn.init.xavier_normal_(self.drift_fc2.weight, gain)
            nn.init.xavier_normal_(self.drift_fc3.weight, gain)
            nn.init.xavier_normal_(self.diff_fc1.weight, gain)
            nn.init.xavier_normal_(self.diff_fc2.weight, gain)
            nn.init.xavier_normal_(self.diff_fc3.weight, gain)

    # Drift (ドリフト)
    def f(self, t, y):
        out = self.act(self.drift_fc1(y[:,:state_dim]))
        out = self.act(self.drift_fc2(out))
        out = self.drift_fc3(out)

        dX1dt = out - torch.exp(-self.p * t) * y[:,state_dim:]  
        dX2dt = torch.zeros_like(y[:,:1])

        #print("1;{}".format(torch.exp(-self.p * t) * y[:,state_dim:]))
        #print("2;{}".format(out))
        #print("3;{}".format(dX1))
        #print("4;{}".format(dX2))
        #print(dX1.size())
        #print(dX2.size())
        #print("5;{}".format(torch.cat([dX1, dX2], dim=1)))
        #print(torch.cat([dX1, dX2], dim=1).size())
        return torch.cat([dX1dt, dX2dt], dim=1)

    # Diffusion (拡散)
    def g(self, t, y):
        def l_func(u):
            numerator = 2 * self.theta * (self.p - self.theta)
            denominator = (((2 * self.p - self.theta) ** 2) * torch.exp(2 * (self.p - self.theta) * u)) - self.theta ** 2
            return self.theta * torch.exp(self.p * u) * (1 - numerator / denominator)

        l_vals = l_func(t).expand(batch_dim).reshape(batch_dim,-1) 
        #l_vals = torch.ones([]).expand(batch_dim).reshape(batch_dim,-1) 
        dX2dw = l_vals
        #print("1;{}".format(l_vals))

        out = self.act(self.diff_fc1(y[:,:state_dim]))
        out = self.act(self.diff_fc2(out))
        out = self.diff_fc3(out)
        dX1dw = out

        #print(torch.cat([dX1dw, dX2dw], dim=1).reshape(batch_dim,state_dim+1,-1).size())

        return torch.cat([dX1dw, dX2dw], dim=1).reshape(batch_dim,state_dim+1,-1)