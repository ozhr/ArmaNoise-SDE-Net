# This file is based on code from:
#   Original Copyright (c) 2022 Kohei Hayashi
#   Licensed under the MIT License
#
# Modifications:
#   Copyright (c) 2025 Hiromu Ozai
#   Released under the MIT License
#
# See the LICENSE file in the repository root for full license text.

import os
import argparse
import logging
import time
import sys

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import numpy as np
# from numpy.core.arrayprint import printoptions
import numpy.random as npr
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data.data import get_stock_data, get_fOU_data, get_other_data, get_short_memory_data, get_rough_data, get_short_fOU_data, get_OU_data, get_FBM_data
from utils.neural_net import LatentFSDEfunc, LatentODEfunc, GeneratorRNN, LatentArmaSDEfunc, LatentNNKernelArmaSDEfunc
from utils.neural_net import LatentSDEfunc, latent_dim, batch_dim, nhidden_rnn
from utils.utils import RunningAverageMeter, log_normal_pdf, normal_kl, calculate_log_likelihood, acf_loss, pathwise_mse_loss
from utils.plots import plot_generated_paths, plot_original_path, plot_hist
from utils.utils import save_csv, tensor_to_numpy

#sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser()
parser.add_argument('--ode_adjoint', type=eval, default=False)
parser.add_argument('--sde_adjoint', type=eval, default=False)
parser.add_argument('--niters', type=int, default=1000) # originally 5000
parser.add_argument('--lr', type=float, default=0.004)
parser.add_argument('--reg_lambda', type=float, default=0)
parser.add_argument('--hurst', type=float, default=0.7)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--num_paths', type=int, default=10)
args = parser.parse_args()

#DICT_DATANAME_STOCK = ["SPX", "TPX", "SX5E"]
#DICT_DATANAME_STOCK = ["SX5E"]
#DICT_DATANAME_STOCK = ["SPX"]
DICT_DATANAME_STOCK = ["TPX"]
DICT_DATANAME_fOU = ['fOU_H0.7', 'fOU_H0.8', 'fOU_H0.9']
#DICT_DATANAME_fOU = ['fOU_H0.7']
#DICT_DATANAME_fOU = ['fOU_H0.8']
#DICT_DATANAME_fOU = ['fOU_H0.9']
DICT_DATANAME_SHORT_fOU = ['fOU_H0.1', 'fOU_H0.2', 'fOU_H0.3', 'fOU_H0.4']
#DICT_DATANAME_OTHER = ['NileMin', 'ethernet', 'videoVBR', 'NBSdiff', 'NhemiTemp']
#DICT_DATANAME_OTHER = ['ethernet','videoVBR', 'NBSdiff']
#DICT_DATANAME_OTHER = ['NileMin']
#DICT_DATANAME_OTHER = ['ethernet']
#DICT_DATANAME_OTHER = ['videoVBR']
DICT_DATANAME_OTHER = ['NBSdiff']
#DICT_DATANAME_OTHER = ['NhemiTemp']
#DICT_DATANAME_OTHER = ['NileMin', 'videoVBR', 'NBSdiff', 'NhemiTemp']
#DICT_DATANAME_OTHER = ['NileMin', 'ethernet', 'NBSdiff', 'NhemiTemp']
DICT_DATANAME_SHORT = ['ar1_short_memory']
DICT_DATANAME_ROUGH = ['log_volatility_sp500']
DICT_DATANAME_OU = ['alpha=-2', 'alpha=-10', 'alpha=-20', 'alpha=-50', 'alpha=-100']
#DICT_DATANAME_OU = ['alpha=-20', 'alpha=-50', 'alpha=-100']
#DICT_DATANAME_OU = ['alpha=-50']
#DICT_DATANAME_FBM = ['fBM_H0.1', 'fBM_H0.2', 'fBM_H0.3', 'fBM_H0.4', 'fBM_H0.6', 'fBM_H0.7', 'fBM_H0.8', 'fBM_H0.9']
#DICT_DATANAME_FBM = ['fBM_H0.2', 'fBM_H0.3', 'fBM_H0.4', 'fBM_H0.6', 'fBM_H0.7', 'fBM_H0.8', 'fBM_H0.9']
DICT_DATANAME_FBM = ['fBM_H0.2', 'fBM_H0.3', 'fBM_H0.4']
#DICT_DATANAME_FBM = ['fBM_H0.4']


#DICT_DATANAME = ['NileMin']
#DICT_DATANAME = ['ethernet']
#DICT_DATANAME = ['NhemiTemp','videoVBR']
#DICT_DATANAME = DICT_DATANAME_OTHER
#DICT_DATANAME = DICT_DATANAME_STOCK + DICT_DATANAME_fOU + DICT_DATANAME_OTHER
#DICT_DATANAME =  DICT_DATANAME_fOU + DICT_DATANAME_OTHER
#DICT_DATANAME = DICT_DATANAME_fOU
#DICT_DATANAME = DICT_DATANAME_STOCK
DICT_DATANAME = DICT_DATANAME_OTHER
#DICT_DATANAME = DICT_DATANAME_SHORT
#DICT_DATANAME = DICT_DATANAME_ROUGH
#DICT_DATANAME = DICT_DATANAME_SHORT_fOU
#DICT_DATANAME = DICT_DATANAME_OU
#DICT_DATANAME = DICT_DATANAME_FBM

#DICT_METHOD = ['fSDE']
#DICT_METHOD = ['SDE']
#DICT_METHOD = ['RNN', 'SDE', 'fSDE']
#DICT_METHOD = ['ArmaSDE']
#DICT_METHOD = ['NNKernelArmaSDE']
#DICT_METHOD = ['RNN', 'SDE', 'fSDE', 'ArmaSDE']
DICT_METHOD = ['RNN', 'SDE', 'fSDE', 'ArmaSDE', 'NNKernelArmaSDE']

#ts_points = ['2010/1/4', '2020/12/31', '2021/11/11'] # train_start, train_end=test_start, test_end 
#ts_points = ['1986/4/10', '2015/12/31', '2021/11/11'] 
#ts_points = ['2000/1/3', '2020/12/31', '2021/11/11'] 
stock_ts_points = ['2000/1/3', '2020/12/31', '2021/11/11'] 
split_rate = 0.8
#fOU_ts_points = ['0', '900', '1000']
#other_ts_points = ['0', '600', '663']

resume_training = True #true when using saved parameters

if args.ode_adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

if args.sde_adjoint:
    from torchsde import sdeint_adjoint as sdeint
else:
    from torchsde import sdeint
from utils.fsde_solver import fsdeint


def train(data_name, method):
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    
    dir_name = base_dir + "/result/" + data_name + "/train_params"
    save_file_name = dir_name + f"/{data_name}_{method}_params.pt"
    save_file_name = os.path.abspath(save_file_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    # generate data
    if data_name in DICT_DATANAME_STOCK:
        train_data, test_data, train_ts_str, test_ts_str, train_ts, test_ts = get_stock_data(stock_ts_points, data_name)
    elif data_name in DICT_DATANAME_fOU: 
        train_data, test_data, train_ts_str, test_ts_str, train_ts, test_ts = get_fOU_data(data_name, split_rate)   
    elif data_name in DICT_DATANAME_OTHER:
        train_data, test_data, train_ts_str, test_ts_str, train_ts, test_ts = get_other_data(data_name, split_rate)
    elif data_name in DICT_DATANAME_SHORT: 
        train_data, test_data, train_ts_str, test_ts_str, train_ts, test_ts = get_short_memory_data(split_rate)
    elif data_name in DICT_DATANAME_ROUGH: 
        train_data, test_data, train_ts_str, test_ts_str, train_ts, test_ts = get_rough_data(split_rate)
    elif data_name in DICT_DATANAME_SHORT_fOU: 
        train_data, test_data, train_ts_str, test_ts_str, train_ts, test_ts = get_short_fOU_data(data_name, split_rate)
    elif data_name in DICT_DATANAME_OU: 
        train_data, test_data, train_ts_str, test_ts_str, train_ts, test_ts = get_OU_data(data_name, split_rate)
    elif data_name in DICT_DATANAME_FBM: 
        train_data, test_data, train_ts_str, test_ts_str, train_ts, test_ts = get_FBM_data(data_name, split_rate)
    train_data = torch.from_numpy(train_data).float().to(device) 
    test_data = torch.from_numpy(test_data).float().to(device)
    train_ts = torch.from_numpy(train_ts).float().to(device)
    test_ts = torch.from_numpy(test_ts).float().to(device) 

    #print("train_ts:{}".format(train_ts))
    
    ts_total = torch.cat((train_ts.reshape(-1), test_ts[1:]))
    data_total = torch.cat((train_data.reshape(-1), test_data.reshape(-1)[1:]))
    ts_total_str = list(train_ts_str) + list(test_ts_str[1:])

    # model
    # Call instance
    #rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, batch_dim).to(device)
    #dec = Decoder(latent_dim, obs_dim, nhidden).to(device)
    #if method == "ODE":
    #    func_ODE = LatentODEfunc().to(device)
    #    params = (list(func_ODE.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    #elif method == "SDE":
    #    func_SDE = LatentSDEfunc().to(device)
    #    params = (list(func_SDE.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    #elif method == "fSDE":
    #    func_fSDE = LatentFSDEfunc().to(device)
    #    params = (list(func_fSDE.parameters())) 
    
    
    #params = []
    if method == "RNN":
        rnn = GeneratorRNN().to(device)
        params = list(rnn.parameters()) 
    elif method == "SDE":
        func_SDE = LatentSDEfunc().to(device)
        params = list(func_SDE.parameters()) 
    elif method == "fSDE":
        func_fSDE = LatentFSDEfunc().to(device)
        #fsdenet = FSDENet().to(device)
        params = (list(func_fSDE.parameters())) 
        #params = (list(fsdenet.parameters()))
    elif method == "ArmaSDE":
        func_ArmaSDE = LatentArmaSDEfunc().to(device)
        params = list(func_ArmaSDE.parameters())
    elif method == "NNKernelArmaSDE":
        func_NNKernelArmaSDE = LatentNNKernelArmaSDEfunc().to(device)
        params = list(func_NNKernelArmaSDE.parameters())
        """
        p = torch.tensor(p, dtype=torch.float32, requires_grad=True)
        theta = torch.tensor(theta, dtype=torch.float32, requires_grad=True)
        params.append(p)
        params.append(theta)
        """

    #print(params)
    #print([type(p) for p in params])

    optimizer = optim.Adam(params, lr=args.lr)
    loss_meter = RunningAverageMeter()

    if resume_training and os.path.exists(save_file_name):
        print(f"Loading checkpoint from {save_file_name}")
        checkpoint = torch.load(save_file_name)
        if method == "RNN":
            rnn.load_state_dict(checkpoint["model_state_dict"])
        elif method == "SDE":
            func_SDE.load_state_dict(checkpoint["model_state_dict"])
        elif method == "fSDE":
            func_fSDE.load_state_dict(checkpoint["model_state_dict"])
        elif method == "ArmaSDE":
            func_ArmaSDE.load_state_dict(checkpoint["model_state_dict"])
        elif method == "NNKernelArmaSDE":
            func_NNKernelArmaSDE.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_loss = checkpoint["best_loss"]
        print(f"Resumed training with best loss = {best_loss:.4f}")
    else:
        best_loss = float("inf")

    patience = args.niters  # Number to determine how many times in a row to stop if there is no improvement
    #patience = args.niters*0.2
    patience_counter = 0

    #for name, param in func_ArmaSDE.named_parameters():
        #print(f"{name}: requires_grad = {param.requires_grad}")
    
    for itr in range(1, args.niters + 1): #tqdm(range(1, args.niters + 1)):
        optimizer.zero_grad()
        #h = rec.initHidden().to(device)
        #for t in range(sample_trajs.size(1)):
        #    obs = sample_trajs[:, t].reshape(-1, 1)
        #    out, h = rec.forward(obs, h)
        #qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
        #epsilon = torch.randn(qz0_mean.size()).to(device)
        #z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean # dimension (batch_size, latent_size)
        #print(train_data.shape)
        z0 = torch.zeros(batch_dim, latent_dim) + train_data[0, 0]
        #print("z0: {}".format(z0))
        #print("z0_size:{}".format{z0.size()})
        #print(z0.shape)
        
        if method == "RNN":
            h = torch.randn(train_data.size(0), batch_dim, nhidden_rnn)
            z = z0
            pred_return = torch.zeros(batch_dim, latent_dim)
            for k in range(train_data.size(0)-1):        
                z, h_out = rnn(z, h[k])
                pred_return = torch.cat((pred_return, h_out), dim=1)
            pred_return = torch.cumsum(pred_return.unsqueeze(-1), dim=1)
            pred_z = torch.zeros(batch_dim, train_data.size(0), latent_dim) + train_data[0, 0] - pred_return
        elif method == "SDE":
            # dimension of sdeint is (t_size, batch_size, latent_size)
            pred_z = sdeint(func_SDE, z0, train_ts).permute(1, 0, 2)
        elif method == "fSDE":
            # dimension of fsdeint is (batch_size, t_size, latent_size)
            pred_z = fsdeint(func_fSDE, args.hurst, z0, train_ts) #.permute(0, 2, 1)
        elif method == "ArmaSDE":
            z_noise_state = torch.zeros(batch_dim, latent_dim)
            z0 = torch.cat([z0,z_noise_state], dim=1)
            pred_z = sdeint(func_ArmaSDE, z0, train_ts).permute(1, 0, 2)
            #print("pred_z: {}".format(pred_z))
            #print(pred_z.size())
            pred_z = pred_z[:,:,:latent_dim]
            #print("p: {:.4f}, theta: {:.4f}".format(p, theta))
        elif method == "NNKernelArmaSDE":
            z_noise_state = torch.zeros(batch_dim, latent_dim)
            z0 = torch.cat([z0,z_noise_state], dim=1)
            pred_z = sdeint(func_NNKernelArmaSDE, z0, train_ts).permute(1, 0, 2)
            #print("pred_z: {}".format(pred_z))
            #print(pred_z.size())
            pred_z = pred_z[:,:,:latent_dim]
        
        # compute loss
        #noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
        #noise_logvar = 2. * torch.log(noise_std_).to(device)
        #logpx = log_normal_pdf(
        #    sample_trajs, pred_x, noise_logvar).sum(-1).sum(-1)
        #pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
        #analytic_kl = normal_kl(qz0_mean, qz0_logvar,
        #                        pz0_mean, pz0_logvar).sum(-1)
        #loss = torch.mean(-logpx + analytic_kl, dim=0)
        #loss_meter.update(loss.item())
        #if itr%5==0:
        #    print('Iter: {}, loss: {:.4f}'.format(itr, -loss_meter.avg))

        with torch.autograd.set_detect_anomaly(True):
            loss = - calculate_log_likelihood(pred_z[:,:,0], train_data[:,0])
            #loss_acf = acf_loss(pred_z[:,:,0], train_data[:,0])
            #loss_path = pathwise_mse_loss(pred_z[:,:,0], train_data[:,0])

            #loss = 0
            #loss += loss_acf
            #loss += loss_path
        
            reg_lambda = args.reg_lambda
            reg = torch.tensor(0.) 
            for param in params:
                reg += torch.norm(param, 1)
            loss += reg_lambda * reg

            loss.backward()

            #print("raw_pgrad: {}, raw_thetagrad: {}".format(func_ArmaSDE.raw_p.grad, func_ArmaSDE.raw_theta.grad))
            #print("raw_p: {}, raw_theta: {}".format(func_ArmaSDE.raw_p, func_ArmaSDE.raw_theta))
            #print("p: {}, theta: {}".format(func_ArmaSDE.p, func_ArmaSDE.theta.item()))
            
            optimizer.step()
        
        #if itr%5==0:
        print("Iter: {}, Log Likelihood: {:.4f}, Regularization: {:.4f}".format(itr, -loss, reg)) 

         
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            # Save model
            torch.save({
                'model_state_dict': (rnn if method == "RNN" else
                                    func_SDE if method == "SDE" else
                                    func_fSDE if method == "fSDE" else
                                    func_ArmaSDE if method == "ArmaSDE" else
                                    func_NNKernelArmaSDE).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, save_file_name)
            print(f"Saved new best model at iter {itr}, loss = {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at iter {itr}, best loss = {best_loss:.4f}")
                break
         
    print(f'Training complete after {itr} iters.\n')


    # Restore best model before generation
    print(f"Loading best model from {save_file_name} for generation...")
    checkpoint = torch.load(save_file_name)
    if method == "RNN":
        rnn.load_state_dict(checkpoint["model_state_dict"])
    elif method == "SDE":
        func_SDE.load_state_dict(checkpoint["model_state_dict"])
    elif method == "fSDE":
        func_fSDE.load_state_dict(checkpoint["model_state_dict"])
    elif method == "ArmaSDE":
        func_ArmaSDE.load_state_dict(checkpoint["model_state_dict"])
    elif method == "NNKernelArmaSDE":
        func_NNKernelArmaSDE.load_state_dict(checkpoint["model_state_dict"])
    
    
    # Generation of sample paths
    with torch.no_grad():
        # sample from trajectorys' approx. posterior
        #h = rec.initHidden().to(device)
        #for t in range(sample_trajs.size(1)):
        #    obs = sample_trajs[:, t].reshape(-1, 1)
        #    out, h = rec.forward(obs, h)
        #qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
        #epsilon = torch.randn(qz0_mean.size()).to(device)
        #z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        x0 = torch.zeros(batch_dim, latent_dim) + train_data[0, 0]
        #xs_gen = []

        
        if method == 'RNN':
            h = torch.randn(data_total.size(0), batch_dim, nhidden_rnn)
            x = x0
            return_pred = torch.zeros(batch_dim, latent_dim)
            for k in range(data_total.size(0)-1):        
                x, h_out = rnn(x, h[k])
                return_pred = torch.cat((return_pred, h_out), dim=1)
            return_pred = torch.cumsum(return_pred.unsqueeze(-1), dim=1)
            xs_gen = torch.zeros(batch_dim, data_total.size(0), latent_dim) + train_data[0, 0] - return_pred
        elif method == 'SDE':
            xs_gen = sdeint(func_SDE, x0, ts_total).permute(1, 0, 2)
        elif method == 'fSDE':
            xs_gen = fsdeint(func_fSDE, args.hurst, x0, ts_total)
        elif method == 'ArmaSDE':
            x_noise_state = torch.zeros(batch_dim, latent_dim)
            x0 = torch.cat([x0,x_noise_state], dim=1)
            xs_gen = sdeint(func_ArmaSDE, x0, ts_total).permute(1, 0, 2)
            xs_gen = xs_gen[:,:,:latent_dim]
        elif method == 'NNKernelArmaSDE':
            x_noise_state = torch.zeros(batch_dim, latent_dim)
            x0 = torch.cat([x0,x_noise_state], dim=1)
            xs_gen = sdeint(func_NNKernelArmaSDE, x0, ts_total).permute(1, 0, 2)
            xs_gen = xs_gen[:,:,:latent_dim]
        
        plot_original_path(data_name, ts_total, data_total)
        plot_generated_paths(min([args.num_paths, batch_dim]), data_name, method, ts_total, data_total, xs_gen)
        xs_gen_np = tensor_to_numpy(xs_gen[:,:,0]) 
        save_csv(data_name, method, ts_total_str, data_total.reshape(-1), xs_gen_np)
        plot_hist(data_name, method, xs_gen_np[0], train_data)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("base_dir",base_dir)
    for key_data in DICT_DATANAME:
        for key_method in DICT_METHOD:
            print(f"Training begin with data:{key_data}, method:{key_method}") 
            train(data_name = key_data, method=key_method)
    

