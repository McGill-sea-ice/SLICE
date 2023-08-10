#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:26:44 2020

@author: Amelie
"""
import tensorflow as tf

import copy
import os

import numpy as np
import matplotlib.pyplot as plt

import datetime as dt
import dateutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
#=========================================================================================
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
#=========================================================================================
def plot_series(time, series, format="-", start=0, end=None, ax_labels=['Time','Value']):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])
    plt.grid(True)


# Load the series
fp = './'
series_data = np.load(fp+'ML_timeseries_3years_ALXN6_Kingston.npz',allow_pickle='TRUE')
Twater = series_data['Tw']
Tair = series_data['Ta']
# time = series_data['time']

time = np.arange((15*364))
period = 365.
A = 20
offset = 15
Twater = A*np.sin((2*np.pi/period)*time) + offset

# Tw_min = np.nanmin(Twater)
# Tw_max = np.nanmax(Twater)
# Twater = (Twater-np.nanmin(Twater))/(np.nanmax(Twater)-np.nanmin(Twater))

Tw_mean = np.nanmean(Twater)
Tw_std = np.nanstd(Twater)
Twater = (Twater-np.nanmean(Twater))/np.nanstd(Twater)


target = Twater
plt.figure()
plt.plot(time,Twater)

# Split predictor and traget series in train-Valid sets
# (Here it is assumed that the test set will be available in the future as new observations
# are available, since the model should be trained with all the latest data as possible)
split_time = int(0.8*(len(time)))
time_train = time[:split_time]
x1_train = Twater[:split_time]
y_train = target[:split_time]
time_valid = time[split_time:]
x1_valid = Twater[split_time:]
y_valid = target[split_time:]


#%%=========================================================================================
def generate_sliding_window_samples_lstm(Xin,Yin,ninput,ntarget,nslide,input_dim=1,output_dim=1):
# Only complete samples are created, remainder of inout and output arrays aere ignored.
    nsamples = np.floor((len(Xin[0])-(ninput+ntarget))/float(nslide)).astype(int)+1

    Xout = np.zeros((nsamples,ninput,len(Xin)))*np.nan
    Yout = np.zeros((nsamples,ntarget,len(Yin)))*np.nan

    for iw in range(nsamples):
        istart = (iw*nslide)
        iend   = istart+ninput
        for il in range(len(Xin)):
            Xl = Xin[il]
            Xout[iw,:,il] = Xl[istart:iend]
        for jl in range(len(Yin)):
            Yl = Yin[jl]
            Yout[iw,:,jl] = Yl[iend:iend+ntarget]

    return Xout, Yout

def generate_sliding_window_samples_ndim(Xin,Yin,ninput,ntarget,nslide,input_dim=1,output_dim=1):
# Only complete samples are created, remainder of inout and output arrays aere ignored.
    nsamples = np.floor((len(Xin[0])-(ninput+ntarget))/float(nslide)).astype(int)+1

    Xout = np.zeros((nsamples,ninput*len(Xin)))*np.nan
    Yout = np.zeros((nsamples,ntarget))*np.nan

    for iw in range(nsamples):
        istart = (iw*nslide)
        iend   = istart+ninput
        for il in range(len(Xin)):
            Xl = Xin[il]
            Xout[iw,il*ninput:(il+1)*ninput] = Xl[istart:iend]
        Yout[iw] = Yin[iend:iend+ntarget]

    return Xout, Yout

def generate_sliding_window_samples(Xin,Yin,ninput,ntarget,nslide):
# Only complete samples are created, remainder of inout and output arrays aere ignored.
    nsamples = np.floor((Xin.shape[0]-(ninput+ntarget))/float(nslide)).astype(int)+1

    Xout = np.zeros((nsamples,ninput))*np.nan
    Yout = np.zeros((nsamples,ntarget))*np.nan

    for iw in range(nsamples):
        istart = (iw*nslide)
        iend   = istart+ninput
        Xout[iw] = Xin[istart:iend]
        Yout[iw] = Yin[iend:iend+ntarget]

    return Xout, Yout

def windowed_dataset_lstm(xin,yin,len_input,len_targets,nslide,batch_size,shuffle_opt):
    x_window, y_window = generate_sliding_window_samples_lstm(xin,yin,len_input,len_targets,nslide)
    x_tensor = torch.from_numpy(x_window).float()
    y_tensor = torch.from_numpy(y_window).float()
    data_loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size, shuffle=shuffle_opt)
    return data_loader,x_window,y_window

def windowed_dataset_ndim(xin,yin,len_input,len_targets,nslide,batch_size,shuffle_opt):
    x_window, y_window = generate_sliding_window_samples_ndim(xin,yin,len_input,len_targets,nslide)
    x_tensor = torch.from_numpy(x_window).float()
    y_tensor = torch.from_numpy(y_window).float()
    data_loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size, shuffle=shuffle_opt)
    return data_loader,x_window,y_window

def windowed_dataset(xin,yin,len_input,len_targets,nslide,batch_size,shuffle_opt):
    x_window, y_window = generate_sliding_window_samples(xin,yin,len_input,len_targets,nslide)
    x_tensor = torch.from_numpy(x_window).float()
    y_tensor = torch.from_numpy(y_window).float()
    data_loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size, shuffle=shuffle_opt)
    return data_loader,x_window,y_window

class LinearNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.linear_in = nn.Linear(input_size, 200)
        self.linear_mid1 = nn.Linear(200, 100)
        self.linear_mid2 = nn.Linear(100, 40)
        self.linear_out = nn.Linear(40, output_size)

    def forward(self, x):
        x = F.relu(self.linear_in(x))
        x = F.relu(self.linear_mid1(x))
        x = F.relu(self.linear_mid2(x))
        out = self.linear_out(x)

        return out

class LSTMLinear(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size, n_layers):
        super(LSTMLinear, self).__init__()
        self.LSTM = nn.LSTM(input_dim,
                            hidden_size,
                            n_layers)

        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # the input to an LSTM must be of size (seq_len, batch_size, input_dim)
        x = x.transpose(0, 1)
        output, _ = self.LSTM(x)
        pred = self.linear(output[-1])
        return pred

def train_model(epoch, model, train_loader, optimizer, loss_fct, device):
    # activate the training mode
    model.train()
    torch.set_grad_enabled(True)
    total_loss = 0

    # iteration over the mini-batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # transfer the data on the chosen device
        data, target = data.to(device), target.to(device)
        # reinitialize the gradients to zero
        optimizer.zero_grad()
        # forward propagation on the data
        prediction = model(data)
        # compute the cost function w.r.t. the targets
        loss = (loss_fct(prediction, target))
        # execute the backpropagation
        loss.backward()
        # execute an optimization step
        optimizer.step()
        # accumulate the loss
        total_loss += loss.item()*len(data)

    # compute the average cost per epoch
    mean_loss = total_loss/len(train_loader.dataset)

    print('Train Epoch: {}   Avg_Loss: {:.5f}  '.format(epoch, mean_loss))
    return mean_loss

def evaluate_model(model, eval_loader, loss_fct, device):
    # activate the evaluation mode
    model.eval()
    total_loss = 0

    with torch.no_grad():
        # iterate over the batches
        for batch_idx, (data, target) in enumerate(eval_loader):
            # transfer the data on the chosen device
            data, target = data.to(device), target.to(device)
            # forward propagation on the data
            prediction = model(data)
            # compute the cost function w.r.t. the targets
            loss = (loss_fct(prediction, target))
            # accumulate the loss
            total_loss += loss.item()*len(data)

    # compute the average cost per epoch
    mean_loss = total_loss/len(eval_loader.dataset)

    print('Eval:  Avg_Loss: {:.5f}  '.format(mean_loss))
    return mean_loss


def plot_losses(t_loss,v_loss,new_fig):
    xplot = list(range(len(t_loss)))
    if new_fig:
        plt.figure()
        ax = plt.subplot(111)
        plt.plot(xplot, t_loss, 'r', label="Train")
        plt.plot(xplot, v_loss, 'm', label="Validation")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=False, fancybox=False)
        leg.get_frame().set_alpha(0.99)
        plt.grid()
    else:
        ax = plt.subplot(111)
        plt.plot(xplot, t_loss, 'r', label="Train")
        plt.plot(xplot, v_loss, 'g', label="Validation")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=False, fancybox=False)
        leg.get_frame().set_alpha(0.99)
        plt.grid()


# Setting the seed to a fixed value can be helpful in reproducing results
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Create windowed samples and batched dataset
seqlen_input = 21
seqlen_target = 1
nslide = 1
batch_size = 8
shuffle_train = True
shuffle_valid = False

predictors_train = [x1_train]
predictors_valid = [x1_valid]
dl_train,input_train,target_train = windowed_dataset_lstm(predictors_train,[y_train],seqlen_input,seqlen_target,nslide,batch_size,shuffle_train)
dl_valid,input_valid,target_valid = windowed_dataset_lstm(predictors_valid,[y_valid],seqlen_input,seqlen_target,nslide,batch_size,shuffle_valid)

# predictors_train = [x1_train,x2_train]
# predictors_valid = [x1_valid,x2_valid]
# dl_train,input_train,target_train = windowed_dataset_ndim(predictors_train,y_train,seqlen_input,seqlen_target,nslide,batch_size,shuffle_train)
# dl_valid,input_valid,target_valid = windowed_dataset_ndim(predictors_valid,y_valid,seqlen_input,seqlen_target,nslide,batch_size,shuffle_valid)

# predictors_train = [x1_train,x2_train,x3_train]
# predictors_valid = [x1_valid,x2_valid,x3_valid]
# dl_train,input_train,target_train = windowed_dataset_ndim(predictors_train,y_train,seqlen_input,seqlen_target,nslide,batch_size,shuffle_train)
# dl_valid,input_valid,target_valid = windowed_dataset_ndim(predictors_valid,y_valid,seqlen_input,seqlen_target,nslide,batch_size,shuffle_valid)


LSTMModel = LSTMLinear(input_dim=1, output_dim=1, hidden_size=24, n_layers=3)
LSTMModel.to(device)

init_rnn_weights = copy.deepcopy(LSTMModel.state_dict())
LSTMModel.load_state_dict(init_rnn_weights)

train_losses = []
valid_losses = []

# optimizer = optim.SGD(LSTMModel.parameters(),lr=1e-4, momentum=0.9)
optimizer = optim.Adam(LSTMModel.parameters(),lr=1e-4)
loss_function = nn.MSELoss() # Mean square error
# loss_function = nn.L1Loss() # Mean absolute error

n_epochs = 100

for epoch in range(1, n_epochs + 1):
    train_loss = train_model(epoch, LSTMModel, dl_train, optimizer, loss_function, device)
    valid_loss = evaluate_model(LSTMModel, dl_valid, loss_function, device)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

#%%
print("\n\n\nOptimization ended.\n")
plot_losses(train_losses,valid_losses,new_fig=True)

#%%
predictions = []
for it in range(input_valid.shape[0]):
    input_pred = np.expand_dims(input_valid[it,:,:],0)
    predictions.append(LSTMModel(torch.from_numpy(input_pred).float()).detach().numpy())
plt.figure()
plot_series(time_valid[seqlen_input:seqlen_input+target_valid.size],target_valid.ravel(),'.-',ax_labels=['Time', 'T$_{w}$'])
plot_series(time_valid[seqlen_input:seqlen_input+target_valid.size],np.array(predictions).ravel(),'.-',ax_labels=['Time', 'T$_{w}$'])
plt.title('Predictors: T$_{air}$, T$_{water}$, |T$_{water}$-T$_{air}$| (previous ' + '%2i'%seqlen_input + 'days)')
MAE=nn.L1Loss()
print(MAE(torch.from_numpy(np.squeeze(target_valid)).float(),torch.from_numpy(np.array(predictions).astype(float)).float()))
print(np.corrcoef(np.squeeze(target_valid),np.squeeze(np.array(predictions)))[0,1])



#%%
predictions = []
for it in range(input_train.shape[0]):
    input_pred = np.expand_dims(input_train[it,:,:],0)
    predictions.append(LSTMModel(torch.from_numpy(input_pred).float()).detach().numpy())


plt.figure()
plot_series(time_train[seqlen_input:seqlen_input+target_train.size],np.squeeze(target_train.ravel()))
plot_series(time_train[seqlen_input:seqlen_input+target_train.size],np.squeeze(np.array(predictions).ravel()),ax_labels=['Time', 'T$_{w}$'])

print(np.corrcoef(np.squeeze(target_train),np.squeeze(np.array(predictions)))[0,1])


