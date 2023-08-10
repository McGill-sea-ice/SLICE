#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:26:44 2020

@author: Amelie
"""
import tensorflow as tf

import copy
import time
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
def plot_series(time, series, format="-", start=0, end=None, ax_labels=['Time','Value'],linecolor=''):
    plt.plot(time[start:end], series[start:end], format, color=linecolor)
    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])
    plt.grid(True)

def plot_series_1step(time, series, format="-", ax_labels=['Time','Value'],linecolor=''):
    plt.plot(time, series, format, color=linecolor)
    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])
    plt.grid(True)

#%%=========================================================================================
def normalize_data(data_in, norm = 'mean_std', time_included = True):
    data_out = np.zeros(data_in.shape)

    if time_included:
        data_tmp = data_in[:,1:]
        data_out[:,0] = data_in[:,0]
        i0 = 1
    else:
        data_tmp = data_in
        i0 = 0

    if norm == 'max_min':
        for i in range(data_tmp.shape[1]):
            c_max = np.nanmax(data_tmp[:,i])
            c_min = np.nanmin(data_tmp[:,i])
            c_norm = (data_tmp[:,i]-c_min)/(c_max-c_min)
            data_out[:,i+i0] = c_norm

    if norm == 'mean_std':
        for i in range(data_tmp.shape[1]):
            c_mean = np.nanmean(data_tmp[:,i])
            c_std = np.nanstd(data_tmp[:,i])
            c_norm = (data_tmp[:,i]-c_mean)/(c_std)
            data_out[:,i+i0] = c_norm

    return data_out


def load_data(filename='../../data/ML_timeseries/ML_dataset_Lasallepatchlinear_SouthShoreCanal_MontrealDorval.npz'):
    # Load full data set
    vars_out = ['Twater','TEMP']
    period=[2005,2020]
    normalize=True
    norm_f='max_min'
    split =0.7

    with np.load(filename, allow_pickle='TRUE') as data:
        ds = data['data']
        date_ref = data['date_ref']
        var = [k.decode('UTF-8') for k in data['labels']]

    # Initialize output array and list of selected variables
    data_set = np.zeros((ds.shape[0],len(vars_out)+1))
    var_list_out = []

    # First column is always time
    data_set[:,0] = ds[:,0]
    var_list_out.append(var[0])

    # Fill other columns with selected variables
    for k in range(len(vars_out)):
        idx = [i for i,v in enumerate(np.array(var)) if vars_out[k] in v]
        data_set[:,k+1] = np.squeeze(ds[:,idx[0]])
        var_list_out.append(var[idx[0]])

    # Keep only rows corresponding to the selected period
    # date_start=(dt.date(period[0],1,1)-date_ref).days
    date_start=(dt.date(period[0],12,8)-date_ref).days
    date_end=(dt.date(period[1],12,31)-date_ref).days

    istart = np.where(data_set[:,0]==date_start)[0][0]
    iend = np.where(data_set[:,0]==date_end)[0][0]

    # data_set_select = data_set[istart:iend,:]

    data_set_tmp = data_set[istart:iend,:]
    data_set_select = np.zeros((data_set_tmp.shape[0],data_set_tmp.shape[1]+1))
    data_set_select[:,0] = data_set_tmp[:,0]
    data_set_select[:,1] = data_set_tmp[:,1]
    data_set_select[:,2] = data_set_tmp[:,2]
    data_set_select[:,3] = data_set_tmp[:,1] - data_set_tmp[:,2]

    if normalize:
        data_set_select = normalize_data(data_set_select,norm_f)

    # Split predictor and traget series in train-valid sets
    # (here it is assumed that the test set will be available in the future as new observations
    # are available, since the model should be trained with all the latest data as possible)
    split_time = int(np.round(split*data_set_select.shape[0]))
    train_dataset = data_set_select[:split_time,1:]
    valid_dataset = data_set_select[split_time:,1:]
    time_train = data_set_select[:split_time,0]
    time_valid = data_set_select[split_time:,0]

    train_dataset[np.isnan(train_dataset)]= -2 # Quick fix: two data points were not cleaned and are still Nan...
    valid_dataset[np.isnan(valid_dataset)]= -2 # Quick fix: two data points were not cleaned and are still Nan...


    return train_dataset, valid_dataset, time_train, time_valid

#%%
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

def windowed_dataset_lstm(xin,yin,len_input,len_targets,nslide,batch_size,shuffle_opt):
    x_window, y_window = generate_sliding_window_samples_lstm(xin,yin,len_input,len_targets,nslide)
    x_tensor = torch.from_numpy(x_window).float()
    y_tensor = torch.from_numpy(y_window).float()
    data_loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size, shuffle=shuffle_opt)
    return data_loader,x_window,y_window


# class LSTMLinear(nn.Module):

#     def __init__(self, input_dim, output_dim, hidden_size, n_layers):
#         super(LSTMLinear, self).__init__()
#         self.LSTM = nn.LSTM(input_dim,
#                             hidden_size,
#                             n_layers)

#         self.linear = nn.Linear(hidden_size, output_dim)

#     def forward(self, x):
#         # the input to an LSTM must be of size (seq_len, batch_size, input_dim)
#         x = x.transpose(0, 1)
#         output, _ = self.LSTM(x)
#         pred = self.linear(output[-1])
#         return pred

class LSTMLinear(nn.Module):

    def __init__(self, input_dim, output_dim, seqlen_target, hidden_size, n_layers):
        super(LSTMLinear, self).__init__()
        self.LSTM = nn.LSTM(input_dim,
                            hidden_size,
                            n_layers)
        self.fwindow = seqlen_target
        self.linear = nn.Linear(hidden_size, output_dim)
        self.linout = nn.Linear(14, seqlen_target)

    def forward(self, x):
        # the input to an LSTM must be of size (seq_len, batch_size, input_dim)
        x = x.transpose(0, 1)
        output, _ = self.LSTM(x) # output size is: [seq_len, batch_size, hidden_size]
        pred = self.linear(output[-self.fwindow:])
        pred = pred.transpose(0,1)
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

#%%
# Setting the seed to a fixed value can be helpful in reproducing results
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Load data
filename='../../data/ML_timeseries/ML_dataset_DesBaillets_SouthShoreCanal_MontrealDorvalMontrealPETMontrealMcTavishmerged.npz'

train_ds, valid_ds, time_train, time_valid = load_data(filename)
#%%
# Create windowed samples and batched dataset
seqlen_input = 30
seqlen_target = 1
nslide = 1
batch_size = 4
shuffle_train = True
shuffle_valid = False

# predictors_train = [train_ds[:,0]]
# predictors_valid = [valid_ds[:,0]]
# predictors_train = [train_ds[:,0],train_ds[:,1]]
# predictors_valid = [valid_ds[:,0],valid_ds[:,1]]
# predictors_train = [train_ds[:,0],train_ds[:,2]]
# predictors_valid = [valid_ds[:,0],valid_ds[:,2]]
predictors_train = [train_ds[:,0],train_ds[:,1],train_ds[:,2]]
predictors_valid = [valid_ds[:,0],valid_ds[:,1],valid_ds[:,2]]
y_train = train_ds[:,0]
y_valid = valid_ds[:,0]
dl_train,input_train,target_train = windowed_dataset_lstm(predictors_train,[y_train],seqlen_input,seqlen_target,nslide,batch_size,shuffle_train)
dl_valid,input_valid,target_valid = windowed_dataset_lstm(predictors_valid,[y_valid],seqlen_input,seqlen_target,nslide,batch_size,shuffle_valid)
#%%
# predictors_train = [x1_train,x2_train]
# predictors_valid = [x1_valid,x2_valid]
# dl_train,input_train,target_train = windowed_dataset_ndim(predictors_train,y_train,seqlen_input,seqlen_target,nslide,batch_size,shuffle_train)
# dl_valid,input_valid,target_valid = windowed_dataset_ndim(predictors_valid,y_valid,seqlen_input,seqlen_target,nslide,batch_size,shuffle_valid)

# predictors_train = [x1_train,x2_train,x3_train]
# predictors_valid = [x1_valid,x2_valid,x3_valid]
# dl_train,input_train,target_train = windowed_dataset_ndim(predictors_train,y_train,seqlen_input,seqlen_target,nslide,batch_size,shuffle_train)
# dl_valid,input_valid,target_valid = windowed_dataset_ndim(predictors_valid,y_valid,seqlen_input,seqlen_target,nslide,batch_size,shuffle_valid)


LSTMModel = LSTMLinear(input_dim=len(predictors_train), output_dim=1, seqlen_target=seqlen_target, hidden_size=128, n_layers=2)
LSTMModel.to(device)

init_rnn_weights = copy.deepcopy(LSTMModel.state_dict())
LSTMModel.load_state_dict(init_rnn_weights)

train_losses = []
valid_losses = []

optimizer = optim.Adam(LSTMModel.parameters(),lr=2e-5)
# optimizer = optim.SGD(LSTMModel.parameters(),lr=2e-4, momentum=0.9)
loss_function = nn.MSELoss() # Mean square error
# loss_function = nn.L1Loss() # Mean absolute error

n_epochs = 50

for epoch in range(1, n_epochs + 1):
    train_loss = train_model(epoch, LSTMModel, dl_train, optimizer, loss_function, device)
    valid_loss = evaluate_model(LSTMModel, dl_valid, loss_function, device)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)


print("\n\n\nOptimization ended.\n")

#%%
plot_losses(train_losses,valid_losses,new_fig=True)

#%%
predictions = []
for it in range(input_valid.shape[0]):
    input_pred = np.expand_dims(input_valid[it,:,:],0)
    predictions.append(LSTMModel(torch.from_numpy(input_pred).float()).detach().numpy())

MAE=nn.L1Loss()
MSE=nn.MSELoss()

# One-step ahead
plt.figure()
plot_series_1step(time_valid[seqlen_input:],target_valid.ravel(),'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
plot_series_1step(time_valid[seqlen_input:],np.array(predictions).ravel(),'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))
plt.title('Predictors: T$_{air}$, T$_{water}$, |T$_{water}$-T$_{air}$| (previous ' + '%2i'%seqlen_input + 'days)')
print(MAE(np.squeeze(torch.from_numpy(target_valid).float()),np.squeeze(torch.from_numpy(np.array(predictions).astype(float)).float())))
print(MSE(np.squeeze(torch.from_numpy(target_valid).float()),np.squeeze(torch.from_numpy(np.array(predictions).astype(float)).float())))
print(np.corrcoef(target_valid.ravel(),np.array(predictions).ravel())[0,1])

# Multi-step ahead
# plt.figure()
# for s in range(target_valid.shape[0]):
# # for s in np.arange(0,target_valid.shape[0],seqlen_target):
#     plot_series(time_valid[seqlen_input+(nslide*s):seqlen_input+seqlen_target+(nslide*s)],np.squeeze(target_valid[s,:]),'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
#     plot_series(time_valid[seqlen_input+(nslide*s):seqlen_input+seqlen_target+(nslide*s)],np.squeeze(np.array(predictions[s])),'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(s))
# plt.title('Predictors: T$_{air}$, T$_{water}$, |T$_{water}$-T$_{air}$| (previous ' + '%2i'%seqlen_input + 'days)')
# print(MAE(torch.from_numpy(target_valid).float(),torch.from_numpy(np.array(predictions).astype(float)).float()))
# print(MSE(torch.from_numpy(target_valid).float(),torch.from_numpy(np.array(predictions).astype(float)).float()))
# print(np.corrcoef(target_valid.ravel(),np.array(predictions).ravel())[0,1])


#%%
predictions = []
for it in range(input_train.shape[0]):
    input_pred = np.expand_dims(input_train[it,:],0)
    predictions.append(LSTMModel(torch.from_numpy(input_pred).float()).detach().numpy())

plt.figure()
for s in range(target_train.shape[0]):
# for s in np.arange(0,target_train.shape[0],seqlen_target):
    plot_series(time_train[seqlen_input+(nslide*s):seqlen_input+seqlen_target+(nslide*s)],target_train[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
    plot_series(time_train[seqlen_input+(nslide*s):seqlen_input+seqlen_target+(nslide*s)],np.squeeze(np.array(predictions[s])),'-',ax_labels=['Time', 'T$_{w}$'],linecolor='orange')


# plt.figure()
# plot_series(time_train[seqlen_input:seqlen_input+target_train.size],np.squeeze(target_train.ravel()))
# plot_series(time_train[seqlen_input:seqlen_input+target_train.size],np.squeeze(np.array(predictions).ravel()),ax_labels=['Time', 'T$_{w}$'])

print(np.corrcoef(target_train.ravel(),np.array(predictions).ravel())[0,1])


