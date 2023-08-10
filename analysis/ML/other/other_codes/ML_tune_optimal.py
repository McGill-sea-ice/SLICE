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
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import datetime as dt
import dateutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, MedianStoppingRule

from torch.utils.data import DataLoader, TensorDataset
#=========================================================================================
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
#=========================================================================================

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
    period=[2012,2014]
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
    date_start=(dt.date(period[0],1,1)-date_ref).days
    date_end=(dt.date(period[1]+1,1,1)-date_ref).days

    istart = np.where(data_set[:,0]==date_start)[0][0]
    iend = np.where(data_set[:,0]==date_end)[0][0]

    data_set_select = data_set[istart:iend,:]

    if normalize:
        data_set_select = normalize_data(data_set_select,norm_f)

    # Split predictor and traget series in train-valid sets
    # (here it is assumed that the test set will be available in the future as new observations
    # are available, since the model should be trained with all the latest data as possible)
    split_time = int(np.round(split*data_set_select.shape[0]))
    train_dataset = data_set_select[:split_time,1:]
    valid_dataset = data_set_select[split_time:,1:]
    # time_train = data_set_select[:split_time,0]
    # time_valid = data_set_select[split_time:,0]

    # train_dataset[np.isnan(train_dataset)]=-1.2 # Quick fix: two data points were not cleaned and are still Nan...

    return train_dataset, valid_dataset


#%%
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

def windowed_dataset_ndim(xin,yin,len_input,len_targets,nslide,batch_size,shuffle_opt):
    x_window, y_window = generate_sliding_window_samples_ndim(xin,yin,len_input,len_targets,nslide)
    x_tensor = torch.from_numpy(x_window).float()
    y_tensor = torch.from_numpy(y_window).float()
    data_loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size, shuffle=shuffle_opt)
    return data_loader,x_window,y_window


#%%
class Net(nn.Module):
    def __init__(self, ws=98, l1=200, l2=100, l3=40):
        super(Net, self).__init__()
        self.fcin = nn.Linear(ws, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, l3)
        self.fcout = nn.Linear(l3, 14)

    def forward(self, x):
        x = F.relu(self.fcin(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fcout(x)
        return x

#%%
# def train_model_config(config, checkpoint_dir=None, filename=None):
#     # Setting the seed to a fixed value can be helpful in reproducing results
#     seed = 42
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)

#     # Load the series
#     train_ds, valid_ds = load_data(filename)

#     seqlen_input = config["ws"]
#     seqlen_target = 32
#     nslide = 32
#     shuffle_train = True
#     shuffle_valid = False

#     predictors_train = [train_ds[:,0],train_ds[:,1]]
#     predictors_valid = [valid_ds[:,0],valid_ds[:,1]]
#     # predictors_train = [train_ds[:,0]]
#     # predictors_valid = [valid_ds[:,0]]
#     target_train = train_ds[:,0]
#     target_valid = valid_ds[:,0]
#     trainloader,input_train,target_train = windowed_dataset_ndim(predictors_train,target_train,seqlen_input,seqlen_target,nslide,config["batch_size"],shuffle_train)
#     validloader,input_valid,target_valid = windowed_dataset_ndim(predictors_valid,target_valid,seqlen_input,seqlen_target,nslide,config["batch_size"],shuffle_valid)

#     net = Net(config["ws"]*2,config["l1"],config["l2"],config["l3"])
#     # net = Net(seqlen_input*len(predictors_train),seqlen_target,config["l1"],config["l2"],config["l3"])
#     net.to(device)

#     loss_function = nn.MSELoss()
#     optimizer = optim.SGD(net.parameters(),lr=config["lr"], momentum=0.9)
#     # optimizer = optim.Adam(net.parameters(),lr=config["lr"])

#     if checkpoint_dir:
#         model_state, optimizer_state = torch.load(
#             os.path.join(checkpoint_dir, "checkpoint"))
#         net.load_state_dict(model_state)
#         optimizer.load_state_dict(optimizer_state)

#     train_losses = []
#     valid_losses = []

#     n_epochs = 500

#     for epoch in range(1, n_epochs + 1):

#         # ----- TRAINING -----
#         # activate the training mode
#         net.train()
#         torch.set_grad_enabled(True)

#         train_loss = 0

#         for batch_idx, (data, target) in enumerate(trainloader):
#             # transfer the data on the chosen device
#             inputs, target = data.to(device), target.to(device)
#             # reinitialize the gradients to zero
#             optimizer.zero_grad()
#             # forward + backward + optimize
#             outputs = net(inputs)
#             loss = loss_function(outputs, target)
#             loss.backward()
#             optimizer.step()
#             # accumulate the loss
#             train_loss += loss.item()*len(inputs)

#         # compute the average loss (accross batches) for this epoch
#         mean_train_loss = train_loss/len(trainloader.dataset)
#         train_losses.append(mean_train_loss)
#         print('Train Epoch: {}   Avg_Loss: {:.5f}  '.format(epoch, mean_train_loss))

#         # ----- VALIDATION -----
#         # activate the evaluation mode
#         net.eval()
#         valid_loss = 0

#         val_loss_cpu = 0
#         val_steps = 0

#         with torch.no_grad():
#              for batch_idx, (data, target) in enumerate(validloader):
#                  # transfer the data on the chosen device
#                  inputs, target = data.to(device), target.to(device)
#                  # forward propagation on the data
#                  outputs = net(inputs)
#                  loss = loss_function(outputs, target)
#                  valid_loss += loss.item()*len(data)

#                  val_loss_cpu += loss.cpu().numpy()
#                  val_steps += 1

#         mean_valid_loss = valid_loss/len(validloader.dataset)
#         valid_losses.append(mean_valid_loss)
#         print('Eval:  Avg_Loss: {:.5f}  '.format(mean_valid_loss))

#         with tune.checkpoint_dir(epoch) as checkpoint_dir:
#                 path = os.path.join(checkpoint_dir, "checkpoint")
#                 torch.save((net.state_dict(), optimizer.state_dict()), path)

#         tune.report(loss=(val_loss_cpu / val_steps))

#     print("\n\n\nTraining ended.\n")

#     return train_losses, valid_losses

#%%
def train_model(config, filename=None):
    # Setting the seed to a fixed value can be helpful in reproducing results
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



    # Load the series
    train_ds, valid_ds = load_data(filename)

    seqlen_input = config["ws"]
    seqlen_target = 14
    nslide = 14
    shuffle_train = True
    shuffle_valid = False

    predictors_train = [train_ds[:,0],train_ds[:,1]]
    predictors_valid = [valid_ds[:,0],valid_ds[:,1]]
    # predictors_train = [train_ds[:,0]]
    # predictors_valid = [valid_ds[:,0]]
    target_train = train_ds[:,0]
    target_valid = valid_ds[:,0]
    trainloader,input_train,target_train = windowed_dataset_ndim(predictors_train,target_train,seqlen_input,seqlen_target,nslide,config["batch_size"],shuffle_train)
    validloader,input_valid,target_valid = windowed_dataset_ndim(predictors_valid,target_valid,seqlen_input,seqlen_target,nslide,config["batch_size"],shuffle_valid)

    net = Net(config["ws"]*2,config["l1"],config["l2"],config["l3"])
    # net = Net(seqlen_input*len(predictors_train),seqlen_target,config["l1"],config["l2"],config["l3"])
    net.to(device)

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(),lr=config["lr"], momentum=0.9)
    # optimizer = optim.Adam(net.parameters(),lr=config["lr"])

    train_losses = []
    valid_losses = []

    n_epochs = 1000

    for epoch in range(1, n_epochs + 1):

        # ----- TRAINING -----
        # activate the training mode
        net.train()
        torch.set_grad_enabled(True)

        train_loss = 0

        for batch_idx, (data, target) in enumerate(trainloader):
            # transfer the data on the chosen device
            inputs, target = data.to(device), target.to(device)
            # reinitialize the gradients to zero
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, target)
            loss.backward()
            optimizer.step()
            # accumulate the loss
            train_loss += loss.item()*len(inputs)

        # compute the average loss (accross batches) for this epoch
        mean_train_loss = train_loss/len(trainloader.dataset)
        train_losses.append(mean_train_loss)
        print('Train Epoch: {}   Avg_Loss: {:.5f}  '.format(epoch, mean_train_loss))

        # ----- VALIDATION -----
        # activate the evaluation mode
        net.eval()
        valid_loss = 0

        val_loss_cpu = 0
        val_steps = 0

        with torch.no_grad():
             for batch_idx, (data, target) in enumerate(validloader):
                 # transfer the data on the chosen device
                 inputs, target = data.to(device), target.to(device)
                 # forward propagation on the data
                 outputs = net(inputs)
                 loss = loss_function(outputs, target)
                 valid_loss += loss.item()*len(data)

                 val_loss_cpu += loss.cpu().numpy()
                 val_steps += 1

        mean_valid_loss = valid_loss/len(validloader.dataset)
        valid_losses.append(mean_valid_loss)
        print('Eval:  Avg_Loss: {:.5f}  '.format(mean_valid_loss))

    print("\n\n\nTraining ended.\n")

    return train_losses, valid_losses


#%%
# def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):

#     #Load data
#     filename = os.path.abspath('../../data/ML_timeseries/ML_dataset_Lasallepatchlinear_SouthShoreCanal_MontrealDorval.npz')
#     load_data(filename)
#     # Define the sampling range/method for each parameter, e.g. :
#     # l1,l2,l3 should be powers of 2 between 4 and 256, so either 4, 8, 16, 32, 64, 128, or 256.
#     # lr (learning rate) should be uniformly sampled between 0.0001 and 0.1.
#     # batch_size is a choice between 2, 4, 8, and 16.
#     config = {
#         "ws": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
#         "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
#         "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
#         "l3": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
#         "lr": tune.loguniform(1e-5, 1e-1),
#         "batch_size": tune.choice([2, 4, 8, 16])
#     }
#     # The ASHAScheduler will terminate bad performing trials early.
#     scheduler = ASHAScheduler(
#         metric="loss",
#         mode="min",
#         max_t=max_num_epochs,
#         grace_period=400,
#         reduction_factor=2)

#     # scheduler = MedianStoppingRule(
#     #     metric="loss",
#     #     mode="min",
#     #     # min_time_slice=max_num_epochs,
#     #     min_samples_required = 5,
#     #     grace_period=25)


#     reporter = CLIReporter(
#         # parameter_columns=["ws", "l1", "l2","l3", "lr", "batch_size"],
#         metric_columns=["loss", "accuracy", "training_iteration"])

#     result = tune.run(
#         partial(train_model_config, filename=filename),
#         resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
#         config=config,
#         num_samples=num_samples,
#         scheduler=scheduler,
#         progress_reporter=reporter)


#     best_trial = result.get_best_trial("loss", "min", "last")
#     print("Best trial config: {}".format(best_trial.config))
#     print("Best trial final validation loss: {}".format(
#         best_trial.last_result["loss"]))

#     best_trained_model = Net(best_trial.config["ws"]*2, best_trial.config["l1"], best_trial.config["l2"], best_trial.config["l3"])
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if gpus_per_trial > 1:
#             best_trained_model = nn.DataParallel(best_trained_model)
#     best_trained_model.to(device)

#     best_checkpoint_dir = best_trial.checkpoint.value
#     model_state, optimizer_state = torch.load(os.path.join(
#         best_checkpoint_dir, "checkpoint"))
#     best_trained_model.load_state_dict(model_state)
#%%
# if __name__ == "__main__":
#     # You can change the number of GPUs per trial here:
#     main(num_samples=15, max_num_epochs=400, gpus_per_trial=0)
#%%

filename = os.path.abspath('../../data/ML_timeseries/ML_dataset_Lasallepatchlinear_SouthShoreCanal_MontrealDorval.npz')

config = {'ws': 49,
          'l1': 200,
          'l2': 100,
          'l3': 40,
          'lr': 5e-4,
          'batch_size': 8}

#  batch_size |   l1 |   l2 |   l3 |          lr |   ws
# 8 |  128 |   16 |  512 | 0.00267093  |   32
# 8 |   16 |  128 |  256 | 0.0023513   |   16

tr_lss, vl_lss = train_model(config,filename)


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


plot_losses(tr_lss,vl_lss,new_fig=True)

#%%




# predictions = []
# for it in range(input_valid.shape[0]):
#     predictions.append(LinearModel(torch.from_numpy(input_valid[it,:]).float()).detach().numpy())
# plt.figure()
# plot_series(time_valid[seqlen_input:seqlen_input+target_valid.size],target_valid.ravel(),'.-',ax_labels=['Time', 'T$_{w}$'])
# plot_series(time_valid[seqlen_input:seqlen_input+target_valid.size],np.array(predictions).ravel(),'.-',ax_labels=['Time', 'T$_{w}$'])
# plt.title('Predictors: T$_{air}$, T$_{water}$, |T$_{water}$-T$_{air}$| (previous ' + '%2i'%seqlen_input + 'days)')
# MAE=nn.L1Loss()
# print(MAE(torch.from_numpy(np.squeeze(target_valid)).float(),torch.from_numpy(np.array(predictions).astype(float)).float()))
# print(np.corrcoef(np.squeeze(target_valid),np.squeeze(np.array(predictions)))[0,1])


