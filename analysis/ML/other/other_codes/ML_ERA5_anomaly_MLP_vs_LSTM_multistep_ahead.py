#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:23:02 2021

Tutorial from: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/structured_data/time_series.ipynb#scrollTo=YeCWbq6KLmL7
Alternative link : https://www.tensorflow.org/tutorials/structured_data/time_series

@author: Amelie
"""

import copy
import time
import os

import numpy as np
import matplotlib.pyplot as plt

import datetime as dt
import dateutil
import calendar

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

import tensorflow as tf

from functions import rolling_climo

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

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



# def rolling_clim(var_in, clim_years, recons_years, t, Nwindow = 31):
#     # NOTE: Only odd window size are possible
#     # t = time[istart:iend]
#     # clim_years = years[yr_start:yr_end+1]
#     # var_in = Twater[:,icity]

#     date_ref = dt.date(1900,1,1)

#     # First re-arrange data to have each window, for each DOY, each year
#     data = np.zeros((Nwindow,366,len(clim_years)))*np.nan
#     var_tmp = var_in.copy()

#     for it in range(var_tmp.shape[0]):
#         iw0 = np.max([0,it-int((Nwindow-1)/2)])
#         iw1 = np.min([it+int((Nwindow-1)/2)+1,len(var_tmp)])

#         var_window = np.zeros(Nwindow)*np.nan
#         var_window[0:len(var_tmp[iw0:iw1])] = var_tmp[iw0:iw1]

#         date_mid = date_ref+dt.timedelta(days=int(t[it]))
#         year_mid = date_mid.year
#         month_mid = date_mid.month
#         day_mid = date_mid.day

#         if len(np.where(clim_years == year_mid)[0]) > 0:
#             iyear = np.where(clim_years == year_mid)[0][0]
#             doy = (dt.date(year_mid,month_mid,day_mid)-dt.date(year_mid,1,1)).days

#             data[:,doy,iyear] = var_window

#             if not calendar.isleap(year_mid) and (doy == 364):
#                 imid = int((Nwindow-1)/2)

#                 var_window_366 = np.zeros((Nwindow))*np.nan
#                 var_window_366[imid] = np.array(np.nanmean([var_tmp[it],var_tmp[it+1]]))
#                 var_window_366[0:imid] = var_tmp[int(it+1-((Nwindow-1)/2)):it+1]
#                 var_window_366[imid+1:Nwindow] = var_tmp[it+1:int(it+1+((Nwindow-1)/2))]
#                 data[:,365,iyear] = var_window_366

#     # Then, find the window climatological mean and std for each DOY
#     clim_mean = np.zeros((len(t)))*np.nan
#     clim_std = np.zeros((len(t)))*np.nan

#     clim_mean_tmp = np.nanmean(data,axis=(0,2))
#     clim_std_tmp = np.nanstd(data,axis=(0,2))

#     for iyr,year in enumerate(recons_years):

#         date=(dt.date(year,1,1)-date_ref).days
#         i0 = np.where(t == date)[0][0]
#         i1 = i0+365+calendar.isleap(year)

#         # clim_mean[i0:i1] = clim_mean_tmp[0:365+int(calendar.isleap(year))]
#         # clim_std[i0:i1] = clim_std_tmp[0:365+int(calendar.isleap(year))]

#         if year == recons_years[-1]:
#             clim_mean[i0:i1] = clim_mean_tmp[0:365+int(calendar.isleap(year))-1]
#             clim_std[i0:i1] = clim_std_tmp[0:365+int(calendar.isleap(year))-1]
#         else:
#             clim_mean[i0:i1] = clim_mean_tmp[0:365+int(calendar.isleap(year))]
#             clim_std[i0:i1] = clim_std_tmp[0:365+int(calendar.isleap(year))]



#     return clim_mean, clim_std

#%%
# Options
plot = True
verbose = True

filename='../../../data/ML_timeseries/ML_dataset_TwaterLongueuil_ERA5regionD_QLasallePointeClaireSteAnnedeBellevue_PDOersstv3.npz'
# These are the vars in file:
 # 'Days since 1900-01-01',
 # 'Twater city [deg.C] - Longueuil',
 # 'Binary ice/no ice (1/0)',
 # 'Twater-Tair [deg.C]',
 # 'Avg. Ta_max [deg C]',
 # 'Avg. Ta_min [deg C]',
 # 'Avg. Ta_mean [deg C]',
 # 'Tot. TDD [deg C]',
 # 'Tot. FDD [deg C]',
 # 'Tot. precip. [m]',
 # 'Avg. SLP [kPa]',
 # 'Avg. uwind [m/s]',
 # 'Avg. vwind [m/s]',
 # 'Avg. wind speed [m/s]',
 # 'Tot. snowfall [m of water equiv.]',
 # 'Avg. cloud cover [%]',
 # 'Avg. spec. hum. []',
 # 'Avg. rel. hum. [%]',
 # 'Avg. St-L. level [m]',
 # 'Avg. Ott. Riv. level [m]',
 # 'Avg. St-L. discharge [m^3/s]',
 # 'Monthly NAO index',
 # 'Monthly PDO index'

vars_out = ['Twater ','Twater-Tair', 'Ta_mean']
# vars_out = ['Twater ','Twater-Tair', 'Ta_mean','discharge']
# vars_out = ['Twater ','Twater-Tair', 'Ta_mean','NAO']
period=[1992,2016]
normalize=True
split_valid = 0.62
split_test = 0.999

context_window = 96
pred_window = 32

#%%
# Load Data
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


#%%
# Compute Tw climatology;
nw = 1
years = np.arange(30)+1991
Tw_climatology_mean, Tw_climatology_std, _ = rolling_climo(nw, data_set[:,1],'all_time',data_set[:,0],years)

# Compute Tw anomaly w/r to climatology:
Tw_anomaly = data_set[:,1]-Tw_climatology_mean

# Keep only rows corresponding to the selected period
# date_start=(dt.date(period[0],1,1)-date_ref).days
date_start=(dt.date(period[0],1,1)-date_ref).days
date_end=(dt.date(period[1]+1,2,1)-date_ref).days

istart = np.where(data_set[:,0]==date_start)[0][0]
iend = np.where(data_set[:,0]==date_end)[0][0]

data_set_select = data_set[istart:iend,:]
Tw_climatology_mean, Tw_climatology_std = Tw_climatology_mean[istart:iend], Tw_climatology_std[istart:iend]
Tw_anomaly = Tw_anomaly[istart:iend]

colabels = var_list_out

# Print statistics
if verbose:
    for i in range(data_set_select.shape[1]):
        print("count: "+ str(np.sum(~np.isnan(data_set_select[:,i]))))
        print("nanmean: " + str(np.nanmean(data_set_select[:,i])))
        print("nanstd: " + str(np.nanstd(data_set_select[:,i])))
        print("nanmin: " + str(np.nanmin(data_set_select[:,i])))
        print("nan25%: " + str(np.nanpercentile(data_set_select[:,i],25)))
        print("nan50%: " + str(np.nanpercentile(data_set_select[:,i],50)))
        print("nan75%: " + str(np.nanpercentile(data_set_select[:,i],75)))
        print("nanmax: " + str(np.nanmax(data_set_select[:,i])))
        print("---------------------------")

#%%
# Feature Engineering: add DOY feature with a sin+cos
year = 365.2425
data_set_select=np.hstack((data_set_select,np.zeros((data_set_select.shape[0],2))))

for it,t in enumerate(data_set_select[:,0]):
    date = date_ref+dt.timedelta(days=t)
    doy = (dt.date(date.year,date.month,date.day)-dt.date(date.year,1,1)).days +1
    data_set_select[it,-2] = np.sin(doy * (2*np.pi/year))
    data_set_select[it,-1] = np.cos(doy * (2*np.pi/year))

colabels.append("sin(DOY)")
colabels.append("cos(DOY)")


#TEST!!!
# Replace Twater with Twater_anomaly
data_set_select[:,1] = Tw_anomaly
colabels[1] = 'Twater - climatology'

#%%
# Add climatology as the last column
Tw_climatology_mean[np.isnan(Tw_anomaly)] = np.nan
data_set_select=np.hstack((data_set_select,np.zeros((data_set_select.shape[0],1))))
data_set_select[:,-1] = Tw_climatology_mean


# Visualize data
if plot:
    fig,ax = plt.subplots(nrows=data_set_select.shape[1],ncols=1,figsize=(8,8),sharex=True)
    for i in range(data_set_select.shape[1]):
        ax[i].plot(data_set_select[:,0],data_set_select[:,i])


#%%

# Split predictor and traget series in train-valid-test sets
split_valid_it = int(np.round(split_valid*data_set_select.shape[0]))
split_test_it = int(np.round(split_test*data_set_select.shape[0]))

train_dataset = data_set_select[:split_valid_it,1:]
valid_dataset = data_set_select[split_valid_it:split_test_it,1:]
test_dataset = data_set_select[split_test_it:,1:]
time_train = data_set_select[:split_valid_it,0]
time_valid = data_set_select[split_valid_it:split_test_it,0]
time_test = data_set_select[split_test_it:,0]

Tw_climatology_mean_train = Tw_climatology_mean[:split_valid_it]
Tw_climatology_mean_valid = Tw_climatology_mean[split_valid_it:split_test_it]
Tw_climatology_mean_test = Tw_climatology_mean[split_test_it:]

colabels.pop(0) # Remove time as a variable from column labels

#%%
# Normalize data
if normalize:
    train_mean = np.zeros(train_dataset.shape[1])
    train_std = np.zeros(train_dataset.shape[1])
    for i in range(train_dataset.shape[1]):
        if (i == (train_dataset.shape[1]-1)):
            train_mean[i] = train_mean[0]
            train_std[i] = train_std[0]

            train_dataset[:,i] = (train_dataset[:,i] - train_mean[i]) / train_std[i]
            valid_dataset[:,i] = (valid_dataset[:,i] - train_mean[i]) / train_std[i]
            test_dataset[:,i] = (test_dataset[:,i] - train_mean[i]) / train_std[i]

        else:
            train_mean[i] = np.nanmean(train_dataset[:,i])
            train_std[i] = np.nanstd(train_dataset[:,i])

            train_dataset[:,i] = (train_dataset[:,i] - train_mean[i]) / train_std[i]
            valid_dataset[:,i] = (valid_dataset[:,i] - train_mean[i]) / train_std[i]
            test_dataset[:,i] = (test_dataset[:,i] - train_mean[i]) / train_std[i]

    Tw_climatology_mean_valid = (Tw_climatology_mean_valid - train_mean[0])/train_std[0]
    Tw_climatology_mean_test = (Tw_climatology_mean_test - train_mean[0])/train_std[0]
    Tw_climatology_mean_train = (Tw_climatology_mean_train - train_mean[0])/train_std[0]

#%%
    if plot:
        fig,ax = plt.subplots(nrows=train_dataset.shape[1],ncols=1,figsize=(8,8),sharex = True)
        for i in range(train_dataset.shape[1]):
            if i == 0:
                ax[i].plot(time_train,Tw_climatology_mean_train,color='gray')
            ax[i].plot(time_train,train_dataset[:,i])

        fig,ax = plt.subplots(nrows=valid_dataset.shape[1],ncols=1,figsize=(8,8),sharex = True)
        for i in range(valid_dataset.shape[1]):
            if i == 0:
                ax[i].plot(time_valid,Tw_climatology_mean_valid,color='gray')
            ax[i].plot(time_valid,valid_dataset[:,i])

        fig,ax = plt.subplots(nrows=test_dataset.shape[1],ncols=1,figsize=(8,8),sharex = True)
        for i in range(test_dataset.shape[1]):
            if i == 0:
                ax[i].plot(time_test,Tw_climatology_mean_test,color='gray')
            ax[i].plot(time_test,test_dataset[:,i])



#%%
def sliding_window_samples(data,time_in,input_width,label_width,shift,nslide,input_columns,label_columns):
    # data shape is (time, features)
    input_data = data[:,input_columns]
    label_data = data[:,label_columns]


    # Only complete samples are created, remainder of inout and output arrays aere ignored.
    nsamples = np.floor((data.shape[0]-(input_width+shift))/float(nslide)).astype(int)+1

    Xout = np.zeros((nsamples,input_width,len(input_columns)))*np.nan
    Yout = np.zeros((nsamples,label_width,len(label_columns)))*np.nan

    input_time = np.zeros((nsamples,input_width))*np.nan
    label_time = np.zeros((nsamples,label_width))*np.nan

    iw_n = 0
    for iw in range(nsamples):
        istart = (iw*nslide)
        iend   = istart+input_width

        if (np.sum(np.isnan(input_data[istart:iend])) > 0 ) | (np.sum(np.isnan(label_data[iend+shift-label_width:iend+shift])) > 0 ):
            # Do not consider example if there is one or more "nan" in it.
            continue
        else:
            Xout[iw_n] = input_data[istart:iend]
            Yout[iw_n] = label_data[iend+shift-label_width:iend+shift]
            input_time[iw_n] = time_in[istart:iend]
            label_time[iw_n] = time_in[iend+shift-label_width:iend+shift]

            iw_n += 1

    Xout = Xout[0:iw_n]
    Yout = Yout[0:iw_n]
    input_time = input_time[0:iw_n]
    label_time = label_time[0:iw_n]

    return Xout, Yout, input_time, label_time


def make_dataset(x_window,y_window,batch_size,shuffle_opt):
    x = torch.from_numpy(x_window).float()
    y = torch.from_numpy(y_window).float()
    data_loader = DataLoader(TensorDataset(x, y), batch_size, shuffle=shuffle_opt)

    return data_loader

#%%
# # EXAMPLE OF HOW TO USE WINDOWING AND DATALOADERS:
# # Windowing infos:
# input_width,label_width,shift,nslide = 6, 1, 1, 1
# input_col = [0,1,2,3,4]
# label_col = [0]

# train_input_windows, train_label_windows = sliding_window_samples(train_dataset,input_width,label_width,shift,nslide,input_col,label_col)
# valid_input_windows, valid_label_windows = sliding_window_samples(valid_dataset,input_width,label_width,shift,nslide,input_col,label_col)
# test_input_windows, test_label_windows = sliding_window_samples(test_dataset,input_width,label_width,shift,nslide,input_col,label_col)

# # Data Loader info:
# bs = 4
# shuffle_train = True
# shuffle_valid = False
# shuffle_test  = False

# train_dl = make_dataset(train_input_windows,train_label_windows,bs,shuffle_train)
# valid_dl = make_dataset(valid_input_windows,valid_label_windows,bs,shuffle_valid)
# test_dl  = make_dataset(test_input_windows,test_label_windows,bs,shuffle_test)

#%%
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

    if verbose:
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

    if verbose:
        print('Eval:  Avg_Loss: {:.5f}  '.format(mean_loss))
    return mean_loss


#%%
# ONE-STEP AHEAD NAIVE FORECAST (BASELINE = NO CHANGE):
# USE CURRENT VALUE AT T AS PREDICTION FOR VALUE AT T+1

# class Baseline_onestep(nn.Module):
#   def __init__(self, label_index=None):
#     super(Baseline_onestep,self).__init__()
#     self.label_index = label_index

#   def forward(self, inputs):
#     if self.label_index is None:
#       return inputs
#     result = inputs[:, :, self.label_index]
#     return result[:, :, np.newaxis]

# # Setting the seed to a fixed value can be helpful in reproducing results
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

# # Windowing infos:
# input_width,label_width,shift,nslide = 1, 1, 1, 1
# input_col = [0,1,2,3,4]
# label_col = [0]

# input_train, target_train = sliding_window_samples(train_dataset,input_width,label_width,shift,nslide,input_col,label_col)
# input_valid, target_valid = sliding_window_samples(valid_dataset,input_width,label_width,shift,nslide,input_col,label_col)
# input_test, target_test = sliding_window_samples(test_dataset,input_width,label_width,shift,nslide,input_col,label_col)

# # Data Loader info:
# bs = 4
# shuffle_train = True
# shuffle_valid = False
# shuffle_test  = False

# train_dl = make_dataset(input_train,target_train,bs,shuffle_train)
# valid_dl = make_dataset(input_valid,target_valid,bs,shuffle_valid)
# test_dl  = make_dataset(input_test,target_test,bs,shuffle_test)

# # Instantiate Baseline model
# model = Baseline_onestep(label_index=0)
# model.to(device)

# predictions = []
# for it in range(input_valid.shape[0]):
#     input_pred = np.expand_dims(input_valid[it,:,:],0)
#     predictions.append(model(torch.from_numpy(input_pred).float()).detach().numpy())


# # Evaluate skill on valid set...
# MAE=nn.L1Loss()
# MSE=nn.MSELoss()
# x_renorm = (np.squeeze(torch.from_numpy(target_valid).float())*train_std[0]) + train_mean[0]
# y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions).astype(float)).float())*train_std[0]) + train_mean[0]
# print(MAE(x_renorm,y_renorm))
# print(np.sqrt(MSE(x_renorm,y_renorm)))
# print(np.corrcoef(np.array(x_renorm).ravel(),np.array(y_renorm).ravel())[0,1])
# plt.figure()
# plot_series_1step(time_valid[input_width:],np.array(x_renorm).ravel(),'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
# plot_series_1step(time_valid[input_width:],np.array(y_renorm).ravel(),'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))
# plt.title('Predictors: T$_{air}$, T$_{water}$, |T$_{water}$-T$_{air}$| (previous ' + '%2i'%input_width + 'days)')

# # ...and compare with climatology
# x_renorm = (np.squeeze(torch.from_numpy(target_valid).float())*train_std[0]) + train_mean[0]
# y_renorm = torch.from_numpy((Tw_climatology_mean_valid[input_width:]*train_std[0]) + train_mean[0]).float()
# print(MAE(x_renorm,y_renorm))
# print(np.sqrt(MSE(x_renorm,y_renorm)))
# print(np.corrcoef(np.array(x_renorm).ravel(),np.array(y_renorm).ravel())[0,1])
# plt.figure()
# plot_series_1step(time_valid[input_width:],np.array(x_renorm).ravel(),'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
# plot_series_1step(time_valid[input_width:],np.array(y_renorm).ravel(),'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))
# plt.title('Predictors: T$_{air}$, T$_{water}$, |T$_{water}$-T$_{air}$| (previous ' + '%2i'%input_width + 'days)')


#%%
# LINEAR REGRESSION MODEL USING MULTIPLE INPUT TIME STEPS TO
# PREDICT ONE-STEP AHEAD

# class LinearMulti(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(LinearMulti, self).__init__()
#         self.linear = nn.Linear(input_size, output_size)
#         self.flatten = nn.Flatten()

#     def forward(self, x):
#         x = self.flatten(x)
#         out = self.linear(x)
#         out = torch.unsqueeze(out, 1) # This works for now because there is only one feature out... but not sure how to deal with multiple features out.
#         return out

# # Setting the seed to a fixed value can be helpful in reproducing results
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

# # Windowing infos:
# input_width,label_width,shift,nslide = 24, 1, 1, 1
# input_col = [0,1,2,3,4]
# label_col = [0]

# input_train, target_train = sliding_window_samples(train_dataset,input_width,label_width,shift,nslide,input_col,label_col)
# input_valid, target_valid = sliding_window_samples(valid_dataset,input_width,label_width,shift,nslide,input_col,label_col)
# input_test, target_test = sliding_window_samples(test_dataset,input_width,label_width,shift,nslide,input_col,label_col)

# # Data Loader info:
# bs = 4
# shuffle_train = True
# shuffle_valid = False
# shuffle_test  = False

# train_dl = make_dataset(input_train,target_train,bs,shuffle_train)
# valid_dl = make_dataset(input_valid,target_valid,bs,shuffle_valid)
# test_dl  = make_dataset(input_test,target_test,bs,shuffle_test)

# # Instantiate Baseline model
# model = LinearMulti(input_width*len(input_col),len(label_col))
# model.to(device)

# train_losses = []
# valid_losses = []

# optimizer = optim.Adam(model.parameters(),lr=1e-5)
# loss_function = nn.MSELoss() # Mean square error

# n_epochs = 100

# for epoch in range(1, n_epochs + 1):
#     train_loss = train_model(epoch, model, train_dl, optimizer, loss_function, device)
#     valid_loss = evaluate_model(model, valid_dl, loss_function, device)

#     train_losses.append(train_loss)
#     valid_losses.append(valid_loss)


# print("\n\n\nOptimization ended.\n")
# plot_losses(train_losses,valid_losses,new_fig=True)

# predictions = []
# for it in range(input_valid.shape[0]):
#     input_pred = np.expand_dims(input_valid[it,:,:],0)
#     predictions.append(model(torch.from_numpy(input_pred).float()).detach().numpy())


# # Evaluate skill on valid set...
# MAE=nn.L1Loss()
# MSE=nn.MSELoss()
# x_renorm = (np.squeeze(torch.from_numpy(target_valid).float())*train_std[0]) + train_mean[0]
# y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions).astype(float)).float())*train_std[0]) + train_mean[0]
# print(MAE(x_renorm,y_renorm))
# print(np.sqrt(MSE(x_renorm,y_renorm)))
# print(np.corrcoef(np.array(x_renorm).ravel(),np.array(y_renorm).ravel())[0,1])
# plt.figure()
# plot_series_1step(time_valid[input_width:],np.array(x_renorm).ravel(),'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
# plot_series_1step(time_valid[input_width:],np.array(y_renorm).ravel(),'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))
# plt.title('Predictors: T$_{air}$, T$_{water}$, |T$_{water}$-T$_{air}$| (previous ' + '%2i'%input_width + 'days)')


#%%
# DENSE (MLP) MODEL, USING MULTIPLE INPUT STEPS TO PREDICT
# MULTI-STEP AHEAD

class DenseMulti(nn.Module):
    def __init__(self, input_size, output_size):
        super(DenseMulti, self).__init__()
        self.linear_in = nn.Linear(input_size, 64)
        self.linear_mid = nn.Linear(64, 64)
        self.linear_out = nn.Linear(64, output_size)
        self.flatten = nn.Flatten()

        # self.linear_in = nn.Linear(input_size, 100)
        # self.linear_mid1 = nn.Linear(100, 100)
        # self.linear_mid2 = nn.Linear(200, 200)
        # self.linear_mid3 = nn.Linear(100, 40)
        # self.linear_out = nn.Linear(40, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.linear_in(x))
        x = F.relu(self.linear_mid(x))
        out = self.linear_out(x)
        # x = F.relu(self.linear_in(x))
        # x = F.relu(self.linear_mid1(x))
        # x = F.relu(self.linear_mid3(x))
        # out = self.linear_out(x)
        out = torch.unsqueeze(out, -1) # This works for now because there is only one feature out... but not sure how to deal with multiple features out.
        return out



# Setting the seed to a fixed value can be helpful in reproducing results
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Windowing infos:
input_width,label_width,shift,nslide = context_window, pred_window, pred_window, 1
input_col = [0,1,2,3,4,5]
label_col = [0]
clim_col = [-1]

input_train, target_train, time_input_train, time_target_train = sliding_window_samples(train_dataset,time_train,input_width,label_width,shift,nslide,input_col,label_col)
input_valid, target_valid, time_input_valid, time_target_valid  = sliding_window_samples(valid_dataset,time_valid,input_width,label_width,shift,nslide,input_col,label_col)
# input_test, target_test, time_input_test, time_target_test  = sliding_window_samples(test_dataset,time_test,input_width,label_width,shift,nslide,input_col,label_col)

_, clim_train, _, _ = sliding_window_samples(train_dataset,time_train,input_width,label_width,shift,nslide,input_col,clim_col)
_, clim_valid, _, _  = sliding_window_samples(valid_dataset,time_valid,input_width,label_width,shift,nslide,input_col,clim_col)
# _, clim_test, _, _  = sliding_window_samples(test_dataset,time_test,input_width,label_width,shift,nslide,input_col,clim_col)

# plt.figure()
# plt.plot(target_test[1,:,0]*train_std[0]+train_mean[0])
# plt.plot(clim_test[1,:,0]*train_std[0]+train_mean[0])
# plt.plot(target_test[1,:,0]*train_std[0]+train_mean[0]+clim_test[1,:,0]*train_std[0]+train_mean[0])



#%%

# Data Loader info:
bs = 4
shuffle_train = True
shuffle_valid = False
shuffle_test  = False

train_dl = make_dataset(input_train,target_train,bs,shuffle_train)
valid_dl = make_dataset(input_valid,target_valid,bs,shuffle_valid)
# test_dl  = make_dataset(input_test,target_test,bs,shuffle_test)

# Instantiate Baseline model
model = DenseMulti(input_width*len(input_col),label_width*len(label_col))
model.to(device)

train_losses = []
valid_losses = []

optimizer = optim.Adam(model.parameters(),lr=5e-6)
# optimizer = optim.SGD(model.parameters(),lr=5e-5, momentum=0.9)
loss_function = nn.MSELoss() # Mean square error

n_epochs = 20

for epoch in range(1, n_epochs + 1):
    train_loss = train_model(epoch, model, train_dl, optimizer, loss_function, device)
    valid_loss = evaluate_model(model, valid_dl, loss_function, device)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
if verbose:
    print("\n\n\nOptimization ended.\n")

#%%
plot_losses(train_losses,valid_losses,new_fig=True)

#%%
# Evaluate predictions
predictions_train = []
for it in range(input_train.shape[0]):
    input_pred = np.expand_dims(input_train[it,:,:],0)
    predictions_train.append(model(torch.from_numpy(input_pred).float()).detach().numpy())

predictions_valid = []
for it in range(input_valid.shape[0]):
    input_pred = np.expand_dims(input_valid[it,:,:],0)
    predictions_valid.append(model(torch.from_numpy(input_pred).float()).detach().numpy())

# predictions_test = []
# for it in range(input_test.shape[0]):
#     input_pred = np.expand_dims(input_test[it,:,:],0)
#     predictions_test.append(model(torch.from_numpy(input_pred).float()).detach().numpy())


#%%
np.savez('./test_ML_anomaly_pred',
          train_dataset = train_dataset,
          valid_dataset = valid_dataset,
          test_dataset = test_dataset,
          target_train = target_train,
          target_valid = target_valid,
          # target_test = target_test,
          predictions_train = predictions_train,
          predictions_valid = predictions_valid,
          # predictions_test = predictions_test,
          Tw_climatology_mean_train = Tw_climatology_mean_train,
          Tw_climatology_mean_valid = Tw_climatology_mean_valid,
          Tw_climatology_mean_test = Tw_climatology_mean_test,
          clim_train = clim_train,
          clim_valid = clim_valid,
          # clim_test = clim_test,
          time_train = time_train,
          time_valid = time_valid,
          time_test = time_test,
          time_input_train = time_input_train,
          time_input_valid = time_input_valid,
          # time_input_test = time_input_test,
          time_target_train = time_target_train,
          time_target_valid = time_target_valid,
          # time_target_test = time_target_test,
          train_std = train_std[0],
          train_mean = train_mean[0],
          input_width = input_width,
          label_width = label_width,
          shift = shift,
          nslide = nslide,
          nepochs = epoch,
          date_ref = date_ref)


# #%%
# # Evaluate skill on valid set...
# predictions = []
# for it in range(input_valid.shape[0]):
#     input_pred = np.expand_dims(input_valid[it,:,:],0)
#     predictions.append(model(torch.from_numpy(input_pred).float()).detach().numpy())

# MAE=nn.L1Loss()
# MSE=nn.MSELoss()
# x_renorm = (np.squeeze(torch.from_numpy(target_valid).float())*train_std[0]) + train_mean[0]
# y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions).astype(float)).float())*train_std[0]) + train_mean[0]
# print('MLP MODEL, '+ str(pred_window)+ '-STEP AHEAD -----------')
# print('VALID')
# print(MAE(x_renorm,y_renorm))
# print(np.sqrt(MSE(x_renorm,y_renorm)))
# print(np.corrcoef(np.array(x_renorm).ravel(),np.array(y_renorm).ravel())[0,1])
# plt.figure()
# # First add climatology
# x_renorm_clim = (np.squeeze(torch.from_numpy(target_valid).float())*train_std[0]) + train_mean[0]
# y_renorm_clim = torch.from_numpy((Tw_climatology_mean_valid[input_width:]*train_std[0]) + train_mean[0]).float()
# plot_series_1step(time_valid[input_width:],np.array(y_renorm_clim).ravel(),'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(0))
# # Then add model:
# # x_renorm = (np.squeeze(torch.from_numpy(target_valid).float())*train_std[0]) + train_mean[0]
# # y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions).astype(float)).float())*train_std[0]) + train_mean[0]
# clim_recons = (np.squeeze(torch.from_numpy(clim_valid).float())*train_std[0]) + train_mean[0]
# x_renorm = (np.squeeze(torch.from_numpy(target_valid).float())*train_std[0]) + train_mean[0] + clim_recons
# y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions).astype(float)).float())*train_std[0]) + train_mean[0] + clim_recons
# for s in np.arange(0,target_valid.shape[0]-(input_width+shift),label_width):
#     plot_series(time_valid[input_width+(nslide*s):input_width+label_width+(nslide*s)],np.array(x_renorm)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
#     plot_series(time_valid[input_width+(nslide*s):input_width+label_width+(nslide*s)],np.array(y_renorm)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))
# plt.title('Predictors: T$_{air}$, T$_{water}$, |T$_{water}$-T$_{air}$| (previous ' + '%2i'%input_width + 'days)')


#%%
# # Evaluate skill on train set...
# predictions = []
# for it in range(input_train.shape[0]):
#     input_pred = np.expand_dims(input_train[it,:,:],0)
#     predictions.append(model(torch.from_numpy(input_pred).float()).detach().numpy())

MAE=nn.L1Loss()
MSE=nn.MSELoss()
x_renorm = (np.squeeze(torch.from_numpy(target_train).float())*train_std[0]) + train_mean[0]
y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions_train).astype(float)).float())*train_std[0]) + train_mean[0]
print('MLP MODEL, '+ str(pred_window)+ '-STEP AHEAD -----------')
print('TRAIN')
print(MAE(x_renorm,y_renorm))
print(np.sqrt(MSE(x_renorm,y_renorm)))
print(np.corrcoef(np.array(x_renorm).ravel(),np.array(y_renorm).ravel())[0,1])
plt.figure()
clim_recons = (np.squeeze(torch.from_numpy(clim_train).float())*train_std[0]) + train_mean[0]
x_renorm = (np.squeeze(torch.from_numpy(target_train).float())*train_std[0]) + train_mean[0] + clim_recons
y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions_train).astype(float)).float())*train_std[0]) + train_mean[0] + clim_recons
for s in np.arange(0,target_train.shape[0]-(input_width+shift),label_width):
    plot_series(time_target_train[s,:],np.array(x_renorm)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
    plot_series(time_target_train[s,:],np.array(y_renorm)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))
plt.title('Predictors: T$_{air}$, T$_{water}$, |T$_{water}$-T$_{air}$| (previous ' + '%2i'%input_width + 'days)')


#%%
# LSTM MODEL TO PREDICT MULTI-STEP AHEAD

# class LSTMLinear(nn.Module):
#     def __init__(self, input_dim, output_dim, seqlen_target, hidden_size, n_layers):
#         super(LSTMLinear, self).__init__()
#         self.LSTM = nn.LSTM(input_dim,
#                             hidden_size,
#                             n_layers)
#         self.fwindow = seqlen_target
#         self.linear = nn.Linear(hidden_size, output_dim)
#         self.linear_target = nn.Linear(hidden_size, seqlen_target)

#     def forward(self, x):
#         # the input to an LSTM must be of size (seq_len, batch_size, input_features)
#         x = x.transpose(0,1)
#         output, _ = self.LSTM(x) # output size is: [seq_len, batch_size, hidden_size]
#         # pred = self.linear(output[-self.fwindow:])
#         pred = self.linear_target(output[-1])
#         # pred = pred.transpose(0,1)
#         pred = torch.unsqueeze(pred, -1)

#         return pred

# # Setting the seed to a fixed value can be helpful in reproducing results
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

# # Windowing infos:
# input_width,label_width,shift,nslide = context_window, pred_window, pred_window, 1
# input_col = [0,1,2,3,4]
# label_col = [0]

# input_train, target_train = sliding_window_samples(train_dataset,input_width,label_width,shift,nslide,input_col,label_col)
# input_valid, target_valid = sliding_window_samples(valid_dataset,input_width,label_width,shift,nslide,input_col,label_col)
# # input_test, target_test = sliding_window_samples(test_dataset,input_width,label_width,shift,nslide,input_col,label_col)

# # Data Loader info:
# bs = 4
# shuffle_train = True
# shuffle_valid = False
# shuffle_test  = False

# train_dl = make_dataset(input_train,target_train,bs,shuffle_train)
# valid_dl = make_dataset(input_valid,target_valid,bs,shuffle_valid)
# # test_dl  = make_dataset(input_test,target_test,bs,shuffle_test)

# # Instantiate Baseline model
# model = LSTMLinear(len(input_col),len(label_col), seqlen_target=label_width, hidden_size=64, n_layers=1)
# model.to(device)

# train_losses = []
# valid_losses = []

# optimizer = optim.Adam(model.parameters(),lr=1e-5)
# loss_function = nn.MSELoss() # Mean square error

# n_epochs = 50

# for epoch in range(1, n_epochs + 1):
#     train_loss = train_model(epoch, model, train_dl, optimizer, loss_function, device)
#     valid_loss = evaluate_model(model, valid_dl, loss_function, device)

#     train_losses.append(train_loss)
#     valid_losses.append(valid_loss)
# if verbose:
#     print("\n\n\nOptimization ended.\n")

# plot_losses(train_losses,valid_losses,new_fig=True)


# #%%
# # Evaluate predictions
# predictions_train = []
# for it in range(input_train.shape[0]):
#     input_pred = np.expand_dims(input_train[it,:,:],0)
#     predictions_train.append(model(torch.from_numpy(input_pred).float()).detach().numpy())

# predictions_valid = []
# for it in range(input_valid.shape[0]):
#     input_pred = np.expand_dims(input_valid[it,:,:],0)
#     predictions_valid.append(model(torch.from_numpy(input_pred).float()).detach().numpy())

# predictions_test = []
# for it in range(input_test.shape[0]):
#     input_pred = np.expand_dims(input_test[it,:,:],0)
#     predictions_test.append(model(torch.from_numpy(input_pred).float()).detach().numpy())


# #%%
# np.savez('./test_ML_pred',
#           target_train = target_train,
#           target_valid = target_valid,
#           target_test = target_test,
#           predictions_train = predictions_train,
#           predictions_valid = predictions_valid,
#           predictions_test = predictions_test,
#           Tw_climatology_mean_train = Tw_climatology_mean_train,
#           Tw_climatology_mean_valid = Tw_climatology_mean_valid,
#           Tw_climatology_mean_test = Tw_climatology_mean_test,
#           clim_train = clim_train,
#           clim_valid = clim_valid,
#           clim_test = clim_test,
#           time_train = time_train,
#           time_valid = time_valid,
#           time_test = time_test,
#           train_std = train_std[0],
#           train_mean = train_mean[0],
#           input_width = input_width,
#           label_width = label_width,
#           shift = shift,
          # nslide = nslide,
          # date_ref = date_ref)



# # Evaluate skill on valid set...
# predictions = []
# for it in range(input_valid.shape[0]):
#     input_pred = np.expand_dims(input_valid[it,:,:],0)
#     predictions.append(model(torch.from_numpy(input_pred).float()).detach().numpy())

# MAE=nn.L1Loss()
# MSE=nn.MSELoss()
# x_renorm = (np.squeeze(torch.from_numpy(target_valid).float())*train_std[0]) + train_mean[0]
# y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions).astype(float)).float())*train_std[0]) + train_mean[0]
# print('LSTM MODEL, '+ str(pred_window)+ '-STEP AHEAD -----------')
# print('VALID')
# print(MAE(x_renorm,y_renorm))
# print(np.sqrt(MSE(x_renorm,y_renorm)))
# print(np.corrcoef(np.array(x_renorm).ravel(),np.array(y_renorm).ravel())[0,1])
# plt.figure()
# # First add climatology
# x_renorm = (np.squeeze(torch.from_numpy(target_valid).float())*train_std[0]) + train_mean[0]
# y_renorm = torch.from_numpy((Tw_climatology_mean_valid[input_width:]*train_std[0]) + train_mean[0]).float()
# plot_series_1step(time_valid[input_width:],np.array(y_renorm).ravel(),'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(0))
# # Then add model:
# x_renorm = (np.squeeze(torch.from_numpy(target_valid).float())*train_std[0]) + train_mean[0]
# y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions).astype(float)).float())*train_std[0]) + train_mean[0]
# for s in np.arange(0,target_valid.shape[0]-(input_width+shift),label_width):
#     plot_series(time_valid[input_width+(nslide*s):input_width+label_width+(nslide*s)],np.array(x_renorm)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
#     plot_series(time_valid[input_width+(nslide*s):input_width+label_width+(nslide*s)],np.array(y_renorm)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))
# plt.title('Predictors: T$_{air}$, T$_{water}$, |T$_{water}$-T$_{air}$| (previous ' + '%2i'%input_width + 'days)')

# # # Evaluate skill on train set...
# # predictions = []
# # for it in range(input_train.shape[0]):
# #     input_pred = np.expand_dims(input_train[it,:,:],0)
# #     predictions.append(model(torch.from_numpy(input_pred).float()).detach().numpy())

# # MAE=nn.L1Loss()
# # MSE=nn.MSELoss()
# # x_renorm = (np.squeeze(torch.from_numpy(target_train).float())*train_std[0]) + train_mean[0]
# # y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions).astype(float)).float())*train_std[0]) + train_mean[0]
# # print('LSTM MODEL, '+ str(pred_window)+ '-STEP AHEAD -----------')
# # print('TRAIN')
# # print(MAE(x_renorm,y_renorm))
# # print(np.sqrt(MSE(x_renorm,y_renorm)))
# # print(np.corrcoef(np.array(x_renorm).ravel(),np.array(y_renorm).ravel())[0,1])
# # plt.figure()
# # for s in np.arange(0,target_train.shape[0]-(input_width+shift),label_width):
# #     plot_series(time_train[input_width+(nslide*s):input_width+label_width+(nslide*s)],np.array(x_renorm)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
# #     plot_series(time_train[input_width+(nslide*s):input_width+label_width+(nslide*s)],np.array(y_renorm)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))
# # plt.title('Predictors: T$_{air}$, T$_{water}$, |T$_{water}$-T$_{air}$| (previous ' + '%2i'%input_width + 'days)')

