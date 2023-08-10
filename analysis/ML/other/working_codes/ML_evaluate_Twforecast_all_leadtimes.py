#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:23:02 2021

Tutorial from: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/structured_data/time_series.ipynb#scrollTo=YeCWbq6KLmL7
Alternative link : https://www.tensorflow.org/tutorials/structured_data/time_series

@author: Amelie
"""
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import copy
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import cmocean

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
from functions import rolling_climo,running_nanmean
from functions import find_freezeup_Tw,find_freezeup_Tw_all_yrs
from functions_ML import plot_series_1step,plot_series

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

#%%
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

# eval_period = 'test'
eval_period = 'valid'
# eval_period = 'train'

# metric = 'RMSE'
metric = 'MAE'
anomaly = False
freezeup_opt = 1

plot_samples = False
plot_Tw_diagnostics = True
plot_FUD_diagnostics = False

recalibrate = False
# offset_type = 'mean_valid'
offset_type = 'mean_clim'

model_name = 'MLP'
norm_type = 'min_max'
use_softplus = False
lblw = 75
inpw = 240
nlayers = 3
nneurons = 10
ne = 250
suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_mean'
# suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanAvg.levelOttawaRiver'
# suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanNAOAOPDO'
# suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanTot.FDDTot.snowfallAvg.SLP'
# suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanNAOAOPDOTot.FDDTot.snowfallAvg.SLP'
# suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanTot.FDDTot.snowfallAvg.SLP'
horizon_arr = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]

# model_name = 'LSTM'
# norm_type = 'min_max'
# use_softplus = False
# lblw = 75
# inpw = 60
# nlayers = 1
# hs = 8
# ne = 100
# suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_mean'
# # suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanAvg.levelOttawaRiver'
# # suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanNAONAOPNASOI'
# # suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanNAONAOPNASOIAvg.levelOttawaRiverTot.FDDTot.snowfallAvg.SLP'
# # suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanTot.FDDTot.snowfallAvg.SLP'
# horizon_arr = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]



#%%
# Load data
if use_softplus:
    if model_name == 'MLP':
        pred_data = np.load('./'+model_name+'_standardvalid/softplus/'+model_name+'_horizon'+str(lblw)+'_context'+str(inpw)+'_nlayers'+str(nlayers)+'_nneurons'+str(nneurons)+'_nepochs'+str(ne)+'_'+norm_type+'_'+suffix+'.npz',allow_pickle='TRUE')
    if model_name == 'LSTM':
        pred_data = np.load('./'+model_name+'_standardvalid/softplus/'+model_name+'_horizon'+str(lblw)+'_context'+str(inpw)+'_nlayers'+str(nlayers)+'_hiddensize'+str(hs)+'_nepochs'+str(ne)+'_'+norm_type+'_'+suffix+'.npz',allow_pickle='TRUE')
else:
    if model_name == 'MLP':
        pred_data = np.load('./'+model_name+'_standardvalid/'+model_name+'_horizon'+str(lblw)+'_context'+str(inpw)+'_nlayers'+str(nlayers)+'_nneurons'+str(nneurons)+'_nepochs'+str(ne)+'_'+norm_type+'_'+suffix+'.npz',allow_pickle='TRUE')
    if model_name == 'LSTM':
        pred_data = np.load('./'+model_name+'_standardvalid/'+model_name+'_horizon'+str(lblw)+'_context'+str(inpw)+'_nlayers'+str(nlayers)+'_hiddensize'+str(hs)+'_nepochs'+str(ne)+'_'+norm_type+'_'+suffix+'.npz',allow_pickle='TRUE')

train_dataset = np.squeeze(pred_data['train_dataset'])
valid_dataset = np.squeeze(pred_data['valid_dataset'])
test_dataset = np.squeeze(pred_data['test_dataset'])
predictors = np.squeeze(pred_data['predictors'])
target_train = np.squeeze(pred_data['target_train'])
target_valid = np.squeeze(pred_data['target_valid'])
target_test = np.squeeze(pred_data['target_test'])
input_train = np.squeeze(pred_data['input_train'])
input_valid = np.squeeze(pred_data['input_valid'])
input_test = np.squeeze(pred_data['input_test'])
predictions_train = np.squeeze(pred_data['predictions_train'])
predictions_valid = np.squeeze(pred_data['predictions_valid'])
predictions_test = np.squeeze(pred_data['predictions_test'])
clim_target_train = pred_data['clim_target_train']
clim_target_valid = pred_data['clim_target_valid']
clim_target_test  = pred_data['clim_target_test']
# Tw_climatology_mean_train = pred_data['Tw_climatology_mean_train']
# Tw_climatology_mean_valid = pred_data['Tw_climatology_mean_valid']
# Tw_climatology_mean_test = pred_data['Tw_climatology_mean_test']
time_train = pred_data['time_train']
time_valid = pred_data['time_valid']
time_test = pred_data['time_test']
time_target_train = pred_data['time_target_train']
time_target_valid = pred_data['time_target_valid']
time_target_test = pred_data['time_target_test']
time_input_train = pred_data['time_input_train']
time_input_valid = pred_data['time_input_valid']
time_input_test = pred_data['time_input_test']
train_range_pred = pred_data['train_range_pred']
train_offset_pred = pred_data['train_offset_pred']
train_range_tar = pred_data['train_range_tar']
train_offset_tar = pred_data['train_offset_tar']
input_width = int(pred_data['input_width'])
label_width = int(pred_data['label_width'])
shift = int(pred_data['shift'])
nslide = int(pred_data['nslide'])
date_ref = pred_data['date_ref']
nepochs = pred_data['nepochs']
train_losses = pred_data['train_losses']
valid_losses = pred_data['valid_losses']
normalize_predictors =pred_data['normalize_predictors']
normalize_target = pred_data['normalize_target']
norm_type = pred_data['norm_type']
ynoneg = pred_data['ynoneg']
ytransform = pred_data['ytransform']
valid_scheme = pred_data['valid_scheme']
train_yr_start = pred_data['train_yr_start']
valid_yr_start = pred_data['valid_yr_start']
test_yr_start = pred_data['test_yr_start']
yr_input_train = pred_data['yr_input_train']
yr_input_valid = pred_data['yr_input_valid']
yr_input_test = pred_data['yr_input_test']
# seed = pred_data['seed']
# nb_neurons = pred_data['nb_neurons']
# nb_layers = pred_data['nb_layers']
# learning_rate = pred_data['learning_rate']
# momentum = pred_data['momentum']
# optimizer_name = pred_data['optimizer_name']
# loss_name = pred_data['loss_name']
# model=pred_data['model']

plot_losses(train_losses,valid_losses,new_fig=True)

def reconstruct_ts(targets,predictions,yr_input,time_in,Tw_climatology_mean,t_range,t_offset,plot_pred_ts = True):
    global input_width, shift, label_width, nslide
    global normalize_target, normalize_predictors, ytransform
    global train_range_pred, train_offset_pred, predictors

    if normalize_target:
        targets = (np.squeeze(torch.from_numpy(targets).float())*t_range) + t_offset
        predictions = (np.squeeze(torch.from_numpy(np.array(predictions).astype(float)).float())*t_range) + t_offset

    else:
        targets = (np.squeeze(torch.from_numpy(targets).float()))
        predictions = (np.squeeze(torch.from_numpy(np.array(predictions).astype(float)).float()))


    if ytransform == 'diff':
        if normalize_predictors:
            # the yr time series was also normalized so we have
            # to transform it back before we can use it as the
            # base values for the diff time series
            yr = (np.squeeze(torch.from_numpy(yr_input).float())*train_range_pred[np.where(predictors == 'Avg. Twater')[0][0]]) + train_offset_pred[np.where(predictors == 'Avg. Twater')[0][0]]
        else:
            yr = (np.squeeze(torch.from_numpy(yr_input).float()))

        targets_recons = np.zeros(targets.shape)
        predictions_recons = np.zeros(predictions.shape)
        for s in range(targets.shape[0]):
            for it in range(label_width):
                if it == 0:
                    targets_recons[s,it] = yr[s,-1] + targets[s,it]
                    predictions_recons[s,it] = yr[s,-1] + predictions[s,it]
                else:
                    targets_recons[s,it] = targets_recons[s,it-1] + targets[s,it]
                    predictions_recons[s,it] = predictions_recons[s,it-1] + predictions[s,it]

        targets_recons = torch.from_numpy((targets_recons)).float()
        predictions_recons = torch.from_numpy((predictions_recons)).float()

    elif ytransform == 'None':
        targets_recons = targets
        predictions_recons = predictions

    clim_recons = (np.squeeze(torch.from_numpy((Tw_climatology_mean)).float()))

    # Plot predictions:
    if plot_pred_ts:
        plt.figure()
        if label_width == 1:
            plot_series_1step(time_in,np.squeeze(clim_recons),'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(0))
            plot_series_1step(time_in,np.array(targets_recons),'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
            plot_series_1step(time_in,np.array(predictions_recons),'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))
        else:
            plot_series_1step(time_in[:,0],np.squeeze(clim_recons[:,0]),'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(0))
            for s in np.arange(0,targets_recons.shape[0]-(input_width+shift),label_width):
                plot_series(time_in[s],np.array(targets_recons)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
                plot_series(time_in[s],np.array(predictions_recons)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))

    return predictions_recons, targets_recons, clim_recons



if eval_period == 'train':
    # TRAIN PREDICTIONS
    pred_recons_train, target_recons_train, clim_recons_train = reconstruct_ts(target_train,predictions_train,yr_input_train,time_target_train,clim_target_train,train_range_tar,train_offset_tar,plot_pred_ts = True)
    Tw_target_0lead_train = np.array(target_recons_train[:,0])
    time_all_train  = np.array(time_target_train[:,0])

    pred_arr =  np.array(pred_recons_train)
    target_arr = np.array(target_recons_train)
    clim_recons = np.array(clim_recons_train)
    time_arr = time_target_train
    Twater = np.array(target_recons_train[:,0])
    time_all = np.array(time_target_train[:,0])
    years = np.arange(train_yr_start,valid_yr_start)

if eval_period == 'valid':
    # VALIDATION PREDICTIONS
    pred_recons_valid, target_recons_valid, clim_recons_valid = reconstruct_ts(target_valid,predictions_valid,yr_input_valid,time_target_valid,clim_target_valid,train_range_tar,train_offset_tar,plot_pred_ts = True)
    Tw_target_0lead_valid = np.array(target_recons_valid[:,0])
    time_all_valid  = np.array(time_target_valid[:,0])

    pred_arr =  np.array(pred_recons_valid)
    target_arr = np.array(target_recons_valid)
    clim_recons = np.array(clim_recons_valid)
    time_arr = time_target_valid
    Twater = np.array(target_recons_valid[:,0])
    time_all = np.array(time_target_valid[:,0])
    # Twater = (valid_dataset[:,np.where(predictors == 'Avg. Twater')[0][0]]*train_range_pred[np.where(predictors == 'Avg. Twater')[0][0]]) + train_offset_pred[np.where(predictors == 'Avg. Twater')[0][0]]
    # time_all = time_valid
    years = np.arange(valid_yr_start,test_yr_start)

if eval_period == 'test':
    # TEST PREDICTIONS
    pred_recons_test, target_recons_test, clim_recons_test = reconstruct_ts(target_test,predictions_test,yr_input_test,time_target_test,clim_target_test,train_range_tar,train_offset_tar,plot_pred_ts = True)
    Tw_target_0lead_test = np.array(target_recons_test[:,0])
    time_all_test  = np.array(time_target_test[:,0])

    pred_arr =  np.array(pred_recons_test)
    target_arr = np.array(target_recons_test)
    clim_recons = np.array(clim_recons_test)
    time_arr = time_target_test
    Twater = np.array(target_recons_test[:,0])
    time_all = np.array(time_target_test[:,0])
    years = np.arange(test_yr_start,2020)

#%%
plt.figure()
plot_series_1step(time_all_valid,np.squeeze(clim_recons[:,0]),'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(0))
plot_series_1step(np.arange(time_all_valid[-1],time_all_valid[-1]+75),np.squeeze(clim_recons[-1,:]),'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(0))
plot_series(time_all_valid,np.array(target_recons_valid)[:,0],'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')

lead=0
plot_series(time_all_valid+lead,np.array(pred_recons_valid)[:,lead],'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))
# lead=40
# plot_series(time_all_valid+lead,np.array(pred_recons_valid)[:,lead],'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))

#%%
plot_Tw_metric = np.zeros((len(horizon_arr),12))
plot_Tw_clim_metric = np.zeros((len(horizon_arr),12))
plot_fu_metric = np.zeros((len(horizon_arr),6))
plot_fu_clim_metric = np.zeros((len(horizon_arr),6))
plot_fu_N = np.zeros((len(horizon_arr),6))
plot_fu_clim_N = np.zeros((len(horizon_arr),6))

for ih,h in enumerate(horizon_arr):
    Tw_eval_arr = np.zeros((12,1))*np.nan
    Tw_clim_eval_arr = np.zeros((12,1))*np.nan
    fu_eval_arr = np.zeros((6,1))*np.nan
    fu_clim_eval_arr = np.zeros((6,1))*np.nan
    fu_N_arr = np.zeros((6,1))*np.nan
    fu_clim_N_arr = np.zeros((6,1))*np.nan

    # EVALUATE TWATER FORECAST ACCORDING TO SELECTED METRIC:
    for imonth,month in enumerate(np.arange(1,13)):
        samples_Tw = pred_arr[np.where(np.array([(date_ref+dt.timedelta(days=int(time_arr[s,0]))).month for s in range(time_arr.shape[0])]) == month )[0]][:,h-1]
        targets_Tw = target_arr[np.where(np.array([(date_ref+dt.timedelta(days=int(time_arr[s,0]))).month for s in range(time_arr.shape[0])]) == month )[0]][:,h-1]
        clim_Tw = clim_recons[np.where(np.array([(date_ref+dt.timedelta(days=int(time_arr[s,0]))).month for s in range(time_arr.shape[0])]) == month )[0]][:,h-1]

        if metric == 'RMSE':
            Tw_metric = 0
            Tw_clim_metric = 0
            for s in range(samples_Tw.shape[0]):
                Tw_metric += (samples_Tw[s]-targets_Tw[s])**2.
                Tw_clim_metric += (clim_Tw[s]-targets_Tw[s])**2.
            Tw_metric = np.sqrt(Tw_metric/samples_Tw.shape[0])
            Tw_clim_metric = np.sqrt(Tw_clim_metric/samples_Tw.shape[0])

        if metric == 'MAE':
            Tw_metric = 0
            Tw_clim_metric = 0
            for s in range(samples_Tw.shape[0]):
                Tw_metric += np.abs(samples_Tw[s]-targets_Tw[s])
                Tw_clim_metric += np.abs(clim_Tw[s]-targets_Tw[s])
            Tw_metric = Tw_metric/samples_Tw.shape[0]
            Tw_clim_metric = Tw_clim_metric/samples_Tw.shape[0]

        if metric == 'NSE':
            print('NEEDS TO BE IMPLEMENTED!!')

        Tw_eval_arr[imonth] = Tw_metric
        Tw_clim_eval_arr[imonth] = Tw_clim_metric


    plot_Tw_metric[ih,:] = np.squeeze(Tw_eval_arr)
    plot_Tw_clim_metric[ih,:] = np.squeeze(Tw_clim_eval_arr)

    # FIND FREEZE-UP ONLY FROM SAMPLES THAT STARTED
    # ON OCT 1st, OCT 15th,
    #    NOV 1st, NOV 15th,
    #    DEC 1st, DEC 15th,
    #  (REPEAT THIS ANALYSIS FOR ALL LEAD TIMES)

    # OPTION 1
    if freezeup_opt == 1:
        def_opt = 1
        smooth_T =False; N_smooth = 3; mean_type='centered'
        round_T = False; round_type= 'half_unit'
        Gauss_filter = False
        T_thresh = 0.75
        # T_thresh = 1.0
        dTdt_thresh = 0.25
        d2Tdt2_thresh = 0.25
        nd = 1
        no_negTw = False

    # OPTION 2
    if freezeup_opt == 2:
        def_opt = 3
        smooth_T =False; N_smooth = 3; mean_type='centered'
        round_T = False; round_type= 'half_unit'
        Gauss_filter = True
        sig_dog = 3.5
        T_thresh = 3.
        dTdt_thresh = 0.15
        d2Tdt2_thresh = 0.15
        # dTdt_thresh = 0.20 # using 0.20 here instead of 0.15 can achieve even lower difference between stations, but then some years do not match the charts timing anymore...
        # d2Tdt2_thresh = 0.20
        nd = 30
        no_negTw = False



    freezeup_dates_sample = np.zeros((fu_eval_arr.shape[0],samples_Tw.shape[0],4))*np.nan
    freezeup_dates_target = np.zeros((fu_eval_arr.shape[0],samples_Tw.shape[0],4))*np.nan
    freezeup_dates_clim_target = np.zeros((fu_eval_arr.shape[0],samples_Tw.shape[0],4))*np.nan

    for istart in range(0,fu_eval_arr.shape[0],2):
        if istart == 0:
            month = 10
        if istart == 2:
            month = 11
        if istart == 4:
            month = 12
        if istart == 6:
            month = 1

        month_array = np.array([(date_ref+dt.timedelta(days=int(time_arr[s,0]))).month for s in range(time_arr.shape[0])])
        samples_tmp = pred_arr[np.where(month_array == month )[0]][:,0:h]
        targets_tmp = target_arr[np.where(month_array == month )[0]][:,0:h]
        time_st_tmp = time_arr[np.where(month_array == month )[0]][:,0:h]
        clim_targets_tmp = clim_recons[np.where(month_array == month )[0]][:,0:h]

        # Find sample starting between the 1st and the 15th of the month
        # day_array = np.array([(date_ref+dt.timedelta(days=int(time_st_tmp[s,0]))).day for s in range(time_st_tmp.shape[0])])
        # samples_1 = samples_tmp[np.where((day_array  >= 1) & (day_array <=15))[0]]
        # targets_1 = targets_tmp[np.where((day_array  >= 1) & (day_array <=15))[0]]
        # time_1 = time_st_tmp[np.where((day_array  >= 1) & (day_array <=15))[0]]
        # clim_targets_1 = clim_targets_tmp[np.where((day_array  >= 1) & (day_array <=15))[0]]

        # Find sample starting on the first of the month:
        day_array = np.array([(date_ref+dt.timedelta(days=int(time_st_tmp[s,0]))).day for s in range(time_st_tmp.shape[0])])
        samples_1 = samples_tmp[np.where((day_array  == 1) )[0]]
        targets_1 = targets_tmp[np.where((day_array  == 1) )[0]]
        time_1 = time_st_tmp[np.where((day_array  == 1) )[0]]
        clim_targets_1 = clim_targets_tmp[np.where((day_array  == 1) )[0]]

        samples = samples_1
        targets = targets_1
        time_st = time_1
        clim_targets = clim_targets_1

        for s in range(samples.shape[0]):

            time = time_st[s,:]

            # FIND DTDt, D2Tdt2,etc. - SAMPLE
            Twater_sample = samples[s]
            Twater_dTdt_sample = np.zeros(Twater_sample.shape)*np.nan
            Twater_d2Tdt2_sample = np.zeros(Twater_sample.shape)*np.nan
            Twater_DoG1_sample = np.zeros(Twater_sample.shape)*np.nan
            Twater_DoG2_sample = np.zeros(Twater_sample.shape)*np.nan

            Twater_tmp = Twater_sample.copy()
            if round_T:
                if round_type == 'unit':
                    Twater_tmp = np.round(Twater_tmp.copy())
                if round_type == 'half_unit':
                    Twater_tmp = np.round(Twater_tmp.copy()* 2) / 2.
            if smooth_T:
                Twater_tmp = running_nanmean(Twater_tmp.copy(),N_smooth,mean_type=mean_type)

            if no_negTw:
                Twater_tmp[Twater_tmp < 0] = 0.0

            dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

            dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
            dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
            dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

            Twater_dTdt_sample= np.nanmean(dTdt_tmp[:,0:2],axis=1)
            Twater_d2Tdt2_sample = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

            if Gauss_filter:
                Twater_dTdt_sample = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=1)
                Twater_d2Tdt2_sample = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=2)

            # FIND DTDt, D2Tdt2,etc. - TARGET
            Twater_target = targets[s]
            Twater_dTdt_target = np.zeros(Twater_target.shape)*np.nan
            Twater_d2Tdt2_target = np.zeros(Twater_target.shape)*np.nan
            Twater_DoG1_target = np.zeros(Twater_target.shape)*np.nan
            Twater_DoG2_target = np.zeros(Twater_target.shape)*np.nan

            Twater_tmp_target = Twater_target.copy()
            if round_T:
                if round_type == 'unit':
                    Twater_tmp_target = np.round(Twater_tmp_target.copy())
                if round_type == 'half_unit':
                    Twater_tmp_target = np.round(Twater_tmp_target.copy()* 2) / 2.
            if smooth_T:
                Twater_tmp_target = running_nanmean(Twater_tmp_target.copy(),N_smooth,mean_type=mean_type)

            dTdt_tmp = np.zeros((Twater_tmp_target.shape[0],3))*np.nan

            dTdt_tmp[0:-1,0]= Twater_tmp_target[1:]- Twater_tmp_target[0:-1] # Forwards
            dTdt_tmp[1:,1] = Twater_tmp_target[1:] - Twater_tmp_target[0:-1] # Backwards
            dTdt_tmp[0:-1,2]= Twater_tmp_target[0:-1]-Twater_tmp_target[1:]  # -1*Forwards

            Twater_dTdt_target= np.nanmean(dTdt_tmp[:,0:2],axis=1)
            Twater_d2Tdt2_target = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

            if Gauss_filter:
                Twater_dTdt_target = scipy.ndimage.gaussian_filter1d(Twater_tmp_target.copy(),sigma=sig_dog,order=1)
                Twater_d2Tdt2_target = scipy.ndimage.gaussian_filter1d(Twater_tmp_target.copy(),sigma=sig_dog,order=2)

            # FIND DTDt, D2Tdt2,etc. - CLIM TARGET
            Twater_clim_target = clim_targets[s]
            Twater_dTdt_clim_target = np.zeros(Twater_clim_target.shape)*np.nan
            Twater_d2Tdt2_clim_target = np.zeros(Twater_clim_target.shape)*np.nan
            Twater_DoG1_clim_target = np.zeros(Twater_clim_target.shape)*np.nan
            Twater_DoG2_clim_target = np.zeros(Twater_clim_target.shape)*np.nan

            Twater_tmp_clim_target = Twater_clim_target.copy()
            if round_T:
                if round_type == 'unit':
                    Twater_tmp_clim_target = np.round(Twater_tmp_clim_target.copy())
                if round_type == 'half_unit':
                    Twater_tmp_clim_target = np.round(Twater_tmp_clim_target.copy()* 2) / 2.
            if smooth_T:
                Twater_tmp_clim_target = running_nanmean(Twater_tmp_clim_target.copy(),N_smooth,mean_type=mean_type)

            dTdt_tmp = np.zeros((Twater_tmp_clim_target.shape[0],3))*np.nan

            dTdt_tmp[0:-1,0]= Twater_tmp_clim_target[1:]- Twater_tmp_clim_target[0:-1] # Forwards
            dTdt_tmp[1:,1] = Twater_tmp_clim_target[1:] - Twater_tmp_clim_target[0:-1] # Backwards
            dTdt_tmp[0:-1,2]= Twater_tmp_clim_target[0:-1]-Twater_tmp_clim_target[1:]  # -1*Forwards

            Twater_dTdt_clim_target= np.nanmean(dTdt_tmp[:,0:2],axis=1)
            Twater_d2Tdt2_clim_target = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

            if Gauss_filter:
                Twater_dTdt_clim_target = scipy.ndimage.gaussian_filter1d(Twater_tmp_clim_target.copy(),sigma=sig_dog,order=1)
                Twater_d2Tdt2_clim_target = scipy.ndimage.gaussian_filter1d(Twater_tmp_clim_target.copy(),sigma=sig_dog,order=2)


            if (h == 75) & (istart == 4):
                if plot_samples:
                    plt.figure()
                    plt.plot(Twater_target, color='black')
                    plt.plot(Twater_clim_target)
                    plt.plot(Twater_sample)
                    # plt.figure()
                    # plt.plot(Twater_dTdt_clim_target,'-', color='blue')
                    # plt.plot(Twater_d2Tdt2_clim_target,':', color='blue')
                    # plt.plot(Twater_dTdt_target,'-', color='black')
                    # plt.plot(Twater_d2Tdt2_target,':', color='black')
                    # plt.plot(Twater_dTdt_sample,'-', color='orange')
                    # plt.plot(Twater_d2Tdt2_sample,':', color='orange')

            # FIND FREEZE-UP FOR BOTH SAMPLE AND TARGET
            # ifz = 0
            date_start = dt.timedelta(days=int(time[0])) + date_ref
            if date_start.month < 3:
                year = date_start.year-1
            else:
                year = date_start.year

            if year >= years[0]:
                iyr = np.where(years == year)[0][0]
                fd_sample, ftw_sample, T_freezeup_sample, mask_freeze_sample = find_freezeup_Tw(def_opt,Twater_tmp,Twater_dTdt_sample,Twater_d2Tdt2_sample,time,year,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
                fd_target, ftw_target, T_freezeup_target, mask_freeze_target = find_freezeup_Tw(def_opt,Twater_tmp_target,Twater_dTdt_target,Twater_d2Tdt2_target,time,year,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
                fd_clim_target, ftw_clim_target, T_freezeup_clim_target, mask_freeze_clim_target = find_freezeup_Tw(def_opt,Twater_tmp_clim_target,Twater_dTdt_clim_target,Twater_d2Tdt2_clim_target,time,year,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)


                if (np.sum(mask_freeze_sample) > 0):
                    freezeup_dates_sample[istart,s,0] = iyr
                    freezeup_dates_sample[istart,s,1:4] = fd_sample

                if (np.sum(mask_freeze_target) > 0):
                    freezeup_dates_target[istart,s,0] = iyr
                    freezeup_dates_target[istart,s,1:4] = fd_target

                if (np.sum(mask_freeze_clim_target) > 0):
                    freezeup_dates_clim_target[istart,s,0] = iyr
                    freezeup_dates_clim_target[istart,s,1:4] = fd_clim_target


        if istart < 6: # (We do not look at forecasts starting Jan 15th...)

            # # Find sample starting between the 15th and the end of the month
            # samples_15 = samples_tmp[np.where((day_array  > 15) & (day_array <=31))[0]]
            # targets_15 = targets_tmp[np.where((day_array  > 15) & (day_array <=31))[0]]
            # time_15 = time_st_tmp[np.where((day_array  > 15) & (day_array <=31))[0]]
            # clim_targets_15 = clim_targets_tmp[np.where((day_array  > 15) & (day_array <=31))[0]]

            # Find sample starting on the 15th of the month
            day_array = np.array([(date_ref+dt.timedelta(days=int(time_st_tmp[s,0]))).day for s in range(time_st_tmp.shape[0])])
            samples_15 = samples_tmp[np.where((day_array  ==15) )[0]]
            targets_15 = targets_tmp[np.where((day_array  ==15) )[0]]
            time_15 = time_st_tmp[np.where((day_array  == 15) )[0]]
            clim_targets_15 = clim_targets_tmp[np.where((day_array  == 15) )[0]]

            samples = samples_15
            targets = targets_15
            time_st = time_15
            clim_targets = clim_targets_15

            for s in range(samples.shape[0]):

                time = time_st[s,:]

                # FIND DTDt, D2Tdt2,etc. - SAMPLE
                Twater_sample = samples[s]
                Twater_dTdt_sample = np.zeros(Twater_sample.shape)*np.nan
                Twater_d2Tdt2_sample = np.zeros(Twater_sample.shape)*np.nan
                Twater_DoG1_sample = np.zeros(Twater_sample.shape)*np.nan
                Twater_DoG2_sample = np.zeros(Twater_sample.shape)*np.nan

                Twater_tmp = Twater_sample.copy()
                if round_T:
                    if round_type == 'unit':
                        Twater_tmp = np.round(Twater_tmp.copy())
                    if round_type == 'half_unit':
                        Twater_tmp = np.round(Twater_tmp.copy()* 2) / 2.
                if smooth_T:
                    Twater_tmp = running_nanmean(Twater_tmp.copy(),N_smooth,mean_type=mean_type)

                if no_negTw:
                    Twater_tmp[Twater_tmp < 0] = 0.0

                dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

                dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
                dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
                dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

                Twater_dTdt_sample= np.nanmean(dTdt_tmp[:,0:2],axis=1)
                Twater_d2Tdt2_sample = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

                if Gauss_filter:
                    Twater_dTdt_sample = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=1)
                    Twater_d2Tdt2_sample = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=2)

                # FIND DTDt, D2Tdt2,etc. - TARGET
                Twater_target = targets[s]
                Twater_dTdt_target = np.zeros(Twater_target.shape)*np.nan
                Twater_d2Tdt2_target = np.zeros(Twater_target.shape)*np.nan
                Twater_DoG1_target = np.zeros(Twater_target.shape)*np.nan
                Twater_DoG2_target = np.zeros(Twater_target.shape)*np.nan

                Twater_tmp_target = Twater_target.copy()
                if round_T:
                    if round_type == 'unit':
                        Twater_tmp_target = np.round(Twater_tmp_target.copy())
                    if round_type == 'half_unit':
                        Twater_tmp_target = np.round(Twater_tmp_target.copy()* 2) / 2.
                if smooth_T:
                    Twater_tmp_target = running_nanmean(Twater_tmp_target.copy(),N_smooth,mean_type=mean_type)

                dTdt_tmp = np.zeros((Twater_tmp_target.shape[0],3))*np.nan

                dTdt_tmp[0:-1,0]= Twater_tmp_target[1:]- Twater_tmp_target[0:-1] # Forwards
                dTdt_tmp[1:,1] = Twater_tmp_target[1:] - Twater_tmp_target[0:-1] # Backwards
                dTdt_tmp[0:-1,2]= Twater_tmp_target[0:-1]-Twater_tmp_target[1:]  # -1*Forwards

                Twater_dTdt_target= np.nanmean(dTdt_tmp[:,0:2],axis=1)
                Twater_d2Tdt2_target = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

                if Gauss_filter:
                    Twater_dTdt_target = scipy.ndimage.gaussian_filter1d(Twater_tmp_target.copy(),sigma=sig_dog,order=1)
                    Twater_d2Tdt2_target = scipy.ndimage.gaussian_filter1d(Twater_tmp_target.copy(),sigma=sig_dog,order=2)

                # FIND DTDt, D2Tdt2,etc. - CLIM TARGET
                Twater_clim_target = clim_targets[s]
                Twater_dTdt_clim_target = np.zeros(Twater_clim_target.shape)*np.nan
                Twater_d2Tdt2_clim_target = np.zeros(Twater_clim_target.shape)*np.nan
                Twater_DoG1_clim_target = np.zeros(Twater_clim_target.shape)*np.nan
                Twater_DoG2_clim_target = np.zeros(Twater_clim_target.shape)*np.nan

                Twater_tmp_clim_target = Twater_clim_target.copy()
                if round_T:
                    if round_type == 'unit':
                        Twater_tmp_clim_target = np.round(Twater_tmp_clim_target.copy())
                    if round_type == 'half_unit':
                        Twater_tmp_clim_target = np.round(Twater_tmp_clim_target.copy()* 2) / 2.
                if smooth_T:
                    Twater_tmp_clim_target = running_nanmean(Twater_tmp_clim_target.copy(),N_smooth,mean_type=mean_type)

                dTdt_tmp = np.zeros((Twater_tmp_clim_target.shape[0],3))*np.nan

                dTdt_tmp[0:-1,0]= Twater_tmp_clim_target[1:]- Twater_tmp_clim_target[0:-1] # Forwards
                dTdt_tmp[1:,1] = Twater_tmp_clim_target[1:] - Twater_tmp_clim_target[0:-1] # Backwards
                dTdt_tmp[0:-1,2]= Twater_tmp_clim_target[0:-1]-Twater_tmp_clim_target[1:]  # -1*Forwards

                Twater_dTdt_clim_target= np.nanmean(dTdt_tmp[:,0:2],axis=1)
                Twater_d2Tdt2_clim_target = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

                if Gauss_filter:
                    Twater_dTdt_clim_target = scipy.ndimage.gaussian_filter1d(Twater_tmp_clim_target.copy(),sigma=sig_dog,order=1)
                    Twater_d2Tdt2_clim_target = scipy.ndimage.gaussian_filter1d(Twater_tmp_clim_target.copy(),sigma=sig_dog,order=2)


                # FIND FREEZE-UP FOR BOTH SAMPLE AND TARGET
                # ifz = 0
                date_start = dt.timedelta(days=int(time[0])) + date_ref
                if date_start.month < 3:
                    year = date_start.year-1
                else:
                    year = date_start.year

                if year >= years[0]:
                    iyr = np.where(years == year)[0][0]
                    fd_sample, ftw_sample, T_freezeup_sample, mask_freeze_sample = find_freezeup_Tw(def_opt,Twater_tmp,Twater_dTdt_sample,Twater_d2Tdt2_sample,time,year,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
                    fd_target, ftw_target, T_freezeup_target, mask_freeze_target = find_freezeup_Tw(def_opt,Twater_tmp_target,Twater_dTdt_target,Twater_d2Tdt2_target,time,year,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
                    fd_clim_target, ftw_clim_target, T_freezeup_clim_target, mask_freeze_clim_target = find_freezeup_Tw(def_opt,Twater_tmp_clim_target,Twater_dTdt_clim_target,Twater_d2Tdt2_clim_target,time,year,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)


                    if (np.sum(mask_freeze_sample) > 0):
                        freezeup_dates_sample[istart+1,s,0] = iyr
                        freezeup_dates_sample[istart+1,s,1:4] = fd_sample

                    if (np.sum(mask_freeze_target) > 0):
                        freezeup_dates_target[istart+1,s,0] = iyr
                        freezeup_dates_target[istart+1,s,1:4] = fd_target

                    if (np.sum(mask_freeze_clim_target) > 0):
                        freezeup_dates_clim_target[istart+1,s,0] = iyr
                        freezeup_dates_clim_target[istart+1,s,1:4] = fd_clim_target



    # NOW MAKE ARRAY WITH DETECTED DATES WHEN USING ALL YEARS TIME SERIES
    # (INSTEAD OF THE SAME SAMPLES AS FOR THE MODEL). THIS MIGHT BE DIFFERENT THAN
    # THE DETECTED DATES FROM THE SAMPLE ONLY SINCE SAMPLES THAT START DURING THE
    # WINTER AFTER THE INITIAL FREEZE-UP MIGHT DETECT A NEW FREEZE-UP IF THE WATER
    # TEMPERATURE GOES ABOVE ZERO MOMENTARILY AND THEN DROPS BACK (THIS IS
    # ESPECIALLY POSSIBLE WHEN USING THE FIRST DEFINITION OF FREEZE-UP WITH THE
    # TWATER THRESHOLD, BUT MAYBE BE LESS OF A PROBLEM WHEN USING THE SECOND FREEZE-IP
    # DEFINITION USING THE DERIVATIVE OF GAUSSIAN FILTER).

    freezeup_dates_obs = np.zeros((len(years),3))*np.nan

    Twater_tmp = Twater.copy()
    Twater_dTdt_sample = np.zeros(Twater_tmp.shape)*np.nan
    Twater_d2Tdt2_sample = np.zeros(Twater_tmp.shape)*np.nan
    Twater_DoG1_sample = np.zeros(Twater_tmp.shape)*np.nan
    Twater_DoG2_sample = np.zeros(Twater_tmp.shape)*np.nan

    if round_T:
        if round_type == 'unit':
            Twater_tmp = np.round(Twater_tmp.copy())
        if round_type == 'half_unit':
            Twater_tmp = np.round(Twater_tmp.copy()* 2) / 2.

    if smooth_T:
        Twater_tmp = running_nanmean(Twater_tmp.copy(),N_smooth,mean_type=mean_type)

    if no_negTw:
        Twater_tmp[Twater_tmp < 0] = 0.0

    dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

    dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
    dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
    dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

    Twater_dTdt = np.nanmean(dTdt_tmp[:,0:2],axis=1)
    Twater_d2Tdt2 = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

    if Gauss_filter:
        Twater_dTdt = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=1)
        Twater_d2Tdt2 = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=2)

    fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw_all_yrs(def_opt,Twater_tmp,Twater_dTdt,Twater_d2Tdt2,time_all,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
    freezeup_dates_obs = fd


    # Compute climatological freeze-up date
    doy_arr = np.zeros((freezeup_dates_obs.shape[0]))*np.nan
    for ifd,fd in enumerate(freezeup_dates_obs):

        if ~np.isnan(fd[0]):
            fd_date = dt.date(int(fd[0]),int(fd[1]),int(fd[2]))

            if fd[1] > 11:
                yr_1 = dt.date(int(fd[0]),1,1)
                if calendar.isleap(int(fd[0])):
                    fd_doy = (fd_date-yr_1).days
                else:
                    fd_doy = (fd_date-yr_1).days+1

            else:
                yr_1 = dt.date(int(fd[0]),1,1)
                fd_doy = 365+(fd_date-yr_1).days+1

            doy_arr[ifd] = fd_doy


    clim_doy = np.nanmean(doy_arr)
    std_doy = np.nanstd(doy_arr)
    clim_dt = dt.date(2015,1,1)+dt.timedelta(days=clim_doy-1)

    clim_dates = np.zeros(freezeup_dates_obs.shape)*np.nan
    for ifd,fd in enumerate(freezeup_dates_obs):
        if ~np.isnan(fd[0]):
            clim_dates[ifd,0] = years[ifd]
            clim_dates[ifd,1] = clim_dt.month
            clim_dates[ifd,2] = clim_dt.day


    # EVALUATE FREEZE-UP FORECAST ACCORDING TO SELECTED METRIC:

    fu_metric = np.zeros((freezeup_dates_sample.shape[0],freezeup_dates_sample.shape[1]))*np.nan
    fu_clim_metric = np.zeros((freezeup_dates_sample.shape[0],freezeup_dates_sample.shape[1]))*np.nan

    # for istart in range(freezeup_dates_sample.shape[0]):

    #     sample_freezeup = freezeup_dates_sample[istart,:,:]
    #     target_freezeup = freezeup_dates_target[istart,:,:]
    #     clim_target_freezeup = freezeup_dates_clim_target[istart,:,:]

    #     # sample_freezeup = sample_freezeup_tmp[~np.isnan(sample_freezeup_tmp[:,0])]
    #     # target_freezeup = target_freezeup_tmp[~np.isnan(sample_freezeup_tmp[:,0])]
    #     n = np.nan
    #     n_clim = np.nan
    #     if np.sum(~np.isnan(sample_freezeup)) > 0:

    #         n = 0
    #         n_clim = 0
    #         for s in range(sample_freezeup.shape[0]):
    #             if ~np.isnan(sample_freezeup[s,0]):
    #                 iyr = int(sample_freezeup[s,0])
    #                 fs = dt.date(int(sample_freezeup[s,1]),int(sample_freezeup[s,2]),int(sample_freezeup[s,3]))
    #                 fo = dt.date(int(freezeup_dates_obs[iyr,0]),int(freezeup_dates_obs[iyr,1]),int(freezeup_dates_obs[iyr,2]))
    #                 # fc = dt.date(int(clim_dates[iyr,0]),int(clim_dates[iyr,1]),int(clim_dates[iyr,2]))
    #                 # fc = dt.date(int(clim_target_freezeup[s,1]),int(clim_target_freezeup[s,2]),int(clim_target_freezeup[s,3]))

    #                 if np.where( target_freezeup[:,0] == iyr )[0].shape[0] > 0:
    #                     s_t = np.where( target_freezeup[:,0] == iyr )[0][0]
    #                     ft = dt.date(int(target_freezeup[s_t,1]),int(target_freezeup[s_t,2]),int(target_freezeup[s_t,3]))
    #                 else:
    #                     ft = np.nan

    #                 # print(istart,fs,fo,ft)
    #                 if metric == 'RMSE':
    #                     fu_metric[istart,s] = ((fs-fo).days)**2.
    #                     # fu_clim_metric[istart,s] = ((fc-fo).days)**2.
    #                     n += 1

    #                 if metric == 'MAE':
    #                     fu_metric[istart,s] = np.abs((fs-fo).days)
    #                     # fu_clim_metric[istart,s] = np.abs((fc-fo).days)
    #                     n +=1

    #                 if metric == 'NSE':
    #                     print('NEEDS TO BE IMPLEMENTED!!')

    #             if ~np.isnan(clim_target_freezeup[s,0]):
    #                 iyr = int(clim_target_freezeup[s,0])
    #                 fo = dt.date(int(freezeup_dates_obs[iyr,0]),int(freezeup_dates_obs[iyr,1]),int(freezeup_dates_obs[iyr,2]))
    #                 fc = dt.date(int(clim_target_freezeup[s,1]),int(clim_target_freezeup[s,2]),int(clim_target_freezeup[s,3]))

    #                 # print(istart,fs,fo,ft)
    #                 if metric == 'RMSE':
    #                     fu_clim_metric[istart,s] = ((fc-fo).days)**2.
    #                     n_clim += 1

    #                 if metric == 'MAE':
    #                     fu_clim_metric[istart,s] = np.abs((fc-fo).days)
    #                     n_clim +=1

    #                 if metric == 'NSE':
    #                     print('NEEDS TO BE IMPLEMENTED!!')


    #     if metric == 'RMSE':
    #         fu_eval_arr[istart] = np.sqrt(np.nanmean(fu_metric[istart,:]))
    #         fu_clim_eval_arr[istart] = np.sqrt(np.nanmean(fu_clim_metric[istart,:]))
    #         fu_N_arr[istart] = n
    #         fu_clim_N_arr[istart] = n_clim

    #     if metric == 'MAE':
    #         fu_eval_arr[istart] = np.nanmean(fu_metric[istart,:])
    #         fu_clim_eval_arr[istart] = np.nanmean(fu_clim_metric[istart,:])
    #         fu_N_arr[istart] = n
    #         fu_clim_N_arr[istart] = n_clim

    #     if metric == 'NSE':
    #         print('NEEDS TO BE IMPLEMENTED!!')


    # plot_fu_metric[ih,:] = np.squeeze(fu_eval_arr)
    # plot_fu_clim_metric[ih,:] = np.squeeze(fu_clim_eval_arr)
    # plot_fu_N[ih,:] = np.squeeze(fu_N_arr)
    # plot_fu_clim_N[ih,:] = np.squeeze(fu_clim_N_arr)

    # # Save Tw and freeze-up metrics for the given prediction horizon
    # np.savez('./eval_metrics/'+model_name+'_FORECAST_EVAL_horizon_'+str(label_width)+'_input_'+str(input_width)+'_nepochs_'+str(nepochs)+suffix,
    #           fu_eval_arr = fu_eval_arr,
    #           Tw_eval_arr = Tw_eval_arr,
    #           fu_clim_eval_arr = fu_clim_eval_arr,
    #           Tw_clim_eval_arr = Tw_clim_eval_arr,
    #           freezeup_opt = freezeup_opt,
    #           metric = metric,
    #           anomaly = anomaly,
    #           label_width = label_width,
    #           input_width = input_width,
    #           nepochs = nepochs,
    #           )

#%%
# PLOT WATER TEMP ERROR METRIC
if plot_Tw_diagnostics:
    vmin = 0; vmax = 1.5; pivot = 1
    cmap = cmocean.cm.tempo
    cmap = cmocean.cm.dense
    cmap = cmocean.cm.deep
    cmap = cmocean.cm.thermal
    cmap = plt.get_cmap('cividis')
    cmap = plt.get_cmap('viridis')
    cmap = plt.get_cmap('magma')

    fig, axs = plt.subplots(1, 1, figsize=(6,4))
    mappable = axs.pcolormesh(np.flipud(plot_Tw_metric.T), cmap=cmap, vmin=vmin, vmax=vmax)
    axs.set_title('Water temp. MAE (deg. C) - ' + model_name)
    fig.colorbar(mappable, ax=[axs], location='left')

    fig_clim, axs_clim = plt.subplots(1, 1, figsize=(6,4))
    mappable = axs_clim.pcolormesh(np.flipud(plot_Tw_clim_metric.T), cmap=cmap, vmin=vmin, vmax=vmax)
    axs_clim.set_title('Water temp. MAE (deg. C) - Climatology')
    fig_clim.colorbar(mappable, ax=[axs_clim], location='left')

    vmin = -0.5; vmax = 0.5; pivot = 0
    cmap = cmocean.cm.balance
    fig_diff, axs_diff = plt.subplots(1, 1, figsize=(6,4))
    mappable = axs_diff.pcolormesh(np.flipud(plot_Tw_metric.T)-np.flipud(plot_Tw_clim_metric.T), cmap=cmap, vmin=vmin, vmax=vmax)
    axs_diff.set_title('$T_{w}$ mean absolute error diff. (deg. C)\n ' + model_name+' - Climatology')
    fig_diff.colorbar(mappable, ax=[axs_diff], location='left')


