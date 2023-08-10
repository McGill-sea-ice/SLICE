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

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

#%%

# metric = 'RMSE'
metric = 'MAE'
anomaly = False
freezeup_opt = 1


years = np.array([1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020])

model_name = 'MLP'
lblw = 75
inpw = 120
# ne = 271
ne = 300
# suffix = 'Ta'
suffix = '22_Tw_Ta'
# suffix = 'Tw_Ta_precip'
# suffix = 'Tw_Ta_NAO'
# suffix = 'Tw_Ta_level'
# suffix = 'Tw_Ta_snowfall'
# suffix = 'Ta_snowfall'
# suffix = '64_Tw_Ta_NAO_precip_snowfall_cloud_discharge_FDD_TDD'
# suffix = '32_Tw_Ta_NAO_precip_snowfall_cloud_discharge_FDD_TDD'
horizon_arr = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]
# horizon_arr = [4,8,16,32,64]

lblw = 75
inpw = 120
ne = 500
# suffix = '22_Tw_Ta'
# suffix = '22_Ta'
suffix = '22_Tw_Ta_NAO_precip_snowfall_cloud_discharge_FDD_TDD'
horizon_arr = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]

model_name = 'MLP'
lblw = 75
inpw = 120
ne = 250
suffix = '10_3_Tw_Ta'
# suffix = '10_3_Tw_Ta_snowfall'
# suffix = '10_3_Tw_Ta_NAO_snowfall_discharge_FDD'
horizon_arr = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]

# model_name = 'LSTM'
# lblw = 64
# inpw = 128
# ne = 300
# # suffix = '10_3_Tw_Ta'
# # suffix = '10_3_Tw_Ta_NAO_snowfall_discharge_FDD'
# horizon_arr = [5,10,15,20,25,30,35,40,45,50,55,60,64]
# suffix ='32_Tw_Ta_NAO_precip_cloud_FDD_TDD'

# lblw = 64
# inpw = 128
# ne = 300
# suffix = '32_Tw_Ta_NAO_precip_snowfall_cloud_discharge_FDD_TDD'
# horizon_arr = [5,10,15,20,25,30,35,40,45,50,55,60,64]




# model_name = 'LSTM'
# lblw = 75
# inpw = 96
# ne = 200
# suffix = ''
# suffix = 'hidden_16'
# horizon_arr = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]

plot_Tw_metric = np.zeros((len(horizon_arr),12))
plot_Tw_clim_metric = np.zeros((len(horizon_arr),12))
plot_fu_metric = np.zeros((len(horizon_arr),6))
plot_fu_clim_metric = np.zeros((len(horizon_arr),6))
plot_fu_N = np.zeros((len(horizon_arr),6))
plot_fu_clim_N = np.zeros((len(horizon_arr),6))



# Load data
# pred_data = np.load('./ML_pred_'+model_name+'_horizon_'+str(lblw)+'_input_'+str(inpw)+'_nepochs_'+str(ne)+suffix+'.npz',allow_pickle='TRUE')

pred_data = np.load('./ML_pred_'+model_name+'_horizon_'+str(lblw)+'_input_'+str(inpw)+'_nepochs_'+str(ne)+'_'+suffix+'.npz',allow_pickle='TRUE')

# pred_data = np.load('./ML_pred_MLP_horizon_32_input_128_nepochs_136.npz',allow_pickle='TRUE')
# model_name = 'MLP'
# lblw = 32
# inpw = 128
# ne = 136

# pred_data = np.load('./ML_pred_MLP_horizon_32_input_128_nepochs_128.npz',allow_pickle='TRUE')
# model_name = 'MLP'
# lblw = 32
# inpw = 128
# ne = 128

# pred_data = np.load('./ML_pred_MLP_horizon_32_input_96_nepochs_137.npz',allow_pickle='TRUE')
# model_name = 'MLP'
# lblw = 32
# inpw = 96
# ne = 137

train_dataset = np.squeeze(pred_data['train_dataset'])
valid_dataset = np.squeeze(pred_data['valid_dataset'])
test_dataset = np.squeeze(pred_data['test_dataset'])
target_train = np.squeeze(pred_data['target_train'])
target_valid = np.squeeze(pred_data['target_valid'])
# target_test = np.squeeze(pred_data['target_test'])
predictions_train = np.squeeze(pred_data['predictions_train'])
predictions_valid = np.squeeze(pred_data['predictions_valid'])
# predictions_test = np.squeeze(pred_data['predictions_test'])
Tw_climatology_mean_train = pred_data['Tw_climatology_mean_train']
Tw_climatology_mean_valid = pred_data['Tw_climatology_mean_valid']
Tw_climatology_mean_test = pred_data['Tw_climatology_mean_test']
clim_train = pred_data['clim_train']
clim_valid = pred_data['clim_valid']
# clim_test = pred_data['clim_test']
time_train = pred_data['time_train']
time_valid = pred_data['time_valid']
time_test = pred_data['time_test']
time_target_train = pred_data['time_target_train']
time_target_valid = pred_data['time_target_valid']
# time_target_test = pred_data['time_target_test']
train_std = float(pred_data['train_std'])
train_mean = float(pred_data['train_mean'])
input_width = int(pred_data['input_width'])
label_width = int(pred_data['label_width'])
shift = int(pred_data['shift'])
nslide = int(pred_data['nslide'])
date_ref = pred_data['date_ref']
nepochs = pred_data['nepochs']
train_losses = pred_data['train_losses']
valid_losses = pred_data['valid_losses']


for ih,h in enumerate(horizon_arr):

    # plot_losses(train_losses,valid_losses,new_fig=True)


    Tw_eval_arr = np.zeros((12,1))*np.nan
    Tw_clim_eval_arr = np.zeros((12,1))*np.nan
    fu_eval_arr = np.zeros((6,1))*np.nan
    fu_clim_eval_arr = np.zeros((6,1))*np.nan
    fu_N_arr = np.zeros((6,1))*np.nan
    fu_clim_N_arr = np.zeros((6,1))*np.nan

    pred_arr = predictions_train
    target_arr = target_train
    clim_arr = clim_train
    time_arr = time_target_train
    Twater = train_dataset[:,0]
    Twater_clim = Tw_climatology_mean_train
    time_all = time_train

    pred_arr = predictions_valid
    target_arr = target_valid
    clim_arr = clim_valid
    time_arr = time_target_valid
    Twater = valid_dataset[:,0]
    Twater_clim = Tw_climatology_mean_valid
    time_all = time_valid


    # Reconstruct the Tw series from the anomaly and climatology
    if anomaly:
        clim_recons = (np.squeeze(clim_arr)*train_std) + train_mean
        target_arr = ((target_arr*train_std) + train_mean ) + clim_recons
        pred_arr = ((pred_arr*train_std) + train_mean ) + clim_recons
        Twater_clim = (Twater_clim*train_std) + train_mean
        Twater =  ((Twater*train_std) + train_mean ) + Twater_clim
    else:
        clim_recons = (np.squeeze(clim_arr)*train_std) + train_mean
        target_arr = ((target_arr*train_std) + train_mean )
        pred_arr = ((pred_arr*train_std) + train_mean )
        Twater =  ((Twater*train_std) + train_mean )


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
    #    JAN 1st
    #  (REPEAT THIS ANALYSIS FOR LEAD TIMES OF 60, 75, 90 DAYS)

    # OPTION 1
    if freezeup_opt == 1:
        def_opt = 1
        smooth_T =False; N_smooth = 3; mean_type='centered'
        round_T = False; round_type= 'half_unit'
        Gauss_filter = False
        T_thresh = 0.75
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
                plt.figure()
                plt.plot(Twater_target, color='black')
                plt.plot(Twater_clim_target)
                plt.plot(Twater_sample)

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

    for istart in range(freezeup_dates_sample.shape[0]):

        sample_freezeup = freezeup_dates_sample[istart,:,:]
        target_freezeup = freezeup_dates_target[istart,:,:]
        clim_target_freezeup = freezeup_dates_clim_target[istart,:,:]

        # sample_freezeup = sample_freezeup_tmp[~np.isnan(sample_freezeup_tmp[:,0])]
        # target_freezeup = target_freezeup_tmp[~np.isnan(sample_freezeup_tmp[:,0])]
        n = np.nan
        n_clim = np.nan
        if np.sum(~np.isnan(sample_freezeup)) > 0:

            n = 0
            n_clim = 0
            for s in range(sample_freezeup.shape[0]):
                if ~np.isnan(sample_freezeup[s,0]):
                    iyr = int(sample_freezeup[s,0])
                    fs = dt.date(int(sample_freezeup[s,1]),int(sample_freezeup[s,2]),int(sample_freezeup[s,3]))
                    fo = dt.date(int(freezeup_dates_obs[iyr,0]),int(freezeup_dates_obs[iyr,1]),int(freezeup_dates_obs[iyr,2]))
                    # fc = dt.date(int(clim_dates[iyr,0]),int(clim_dates[iyr,1]),int(clim_dates[iyr,2]))
                    # fc = dt.date(int(clim_target_freezeup[s,1]),int(clim_target_freezeup[s,2]),int(clim_target_freezeup[s,3]))

                    if np.where( target_freezeup[:,0] == iyr )[0].shape[0] > 0:
                        s_t = np.where( target_freezeup[:,0] == iyr )[0][0]
                        ft = dt.date(int(target_freezeup[s_t,1]),int(target_freezeup[s_t,2]),int(target_freezeup[s_t,3]))
                    else:
                        ft = np.nan

                    # print(istart,fs,fo,ft)
                    if metric == 'RMSE':
                        fu_metric[istart,s] = ((fs-fo).days)**2.
                        # fu_clim_metric[istart,s] = ((fc-fo).days)**2.
                        n += 1

                    if metric == 'MAE':
                        fu_metric[istart,s] = np.abs((fs-fo).days)
                        # fu_clim_metric[istart,s] = np.abs((fc-fo).days)
                        n +=1

                    if metric == 'NSE':
                        print('NEEDS TO BE IMPLEMENTED!!')

                if ~np.isnan(clim_target_freezeup[s,0]):
                    iyr = int(clim_target_freezeup[s,0])
                    fo = dt.date(int(freezeup_dates_obs[iyr,0]),int(freezeup_dates_obs[iyr,1]),int(freezeup_dates_obs[iyr,2]))
                    fc = dt.date(int(clim_target_freezeup[s,1]),int(clim_target_freezeup[s,2]),int(clim_target_freezeup[s,3]))

                    # print(istart,fs,fo,ft)
                    if metric == 'RMSE':
                        fu_clim_metric[istart,s] = ((fc-fo).days)**2.
                        n_clim += 1

                    if metric == 'MAE':
                        fu_clim_metric[istart,s] = np.abs((fc-fo).days)
                        n_clim +=1

                    if metric == 'NSE':
                        print('NEEDS TO BE IMPLEMENTED!!')


        if metric == 'RMSE':
            fu_eval_arr[istart] = np.sqrt(np.nanmean(fu_metric[istart,:]))
            fu_clim_eval_arr[istart] = np.sqrt(np.nanmean(fu_clim_metric[istart,:]))
            fu_N_arr[istart] = n
            fu_clim_N_arr[istart] = n_clim

        if metric == 'MAE':
            fu_eval_arr[istart] = np.nanmean(fu_metric[istart,:])
            fu_clim_eval_arr[istart] = np.nanmean(fu_clim_metric[istart,:])
            fu_N_arr[istart] = n
            fu_clim_N_arr[istart] = n_clim

        if metric == 'NSE':
            print('NEEDS TO BE IMPLEMENTED!!')


    plot_fu_metric[ih,:] = np.squeeze(fu_eval_arr)
    plot_fu_clim_metric[ih,:] = np.squeeze(fu_clim_eval_arr)
    plot_fu_N[ih,:] = np.squeeze(fu_N_arr)
    plot_fu_clim_N[ih,:] = np.squeeze(fu_clim_N_arr)

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
# PLOT FUD ERRROR METRIC
vmin = 0; vmax = 21; pivot = 10
cmap = cmocean.cm.curl
crop_cmap = cmocean.tools.crop(cmap, vmin, vmax, pivot)

fig, axs = plt.subplots(1, 1, figsize=(6,4))
mappable = axs.pcolormesh(np.flipud(plot_fu_metric.T), cmap=crop_cmap, vmin=vmin, vmax=vmax)
axs.set_title('Freeze-up date MAE (days) - ' + model_name)
fig.colorbar(mappable, ax=axs)

fig_clim, axs_clim = plt.subplots(1, 1, figsize=(6,4))
mappable = axs_clim.pcolormesh(np.flipud(plot_fu_clim_metric.T), cmap=crop_cmap, vmin=vmin, vmax=vmax)
axs_clim.set_title('Freeze-up date MAE (days) - Climatology')
fig_clim.colorbar(mappable, ax=axs_clim)

vmin = -10; vmax = 10; pivot = 0
cmap = cmocean.cm.balance
fig_diff, axs_diff = plt.subplots(1, 1, figsize=(6,4))
mappable = axs_diff.pcolormesh(np.flipud(plot_fu_metric.T)-np.flipud(plot_fu_clim_metric.T), cmap=cmap, vmin=vmin, vmax=vmax)
axs_diff.set_title('FUD mean absolute error diff. (days)\n ' + model_name+' - Climatology')
fig_diff.colorbar(mappable, ax=axs_diff)

# vmin = -10; vmax = 10; pivot = 0
# cmap = cmocean.cm.balance
# fig_diff, axs_diff = plt.subplots(1, 1, figsize=(6,4))
# mappable = axs_diff.pcolormesh(np.flipud(plot_fu_metric.T)-10.0, cmap=cmap, vmin=vmin, vmax=vmax)
# axs_diff.set_title('FUD mean absolute error diff. (days)\n ' + model_name+' - Climatology')
# fig_diff.colorbar(mappable, ax=axs_diff)
#%%

# PLOT WATER TEMP ERROR METRIC
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

#%%
fig_N, axs_N = plt.subplots(1, 1, figsize=(6,4))
cmap = plt.get_cmap('viridis')
cmap = plt.get_cmap('cividis')
cmap = plt.get_cmap('pink')
mappable = axs_N.pcolormesh(np.flipud(plot_fu_N.T), cmap=cmap, vmin=0, vmax=145)
axs_N.set_title('Number of forecast samples')
fig_N.colorbar(mappable, ax=axs_N)

fig_N_clim, axs_N_clim = plt.subplots(1, 1, figsize=(6,4))
cmap = plt.get_cmap('viridis')
cmap = plt.get_cmap('cividis')
cmap = plt.get_cmap('pink')
mappable = axs_N_clim.pcolormesh(np.flipud(plot_fu_clim_N.T), cmap=cmap, vmin=0, vmax=145)
axs_N_clim.set_title('Number of forecast samples')
fig_N_clim.colorbar(mappable, ax=axs_N_clim)

#%%
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

MAE=nn.L1Loss()
MSE=nn.MSELoss()
x_renorm = (np.squeeze(torch.from_numpy(target_valid).float())*train_std) + train_mean
y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions_valid).astype(float)).float())*train_std) + train_mean
print('MLP MODEL, '+ str(label_width)+ '-STEP AHEAD -----------')
print('VALID')
print(MAE(x_renorm,y_renorm))
print(np.sqrt(MSE(x_renorm,y_renorm)))
print(np.corrcoef(np.array(x_renorm).ravel(),np.array(y_renorm).ravel())[0,1])
plt.figure()
if anomaly:
    clim_recons = (np.squeeze(torch.from_numpy(clim_valid).float())*train_std) + train_mean
    x_renorm = (np.squeeze(torch.from_numpy(target_valid).float())*train_std) + train_mean + clim_recons
    y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions_valid).astype(float)).float())*train_std) + train_mean + clim_recons
else:
    x_renorm = (np.squeeze(torch.from_numpy(target_valid).float())*train_std) + train_mean
    y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions_valid).astype(float)).float())*train_std) + train_mean

# First add climatology
x_renorm_clim = (np.squeeze(torch.from_numpy(target_valid).float())*train_std) + train_mean
y_renorm_clim = torch.from_numpy((Tw_climatology_mean_valid[input_width:]*train_std) + train_mean).float()
plot_series_1step(time_valid[input_width:],np.array(y_renorm_clim).ravel(),'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(0))

for s in np.arange(0+30,target_valid.shape[0],label_width):
    plot_series(time_target_valid[s,:],np.array(x_renorm)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
    plot_series(time_target_valid[s,:],np.array(y_renorm)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))

plt.title('Predictors: T$_{air}$, T$_{water}$, |T$_{water}$-T$_{air}$| (previous ' + '%2i'%input_width + 'days)')



# %%
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


MAE=nn.L1Loss()
MSE=nn.MSELoss()
x_renorm = (np.squeeze(torch.from_numpy(target_train).float())*train_std) + train_mean
y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions_train).astype(float)).float())*train_std) + train_mean
print('MLP MODEL, '+ str(label_width)+ '-STEP AHEAD -----------')
print('TRAIN')
print(MAE(x_renorm,y_renorm))
print(np.sqrt(MSE(x_renorm,y_renorm)))
print(np.corrcoef(np.array(x_renorm).ravel(),np.array(y_renorm).ravel())[0,1])


plt.figure()
if anomaly:
    clim_recons = (np.squeeze(torch.from_numpy(clim_train).float())*train_std) + train_mean
    x_renorm = (np.squeeze(torch.from_numpy(target_train).float())*train_std) + train_mean + clim_recons
    y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions_train).astype(float)).float())*train_std) + train_mean + clim_recons
else:
    x_renorm = (np.squeeze(torch.from_numpy(target_train).float())*train_std) + train_mean
    y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions_train).astype(float)).float())*train_std) + train_mean

# First add climatology
x_renorm_clim = (np.squeeze(torch.from_numpy(target_train).float())*train_std) + train_mean
y_renorm_clim = torch.from_numpy((Tw_climatology_mean_train[input_width:]*train_std) + train_mean).float()
plot_series_1step(time_train[input_width:],np.array(y_renorm_clim).ravel(),'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(0))


for s in np.arange(0,target_train.shape[0],label_width):
    plot_series(time_target_train[s,:],np.array(x_renorm)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
    # plot_series(time_target_train[s,:],np.array(y_renorm)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))
plt.title('Predictors: T$_{air}$, T$_{water}$, |T$_{water}$-T$_{air}$| (previous ' + '%2i'%input_width + 'days)')


#%%
istart_label = ['Nov 1st', 'Nov 15th', 'Dec. 1st', 'Dec.15th']
plt.figure()
years_plot = np.arange(1992,2020)
avg_freezeup_doy = np.array([ 361., 358., 364., 342., 366., 350. ,365., 360.,
                     344., 367., 339., 348., 355., 350., 381., 345.,
                     347., 352., 352., 364., 361., 348., 366., 375.,
                     352., 349., 360., np.nan])

np.nanmean(avg_freezeup_doy[0:18])
MLR_pred = np.array([354.98402352,
       363.54500541,
       349.8139212 ,
       349.91221944,
       352.48026772,
       363.77961087,
       359.96231006,
       349.02877679,
       341.08672315])

# plt.plot(years[19:28],doy_arr[19:28],'o-',color='black')
plt.plot(years_plot,np.ones(len(years_plot))*(365),color=plt.get_cmap('tab20c')(2))
plt.plot(years_plot,np.ones(len(years_plot))*(np.nanmean(avg_freezeup_doy)),color=[0.7,0.7,0.7])
plt.plot(years_plot,avg_freezeup_doy ,'o-',color='black')
plt.plot(years[19:28],MLR_pred ,'o:',color=plt.get_cmap('tab20')(4),label='Lin.Reg. - $T_{a}$ Nov.')
ic = 0
for istart in [3,4,5]:
    fd_ML_forcast = np.zeros((9))*np.nan
    iarr = 0
    # istart = 2 # Dec. 1st
    for iyr in range(19,28):
        # print(iyr)
        select_yr_fud = freezeup_dates_sample[istart,np.where(freezeup_dates_sample[istart,:,0] == iyr)[0]]

        fd_avg = 0
        if select_yr_fud.shape[0] > 0:
            for j in range(select_yr_fud.shape[0]):

                fd_date = dt.date(int(select_yr_fud[j,1]),int(select_yr_fud[j,2]),int(select_yr_fud[j,3]))

                if select_yr_fud[j,2] > 11:
                    yr_1 = dt.date(int(select_yr_fud[j,1]),1,1)
                    if calendar.isleap(int(select_yr_fud[j,1])):
                        fd_doy = (fd_date-yr_1).days
                    else:
                        fd_doy = (fd_date-yr_1).days+1

                else:
                    yr_1 = dt.date(int(select_yr_fud[j,1]),1,1)
                    fd_doy = 365+(fd_date-yr_1).days+1

                fd_avg += fd_doy

            fd_ML_forcast[iarr]= fd_avg/select_yr_fud.shape[0]
        else:
            fd_ML_forcast[iarr]= np.nan

        iarr +=1



    # plt.figure()
    plt.plot(years[19:28],fd_ML_forcast,'o:', color=plt.get_cmap('tab20c')(7-ic), label= 'MLP - '+istart_label[ic])
    ic+=1
    # plt.plot(doy_arr[19:28],'*-',color='black')

    import statsmodels.api as sm
    # model = sm.OLS(doy_arr[19:28], sm.add_constant(fd_ML_forcast,has_constant='skip'), missing='drop').fit()
    # print('------')
    # print(model.rsquared, model.rsquared_adj)
    # print(np.nanmean(np.abs(doy_arr[19:28]-fd_ML_forcast)),np.nanmean(np.abs(doy_arr[19:28]-fd_clim_forcast)))
    # print(np.nansum(np.abs(doy_arr[19:28]-fd_ML_forcast)<=7),np.nansum(np.abs(doy_arr[19:28]-365)<=7))

    model = sm.OLS(avg_freezeup_doy[18:27], sm.add_constant(fd_ML_forcast,has_constant='skip'), missing='drop').fit()
    print('------')
    print(model.rsquared, model.rsquared_adj)
    print(np.nanmean(np.abs(avg_freezeup_doy[18:27]-fd_ML_forcast)),np.nanmean(np.abs(avg_freezeup_doy[18:27]-365)))
    print(np.nansum(np.abs(avg_freezeup_doy[18:27]-fd_ML_forcast)<=7),np.nansum(np.abs(avg_freezeup_doy[18:27]-365)<=7))


plt.legend()



#%%

plot_losses(train_losses,valid_losses,new_fig=True)