#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:23:02 2021

Tutorial from: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/structured_data/time_series.ipynb#scrollTo=YeCWbq6KLmL7
Alternative link : https://www.tensorflow.org/tutorials/structured_data/time_series

@author: Amelie
"""

import copy
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage

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
# horizon_arr = [8,16,24,32,42,64,96,128]
horizon_arr = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
# horizon_arr = [20,25]
horizon_arr = [5,10,15,20,25,30]
inpw = 96
ne = 200
# suffix = ''
suffix = '_withsnow'

# model_name = 'LSTM'
# horizon_arr = [5,10,15,20,25,30]
# horizon_arr = [45]
# inpw = 96
# ne = 100
# suffix = ''

plot_Tw_metric = np.zeros((len(horizon_arr),12))
plot_fu_metric = np.zeros((len(horizon_arr),6))
plot_fu_clim_metric = np.zeros((len(horizon_arr),6))

for ilbl,lblw in enumerate(horizon_arr):

    # Load data
    pred_data = np.load('./ML_pred_'+model_name+'_horizon_'+str(lblw)+'_input_'+str(inpw)+'_nepochs_'+str(ne)+suffix+'.npz',allow_pickle='TRUE')

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


    # plot_losses(train_losses,valid_losses,new_fig=True)

    #%%
    # Reshape time arrays
    # time_train = np.zeros((target_train.shape[0],label_width))*np.nan
    # for s in np.arange(0,target_train.shape[0]):
    #     time_train[s,:]= time_train_all[input_width+(nslide*s):input_width+label_width+(nslide*s)]

    # time_valid = np.zeros((target_valid.shape[0],label_width))*np.nan
    # for s in np.arange(0,target_valid.shape[0]):
    #     time_valid[s,:]= time_valid_all[input_width+(nslide*s):input_width+label_width+(nslide*s)]

    # # time_test = np.zeros((target_test.shape[0],label_width))*np.nan
    # # for s in np.arange(0,target_test.shape[0]):
    # #     time_test[s,:]= time_test_tmp[input_width+(nslide*s):input_width+label_width+(nslide*s)]

    #%%
    Tw_eval_arr = np.zeros((12,1))*np.nan
    Tw_clim_eval_arr = np.zeros((12,1))*np.nan
    fu_eval_arr = np.zeros((6,1))*np.nan
    fu_clim_eval_arr = np.zeros((6,1))*np.nan

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

    #%%
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

    #%%
    # # EVALUATE TWATER FORECAST ACCORDING TO SELECTED METRIC:
    for imonth,month in enumerate(np.arange(1,13)):
        samples_Tw = pred_arr[np.where(np.array([(date_ref+dt.timedelta(days=int(time_arr[s,0]))).month for s in range(time_arr.shape[0])]) == month )[0]][:,-1]
        targets_Tw = target_arr[np.where(np.array([(date_ref+dt.timedelta(days=int(time_arr[s,0]))).month for s in range(time_arr.shape[0])]) == month )[0]][:,-1]
        clim_Tw = clim_recons[np.where(np.array([(date_ref+dt.timedelta(days=int(time_arr[s,0]))).month for s in range(time_arr.shape[0])]) == month )[0]][:,-1]

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

    #%%
    # EVALUATE TWATER FORECAST ACCORDING TO SELECTED METRIC:
    # for imonth,month in enumerate(np.arange(1,13)):
    #     samples_Tw = pred_arr[np.where(np.array([(date_ref+dt.timedelta(days=int(time_arr[s,0]))).month for s in range(time_arr.shape[0])]) == month )[0]][:,:]
    #     targets_Tw = target_arr[np.where(np.array([(date_ref+dt.timedelta(days=int(time_arr[s,0]))).month for s in range(time_arr.shape[0])]) == month )[0]][:,:]
    #     clim_Tw = clim_recons[np.where(np.array([(date_ref+dt.timedelta(days=int(time_arr[s,0]))).month for s in range(time_arr.shape[0])]) == month )[0]][:,:]

    #     if metric == 'RMSE':
    #         Tw_metric = 0
    #         Tw_clim_metric = 0
    #         for s in range(samples_Tw.shape[0]):
    #             Tw_metric += (samples_Tw[s]-targets_Tw[s])**2.
    #             Tw_clim_metric += (clim_Tw[s]-targets_Tw[s])**2.
    #         Tw_metric = np.sqrt(Tw_metric/samples_Tw.shape[0])
    #         Tw_clim_metric = np.sqrt(Tw_clim_metric/samples_Tw.shape[0])

    #     if metric == 'MAE':
    #         Tw_metric = 0
    #         Tw_clim_metric = 0
    #         for s in range(samples_Tw.shape[0]):
    #             Tw_metric += np.nanmean(np.abs(samples_Tw[s]-targets_Tw[s]))
    #             Tw_clim_metric += np.nanmean(np.abs(clim_Tw[s]-targets_Tw[s]))
    #         Tw_metric = Tw_metric/samples_Tw.shape[0]
    #         Tw_clim_metric = Tw_clim_metric/samples_Tw.shape[0]

    #     if metric == 'NSE':
    #         print('NEEDS TO BE IMPLEMENTED!!')

    #     Tw_eval_arr[imonth] = Tw_metric
    #     Tw_clim_eval_arr[imonth] = Tw_clim_metric

    #%%
    # FIND FREEZE-UP FROM ALL SAMPLES

    # freezeup_opt = 1
    # month_start_day = 1

    # # OPTION 1
    # if freezeup_opt == 1:
    #     def_opt = 1
    #     smooth_T =False; N_smooth = 3; mean_type='centered'
    #     round_T = False; round_type= 'half_unit'
    #     Gauss_filter = False
    #     T_thresh = 0.75
    #     dTdt_thresh = 0.25
    #     d2Tdt2_thresh = 0.25
    #     nd = 1

    # # OPTION 2
    # if freezeup_opt == 2:
    #     def_opt = 3
    #     smooth_T =False; N_smooth = 3; mean_type='centered'
    #     round_T = False; round_type= 'half_unit'
    #     Gauss_filter = True
    #     sig_dog = 3.5
    #     T_thresh = 3.
    #     dTdt_thresh = 0.15
    #     d2Tdt2_thresh = 0.15
    #     # dTdt_thresh = 0.20 # using 0.20 here instead of 0.15 can achieve even lower difference between stations, but then some years do not match the charts timing anymore...
    #     # d2Tdt2_thresh = 0.20
    #     nd = 30


    # freezeup_dates_sample = np.zeros((12,samples.shape[0],4))*np.nan
    # freezeup_dates_target = np.zeros((12,samples.shape[0],4))*np.nan

    # for imonth,month in enumerate(np.arange(1,13)):
    #     samples = pred_arr[np.where(np.array([(date_ref+dt.timedelta(days=int(time_arr[s,0]))).month for s in range(time_arr.shape[0])]) == month )[0]]
    #     targets = target_arr[np.where(np.array([(date_ref+dt.timedelta(days=int(time_arr[s,0]))).month for s in range(time_arr.shape[0])]) == month )[0]]
    #     time_st = time_arr[np.where(np.array([(date_ref+dt.timedelta(days=int(time_arr[s,0]))).month for s in range(time_arr.shape[0])]) == month )[0]]

    #     for s in range(samples.shape[0]):

    #         time = time_st[s,:]

    #         # FIND DTDt, D2Tdt2,etc. - SAMPLE
    #         Twater_sample = samples[s]
    #         Twater_dTdt_sample = np.zeros(Twater_sample.shape)*np.nan
    #         Twater_d2Tdt2_sample = np.zeros(Twater_sample.shape)*np.nan
    #         Twater_DoG1_sample = np.zeros(Twater_sample.shape)*np.nan
    #         Twater_DoG2_sample = np.zeros(Twater_sample.shape)*np.nan

    #         Twater_tmp = Twater_sample.copy()
    #         if round_T:
    #             if round_type == 'unit':
    #                 Twater_tmp = np.round(Twater_tmp.copy())
    #             if round_type == 'half_unit':
    #                 Twater_tmp = np.round(Twater_tmp.copy()* 2) / 2.
    #         if smooth_T:
    #             Twater_tmp = running_nanmean(Twater_tmp.copy(),N_smooth,mean_type=mean_type)

    #         dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

    #         dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
    #         dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
    #         dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

    #         Twater_dTdt_sample= np.nanmean(dTdt_tmp[:,0:2],axis=1)
    #         Twater_d2Tdt2_sample = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

    #         if Gauss_filter:
    #             Twater_dTdt_sample = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=1)
    #             Twater_d2Tdt2_sample = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=2)



    #         # FIND DTDt, D2Tdt2,etc. - TARGET
    #         Twater_target = targets[s]
    #         Twater_dTdt_target = np.zeros(Twater_sample.shape)*np.nan
    #         Twater_d2Tdt2_target = np.zeros(Twater_sample.shape)*np.nan
    #         Twater_DoG1_target = np.zeros(Twater_sample.shape)*np.nan
    #         Twater_DoG2_target = np.zeros(Twater_sample.shape)*np.nan

    #         Twater_tmp_target = Twater_target.copy()
    #         if round_T:
    #             if round_type == 'unit':
    #                 Twater_tmp_target = np.round(Twater_tmp_target.copy())
    #             if round_type == 'half_unit':
    #                 Twater_tmp_target = np.round(Twater_tmp_target.copy()* 2) / 2.
    #         if smooth_T:
    #             Twater_tmp_target = running_nanmean(Twater_tmp_target.copy(),N_smooth,mean_type=mean_type)

    #         dTdt_tmp = np.zeros((Twater_tmp_target.shape[0],3))*np.nan

    #         dTdt_tmp[0:-1,0]= Twater_tmp_target[1:]- Twater_tmp_target[0:-1] # Forwards
    #         dTdt_tmp[1:,1] = Twater_tmp_target[1:] - Twater_tmp_target[0:-1] # Backwards
    #         dTdt_tmp[0:-1,2]= Twater_tmp_target[0:-1]-Twater_tmp_target[1:]  # -1*Forwards

    #         Twater_dTdt_target= np.nanmean(dTdt_tmp[:,0:2],axis=1)
    #         Twater_d2Tdt2_target = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

    #         if Gauss_filter:
    #             Twater_dTdt_target = scipy.ndimage.gaussian_filter1d(Twater_tmp_target.copy(),sigma=sig_dog,order=1)
    #             Twater_d2Tdt2_target = scipy.ndimage.gaussian_filter1d(Twater_tmp_target.copy(),sigma=sig_dog,order=2)



    #         # FIND FREEZE-UP FOR BOTH SAMPLE AND TARGET
    #         # ifz = 0
    #         date_start = dt.timedelta(days=int(time[0])) + date_ref
    #         if date_start.month < 3:
    #             year = date_start.year-1
    #         else:
    #             year = date_start.year

    #         if year >= years[0]:
    #             iyr = np.where(years == year)[0][0]
    #             fd_sample, ftw_sample, T_freezeup_sample, mask_freeze_sample = find_freezeup_Tw(def_opt,Twater_tmp,Twater_dTdt_sample,Twater_d2Tdt2_sample,time,year,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
    #             fd_target, ftw_target, T_freezeup_target, mask_freeze_target = find_freezeup_Tw(def_opt,Twater_tmp_target,Twater_dTdt_target,Twater_d2Tdt2_target,time,year,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)

    #             if (np.sum(mask_freeze_sample) > 0):
    #                 freezeup_dates_sample[imonth,s,0] = iyr
    #                 freezeup_dates_sample[imonth,s,1:4] = fd_sample

    #             if (np.sum(mask_freeze_target) > 0):
    #                 freezeup_dates_target[imonth,s,0] = iyr
    #                 freezeup_dates_target[imonth,s,1:4] = fd_target



    #         # # FINALLY, TRANSFORM FREEZEUP FROM DATE FORMAT TO DOY FORMAT:
    #         # for iyr,year in enumerate(years):
    #         #     if ~np.isnan(freezeup_dates[iyr,0,iloc]):
    #         #         fd_yy = int(freezeup_dates[iyr,0,iloc])
    #         #         fd_mm = int(freezeup_dates[iyr,1,iloc])
    #         #         fd_dd = int(freezeup_dates[iyr,2,iloc])

    #         #         fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
    #         #         if fd_doy < 60: fd_doy += 365

    #         #         freezeup_doy[iyr,iloc]=fd_doy


    #%%
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

    for istart in range(0,fu_eval_arr.shape[0],2):
        if istart == 0:
            month = 10
        if istart == 2:
            month = 11
        if istart == 4:
            month = 12
        if istart == 6:
            month = 1

        samples_tmp = pred_arr[np.where(np.array([(date_ref+dt.timedelta(days=int(time_arr[s,0]))).month for s in range(time_arr.shape[0])]) == month )[0]]
        targets_tmp = target_arr[np.where(np.array([(date_ref+dt.timedelta(days=int(time_arr[s,0]))).month for s in range(time_arr.shape[0])]) == month )[0]]
        time_st_tmp = time_arr[np.where(np.array([(date_ref+dt.timedelta(days=int(time_arr[s,0]))).month for s in range(time_arr.shape[0])]) == month )[0]]

        # Find sample starting on the 1st of the month
        samples_1 = samples_tmp[np.where( np.array([(date_ref+dt.timedelta(days=int(time_st_tmp[s,0]))).day for s in range(time_st_tmp.shape[0])])  == 1)[0]]
        targets_1 = targets_tmp[np.where( np.array([(date_ref+dt.timedelta(days=int(time_st_tmp[s,0]))).day for s in range(time_st_tmp.shape[0])])  == 1)[0]]
        time_1 = time_st_tmp[np.where( np.array([(date_ref+dt.timedelta(days=int(time_st_tmp[s,0]))).day for s in range(time_st_tmp.shape[0])])  == 1)[0]]

        samples = samples_1
        targets = targets_1
        time_st = time_1

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
            Twater_dTdt_target = np.zeros(Twater_sample.shape)*np.nan
            Twater_d2Tdt2_target = np.zeros(Twater_sample.shape)*np.nan
            Twater_DoG1_target = np.zeros(Twater_sample.shape)*np.nan
            Twater_DoG2_target = np.zeros(Twater_sample.shape)*np.nan

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

                if (np.sum(mask_freeze_sample) > 0):
                    freezeup_dates_sample[istart,s,0] = iyr
                    freezeup_dates_sample[istart,s,1:4] = fd_sample

                if (np.sum(mask_freeze_target) > 0):
                    freezeup_dates_target[istart,s,0] = iyr
                    freezeup_dates_target[istart,s,1:4] = fd_target



        if istart < 6: # (We do not look at forecasts starting Jan 15th...)
            # Find sample starting on the 15th of the month
            samples_15 = samples_tmp[np.where( np.array([(date_ref+dt.timedelta(days=int(time_st_tmp[s,0]))).day for s in range(time_st_tmp.shape[0])])  == 15)[0]]
            targets_15 = targets_tmp[np.where( np.array([(date_ref+dt.timedelta(days=int(time_st_tmp[s,0]))).day for s in range(time_st_tmp.shape[0])])  == 15)[0]]
            time_15 = time_st_tmp[np.where( np.array([(date_ref+dt.timedelta(days=int(time_st_tmp[s,0]))).day for s in range(time_st_tmp.shape[0])])  == 15)[0]]

            samples = samples_15
            targets = targets_15
            time_st = time_15

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
                Twater_dTdt_target = np.zeros(Twater_sample.shape)*np.nan
                Twater_d2Tdt2_target = np.zeros(Twater_sample.shape)*np.nan
                Twater_DoG1_target = np.zeros(Twater_sample.shape)*np.nan
                Twater_DoG2_target = np.zeros(Twater_sample.shape)*np.nan

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

                    if (np.sum(mask_freeze_sample) > 0):
                        freezeup_dates_sample[istart+1,s,0] = iyr
                        freezeup_dates_sample[istart+1,s,1:4] = fd_sample

                    if (np.sum(mask_freeze_target) > 0):
                        freezeup_dates_target[istart+1,s,0] = iyr
                        freezeup_dates_target[istart+1,s,1:4] = fd_target



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

    #%%
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


    #%%
    # EVALUATE FREEZE-UP FORECAST ACCORDING TO SELECTED METRIC:

    fu_metric = np.zeros((freezeup_dates_sample.shape[0],freezeup_dates_sample.shape[1]))*np.nan

    for istart in range(freezeup_dates_sample.shape[0]):

        sample_freezeup = freezeup_dates_sample[istart,:,:]
        target_freezeup = freezeup_dates_target[istart,:,:]

        # sample_freezeup = sample_freezeup_tmp[~np.isnan(sample_freezeup_tmp[:,0])]
        # target_freezeup = target_freezeup_tmp[~np.isnan(sample_freezeup_tmp[:,0])]

        if np.sum(~np.isnan(sample_freezeup)) > 0:

            n = 0

            for s in range(sample_freezeup.shape[0]):
                if ~np.isnan(sample_freezeup[s,0]):
                    iyr = int(sample_freezeup[s,0])
                    fs = dt.date(int(sample_freezeup[s,1]),int(sample_freezeup[s,2]),int(sample_freezeup[s,3]))
                    fo = dt.date(int(freezeup_dates_obs[iyr,0]),int(freezeup_dates_obs[iyr,1]),int(freezeup_dates_obs[iyr,2]))

                    if np.where( target_freezeup[:,0] == iyr )[0].shape[0] > 0:
                        s_t = np.where( target_freezeup[:,0] == iyr )[0][0]
                        ft = dt.date(int(target_freezeup[s_t,1]),int(target_freezeup[s_t,2]),int(target_freezeup[s_t,3]))
                    else:
                        ft = np.nan

                    # print(istart,fs,fo,ft)
                    if metric == 'RMSE':
                        fu_metric[istart,s] = ((fs-fo).days)**2.
                        n += 1

                    if metric == 'MAE':
                        fu_metric[istart,s] = np.abs((fs-fo).days)
                        n +=1

                    if metric == 'NSE':
                        print('NEEDS TO BE IMPLEMENTED!!')




        if metric == 'RMSE':
            fu_eval_arr[istart] = np.sqrt(np.nanmean(fu_metric[istart,:]))

        if metric == 'MAE':
            fu_eval_arr[istart] = np.nanmean(fu_metric[istart,:])

        if metric == 'NSE':
            print('NEEDS TO BE IMPLEMENTED!!')



    #%%
    # EVALUATE FREEZE-UP FORECAST USING CLIMATOLOGY:

    fu_clim_metric = np.zeros((freezeup_dates_sample.shape[0],freezeup_dates_sample.shape[1]))*np.nan

    for istart in range(freezeup_dates_sample.shape[0]):

        cd = clim_dates[istart,:]

        sample_freezeup = freezeup_dates_sample[istart,:,:]
        target_freezeup = freezeup_dates_target[istart,:,:]

        # sample_freezeup = sample_freezeup_tmp[~np.isnan(sample_freezeup_tmp[:,0])]
        # target_freezeup = target_freezeup_tmp[~np.isnan(sample_freezeup_tmp[:,0])]

        if np.sum(~np.isnan(sample_freezeup)) > 0:
            n = 0
            for s in range(sample_freezeup.shape[0]):
                if ~np.isnan(sample_freezeup[s,0]):
                    iyr = int(sample_freezeup[s,0])
                    fo = dt.date(int(freezeup_dates_obs[iyr,0]),int(freezeup_dates_obs[iyr,1]),int(freezeup_dates_obs[iyr,2]))
                    fc = dt.date(int(clim_dates[iyr,0]),int(clim_dates[iyr,1]),int(clim_dates[iyr,2]))

                    # print(istart,fs,fo,ft)
                    if metric == 'RMSE':
                        fu_clim_metric[istart,s] = ((fc-fo).days)**2.
                        n += 1

                    if metric == 'MAE':
                        fu_clim_metric[istart,s] = np.abs((fc-fo).days)
                        n +=1

                    if metric == 'NSE':
                        print('NEEDS TO BE IMPLEMENTED!!')




        if metric == 'RMSE':
            fu_clim_eval_arr[istart] = np.sqrt(np.nanmean(fu_clim_metric[istart,:]))

        if metric == 'MAE':
            fu_clim_eval_arr[istart] = np.nanmean(fu_clim_metric[istart,:])

        if metric == 'NSE':
            print('NEEDS TO BE IMPLEMENTED!!')


    #%%
    # Save Tw and freeze-up metrics for the given prediction horizon

    np.savez('./eval_metrics/'+model_name+'_FORECAST_EVAL_horizon_'+str(label_width)+'_input_'+str(input_width)+'_nepochs_'+str(nepochs)+suffix,
              fu_eval_arr = fu_eval_arr,
              Tw_eval_arr = Tw_eval_arr,
              fu_clim_eval_arr = fu_clim_eval_arr,
              Tw_clim_eval_arr = Tw_clim_eval_arr,
              freezeup_opt = freezeup_opt,
              metric = metric,
              anomaly = anomaly,
              label_width = label_width,
              input_width = input_width,
              nepochs = nepochs,
              )


    #%%
    # def plot_series(time, series, format="-", start=0, end=None, ax_labels=['Time','Value'],linecolor=''):
    #     plt.plot(time[start:end], series[start:end], format, color=linecolor)
    #     plt.xlabel(ax_labels[0])
    #     plt.ylabel(ax_labels[1])
    #     plt.grid(True)

    # def plot_series_1step(time, series, format="-", ax_labels=['Time','Value'],linecolor=''):
    #     plt.plot(time, series, format, color=linecolor)
    #     plt.xlabel(ax_labels[0])
    #     plt.ylabel(ax_labels[1])
    #     plt.grid(True)

    # MAE=nn.L1Loss()
    # MSE=nn.MSELoss()
    # x_renorm = (np.squeeze(torch.from_numpy(target_valid).float())*train_std) + train_mean
    # y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions_valid).astype(float)).float())*train_std) + train_mean
    # print('MLP MODEL, '+ str(label_width)+ '-STEP AHEAD -----------')
    # print('VALID')
    # print(MAE(x_renorm,y_renorm))
    # print(np.sqrt(MSE(x_renorm,y_renorm)))
    # print(np.corrcoef(np.array(x_renorm).ravel(),np.array(y_renorm).ravel())[0,1])
    # plt.figure()
    # if anomaly:
    #     clim_recons = (np.squeeze(torch.from_numpy(clim_valid).float())*train_std) + train_mean
    #     x_renorm = (np.squeeze(torch.from_numpy(target_valid).float())*train_std) + train_mean + clim_recons
    #     y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions_valid).astype(float)).float())*train_std) + train_mean + clim_recons
    # else:
    #     x_renorm = (np.squeeze(torch.from_numpy(target_valid).float())*train_std) + train_mean
    #     y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions_valid).astype(float)).float())*train_std) + train_mean

    # # First add climatology
    # x_renorm_clim = (np.squeeze(torch.from_numpy(target_valid).float())*train_std) + train_mean
    # y_renorm_clim = torch.from_numpy((Tw_climatology_mean_valid[input_width:]*train_std) + train_mean).float()
    # plot_series_1step(time_valid[input_width:],np.array(y_renorm_clim).ravel(),'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(0))

    # for s in np.arange(0,target_valid.shape[0],label_width):
    #     plot_series(time_target_valid[s,:],np.array(x_renorm)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
    #     plot_series(time_target_valid[s,:],np.array(y_renorm)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))

    # plt.title('Predictors: T$_{air}$, T$_{water}$, |T$_{water}$-T$_{air}$| (previous ' + '%2i'%input_width + 'days)')



    #%%
    # def plot_series(time, series, format="-", start=0, end=None, ax_labels=['Time','Value'],linecolor=''):
    #     plt.plot(time[start:end], series[start:end], format, color=linecolor)
    #     plt.xlabel(ax_labels[0])
    #     plt.ylabel(ax_labels[1])
    #     plt.grid(True)

    # def plot_series_1step(time, series, format="-", ax_labels=['Time','Value'],linecolor=''):
    #     plt.plot(time, series, format, color=linecolor)
    #     plt.xlabel(ax_labels[0])
    #     plt.ylabel(ax_labels[1])
    #     plt.grid(True)


    # MAE=nn.L1Loss()
    # MSE=nn.MSELoss()
    # x_renorm = (np.squeeze(torch.from_numpy(target_train).float())*train_std) + train_mean
    # y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions_train).astype(float)).float())*train_std) + train_mean
    # print('MLP MODEL, '+ str(label_width)+ '-STEP AHEAD -----------')
    # print('TRAIN')
    # print(MAE(x_renorm,y_renorm))
    # print(np.sqrt(MSE(x_renorm,y_renorm)))
    # print(np.corrcoef(np.array(x_renorm).ravel(),np.array(y_renorm).ravel())[0,1])


    # plt.figure()
    # if anomaly:
    #     clim_recons = (np.squeeze(torch.from_numpy(clim_train).float())*train_std) + train_mean
    #     x_renorm = (np.squeeze(torch.from_numpy(target_train).float())*train_std) + train_mean + clim_recons
    #     y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions_train).astype(float)).float())*train_std) + train_mean + clim_recons
    # else:
    #     x_renorm = (np.squeeze(torch.from_numpy(target_train).float())*train_std) + train_mean
    #     y_renorm = (np.squeeze(torch.from_numpy(np.array(predictions_train).astype(float)).float())*train_std) + train_mean


    # for s in np.arange(5,target_train.shape[0],15):
    #     plot_series(time_target_train[s,:],np.array(x_renorm)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
    #     plot_series(time_target_train[s,:],np.array(y_renorm)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))
    # plt.title('Predictors: T$_{air}$, T$_{water}$, |T$_{water}$-T$_{air}$| (previous ' + '%2i'%input_width + 'days)')


