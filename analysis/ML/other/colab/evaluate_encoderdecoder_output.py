#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 09:56:04 2022

@author: amelie
"""
#%%
local_path = '/storage/amelie/'
# local_path = '/Volumes/SeagateUSB/McGill/Postdoc/'

#%%
import tensorflow as tf
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath(local_path +'slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from keras.layers import Lambda
import sklearn.metrics as metrics

import numpy as np
import pandas as pd
import datetime as dt
import time as timer

from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import cmocean

from functions import rolling_climo
from functions_ML import regression_metrics, plot_sample, plot_prediction_timeseries

from compare_encoderdecoder_output import load_data_year
from compare_encoderdecoder_output import plot_Tw_metric, evaluate_Tw_forecasts
from compare_encoderdecoder_output import detect_FUD_from_Tw_samples, evaluate_FUD_forecasts
#%%
# RUN OPTIONS:
suffix = '_perfectexp_noTw'

pred_len = 30
input_len = 128
n_epochs = 50
latent_dim = 50
nb_layers = 1

loss_name = 'MSETw'
# loss_name = 'MSETw_MSEdTwdt'
# loss_name = 'MSETw_with_weights1_on_thresh0_75'

# dense_act_func_name = 'sigmoid'
# norm_type='MinMax'
dense_act_func_name = 'None'
norm_type='Standard'

anomaly_target = True
# anomaly_target = False

# valid_scheme = 'LOOk'
# valid_scheme = 'standard'
# train_yr_start = 1992 # Training dataset: 1992 - 2010
# valid_yr_start = 2011 # Validation dataset: 2011 - 2015
# test_yr_start = 2016 # Testing dataset: 2016 - 2021

plot_series = True
# plot_series = False

#%%
test_years = np.arange(1992,2020)

pred_type = 'test'
target_time = np.zeros((len(test_years),10220,pred_len))*np.nan
y_pred = np.zeros((len(test_years),10220,pred_len))*np.nan
y = np.zeros((len(test_years),10220,pred_len))*np.nan
y_clim = np.zeros((len(test_years),10220,pred_len))*np.nan

# pred_type = 'valid'
# target_time = np.zeros((len(test_years),5,10220,pred_len))*np.nan
# y_pred = np.zeros((len(test_years),5,10220,pred_len))*np.nan
# y = np.zeros((len(test_years),5,10220,pred_len))*np.nan
# y_clim = np.zeros((len(test_years),5,10220,pred_len))*np.nan

for iyr,yr_test in enumerate(test_years):
    if yr_test != 2019:
        [target_time_yr,
          y_pred_yr, y_yr, y_clim_yr,
          valid_scheme, nfolds] = load_data_year(yr_test,pred_type,pred_len,input_len,latent_dim,n_epochs,nb_layers,norm_type,dense_act_func_name,loss_name,anomaly_target,suffix)

        target_time[iyr,0:y_yr.shape[0],:] = target_time_yr
        y_pred[iyr,0:y_yr.shape[0],:] = y_pred_yr
        y[iyr,0:y_yr.shape[0],:] = y_yr
        y_clim[iyr,0:y_yr.shape[0],:] = y_clim_yr


#%% PLOT PREDICTION TIME SERIES
if plot_series:

    if valid_scheme == 'standard':
        plot_prediction_timeseries(y_pred,y,y_clim,target_time,pred_type, lead=0, nyrs_plot= 28)
        plot_prediction_timeseries(y_pred,y,y_clim,target_time,pred_type, lead=50, nyrs_plot= 28)

    if valid_scheme == 'LOOk':
        if (pred_type == 'test') | (pred_type == 'train') :
            for iyr in range(len(test_years)):
                y_pred_in = y_pred[iyr,:,:]
                y_pred_in = y_pred_in[np.where(np.all(~np.isnan(y_pred_in),axis=1))[0]]
                y_in = y[iyr,:,:]
                y_in = y_in[np.where(np.all(~np.isnan(y_in),axis=1))[0]]
                y_clim_in = y_clim[iyr,:,:]
                y_clim_in = y_clim_in[np.where(np.all(~np.isnan(y_clim_in),axis=1))[0]]
                time_in = target_time[iyr,:,:]
                time_in = time_in[np.where(np.all(~np.isnan(time_in),axis=1))[0]]
                if y_pred_in.shape[0]>0:
                    # plot_prediction_timeseries(y_pred_in,y_in,y_clim_in,time_in,pred_type,lead=0,nyrs_plot= 28)
                    # plot_prediction_timeseries(y_pred_in,y_in,y_clim_in,time_in,pred_type,lead=50,nyrs_plot= 28)
                    plot_sample(y_pred_in,y_in,y_clim_in,time_in,it=125,pred_type=pred_type,show_clim=True)


        if (pred_type == 'valid'):
            for iyr in range(len(test_years)):
                for ifold in range(nfolds):
                    y_pred_in = y_pred[iyr,ifold,:,:]
                    y_pred_in = y_pred_in[np.where(np.all(~np.isnan(y_pred_in),axis=1))[0]]
                    y_in = y[iyr,ifold,:,:]
                    y_in = y_in[np.where(np.all(~np.isnan(y_in),axis=1))[0]]
                    y_clim_in = y_clim[iyr,ifold,:,:]
                    y_clim_in = y_clim_in[np.where(np.all(~np.isnan(y_clim_in),axis=1))[0]]
                    time_in = target_time[iyr,ifold,:,:]
                    time_in = time_in[np.where(np.all(~np.isnan(time_in),axis=1))[0]]
                    if y_pred_in.shape[0]>0:
                        plot_prediction_timeseries(y_pred_in,y_in,y_clim_in,time_in,pred_type,lead=0,nyrs_plot=28)
                        plot_prediction_timeseries(y_pred_in,y_in,y_clim_in,time_in,pred_type,lead=50,nyrs_plot=28)

#%% EVALUATE TWATER FORECASTS
if valid_scheme == 'LOOk':
    if (pred_type == 'test') | (pred_type == 'train') :
        model_name = 'Encoder-Decoder LSTM - ' + pred_type
        Tw_MAE_yr = np.zeros((12,pred_len,len(test_years)))*np.nan
        Tw_clim_MAE_yr = np.zeros((12,pred_len,len(test_years)))*np.nan
        for iyr in range(len(test_years)):
            y_pred_in_tmp = y_pred[iyr,:,:]
            y_pred_in = y_pred_in_tmp[np.where(np.all(~np.isnan(y_pred_in_tmp),axis=1))[0]]
            y_in = y[iyr,:,:]
            y_in = y_in[np.where(np.all(~np.isnan(y_pred_in_tmp),axis=1))[0]]
            y_clim_in = y_clim[iyr,:,:]
            y_clim_in = y_clim_in[np.where(np.all(~np.isnan(y_pred_in_tmp),axis=1))[0]]
            time_in = target_time[iyr,:,:]
            time_in = time_in[np.where(np.all(~np.isnan(y_pred_in_tmp),axis=1))[0]]
            if y_pred_in.shape[0]>0:
                Tw_MAE_yr[:,:,iyr], Tw_clim_MAE_yr[:,:,iyr] = evaluate_Tw_forecasts(y_pred_in,y_in,y_clim_in,time_in,pred_len,model_name)
        Tw_MAE = np.nanmean(Tw_MAE_yr,axis=2)
        Tw_clim_MAE = np.nanmean(Tw_clim_MAE_yr,axis=2)
        Tw_MAE_std = np.nanstd(Tw_MAE_yr,axis=2)
        Tw_clim_MAE_std = np.nanstd(Tw_clim_MAE_yr,axis=2)
        plot_Tw_metric(Tw_MAE,Tw_clim_MAE,'MAE',model_name)
        plot_Tw_metric(Tw_MAE_std,Tw_clim_MAE_std,'MAE',model_name,vmin=0,vmax=0.5)

    if pred_type == 'valid' :
        model_name = 'Encoder-Decoder LSTM - ' + pred_type
        show_all_folds = False
        Tw_MAE_fold = np.zeros((12,pred_len,len(test_years),nfolds))*np.nan
        Tw_clim_MAE_fold = np.zeros((12,pred_len,len(test_years),nfolds))*np.nan
        for iyr in range(len(test_years)):
            for ifold in range(nfolds):
                y_pred_in_tmp = y_pred[iyr,ifold,:,:]
                y_pred_in = y_pred_in_tmp[np.where(np.all(~np.isnan(y_pred_in_tmp),axis=1))[0]]
                y_in = y[iyr,ifold,:,:]
                y_in = y_in[np.where(np.all(~np.isnan(y_pred_in_tmp),axis=1))[0]]
                y_clim_in = y_clim[iyr,ifold,:,:]
                y_clim_in = y_clim_in[np.where(np.all(~np.isnan(y_pred_in_tmp),axis=1))[0]]
                time_in = target_time[iyr,ifold,:,:]
                time_in = time_in[np.where(np.all(~np.isnan(y_pred_in_tmp),axis=1))[0]]
                if y_pred_in.shape[0]>0:
                    Tw_MAE_fold[:,:,iyr,ifold], Tw_clim_MAE_fold[:,:,iyr,ifold] = evaluate_Tw_forecasts(y_pred_in,y_in,y_clim_in,time_in,pred_len,model_name)
            if show_all_folds:
                if ~np.all(np.isnan(Tw_MAE_fold[:,:,iyr,:])):
                    Tw_MAE_yr = np.nanmean(Tw_MAE_fold,axis=3)
                    Tw_clim_MAE_yr = np.nanmean(Tw_clim_MAE_fold,axis=3)
                    Tw_MAE_std_yr = np.nanstd(Tw_MAE_fold,axis=3)
                    Tw_clim_MAE_std_yr = np.nanstd(Tw_clim_MAE_fold,axis=3)
                    plot_Tw_metric(Tw_MAE_yr[:,:,iyr],Tw_clim_MAE_yr[:,:,iyr],'MAE_mean',model_name)
                    # plot_Tw_metric(Tw_MAE_std_yr[:,:,iyr],Tw_clim_MAE_std_yr[:,:,iyr],'MAE_std',model_name,vmin=0,vmax=0.5)
        Tw_MAE = np.nanmean(Tw_MAE_fold,axis=(2,3))
        Tw_clim_MAE = np.nanmean(Tw_clim_MAE_fold,axis=(2,3))
        Tw_MAE_std = np.nanstd(Tw_MAE_fold,axis=(2,3))
        Tw_clim_MAE_std = np.nanstd(Tw_clim_MAE_fold,axis=(2,3))
        plot_Tw_metric(Tw_MAE,Tw_clim_MAE,'MAE_mean',model_name)
        # plot_Tw_metric(Tw_MAE_std,Tw_clim_MAE_std,'MAE_std',model_name,vmin=0,vmax=0.5)

if valid_scheme == 'standard':
    model_name = 'Encoder-Decoder LSTM - ' + pred_type
    y_clim = np.squeeze(y_clim)
    Tw_MAE, Tw_clim_MAE = evaluate_Tw_forecasts(y_pred,y,y_clim,target_time,pred_len,model_name)
    plot_Tw_metric(Tw_MAE,Tw_clim_MAE,'MAE_mean',model_name)



#%% EVALUATE FUD FORECASTS
#-----------------
freezeup_opt = 1
start_doy_arr =[307,        314,         321,         328,         335        ]
istart_label = ['Nov. 3rd', 'Nov. 10th', 'Nov. 17th', 'Nov. 24th', 'Dec. 1st' ]
month_istart = [11,11,11,11,12]
day_istart   = [ 3,10,17,24, 1]
date_ref = dt.date(1900,1,1)
years_eval = np.arange(1992,2020)

plot_samples = False
# plot_samples = True

#-----------------

if valid_scheme == 'standard':
    y_true = y
    y_pred = y_pred
    y_clim = y_clim
    time_y = target_time
    model_name = 'Encoder-Decoder LSTM - '+pred_type


    [freezeup_dates_sample,freezeup_dates_sample_doy,
      freezeup_dates_target,freezeup_dates_target_doy,
      freezeup_dates_clim_target,freezeup_dates_clim_target_doy,
      mean_clim_FUD]  = detect_FUD_from_Tw_samples(y_true,y_pred,y_clim,time_y,years_eval,freezeup_opt,start_doy_arr,istart_label,day_istart,month_istart,plot_samples)

    plot_FUD_ts = True
    plot_metrics = False
    MAE_arr, RMSE_arr = evaluate_FUD_forecasts(freezeup_dates_sample,freezeup_dates_sample_doy,freezeup_dates_target,freezeup_dates_target_doy,mean_clim_FUD,model_name,years_eval,start_doy_arr,istart_label,plot_FUD_ts,plot_metrics,verbose=True)
    plot_FUD_ts = False
    plot_metrics = False
    MAE_arr_clim, RMSE_arr_clim = evaluate_FUD_forecasts(freezeup_dates_clim_target,freezeup_dates_clim_target_doy,freezeup_dates_target,freezeup_dates_target_doy,mean_clim_FUD,model_name,years_eval,start_doy_arr,istart_label,plot_FUD_ts,plot_metrics,verbose=True)

    istart_plot = [0,1,2,3,4]
    fig_mae_ts,ax_mae_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
    ax_mae_ts.plot(np.arange(len(istart_plot)),MAE_arr,'o-',label='Model')
    ax_mae_ts.plot(np.arange(len(istart_plot)),MAE_arr_clim,'x-',label='Climatology')
    ax_mae_ts.set_ylabel('MAE (days)')
    ax_mae_ts.set_xlabel('Forecast start date')
    ax_mae_ts.set_xticks(np.arange(len(istart_plot)))
    ax_mae_ts.set_xticklabels([istart_label[k] for k in range(len(istart_plot))])
    ax_mae_ts.legend()

    fig_mae_ss,ax_mae_ss = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
    ax_mae_ss.plot(np.arange(len(istart_plot)),1-(MAE_arr/MAE_arr_clim),'o-')
    ax_mae_ss.set_ylabel('Skill Score (MAE)')
    ax_mae_ss.set_xlabel('Forecast start date')
    ax_mae_ss.set_xticks(np.arange(len(istart_plot)))
    ax_mae_ss.set_xticklabels([istart_label[k] for k in range(len(istart_plot))])



if valid_scheme == 'LOOk':

    if (pred_type == 'test') | (pred_type == 'train'):
        for iyr in range(len(test_years)):
            y_pred_tmp = y_pred[iyr,:,:]
            y_true_yr = y[iyr,:,:]
            y_clim_yr = y_clim[iyr,:,:]
            time_y_yr = target_time[iyr,:,:]
            model_name = 'Encoder-Decoder LSTM - '+pred_type

            y_pred_yr = y_pred_tmp[np.where(np.all(~np.isnan(y_pred_tmp),axis=1))[0]]
            y_true_yr = y_true_yr[np.where(np.all(~np.isnan(y_pred_tmp),axis=1))[0]]
            y_clim_yr = y_clim_yr[np.where(np.all(~np.isnan(y_pred_tmp),axis=1))[0]]
            time_y_yr = time_y_yr[np.where(np.all(~np.isnan(y_pred_tmp),axis=1))[0]]

            [freezeup_dates_sample_yr,freezeup_dates_sample_doy_yr,
              freezeup_dates_target_yr,freezeup_dates_target_doy_yr,
              freezeup_dates_clim_target_yr,freezeup_dates_clim_target_doy_yr,
              mean_clim_FUD_yr]  = detect_FUD_from_Tw_samples(y_true_yr,y_pred_yr,y_clim_yr,time_y_yr,years_eval,freezeup_opt,start_doy_arr,istart_label,day_istart,month_istart,plot_samples)

            if iyr == 0:
                [freezeup_dates_sample,freezeup_dates_sample_doy,
                  freezeup_dates_target,freezeup_dates_target_doy,
                  freezeup_dates_clim_target,freezeup_dates_clim_target_doy,
                  mean_clim_FUD]  = [freezeup_dates_sample_yr,freezeup_dates_sample_doy_yr,
                                      freezeup_dates_target_yr,freezeup_dates_target_doy_yr,
                                      freezeup_dates_clim_target_yr,freezeup_dates_clim_target_doy_yr,
                                      mean_clim_FUD_yr]
            else:
                freezeup_dates_sample = np.concatenate((freezeup_dates_sample,freezeup_dates_sample_yr),axis=1)
                freezeup_dates_sample_doy = np.concatenate((freezeup_dates_sample_doy,freezeup_dates_sample_doy_yr),axis=1)
                freezeup_dates_target = np.concatenate((freezeup_dates_target,freezeup_dates_target_yr),axis=1)
                freezeup_dates_target_doy = np.concatenate((freezeup_dates_target_doy,freezeup_dates_target_doy_yr),axis=1)
                freezeup_dates_clim_target = np.concatenate((freezeup_dates_clim_target,freezeup_dates_clim_target_yr),axis=1)
                freezeup_dates_clim_target_doy = np.concatenate((freezeup_dates_clim_target_doy,freezeup_dates_clim_target_doy_yr),axis=1)
                mean_clim_FUD = np.nanmean((mean_clim_FUD,mean_clim_FUD_yr))


        if freezeup_dates_sample.shape[1] > 0:
            plot_FUD_ts = True
            plot_metrics = True
            MAE_arr, RMSE_arr = evaluate_FUD_forecasts(freezeup_dates_sample,freezeup_dates_sample_doy,freezeup_dates_target,freezeup_dates_target_doy,mean_clim_FUD,model_name,years_eval,start_doy_arr,istart_label,plot_FUD_ts,plot_metrics,verbose=True)
            plot_FUD_ts = False
            plot_metrics = False
            MAE_arr_clim, RMSE_arr_clim = evaluate_FUD_forecasts(freezeup_dates_clim_target,freezeup_dates_clim_target_doy,freezeup_dates_target,freezeup_dates_target_doy,mean_clim_FUD,model_name,years_eval,start_doy_arr,istart_label,plot_FUD_ts,plot_metrics,verbose=True)

        istart_plot = [0,1,2,3,4]
        fig_mae_ts,ax_mae_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
        ax_mae_ts.plot(np.arange(len(istart_plot)),MAE_arr,'o-',label='Model')
        ax_mae_ts.plot(np.arange(len(istart_plot)),MAE_arr_clim,'x-',label='Climatology')
        ax_mae_ts.set_ylabel('MAE (days)')
        ax_mae_ts.set_xlabel('Forecast start date')
        ax_mae_ts.set_xticks(np.arange(len(istart_plot)))
        ax_mae_ts.set_xticklabels([istart_label[k] for k in range(len(istart_plot))])
        ax_mae_ts.legend()

        fig_mae_ss,ax_mae_ss = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
        ax_mae_ss.plot(np.arange(len(istart_plot)),1-(MAE_arr/MAE_arr_clim),'o-')
        ax_mae_ss.set_ylabel('Skill Score (MAE)')
        ax_mae_ss.set_xlabel('Forecast start date')
        ax_mae_ss.set_xticks(np.arange(len(istart_plot)))
        ax_mae_ss.set_xticklabels([istart_label[k] for k in range(len(istart_plot))])



    if pred_type == 'valid':
        SS_MAE_arr_valid = np.zeros((len(start_doy_arr),len(test_years),nfolds))*np.nan
        MAE_arr_valid = np.zeros((len(start_doy_arr),len(test_years),nfolds))*np.nan
        RMSE_arr_valid = np.zeros((len(start_doy_arr),len(test_years),nfolds))*np.nan
        MAE_arr_clim_valid = np.zeros((len(start_doy_arr),len(test_years),nfolds))*np.nan
        RMSE_arr_clim_valid = np.zeros((len(start_doy_arr),len(test_years),nfolds))*np.nan
        for iyr in range(len(test_years)):
            for ifold in range(nfolds):
                y_pred_tmp = y_pred[iyr,ifold,:,:]
                y_true_fold = y[iyr,ifold,:,:]
                y_clim_fold = y_clim[iyr,ifold,:,:]
                time_y_fold = target_time[iyr,ifold,:,:]
                model_name = 'Encoder-Decoder LSTM -' + pred_type

                y_pred_fold = y_pred_tmp[np.where(np.all(~np.isnan(y_pred_tmp),axis=1))[0]]
                y_true_fold = y_true_fold[np.where(np.all(~np.isnan(y_pred_tmp),axis=1))[0]]
                y_clim_fold = y_clim_fold[np.where(np.all(~np.isnan(y_pred_tmp),axis=1))[0]]
                time_y_fold = time_y_fold[np.where(np.all(~np.isnan(y_pred_tmp),axis=1))[0]]

                [freezeup_dates_sample_fold,freezeup_dates_sample_doy_fold,
                  freezeup_dates_target_fold,freezeup_dates_target_doy_fold,
                  freezeup_dates_clim_target_fold,freezeup_dates_clim_target_doy_fold,
                  mean_clim_FUD]  = detect_FUD_from_Tw_samples(y_true_fold,y_pred_fold,y_clim_fold,time_y_fold,years_eval,freezeup_opt,start_doy_arr,istart_label,day_istart,month_istart,plot_samples)

                if freezeup_dates_sample_fold.shape[1] > 0:
                    plot_FUD_ts = False
                    plot_metrics = False
                    MAE_arr_valid[:,iyr,ifold], RMSE_arr_valid[:,iyr,ifold] = evaluate_FUD_forecasts(freezeup_dates_sample_fold,freezeup_dates_sample_doy_fold,freezeup_dates_target_fold,freezeup_dates_target_doy_fold,mean_clim_FUD,model_name,years_eval,start_doy_arr,istart_label,plot_FUD_ts,plot_metrics,verbose=True)
                    plot_FUD_ts = False
                    plot_metrics = False
                    MAE_arr_clim_valid[:,iyr,ifold], RMSE_arr_clim_valid[:,iyr,ifold] = evaluate_FUD_forecasts(freezeup_dates_clim_target_fold,freezeup_dates_clim_target_doy_fold,freezeup_dates_target_fold,freezeup_dates_target_doy_fold,mean_clim_FUD,model_name,years_eval,start_doy_arr,istart_label,plot_FUD_ts,plot_metrics,verbose=True)
                    SS_MAE_arr_valid[:,iyr,ifold] = 1-(MAE_arr_valid[:,iyr,ifold]/MAE_arr_clim_valid[:,iyr,ifold])



        istart_plot = [0,1,2,3,4]
        fig_mae_CV,ax_mae_CV = plt.subplots(nrows = 1, ncols = len(istart_plot),figsize=(12,3),sharex=True,sharey=True)
        fig_mae_SS,ax_mae_SS = plt.subplots(nrows = 1, ncols = len(istart_plot),figsize=(12,3),sharex=True,sharey=True)
        letter = ['a)','b)','c)','d)','e)','f)']
        for i,istart in enumerate(istart_plot):
            for iyr in range(len(test_years)):
                ax_mae_CV[i].plot(test_years[iyr],np.nanmean(MAE_arr_valid[istart,iyr,:]),'x',color=plt.get_cmap('tab20')(1))
                ax_mae_CV[i].plot([test_years[iyr],test_years[iyr]],[np.nanmin(MAE_arr_valid[istart,iyr,:]),np.nanmax(MAE_arr_valid[istart,iyr,:])],'-',linewidth=0.5,color=plt.get_cmap('tab20')(1))
            ax_mae_CV[i].plot(test_years,np.ones(len(test_years))*np.nanmean(MAE_arr_valid[istart,:,:],axis=(0,1)),'-',color=plt.get_cmap('tab20')(0))
            # !!!! ADD OPTION HERE TO ADD TEST RESULTS AS WELL FOR EACH START DATE.
            ax_mae_CV[i].set_xlabel('Years')
            ax_mae_CV[i].set_xticks(test_years)
            ax_mae_CV[i].set_xticklabels(['' for k in range(len(test_years))])
            # ax_mae_CV[i].set_xticklabels([str(test_years[k]) for k in range(len(test_years))])
            # ax_mae_CV[i].legend()
            ax_mae_CV[i].set_title(letter[i]+' '+istart_label[istart])


            for iyr in range(len(test_years)):
                ax_mae_SS[i].plot(test_years[iyr],np.nanmean(SS_MAE_arr_valid[istart,iyr,:]),'x',color=plt.get_cmap('tab20')(1))
                ax_mae_SS[i].plot([test_years[iyr],test_years[iyr]],[np.nanmin(SS_MAE_arr_valid[istart,iyr,:]),np.nanmax(SS_MAE_arr_valid[istart,iyr,:])],'-',linewidth=0.5,color=plt.get_cmap('tab20')(1))
            ax_mae_SS[i].plot(test_years,np.ones(len(test_years))*np.nanmean(SS_MAE_arr_valid[istart,:,:],axis=(0,1)),'-',color=plt.get_cmap('tab20')(0))
            # !!!! ADD OPTION HERE TO ADD TEST RESULTS AS WELL FOR EACH START DATE.
            ax_mae_SS[i].set_xlabel('Years')
            ax_mae_SS[i].set_xticks(test_years)
            ax_mae_SS[i].set_xticklabels(['' for k in range(len(test_years))])
            # ax_mae_SS[i].set_xticklabels([str(test_years[k]) for k in range(len(test_years))])
            # ax_mae_SS[i].legend()
            ax_mae_SS[i].set_title(letter[i]+' '+istart_label[istart])


        fig_mae_CV.suptitle('VALID')
        fig_mae_CV.subplots_adjust(top=0.8,bottom=0.15,left=0.05,right=0.95)
        ax_mae_CV[0].set_ylabel('MAE (days)')

        fig_mae_SS.suptitle('VALID')
        fig_mae_SS.subplots_adjust(top=0.8,bottom=0.15,left=0.05,right=0.95)
        ax_mae_SS[0].set_ylabel(r'Skill Score (1- $\frac{MAE_{model}}{MAE_{clim}}$)')











