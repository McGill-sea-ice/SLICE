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
#%%

def load_data_year(eval_set,pred_len,input_len,latent_dim,n_epochs,nb_layers,norm_type,dense_act_func_name,loss_name,anomaly_target,suffix,yr_test=None,load_history=False,fpath='./'):

    fname = 'encoderdecoder_horizon'+str(pred_len)+'_context'+str(input_len)+'_nneurons'+str(latent_dim)+'_nepochs'+str(n_epochs)+(nb_layers > 1)*('_'+str(nb_layers)+'layers')+'_'+norm_type+'_'+dense_act_func_name+'_'+loss_name+'_'+anomaly_target*'anomaly_target_'+suffix
    if yr_test is None:
        data = np.load(fpath+'output/'+fname+'.npz', allow_pickle=True)
    else:
        data = np.load(fpath+'output/'+fname+'_'+str(yr_test)+'.npz', allow_pickle=True)

    if eval_set == 'train':
        target_time = data['target_time_train']
        y_pred = data['y_pred_train']
        y = data['y_train']
        y_clim = data['y_clim_train']

    elif eval_set == 'test':
        target_time = data['target_time_test']
        y_pred = data['y_pred_test']
        y = data['y_test']
        y_clim = data['y_clim_test']

    elif eval_set == 'valid':
        target_time = data['target_time_valid']
        y_pred = data['y_pred_valid']
        y = data['y_valid']
        y_clim = data['y_clim_valid']

    print(data['seed'])
    valid_scheme = data['valid_scheme']
    nfolds = data['nfolds']
    if load_history: test_history = data['test_history'].ravel()[0]
    if load_history: valid_history = data['valid_history'].ravel()[0]

    print('Predictor vars:', data['predictor_vars'])
    print('Forecast vars:', data['forecast_vars'])
    print('Tagret var:', data['target_var'])
    print('Anomaly predictors:', data['anomaly_past'] )
    print('Anomaly frcst:', data['anomaly_frcst'] )
    print('Anomaly target:', data['anomaly_target'] )
    print('Learning rate:', data['learning_rate'] )
    print('Batch Size:', data['batch_size'] )
    print('Optimizer:', data['optimizer_name'] )
    print('fixed seed:', data['fixed_seed'])

    # predictor_vars = data['predictor_vars']
    # target_var = data['target_var']
    # date_ref = data['date_ref']
    # anomaly_target = data['anomaly_target']
    # anomaly_past =  data['anomaly_past']
    # anomaly_frcst = data['anomaly_frcst']
    # train_yr_start = data['train_yr_start']
    # valid_yr_start = data['valid_yr_start']
    # test_yr_start = data['test_yr_start']
    # seed = data['seed']
    # fixed_seed = data['fixed_seed']
    # batch_size = ['batch_size']
    # nb_layers = ['nb_layers']
    # learning_rate = ['lr']
    # inp_dropout = data['inp_dropout']
    # rec_dropout = data['rec_dropout']
    # optimizer_name = data['optimizer_name']
    if load_history:
        return fname, target_time, y_pred, y, y_clim, valid_scheme, nfolds, test_history, valid_history
    else:
        return fname, target_time, y_pred, y, y_clim, valid_scheme, nfolds


def detect_FUD_from_Tw_samples(y_true,y_pred,y_clim,time_y,years,freezeup_opt,start_doy_arr,istart_label,day_istart,month_istart,plot_samples,date_ref = dt.date(1900,1,1) ):

    from functions import running_nanmean
    import scipy
    import scipy.ndimage
    import calendar


    def find_freezeup_Tw_threshold(def_opt,Twater_in,time,thresh_T=2.0,ndays=7,date_ref=dt.date(1900,1,1)):

        def record_event(istart,time,year,Twater_in,freezeup_date,mask_freezeup,date_ref):
            # Temperature has been lower than thresh_T
            # for more than (or equal to) ndays.
            # Define freezeup date as first date of group

            date_start = date_ref+dt.timedelta(days=int(time[istart]))
            doy_start = (date_start - dt.date(int(date_start.year),1,1)).days+1

            if ((date_start.year > 1992) | ((date_start.year == 1992) & (date_start.month > 10)) ):
                if ( (date_start.year == year) & (doy_start > 319) ) | ((date_start.year == year+1) & (doy_start < 46)):
                        freezeup_date[0] = date_start.year
                        freezeup_date[1] = date_start.month
                        freezeup_date[2] = date_start.day
                        freezeup_Tw = Twater_in[istart]
                        mask_freezeup[istart] = True
                else:
                    freezeup_date[0] = np.nan
                    freezeup_date[1] = np.nan
                    freezeup_date[2] = np.nan
                    freezeup_Tw = np.nan
                    mask_freezeup[istart] = False
            else:
            # I think this condition exists because the Tw time series
            # starts in January 1992, so it is already frozen, but we
            # do not want to detect this as freezeup for 1992, so we
            # have to wait until at least October 1992 before recording
            # any FUD events.
                freezeup_date[0] = np.nan
                freezeup_date[1] = np.nan
                freezeup_date[2] = np.nan
                freezeup_Tw = np.nan
                mask_freezeup[istart] = False

            return freezeup_date, freezeup_Tw, mask_freezeup

        date_start = dt.timedelta(days=int(time[0])) + date_ref
        if date_start.month < 3:
            year = date_start.year-1
        else:
            year = date_start.year

        mask_tmp = Twater_in <= thresh_T
        mask_freezeup = mask_tmp.copy()
        mask_freezeup[:] = False

        freezeup_Tw = np.nan
        freezeup_date=np.zeros((3))*np.nan

        # Loop on sample time steps
        for im in range(mask_freezeup.size):

            if (im == 0): # First time step cannot be detected as freeze-up
                sum_m = 0
                istart = -1 # This ensures that a freeze-up is not detected if the time series started already below the freezing temp.
            else:
                if (np.sum(mask_freezeup) == 0): # Only continue while no prior freeze-up was detected for the sequence
                    if (~mask_tmp[im-1]):
                        sum_m = 0
                        if ~mask_tmp[im]:
                            sum_m = 0
                        else:
                            # start new group
                            sum_m +=1
                            istart = im
                            # Below will only occur if ndays is set to 1, e.g. first day of freezing temp.
                            if (sum_m >= ndays):
                                freezeup_date,freezeup_Tw,mask_freezeup = record_event(istart,time,year,Twater_in,freezeup_date,mask_freezeup,date_ref)
                    else:
                        if (mask_tmp[im]) & (istart > 0):
                            sum_m += 1
                            if (sum_m >= ndays):
                                freezeup_date,freezeup_Tw,mask_freezeup = record_event(istart,time,year,Twater_in,freezeup_date,mask_freezeup,date_ref)

        return freezeup_date, mask_freezeup


    if freezeup_opt == 1:
        def_opt = 1
        smooth_T =False; N_smooth = 3; mean_type='centered'
        round_T = False; round_type= 'half_unit'
        Gauss_filter = False
        sig_dog = 3.5
        T_thresh = 0.75
        dTdt_thresh = 0.25
        d2Tdt2_thresh = 0.25
        nd = 1
        no_negTw = False

    # FIND FREEZE-UP FROM SIMULATED SAMPLES
    nsamples_per_doy = np.zeros(len(start_doy_arr))*np.nan

    for istart in range(len(start_doy_arr)):
        month = month_istart[istart]
        day = day_istart[istart]
        month_ind = np.where((np.array([(date_ref+dt.timedelta(days=int(time_y[s,0]))).month for s in range(time_y.shape[0])]) == month))[0]
        day_ind = np.where((np.array([(date_ref+dt.timedelta(days=int(time_y[s,0]))).day for s in range(time_y.shape[0])]) == day))[0]
        istart_ind = np.sort(list( set(month_ind).intersection(day_ind) ))
        nsamples_per_doy[istart] = len(istart_ind)

    freezeup_dates_sample = np.zeros((len(start_doy_arr),int(np.nanmax(nsamples_per_doy)),4))*np.nan
    freezeup_dates_sample_doy = np.zeros((len(start_doy_arr),int(np.nanmax(nsamples_per_doy))))*np.nan

    freezeup_dates_target = np.zeros((len(start_doy_arr),int(np.nanmax(nsamples_per_doy)),4))*np.nan
    freezeup_dates_target_doy = np.zeros((len(start_doy_arr),int(np.nanmax(nsamples_per_doy))))*np.nan

    freezeup_dates_clim_target = np.zeros((len(start_doy_arr),int(np.nanmax(nsamples_per_doy)),4))*np.nan
    freezeup_dates_clim_target_doy = np.zeros((len(start_doy_arr),int(np.nanmax(nsamples_per_doy))))*np.nan

    for istart,start_doy in enumerate(start_doy_arr):

        if nsamples_per_doy[istart] > 0:
            # Find sample starting on the day & month of start date:
            month = month_istart[istart]
            day = day_istart[istart]
            month_ind = np.where((np.array([(date_ref+dt.timedelta(days=int(time_y[s,0]))).month for s in range(time_y.shape[0])]) == month))[0]
            day_ind = np.where((np.array([(date_ref+dt.timedelta(days=int(time_y[s,0]))).day for s in range(time_y.shape[0])]) == day))[0]
            istart_ind = np.sort(list( set(month_ind).intersection(day_ind) ))

            samples = y_pred[istart_ind]
            targets = y_true[istart_ind]
            clim_targets = y_clim[istart_ind]
            time_st = time_y[istart_ind]

            for s in range(samples.shape[0]):

                time = time_st[s,:]

                # FIND DTDt, D2Tdt2,etc. - SAMPLE
                Twater_sample = samples[s]

                Twater_tmp = Twater_sample.copy()
                Twater_dTdt_sample = np.zeros(Twater_sample.shape)*np.nan
                Twater_d2Tdt2_sample = np.zeros(Twater_sample.shape)*np.nan
                Twater_DoG1_sample = np.zeros(Twater_sample.shape)*np.nan
                Twater_DoG2_sample = np.zeros(Twater_sample.shape)*np.nan

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


                if (istart == 4):
                    if plot_samples:
                        plt.figure()
                        plt.title(istart_label[istart]+', '+str((dt.timedelta(days=int(time[0])) + date_ref).year))
                        plt.plot(Twater_target, color='black',label='Observed')
                        plt.plot(Twater_sample,label='Forecast')
                        plt.plot(Twater_clim_target,label='Climatology')
                        plt.plot(np.arange(len(Twater_clim_target)),np.ones(len(Twater_clim_target))*0.75,'--', color='gray')
                        plt.legend()


                # FIND FREEZE-UP FOR BOTH SAMPLE AND TARGET
                date_start = dt.timedelta(days=int(time[0])) + date_ref
                year = date_start.year

                if (year >= years[0]) & (year <= years[-1]):
                    iyr = np.where(years == year)[0][0]
                    fd_sample, mask_freeze_sample = find_freezeup_Tw_threshold(def_opt,Twater_tmp,time,thresh_T = T_thresh,ndays = nd)
                    fd_target, mask_freeze_target = find_freezeup_Tw_threshold(def_opt,Twater_tmp_target,time,thresh_T = T_thresh,ndays = nd)
                    fd_clim_target, mask_freeze_clim_target = find_freezeup_Tw_threshold(def_opt,Twater_tmp_clim_target,time,thresh_T = T_thresh,ndays = nd)

                    if (np.sum(mask_freeze_sample) > 0): # A freeze-up was detected in sample
                        if fd_sample[0] == year:
                            if calendar.isleap(years[iyr]):
                                freezeup_dates_sample[istart,s,0] = iyr
                                freezeup_dates_sample[istart,s,1:4] = fd_sample
                                freezeup_dates_sample_doy[istart,s]= (dt.date(int(fd_sample[0]),int(fd_sample[1]),int(fd_sample[2]))-dt.date(int(fd_sample[0]),1,1)).days
                            else:
                                freezeup_dates_sample[istart,s,0] = iyr
                                freezeup_dates_sample[istart,s,1:4] = fd_sample
                                freezeup_dates_sample_doy[istart,s]= (dt.date(int(fd_sample[0]),int(fd_sample[1]),int(fd_sample[2]))-dt.date(int(fd_sample[0]),1,1)).days+1
                        else:
                            if calendar.isleap(years[iyr]):
                                freezeup_dates_sample[istart,s,0] = iyr
                                freezeup_dates_sample[istart,s,1:4] = fd_sample
                                freezeup_dates_sample_doy[istart,s]= fd_sample[2]+365
                            else:
                                freezeup_dates_sample[istart,s,0] = iyr
                                freezeup_dates_sample[istart,s,1:4] = fd_sample
                                freezeup_dates_sample_doy[istart,s]= fd_sample[2]+365


                    if (np.sum(mask_freeze_target) > 0): # A freeze-up was detected in target
                        if fd_target[0] == year:
                            if calendar.isleap(years[iyr]):
                                freezeup_dates_target[istart,s,0] = iyr
                                freezeup_dates_target[istart,s,1:4] = fd_target
                                freezeup_dates_target_doy[istart,s]= (dt.date(int(fd_target[0]),int(fd_target[1]),int(fd_target[2]))-dt.date(int(fd_target[0]),1,1)).days
                            else:
                                freezeup_dates_target[istart,s,0] = iyr
                                freezeup_dates_target[istart,s,1:4] = fd_target
                                freezeup_dates_target_doy[istart,s]= (dt.date(int(fd_target[0]),int(fd_target[1]),int(fd_target[2]))-dt.date(int(fd_target[0]),1,1)).days+1
                        else:
                            if calendar.isleap(years[iyr]):
                                freezeup_dates_target[istart,s,0] = iyr
                                freezeup_dates_target[istart,s,1:4] = fd_target
                                freezeup_dates_target_doy[istart,s]= fd_target[2]+365
                            else:
                                freezeup_dates_target[istart,s,0] = iyr
                                freezeup_dates_target[istart,s,1:4] = fd_target
                                freezeup_dates_target_doy[istart,s]= fd_target[2]+365

                    if (np.sum(mask_freeze_clim_target) > 0): # A freeze-up was detected in climatology
                        if fd_clim_target[0] == year:
                            freezeup_dates_clim_target[istart,s,0] = iyr
                            if calendar.isleap(years[iyr]):
                                freezeup_dates_clim_target[istart,s,1] = fd_clim_target[0]
                                freezeup_dates_clim_target[istart,s,2] = fd_clim_target[1]
                                freezeup_dates_clim_target[istart,s,3] = fd_clim_target[2]+1
                                freezeup_dates_clim_target_doy[istart,s]= (dt.date(int(fd_clim_target[0]),int(fd_clim_target[1]),int(fd_clim_target[2]))-dt.date(int(fd_clim_target[0]),1,1)).days+1
                            else:
                                freezeup_dates_clim_target[istart,s,1:4] = fd_clim_target
                                freezeup_dates_clim_target_doy[istart,s]= (dt.date(int(fd_clim_target[0]),int(fd_clim_target[1]),int(fd_clim_target[2]))-dt.date(int(fd_clim_target[0]),1,1)).days+1


    # FIND MEAN FUD FROM Tw CLIM:
    mean_clim_FUD = np.nanmean(freezeup_dates_clim_target_doy)
    # if recalibrate:
    #     if offset_type == 'mean_clim':
    #         offset_forecasts = mean_clim_FUD
    #     freezeup_dates_sample_doy = freezeup_dates_sample_doy - offset_forecasts + mean_FUD_Longueuil_train


    return freezeup_dates_sample,freezeup_dates_sample_doy,freezeup_dates_target,freezeup_dates_target_doy,freezeup_dates_clim_target,freezeup_dates_clim_target_doy,mean_clim_FUD


def evaluate_FUD_forecasts(freezeup_dates_sample,freezeup_dates_sample_doy,freezeup_dates_target,freezeup_dates_target_doy,mean_clim_FUD,model_name,years,start_doy_arr,istart_label,plot_FUD_ts,plot_metrics,verbose=True):

    import statsmodels.api as sm

    # EVALUATE FREEZE-UP FORECAST ACCORDING TO SELECTED METRIC:
    fu_rmse = np.zeros((len(start_doy_arr),len(years)))*np.nan
    fu_acc = np.zeros((len(start_doy_arr),len(years)))*np.nan
    fu_acc7 = np.zeros((len(start_doy_arr),len(years)))*np.nan
    fu_mae = np.zeros((len(start_doy_arr),len(years)))*np.nan

    MAE_arr = np.zeros((len(start_doy_arr)))*np.nan
    RMSE_arr = np.zeros((len(start_doy_arr)))*np.nan
    Rsqr_arr = np.zeros((len(start_doy_arr)))*np.nan
    Rsqradj_arr = np.zeros((len(start_doy_arr)))*np.nan
    Acc_arr = np.zeros((len(start_doy_arr)))*np.nan
    Acc7_arr = np.zeros((len(start_doy_arr)))*np.nan

    for istart in range(len(start_doy_arr)):
        n = 0

        # The FUD is not detectabble because the forecast
        # length doesn't reach the FUD (or because there
        # were no samples for that period, e.g. because
        # of the presence of nans in the predictors time
        # series)

        # years_impossible =[]
        # tmp_arr = years.copy().astype('float')
        # for iyr in range(len(years)):
        #     if len(np.where(freezeup_dates_target[istart,:,0]==iyr)[0])>0:
        #         i_s = np.where(freezeup_dates_target[istart,:,0]==iyr)[0]
        #         if ~np.isnan(freezeup_dates_target[istart,i_s,0]):
        #             tmp_arr[iyr] = np.nan
        # years_impossible = years_impossible + tmp_arr[~np.isnan(tmp_arr)].tolist()

        # print(years_impossible)
        # years_impossible = np.array(years_impossible).astype('int')
        sample_freezeup_doy = freezeup_dates_sample_doy[istart,:]
        target_freezeup_doy = freezeup_dates_target_doy[istart,:]
        ts_doy = np.zeros(len(years))*np.nan
        ts_doy_obs = np.zeros(len(years))*np.nan

        # Evaluate the performance only on detectable FUDs
        if np.sum(~np.isnan(target_freezeup_doy)) > 0:

            for iyr in range(len(years)):
                if len(np.where(freezeup_dates_target[istart,:,0]==iyr)[0])>0:
                    i_st = np.where(freezeup_dates_target[istart,:,0]==iyr)[0]
                    fo_doy = target_freezeup_doy[i_st]
                    ts_doy_obs[iyr] = fo_doy
                    print

                    if (~np.isnan(target_freezeup_doy[i_st])):
                        n += 1
                        if len(np.where(freezeup_dates_sample[istart,:,0]==iyr)[0])>0:
                            i_ss = np.where(freezeup_dates_sample[istart,:,0]==iyr)[0][0]
                            fs_doy = sample_freezeup_doy[i_ss]
                            ts_doy[iyr] = fs_doy
                            fc_doy = mean_clim_FUD

                            # obs_cat = Longueuil_FUD_period_cat[iyr]
                            # if ~np.isnan(fs_doy):
                            #     if fs_doy <= tercile1_FUD_Longueuil_train:
                            #         sample_cat = -1
                            #     elif fs_doy > tercile2_FUD_Longueuil_train:
                            #         sample_cat = 1
                            #     else:
                            #         sample_cat = 0
                            #     if (sample_cat == obs_cat):
                            #         fu_acc[istart,iyr] = 1
                            #     else:
                            #         fu_acc[istart,iyr] = 0
                            # else:
                            #     fu_acc[istart,iyr] = np.nan

                            fu_rmse[istart,iyr] = (fs_doy-fo_doy)**2.
                            fu_mae[istart,iyr] = np.abs(fs_doy-fo_doy)
                            # print(istart,iyr,n,fo_doy,fs_doy,fc_doy,obs_cat,sample_cat,fu_acc[istart,iyr])

                        else:
                            # print(istart,iyr,'REPLACING FUD WITH CLIM...')
                            fs_doy = 355.0
                            ts_doy[iyr] = fs_doy

                            fu_rmse[istart,iyr] = (fs_doy-fo_doy)**2.
                            fu_mae[istart,iyr] = np.abs(fs_doy-fo_doy)

                            # fu_rmse[istart,iyr] = np.nan
                            # fu_mae[istart,iyr] = np.nan
                            # # fu_acc[istart,iyr] = np.nan

                            # # print(istart,iyr,n,Longueuil_FUD_period[iyr],np.nan,mean_clim_FUD,Longueuil_FUD_period_cat[iyr],np.nan,fu_acc[istart,iyr])

        # Here we don't use nanmean, but we nansum and divide by "n", where
        # "n" is the number of detectable FUDs for the given start date.
        # If an FUD was not forecasted/detected but it could have been,
        # then n will be larger than the available number of ML forecasts so the
        # performance is penalized.
        # print(istart,n,'Y')
        if np.all(np.isnan(fu_mae[istart,:])):
            MAE_arr[istart] = np.nan
        else:
            MAE_arr[istart] = np.nansum(fu_mae[istart,:])/n

        if np.all(np.isnan(fu_rmse[istart,:])):
            RMSE_arr[istart] = np.nan
        else:
            RMSE_arr[istart] = np.sqrt(np.nansum(fu_rmse[istart,:])/n)



        if (np.all(np.isnan(ts_doy))) | (np.all(np.isnan(ts_doy_obs))) :
            Rsqr_arr[istart] = np.nan
            Rsqradj_arr[istart] = np.nan
        else:
            model = sm.OLS(ts_doy_obs, sm.add_constant(ts_doy,has_constant='skip'), missing='drop').fit()
            Rsqr_arr[istart] = model.rsquared
            Rsqradj_arr[istart] = model.rsquared_adj


        # COMPUTE CATEGORICAL ACCURACY:
        obs_cat = np.array([0., 0.,  1., -1.,  1., -1.,  1.,  0., -1.,  1., -1., -1.,  0., -1.,1.,
                   -1., -1.,  0.,  0.,  1.,  0., -1.,  1.,  1.,  0., -1.,  0., 0.,])
        for iyr in range(len(years)):
            if ~np.isnan(ts_doy[iyr]):
                if ts_doy[iyr] <= 350.0:
                    sample_cat = -1
                elif ts_doy[iyr] > 360.0:
                    sample_cat = 1
                else:
                    sample_cat = 0
                if (sample_cat == obs_cat[iyr]):
                    fu_acc[istart,iyr] = 1
                else:
                    fu_acc[istart,iyr] = 0
            else:
                fu_acc[istart,iyr] = np.nan

        if  np.all(np.isnan(fu_acc[istart,:])):
            Acc_arr[istart] = np.nan
        else:
            Acc_arr[istart] = np.nansum(fu_acc[istart,:])/n

        # COMPUTE 7-day ACCURACY:
        for iyr in range(len(years)):
            if ~np.isnan(ts_doy[iyr]):
                if (np.abs(ts_doy[iyr]-ts_doy_obs[iyr]) <= 7):
                    fu_acc7[istart,iyr] = 1
                else:
                    fu_acc7[istart,iyr] = 0
            else:
                fu_acc7[istart,iyr] = np.nan

        if  np.all(np.isnan(fu_acc7[istart,:])):
            Acc7_arr[istart] = np.nan
        else:
            Acc7_arr[istart] = np.nansum(fu_acc7[istart,:])/n

    # PLOT FUD TIME SERIES
    fd_ML_forcast = np.zeros((5,len(years)))*np.nan
    if plot_FUD_ts:
        fig, ax = plt.subplots()
        # ax.plot(years,np.ones(len(years))*(mean_clim_FUD),color=plt.get_cmap('tab20c')(2))
        # ax.plot(years,np.ones(len(years))*(mean_FUD_Longueuil_train),color=[0.7,0.7,0.7])
        ax.plot(years,ts_doy_obs,'o-',color='black')

        for ic,istart in enumerate([0,1,2,3,4]):

            figi, axi = plt.subplots()
            # axi.plot(years,np.ones(len(years))*(mean_clim_FUD),color=plt.get_cmap('tab20c')(2))
            axi.plot(years,ts_doy_obs,'o-',color='black')

            for iyr in range(len(years)):
                select_yr_fud = freezeup_dates_sample[istart,np.where(freezeup_dates_sample[istart,:,0] == iyr)[0]]
                if select_yr_fud.shape[0] > 0:
                    fd_ML_forcast[istart,iyr] = freezeup_dates_sample_doy[istart,np.where(freezeup_dates_sample[istart,:,0] == iyr)[0]]
                else:
                    fd_ML_forcast[istart,iyr] = 355.0
            # print(fd_ML_forcast)
            if ic == 4:
                ax.plot(years,fd_ML_forcast[istart,:],'o:', color=plt.get_cmap('tab20c')(0), label= model_name + ' - '+istart_label[istart])
                axi.plot(years,fd_ML_forcast[istart,:],'o:', color=plt.get_cmap('tab20c')(0), label= model_name + ' - '+istart_label[istart])

            else:
                ax.plot(years,fd_ML_forcast[istart,:],'o:', color=plt.get_cmap('tab20c')(7-ic), label= model_name + ' - '+istart_label[istart])
                axi.plot(years,fd_ML_forcast[istart,:],'o:', color=plt.get_cmap('tab20c')(7-ic), label= model_name + ' - '+istart_label[istart])

            if ~np.all(np.isnan(fd_ML_forcast[istart,:])):
                model = sm.OLS(ts_doy_obs, sm.add_constant(fd_ML_forcast[istart,:],has_constant='skip'), missing='drop').fit()
                if verbose:
                    print('-----------------------------')
                    print('START DATE: ' + istart_label[istart])
                    print('Rsqr: ',model.rsquared, model.rsquared_adj)
                    print('MAE: ', MAE_arr[istart])
                    print('RMSE: ',RMSE_arr[istart])
                    print('Acc.: ',Acc_arr[istart])
                    print('FUD forecast: ', fd_ML_forcast[istart,:])

            # if recalibrate:
            #     if offset_type == 'mean_clim':
            #         plt.title('Recalibrated Forecasts - Mean clim\n'+'nlayers:'+str(nlayers)+', input window: '+str(inpw))
            # else:
            #     plt.title('Raw Forecasts\n'+'nlayers:'+str(nlayers)+', input window: '+str(inpw))
            plt.title('Raw Forecasts')

        ax.legend()


    return MAE_arr, RMSE_arr, Rsqr_arr, Rsqradj_arr, Acc_arr, Acc7_arr, fd_ML_forcast


def plot_Tw_metric(plot_Tw_metric,plot_Tw_clim_metric,metric_name,mname,vmin=0,vmax=0.9,vmin_diff=-0.95,vmax_diff=0.95):
    # cmap = cmocean.cm.tempo
    # cmap = cmocean.cm.dense
    # cmap = cmocean.cm.deep
    # cmap = cmocean.cm.solar
    # cmap = cmocean.cm.thermal
    # cmap = plt.get_cmap('cividis')
    cmap = plt.get_cmap('viridis')
    # cmap = plt.get_cmap('magma')
    # cmap = plt.get_cmap('inferno')

    fig, axs = plt.subplots(1, 1, figsize=(6,4))
    mappable = axs.pcolormesh(np.flipud(plot_Tw_metric), cmap=cmap, vmin=vmin, vmax=vmax)
    axs.set_title('$T_{w}$ '+metric_name+' (deg. C) - ' + mname)
    fig.colorbar(mappable, ax=[axs], location='left')
    for imonth in range(12):
        axs.text(62,11.35-imonth,str(np.nanmean(plot_Tw_metric[imonth,:])))

    fig_clim, axs_clim = plt.subplots(1, 1, figsize=(6,4))
    mappable = axs_clim.pcolormesh(np.flipud(plot_Tw_clim_metric), cmap=cmap, vmin=vmin, vmax=vmax)
    axs_clim.set_title('$T_{w}$ '+metric_name+' (deg. C) - Climatology')
    fig_clim.colorbar(mappable, ax=[axs_clim], location='left')

    cmap = cmocean.cm.balance
    fig_diff, axs_diff = plt.subplots(1, 1, figsize=(6,4))
    mappable = axs_diff.pcolormesh(np.flipud(plot_Tw_metric)-np.flipud(plot_Tw_clim_metric), cmap=cmap, vmin=vmin_diff, vmax=vmax_diff)
    axs_diff.set_title('$T_{w}$ '+metric_name+' diff. (deg. C)\n ' + mname+' - Climatology')
    fig_diff.colorbar(mappable, ax=[axs_diff], location='left')


def evaluate_Tw_forecasts(y_pred_in,y_in,yclim_in,time_in,pred_len_in,mname,metric_name = 'MAE', plot = False, date_ref = dt.date(1900,1,1) ):
    rsqr_Tw = np.zeros((12,pred_len_in))*np.nan; rsqr_Tw_clim = np.zeros((12,pred_len_in))*np.nan
    mae_Tw = np.zeros((12,pred_len_in))*np.nan; mae_Tw_clim = np.zeros((12,pred_len_in))*np.nan
    rmse_Tw = np.zeros((12,pred_len_in))*np.nan; rmse_Tw_clim = np.zeros((12,pred_len_in))*np.nan

    # for h in range(pred_len):
    for imonth in range(12):
        month = imonth+1
        samples_Tw = y_pred_in[np.where(np.array([(date_ref+dt.timedelta(days=int(time_in[s,0]))).month for s in range(time_in.shape[0])]) == month )[0]]
        targets_Tw = y_in[np.where(np.array([(date_ref+dt.timedelta(days=int(time_in[s,0]))).month for s in range(time_in.shape[0])]) == month )[0]]
        clim_Tw = yclim_in[np.where(np.array([(date_ref+dt.timedelta(days=int(time_in[s,0]))).month for s in range(time_in.shape[0])]) == month )[0]]
        if samples_Tw.shape[0] > 1:
            rsqr_Tw[imonth,:],mae_Tw[imonth,:],rmse_Tw[imonth,:] = regression_metrics(targets_Tw,samples_Tw)
            rsqr_Tw_clim[imonth,:], mae_Tw_clim[imonth,:], rmse_Tw_clim[imonth,:] =  regression_metrics(targets_Tw,clim_Tw)

            if month == 11:
                rsqr_nov,mae_nov,rmse_nov = regression_metrics(targets_Tw,samples_Tw,output_opt='uniform_average')
                # print('Nov. Tw. MAE (days): '+ str(mae_nov) )
                # print(np.nanmean(mae_Tw[imonth,:]))
            if month == 12:
                rsqr_dec,mae_dec,rmse_dec = regression_metrics(targets_Tw,samples_Tw,output_opt='uniform_average')
                # print('Dec. Tw. MAE (days): '+ str(mae_dec) )
                # print(np.nanmean(mae_Tw[imonth,:]))

    # Plot Tw forecast metrics
    if metric_name == 'Rsqr':
        if plot: plot_Tw_metric(rsqr_Tw,rsqr_Tw_clim,'R$^2$',mname)
        return rsqr_Tw,rsqr_Tw_clim
    if metric_name == 'MAE':
        if plot: plot_Tw_metric(mae_Tw,mae_Tw_clim,'MAE',mname)
        return mae_Tw,mae_Tw_clim
    if metric_name == 'RMSE':
        if plot: plot_Tw_metric(rmse_Tw,rmse_Tw_clim,'RMSE',mname)
        return rmse_Tw,rmse_Tw_clim




#%%
if __name__ == "__main__":

    # RUN OPTIONS:
    # suffix = ['_perfectexp','_perfectexp']#,'_perfectexp','_perfectexp']
    # pred_len = [60,60,60,60]
    # input_len = [128,128,128,128]
    # n_epochs = [50,50,50,50]
    # latent_dim = [25,25,25,25]
    # loss_name = ['MSETw','MSETw_MSEdTwdt','MSETw_with_weights10_on_thresh0_75','MSETw_MSEdTwdt_plus_snowfall']
    # dense_act_func_name = ['None','None','None','None']
    # norm_type=['Standard','Standard','Standard','Standard']
    # anomaly_target = [False, False, False, False]

    # suffix = ['_REF_MSETw']
    suffix = ['REF_MSETw_weigth_1_Tw_thresh_3']
    pred_len = [60]
    input_len = [128]
    n_epochs = [50]
    latent_dim = [50]
    nb_layers = [1]
    # loss_name = ['MSETw']
    loss_name = ['MSETw_with_weights1_on_thresh3_0_']
    dense_act_func_name = ['None']
    norm_type=['Standard']
    # anomaly_target = [True]
    anomaly_target = [False]


    suffix = ['_REF_MSETw','REF_MSETw','REF_MSETw_expdecay','REF_MSETw_expdecay']
    pred_len = [60,60,60,60]
    input_len = [128,128,128,128]
    n_epochs = [50,50,50,50]
    latent_dim = [50,50,50,50]
    nb_layers = [1,1,1,1]
    loss_name = ['MSETw','MSETw','MSETw_exp_decay_tau30_','MSETw_exp_decay_tau30']
    dense_act_func_name = ['None','None','None','None']
    norm_type=['Standard','Standard','Standard','Standard']
    anomaly_target = [False,True,False,True]


    suffix = ['REF_MSETw','REF_MSETw_dTw','REF_MSETw_expdecay','REF_MSETw_dTw']
    pred_len = [60,60,60,60]
    input_len = [128,128,128,128]
    n_epochs = [50,50,50,50]
    latent_dim = [50,50,50,50]
    nb_layers = [1,1,1,1]
    loss_name = ['MSETw','MSETw_MSEdTwdt','MSETw_exp_decay_tau30','MSETw_MSEdTwdt_exp_decay_tau30']
    dense_act_func_name = ['None','None','None','None']
    norm_type=['Standard','Standard','Standard','Standard']
    anomaly_target = [True,True,True,True]



    suffix = ['REF_MSETw_expdecay','REF_MSETw_dTw']
    pred_len = [60,60,60,60]
    input_len = [128,128,128,128]
    n_epochs = [50,50,50,50]
    latent_dim = [50,50,50,50]
    nb_layers = [1,1,1,1]
    loss_name = ['MSETw_exp_decay_tau30','MSETw_MSEdTwdt_exp_decay_tau30']
    dense_act_func_name = ['None','None','None','None']
    norm_type=['Standard','Standard','Standard','Standard']
    anomaly_target = [True,True,True,True]


    suffix = ['REF_MSETw_expdecay','REF_MSETw_dTw']
    pred_len = [60,60,60,60]
    input_len = [128,128,128,128]
    n_epochs = [50,50,50,50]
    latent_dim = [50,50,50,50]
    nb_layers = [1,1,1,1]
    loss_name = ['MSETw_exp_decay_tau30','MSETw_exp_decay_tau45']
    dense_act_func_name = ['None','None','None','None']
    norm_type=['Standard','Standard','Standard','Standard']
    anomaly_target = [True,True,True,True]


    suffix = ['REF_MSETw_expdecay','REF_MSETw_test_new_forecast_method_perf_frcst']
    pred_len = [60,60,60,60]
    input_len = [128,128,128,128]
    n_epochs = [50,50,50,50]
    latent_dim = [50,50,50,50]
    nb_layers = [1,1,1,1]
    loss_name = ['MSETw_exp_decay_tau30','MSETw_exp_decay_tau30']
    dense_act_func_name = ['None','None','None','None']
    norm_type=['Standard','Standard','Standard','Standard']
    anomaly_target = [True,True,True,True]


    # suffix = ['_perfectexp']#,'_perfectexp']#,'_perfectexp','_perfectexp']
    # pred_len = [60,60,60,60]
    # input_len = [128,128,128,128]
    # n_epochs = [50,50,50,50]
    # latent_dim = [50,25,25,25]
    # nb_layers = [1]
    # loss_name = ['MSETw','MSETw_MSEdTwdt','MSETw_with_weights10_on_thresh0_75','MSETw_MSEdTwdt_plus_snowfall']
    # dense_act_func_name = ['None','None','None','None']
    # norm_type=['Standard','Standard','Standard','Standard']
    # anomaly_target = [False, False, False, False]

    # suffix = ['_perfectexp']#,'_perfectexp','_perfectexp','_perfectexp','_perfectexp','_perfectexp','_perfectexp']#,'_perfectexp']
    # pred_len = [60,60,60,60,60,60,60]
    # input_len = [128,128,128,128,128,128,128]
    # n_epochs = [50,50,50,50,50,50,50]
    # # latent_dim = [25,25,25,25]
    # # latent_dim = [1,3,6,12,25,50,100]
    # latent_dim = [50]
    # # nb_layers = [1,2,4]
    # nb_layers = [1,1,1,1,1,1,1]
    # loss_name = ['MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75']
    # # loss_name = ['MSETw','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights10_on_thresh0_75']
    # # loss_name = ['MSETw','MSETw_MSEdTwdt','MSETw_MSEdTwdt_with_weights1_on_thresh0_75']
    # # loss_name = ['MSETw','MSETw_MSEdTwdt','MSETw_MSEdTwdt_plus_snowfall']
    # dense_act_func_name = ['None','None','None','None','None','None','None']
    # norm_type=['Standard','Standard','Standard','Standard','Standard','Standard','Standard']
    # anomaly_target = [False, False, False, False, False, False, False]


    # # suffix = ['_no_forecast_vars','_perfectexp_Ta_mean','_perfectexp_Ta_mean_cloud_cover_snowfall','_perfectexp_Ta_mean_cloud_cover_snowfall']
    # suffix = ['_perfectexp_Ta_mean']
    # # suffix = ['_perfectexp_Ta_mean_cloud_cover_snowfall']
    # pred_len = [60,60,60,60]
    # input_len = [128,128,128,128]
    # n_epochs = [100,100,100,100]
    # latent_dim = [50,50,50,25]
    # nb_layers = [1,1,1,1,1,1,1]
    # loss_name = ['MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75']
    # # loss_name = ['MSETw','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights10_on_thresh0_75']
    # # loss_name = ['MSETw','MSETw_MSEdTwdt','MSETw_MSEdTwdt_with_weights1_on_thresh0_75']
    # # loss_name = ['MSETw','MSETw_MSEdTwdt','MSETw_MSEdTwdt_plus_snowfall']
    # dense_act_func_name = ['None','None','None','None','None','None','None']
    # norm_type=['Standard','Standard','Standard','Standard','Standard','Standard','Standard']
    # anomaly_target = [False, False, False, False, False, False, False]

    # suffix = ['_perfectexp_Ta_mean_cloud_cover_snowfall','_perfectexp_Ta_mean_cloud_cover_snowfall','_perfectexp_Ta_mean_cloud_cover_snowfall','_perfectexp_Ta_mean_cloud_cover_snowfall',]
    # pred_len = [60,60,60,60]
    # input_len = [128,128,128,128]
    # n_epochs = [100,100,100,100]
    # latent_dim = [25,50,100,150]
    # nb_layers = [1,1,1,1,1,1,1]
    # loss_name = ['MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75']
    # # loss_name = ['MSETw','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights10_on_thresh0_75']
    # # loss_name = ['MSETw','MSETw_MSEdTwdt','MSETw_MSEdTwdt_with_weights1_on_thresh0_75']
    # # loss_name = ['MSETw','MSETw_MSEdTwdt','MSETw_MSEdTwdt_plus_snowfall']
    # dense_act_func_name = ['None','None','None','None','None','None','None']
    # norm_type=['Standard','Standard','Standard','Standard','Standard','Standard','Standard']
    # anomaly_target = [False, False, False, False, False, False, False]


    fpath = local_path + 'slice/prog/analysis/ML/encoderdecoder_src/'
    suffix = ['REF_perfectexp_Ta_mean_snowfall','REF_perfectexp_Ta_mean_snowfall_cloudcover','REF_perfectexp_Ta_mean_snowfall_dischargeSLR']
    # suffix = ['REF_perfectexp_Ta_mean_snowfall']
    pred_len = [60,60,60,60]
    input_len = [128,128,128,128]
    n_epochs = [50,50,50,50]
    latent_dim = [50,50,50,50]
    nb_layers = [1,1,1,1]
    loss_name = ['MSETw_MSEdTwdt_exp_decay_tau30','MSETw_MSEdTwdt_exp_decay_tau30','MSETw_MSEdTwdt_exp_decay_tau30']
    dense_act_func_name = ['None','None','None','None']
    norm_type=['Standard','Standard','Standard','Standard']
    anomaly_target = [True,True,True,True]


    # MEOPAR POSTER:
    fpath = ['./']
    suffix = ['perfectexp_Ta_mean']
    pred_len = [60,60,60,60]
    input_len = [128,128,128,128]
    n_epochs = [100,50,50,50]
    latent_dim = [50,50,50,50]
    nb_layers = [1,1,1,1]
    loss_name = ['MSETw_with_weights1_on_thresh0_75']
    dense_act_func_name = ['None','None','None','None']
    norm_type=['Standard','Standard','Standard','Standard']
    anomaly_target = [False,True,True,True]


    # REDO POSTER RUN WITH NEW PIPELINE:
    fpath = [local_path + 'slice/prog/analysis/ML/encoderdecoder_src/']
    suffix = ['redoPOSTER','redoPOSTER_nofluxes','redoPOSTER_noNAO','redoPOSTER_nosnowfall','redoPOSTER_nodischargeSLR']#,'only_Ta_mean_snowfall_pred_and_frcst']
    pred_len = [60,60,60,60,60]
    input_len = [128,128,128,128,128]
    n_epochs = [100,100,100,100,100]
    latent_dim = [50,50,50,50,50]
    nb_layers = [1,1,1,1,1]
    loss_name = ['MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75']
    dense_act_func_name = ['None','None','None','None','None']
    norm_type=['Standard','Standard','Standard','Standard','Standard']
    anomaly_target = [False,False,False,False,False]

    # EFFECTS OF FORECAST VARIABLES
    # fpath = [local_path + 'slice/prog/analysis/ML/encoderdecoder_src/']
    # suffix = ['REF_perfectexp_Ta_mean','REF_perfectexp_Ta_mean_snowfall','REF_perfectexp_Ta_mean_snowfall_cloudcover']
    # pred_len = [60,60,60,60]
    # input_len = [128,128,128,128]
    # n_epochs = [50,50,50,50]
    # latent_dim = [50,50,50,50]
    # nb_layers = [1,1,1,1]
    # loss_name = ['MSETw_MSEdTwdt_exp_decay_tau30','MSETw_MSEdTwdt_exp_decay_tau30','MSETw_MSEdTwdt_exp_decay_tau30']
    # dense_act_func_name = ['None','None','None','None']
    # norm_type=['Standard','Standard','Standard','Standard']
    # anomaly_target = [True,True,True]


    # EFFECTS OF LOSS SCHEME AND ANOMALY
    # fpath = [local_path + 'slice/prog/analysis/ML/encoderdecoder_src/']
    # suffix = ['with_POSTERpredictors','redoPOSTER']
    # pred_len = [60,60,60,60]
    # input_len = [128,128,128,128]
    # n_epochs = [100,100,50,50]
    # latent_dim = [50,50,50,50]
    # nb_layers = [1,1,1,1]
    # loss_name = ['MSETw_exp_decay_tau30','MSETw_with_weights1_on_thresh0_75']
    # dense_act_func_name = ['None','None','None','None']
    # norm_type=['Standard','Standard','Standard','Standard']
    # anomaly_target = [True,False,True]

    # NEW RUNS WITH SEAS5 ENSEMBLE MEAN FORECASTS
    fpath = [local_path + 'slice/prog/analysis/ML/encoderdecoder_src/']
    suffix = ['redoPOSTER','PFE','RWE','RWE','RWE']#,'only_Ta_mean_snowfall_pred_and_frcst']
    pred_len = [60,60,60,30,14]
    input_len = [128,128,128,128,128]
    n_epochs = [100,100,100,100,100]
    latent_dim = [50,50,50,50,50]
    nb_layers = [1,1,1,1,1]
    loss_name = ['MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75']
    dense_act_func_name = ['None','None','None','None','None']
    norm_type=['Standard','Standard','Standard','Standard','Standard']
    anomaly_target = [False,False,False,False,False]


    # fpath = ['./',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/']
    # suffix = ['perfectexp_Ta_mean','redoPOSTER']
    fpath = [local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/']
    suffix = ['reducelrexp_0_05_lr0_004_PFE','reducelrexp_0_05_lr0_004_PFE','reducelrexp_0_05_lr0_004_RWE','reducelrexp_0_05_lr0_004_RWE']
    pred_len = [60,60,60,60]
    input_len = [128,128,128,128]
    n_epochs = [100,100,100,100]
    latent_dim = [50,50,50,50]
    nb_layers = [1,1,1,1]
    loss_name = ['MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75']
    dense_act_func_name = ['None','sigmoid','None','sigmoid']
    norm_type=['Standard','MinMax','Standard','MinMax']
    anomaly_target = [False,False,False,False]


    # fpath = [local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/']
    # suffix = ['reducelrexp_0_05_lr0_004_RWE','reducelrexp_0_05_lr0_004_RWE_redo']
    # pred_len = [60,60,60,60]
    # input_len = [128,128,128,128]
    # n_epochs = [100,100,100,100]
    # latent_dim = [50,50,50,50]
    # nb_layers = [1,1,1,1]
    # loss_name = ['MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75']
    # dense_act_func_name = ['None','None']
    # norm_type=['Standard','Standard']
    # anomaly_target = [False,False,False,False]

    # fpath = [local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/']
    # suffix = ['reducelrexp_0_05_lr0_004_PFE','reducelrexp_0_05_lr0_004_RWE','reducelrexp_0_05_lr0_004_RWE_redo','reducelrexp_0_05_lr0_004_RWE_allmembers']
    # pred_len = [60,60,60,60]
    # input_len = [128,128,128,128]
    # n_epochs = [100,100,100,100]
    # latent_dim = [50,50,50,50]
    # nb_layers = [1,1,1,1]
    # loss_name = ['MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75']
    # dense_act_func_name = ['None','None','None','None']
    # norm_type=['Standard','Standard','Standard','Standard']
    # anomaly_target = [False,False,False,False]

    fpath = [local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/']
    suffix = ['reducelrexp_0_05_lr0_004_PFE','reducelrexp_0_05_lr0_004_RWE']
    pred_len = [60,60,60,60]
    input_len = [128,128,128,128]
    n_epochs = [100,100,100,100]
    latent_dim = [50,50,50,50]
    nb_layers = [1,1,1,1]
    loss_name = ['MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75','MSETw_with_weights1_on_thresh0_75']
    dense_act_func_name = ['None','None','None','None']
    norm_type=['Standard','Standard','Standard','Standard']
    anomaly_target = [False,False,False,False]

    # fpath = [local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/',local_path + 'slice/prog/analysis/ML/encoderdecoder_src/']
    # # suffix = ['reducelrexp_0_05_lr0_004_PFE','lr0_0005_nocallbacks_Tafcstonly_PFE']
    # suffix = ['reducelrexp_0_05_lr0_004_RWE','lr0_0005_nocallbacks_Tafcstonly_RWE_ensemblemean','lr0_0005_nocallbacks_Tafcstonly_RWE_allmembers']
    # pred_len = [60,60,60,60]
    # input_len = [128,128,128,128]
    # n_epochs = [100,100,100,100]
    # latent_dim = [50,10,10,10]
    # nb_layers = [1,1,1,1]
    # loss_name = ['MSETw_with_weights1_on_thresh0_75','MSETw_MSEdTwdt_exp_decay_tau30','MSETw_MSEdTwdt_exp_decay_tau30','MSETw_MSEdTwdt_exp_decay_tau30']
    # dense_act_func_name = ['None','None','None','None']
    # norm_type=['Standard','Standard','Standard','Standard']
    # anomaly_target = [False,True,True,True]

    #-----------------
    pred_type = 'test'
    # pred_type = 'valid'

    valid_scheme = 'LOOk'
    # valid_scheme = 'standard'

    #-----------------
    # save_FUD_metrics = False
    save_FUD_metrics = True

    #-----------------
    plot_series = False
    evaluate_Twater = False
    evaluate_FUD = True
    plot_samples = False
    #-----------------

    freezeup_opt = 1
    start_doy_arr =[307,        314,         321,         328,         335        ]
    istart_label = ['Nov. 3rd', 'Nov. 10th', 'Nov. 17th', 'Nov. 24th', 'Dec. 1st' ]
    # month_istart = [11,11,11,11,12]
    # day_istart   = [ 3,10,17,24, 1]

    # start_doy_arr =[307,        314,         321,         328,         335,         342        ]
    # istart_label = ['Nov. 3rd', 'Nov. 10th', 'Nov. 17th', 'Nov. 24th', 'Dec. 1st', 'Dec. 8th' ]
    month_istart = [11,11,11,11,12,12]
    day_istart   = [ 3,10,17,24, 1, 8]
    date_ref = dt.date(1900,1,1)
    years_eval = np.arange(1992,2020)

    #-----------------
    nruns = len(suffix)
    test_years = np.arange(1993,2020)
    # test_years = np.arange(1993,2007)

    #-----------------
    istart_plot = [0,1,2,3,4]
    letter = ['a)','b)','c)','d)','e)','f)']
    # istart_plot = [4]
    # letter = ['']
    # istart_plot = [0,1,2,3,4,5]
    # letter = ['','','','','','']

    #-----------------
    # Initialize plots
    if valid_scheme == 'standard':
        fig_mae_ts,ax_mae_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
        fig_mae_ss,ax_mae_ss = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
        fig_rsqr_ts,ax_rsqr_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
        fig_rmse_ts,ax_rmse_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
        fig_acc_ts,ax_acc_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
        fig_acc7_ts,ax_acc7_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
        fig_mae_ss,ax_mae_ss = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))

    if valid_scheme == 'LOOk':
        fig_mae_CV,ax_mae_CV = plt.subplots(nrows = 1, ncols = len(istart_plot),figsize=(2.5*nruns,3),sharex=True,sharey=True)
        fig_rsqr_CV,ax_rsqr_CV = plt.subplots(nrows = 1, ncols = len(istart_plot),figsize=(2.5*nruns,3),sharex=True,sharey=True)
        fig_mae_SS,ax_mae_SS = plt.subplots(nrows = 1, ncols = len(istart_plot),figsize=(2.5*nruns,3),sharex=True,sharey=True)

        # ax_mae_CV.grid(axis='y')
        # ax_rsqr_CV.grid(axis='y')
        # ax_mae_SS.grid(axis='y')

        if (pred_type == 'test') | (pred_type == 'train'):
            fig_mae_ts,ax_mae_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
            fig_rsqr_ts,ax_rsqr_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
            fig_rmse_ts,ax_rmse_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
            fig_acc_ts,ax_acc_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
            fig_acc7_ts,ax_acc7_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
            fig_mae_ss,ax_mae_ss = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))

    #-----------------
    # LOOP ON ALL RUNS:
    for irun in range(nruns):

        # LOAD DATA
        if (pred_type == 'test'):
            target_time = np.zeros((len(test_years),10220,pred_len[irun]))*np.nan
            y_pred = np.zeros((len(test_years),10220,pred_len[irun]))*np.nan
            y = np.zeros((len(test_years),10220,pred_len[irun]))*np.nan
            y_clim = np.zeros((len(test_years),10220,pred_len[irun]))*np.nan

        if (pred_type == 'valid'):
            target_time = np.zeros((len(test_years),5,10220,pred_len[irun]))*np.nan
            y_pred = np.zeros((len(test_years),5,10220,pred_len[irun]))*np.nan
            y = np.zeros((len(test_years),5,10220,pred_len[irun]))*np.nan
            y_clim = np.zeros((len(test_years),5,10220,pred_len[irun]))*np.nan


        if valid_scheme == 'standard':
            [fname,target_time,
              y_pred, y, y_clim,
              valid_scheme, nfolds] = load_data_year(pred_type,pred_len[irun],input_len[irun],latent_dim[irun],n_epochs[irun],nb_layers[irun],norm_type[irun],dense_act_func_name[irun],loss_name[irun],anomaly_target[irun],suffix[irun],fpath = fpath[irun])

        if valid_scheme == 'LOOk':
            for iyr,yr_test in enumerate(test_years):
                if yr_test != 2020:
                # if yr_test < 2017:
                    [fname,target_time_yr,
                      y_pred_yr, y_yr, y_clim_yr,
                      valid_scheme, nfolds] = load_data_year(pred_type,pred_len[irun],input_len[irun],latent_dim[irun],n_epochs[irun],nb_layers[irun],norm_type[irun],dense_act_func_name[irun],loss_name[irun],anomaly_target[irun],suffix[irun],yr_test=yr_test,fpath = fpath[irun])

                    target_time[iyr,0:y_yr.shape[0],:] = target_time_yr
                    y_pred[iyr,0:y_yr.shape[0],:] = y_pred_yr
                    y[iyr,0:y_yr.shape[0],:] = y_yr
                    y_clim[iyr,0:y_yr.shape[0],:] = y_clim_yr

        #-----------------
        # PLOT PREDICTION TIME SERIES
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
                            # plot_prediction_timeseries(y_pred_in,y_in,y_clim_in,time_in,pred_type,lead=0,nyrs_plot= 28, year_in = test_years[iyr])
                            plot_prediction_timeseries(y_pred_in,y_in,y_clim_in,time_in,pred_type,lead=50,nyrs_plot= 28, year_in = test_years[iyr])
                            # plot_sample(y_pred_in,y_in,y_clim_in,time_in,it=116,pred_type=pred_type,show_clim=True)


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

        #-----------------
        # EVALUATE TWATER FORECASTS
        if evaluate_Twater:

            if valid_scheme == 'LOOk':
                if (pred_type == 'test') | (pred_type == 'train') :
                    model_name = 'Encoder-Decoder LSTM - ' + pred_type
                    Tw_MAE_yr = np.zeros((12,pred_len[irun],len(test_years)))*np.nan
                    Tw_clim_MAE_yr = np.zeros((12,pred_len[irun],len(test_years)))*np.nan
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
                            Tw_MAE_yr[:,:,iyr], Tw_clim_MAE_yr[:,:,iyr] = evaluate_Tw_forecasts(y_pred_in,y_in,y_clim_in,time_in,pred_len[irun],model_name)
                    Tw_MAE = np.nanmean(Tw_MAE_yr,axis=2)
                    Tw_clim_MAE = np.nanmean(Tw_clim_MAE_yr,axis=2)
                    Tw_MAE_std = np.nanstd(Tw_MAE_yr,axis=2)
                    Tw_clim_MAE_std = np.nanstd(Tw_clim_MAE_yr,axis=2)
                    plot_Tw_metric(Tw_MAE,Tw_clim_MAE,'MAE',model_name)
                    plot_Tw_metric(Tw_MAE_std,Tw_clim_MAE_std,'MAE',model_name,vmin=0,vmax=0.5)

                if pred_type == 'valid' :
                    model_name = 'Encoder-Decoder LSTM - ' + pred_type
                    show_all_folds = False
                    Tw_MAE_fold = np.zeros((12,pred_len[irun],len(test_years),nfolds))*np.nan
                    Tw_clim_MAE_fold = np.zeros((12,pred_len[irun],len(test_years),nfolds))*np.nan
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
                                Tw_MAE_fold[:,:,iyr,ifold], Tw_clim_MAE_fold[:,:,iyr,ifold] = evaluate_Tw_forecasts(y_pred_in,y_in,y_clim_in,time_in,pred_len[irun],model_name)
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
                Tw_MAE, Tw_clim_MAE = evaluate_Tw_forecasts(y_pred,y,y_clim,target_time,pred_len[irun],model_name)
                plot_Tw_metric(Tw_MAE,Tw_clim_MAE,'MAE_mean',model_name)

        #-----------------
        # EVALUATE FUD FORECASTS
        if evaluate_FUD:

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
                MAE_arr, RMSE_arr, Rsqr_arr, Rsqr_adj_arr, Acc_arr, Acc7_arr,_ = evaluate_FUD_forecasts(freezeup_dates_sample,freezeup_dates_sample_doy,freezeup_dates_target,freezeup_dates_target_doy,mean_clim_FUD,model_name,years_eval,start_doy_arr,istart_label,plot_FUD_ts,plot_metrics,verbose=True)
                plot_FUD_ts = False
                plot_metrics = False
                MAE_arr_clim, RMSE_arr_clim, Rsqr_arr_clim, Rsqr_adj_arr_clim, Acc_arr_clim, Acc7_arr_clim,_ = evaluate_FUD_forecasts(freezeup_dates_clim_target,freezeup_dates_clim_target_doy,freezeup_dates_target,freezeup_dates_target_doy,mean_clim_FUD,model_name,years_eval,start_doy_arr,istart_label,plot_FUD_ts,plot_metrics,verbose=True)

                ax_mae_ts.plot(np.arange(len(istart_plot)),MAE_arr,'o-',label='Model')
                ax_mae_ts.plot(np.arange(len(istart_plot)),MAE_arr_clim,'x-',label='Climatology')
                ax_mae_ts.set_ylabel('MAE (days)')
                ax_mae_ts.set_xlabel('Forecast start date')
                ax_mae_ts.set_xticks(np.arange(len(istart_plot)))
                ax_mae_ts.set_xticklabels([istart_label[k] for k in range(len(istart_plot))])
                ax_mae_ts.legend()

                ax_rmse_ts.plot(np.arange(len(istart_plot)),RMSE_arr,'o-',label='Model')
                ax_rmse_ts.plot(np.arange(len(istart_plot)),RMSE_arr_clim,'x-',label='Climatology')
                ax_rmse_ts.set_ylabel('RMSE (days)')
                ax_rmse_ts.set_xlabel('Forecast start date')
                ax_rmse_ts.set_xticks(np.arange(len(istart_plot)))
                ax_rmse_ts.set_xticklabels([istart_label[k] for k in range(len(istart_plot))])
                ax_rmse_ts.legend()

                ax_rsqr_ts.plot(np.arange(len(istart_plot)),Rsqr_arr,'o-',label='Model')
                ax_rsqr_ts.plot(np.arange(len(istart_plot)),Rsqr_arr_clim,'x-',label='Climatology')
                ax_rsqr_ts.set_ylabel('Rsqr')
                ax_rsqr_ts.set_xlabel('Forecast start date')
                ax_rsqr_ts.set_xticks(np.arange(len(istart_plot)))
                ax_rsqr_ts.set_xticklabels([istart_label[k] for k in range(len(istart_plot))])
                ax_rsqr_ts.legend()

                ax_acc7_ts.plot(np.arange(len(istart_plot)),Acc7_arr,'o-',label='Model')
                ax_acc7_ts.plot(np.arange(len(istart_plot)),Acc7_arr_clim,'x-',label='Climatology')
                ax_acc7_ts.set_ylabel('7-day Accuracy')
                ax_acc7_ts.set_xlabel('Forecast start date')
                ax_acc7_ts.set_xticks(np.arange(len(istart_plot)))
                ax_acc7_ts.set_xticklabels([istart_label[k] for k in range(len(istart_plot))])
                ax_acc7_ts.legend()

                ax_mae_ss.plot(np.arange(len(istart_plot)),1-(MAE_arr/MAE_arr_clim),'o-')
                ax_mae_ss.set_ylabel('Skill Score (MAE)')
                ax_mae_ss.set_xlabel('Forecast start date')
                ax_mae_ss.set_xticks(np.arange(len(istart_plot)))
                ax_mae_ss.set_xticklabels([istart_label[k] for k in range(len(istart_plot))])


            if valid_scheme == 'LOOk':

                if (pred_type == 'test') | (pred_type == 'train'):

                    MAE_arr = np.zeros((nruns, len(start_doy_arr)))*np.nan
                    RMSE_arr = np.zeros((nruns, len(start_doy_arr)))*np.nan
                    Rsqr_arr = np.zeros((nruns, len(start_doy_arr)))*np.nan
                    Rsqr_adj_arr = np.zeros((nruns, len(start_doy_arr)))*np.nan
                    Acc_arr = np.zeros((nruns, len(start_doy_arr)))*np.nan
                    Acc7_arr = np.zeros((nruns, len(start_doy_arr)))*np.nan
                    FUD_frcst_arr = np.zeros((nruns, len(start_doy_arr),len(years_eval)))*np.nan

                    MAE_arr_clim = np.zeros((nruns, len(start_doy_arr)))*np.nan
                    RMSE_arr_clim = np.zeros((nruns, len(start_doy_arr)))*np.nan
                    Rsqr_arr_clim = np.zeros((nruns, len(start_doy_arr)))*np.nan
                    Rsqr_adj_arr_clim = np.zeros((nruns, len(start_doy_arr)))*np.nan
                    Acc_arr_clim = np.zeros((nruns, len(start_doy_arr)))*np.nan
                    Acc7_arr_clim = np.zeros((nruns, len(start_doy_arr)))*np.nan
                    FUD_frcst_arr_clim = np.zeros((nruns,  len(start_doy_arr),len(years_eval)))*np.nan

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
                        MAE_arr[irun,:], RMSE_arr[irun,:], Rsqr_arr[irun,:], Rsqr_adj_arr[irun,:], Acc_arr[irun,:], Acc7_arr[irun,:], FUD_frcst_arr[irun,:,:] = evaluate_FUD_forecasts(freezeup_dates_sample,freezeup_dates_sample_doy,freezeup_dates_target,freezeup_dates_target_doy,mean_clim_FUD,model_name,years_eval,start_doy_arr,istart_label,plot_FUD_ts,plot_metrics,verbose=True)
                        plot_FUD_ts = False
                        plot_metrics = False
                        MAE_arr_clim[irun,:], RMSE_arr_clim[irun,:], Rsqr_arr_clim[irun,:], Rsqr_adj_arr_clim[irun,:], Acc_arr_clim[irun,:], Acc7_arr_clim[irun,:], FUD_frcst_arr_clim[irun,:,:] = evaluate_FUD_forecasts(freezeup_dates_clim_target,freezeup_dates_clim_target_doy,freezeup_dates_target,freezeup_dates_target_doy,mean_clim_FUD,model_name,years_eval,start_doy_arr,istart_label,plot_FUD_ts,plot_metrics,verbose=True)


                    ax_mae_ts.plot(np.arange(len(istart_plot)),MAE_arr[irun,istart_plot],'o-',label='Model')
                    # ax_mae_ts.plot(np.arange(len(istart_plot)),MAE_arr_clim[irun,istart_plot],'x-',label='Climatology')
                    ax_mae_ts.set_ylabel('MAE (days)')
                    ax_mae_ts.set_xlabel('Forecast start date')
                    ax_mae_ts.set_xticks(np.arange(len(istart_plot)))
                    ax_mae_ts.set_xticklabels([istart_label[k] for k in istart_plot])
                    ax_mae_ts.legend()

                    ax_mae_ss.plot(np.arange(len(istart_plot)),1-(MAE_arr[irun,istart_plot]/MAE_arr_clim[irun,istart_plot]),'o-')
                    ax_mae_ss.set_ylabel('Skill Score (MAE)')
                    ax_mae_ss.set_xlabel('Forecast start date')
                    ax_mae_ss.set_xticks(np.arange(len(istart_plot)))
                    ax_mae_ss.set_xticklabels([istart_label[k] for k in istart_plot])

                    ax_rmse_ts.plot(np.arange(len(istart_plot)),RMSE_arr[irun,istart_plot],'o-',label='Model')
                    # ax_rmse_ts.plot(np.arange(len(istart_plot)),RMSE_arr_clim[irun,istart_plot],'x-',label='Climatology')
                    ax_rmse_ts.set_ylabel('RSME (days)')
                    ax_rmse_ts.set_xlabel('Forecast start date')
                    ax_rmse_ts.set_xticks(np.arange(len(istart_plot)))
                    ax_rmse_ts.set_xticklabels([istart_label[k] for k in istart_plot])
                    ax_rmse_ts.legend()

                    ax_rsqr_ts.plot(np.arange(len(istart_plot)),Rsqr_arr[irun,istart_plot],'o-',label='Model')
                    # ax_rsqr_ts.plot(np.arange(len(istart_plot)),Rsqr_arr_clim[irun,istart_plot],'x-',label='Climatology')
                    ax_rsqr_ts.set_ylabel('R$^{2}$')
                    ax_rsqr_ts.set_xlabel('Forecast start date')
                    ax_rsqr_ts.set_xticks(np.arange(len(istart_plot)))
                    ax_rsqr_ts.set_xticklabels([istart_label[k] for k in istart_plot])
                    ax_rsqr_ts.legend()

                    ax_acc_ts.plot(np.arange(len(istart_plot)),Acc_arr[irun,istart_plot],'o-',label='Model')
                    # ax_acc_ts.plot(np.arange(len(istart_plot)),Acc_arr_clim[istart_plot],'x-',label='Climatology')
                    ax_acc_ts.set_ylabel('Categorical Accuracy (%)')
                    ax_acc_ts.set_xlabel('Forecast start date')
                    ax_acc_ts.set_xticks(np.arange(len(istart_plot)))
                    ax_acc_ts.set_xticklabels([istart_label[k] for k in istart_plot])
                    ax_acc_ts.legend()

                    ax_acc7_ts.plot(np.arange(len(istart_plot)),Acc7_arr[irun,istart_plot],'o-',label='Model')
                    ax_acc7_ts.plot(np.arange(len(istart_plot)),Acc7_arr_clim[irun,istart_plot],'x-',label='Climatology')
                    ax_acc7_ts.set_ylabel('7-day Accuracy (%)')
                    ax_acc7_ts.set_xlabel('Forecast start date')
                    ax_acc7_ts.set_xticks(np.arange(len(istart_plot)))
                    ax_acc7_ts.set_xticklabels([istart_label[k] for k in istart_plot])
                    ax_acc7_ts.legend()

                    if save_FUD_metrics:
                        np.savez('./metrics/'+fname+'_TEST_METRICS',
                                MAE_arr = MAE_arr,
                                RMSE_arr = RMSE_arr,
                                Rsqr_arr = Rsqr_arr,
                                Rsqr_adj_arr = Rsqr_adj_arr,
                                Acc_arr = Acc_arr,
                                Acc7_arr = Acc7_arr,
                                FUD_frcst_arr = FUD_frcst_arr,
                                MAE_arr_clim = MAE_arr_clim,
                                RMSE_arr_clim = RMSE_arr_clim,
                                Rsqr_arr_clim = Rsqr_arr_clim,
                                Rsqr_adj_arr_clim = Rsqr_adj_arr_clim,
                                Acc_arr_clim = Acc_arr_clim,
                                Acc7_arr_clim = Acc7_arr_clim,
                                FUD_frcst_arr_clim = FUD_frcst_arr_clim,
                                )


                if pred_type == 'valid':
                    SS_MAE_arr_valid = np.zeros((len(start_doy_arr),len(test_years),nfolds))*np.nan
                    MAE_arr_valid = np.zeros((len(start_doy_arr),len(test_years),nfolds))*np.nan
                    RMSE_arr_valid = np.zeros((len(start_doy_arr),len(test_years),nfolds))*np.nan
                    Rsqr_arr_valid = np.zeros((len(start_doy_arr),len(test_years),nfolds))*np.nan
                    Rsqr_adj_arr_valid = np.zeros((len(start_doy_arr),len(test_years),nfolds))*np.nan
                    Acc_arr_valid = np.zeros((len(start_doy_arr),len(test_years),nfolds))*np.nan
                    Acc7_arr_valid = np.zeros((len(start_doy_arr),len(test_years),nfolds))*np.nan
                    MAE_arr_clim_valid = np.zeros((len(start_doy_arr),len(test_years),nfolds))*np.nan
                    RMSE_arr_clim_valid = np.zeros((len(start_doy_arr),len(test_years),nfolds))*np.nan
                    Rsqr_arr_clim_valid = np.zeros((len(start_doy_arr),len(test_years),nfolds))*np.nan
                    Rsqr_adj_arr_clim_valid = np.zeros((len(start_doy_arr),len(test_years),nfolds))*np.nan
                    Acc_arr_clim_valid = np.zeros((len(start_doy_arr),len(test_years),nfolds))*np.nan
                    Acc7_arr_clim_valid = np.zeros((len(start_doy_arr),len(test_years),nfolds))*np.nan
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

                            if (iyr == 2) & (ifold == 1):
                                plot_samples = True
                            else:
                                plot_samples = False

                            [freezeup_dates_sample_fold,freezeup_dates_sample_doy_fold,
                              freezeup_dates_target_fold,freezeup_dates_target_doy_fold,
                              freezeup_dates_clim_target_fold,freezeup_dates_clim_target_doy_fold,
                              mean_clim_FUD]  = detect_FUD_from_Tw_samples(y_true_fold,y_pred_fold,y_clim_fold,time_y_fold,years_eval,freezeup_opt,start_doy_arr,istart_label,day_istart,month_istart,plot_samples)

                            if freezeup_dates_sample_fold.shape[1] > 0:
                                plot_FUD_ts = False
                                plot_metrics = False
                                MAE_arr_valid[:,iyr,ifold], RMSE_arr_valid[:,iyr,ifold], Rsqr_arr_valid[:,iyr,ifold], Rsqr_adj_arr_valid[:,iyr,ifold], Acc_arr_valid[:,iyr,ifold], Acc7_arr_valid[:,iyr,ifold], _ = evaluate_FUD_forecasts(freezeup_dates_sample_fold,freezeup_dates_sample_doy_fold,freezeup_dates_target_fold,freezeup_dates_target_doy_fold,mean_clim_FUD,model_name,years_eval,start_doy_arr,istart_label,plot_FUD_ts,plot_metrics,verbose=True)
                                plot_FUD_ts = False
                                plot_metrics = False
                                MAE_arr_clim_valid[:,iyr,ifold], RMSE_arr_clim_valid[:,iyr,ifold], Rsqr_arr_clim_valid[:,iyr,ifold], Rsqr_adj_arr_clim_valid[:,iyr,ifold], Acc_arr_clim_valid[:,iyr,ifold], Acc7_arr_clim_valid[:,iyr,ifold], _ = evaluate_FUD_forecasts(freezeup_dates_clim_target_fold,freezeup_dates_clim_target_doy_fold,freezeup_dates_target_fold,freezeup_dates_target_doy_fold,mean_clim_FUD,model_name,years_eval,start_doy_arr,istart_label,plot_FUD_ts,plot_metrics,verbose=True)
                                SS_MAE_arr_valid[:,iyr,ifold] = 1-(MAE_arr_valid[:,iyr,ifold]/MAE_arr_clim_valid[:,iyr,ifold])




                    if len(istart_plot) > 1:
                        for i,istart in enumerate(istart_plot):
                            for iyr in range(len(test_years)):
                                ax_mae_CV[i].plot(test_years[iyr]+irun*30,np.nanmean(MAE_arr_valid[istart,iyr,:]),'x',color=plt.get_cmap('tab20c')(2*irun+1))
                                ax_mae_CV[i].plot([test_years[iyr]+irun*30,test_years[iyr]+irun*30],[np.nanmin(MAE_arr_valid[istart,iyr,:]),np.nanmax(MAE_arr_valid[istart,iyr,:])],'-',linewidth=0.5,color=plt.get_cmap('tab20c')(2*irun+1))
                            ax_mae_CV[i].plot(test_years+irun*30,np.ones(len(test_years))*np.nanmean(MAE_arr_valid[istart,:,:],axis=(0,1)),'-',color=plt.get_cmap('tab20c')(2*irun))
                            # !!!! ADD OPTION HERE TO ADD TEST RESULTS AS WELL FOR EACH START DATE.
                            ax_mae_CV[i].set_xlabel('Years')
                            # ax_mae_CV[i].set_xticks(test_years+irun*30)
                            # ax_mae_CV[i].set_xticklabels(['' for k in range(len(test_years))])
                            # ax_mae_CV[i].set_xticklabels([str(test_years[k]) for k in range(len(test_years))])
                            # ax_mae_CV[i].legend()
                            ax_mae_CV[i].set_title(letter[i]+' '+istart_label[istart])
                            ax_mae_CV[0].set_ylabel('MAE (days)')
                            print(np.nanmean(MAE_arr_valid[istart,:,:],axis=(0,1)))

                            for iyr in range(len(test_years)):
                                ax_rsqr_CV[i].plot(test_years[iyr]+irun*30,np.nanmean(Rsqr_arr_valid[istart,iyr,:]),'x',color=plt.get_cmap('tab20')(2*irun+1))
                                ax_rsqr_CV[i].plot([test_years[iyr]+irun*30,test_years[iyr]+irun*30],[np.nanmin(Rsqr_arr_valid[istart,iyr,:]),np.nanmax(Rsqr_arr_valid[istart,iyr,:])],'-',linewidth=0.5,color=plt.get_cmap('tab20')(2*irun+1))
                            ax_rsqr_CV[i].plot(test_years+irun*30,np.ones(len(test_years))*np.nanmean(Rsqr_arr_valid[istart,:,:],axis=(0,1)),'-',color=plt.get_cmap('tab20')(2*irun))
                            # !!!! ADD OPTION HERE TO ADD TEST RESULTS AS WELL FOR EACH START DATE.
                            ax_rsqr_CV[i].set_xlabel('Years')
                            # ax_rsqr_CV[i].set_xticks(test_years+irun*30)
                            # ax_rsqr_CV[i].set_xticklabels(['' for k in range(len(test_years))])
                            # ax_rsqr_CV[i].set_xticklabels([str(test_years[k]) for k in range(len(test_years))])
                            # ax_rsqr_CV[i].legend()
                            ax_rsqr_CV[i].set_title(letter[i]+' '+istart_label[istart])
                            ax_rsqr_CV[0].set_ylabel('Rsqr')
                            print(np.nanmean(Rsqr_arr_valid[istart,:,:],axis=(0,1)))


                            for iyr in range(len(test_years)):
                                ax_mae_SS[i].plot(test_years[iyr]+irun*30,np.nanmean(SS_MAE_arr_valid[istart,iyr,:]),'x',color=plt.get_cmap('tab20')(2*irun+1))
                                ax_mae_SS[i].plot([test_years[iyr]+irun*30,test_years[iyr]+irun*30],[np.nanmin(SS_MAE_arr_valid[istart,iyr,:]),np.nanmax(SS_MAE_arr_valid[istart,iyr,:])],'-',linewidth=0.5,color=plt.get_cmap('tab20')(2*irun+1))
                            ax_mae_SS[i].plot(test_years+irun*30,np.ones(len(test_years))*np.nanmean(SS_MAE_arr_valid[istart,:,:],axis=(0,1)),'-',color=plt.get_cmap('tab20')(2*irun))
                            # !!!! ADD OPTION HERE TO ADD TEST RESULTS AS WELL FOR EACH START DATE.
                            ax_mae_SS[i].set_xlabel('Years')
                            # ax_mae_SS[i].set_xticks(test_years+irun*30)
                            # ax_mae_SS[i].set_xticklabels(['' for k in range(len(test_years))])
                            # ax_mae_SS[i].set_xticklabels([str(test_years[k]) for k in range(len(test_years))])
                            # ax_mae_SS[i].legend()
                            ax_mae_SS[i].set_title(letter[i]+' '+istart_label[istart])
                            ax_mae_SS[0].set_ylabel(r'Skill Score (1- $\frac{MAE_{model}}{MAE_{clim}}$)')

                    else:
                        for i,istart in enumerate(istart_plot):
                            if irun == 0:
                                cplot = plt.get_cmap('Accent')(4)
                            if irun > 0:
                                if irun < 4:
                                    cplot = plt.get_cmap('tab20c')(irun)
                                else:
                                    cplot = plt.get_cmap('tab20b')(irun)

                            for iyr in range(len(test_years)):
                                ax_mae_CV.plot(test_years[iyr]+irun*30,np.nanmean(MAE_arr_valid[istart,iyr,:]),'x',color=cplot)
                                ax_mae_CV.plot([test_years[iyr]+irun*30,test_years[iyr]+irun*30],[np.nanmin(MAE_arr_valid[istart,iyr,:]),np.nanmax(MAE_arr_valid[istart,iyr,:])],'-',linewidth=0.5,color=cplot)
                            ax_mae_CV.plot(test_years+irun*30,np.ones(len(test_years))*np.nanmean(MAE_arr_valid[istart,:,:],axis=(0,1)),'-',color=cplot)
                            # !!!! ADD OPTION HERE TO ADD TEST RESULTS AS WELL FOR EACH START DATE.
                            # ax_mae_CV.set_xlabel('Years')
                            # ax_mae_CV.set_xticks(test_years+irun*30)
                            # ax_mae_CV.set_xticklabels(['' for k in range(len(test_years))])
                            # ax_mae_CV.set_xticklabels([str(test_years[k]) for k in range(len(test_years))])
                            # ax_mae_CV.legend()
                            ax_mae_CV.set_title(letter[i]+' '+istart_label[istart])
                            ax_mae_CV.set_ylabel('MAE (days)')
                            print(np.nanmean(MAE_arr_valid[istart,:,:],axis=(0,1)))

                            for iyr in range(len(test_years)):
                                ax_rsqr_CV.plot(test_years[iyr]+irun*30,np.nanmean(Rsqr_arr_valid[istart,iyr,:]),'x',color=plt.get_cmap('tab20')(2*irun+1))
                                ax_rsqr_CV.plot([test_years[iyr]+irun*30,test_years[iyr]+irun*30],[np.nanmin(Rsqr_arr_valid[istart,iyr,:]),np.nanmax(Rsqr_arr_valid[istart,iyr,:])],'-',linewidth=0.5,color=plt.get_cmap('tab20')(2*irun+1))
                            ax_rsqr_CV.plot(test_years+irun*30,np.ones(len(test_years))*np.nanmean(Rsqr_arr_valid[istart,:,:],axis=(0,1)),'-',color=plt.get_cmap('tab20')(2*irun))
                            # !!!! ADD OPTION HERE TO ADD TEST RESULTS AS WELL FOR EACH START DATE.
                            ax_rsqr_CV.set_xlabel('Years')
                            # ax_rsqr_CV[i].set_xticks(test_years+irun*30)
                            # ax_rsqr_CV[i].set_xticklabels(['' for k in range(len(test_years))])
                            # ax_rsqr_CV[i].set_xticklabels([str(test_years[k]) for k in range(len(test_years))])
                            # ax_rsqr_CV[i].legend()
                            ax_rsqr_CV.set_title(letter[i]+' '+istart_label[istart])
                            ax_rsqr_CV.set_ylabel('Rsqr')
                            print(np.nanmean(Rsqr_arr_valid[istart,:,:],axis=(0,1)))


                            for iyr in range(len(test_years)):
                                ax_mae_SS.plot(test_years[iyr]+irun*30,np.nanmean(SS_MAE_arr_valid[istart,iyr,:]),'x',color=plt.get_cmap('tab20')(2*irun+1))
                                ax_mae_SS.plot([test_years[iyr]+irun*30,test_years[iyr]+irun*30],[np.nanmin(SS_MAE_arr_valid[istart,iyr,:]),np.nanmax(SS_MAE_arr_valid[istart,iyr,:])],'-',linewidth=0.5,color=plt.get_cmap('tab20')(2*irun+1))
                            ax_mae_SS.plot(test_years+irun*30,np.ones(len(test_years))*np.nanmean(SS_MAE_arr_valid[istart,:,:],axis=(0,1)),'-',color=plt.get_cmap('tab20')(2*irun))
                            # !!!! ADD OPTION HERE TO ADD TEST RESULTS AS WELL FOR EACH START DATE.
                            # ax_mae_SS.set_xlabel('Years')
                            # ax_mae_SS.set_xticks(test_years+irun*30)
                            # ax_mae_SS.set_xticklabels(['' for k in range(len(test_years))])
                            # ax_mae_SS.set_xticklabels([str(test_years[k]) for k in range(len(test_years))])
                            # ax_mae_SS.legend()
                            ax_mae_SS.set_title(letter[i]+' '+istart_label[istart])
                            ax_mae_SS.set_ylabel(r'Skill Score (1- $\frac{MAE_{model}}{MAE_{clim}}$)')
                            print(np.nanmean(SS_MAE_arr_valid[istart,:,:],axis=(0,1)))


                    fig_mae_CV.suptitle('VALID')
                    fig_mae_CV.subplots_adjust(top=0.8,bottom=0.15,left=0.17,right=0.95)

                    fig_rsqr_CV.suptitle('VALID')
                    fig_rsqr_CV.subplots_adjust(top=0.8,bottom=0.15,left=0.17,right=0.95)


                    fig_mae_SS.suptitle('VALID')
                    fig_mae_SS.subplots_adjust(top=0.8,bottom=0.15,left=0.17,right=0.95)

        #-----------------









