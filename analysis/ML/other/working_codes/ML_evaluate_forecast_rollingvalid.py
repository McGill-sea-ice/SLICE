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


FCT_DIR = os.path.dirname(os.path.abspath('/storage/amelie/slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

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
import statsmodels.api as sm

from functions import rolling_climo,running_nanmean
from functions import find_freezeup_Tw,find_freezeup_Tw_all_yrs
from functions_ML import plot_series_1step,plot_series,reconstruct_ts

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
# base_path = '/Volumes/SeagateUSB/McGill/Postdoc'
base_path = '/storage/amelie'

#%%
# OPTIONS

start_doy_arr = [300,         307,       314,         321,         328,         335,       349]
istart_label = ['Oct. 27th', 'Nov. 3rd', 'Nov. 10th', 'Nov. 17th', 'Nov. 24th', 'Dec. 1st','Dec. 15th']
month_istart = [10,11,11,11,11,12,12]
day_istart   = [27, 3,10,17,24, 1,15]

valid_type = 'rollinglong'
train_yr_start = 1992 # [1992 - 2007] = 16 years
valid_yr_start = 2008 # [2008 - 2013] = 6 years
test_yr_start = 2014  # [2014 - 2019] = 6 years
# train_yr_start = 1992 # [1992 - 2007] = 16 years
# valid_yr_start = 2008 # [2008 - 2013] = 6 years
# test_yr_start = 2018  # [2014 - 2019] = 6 years

# eval_period = 'test'
eval_period = 'valid'

anomaly = False
freezeup_opt = 1

save_results = False

plot_samples = True
plot_FUD_ts = True
plot_loss = True

recalibrate = True
offset_type = 'mean_clim'

#%%
# MLP EVALUATION OPTIONS
hidden_size_list = [0]
# nb_layers_list = [3,6,12]
# input_window_list = [30,60,120,240]
input_window_list = [30,60,90]
# nb_layers_list = [2,3,6]
# input_window_list = [240]
# nb_layers_list = [3]
# input_window_list = [120,240]
nb_layers_list = [1]
# input_window_list = [30,60]

model_name = 'MLP'
norm_type = 'min_max'
use_softplus = False
lblw = 75
nneurons = 10
ne = 250
suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_mean'
# suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanNAOAOPDO'
# suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanTot.FDDTot.snowfallAvg.SLP'
# suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanNAOAOPDOTot.FDDTot.snowfallAvg.SLP'

#%%
# LSTM EVALUATION OPTIONS
hidden_size_list = [2]
# nb_layers_list = [3,6,12]
input_window_list = [30,60,120,240]
# input_window_list = [30,60,90]
# nb_layers_list = [2,3,6]
# input_window_list = [240]
# nb_layers_list = [3]
# input_window_list = [120,240]
nb_layers_list = [1]
# input_window_list = [30,60]
model_name = 'LSTM'
norm_type = 'min_max'
use_softplus = False
lblw = 75
hs = 2
ne = 250
suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_mean'
# suffix = 'Avg.TwaterTot.TDDTot.FDD'
# # # suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanAvg.levelOttawaRiver'
# # # suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanNAOAOPDO'
# suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanNAOAOPDOTot.FDDTot.snowfallAvg.SLP'
# # suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanTot.FDDTot.snowfallAvg.SLP'

#%%
years_all = np.array([1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001,
       2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
       2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021])
if freezeup_opt == 1:
    Longueuil_FUD = np.array([ np.nan, 360., 358., 364., 342., 365., 350., 365., 360., 343., 367.,
            339., 348., 354., 350., 381., 341., 347., 352., 357., 364., 358.,
            347., 365., 371., 351., 348., 356., 354.,  386.,  np.nan])
if freezeup_opt == 2:
    Longueuil_FUD = np.array([ np.nan, 362., 361., 368., 339., 366., 351., 363., 363., 346., 368.,
           341., 345., 355., 349., 384., 341., 347., 355., 350., 366., 360.,
           350., 368., 371., 353., 352., 346., 355.,  np.nan,  np.nan])

Longueuil_FUD[np.where(years_all == 2020)[0][0]]= np.nan
iyr_train_start = np.where(years_all == train_yr_start )[0][0]
iyr_valid_start = np.where(years_all == valid_yr_start )[0][0]
iyr_test_start = np.where(years_all == test_yr_start )[0][0]
mean_FUD_Longueuil_train = np.nanmean(Longueuil_FUD[iyr_train_start:iyr_valid_start])
std_FUD_Longueuil_train = np.nanstd(Longueuil_FUD[iyr_train_start:iyr_valid_start])
tercile1_FUD_Longueuil_train = np.nanpercentile(Longueuil_FUD[iyr_train_start:iyr_valid_start],(1/3.)*100)
tercile2_FUD_Longueuil_train = np.nanpercentile(Longueuil_FUD[iyr_train_start:iyr_valid_start],(2/3.)*100)

# FREEZE-UP OPTION 1
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

# FREEZE-UP OPTION 2
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



#%%

MAE_arr = np.zeros((len(start_doy_arr),len(nb_layers_list),len(input_window_list),len(hidden_size_list)))*np.nan
RMSE_arr = np.zeros((len(start_doy_arr),len(nb_layers_list),len(input_window_list),len(hidden_size_list)))*np.nan
Rsqr_arr = np.zeros((len(start_doy_arr),len(nb_layers_list),len(input_window_list),len(hidden_size_list)))*np.nan
Rsqradj_arr = np.zeros((len(start_doy_arr),len(nb_layers_list),len(input_window_list),len(hidden_size_list)))*np.nan
Acc_arr = np.zeros((len(start_doy_arr),len(nb_layers_list),len(input_window_list),len(hidden_size_list)))*np.nan

for il,nlayers in enumerate(nb_layers_list):
    for iw,inpw in enumerate(input_window_list):
        for ih,hs in enumerate(hidden_size_list):
    
            print('-------------------------')
            print('# Layers = '+str(nlayers))
            print('Input window  = '+str(inpw))
    
            if eval_period == 'valid':
                nsplits = test_yr_start-valid_yr_start
                start_yr = valid_yr_start
                years_all_splits = np.arange(valid_yr_start,test_yr_start)
                Longueuil_FUD_period = Longueuil_FUD[iyr_valid_start:iyr_test_start]
            if eval_period == 'test':
                nsplits = 2020-test_yr_start
                start_yr = test_yr_start
                years_all_splits = np.arange(test_yr_start,2020)
                Longueuil_FUD_period = Longueuil_FUD[iyr_test_start:iyr_test_start+len(years_all_splits)]
                # nsplits = 2021-test_yr_start
                # start_yr = test_yr_start
                # years_all_splits = np.arange(test_yr_start,2021)
                # Longueuil_FUD_period = Longueuil_FUD[iyr_test_start:iyr_test_start+len(years_all_splits)]
    
            # Find Categorical forecasts for observations
            Longueuil_FUD_period_cat = np.zeros(Longueuil_FUD_period.shape)*np.nan
            for iyr in range(Longueuil_FUD_period_cat.shape[0]):
                if Longueuil_FUD_period[iyr] <= tercile1_FUD_Longueuil_train:
                    Longueuil_FUD_period_cat[iyr] = -1
                elif Longueuil_FUD_period[iyr] > tercile2_FUD_Longueuil_train:
                    Longueuil_FUD_period_cat[iyr] = 1
                else:
                    Longueuil_FUD_period_cat[iyr] = 0
    
    
            # Initalize arrays to put results of each year
            freezeup_dates_sample = np.zeros((MAE_arr.shape[0],len(years_all_splits),4))*np.nan
            freezeup_dates_sample_doy = np.zeros((MAE_arr.shape[0],len(years_all_splits)))*np.nan
    
            freezeup_dates_target = np.zeros((MAE_arr.shape[0],len(years_all_splits),4))*np.nan
            freezeup_dates_target_doy = np.zeros((MAE_arr.shape[0],len(years_all_splits)))*np.nan
    
            freezeup_dates_clim_target = np.zeros((MAE_arr.shape[0],len(years_all_splits),4))*np.nan
            freezeup_dates_clim_target_doy = np.zeros((MAE_arr.shape[0],len(years_all_splits)))*np.nan
    
            # Go trhough all years of period separately
            for i_split in range(nsplits):
                year_split = start_yr+i_split
    
                # Load data
                if use_softplus:
                    if model_name == 'MLP':
                        pred_data = np.load(base_path+'/slice/data/ML_output/'+model_name+'_'+ valid_type +'valid/softplus/'+model_name+'_valid'+ str(start_yr+i_split) +'_horizon'+str(lblw)+'_context'+str(inpw)+'_nlayers'+str(nlayers)+'_nneurons'+str(nneurons)+'_nepochs'+str(ne)+'_'+norm_type+'_'+suffix+'.npz',allow_pickle='TRUE')
                    if model_name == 'LSTM':
                        pred_data = np.load(base_path+'/slice/data/ML_output/'+model_name+'_'+ valid_type +'valid/softplus/'+model_name+'_valid'+ str(start_yr+i_split) +'_horizon'+str(lblw)+'_context'+str(inpw)+'_nlayers'+str(nlayers)+'_hiddensize'+str(hs)+'_nepochs'+str(ne)+'_'+norm_type+'_'+suffix+'.npz',allow_pickle='TRUE')
                else:
                    if model_name == 'MLP':
                        pred_data = np.load(base_path+'/slice/data/ML_output/'+model_name+'_'+ valid_type +'valid/'+model_name+'_valid'+ str(start_yr+i_split) +'_horizon'+str(lblw)+'_context'+str(inpw)+'_nlayers'+str(nlayers)+'_nneurons'+str(nneurons)+'_nepochs'+str(ne)+'_'+norm_type+'_'+suffix+'.npz',allow_pickle='TRUE')
                    if model_name == 'LSTM':
                        pred_data = np.load(base_path+'/slice/data/ML_output/'+model_name+'_'+ valid_type +'valid/'+model_name+'_valid'+ str(start_yr+i_split) +'_horizon'+str(lblw)+'_context'+str(inpw)+'_nlayers'+str(nlayers)+'_hiddensize'+str(hs)+'_nepochs'+str(ne)+'_'+norm_type+'_'+suffix+'.npz',allow_pickle='TRUE')
    
                train_dataset = np.squeeze(pred_data['train_dataset'])
                valid_dataset = np.squeeze(pred_data['valid_dataset'])
                predictors = np.squeeze(pred_data['predictors'])
                target_train = np.squeeze(pred_data['target_train'])
                target_valid = np.squeeze(pred_data['target_valid'])
                input_train = np.squeeze(pred_data['input_train'])
                input_valid = np.squeeze(pred_data['input_valid'])
                predictions_train = np.squeeze(pred_data['predictions_train'])
                predictions_valid = np.squeeze(pred_data['predictions_valid'])
                clim_target_train = pred_data['clim_target_train']
                clim_target_valid = pred_data['clim_target_valid']
                # Tw_climatology_mean_train = pred_data['Tw_climatology_mean_train']
                # Tw_climatology_mean_valid = pred_data['Tw_climatology_mean_valid']
                time_train = pred_data['time_train']
                time_valid = pred_data['time_valid']
                time_target_train = pred_data['time_target_train']
                time_target_valid = pred_data['time_target_valid']
                time_input_train = pred_data['time_input_train']
                time_input_valid = pred_data['time_input_valid']
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
                # norm_type = pred_data['norm_type']
                ynoneg = pred_data['ynoneg']
                ytransform = pred_data['ytransform']
                valid_scheme = pred_data['valid_scheme']
                # train_yr_start = pred_data['train_yr_start']
                # valid_yr_start = pred_data['valid_yr_start']
                # test_yr_start = pred_data['test_yr_start']
                yr_input_train = pred_data['yr_input_train']
                yr_input_valid = pred_data['yr_input_valid']
                # seed = pred_data['seed']
                # nb_neurons = pred_data['nb_neurons']
                # nb_layers = pred_data['nb_layers']
                # learning_rate = pred_data['learning_rate']
                # momentum = pred_data['momentum']
                # optimizer_name = pred_data['optimizer_name']
                # loss_name = pred_data['loss_name']
                # model=pred_data['model']
    
                if plot_loss:
                    plot_losses(train_losses,valid_losses,new_fig=True)
    
                # GET PREDICTIONS
                pred_recons_valid, target_recons_valid, clim_recons_valid = reconstruct_ts(target_valid,predictions_valid,yr_input_valid,time_target_valid,clim_target_valid,train_range_tar,train_offset_tar,input_width, shift, label_width, nslide, normalize_target, normalize_predictors, ytransform, train_range_pred, train_offset_pred, predictors, plot_pred_ts = False)
                Tw_target_0lead_valid = np.array(target_recons_valid[:,0])
                time_all_valid  = np.array(time_target_valid[:,0])
    
    #             plt.figure()
    #             plot_series_1step(time_all_valid,np.squeeze(clim_recons_valid[:,0]),'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(0))
    #             plot_series_1step(np.arange(time_all_valid[-1],time_all_valid[-1]+75),np.squeeze(clim_recons_valid[-1,:]),'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(0))
    
    #             lead=0
    #             # for s in range(target_recons_valid.shape[0]):
    #             # #     print(s)
    #             plot_series(time_all_valid+lead,np.array(target_recons_valid)[:,lead],'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
    #             plot_series(time_all_valid+lead,np.array(pred_recons_valid)[:,lead],'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))
    #                 # plot_series(time_in[s],np.array(predictions_recons)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))
    
    # #             itp = 29
    # #             plot_series(np.arange(time_all_valid[itp],time_all_valid[itp]+75),np.array(target_recons_valid)[itp,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
    # #             plot_series(np.arange(time_all_valid[itp],time_all_valid[itp]+75),np.array(pred_recons_valid)[itp,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))
    
                pred_arr =  np.array(pred_recons_valid)
                target_arr = np.array(target_recons_valid)
                clim_recons = np.array(clim_recons_valid)
                time_arr = time_target_valid
                Twater = np.array(target_recons_valid[:,0])
                time_all = np.array(time_target_valid[:,0])
                # Twater = (valid_dataset[:,np.where(predictors == 'Avg. Twater')[0][0]]*train_range_pred[np.where(predictors == 'Avg. Twater')[0][0]]) + train_offset_pred[np.where(predictors == 'Avg. Twater')[0][0]]
                # time_all = time_valid
    
    
                # FIND FREEZE-UP ONLY FROM SAMPLES THAT STARTED
                # ON OCT 1st, OCT 15th,
                #    NOV 1st, NOV 15th,
                #    DEC 1st, DEC 15th,
                #  (REPEAT THIS ANALYSIS FOR ALL LEAD TIMES)
    
                for istart,start_doy in enumerate(start_doy_arr):
                    month = month_istart[istart]
                    day = day_istart[istart]
    
                    # Find all samples starting in the month of start date
                    month_array = np.array([(date_ref+dt.timedelta(days=int(time_arr[s,0]))).month for s in range(time_arr.shape[0])])
                    samples_tmp = pred_arr[np.where(month_array == month )[0]][:,0:lblw]
                    targets_tmp = target_arr[np.where(month_array == month )[0]][:,0:lblw]
                    time_st_tmp = time_arr[np.where(month_array == month )[0]][:,0:lblw]
                    clim_targets_tmp = clim_recons[np.where(month_array == month )[0]][:,0:lblw]
    
                    # Find sample starting on the day & month of start date:
                    day_array = np.array([(date_ref+dt.timedelta(days=int(time_st_tmp[s,0]))).day for s in range(time_st_tmp.shape[0])])
                    samples = samples_tmp[np.where((day_array  == day) )[0]]
                    targets = targets_tmp[np.where((day_array  == day) )[0]]
                    time_st = time_st_tmp[np.where((day_array  == day) )[0]]
                    clim_targets = clim_targets_tmp[np.where((day_array  == day) )[0]]
    
    
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
    
    
                        if (istart == 5):
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
                        date_start = dt.timedelta(days=int(time[0])) + date_ref
                        if date_start.month < 3:
                            year = date_start.year-1
                        else:
                            year = date_start.year
    
                        if year >= years_all_splits[0]:
                            iyr = np.where(years_all_splits == year)[0][0]
                            fd_sample, ftw_sample, T_freezeup_sample, mask_freeze_sample = find_freezeup_Tw(def_opt,Twater_tmp,Twater_dTdt_sample,Twater_d2Tdt2_sample,time,year,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
                            fd_target, ftw_target, T_freezeup_target, mask_freeze_target = find_freezeup_Tw(def_opt,Twater_tmp_target,Twater_dTdt_target,Twater_d2Tdt2_target,time,year,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
                            fd_clim_target, ftw_clim_target, T_freezeup_clim_target, mask_freeze_clim_target = find_freezeup_Tw(def_opt,Twater_tmp_clim_target,Twater_dTdt_clim_target,Twater_d2Tdt2_clim_target,time,year,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
    
                            if (np.sum(mask_freeze_sample) > 0):
                                if fd_sample[0] == year:
                                    if calendar.isleap(years_all_splits[iyr]):
                                        freezeup_dates_sample[istart,i_split,0] = iyr
                                        freezeup_dates_sample[istart,i_split,1:4] = fd_sample
                                        freezeup_dates_sample_doy[istart,i_split]= (dt.date(int(fd_sample[0]),int(fd_sample[1]),int(fd_sample[2]))-dt.date(int(fd_sample[0]),1,1)).days
                                    else:
                                        freezeup_dates_sample[istart,i_split,0] = iyr
                                        freezeup_dates_sample[istart,i_split,1:4] = fd_sample
                                        freezeup_dates_sample_doy[istart,i_split]= (dt.date(int(fd_sample[0]),int(fd_sample[1]),int(fd_sample[2]))-dt.date(int(fd_sample[0]),1,1)).days+1
                                else:
                                    if calendar.isleap(years_all_splits[iyr]):
                                        freezeup_dates_sample[istart,i_split,0] = iyr
                                        freezeup_dates_sample[istart,i_split,1:4] = fd_sample
                                        freezeup_dates_sample_doy[istart,i_split]= fd_sample[2]+365
                                    else:
                                        freezeup_dates_sample[istart,i_split,0] = iyr
                                        freezeup_dates_sample[istart,i_split,1:4] = fd_sample
                                        freezeup_dates_sample_doy[istart,i_split]= fd_sample[2]+365
    
    
                            if (np.sum(mask_freeze_target) > 0):
                                if fd_target[0] == year:
                                    if calendar.isleap(years_all_splits[iyr]):
                                        freezeup_dates_target[istart,i_split,0] = iyr
                                        freezeup_dates_target[istart,i_split,1:4] = fd_target
                                        freezeup_dates_target_doy[istart,i_split]= (dt.date(int(fd_target[0]),int(fd_target[1]),int(fd_target[2]))-dt.date(int(fd_target[0]),1,1)).days
                                    else:
                                        freezeup_dates_target[istart,i_split,0] = iyr
                                        freezeup_dates_target[istart,i_split,1:4] = fd_target
                                        freezeup_dates_target_doy[istart,i_split]= (dt.date(int(fd_target[0]),int(fd_target[1]),int(fd_target[2]))-dt.date(int(fd_target[0]),1,1)).days+1
                                else:
                                    if calendar.isleap(years_all_splits[iyr]):
                                        freezeup_dates_target[istart,i_split,0] = iyr
                                        freezeup_dates_target[istart,i_split,1:4] = fd_target
                                        freezeup_dates_target_doy[istart,i_split]= fd_target[2]+365
                                    else:
                                        freezeup_dates_target[istart,i_split,0] = iyr
                                        freezeup_dates_target[istart,i_split,1:4] = fd_target
                                        freezeup_dates_target_doy[istart,i_split]= fd_target[2]+365
    
                            if (np.sum(mask_freeze_clim_target) > 0):
                                if fd_clim_target[0] == year:
                                    freezeup_dates_clim_target[istart,i_split,0] = iyr
                                    if calendar.isleap(years_all_splits[iyr]):
                                        freezeup_dates_clim_target[istart,i_split,1] = fd_clim_target[0]
                                        freezeup_dates_clim_target[istart,i_split,2] = fd_clim_target[1]
                                        freezeup_dates_clim_target[istart,i_split,3] = fd_clim_target[2]+1
                                        freezeup_dates_clim_target_doy[istart,i_split]= (dt.date(int(fd_clim_target[0]),int(fd_clim_target[1]),int(fd_clim_target[2]))-dt.date(int(fd_clim_target[0]),1,1)).days+1
                                    else:
                                        freezeup_dates_clim_target[istart,i_split,1:4] = fd_clim_target
                                        freezeup_dates_clim_target_doy[istart,i_split]= (dt.date(int(fd_clim_target[0]),int(fd_clim_target[1]),int(fd_clim_target[2]))-dt.date(int(fd_clim_target[0]),1,1)).days+1
    
    
                # FIND MEAN FUD FROM Tw CLIM:
                mean_clim_FUD = np.nanmean(freezeup_dates_clim_target_doy)
                if recalibrate:
                    if offset_type == 'mean_clim':
                        offset_forecasts = mean_clim_FUD
                    freezeup_dates_sample_doy[:,i_split] = freezeup_dates_sample_doy[:,i_split] - offset_forecasts + mean_FUD_Longueuil_train
    
    
            # NOW MAKE ARRAY WITH DETECTED DATES WHEN USING ALL YEARS TIME SERIES
            # (INSTEAD OF THE SAME SAMPLES AS FOR THE MODEL). THIS MIGHT BE DIFFERENT THAN
            # THE DETECTED DATES FROM THE SAMPLE ONLY SINCE SAMPLES THAT START DURING THE
            # WINTER AFTER THE INITIAL FREEZE-UP MIGHT DETECT A NEW FREEZE-UP IF THE WATER
            # TEMPERATURE GOES ABOVE ZERO MOMENTARILY AND THEN DROPS BACK (THIS IS
            # ESPECIALLY POSSIBLE WHEN USING THE FIRST DEFINITION OF FREEZE-UP WITH THE
            # TWATER THRESHOLD, BUT MAYBE BE LESS OF A PROBLEM WHEN USING THE SECOND FREEZE-IP
            # DEFINITION USING THE DERIVATIVE OF GAUSSIAN FILTER).
            freezeup_dates_obs = np.zeros((Longueuil_FUD_period.shape[0],3))*np.nan
    
            # Compute climatological freeze-up dates from DOYs
            for iyr in range(Longueuil_FUD_period.shape[0]):
                fd = Longueuil_FUD_period[iyr]
                if fd > 365:
                    fd_year = years_all_splits[iyr]+1
                    fd_month = 1
                    fd_day = fd-365
                else:
                    if calendar.isleap(years_all_splits[iyr]):
                        fd_year = years_all_splits[iyr]
                        fd_date = dt.date(int(fd_year),1,1)+dt.timedelta(days=int((fd)))
                        fd_month = fd_date.month
                        fd_day = fd_date.day
                    else:
                        fd_year = years_all_splits[iyr]
                        fd_date = dt.date(int(fd_year),1,1)+dt.timedelta(days=int((fd-1)))
                        fd_month = fd_date.month
                        fd_day = fd_date.day
    
                freezeup_dates_obs[iyr,0] = fd_year
                freezeup_dates_obs[iyr,1] = fd_month
                freezeup_dates_obs[iyr,2] = fd_day
    
    
    
    
            # EVALUATE FREEZE-UP FORECAST ACCORDING TO SELECTED METRIC:
            fu_rmse = np.zeros((freezeup_dates_sample.shape[0],len(years_all_splits)))*np.nan
            fu_acc = np.zeros((freezeup_dates_sample.shape[0],len(years_all_splits)))*np.nan
            fu_mae = np.zeros((freezeup_dates_sample.shape[0],len(years_all_splits)))*np.nan
    
            for istart in range(freezeup_dates_sample.shape[0]):
                n = 0
    
                # The FUD is not detectabble because the forecast
                # length doesn't reach the FUD (or because there
                # were no samples for that period, e.g. because
                # of the presence of nans in the predictors time
                # series)
                years_impossible =[]
                tmp_arr = np.arange(len(years_all_splits)).astype('float')
                for iyr in range(len(years_all_splits)):
                    if ~np.isnan(freezeup_dates_target[istart,iyr,0]):
                        tmp_arr[int(freezeup_dates_target[istart,iyr,0])] = np.nan
                years_impossible = years_impossible + tmp_arr[~np.isnan(tmp_arr)].tolist()
    
                # Or the FUD is not detectable because it happened
                # before the forecast start date. (e.g. the observed
                # FUD was Dec. 12th, and the forecast start date is
                # Dec. 15th)
                for iyr in range(len(years_all_splits)):
                    if dt.date(int(freezeup_dates_obs[iyr][0]),int(freezeup_dates_obs[iyr][1]),int(freezeup_dates_obs[iyr][2])) <= dt.date(int(years_all_splits[iyr]),12,31):
                        if dt.date(int(freezeup_dates_obs[iyr][0]),int(freezeup_dates_obs[iyr][1]),int(freezeup_dates_obs[iyr][2])) <= dt.date(int(freezeup_dates_obs[iyr][0]),int(month_istart[istart]),int(day_istart[istart])):
                            if iyr not in years_impossible:
                                years_impossible.append(iyr)
    
                years_impossible = np.array(years_impossible).astype('int')
                sample_freezeup_doy = freezeup_dates_sample_doy[istart,0:len(years_all_splits)]
                clim_target_freezeup_doy = freezeup_dates_clim_target_doy[istart,0:len(years_all_splits)]
                ts_doy = np.zeros(len(years_all_splits))*np.nan
    
                # Evaluate the performance only on detectable FUDs
                obs_fud_tmp = freezeup_dates_obs.copy()
                obs_fud_tmp[years_impossible] = np.nan
    
                if np.sum(~np.isnan(obs_fud_tmp)) > 0:
    
                    for iyr in range(len(years_all_splits)):
                        if (~np.isnan(obs_fud_tmp[iyr,0])):
                            n += 1
                            if len(np.where(freezeup_dates_sample[istart,:,0]==iyr)[0])>0:
                                i_s = np.where(freezeup_dates_sample[istart,:,0]==iyr)[0][0]
                                fs_doy = sample_freezeup_doy[i_s]
                                ts_doy[iyr] = fs_doy
                                fo_doy = Longueuil_FUD_period[iyr]
                                fc_doy = mean_clim_FUD
    
                                obs_cat = Longueuil_FUD_period_cat[iyr]
                                if ~np.isnan(fs_doy):
                                    if fs_doy <= tercile1_FUD_Longueuil_train:
                                        sample_cat = -1
                                    elif fs_doy > tercile2_FUD_Longueuil_train:
                                        sample_cat = 1
                                    else:
                                        sample_cat = 0
                                    if (sample_cat == obs_cat):
                                        fu_acc[istart,iyr] = 1
                                else:
                                    fu_acc[istart,iyr] = np.nan
    
                                fu_rmse[istart,iyr] = (fs_doy-fo_doy)**2.
                                fu_mae[istart,iyr] = np.abs(fs_doy-fo_doy)
                                # print(istart,iyr,n,fo_doy,fs_doy,fc_doy,obs_cat,sample_cat,fu_acc[istart,iyr])
    
                            else:
                                ts_doy[iyr] = np.nan
                                fu_rmse[istart,iyr] = np.nan
                                fu_mae[istart,iyr] = np.nan
                                fu_acc[istart,iyr] = np.nan
    
                                # print(istart,iyr,n,Longueuil_FUD_period[iyr],np.nan,mean_clim_FUD,Longueuil_FUD_period_cat[iyr],np.nan,fu_acc[istart,iyr])
    
                # Here we don't use nanmean, but we nansum and divide by "n", where
                # "n" is the number of detectable FUDs for the given start date.
                # If an FUD was not forecasted/detected but it could have been,
                # then n will be larger than the available number of ML forecasts so the
                # performance is penalized.
                if np.all(np.isnan(fu_mae[istart,:])):
                    MAE_arr[istart,il,iw,ih] = np.nan
                else:
                    MAE_arr[istart,il,iw,ih] = np.nansum(fu_mae[istart,:])/n
    
                if np.all(np.isnan(fu_rmse[istart,:])):
                    RMSE_arr[istart,il,iw,ih] = np.nan
                else:
                    RMSE_arr[istart,il,iw,ih] = np.sqrt(np.nansum(fu_rmse[istart,:])/n)
    
                if  np.all(np.isnan(fu_acc[istart,:])):
                    Acc_arr[istart,il,iw,ih] = np.nan
                else:
                    Acc_arr[istart,il,iw,ih] = np.nansum(fu_acc[istart,:])/n
    
                if (np.all(np.isnan(ts_doy))) | (np.all(np.isnan(Longueuil_FUD_period))) :
                    Rsqr_arr[istart,il,iw,ih] = np.nan
                    Rsqradj_arr[istart,il,iw,ih] = np.nan
                else:
                    model = sm.OLS(Longueuil_FUD_period, sm.add_constant(ts_doy,has_constant='skip'), missing='drop').fit()
                    Rsqr_arr[istart,il,iw,ih] = model.rsquared
                    Rsqradj_arr[istart,il,iw,ih] = model.rsquared_adj
    
    
            if plot_FUD_ts:
                # PLOT FUD TIME SERIES - VALIDATION
    
                # np.nanmean(avg_freezeup_doy[0:18])
                # MLR_pred = np.array([354.98402352,
                #        363.54500541,
                #        349.8139212 ,
                #        349.91221944,
                #        352.48026772,
                #        363.77961087,
                #        359.96231006,
                #        349.02877679,
                #        341.08672315])
    
                plt.figure()
                # plt.plot(years_all,np.ones(len(years_all))*(mean_clim_FUD),color=plt.get_cmap('tab20c')(2))
                plt.plot(years_all,np.ones(len(years_all))*(mean_FUD_Longueuil_train),color=[0.7,0.7,0.7])
                plt.plot(years_all,Longueuil_FUD,'o-',color='black')
                # plt.plot(years_all[iyr_valid_start:iyr_test_start],MLR_pred ,'o:',color=plt.get_cmap('tab20')(4),label='Lin.Reg. - $T_{a}$ Nov.')
    
                fd_ML_forcast = np.zeros((len(start_doy_arr),len(years_all_splits)))*np.nan
                istart_show = [1,2,3,4,5,6]
                for ic,istart in enumerate(np.arange(len(start_doy_arr))):
    
                    for iyr in range(len(years_all_splits)):
                        select_yr_fud = freezeup_dates_sample[istart,np.where(freezeup_dates_sample[istart,:,0] == iyr)[0]]
                        fd_avg = 0
                        if select_yr_fud.shape[0] > 0:
                            fd_ML_forcast[ic,iyr] = freezeup_dates_sample_doy[istart,np.where(freezeup_dates_sample[istart,:,0] == iyr)[0]]
    
                    if istart in istart_show:
                        if istart == 5:
                            # plt.plot(years_all_splits,fd_ML_forcast,'o:', color=plt.get_cmap('Accent')(6), label= 'MLP - '+istart_label[istart])
                            plt.plot(years_all_splits,fd_ML_forcast[ic],'o:', color=plt.get_cmap('Set1')(6), label= 'MLP - '+istart_label[istart])
                        elif istart == 6:
                            plt.plot(years_all_splits,fd_ML_forcast[ic],'o:', color=plt.get_cmap('tab10')(0), label= 'MLP - '+istart_label[istart])
                        else:
                            if istart > 0:
                                plt.plot(years_all_splits,fd_ML_forcast[ic],'o:', color=plt.get_cmap('tab20c')(7-ic+1), label= 'MLP - '+istart_label[istart])
    
    
                    if ~np.all(np.isnan(fd_ML_forcast[ic])):
                        model = sm.OLS(Longueuil_FUD_period, sm.add_constant(fd_ML_forcast[ic],has_constant='skip'), missing='drop').fit()
                        print('-----------------------------')
                        # print('START DATE: ' + istart_label[istart])
                        # print('Rsqr: ',model.rsquared, model.rsquared_adj)
                        # print('MAE: ',np.nanmean(np.abs(Longueuil_FUD_period-fd_ML_forcast)),np.nanmean(np.abs(Longueuil_FUD_period-mean_clim_FUD)))
                        # print('RMSE: ',np.sqrt(np.nanmean((Longueuil_FUD_period-fd_ML_forcast)**2.)),np.sqrt(np.nanmean((Longueuil_FUD_period-mean_clim_FUD)**2.)))
                        # print('Acc.: ',np.nansum(np.abs(Longueuil_FUD_period-fd_ML_forcast)<=7)/len(Longueuil_FUD_period),np.nansum(np.abs(Longueuil_FUD_period-mean_clim_FUD)<=7)/len(Longueuil_FUD_period))
                        print('START DATE: ' + istart_label[istart])
                        print('Rsqr: ',model.rsquared, model.rsquared_adj)
                        print('MAE: ', MAE_arr[istart,il,iw,ih])
                        print('RMSE: ',RMSE_arr[istart,il,iw,ih])
                        print('Acc.: ',Acc_arr[istart,il,iw,ih])
    
                    if recalibrate:
                        if offset_type == 'mean_clim':
                            plt.title('Recalibrated Forecasts - Mean clim\n'+'nlayers:'+str(nlayers)+', input window: '+str(inpw))
                    else:
                        plt.title('Raw Forecasts\n'+'nlayers:'+str(nlayers)+', input window: '+str(inpw))
    
                plt.legend()
    
                if save_results:
                    spath = base_path+'/slice/prog/analysis/plots_IAHR_paper/'
                    savename = spath+eval_period+'_'+model_name+'_'+'horizon'+str(lblw)+'_context'+str(inpw)+'_nlayers'+str(nlayers)+'_nneurons'+str(nneurons)+'_nepochs'+str(ne)+'_'+norm_type+'_'+suffix
                    np.savez(savename,
                             pred = fd_ML_forcast,
                             years = years_all_splits,
                             MAE_arr = MAE_arr[:,il,iw,ih],
                             RMSE_arr = RMSE_arr[:,il,iw,ih],
                             Rsqr_arr = Rsqr_arr[:,il,iw,ih],
                             Rsqradj_arr = Rsqradj_arr[:,il,iw,ih],
                             Acc_arr = Acc_arr[:,il,iw,ih],
                             )
    

#%%
istart_plot = [1,2,3,4,5]

cmap = plt.get_cmap('magma', 7)
# cmap = plt.get_cmap('coolwarm', 7)

fig_mae,ax_mae = plt.subplots(nrows = 1, ncols = len(istart_plot),figsize=(14,3.5),sharey=True)
letter = ['a)','b)','c)','d)','e)','f)']
for i,istart in enumerate(istart_plot):
    # p_mae=ax_mae[i].imshow(MAE_arr[istart],vmin=4,vmax=9,cmap='coolwarm')
    # p_mae=ax_mae[i].imshow(MAE_arr[istart],vmin=4,vmax=9,cmap='YlOrRd')
    # p_mae=ax_mae[i].imshow(MAE_arr[istart],vmin=3,vmax=9,cmap='afmhot_r')
    # p_mae=ax_mae[i].imshow(MAE_arr[istart],vmin=3,vmax=9,cmap='hot_r')
    # p_mae=ax_mae[i].imshow(MAE_arr[istart],vmin=3.75,vmax=9.5,cmap='magma')
    p_mae=ax_mae[i].imshow(np.squeeze(MAE_arr[istart]),vmin=3,vmax=10,cmap=cmap)
    ax_mae[0].set_ylabel('Nb. Layers')
    ax_mae[i].set_xlabel('Context Window (days)')
    ax_mae[i].set_yticks(np.arange(len(nb_layers_list)))
    ax_mae[i].set_yticklabels([str(nb_layers_list[k]) for k in range(len(nb_layers_list))])
    ax_mae[i].set_xticks(np.arange(len(input_window_list)))
    ax_mae[i].set_xticklabels([str(input_window_list[k]) for k in range(len(input_window_list))])
    # ax_mae[i].set_title(letter[i]+' Forecast start: '+istart_label[istart])
    # ax_mae[i].set_title(letter[i]+'                                   ')
    ax_mae[i].text(-0.45,-0.65,letter[i]+'      Forecast start: \n           '+istart_label[istart],fontweight='bold',fontsize=12)
fig_mae.subplots_adjust(top=0.92, bottom=0.32,left=0.041,right=0.988,hspace=0.2,wspace =0.095)
cbar_mae_ax = fig_mae.add_axes([0.29, 0.13, 0.42, 0.07])
fig_mae.colorbar(p_mae, cax=cbar_mae_ax,orientation='horizontal')
cbar_mae_ax.text(6.1,-1.6,'MAE (days)',fontsize=12,fontweight='bold')



fig_acc,ax_acc = plt.subplots(nrows = 1, ncols = len(istart_plot),figsize=(14,3))
letter = ['f)','g)','h)','i)','j)','k)']
for i,istart in enumerate(istart_plot):
    p_acc=ax_acc[i].imshow(np.squeeze(Acc_arr[istart])*100,vmin=10,vmax=80)
    ax_acc[0].set_ylabel('Nb. Layers')
    # ax_acc[1].set_xlabel('Context Window (days)')
    ax_acc[i].set_yticks(np.arange(len(nb_layers_list)))
    ax_acc[i].set_yticklabels([str(nb_layers_list[k]) for k in range(len(nb_layers_list))])
    ax_acc[i].set_xticks(np.arange(len(input_window_list)))
    ax_acc[i].set_xticklabels([str(input_window_list[k]) for k in range(len(input_window_list))])
    # ax_acc[i].set_title(letter[i]+' Forecast start: '+istart_label[istart])
    ax_acc[i].set_title(letter[i]+'                                   ')
fig_acc.subplots_adjust(left=0.2,wspace = 0.45)
cbar_acc_ax = fig_acc.add_axes([0.07, 0.26, 0.02, 0.52])
fig_acc.colorbar(p_acc, cax=cbar_acc_ax)
cbar_acc_ax.text(-0.95,27,'Accuracy (%)',fontsize=14, rotation = 90)


fig_Rsqr,ax_Rsqr = plt.subplots(nrows = 1, ncols = len(istart_plot),figsize=(14,3),sharex=True,sharey=True)
letter = ['k)','l)','m)','n)','o)','p)']
for i,istart in enumerate(istart_plot):
    p_Rsqr=ax_Rsqr[i].imshow(np.squeeze(Rsqr_arr[istart]),vmin=0.1,vmax=0.90)
    ax_Rsqr[0].set_ylabel('Nb. Layers')
    ax_Rsqr[i].set_xlabel('Context Window (days)')
    ax_Rsqr[i].set_yticks(np.arange(len(nb_layers_list)))
    ax_Rsqr[i].set_yticklabels([str(nb_layers_list[k]) for k in range(len(nb_layers_list))])
    ax_Rsqr[i].set_xticks(np.arange(len(input_window_list)))
    ax_Rsqr[i].set_xticklabels([str(input_window_list[k]) for k in range(len(input_window_list))])
    # plt.colorbar(p_Rsqr,ax=ax_Rsqr[i],fraction=0.046, pad=0.04)
    # ax_Rsqr[i].set_title(letter[i]+' Forecast start: '+istart_label[istart])
    ax_Rsqr[i].set_title(letter[i]+'                                   ')
fig_Rsqr.subplots_adjust(left=0.2,wspace = 0.45)
cbar_Rsqr_ax = fig_Rsqr.add_axes([0.07, 0.26, 0.02, 0.52])
fig_Rsqr.colorbar(p_Rsqr, cax=cbar_Rsqr_ax)
cbar_Rsqr_ax.text(-0.95,0.5,'R$^{2}$',fontsize=14, rotation = 90)


