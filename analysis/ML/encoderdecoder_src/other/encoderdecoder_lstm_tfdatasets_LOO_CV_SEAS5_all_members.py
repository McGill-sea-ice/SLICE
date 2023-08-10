#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" EncoderDecoder_LSTM


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

import gc

from functions import rolling_climo
from functions_ML import regression_metrics, plot_sample, plot_prediction_timeseries

from functions_encoderdecoder import fit_scaler, normalize_df, get_predictor_clim, replace_nan_with_clim, encoder_decoder_recursive
from functions_encoderdecoder import create_dataset_tpf, reconstruct_ysamples_tpf, execute_fold_tpf, create_dataset_tpf
from functions_encoderdecoder import SEAS5_dailyarray_to_ts, obs_dailyts_to_forecast_ts

#%%
import warnings
warnings.filterwarnings("ignore")

#%% CHOOSE RUN SETTINGS:

use_GPU = False

print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if use_GPU:
    if len(tf.config.list_physical_devices('GPU')) > 0:
        device_name = '/GPU:0'
    else:
        print('No GPU available... Defaulting to using CPU.')
        device_name = '/CPU:0'
else:
    device_name = '/CPU:0'


# To see on which device the operation are done, uncomment below:
# tf.debugging.set_log_device_placement(True)


#%% CHOOSE MODEL SETTINGS:

start_time = timer.time()

# If using 'LOOk', need to specify how many folds to use for validation:
valid_scheme = 'LOOk'
nfolds = 5

# If using'standard', need to specify bounds for train/valid/test sets:
valid_scheme = 'standard'
train_yr_start = 1993 # Training dataset: 1993 - 2010
valid_yr_start = 2018 # Validation dataset: 2011 - 2015
test_yr_start = 2018 # Testing dataset: 2016 - 2021


# SET MODEL AND TRAINING HYPER-PARAMETERS:

# Choose batch size:
# batch_size = 32
# batch_size = 64
# batch_size = 128
# batch_size = 256
batch_size = 512 # <--- This is the max size for running on bjerknes' GPU
# batch_size = 1024
# batch_size = 2048
# batch_size = 4096
# batch_size = 8192

# Choose learning rate: (Note: Optimizer is Adam)
optimizer_name = 'Adam'
lr = 0.0004
# lr = 0.001
# lr = 0.004
# lr = 0.008
# lr = 0.016 # <---
# lr = 0.032
# lr = 0.064
# lr = 0.128
# lr = 0.256
# lr = 0.512


# Choose loss function:
loss = 'MSETw'  # <---
# loss = 'MSETw_MSEdTwdt'
# loss = 'MSETw_penalize_FN' # Will not work with target anomalies
# loss = 'MSETw_penalize_recall' # Will not work with target anomalies
# loss = 'log_loss'

use_exp_decay_loss = False  # <--- True
tau = 30

# Will not work with target anomalies
weights_on_timesteps = True # <--- False
added_weight = 1.
Tw_thresh = 0.75


loss_name = (loss
             +weights_on_timesteps*('_with_weights'+str(added_weight).rsplit('.')[0]+'_on_thresh'+str(Tw_thresh).rsplit('.')[0]+'_'+str(Tw_thresh).rsplit('.')[1])
             +use_exp_decay_loss*('_exp_decay_tau'+str(tau))
             )


# Set max. number of epochs
# n_epochs = 120
n_epochs = 2
# n_epochs = 200

# Number of hidden neurons in Encoder and Decoder
# latent_dim = 2
latent_dim = 50
# latent_dim = 200
# latent_dim = 10
nb_layers = 1

# Choose Dense layer activation function and data normalization type:
# dense_act_func = 'sigmoid'
# norm_type='MinMax'

dense_act_func = None
norm_type='Standard'

# dense_act_func = None
# norm_type='None'

# Choose Dropout rate:
inp_dropout = 0
rec_dropout = 0

# Prediction window length, in days
pred_len = 60
# pred_len = 30
# pred_len = 14
# pred_len = 4

# Input window length, in days
input_len = 128
# input_len = 240

# Select variables to use:
predictor_vars = ['Avg. Ta_mean',
                  # 'Avg. Ta_min',
                  # 'Avg. Ta_max',
                   'Avg. cloud cover',
                   'Tot. snowfall',
                   'NAO',
                   'Avg. Twater',
                   'Avg. discharge St-L. River',
                  # 'Avg. level Ottawa River',
                  # 'Avg. SLP',
                   'Avg. SH (sfc)',
                   'Avg. SW down (sfc)',
                  ]
# predictor_vars = []

# perfect_forecast = True
perfect_forecast = False
# ensemble_mean_fcst = True
ensemble_mean_fcst = False

forecast_vars = ['Avg. Ta_mean',
                    # 'Tot. snowfall',
                # 'Avg. cloud cover',
                # 'Avg. discharge St-L. River',
                # 'NAO',
                # 'Avg. SH (sfc)',
                # 'Avg. SW down (sfc)',
                # 'Avg. SLP'
                ]
# forecast_vars = []

target_var = ['Avg. Twater']

# Choose if using anomaly timeseries:
# anomaly_target = True
# anomaly_past = True
# anomaly_frcst = True

# anomaly_target = False
# anomaly_past = True
# anomaly_frcst = True

anomaly_target = False
anomaly_past = False
anomaly_frcst = False

# Set random seed:
fixed_seed = True
# fixed_seed = False
seed = 442
# seed = 84
# seed = 2
# seed = 17

# save_model_outputs = True
save_model_outputs = False

# suffix = '_perfectexp_Ta_mean_cloud_cover_snowfall'
# suffix = '_perfectexp_Ta_mean_snowfall'
# suffix = '_perfectexp_Ta_mean'
# suffix = '_perfectexp'
# suffix = '_no_forecast_vars't
# suffix = '_perfectexp_noTw'
# suffix = '_RWE'
# suffix = '_PFE'
# suffix = 'TEST_100epochs_reducelrlin_0_00016_lr0_016_dropout_0_1'
# suffix = '_lr0_0005_nocallbacks_Tafcstonly_RWE_allmembers'
# suffix = '_lr0_0005_nocallbacks_Tafcstonly_RWE_ensemblemean'
# suffix = '_lr0_0005_nocallbacks_Tafcstonly_PFE'
suffix = '_reducelrexp_0_025_lr0_001_PFE'
# suffix = '_reducelrexp_0_025_lr0_001_RWE_withsnow'

#%% RUN MODEL
with tf.device(device_name):

    # LOADING THE DATA AND PRE-PROCESSING
    """
    The dataset is composed of the following variables:

    *   Dates (in days since 1900-01-01)
    *   Daily water temperature (in °C - from the Longueuil water filtration plant)
    *   ERA5 weather daily variables (*see belo*w)
    *   Daily discharge (in m$^3$/s) and level (in m) for the St-Lawrence River (at Lasalle and Pointe-Claire, respectively)
    *   Daily level (in m) for the Ottawa River (at Ste-Anne-de-Bellevue)
    *   Daily time series of climate indices (*see below*)
    *   Daily time series of monthly and seasonal forecasts from CanSIPS (*see below*)

    """

    # The dataset has been copied into an Excel spreadsheet for this example
    filepath = 'slice/data/colab/'
    df = pd.read_excel(local_path+filepath+'predictor_data_daily_timeseries.xlsx')

    # The dates column is not needed
    time = df['Days since 1900-01-01']
    df.drop(columns='Days since 1900-01-01', inplace=True)

    # Keep only data for 1992-2020.
    yr_start = 1993
    yr_end = 2020
    date_ref = dt.date(1900,1,1)
    it_start = np.where(time == (dt.date(yr_start,1,1)-date_ref).days)[0][0]
    it_end = np.where(time == (dt.date(yr_end+1,1,1)-date_ref).days)[0][0]

    df = df.iloc[it_start:it_end,:]
    time = time[it_start:it_end]

    # There is a missing data in the water temperature time series...
    # This gap occurs during winter 2019, so we will fill it with zeros.
    ii = np.where(time == (dt.date(2019,1,25)-date_ref).days )[0][0]
    ff = np.where(time == (dt.date(2019,3,26)-date_ref).days )[0][0]
    df['Avg. Twater'][ii:ff] = 0

    # We also cap all negative water temperature to zero degrees.
    df['Avg. Twater'][df['Avg. Twater'] < 0] = 0

    # And replace nan values in FDD and TDD by zeros
    df['Tot. FDD'][np.isnan(df['Tot. FDD'])] = 0
    df['Tot. TDD'][np.isnan(df['Tot. TDD'])] = 0

    # Note: other nan values due to missing values are replaced by the
    # training climatology below...

    # Create the time vector for plotting purposes:
    first_day = (date_ref+dt.timedelta(days=int(time.iloc[0]))).toordinal()
    last_day = (date_ref+dt.timedelta(days=int(time.iloc[-1]))).toordinal()
    time_plot = pd.DataFrame(np.arange(first_day, last_day + 1),index=time.index)

    plot_target = False
    if plot_target:
        # Plot the observed water temperature time series, which will be the target variable
        fig, ax = plt.subplots(figsize=[12, 6])
        ax.plot(time_plot.values, df['Avg. Twater'][:], color='C0', label='T$_w$')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlabel('Time')
        ax.set_ylabel('Water temperature $[^{\circ}C]$')
        ax.legend(loc='best')
        ax.grid(True)


    # PREDICTOR SELECTION
    """
    The data frame will contain (in order):
      [the target variable] x 1 column
      [the past predictors] x n_predictors columns
      [the forecasts] x n_forecasts columns
    """

    if perfect_forecast:
        #Use observations as forecast variables
        df_f = np.zeros((len(time),len(forecast_vars)*12))*np.nan
        df_f_col = []
        for iv,fv in enumerate(forecast_vars):
            obs_f = df[fv]
            for month in range(1,13):
                df_f_col = df_f_col + [fv]
                obs_ts = obs_dailyts_to_forecast_ts(obs_f,month,time.values)
                df_f[:,iv*12+month-1] = obs_ts

        n_f_vars = len(forecast_vars)

    else:
        # Load SEAS5 as forecast variables
        fdir_SEAS5 = local_path+'slice/prog/analysis/SEAS5/'
        fname_SEAS5 = 'SEAS5_SEAS51_MLR_forecast_daily_predictors.npz'
        data_SEAS5 = np.load(fdir_SEAS5+fname_SEAS5,allow_pickle=True)
        SEAS5_years = np.arange(1981,2022+1)
        time_SEAS5 = np.arange((dt.date(SEAS5_years[0],1,1)-date_ref).days,(dt.date(SEAS5_years[-1]+1,1,1)-date_ref).days)
        it0_SEAS5 = np.where(time_SEAS5 == time.iloc[0])[0][0]
        it1_SEAS5 = np.where(time_SEAS5 == time.iloc[-1])[0][0]


        if ensemble_mean_fcst:
            df_f = np.zeros((len(time),len(forecast_vars)*12))*np.nan
            df_f_col = []
            for iv,fv in enumerate(forecast_vars):
                if fv == 'Avg. Ta_mean':
                    sv = 'SEAS5_Ta_mean'
                elif fv == 'Tot. snowfall':
                    sv = 'SEAS5_snowfall'
                else:
                    raise('SEAS5 variable not yet implemented')
                SEAS5_f = data_SEAS5[sv][:,:,:,:25]
                SEAS5_f_em = np.nanmean(SEAS5_f,axis = 3)
                for month in range(1,13):
                    df_f_col = df_f_col + [sv]
                    SEAS5_ts = SEAS5_dailyarray_to_ts(SEAS5_f_em,SEAS5_years,month,time_SEAS5)
                    df_f[:,iv*12+month-1] = SEAS5_ts[it0_SEAS5:it1_SEAS5+1]
        else:
            df_f = np.zeros((len(time),len(forecast_vars)*12*25))*np.nan
            df_f_col = []
            for iv,fv in enumerate(forecast_vars):
                if fv == 'Avg. Ta_mean':
                    sv = 'SEAS5_Ta_mean'
                elif fv == 'Tot. snowfall':
                    sv = 'SEAS5_snowfall'
                else:
                    raise('SEAS5 variable not yet implemented')
                SEAS5_f = data_SEAS5[sv][:,:,:,:25]
                for im in range(SEAS5_f.shape[3]):
                    for month in range(1,13):
                        df_f_col = df_f_col + [sv+', '+str(im)]
                        SEAS5_ts = SEAS5_dailyarray_to_ts(SEAS5_f[:,:,:,im],SEAS5_years,month,time_SEAS5)
                        df_f[:,12*(im+iv*25)+month-1] = SEAS5_ts[it0_SEAS5:it1_SEAS5+1]

        n_f_vars = (not ensemble_mean_fcst)*len(forecast_vars)*25+ (ensemble_mean_fcst)*len(forecast_vars)

    df_f = pd.DataFrame(df_f,columns=df_f_col,index=df.index) # Forecast variables
    df_t = df[target_var] # Target variable
    df_p = df[predictor_vars] # Predictor variables


    # SPLITTING THE DATASET + FILLING NANS + DATA SCALING + CREATE SAMPLES
    """
    The data is first separated in train-valid-test sets according to the
    chosen CV scheme.

    Then a daily climatology is computed for each predictor by using a moving
    average window and using only values from the training set.
    Nan values in the predictors and the target time series are replaced by
    their daily climatological values.

    The data is then normalized (MinMaxScaler or StandardScaler) to help
    the learning process. The scaler uses only the training dataset for
    calibration.

    Finally, the samples are prepared using a rolling window with
    tf.data.Dataset objects that are batched and pre-fetched.
    """

    #---------------------------
    if valid_scheme == 'LOOk':

        from sklearn.model_selection import KFold

        valid_years = np.nan
        test_years = np.arange(yr_start,yr_end)

        if nfolds > 0: kf = KFold(n_splits=nfolds)

        for iyr_test,yr_test in enumerate(test_years):
        # for iyr_test,yr_test in enumerate(test_years[10:12]):
            istart_yr_test = np.where(time_plot == dt.date(yr_test, 4, 1).toordinal())[0][0]
            iend_yr_test = np.where(time_plot == dt.date(yr_test+1, 4, 1).toordinal())[0][0]
            # istart_yr_test = np.where(time_plot == dt.date(yr_test, 3, 1).toordinal())[0][0]
            # iend_yr_test = np.where(time_plot == dt.date(yr_test+1, 3, 1).toordinal())[0][0]

            is_test = np.zeros(df.shape[0], dtype=bool)
            is_test[istart_yr_test:iend_yr_test] = True

            df_f_kfold = df_f.iloc[~is_test].copy()
            df_t_kfold = df_t.iloc[~is_test].copy()
            df_p_kfold = df_p.iloc[~is_test].copy()

            ind_kfold = df[~is_test].index

            df_f_test_fold = df_f.iloc[is_test].copy()
            df_t_test_fold = df_t.iloc[is_test].copy()
            df_p_test_fold = df_p.iloc[is_test].copy()

            ind_test_fold = df[is_test].index  # NOTE: THESE ARE THE SAME INDICES AS WHERE is_test IS TRUE
            test_dates = [dt.date.fromordinal(d[0]) for d in time_plot.loc[ind_test_fold].values]
            time_test = time.loc[ind_test_fold].values
            time_test_plot = time_plot.loc[ind_test_fold].values

            y_valid_all = np.zeros((nfolds,365*28,pred_len))*np.nan
            y_pred_valid_all = np.zeros((nfolds,365*28,pred_len))*np.nan
            y_clim_valid_all = np.zeros((nfolds,365*28,pred_len))*np.nan
            target_time_valid_all = np.zeros((nfolds,365*28,pred_len))*np.nan
            history_valid_all = dict()
            history_test_all = dict()

            # Get cross-validation folds (determined using all years other than test year)
            if nfolds > 0:
                for ifold,[train_index, valid_index] in enumerate(kf.split(ind_kfold)):
                    # if (ifold == 2) | (ifold == 3): # REMOVE THIS AFTER!!!!!!!!!!!1

                    # GET TRAINING AND VALIDATION DATA SETS FOR THIS FOLD:
                    df_f_train_fold, df_f_valid_fold = df_f_kfold.iloc[train_index], df_f_kfold.iloc[valid_index]
                    df_t_train_fold, df_t_valid_fold = df_t_kfold.iloc[train_index], df_t_kfold.iloc[valid_index]
                    df_p_train_fold, df_p_valid_fold = df_p_kfold.iloc[train_index], df_p_kfold.iloc[valid_index]
                    ind_train_fold, ind_valid_fold = ind_kfold[train_index], ind_kfold[valid_index]

                    train_dates = [dt.date.fromordinal(d[0]) for d in time_plot.loc[ind_train_fold].values]
                    valid_dates = [dt.date.fromordinal(d[0]) for d in time_plot.loc[ind_valid_fold].values]

                    time_train = time.loc[ind_train_fold].values
                    time_valid = time.loc[ind_valid_fold].values
                    time_train_plot = time_plot.loc[ind_train_fold].values
                    time_valid_plot = time_plot.loc[ind_valid_fold].values

                    month_train_fold = np.array([(date_ref+dt.timedelta(days=int(time_train[it]))).month for it in range(len(time_train))])
                    month_valid_fold = np.array([(date_ref+dt.timedelta(days=int(time_valid[it]))).month for it in range(len(time_valid))])
                    month_test_fold  = np.array([(date_ref+dt.timedelta(days=int(time_test[it]))).month for it in range(len(time_test))])

                    train_years = np.unique([dt.date.fromordinal(d[0]).year for d in time_plot.loc[ind_train_fold].values])
                    train_years = np.delete(train_years,np.where(train_years==yr_test))
                    valid_years = np.unique([dt.date.fromordinal(d[0]).year for d in time_plot.loc[ind_valid_fold].values])
                    valid_years = np.delete(valid_years,np.where(valid_years==yr_test))


                    [model_out,
                      history_valid_all['loss'+str(ifold)], history_valid_all['val_loss'+str(ifold)],
                    _,target_time_valid,_,
                    _,y_pred_valid,_,
                    _,y_valid,_,
                    _,y_clim_valid,_] = execute_fold_tpf(df_t_train_fold, df_t_valid_fold, df_t_test_fold,
                                                          df_p_train_fold, df_p_valid_fold, df_p_test_fold,
                                                          df_f_train_fold, df_f_valid_fold, df_f_test_fold,
                                                          [np.where(time.index == ind)[0][0] for ind in ind_train_fold.values], [np.where(time.index == ind)[0][0] for ind in ind_valid_fold.values], [np.where(time.index == ind)[0][0] for ind in ind_test_fold.values],
                                                          train_years,time.values,
                                                          time_train,time_valid,time_test,
                                                          time_train_plot,time_valid_plot,time_test_plot,
                                                          month_train_fold,month_valid_fold,month_test_fold,
                                                          latent_dim,nb_layers,inp_dropout,rec_dropout,dense_act_func,
                                                          len(predictor_vars),n_f_vars,
                                                          input_len,pred_len,
                                                          norm_type,anomaly_target,anomaly_past,anomaly_frcst,
                                                          batch_size,lr,
                                                          loss,use_exp_decay_loss,tau,weights_on_timesteps,added_weight,Tw_thresh,
                                                          n_epochs,
                                                          seed,
                                                          perfect_forecast,
                                                          ensemble_mean_fcst,
                                                          fixed_seed =fixed_seed,
                                                          plot_loss=False,
                                                          plot_predictions=False,
                                                          plot_targets=False,
                                                          show_modelgraph=False,
                                                          show_weights=False)

                    # SAVE VALIDATION PREDICTIONS AND METRICS
                    # FOR ALL FOLDS OF THAT TEST YEAR.
                    y_valid_all[ifold,0:y_valid.shape[0],:] = y_valid
                    y_pred_valid_all[ifold,0:y_valid.shape[0],:] = y_pred_valid
                    y_clim_valid_all[ifold,0:y_valid.shape[0],:] = y_clim_valid
                    target_time_valid_all[ifold,0:y_valid.shape[0],:] = target_time_valid


                    print('=====================================================')
                    print(str(yr_test) + ' --- FOLD ' + str(ifold) + ' OF ' + str(nfolds)+ ' COMPLETE')
                    print('=====================================================')
                    del target_time_valid,y_pred_valid,y_valid,y_clim_valid,model_out
                    gc.collect()

                # Remove other variables that are no longer needed
                del df_t_kfold, df_t_test_fold, df_p_kfold, df_p_test_fold, df_f_kfold, df_f_test_fold
                gc.collect()

            # Then, train with all other years to predict test year:
            df_t_train = df_t[~is_test].copy()
            df_t_valid = df_t[is_test].copy() # This is the same as test data, just as a placeholder
            df_t_test = df_t[is_test].copy()
            df_p_train = df_p[~is_test].copy()
            df_p_valid = df_p[is_test].copy() # This is the same as test data, just as a placeholder
            df_p_test = df_p[is_test].copy()
            df_f_train = df_f[~is_test].copy()
            df_f_valid = df_f[is_test].copy() # This is the same as test data, just as a placeholder
            df_f_test = df_f[is_test].copy()

            ind_train = df[~is_test].index
            ind_valid = df[is_test].index
            ind_test = df[is_test].index

            test_dates = [dt.date.fromordinal(d[0]) for d in time_plot.loc[ind_test].values]

            time_train = time.loc[ind_train].values
            time_valid = time.loc[ind_valid].values
            time_test = time.loc[ind_test].values
            time_train_plot = time_plot.loc[ind_train].values
            time_valid_plot = time_plot.loc[ind_valid].values
            time_test_plot = time_plot.loc[ind_test].values
            month_train = np.array([(date_ref+dt.timedelta(days=int(time_train[it]))).month for it in range(len(time_train))])
            month_valid = np.array([(date_ref+dt.timedelta(days=int(time_valid[it]))).month for it in range(len(time_valid))])
            month_test  = np.array([(date_ref+dt.timedelta(days=int(time_test[it]))).month for it in range(len(time_test))])

            train_years = np.unique([dt.date.fromordinal(d[0]).year for d in time_plot.loc[ind_train].values])
            valid_years = np.unique([dt.date.fromordinal(d[0]).year for d in time_plot.loc[ind_valid].values])


            [model_out_all,
            history_test_all['loss'], history_test_all['val_loss'],
            target_time_train,_,target_time_test,
            y_pred_train,_,y_pred_test,
            y_train,_,y_test,
            y_clim_train,_,y_clim_test] = execute_fold_tpf(df_t_train, df_t_valid, df_t_test,
                                                            df_p_train, df_p_valid, df_p_test,
                                                            df_f_train, df_f_valid, df_f_test,
                                                            [np.where(time.index == ind)[0][0] for ind in ind_train.values], [np.where(time.index == ind)[0][0] for ind in ind_valid.values], [np.where(time.index == ind)[0][0] for ind in ind_test.values],
                                                            train_years,time.values,
                                                            time_train,time_valid,time_test,
                                                            time_train_plot,time_valid_plot,time_test_plot,
                                                            month_train,month_valid,month_test,
                                                            latent_dim,nb_layers,inp_dropout,rec_dropout,dense_act_func,
                                                            len(predictor_vars),n_f_vars,
                                                            input_len,pred_len,
                                                            norm_type,anomaly_target,anomaly_past,anomaly_frcst,
                                                            batch_size,lr,
                                                            loss,use_exp_decay_loss,tau,weights_on_timesteps,added_weight,Tw_thresh,
                                                            n_epochs,
                                                            seed,
                                                            perfect_forecast,
                                                            ensemble_mean_fcst,
                                                            fixed_seed =fixed_seed,
                                                            plot_loss=False,
                                                            plot_predictions=False,
                                                            plot_targets=False,
                                                            show_modelgraph=False,
                                                            show_weights=False)


            # SAVE ALL VARIABLES FOR THAT TEST YEAR.
            if save_model_outputs:
                if dense_act_func is None:
                    dense_act_func_name = 'None'
                else:
                    dense_act_func_name = dense_act_func
                np.savez('./output/encoderdecoder_horizon'+str(pred_len)+'_context'+str(input_len)+'_nneurons'+str(latent_dim)+'_seed'+str(seed)+'_nepochs'+str(n_epochs)+(nb_layers > 1)*('_'+str(nb_layers)+'layers')+'_'+norm_type+'_'+dense_act_func_name+'_'+loss_name+anomaly_target*'_anomaly_target'+suffix+'_'+str(yr_test),
                        target_time_train=target_time_train,
                        target_time_valid=target_time_valid_all,
                        target_time_test=target_time_test,
                        y_pred_train_all=y_pred_train,
                        y_pred_valid=y_pred_valid_all,
                        y_pred_test=y_pred_test,
                        y_train=y_train,
                        y_valid=y_valid_all,
                        y_test=y_test,
                        y_clim_train=y_clim_train,
                        y_clim_valid=y_clim_valid_all,
                        y_clim_test=y_clim_test,
                        predictor_vars = predictor_vars,
                        forecast_vars = forecast_vars,
                        target_var = target_var,
                        input_len = input_len,
                        pred_len = pred_len,
                        n_epochs = n_epochs,
                        date_ref = date_ref,
                        anomaly_target = anomaly_target,
                        anomaly_past =  anomaly_past,
                        anomaly_frcst = anomaly_frcst,
                        train_yr_start = train_yr_start,
                        valid_yr_start = valid_yr_start,
                        test_yr_start = test_yr_start,
                        valid_scheme = valid_scheme,
                        nfolds = nfolds,
                        norm_type = norm_type,
                        dense_act_func = dense_act_func,
                        latent_dim = latent_dim,
                        nb_layers = nb_layers,
                        batch_size = batch_size,
                        learning_rate = lr,
                        seed = seed,
                        fixed_seed = fixed_seed,
                        inp_dropout = inp_dropout,
                        rec_dropout = rec_dropout,
                        test_history = history_test_all,
                        valid_history = history_valid_all,
                        optimizer_name = optimizer_name,
                        loss_name = loss_name,
                        )





            print('=====================================================')
            print(str(yr_test) + ' --- END')
            print('=====================================================')
            del df_t_train, df_t_test, df_t_valid
            del df_p_train, df_p_test, df_p_valid
            del df_f_train, df_f_test, df_f_valid
            del target_time_train,target_time_valid_all,target_time_test
            del y_pred_train,y_pred_valid_all,y_pred_test,
            del y_train,y_valid_all,y_test,
            del y_clim_train,y_clim_valid_all,y_clim_test
            del history_test_all, history_valid_all
            gc.collect()


    #-----------------------------
    if valid_scheme == 'standard':

        # GET TRAINING AND VALIDATION DATA SETS
        train_years = np.arange(train_yr_start,valid_yr_start)
        valid_years = np.arange(valid_yr_start,test_yr_start)
        test_years = np.arange(test_yr_start,yr_end+1)

        istart_train = np.where(time_plot == dt.date(train_yr_start, 4, 1).toordinal())[0][0]
        istart_valid = np.where(time_plot == dt.date(valid_yr_start, 4, 1).toordinal())[0][0]
        istart_test = np.where(time_plot == dt.date(test_yr_start, 4, 1).toordinal())[0][0]

        if istart_valid == istart_test:
            ind_train = np.arange(istart_train,istart_valid)
            ind_valid = np.arange(istart_valid,len(time))
            ind_test = np.arange(istart_test,len(time))
        else:
            ind_train = np.arange(istart_train,istart_valid)
            ind_valid = np.arange(istart_valid,istart_test)
            ind_test = np.arange(istart_test,len(time))


        time_train = time.iloc[ind_train].values
        time_valid = time.iloc[ind_valid].values
        time_test = time.iloc[ind_test].values
        time_train_plot = time_plot.iloc[ind_train].values
        time_valid_plot = time_plot.iloc[ind_valid].values
        time_test_plot = time_plot.iloc[ind_test].values

        month_train = np.array([(date_ref+dt.timedelta(days=int(time_train[it]))).month for it in range(len(time_train))])
        month_valid = np.array([(date_ref+dt.timedelta(days=int(time_valid[it]))).month for it in range(len(time_valid))])
        month_test  = np.array([(date_ref+dt.timedelta(days=int(time_test[it]))).month for it in range(len(time_test))])

        df_f_train = df_f.iloc[ind_train]
        df_f_valid = df_f.iloc[ind_valid]
        df_f_test = df_f.iloc[ind_test]

        df_t_train = df_t.iloc[ind_train]
        df_t_valid = df_t.iloc[ind_valid]
        df_t_test = df_t.iloc[ind_test]

        df_p_train = df_p.iloc[ind_train]
        df_p_valid = df_p.iloc[ind_valid]
        df_p_test = df_p.iloc[ind_test]

        history = dict()

        [model_out,history['loss'], history['val_loss'],
        target_time_train,target_time_valid,target_time_test,
        y_pred_train,y_pred_valid,y_pred_test,
        y_train,y_valid,y_test,
        y_clim_train,y_clim_valid,y_clim_test] = execute_fold_tpf(df_t_train, df_t_valid, df_t_test,
                                                                  df_p_train, df_p_valid, df_p_test,
                                                                  df_f_train, df_f_valid, df_f_test,
                                                                  ind_train, ind_valid, ind_test,
                                                                  train_years,time.values,
                                                                  time_train,time_valid,time_test,
                                                                  time_train_plot,time_valid_plot,time_test_plot,
                                                                  month_train,month_valid,month_test,
                                                                  latent_dim,nb_layers,inp_dropout,rec_dropout,dense_act_func,
                                                                  len(predictor_vars),n_f_vars,
                                                                  input_len,pred_len,
                                                                  norm_type,anomaly_target,anomaly_past,anomaly_frcst,
                                                                  batch_size,lr,
                                                                  loss,use_exp_decay_loss,tau,weights_on_timesteps,added_weight,Tw_thresh,
                                                                  n_epochs,
                                                                  seed,
                                                                  perfect_forecast,
                                                                  ensemble_mean_fcst,
                                                                  fixed_seed =fixed_seed,
                                                                  plot_loss=True,
                                                                  plot_predictions=False,
                                                                  plot_targets=False,
                                                                  show_modelgraph=False,
                                                                  show_weights=False)


        if save_model_outputs:
            if dense_act_func is None:
                dense_act_func_name = 'None'
            else:
                dense_act_func_name = dense_act_func
            np.savez('./output/encoderdecoder_horizon'+str(pred_len)+'_context'+str(input_len)+'_nneurons'+str(latent_dim)+'_seed'+str(seed)+'_nepochs'+str(n_epochs)+(nb_layers > 1)*('_'+str(nb_layers)+'layers')+'_'+norm_type+'_'+dense_act_func_name+'_'+loss_name+anomaly_target*'_anomaly_target'+suffix,
            # np.savez('./output/encoderdecoder_horizon'+str(pred_len)+'_context'+str(input_len)+'_nneurons'+str(latent_dim)+'_nepochs'+str(n_epochs)+(nb_layers > 1)*('_'+str(nb_layers)+'layers')+'_'+norm_type+'_'+dense_act_func_name+'_'+loss_name+anomaly_target*'_anomaly_target'+'TEST_800epochs_nocallbacks',
                    target_time_train=target_time_train,
                    target_time_valid=target_time_valid,
                    target_time_test=target_time_test,
                    y_pred_train_all=y_pred_train,
                    y_pred_valid=y_pred_valid,
                    y_pred_test=y_pred_test,
                    y_train=y_train,
                    y_valid=y_valid,
                    y_test=y_test,
                    y_clim_train=y_clim_train,
                    y_clim_valid=y_clim_valid,
                    y_clim_test=y_clim_test,
                    predictor_vars = predictor_vars,
                    forecast_vars = forecast_vars,
                    target_var = target_var,
                    input_len = input_len,
                    pred_len = pred_len,
                    n_epochs = n_epochs,
                    date_ref = date_ref,
                    anomaly_target = anomaly_target,
                    anomaly_past =  anomaly_past,
                    anomaly_frcst = anomaly_frcst,
                    train_yr_start = train_yr_start,
                    valid_yr_start = valid_yr_start,
                    test_yr_start = test_yr_start,
                    valid_scheme = valid_scheme,
                    nfolds = nfolds,
                    norm_type = norm_type,
                    dense_act_func = dense_act_func,
                    latent_dim = latent_dim,
                    nb_layers = nb_layers,
                    batch_size = batch_size,
                    learning_rate = lr,
                    seed = seed,
                    fixed_seed = fixed_seed,
                    inp_dropout = inp_dropout,
                    rec_dropout = rec_dropout,
                    test_history = history,
                    valid_history = history,
                    optimizer_name = optimizer_name,
                    loss_name = loss_name,
                    )


end_time = timer.time()
print('=====================================================')
print('END!')
print('Total time: ' + str(end_time - start_time) + ' seconds')

#%%

fig_hist,ax_hist = plt.subplots()
ax_hist.plot(history['loss'],label='train loss')
ax_hist.plot(history['val_loss'],label='test loss')
ax_hist.set_xlabel('epoch')
ax_hist.set_ylabel('loss')
# ax_hist.legend()
# ax_hist.set_title(str(yr_test))





#%% MAKE SOME PLOTS
# pred_type = 'valid'
pred_type = 'train'
pred_type = 'test'
if pred_type == 'valid':
    y = y_valid
    y_pred = y_pred_valid
    y_clim = y_clim_valid
    target_time = target_time_valid
if pred_type == 'test':
    y = y_test
    y_pred = y_pred_test
    y_clim = y_clim_test
    target_time = target_time_test
if pred_type == 'train':
    y = y_train
    y_pred = y_pred_train
    y_clim = y_clim_train
    target_time = target_time_train

plot_prediction_timeseries(y_pred,y,y_clim,target_time,pred_type, lead=0, nyrs_plot= 28)
plot_prediction_timeseries(y_pred,y,y_clim,target_time,pred_type, lead=pred_len-1, nyrs_plot= 28)

#%%
plot_sample(y_pred,y,y_clim,target_time,it=120,pred_type=pred_type,show_clim=True)
plot_sample(y_pred,y,y_clim,target_time,it=490,pred_type=pred_type,show_clim=True)

#%%





