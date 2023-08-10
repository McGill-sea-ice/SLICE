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

from functions_encoderdecoder import fit_scaler, normalize_df, get_predictor_clim, replace_nan_with_clim, create_dataset, reconstruct_ysamples, encoder_decoder_recursive, execute_fold
from functions_encoderdecoder import SEAS5_dailyarray_to_ts, obs_dailyts_to_forecast_ts
from functions_encoderdecoder import execute_fold_test
from functions_encoderdecoder import create_dataset_test
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

# Set random seed:
fixed_seed = True
# fixed_seed = False

seed = 422
if fixed_seed:
    tf.keras.utils.set_random_seed(seed)

#%% CHOOSE MODEL SETTINGS:

start_time = timer.time()

# If using 'LOOk', need to specify how many folds to use for validation:
valid_scheme = 'LOOk'
nfolds = 5

# If using'standard', need to specify bounds for train/valid/test sets:
valid_scheme = 'standard'
train_yr_start = 1992 # Training dataset: 1993 - 2010
valid_yr_start = 2011 # Validation dataset: 2011 - 2015
test_yr_start = 2016 # Testing dataset: 2016 - 2021


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
# lr = 0.001
# lr = 0.002
# lr = 0.004
# lr = 0.008
lr = 0.016
# lr = 0.032
# lr = 0.064
# lr = 0.128
# lr = 0.256


# Choose loss function:
loss = 'MSETw'
# loss = 'MSETw_MSEdTwdt'
# loss = 'MSETw_penalize_FN' # Will not work with target anomalies
# loss = 'MSETw_penalize_recall' # Will not work with target anomalies
# loss = 'log_loss'

use_exp_decay_loss = True
tau = 30

# Will not work with target anomalies
weights_on_timesteps = False
added_weight = 1.
Tw_thresh = 0.75


loss_name = (loss
             +weights_on_timesteps*('_with_weights'+str(added_weight).rsplit('.')[0]+'_on_thresh'+str(Tw_thresh).rsplit('.')[0]+'_'+str(Tw_thresh).rsplit('.')[1])
             +use_exp_decay_loss*('_exp_decay_tau'+str(tau))
             )


# Set max. number of epochs
# n_epochs = 50
n_epochs = 2

# Number of hidden neurons in Encoder and Decoder
latent_dim = 50
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
# pred_len = 4

# Input window length, in days
input_len = 128

# Select variables to use:
predictor_vars = ['Avg. Ta_mean',
                  # 'Avg. Ta_min',
                  # 'Avg. Ta_max',
                  'Avg. cloud cover',
                  'Tot. snowfall',
                  'NAO',
                  'Avg. Twater',
                  'Avg. discharge St-L. River',
                  'Avg. level Ottawa River',
                  #'Avg. SH (sfc)',
                  #'Avg. SW down (sfc)',
                  ]
# predictor_vars = ['Avg. Ta_max']

perfect_forecast = True
# perfect_forecast = False
forecast_vars = ['Avg. Ta_mean',
                'Tot. snowfall',
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

# save_model_outputs = True
save_model_outputs = False

# suffix = '_perfectexp_Ta_mean_cloud_cover_snowfall'
# suffix = '_perfectexp_Ta_mean_snowfall'
# suffix = '_perfectexp_Ta_mean'
# suffix = '_perfectexp'
# suffix = '_no_forecast_vars'
# suffix = '_perfectexp_noTw'
suffix = '_REF_MSETw_test_new_forecast_method'

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
    time = df['Days since 1900-01-01'].values
    df.drop(columns='Days since 1900-01-01', inplace=True)

    # Keep only data for 1992-2020.
    yr_start = 1992
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
    first_day = (date_ref+dt.timedelta(days=int(time[0]))).toordinal()
    last_day = (date_ref+dt.timedelta(days=int(time[-1]))).toordinal()
    time_plot = np.arange(first_day, last_day + 1)

    plot_target = False
    if plot_target:
        # Plot the observed water temperature time series, which will be the target variable
        fig, ax = plt.subplots(figsize=[12, 6])
        ax.plot(time_plot[:], df['Avg. Twater'][:], color='C0', label='T$_w$')
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
                obs_ts = obs_dailyts_to_forecast_ts(obs_f,month,time)
                df_f[:,iv*12+month-1] = obs_ts
    else:
        # Load SEAS5 as forecast variables
        fdir_SEAS5 = local_path+'slice/prog/analysis/SEAS5/'
        fname_SEAS5 = 'SEAS5_SEAS51_MLR_forecast_daily_predictors.npz'
        data_SEAS5 = np.load(fdir_SEAS5+fname_SEAS5,allow_pickle=True)
        SEAS5_years = np.arange(1981,2022+1)
        time_SEAS5 = np.arange((dt.date(SEAS5_years[0],1,1)-date_ref).days,(dt.date(SEAS5_years[-1]+1,1,1)-date_ref).days)
        it0_SEAS5 = np.where(time_SEAS5 == time[0])[0][0]
        it1_SEAS5 = np.where(time_SEAS5 == time[-1])[0][0]

        df_f = np.zeros((len(time),len(forecast_vars)*12))*np.nan
        df_f_col = []
        for iv,fv in enumerate(forecast_vars):
            if fv == 'Avg. Ta_mean':
                sv = 'SEAS5_Ta_mean'
            elif fv == 'Tot. snowfall':
                sv = 'SEAS5_snowfall'
            else:
                raise('SEAS5 variable not yet implemented')
            SEAS5_f = data_SEAS5[sv]
            SEAS5_f_em = np.nanmean(SEAS5_f,axis = 3)
            for month in range(1,13):
                df_f_col = df_f_col + [sv]
                SEAS5_ts = SEAS5_dailyarray_to_ts(SEAS5_f_em,SEAS5_years,month,time_SEAS5)
                df_f[:,iv*12+month-1] = SEAS5_ts[it0_SEAS5:it1_SEAS5+1]



    df_f = pd.DataFrame(df_f,columns=df_f_col) # Forecast variables
    df_t = df[target_var] # Target variable
    df_p = df[predictor_vars] # Predictor variables

    df = df[target_var+predictor_vars+forecast_vars]


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

        kf = KFold(n_splits=nfolds)

        for iyr_test,yr_test in enumerate(test_years):
        # for iyr_test,yr_test in enumerate(test_years[10:12]):
            df_tmp = df.copy()
            istart_yr_test = np.where(time_plot == dt.date(yr_test, 4, 1).toordinal())[0][0]
            iend_yr_test = np.where(time_plot == dt.date(yr_test+1, 4, 1).toordinal())[0][0]

            is_test = np.zeros(df.shape[0], dtype=bool)
            is_test[istart_yr_test:iend_yr_test] = True

            # Mask test year from data set:
            df_tmp[is_test] = np.nan

            df_kfold = df[~is_test].copy()
            ind_kfold = df[~is_test].index

            df_test_fold = df[is_test].copy()
            ind_test_fold = df[is_test].index  # NOTE: THESE ARE THE SAME INDICES AS WHERE is_test IS TRUE
            test_dates = [dt.date.fromordinal(d) for d in time_plot[ind_test_fold]]
            time_test = time[ind_test_fold]
            time_test_plot = time_plot[ind_test_fold]

            y_valid_all = np.zeros((nfolds,365*28,pred_len))*np.nan
            y_pred_valid_all = np.zeros((nfolds,365*28,pred_len))*np.nan
            y_clim_valid_all = np.zeros((nfolds,365*28,pred_len))*np.nan
            target_time_valid_all = np.zeros((nfolds,365*28,pred_len))*np.nan
            history_valid_all = dict()
            history_test_all = dict()

            # Get cross-validation folds (determined using all years other than test year)
            for ifold,[train_index, valid_index] in enumerate(kf.split(df_kfold.index)):
                # if (ifold == 2) | (ifold == 3): # REMOVE THIS AFTER!!!!!!!!!!!1

                # GET TRAINING AND VALIDATION DATA SETS FOR THIS FOLD:
                df_train_fold, df_valid_fold = df_kfold.iloc[train_index], df_kfold.iloc[valid_index]
                ind_train_fold, ind_valid_fold = ind_kfold[train_index], ind_kfold[valid_index]

                train_dates = [dt.date.fromordinal(d) for d in time_plot[ind_train_fold]]
                valid_dates = [dt.date.fromordinal(d) for d in time_plot[ind_valid_fold]]

                time_train = time[ind_train_fold]
                time_valid = time[ind_valid_fold]
                time_train_plot = time_plot[ind_train_fold]
                time_valid_plot = time_plot[ind_valid_fold]

                train_years = np.unique([dt.date.fromordinal(d).year for d in time_plot[ind_train_fold]])
                valid_years = np.unique([dt.date.fromordinal(d).year for d in time_plot[ind_valid_fold]])

                [model_out,
                 history_valid_all['loss'+str(ifold)], history_valid_all['val_loss'+str(ifold)],
                _,target_time_valid,_,
                _,y_pred_valid,_,
                _,y_valid,_,
                _,y_clim_valid,_] = execute_fold(df_train_fold, df_valid_fold, df_test_fold,
                                                    ind_train_fold, ind_valid_fold, ind_test_fold,
                                                    train_years,time,
                                                    time_train,time_valid,time_test,
                                                    time_train_plot,time_valid_plot,time_test_plot,
                                                    latent_dim,nb_layers,inp_dropout,rec_dropout,dense_act_func,
                                                    len(predictor_vars),len(forecast_vars),
                                                    input_len,pred_len,
                                                    norm_type,anomaly_target,anomaly_past,anomaly_frcst,
                                                    batch_size,lr,
                                                    loss,use_exp_decay_loss,tau,weights_on_timesteps,added_weight,Tw_thresh,
                                                    n_epochs,
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
            del df_kfold, df_tmp, df_test_fold
            gc.collect()

            # Then, train with all other years to predict test year:
            df_train = df[~is_test].copy()
            df_valid = df[is_test].copy() # This is the same as test data, just as a placeholder
            df_test = df[is_test].copy()

            ind_train = df[~is_test].index
            ind_valid = df[is_test].index
            ind_test = df[is_test].index

            test_dates = [dt.date.fromordinal(d) for d in time_plot[ind_test]]

            time_train = time[ind_train]
            time_valid = time[ind_valid]
            time_test = time[ind_test]
            time_train_plot = time_plot[ind_train]
            time_valid_plot = time_plot[ind_valid]
            time_test_plot = time_plot[ind_test]

            train_years = np.unique([dt.date.fromordinal(d).year for d in time_plot[ind_train]])
            valid_years = np.unique([dt.date.fromordinal(d).year for d in time_plot[ind_valid]])

            [model_out_all,
            history_test_all['loss'], history_test_all['val_loss'],
            target_time_train,_,target_time_test,
            y_pred_train,_,y_pred_test,
            y_train,_,y_test,
            y_clim_train,_,y_clim_test] = execute_fold(df_train, df_valid, df_test,
                                                        ind_train, ind_valid, ind_test,
                                                        train_years,time,
                                                        time_train,time_valid,time_test,
                                                        time_train_plot,time_valid_plot,time_test_plot,
                                                        latent_dim,nb_layers,inp_dropout,rec_dropout,dense_act_func,
                                                        len(predictor_vars),len(forecast_vars),
                                                        input_len,pred_len,
                                                        norm_type,anomaly_target,anomaly_past,anomaly_frcst,
                                                        batch_size,lr,
                                                        loss,use_exp_decay_loss,tau,weights_on_timesteps,added_weight,Tw_thresh,
                                                        n_epochs,
                                                        fixed_seed =fixed_seed,
                                                        plot_loss=False,
                                                        plot_predictions=False,
                                                        plot_targets=False,
                                                        show_modelgraph=False,
                                                        show_weights=False)

            # SAVE ALL VARIABLES FOR THAT TEST YEAR.
            if save_model_outputs:
                if dense_act_func is None: dense_act_func_name = 'None'
                np.savez('./output/encoderdecoder_horizon'+str(pred_len)+'_context'+str(input_len)+'_nneurons'+str(latent_dim)+'_nepochs'+str(n_epochs)+(nb_layers > 1)*('_'+str(nb_layers)+'layers')+'_'+norm_type+'_'+dense_act_func_name+'_'+loss_name+anomaly_target*'_anomaly_target'+suffix+'_'+str(yr_test),
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
            del df_train, df_test, df_valid
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
        ind_train = np.arange(istart_train,istart_valid)
        ind_valid = np.arange(istart_valid,istart_test)
        ind_test = np.arange(istart_test,len(time))

        time_train = time[ind_train]
        time_valid = time[ind_valid]
        time_test = time[ind_test]
        time_train_plot = time_plot[ind_train]
        time_valid_plot = time_plot[ind_valid]
        time_test_plot = time_plot[ind_test]

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

        df_train = df.iloc[ind_train]
        df_valid = df.iloc[ind_valid]
        df_test = df.iloc[ind_test]

        history = dict()

        [model_out,history['loss'], history['val_loss'],
        target_time_train,target_time_valid,target_time_test,
        y_pred_train,y_pred_valid,y_pred_test,
        y_train,y_valid,y_test,
        y_clim_train,y_clim_valid,y_clim_test] = execute_fold_test(df_train, df_valid, df_test,
                                                                  df_t_train, df_t_valid, df_t_test,
                                                                  df_p_train, df_p_valid, df_p_test,
                                                                  df_f_train, df_f_valid, df_f_test,
                                                                  ind_train, ind_valid, ind_test,
                                                                  train_years,time,
                                                                  time_train,time_valid,time_test,
                                                                  time_train_plot,time_valid_plot,time_test_plot,
                                                                  month_train,month_valid,month_test,
                                                                  latent_dim,nb_layers,inp_dropout,rec_dropout,dense_act_func,
                                                                  len(predictor_vars),len(forecast_vars),
                                                                  input_len,pred_len,
                                                                  norm_type,anomaly_target,anomaly_past,anomaly_frcst,
                                                                  batch_size,lr,
                                                                  loss,use_exp_decay_loss,tau,weights_on_timesteps,added_weight,Tw_thresh,
                                                                  n_epochs,
                                                                  perfect_forecast,
                                                                  fixed_seed =fixed_seed,
                                                                  plot_loss=True,
                                                                  plot_predictions=False,
                                                                  plot_targets=False,
                                                                  show_modelgraph=False,
                                                                  show_weights=False)

        # [model_out,history['loss'], history['val_loss'],
        # target_time_train,target_time_valid,target_time_test,
        # y_pred_train,y_pred_valid,y_pred_test,
        # y_train,y_valid,y_test,
        # y_clim_train,y_clim_valid,y_clim_test] = execute_fold(df_train, df_valid, df_test,
        #                                                           ind_train, ind_valid, ind_test,
        #                                                           train_years,time,
        #                                                           time_train,time_valid,time_test,
        #                                                           time_train_plot,time_valid_plot,time_test_plot,
        #                                                           latent_dim,nb_layers,inp_dropout,rec_dropout,dense_act_func,
        #                                                           len(predictor_vars),len(forecast_vars),
        #                                                           input_len,pred_len,
        #                                                           norm_type,anomaly_target,anomaly_past,anomaly_frcst,
        #                                                           batch_size,lr,
        #                                                           loss,use_exp_decay_loss,tau,weights_on_timesteps,added_weight,Tw_thresh,
        #                                                           n_epochs,
        #                                                           fixed_seed =fixed_seed,
        #                                                           plot_loss=True,
        #                                                           plot_predictions=False,
        #                                                           plot_targets=False,
        #                                                           show_modelgraph=False,
        #                                                           show_weights=False)


        if save_model_outputs:
            if dense_act_func is None: dense_act_func_name = 'None'
            np.savez('./output/encoderdecoder_horizon'+str(pred_len)+'_context'+str(input_len)+'_nneurons'+str(latent_dim)+'_nepochs'+str(n_epochs)+(nb_layers > 1)*('_'+str(nb_layers)+'layers')+'_'+norm_type+'_'+dense_act_func_name+'_'+loss_name+anomaly_target*'_anomaly_target'+suffix,
            # np.savez('./output/TEST,
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







#%% MAKE SOME PLOTS
from functions_ML import regression_metrics, plot_sample, plot_prediction_timeseries
from analysis.ML.colab.compare_encoderdecoder_output import load_data_year
from analysis.ML.colab.compare_encoderdecoder_output import plot_Tw_metric, evaluate_Tw_forecasts
from analysis.ML.colab.compare_encoderdecoder_output import detect_FUD_from_Tw_samples, evaluate_FUD_forecasts


pred_type = 'valid'
# pred_type = 'train'
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
plot_prediction_timeseries(y_pred,y,y_clim,target_time,pred_type, lead=29, nyrs_plot= 28)

plot_sample(y_pred,y,y_clim,target_time,it=125,pred_type=pred_type,show_clim=True)


# TRAINING ---
# Rsqr = 0.9943
# MAE = 0.4762
# RMSE = 0.6301

# VALIDATION ---
# Rsqr = 0.9936
# MAE = 0.5161
# RMSE = 0.6808

# TEST ---
# Rsqr = 0.9929
# MAE = 0.5672
# RMSE = 0.7384

#%%



#%% BELOW IS A SPACE TO TEST MAKING WINDOWED PREDICTORS WITH FORECATS VARIABLES SEPARATED BY LEAD TIME
# def create_dataset(df, df_clim, time_in,
#                     n_forecasts,
#                     window_size, forecast_size,
#                     batch_size,
#                     shuffle=False):
#     """
#     SEE WEB EXAMPLE:
#     (https://www.angioi.com/time-series-encoder-decoder-tensorflow/)
#     """

#     # Total size of window is given by the number of steps to be considered
#     # before prediction time + steps that we want to forecast
#     total_size = window_size + forecast_size

#     # Selecting windows
#     data = tf.data.Dataset.from_tensor_slices(df.values)
#     data = data.window(total_size, shift=1, drop_remainder=True)
#     data = data.flat_map(lambda k: k.batch(total_size))

#     data_clim = tf.data.Dataset.from_tensor_slices(df_clim.values)
#     data_clim = data_clim.window(total_size, shift=1, drop_remainder=True)
#     data_clim = data_clim.flat_map(lambda k: k.batch(total_size))

#     time = tf.data.Dataset.from_tensor_slices(time_in)
#     time = time.window(total_size, shift=1, drop_remainder=True)
#     time = time.flat_map(lambda k: k.batch(total_size))

#     # Zip all datasets together so that we can filter out the samples
#     # that are discontinuous in time due to cross-validation splits.
#     all_ds = tf.data.Dataset.zip((data, data_clim, time))
#     all_ds_filtered =  all_ds.filter(lambda d,dc,t: tf.math.equal(t[-1]-t[0]+1,total_size))

#     # Then extract the separate data sets
#     data_filtered = all_ds_filtered.map(lambda d,dc,t: d)
#     data_clim_filtered =  all_ds_filtered.map(lambda d,dc,t: dc)
#     time_filtered =  all_ds_filtered.map(lambda d,dc,t: t)

#     # Shuffling data
#     # !!!!! NOT SURE HOW TO DEAL WITH SHUFFLE AND RECONSTRUCT THE SHUFFLED TIME SERIES...
#     # so we keep shuffle to False for now...
#     shuffle = False
#     if shuffle:
#         shuffle_buffer_size = len(data_filtered) # This number can be changed
#     #     data = data.shuffle(shuffle_buffer_size, seed=42)
#         data_filtered = data_filtered.shuffle(shuffle_buffer_size, seed=42)
#         data_clim_filtered =  data_clim_filtered.shuffle(shuffle_buffer_size, seed=42)
#         time_filtered =  time_filtered.shuffle(shuffle_buffer_size, seed=42)

#     # Extracting (past features, forecasts, decoder initial recurrent input) + targets
#     # NOTE : the initial decoder input is set as the last value of the target.
#     if n_forecasts > 0:
#         data_filtered = data_filtered.map(lambda k: ((k[:-forecast_size,1:-n_forecasts], # Past predictors samples
#                                                       k[-forecast_size:, -n_forecasts:], # Future forecasts samples
#                                                       k[-forecast_size-1:-forecast_size,0:1] # Decoder input: last time step of target before prediction time starts
#                                                       ),
#                                                   k[-forecast_size:, 0:1])) # Target samples during prediction time

#         data_clim_filtered = data_clim_filtered.map(lambda k: (k[:-forecast_size,1:-n_forecasts], # Past predictor climatology samples
#                                                                 k[-forecast_size:,0:1])) # Target climatology samples during prediction time



#     else:
#         data_filtered = data_filtered.map(lambda k: ((k[:-forecast_size,1:],  # Past predictors samples
#                                                       k[-forecast_size-1:-forecast_size,0:1] # Decoder input: last time step of target before prediction time starts
#                                                       ),
#                                                   k[-forecast_size:, 0:1])) # Target samples during prediction time

#         data_clim_filtered = data_clim_filtered.map(lambda k: (k[:-forecast_size,1:], # Past predictor climatology samples
#                                                                 k[-forecast_size:,0:1])) # Target climatology samples during prediction time

#     time_filtered = time_filtered.map(lambda k: (k[:-forecast_size], # Time for past predictors samples
#                                                   k[-forecast_size:]))    # Time for prediction samples



#     return data_filtered.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE), data_clim_filtered.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE), time_filtered.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


df_t_train = df_t.iloc[ind_train]
df_t_valid = df_t.iloc[ind_valid]
df_t_test = df_t.iloc[ind_test]

df_p_train = df_p.iloc[ind_train]
df_p_valid = df_p.iloc[ind_valid]
df_p_test = df_p.iloc[ind_test]

df_f_train = df_f.iloc[ind_train]
df_f_valid = df_f.iloc[ind_valid]
df_f_test = df_f.iloc[ind_test]

df_train = df.iloc[ind_train]
df_valid = df.iloc[ind_valid]
df_test = df.iloc[ind_test]
#%%
print('TRAIN')
print('forecast')
test = df_f_train.iloc[:,0:12].values
print(np.sum(np.nanmean(test,axis=1) != df_train.iloc[:,-2].values))
print(np.nanmean(np.abs(np.nanmean(test,axis=1) - df_train.iloc[:,-2].values)))
# print(np.sum(np.abs(np.nanmean(test,axis=1)- df_train.iloc[:,-2].values) > 0.00000000000001))
# print(np.sum(np.abs(np.nanmean(test,axis=1)- df_train.iloc[:,-2].values) > 0.0000000000000000001))
print('predictors')
test = df_p_train.iloc[:,:].values
print(np.sum(test != df_train.iloc[:,1:-2].values))
print(np.nanmean(np.abs(test - df_train.iloc[:,1:-2].values)))
# print(np.sum(np.abs(test- df_train.iloc[:,1:-2].values) > 0.00000000000001))
# print(np.sum(np.abs(test- df_train.iloc[:,1:-2].values) > 0.0000000000000000000000000001))
print('targets')
test = df_t_train.iloc[:,:].values
print(np.sum(test != df_train.iloc[:,0:1].values))
print(np.nanmean(np.abs(test - df_train.iloc[:,0:1].values)))
# print(np.sum(np.abs(test- df_train.iloc[:,0:1].values) > 0.00000000000001))
# print(np.sum(np.abs(test- df_train.iloc[:,0:1].values) > 0.000000000000000000000001))
print('--------')

#%%
i=0
plt.figure()
plt.plot(time_train_plot,df_train.iloc[:,-2])
plt.plot(time_train_plot,df_f_train.iloc[:,i])

plt.figure()
plt.plot(time_train_plot,np.abs(df_train.iloc[:,-2]-df_f_train.iloc[:,i]))
print(np.nanmean(np.abs(df_train.iloc[:,-2]-df_f_train.iloc[:,i])))

#%%
i=11
plt.figure()
plt.plot(time_train_plot,df_train.iloc[:,-1])
plt.plot(time_train_plot,df_f_train.iloc[:,i+12])

plt.figure()
plt.plot(time_train_plot,np.abs(df_train.iloc[:,-1]-df_f_train.iloc[:,i+12]))
print(np.nanmean(np.abs(df_train.iloc[:,-1]-df_f_train.iloc[:,i+12])))

#%%
print('VALID')
print('forecast')
test = df_f_valid.iloc[:,0:12].values
print(np.sum(np.nanmean(test,axis=1) != df_valid.iloc[:,-2].values))
print(np.nanmean(np.abs(np.nanmean(test,axis=1) - df_valid.iloc[:,-2].values)))
# print(np.sum(np.abs(np.nanmean(test,axis=1)- df_valid.iloc[:,-2].values) > 0.00000000000001))
# print(np.sum(np.abs(np.nanmean(test,axis=1)- df_valid.iloc[:,-2].values) > 0.00000000000000001))
print('predictors')
test = df_p_valid.iloc[:,:].values
print(np.sum(test != df_valid.iloc[:,1:-2].values))
print(np.nanmean(np.abs(test - df_valid.iloc[:,1:-2].values)))
# print(np.sum(np.abs(test- df_valid.iloc[:,1:-2].values) > 0.00000000000001))
# print(np.sum(np.abs(test- df_valid.iloc[:,1:-2].values) > 0.00000000000000001))
print('targets')
test = df_t_valid.iloc[:,:].values
print(np.sum(test != df_valid.iloc[:,0:1].values))
print(np.nanmean(np.abs(test - df_valid.iloc[:,0:1].values)))
# print(np.sum(np.abs(test- df_valid.iloc[:,0:1].values) > 0.00000000000001))
# print(np.sum(np.abs(test- df_valid.iloc[:,0:1].values) > 0.00000000000000001))
print('--------')

#%%
print('TEST')
print('forecast')
test = df_f_test.iloc[:,0:12].values
print(np.sum(np.nanmean(test,axis=1) != df_test.iloc[:,-2].values))
print(np.nanmean(np.abs(np.nanmean(test,axis=1) - df_test.iloc[:,-2].values)))
# print(np.sum(np.abs(np.nanmean(test,axis=1)- df_test.iloc[:,-2].values) > 0.00000000000001))
# print(np.sum(np.abs(np.nanmean(test,axis=1)- df_test.iloc[:,-2].values) > 0.00000000000000001))
print('predictors')
test = df_p_test.iloc[:,:].values
print(np.sum(np.nanmean(test,axis=1) != df_test.iloc[:,1:-2].values))
print(np.nanmean(np.abs(test - df_test.iloc[:,1:-2].values)))
# print(np.sum(np.abs(test- df_test.iloc[:,1:-2].values) > 0.000000000000001))
# print(np.sum(np.abs(test- df_test.iloc[:,1:-2].values) > 0.00000000000000001))
print('targets')
test = df_t_test.iloc[:,:].values
print(np.sum(test != df_test.iloc[:,0:1].values))
print(np.nanmean(np.abs(test - df_test.iloc[:,0:1].values)))
# print(np.sum(np.abs(test- df_test.iloc[:,0:1].values) > 0.000000000000001))
# print(np.sum(np.abs(test- df_test.iloc[:,0:1].values) > 0.000000000000000001))
print('--------')

#%%
df_t_train_in = df_t_train
df_t_valid_in = df_t_valid
df_t_test_in = df_t_test

df_p_train_in = df_p_train
df_p_valid_in = df_p_valid
df_p_test_in = df_p_test

df_f_train_in = df_f_train
df_f_valid_in = df_f_valid
df_f_test_in = df_f_test

ind_train_in = ind_train
ind_valid_in = ind_valid
ind_test_in = ind_test

df_train_in = df_train
df_valid_in = df_valid
df_test_in = df_test


df_t_train_clim, df_t_valid_clim, df_t_test_clim = get_predictor_clim(df_t_train_in,df_t_valid_in,df_t_test_in,
                                                                      ind_train_in,ind_valid_in,ind_test_in,
                                                                      time_train,train_years,time,
                                                                      df_is_frcst = False,nw=1,verbose = True)

df_p_train_clim, df_p_valid_clim, df_p_test_clim = get_predictor_clim(df_p_train_in,df_p_valid_in,df_p_test_in,
                                                                      ind_train_in,ind_valid_in,ind_test_in,
                                                                      time_train,train_years,time,
                                                                      df_is_frcst = False,nw=1,verbose = True)

df_f_train_clim, df_f_valid_clim, df_f_test_clim = get_predictor_clim(df_f_train_in, df_f_valid_in, df_f_test_in,
                                                                      ind_train_in,ind_valid_in,ind_test_in,
                                                                      time_train,train_years,time,
                                                                      df_is_frcst = True,nw=1,verbose = True)

df_train_clim, df_valid_clim, df_test_clim = get_predictor_clim(df_train_in,df_valid_in,df_test_in,
                                                                ind_train_in,ind_valid_in,ind_test_in,
                                                                time_train,train_years,time,
                                                                df_is_frcst = False,nw=1,verbose = True)




#%%

nw = 1
v = 0
month = 11-1
f_ts_clim_m, _, _ = rolling_climo(nw,df_f_train_in.iloc[:,(12*v)+month].values,'other',time_train,train_years,time_other=time)
f_ts_clim, _, _ = rolling_climo(nw,df_train_in.iloc[:,-2].values,'other',time_train,train_years,time_other=time)

print(np.nanmean(np.abs(df_f_train_in.iloc[:,(12*v)+month]-df_train_in.iloc[:,-2])))
print(np.nanmean(np.abs(f_ts_clim_m[ind_train]-f_ts_clim[ind_train])))
plt.figure()
plt.plot(time_train_plot,np.abs(f_ts_clim_m[ind_train]-f_ts_clim[ind_train]))

#%%

nw = 1
v = 0
month = 11-1
f_ts_clim_m, _, _ = rolling_climo(nw,df_f_train_in.iloc[:,(12*v)+month].values,'year',time_train,train_years)
f_ts_clim, _, _ = rolling_climo(nw,df_train_in.iloc[:,-2].values,'year',time_train,train_years)

print(np.nanmean(np.abs(df_f_train_in.iloc[:,(12*v)+month]-df_train_in.iloc[:,-2])))
print(np.nanmean(np.abs(f_ts_clim_m-f_ts_clim)))
plt.figure()
plt.plot(np.abs(f_ts_clim_m-f_ts_clim))
plt.figure()
plt.plot(f_ts_clim)
plt.plot(f_ts_clim_m)


#%%
clim_years = train_years
clim_years = np.arange(1993,2016+1)

nw = 1
v = 0
month = 10
Nwindow = nw
time_in = time_train
years = train_years
# years = clim_years
ts_in =  df_f_train_in.iloc[:,(12*v)+month].values
ts_in = df_train_in.iloc[:,-2].values
output_type = 'year'
time_other = time

# def rolling_climo(Nwindow,ts_in,output_type,time_in,years,time_other=None,date_ref = dt.date(1900,1,1)):

import calendar
ts_daily = np.zeros((Nwindow,366,len(years)))*np.nan

for it in range(ts_in.shape[0]):

    iw0 = np.max([0,it-int((Nwindow-1)/2)])
    iw1 = np.min([it+int((Nwindow-1)/2)+1,len(time_in)-1])

    ts_window = ts_in[iw0:iw1]
    # print(it==iw0,it,len(time_in),iw1,iw1-1,time_in[iw0],time_in[iw1],time_in[iw1-1])

    if ((time_in[iw1-1]-time_in[iw0]+1) <= Nwindow):
        # Only keep windows that cover continuous data, i.e.
        # exclude windows in which time jumps from one year to the
        # next due to non-sequential training intervals (in the
        # case of k-fold cross-validation, for example)

        date_mid = date_ref+dt.timedelta(days=int(time_in[it]))
        year_mid = date_mid.year
        month_mid = date_mid.month
        day_mid = date_mid.day

        if len(np.where(years == year_mid)[0]) > 0:
            iyear = np.where(years == year_mid)[0][0]
            doy = (dt.date(year_mid,month_mid,day_mid)-dt.date(year_mid,1,1)).days+1

            if iyear == 0:
                print(doy, ts_window, iw0,iw1)

            ts_daily[0:len(ts_window),doy-1,iyear] = ts_window

            if not calendar.isleap(year_mid) and (doy == 365) and (year_mid != years[-1]):
                imid = int((Nwindow-1)/2)
                ts_window_366 = np.zeros((Nwindow))*np.nan
                ts_window_366[imid] = np.array(np.nanmean([ts_in[it],ts_in[np.nanmin([len(ts_in)-1,it+1])]]))
                ts_window_366[0:imid] = ts_in[int(it+1-((Nwindow-1)/2)):it+1]
                ts_window_366[imid+1:Nwindow] = ts_in[it+1:int(it+1+((Nwindow-1)/2))]
                ts_daily[:,365,iyear] = ts_window_366

                print(it,'HEYYYYYYYYYY',iyear, year_mid)
        else:
            print('YOOOOOO', year_mid)
    else:
        print('HEREEEEEE')

# Then, find the climatological mean and std for each window/date
if output_type == 'year':
    mean_clim = np.zeros((366))*np.nan
    std_clim = np.zeros((366))*np.nan
    mean_clim[:] = np.nanmean(ts_daily,axis=(0,2))
    std_clim[:] = np.nanstd(ts_daily,axis=(0,2))

if output_type == 'all_time':
    mean_clim = np.zeros(len(time_in))*np.nan
    std_clim = np.zeros(len(time_in))*np.nan

    yr_st = (date_ref+dt.timedelta(days=int(time_in[0]))).year
    yr_end = (date_ref+dt.timedelta(days=int(time_in[-1]))).year
    all_years = np.arange(yr_st,yr_end+1)
    for iyr,year in enumerate(all_years):
        istart = np.where(time_in == (dt.date(int(year),1,1)-date_ref).days)[0][0]
        iend = np.where(time_in == (dt.date(int(year),12,31)-date_ref).days)[0][0]+1
        if not calendar.isleap(year):
            mean_clim[istart:iend] = np.nanmean(ts_daily,axis=(0,2))[:-1]
            std_clim[istart:iend] = np.nanstd(ts_daily,axis=(0,2))[:-1]
        else:
            mean_clim[istart:iend] = np.nanmean(ts_daily,axis=(0,2))
            std_clim[istart:iend] = np.nanstd(ts_daily,axis=(0,2))

if output_type == 'time_in':
    mean_clim = np.zeros(len(time_in))*np.nan
    std_clim = np.zeros(len(time_in))*np.nan

    for it in range(len(time_in)):
        day_it = int((date_ref+dt.timedelta(days=int(time_in[it]))).day)
        month_it = int((date_ref+dt.timedelta(days=int(time_in[it]))).month)
        year_it = int((date_ref+dt.timedelta(days=int(time_in[it]))).year)
        doy_it = (dt.date(year_it,month_it,day_it) - dt.date(year_it,1,1)).days + 1

        mean_clim[it] = np.nanmean(ts_daily,axis=(0,2))[doy_it-1]
        std_clim[it] = np.nanstd(ts_daily,axis=(0,2))[doy_it-1]


if output_type == 'other':
    if time_other is not None:
        mean_clim = np.zeros(len(time_other))*np.nan
        std_clim = np.zeros(len(time_other))*np.nan

        for it in range(len(time_other)):
            day_it = int((date_ref+dt.timedelta(days=int(time_other[it]))).day)
            month_it = int((date_ref+dt.timedelta(days=int(time_other[it]))).month)
            year_it = int((date_ref+dt.timedelta(days=int(time_other[it]))).year)
            doy_it = (dt.date(year_it,month_it,day_it) - dt.date(year_it,1,1)).days + 1

            mean_clim[it] = np.nanmean(ts_daily,axis=(0,2))[doy_it-1]
            std_clim[it] = np.nanstd(ts_daily,axis=(0,2))[doy_it-1]

    else:
        raise Exception('NO TIME ARRAY: The "other" output option requires a target time array.')


# return mean_clim, std_clim, ts_daily





#%%



#%%
print('TRAIN')
print('forecast')
test = df_f_train_clim.iloc[:,0:12].values
print(np.sum(np.nanmean(test,axis=1) != df_train_clim.iloc[:,-2].values))
print(np.nanmean(np.abs(np.nanmean(test,axis=1) - df_train_clim.iloc[:,-2].values)))
# print(np.sum(np.abs(np.nanmean(test,axis=1)- df_train.iloc[:,-2].values) > 0.00000000000001))
# print(np.sum(np.abs(np.nanmean(test,axis=1)- df_train.iloc[:,-2].values) > 0.0000000000000000001))
print('predictors')
test = df_p_train_clim.iloc[:,:].values
print(np.sum(test != df_train_clim.iloc[:,1:-2].values))
print(np.nanmean(np.abs(test - df_train_clim.iloc[:,1:-2].values)))
# print(np.sum(np.abs(test- df_train.iloc[:,1:-2].values) > 0.00000000000001))
# print(np.sum(np.abs(test- df_train.iloc[:,1:-2].values) > 0.0000000000000000000000000001))
print('targets')
test = df_t_train_clim.iloc[:,:].values
print(np.sum(test != df_train_clim.iloc[:,0:1].values))
print(np.nanmean(np.abs(test - df_train_clim.iloc[:,0:1].values)))
# print(np.sum(np.abs(test- df_train.iloc[:,0:1].values) > 0.00000000000001))
# print(np.sum(np.abs(test- df_train.iloc[:,0:1].values) > 0.000000000000000000000001))
print('--------')
#%%
i=10
plt.figure()
plt.plot(time_train_plot,df_train_clim.iloc[:,-2])
plt.plot(time_train_plot,df_f_train_clim.iloc[:,i])
# plt.plot(time_train_plot,df_f_train_clim.iloc[:,1])
# plt.plot(time_train_plot,df_train_clim.iloc[:,-2],color='gray')
# plt.plot(time_train_plot,np.nanmean(df_f_train_clim.iloc[:,0:12],axis=1))
plt.figure()
plt.plot(time_train_plot,np.abs(df_train_clim.iloc[:,-2]-df_f_train_clim.iloc[:,i]))


#%%
print('VALID')
print('forecast')
test = df_f_valid_clim.iloc[:,0:12].values
print(np.sum(np.nanmean(test,axis=1) != df_valid_clim.iloc[:,-2].values))
print(np.nanmean(np.abs(np.nanmean(test,axis=1) - df_valid_clim.iloc[:,-2].values)))
# print(np.sum(np.abs(np.nanmean(test,axis=1)- df_valid.iloc[:,-2].values) > 0.00000000000001))
# print(np.sum(np.abs(np.nanmean(test,axis=1)- df_valid.iloc[:,-2].values) > 0.00000000000000001))
print('predictors')
test = df_p_valid_clim.iloc[:,:].values
print(np.sum(test != df_valid_clim.iloc[:,1:-2].values))
print(np.nanmean(np.abs(test - df_valid_clim.iloc[:,1:-2].values)))
# print(np.sum(np.abs(test- df_valid.iloc[:,1:-2].values) > 0.00000000000001))
# print(np.sum(np.abs(test- df_valid.iloc[:,1:-2].values) > 0.00000000000000001))
print('targets')
test = df_t_valid_clim.iloc[:,:].values
print(np.sum(test != df_valid_clim.iloc[:,0:1].values))
print(np.nanmean(np.abs(test - df_valid_clim.iloc[:,0:1].values)))
# print(np.sum(np.abs(test- df_valid.iloc[:,0:1].values) > 0.00000000000001))
# print(np.sum(np.abs(test- df_valid.iloc[:,0:1].values) > 0.00000000000000001))
print('--------')

#%%
print('TEST')
print('forecast')
test = df_f_test_clim.iloc[:,0:12].values
print(np.sum(np.nanmean(test,axis=1) != df_test_clim.iloc[:,-2].values))
print(np.nanmean(np.abs(np.nanmean(test,axis=1) - df_test_clim.iloc[:,-2].values)))
# print(np.sum(np.abs(np.nanmean(test,axis=1)- df_test.iloc[:,-2].values) > 0.00000000000001))
# print(np.sum(np.abs(np.nanmean(test,axis=1)- df_test.iloc[:,-2].values) > 0.00000000000000001))
print('predictors')
test = df_p_test_clim.iloc[:,:].values
print(np.sum(np.nanmean(test,axis=1) != df_test_clim.iloc[:,1:-2].values))
print(np.nanmean(np.abs(test - df_test_clim.iloc[:,1:-2].values)))
# print(np.sum(np.abs(test- df_test.iloc[:,1:-2].values) > 0.000000000000001))
# print(np.sum(np.abs(test- df_test.iloc[:,1:-2].values) > 0.00000000000000001))
print('targets')
test = df_t_test_clim.iloc[:,:].values
print(np.sum(test != df_test_clim.iloc[:,0:1].values))
print(np.nanmean(np.abs(test - df_test_clim.iloc[:,0:1].values)))
# print(np.sum(np.abs(test- df_test.iloc[:,0:1].values) > 0.000000000000001))
# print(np.sum(np.abs(test- df_test.iloc[:,0:1].values) > 0.000000000000000001))
print('--------')

#%%
# n_frcst_vars = 2
# # REPLACE NAN WITH CLIMATOLOGICAL VALUES:
# df_t_train_in, df_t_valid_in, df_t_test_in = replace_nan_with_clim(df_t_train_in,df_t_valid_in,df_t_test_in,
#                                                                     df_t_train_clim,df_t_valid_clim,df_t_test_clim,
#                                                                     verbose = True)

# df_p_train_in, df_p_valid_in, df_p_test_in = replace_nan_with_clim(df_p_train_in,df_p_valid_in,df_p_test_in,
#                                                                     df_p_train_clim,df_p_valid_clim,df_p_test_clim,
#                                                                     verbose = True)

# # if perfect_fcst:
# #     df_f_train_in, df_f_valid_in, df_f_test_in = replace_nan_with_clim(df_f_train_in,df_f_valid_in,df_f_test_in,
# #                                                                        df_f_train_clim,df_f_valid_clim,df_f_test_clim,
# #                                                                        verbose = True)

# df_train_in_tp, df_valid_in_tp, df_test_in_tp = replace_nan_with_clim(df_train_in.iloc[:,0:-n_frcst_vars],df_valid_in.iloc[:,0:-n_frcst_vars],df_test_in.iloc[:,0:-n_frcst_vars],
#                                                               df_train_clim.iloc[:,0:-n_frcst_vars],df_valid_clim.iloc[:,0:-n_frcst_vars],df_test_clim.iloc[:,0:-n_frcst_vars],
#                                                               verbose = True)
# df_train_in.iloc[:,0:-n_frcst_vars] = df_train_in_tp
# df_valid_in.iloc[:,0:-n_frcst_vars] = df_valid_in_tp
# df_test_in.iloc[:,0:-n_frcst_vars] = df_test_in_tp


# #%%

# print('TRAIN')
# print('forecast')
# test = df_f_train_in.iloc[:,0:12].values
# print(np.sum(np.nanmean(test,axis=1) != df_train_in.iloc[:,-2].values))
# print(np.nanmean(np.abs(np.nanmean(test,axis=1) - df_train_in.iloc[:,-2].values)))
# # print(np.sum(np.abs(np.nanmean(test,axis=1)- df_train.iloc[:,-2].values) > 0.00000000000001))
# # print(np.sum(np.abs(np.nanmean(test,axis=1)- df_train.iloc[:,-2].values) > 0.0000000000000000001))
# print('predictors')
# test = df_p_train_in.iloc[:,:].values
# print(np.sum(test != df_train_in.iloc[:,1:-2].values))
# print(np.nanmean(np.abs(test - df_train_in.iloc[:,1:-2].values)))
# # print(np.sum(np.abs(test- df_train.iloc[:,1:-2].values) > 0.00000000000001))
# # print(np.sum(np.abs(test- df_train.iloc[:,1:-2].values) > 0.0000000000000000000000000001))
# print('targets')
# test = df_t_train_in.iloc[:,:].values
# print(np.sum(test != df_train_in.iloc[:,0:1].values))
# print(np.nanmean(np.abs(test - df_train_in.iloc[:,0:1].values)))
# # print(np.sum(np.abs(test- df_train.iloc[:,0:1].values) > 0.00000000000001))
# # print(np.sum(np.abs(test- df_train.iloc[:,0:1].values) > 0.000000000000000000000001))
# print('--------')
# #%%
# print('VALID')
# print('forecast')
# test = df_f_valid_in.iloc[:,0:12].values
# print(np.sum(np.nanmean(test,axis=1) != df_valid_in.iloc[:,-2].values))
# print(np.nanmean(np.abs(np.nanmean(test,axis=1) - df_valid_in.iloc[:,-2].values)))
# # print(np.sum(np.abs(np.nanmean(test,axis=1)- df_valid.iloc[:,-2].values) > 0.00000000000001))
# # print(np.sum(np.abs(np.nanmean(test,axis=1)- df_valid.iloc[:,-2].values) > 0.00000000000000001))
# print('predictors')
# test = df_p_valid_in.iloc[:,:].values
# print(np.sum(test != df_valid_in.iloc[:,1:-2].values))
# print(np.nanmean(np.abs(test - df_valid_in.iloc[:,1:-2].values)))
# # print(np.sum(np.abs(test- df_valid.iloc[:,1:-2].values) > 0.00000000000001))
# # print(np.sum(np.abs(test- df_valid.iloc[:,1:-2].values) > 0.00000000000000001))
# print('targets')
# test = df_t_valid_in.iloc[:,:].values
# print(np.sum(test != df_valid_in.iloc[:,0:1].values))
# print(np.nanmean(np.abs(test - df_valid_in.iloc[:,0:1].values)))
# # print(np.sum(np.abs(test- df_valid.iloc[:,0:1].values) > 0.00000000000001))
# # print(np.sum(np.abs(test- df_valid.iloc[:,0:1].values) > 0.00000000000000001))
# print('--------')

# #%%
# print('TEST')
# print('forecast')
# test = df_f_test_in.iloc[:,0:12].values
# print(np.sum(np.nanmean(test,axis=1) != df_test_in.iloc[:,-2].values))
# print(np.nanmean(np.abs(np.nanmean(test,axis=1) - df_test_in.iloc[:,-2].values)))
# # print(np.sum(np.abs(np.nanmean(test,axis=1)- df_test.iloc[:,-2].values) > 0.00000000000001))
# # print(np.sum(np.abs(np.nanmean(test,axis=1)- df_test.iloc[:,-2].values) > 0.00000000000000001))
# print('predictors')
# test = df_p_test_in.iloc[:,:].values
# print(np.sum(np.nanmean(test,axis=1) != df_test_in.iloc[:,1:-2].values))
# print(np.nanmean(np.abs(test - df_test_in.iloc[:,1:-2].values)))
# # print(np.sum(np.abs(test- df_test.iloc[:,1:-2].values) > 0.000000000000001))
# # print(np.sum(np.abs(test- df_test.iloc[:,1:-2].values) > 0.00000000000000001))
# print('targets')
# test = df_t_test_in.iloc[:,:].values
# print(np.sum(test != df_test_in.iloc[:,0:1].values))
# print(np.nanmean(np.abs(test - df_test_in.iloc[:,0:1].values)))
# # print(np.sum(np.abs(test- df_test.iloc[:,0:1].values) > 0.000000000000001))
# # print(np.sum(np.abs(test- df_test.iloc[:,0:1].values) > 0.000000000000000001))
# print('--------')
# #%%

# # REMOVE CLIMATOLOGY TO GET ANOMALIES:
# if anomaly_target:
#     df_t_train_in.iloc[:,:] = df_t_train_in.iloc[:,:].values-df_t_train_clim.iloc[:,:].values
#     df_t_valid_in.iloc[:,:] = df_t_valid_in.iloc[:,:].values-df_t_valid_clim.iloc[:,:].values
#     df_t_test_in.iloc[:,:] = df_t_test_in.iloc[:,:].values-df_t_test_clim.iloc[:,:].values

#     df_train_in.iloc[:,0:1] = df_train_in.iloc[:,0:1].values-df_train_clim.iloc[:,0:1].values
#     df_valid_in.iloc[:,0:1] = df_valid_in.iloc[:,0:1].values-df_valid_clim.iloc[:,0:1].values
#     df_test_in.iloc[:,0:1] = df_test_in.iloc[:,0:1].values-df_test_clim.iloc[:,0:1].values

# if anomaly_past:
#     df_p_train_in.iloc[:,:] = df_p_train_in.iloc[:,:].values-df_p_train_clim.iloc[:,:].values
#     df_p_valid_in.iloc[:,:] = df_p_valid_in.iloc[:,:].values-df_p_valid_clim.iloc[:,:].values
#     df_p_test_in.iloc[:,:] = df_p_test_in.iloc[:,:].values-df_p_test_clim.iloc[:,:].values

#     if n_frcst_vars > 0:
#         df_train_in.iloc[:,1:-n_frcst_vars] = df_train_in.iloc[:,1:-n_frcst_vars].values-df_train_clim.iloc[:,1:-n_frcst_vars].values
#         df_valid_in.iloc[:,1:-n_frcst_vars] = df_valid_in.iloc[:,1:-n_frcst_vars].values-df_valid_clim.iloc[:,1:-n_frcst_vars].values
#         df_test_in.iloc[:,1:-n_frcst_vars] = df_test_in.iloc[:,1:-n_frcst_vars].values-df_test_clim.iloc[:,1:-n_frcst_vars].values
#     else:
#         df_train_in.iloc[:,1:] = df_train_in.iloc[:,1:].values-df_train_clim.iloc[:,1:].values
#         df_valid_in.iloc[:,1:] = df_valid_in.iloc[:,1:].values-df_valid_clim.iloc[:,1:].values
#         df_test_in.iloc[:,1:] = df_test_in.iloc[:,1:].values-df_test_clim.iloc[:,1:].values

# if anomaly_frcst:
#     df_f_train_in.iloc[:,:] = df_f_train_in.iloc[:,:].values-df_f_train_clim.iloc[:,:].values
#     df_f_valid_in.iloc[:,:] = df_f_valid_in.iloc[:,:].values-df_f_valid_clim.iloc[:,:].values
#     df_f_test_in.iloc[:,:] = df_f_test_in.iloc[:,:].values-df_f_test_clim.iloc[:,:].values

#     if n_frcst_vars > 0:
#         df_train_in.iloc[:,-n_frcst_vars:] = df_train_in.iloc[:,-n_frcst_vars:].values-df_train_clim.iloc[:,-n_frcst_vars:].values
#         df_valid_in.iloc[:,-n_frcst_vars:] = df_valid_in.iloc[:,-n_frcst_vars:].values-df_valid_clim.iloc[:,-n_frcst_vars:].values
#         df_test_in.iloc[:,-n_frcst_vars:] = df_test_in.iloc[:,-n_frcst_vars:].values-df_test_clim.iloc[:,-n_frcst_vars:].values



# #%%
# # DATA NORMALIZATION
# # Normalize all predictors, forecasts, and targets using only the training data
# if norm_type != 'None':
#     scaler_t = fit_scaler(df_t_train_in,norm_type=norm_type)
#     df_t_train_scaled = normalize_df(df_t_train_in,scaler_t)
#     df_t_valid_scaled = normalize_df(df_t_valid_in,scaler_t)
#     df_t_test_scaled = normalize_df(df_t_test_in,scaler_t)

#     scaler_p = fit_scaler(df_p_train_in,norm_type=norm_type)
#     df_p_train_scaled = normalize_df(df_p_train_in,scaler_p)
#     df_p_valid_scaled = normalize_df(df_p_valid_in,scaler_p)
#     df_p_test_scaled = normalize_df(df_p_test_in,scaler_p)

#     scaler_f = fit_scaler(df_f_train_in,norm_type=norm_type)
#     df_f_train_scaled = normalize_df(df_f_train_in,scaler_f)
#     df_f_valid_scaled = normalize_df(df_f_valid_in,scaler_f)
#     df_f_test_scaled = normalize_df(df_f_test_in,scaler_f)

#     FU_threshold_t = np.ones((len(df_t_train_in),1))*Tw_thresh
#     scaled_FU_threshold_t = scaler_t.transform(FU_threshold_t)[0,0]


#     scaler = fit_scaler(df_train_in,norm_type=norm_type)
#     df_train_scaled = normalize_df(df_train_in,scaler)
#     df_valid_scaled = normalize_df(df_valid_in,scaler)
#     df_test_scaled = normalize_df(df_test_in,scaler)

#     target_scaler = fit_scaler(df_train_in.iloc[:,0:1],norm_type=norm_type)
#     FU_threshold = np.ones((len(df_train_in),1))*Tw_thresh
#     scaled_FU_threshold = target_scaler.transform(FU_threshold)[0,0]
# else:
#     df_train_scaled = df_train_in
#     df_valid_scaled = df_valid_in
#     df_test_scaled = df_test_in

#     df_t_train_scaled = df_t_train_in
#     df_t_valid_scaled = df_t_valid_in
#     df_t_test_scaled = df_t_test_in

#     df_p_train_scaled = df_p_train_in
#     df_p_valid_scaled = df_p_valid_in
#     df_p_test_scaled = df_p_test_in

#     df_f_train_scaled = df_f_train_in
#     df_f_valid_scaled = df_f_valid_in
#     df_f_test_scaled = df_f_test_in

#     FU_threshold = np.ones((len(df_train_in),1))*Tw_thresh
#     scaled_FU_threshold = FU_threshold[0,0]

# #%%
# print('TRAIN - before windowed')
# print(np.sum(df_p_train_scaled.iloc[:,:].values != df_train_scaled.iloc[:,1:-n_frcst_vars].values))
# print(np.sum(df_t_train_scaled.iloc[:,:].values != df_train_scaled.iloc[:,0:1].values))
# print(np.sum(np.isnan(df_p_train_scaled.iloc[:,:].values)),np.sum(np.isnan(df_train_scaled.iloc[:,1:-n_frcst_vars].values)))
# print(np.sum(np.isnan(df_t_train_scaled.iloc[:,:].values)),np.sum(np.isnan(df_train_scaled.iloc[:,0:1].values)))
# print('VALID - before windowed')
# print(np.sum(df_p_valid_scaled.iloc[:,:].values != df_valid_scaled.iloc[:,1:-n_frcst_vars].values))
# print(np.sum(df_t_valid_scaled.iloc[:,:].values != df_valid_scaled.iloc[:,0:1].values))
# print(np.sum(np.isnan(df_p_valid_scaled.iloc[:,:].values)),np.sum(np.isnan(df_valid_scaled.iloc[:,1:-n_frcst_vars].values)))
# print(np.sum(np.isnan(df_t_valid_scaled.iloc[:,:].values)),np.sum(np.isnan(df_valid_scaled.iloc[:,0:1].values)))
# print('TEST - before windowed')
# print(np.sum(df_p_test_scaled.iloc[:,:].values != df_test_scaled.iloc[:,1:-n_frcst_vars].values))
# print(np.sum(df_t_test_scaled.iloc[:,:].values != df_test_scaled.iloc[:,0:1].values))
# print(np.sum(np.isnan(df_p_test_scaled.iloc[:,:].values)),np.sum(np.isnan(df_test_scaled.iloc[:,1:-n_frcst_vars].values)))
# print(np.sum(np.isnan(df_t_test_scaled.iloc[:,:].values)),np.sum(np.isnan(df_test_scaled.iloc[:,0:1].values)))

# #%%

# # GET WINDOWED DATA SETS
# # Now we get training, validation, and test as tf.data.Dataset objects
# # The 'create_dataset' function returns batched datasets ('batch_size')
# # using a rolling window shifted by 1-day.
# train_windowed, train_clim_windowed, time_train_windowed = create_dataset(df_train_scaled, df_train_clim, time_train,
#                                                                           n_frcst_vars,
#                                                                           input_len, pred_len,
#                                                                           batch_size,
#                                                                           shuffle = False)

# valid_windowed, valid_clim_windowed, time_valid_windowed  = create_dataset(df_valid_scaled, df_valid_clim, time_valid,
#                                                                             n_frcst_vars,
#                                                                             input_len, pred_len,
#                                                                             batch_size,
#                                                                             shuffle = False)

# test_windowed, test_clim_windowed, time_test_windowed  = create_dataset(df_test_scaled, df_test_clim, time_test,
#                                                                         n_frcst_vars,
#                                                                         input_len, pred_len,
#                                                                         batch_size=1,
#                                                                         shuffle = False)

# train_windowed_t, train_clim_windowed_t, time_train_windowed_t = create_dataset_test(df_t_train_scaled,df_p_train_scaled,df_f_train_scaled,
#                                                                                 df_t_train_clim,df_p_train_clim, df_f_train_clim,
#                                                                                 time_train,month_train,
#                                                                                 n_frcst_vars,input_len,pred_len,batch_size,
#                                                                                 shuffle = False
#                                                                                 )

# valid_windowed_t, valid_clim_windowed_t, time_valid_windowed_t  = create_dataset_test(df_t_valid_scaled,df_p_valid_scaled,df_f_valid_scaled,
#                                                                                 df_t_valid_clim,df_p_valid_clim, df_f_valid_clim,
#                                                                                 time_valid,month_valid,
#                                                                                 n_frcst_vars,input_len,pred_len,batch_size,
#                                                                                 shuffle = False
#                                                                                 )

# test_windowed_t, test_clim_windowed_t, time_test_windowed_t  = create_dataset_test(df_t_test_scaled,df_p_test_scaled,df_f_test_scaled,
#                                                                               df_t_test_clim,df_p_test_clim, df_f_test_clim,
#                                                                               time_test,month_test,
#                                                                               n_frcst_vars,input_len,pred_len,batch_size=1,
#                                                                               shuffle = False
#                                                                               )

# #%%

# print("============================================")

# print('TRAINING - TARGET')
# at_old = np.zeros((15,512,60,2))
# at_new = np.zeros((15,512,60,2))
# for i,x in enumerate(train_windowed):
#     # if i == 1:
#         # print(x[1].shape)
#         # print(x[1][320:323,0:4,0])
#     at_old[i,0:x[1].shape[0],0:x[1].shape[1], 0:x[1].shape[2]] = x[1]

# for i,x in enumerate(train_windowed_t):
#     # if i == 1:
#     #     print(x[1].shape)
#     #     print(x[1][320:323,0:4,0])
#     at_new[i,0:x[1].shape[0],0:x[1].shape[1], 0:x[1].shape[2]] = x[1]

# print(np.sum(at_old != at_new))
# # print(at_old[at_new != at_old])
# # print(at_new[at_new != at_old])
# print(np.where(at_new != at_old))
# print(np.sum(np.isnan(at_old)))
# print(np.sum(np.isnan(at_new)))

# print('VALID - TARGET')
# at_old = np.zeros((15,512,60,2))
# at_new = np.zeros((15,512,60,2))
# for i,x in enumerate(valid_windowed):
#     # if i == 1:
#         # print(x[1].shape)
#         # print(x[1][320:323,0:4,0])
#     at_old[i,0:x[1].shape[0],0:x[1].shape[1], 0:x[1].shape[2]] = x[1]

# for i,x in enumerate(valid_windowed_t):
#     # if i == 1:
#         # print(x[1].shape)
#         # print(x[1][320:323,0:4,0])
#     at_new[i,0:x[1].shape[0],0:x[1].shape[1], 0:x[1].shape[2]] = x[1]

# print(np.sum(at_old != at_new))
# print(np.where(at_new != at_old))
# print(np.sum(np.isnan(at_old)))
# print(np.sum(np.isnan(at_new)))


# # print('TEST - TARGETS')
# # at_old = np.zeros((2502,60,2))
# # at_new = np.zeros((2502,60,2))
# # for i,x in enumerate(test_windowed):
# #     at_old[i,0:x[1].shape[1],0:x[1].shape[2]] = x[1]

# # for i,x in enumerate(test_windowed_t):
# #     at_new[i,0:x[1].shape[1], 0:x[1].shape[2]] = x[1]

# # print(np.sum(at_old != at_new))
# # print(np.where(at_new != at_old))
# # print(np.sum(np.isnan(at_old)))
# # print(np.sum(np.isnan(at_new)))

# #%%
# print("============================================")

# print('TRAINING - FORECAST')
# at_old = np.zeros((15,512,60,2))
# at_new = np.zeros((15,512,60,2))
# for i,x in enumerate(train_windowed):
#     at_old[i,0:x[0][1].shape[0],0:x[0][1].shape[1], 0:x[0][1].shape[2]] = x[0][1]

# for i,x in enumerate(train_windowed_t):
#     at_new[i,0:x[0][1].shape[0],0:x[0][1].shape[1], 0:x[0][1].shape[2]] = x[0][1]

# print(np.sum(at_old != at_new))
# # print(at_old[at_new != at_old])
# # print(at_new[at_new != at_old])
# print(np.nanmean(np.abs(at_new - at_old)))
# print(np.where(at_new != at_old)[2])
# plt.figure()
# plt.hist(np.abs(at_new - at_old))
# print(np.sum(np.isnan(at_old)))
# print(np.sum(np.isnan(at_new)))

# #%%
# print('VALID - FORECAST')
# at_old = np.zeros((15,512,60,2))
# at_new = np.zeros((15,512,60,2))
# for i,x in enumerate(valid_windowed):
#     # if i == 1:
#     #     print(x[0][1].shape)
#     #     print(x[0][1][320:323,0:4,0])
#     at_old[i,0:x[0][1].shape[0],0:x[0][1].shape[1], 0:x[0][1].shape[2]] = x[0][1]

# for i,x in enumerate(valid_windowed_t):
#     # if i == 1:
#     #     print(x[0][1].shape)
#     #     print(x[0][1][320:323,0:4,0])
#     at_new[i,0:x[0][1].shape[0],0:x[0][1].shape[1], 0:x[0][1].shape[2]] = x[0][1]

# print(np.sum(at_old != at_new))
# # print(at_old[at_new != at_old])
# # print(at_new[at_new != at_old])
# print(np.where(at_new != at_old))
# print(np.sum(np.isnan(at_old)))
# print(np.sum(np.isnan(at_new)))


# print('TEST - FORECAST')
# at_old = np.zeros((2502,60,2))
# at_new = np.zeros((2502,60,2))
# for i,x in enumerate(test_windowed):
#     # print(i,x[0][0].shape,x[0][1].shape,x[0][2].shape, x[1].shape)
#     # if i == 1:
#     #     print(x[0][1].shape)
#     #     print(x[0][1][320:323,0:4,0])
#     at_old[i,0:x[0][1].shape[1], 0:x[0][1].shape[2]] = x[0][1][0]

# for i,x in enumerate(test_windowed_t):
#     # if i == 1:
#     #     print(x[0][1].shape)
#     #     print(x[0][1][320:323,0:4,0])
#     at_new[i,0:x[0][1].shape[1], 0:x[0][1].shape[2]] = x[0][1][0]

# print(np.sum(at_old != at_new))
# # print(at_old[at_new != at_old])
# # print(at_new[at_new != at_old])
# print(np.where(at_new != at_old))
# print(np.sum(np.isnan(at_old)))
# print(np.sum(np.isnan(at_new)))

# print("============================================")



#%%
# n_frcst_vars = int(df_f_train.shape[1]/12)
# train_windowed, train_clim_windowed, time_train_windowed = create_dataset(df_train, df_train_clim, time_train,
#                                                                           n_frcst_vars,
#                                                                           input_len, pred_len,
#                                                                           batch_size,
#                                                                           shuffle = False)


# train_windowed_test, train_clim_windowed_test, time_train_windowed_test = create_dataset_test(df_t_train, df_p_train, df_f_train,
                                                                                              # df_t_train_clim, df_p_train_clim, df_f_train_clim,
                                                                                              # time_train,
                                                                                              # n_frcst_vars,
                                                                                              # input_len, pred_len,
                                                                                              # batch_size,
                                                                                              # shuffle = False)


#%%
# month_train = np.array([(date_ref+dt.timedelta(days=int(time_train[it]))).month for it in range(len(time_train))])

# df_t_in = df_t_train
# df_p_in = df_p_train
# df_f_in = df_f_train
# df_t_clim_in = df_t_train_clim
# df_p_clim_in = df_p_train_clim
# df_f_clim_in = df_f_train_clim
# time_in = time_train
# month_in = month_train
# n_forecasts = n_frcst_vars
# window_size = input_len
# forecast_size = pred_len
# batch_size = batch_size
# shuffle = False



# for i, x in enumerate(train_windowed):
# # for i, x in enumerate(data_filtered):
#     # print(i, x[0][0].shape,x[0][1].shape,x[0][2].shape, x[1].shape)
#     print(i, x[1][0,0:40])


# for i,x in enumerate(all_ds_f_filtered):
#     print(i,x[3][-forecast_size].numpy(), x[0][-forecast_size:-forecast_size+3,0].numpy())
#     # plt.plot(time_train, x.numpy().T)

# for i,x in enumerate(all_ds_f_mfiltered):
#     # print(i, x[0].shape, x[1].shape, x[2].shape)
#     # print(i, x[3][-forecast_size] == month_train[128:][i])
#     # print(i, x[3][-forecast_size:-forecast_size+3])
#     print(i, x[0][-forecast_size:-forecast_size+3,0].numpy())


# # Select forecast lead time corresponding to forecast start date
# all_ds_f_tfiltered_jan =  all_ds_f_filtered.map(lambda d,dc,t: [d[:,0::12], dc[:,0::12], t])
# all_ds_f_tfiltered_feb =  all_ds_f_filtered.map(lambda d,dc,t: [d[:,0+1::12], dc[:,0+1::12], t])
# all_ds_f_tfiltered_mar =  all_ds_f_filtered.map(lambda d,dc,t: [d[:,0+2::12], dc[:,0+2::12], t])
# all_ds_f_tfiltered_apr =  all_ds_f_filtered.map(lambda d,dc,t: [d[:,0+3::12], dc[:,0+3::12], t])
# all_ds_f_tfiltered_may =  all_ds_f_filtered.map(lambda d,dc,t: [d[:,0+4::12], dc[:,0+4::12], t])
# all_ds_f_tfiltered_jun =  all_ds_f_filtered.map(lambda d,dc,t: [d[:,0+5::12], dc[:,0+5::12], t])
# all_ds_f_tfiltered_jul =  all_ds_f_filtered.map(lambda d,dc,t: [d[:,0+6::12], dc[:,0+6::12], t])
# all_ds_f_tfiltered_aug =  all_ds_f_filtered.map(lambda d,dc,t: [d[:,0+7::12], dc[:,0+7::12], t])
# all_ds_f_tfiltered_sep =  all_ds_f_filtered.map(lambda d,dc,t: [d[:,0+8::12], dc[:,0+8::12], t])
# all_ds_f_tfiltered_oct =  all_ds_f_filtered.map(lambda d,dc,t: [d[:,0+9::12], dc[:,0+9::12], t])
# all_ds_f_tfiltered_nov =  all_ds_f_filtered.map(lambda d,dc,t: [d[:,0+10::12], dc[:,0+10::12], t])
# all_ds_f_tfiltered_dec =  all_ds_f_filtered.map(lambda d,dc,t: [d[:,0+11::12], dc[:,0+11::12], t])

# #%%
# datasets = [all_ds_f_tfiltered_jan,
#             all_ds_f_tfiltered_feb,
#             all_ds_f_tfiltered_mar,
#             all_ds_f_tfiltered_apr,
#             all_ds_f_tfiltered_may,
#             all_ds_f_tfiltered_jun,
#             all_ds_f_tfiltered_jul,
#             all_ds_f_tfiltered_aug,
#             all_ds_f_tfiltered_sep,
#             all_ds_f_tfiltered_oct,
#             all_ds_f_tfiltered_nov,
#             all_ds_f_tfiltered_dec
#             ]

# # Define a dataset containing `[0, 1, 2, 0, 1, 2, 0, 1, 2]`.
# choice_dataset = tf.data.Dataset.range(12)
# result = tf.data.Dataset.choose_from_datasets(datasets, choice_dataset)

# #%%
# # for i,x in enumerate(result):
# for i,x in enumerate(all_ds_f_tfiltered_dec):
#     print('=====================')
#     print(i,(dt.timedelta(days=int(x[2][-forecast_size].numpy()))+date_ref))
#     print(x[0].shape)
#     print(x[0][-forecast_size:-forecast_size+5,0])


# #%%
# for i,x in enumerate(all_ds_f_filtered):
#     frcst_start_date = (dt.timedelta(days=int(x[2][-forecast_size].numpy()))+date_ref)
#     sample_start_date = (dt.timedelta(days=int(x[2][0].numpy()))+date_ref)
#     print(i, frcst_start_date)
#     print(x[0].shape,x[0][:,frcst_start_date.month-1].shape)
#     # print(np.sum(np.isnan(x[0][-forecast_size:,0])),np.sum(np.isnan(x[0][-forecast_size:,1])),np.sum(np.isnan(x[0][-forecast_size:,2])))
#     # print(np.sum(np.isnan(x[0][-forecast_size:,3])),np.sum(np.isnan(x[0][-forecast_size:,4])),np.sum(np.isnan(x[0][-forecast_size:,5])))
#     # print(np.sum(np.isnan(x[0][-forecast_size:,6])),np.sum(np.isnan(x[0][-forecast_size:,7])),np.sum(np.isnan(x[0][-forecast_size:,8])))
#     # print(np.sum(np.isnan(x[0][-forecast_size:,9])),np.sum(np.isnan(x[0][-forecast_size:,10])),np.sum(np.isnan(x[0][-forecast_size:,11])))
#     # print(i,x[0][-forecast_size,0].numpy(),all_ds_f_tfiltered_feb[i][0],x[2][-forecast_size].numpy(),(date_ref+dt.timedelta(days = int(x[2][-forecast_size]))))


#%%










































#%% BELOW IS FOR WORKING ON TRANSFERRING FUD DETECTION TO TF.TENSORS

# # y_pred = np.expand_dims(y_pred_test[200:200+512,:],axis=-1)
# # y_true = np.expand_dims(y_test[200:200+512,:],axis=-1)
# y_pred = np.expand_dims(y_pred_train_all[1200:1200+512,:],axis=-1)
# y_true = np.expand_dims(y_train_all[1200:1200+512,:],axis=-1)

# def_opt = 1
# Twater_in = y_true
# time = np.squeeze(target_time_test_all[200+365,:])
# thresh_T=0.75
# ndays=1
# date_ref=dt.date(1900,1,1)

# mask_threshold = tf.math.less_equal(y_true, thresh_T)
# mask_threshold_pred = tf.math.less_equal(y_pred, thresh_T)

# n_same = tf.equal(mask_threshold,mask_threshold_pred)

# n_thresh = tf.reduce_sum(tf.cast(mask_threshold,'int32'),axis=1)
# n_thresh_pred = tf.reduce_sum(tf.cast(mask_threshold_pred,'int32'),axis=1)
#%%
# tf.where(tf.equal(mask_threshold[0],mask_threshold_pred[0]),
#           y_true[0],tf.constant(100,shape=tf.shape(y_true[0]))
#           )

# tf.equal(mask_threshold[0],mask_threshold_pred[0])

# metrics.confusion_matrix(y_true[0],y_pred[0])
# thresh_T=0.75
# mask_threshold = tf.cast(tf.math.less_equal(y_true, thresh_T), 'float')
# mask_threshold_pred = tf.cast(tf.math.less_equal(y_pred, thresh_T), 'float')



# tp = tf.reduce_sum(tf.cast(tf.math.multiply(mask_threshold,mask_threshold_pred), 'float'), axis=1)
# tn = tf.reduce_sum(tf.cast(tf.math.multiply(1-mask_threshold,1-mask_threshold_pred), 'float'), axis=1)
# fp = tf.reduce_sum(tf.cast(tf.math.multiply(1-mask_threshold,mask_threshold_pred), 'float'), axis=1)
# fn = tf.reduce_sum(tf.cast(tf.math.multiply(mask_threshold,1-mask_threshold_pred), 'float'), axis=1)

# precision = tp / (tp + fp + K.epsilon())
# recall = tp / (tp + fn + K.epsilon())

# f1 = 2*precision*recall / (precision+recall+K.epsilon())
# f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
#%%
# [batchsize,pred_len,1]
# y_pred = np.expand_dims(y_pred_train_all[0,1200:1200+512,:],axis=-1)
# y_true = np.expand_dims(y_train_all[0,1200:1200+512,:],axis=-1)

# mse = tf.keras.losses.MeanSquaredError()
# dy_true = y_true[:,1:,:]-y_true[:,:-1,:]
# dy_pred = y_pred[:,1:,:]-y_pred[:,:-1,:]

# # # tmp = tf.concat(dy_true,tf.constant(1.0,shape=[tf.shape(y_true)[0],1,1]))

# # l = 0.8*mse(y_true,y_pred)+0.2*mse(dy_true,dy_pred)

# # wtmp = (tf.cast(tf.math.less_equal(y_true, 0.75),'float'))

# w = (tf.cast(tf.reduce_any(tf.math.less_equal(y_true, 3.0),axis=2),'float'))




#%%
# def record_event_tf(istart,time,year,Twater_in,freezeup_date,mask_freezeup,date_ref):
#     # Temperature has been lower than thresh_T
#     # for more than (or equal to) ndays.
#     # Define freezeup date as first date of group

#     date_start = date_ref+dt.timedelta(days=int(time[istart]))
#     doy_start = (date_start - dt.date(int(date_start.year),1,1)).days+1

#     if ((date_start.year > 1992) | ((date_start.year == 1992) & (date_start.month > 10)) ):
#         if ( (date_start.year == year) & (doy_start > 319) ) | ((date_start.year == year+1) & (doy_start < 46)):
#                 freezeup_date[0] = date_start.year
#                 freezeup_date[1] = date_start.month
#                 freezeup_date[2] = date_start.day
#                 freezeup_Tw = Twater_in[istart]
#                 mask_freezeup[istart] = True
#         else:
#             freezeup_date[0] = np.nan
#             freezeup_date[1] = np.nan
#             freezeup_date[2] = np.nan
#             freezeup_Tw = np.nan
#             mask_freezeup[istart] = False
#     else:
#     # I think this condition exists because the Tw time series
#     # starts in January 1992, so it is already frozen, but we
#     # do not want to detect this as freezeup for 1992, so we
#     # have to wait until at least October 1992 before recording
#     # any FUD events.
#         freezeup_date[0] = np.nan
#         freezeup_date[1] = np.nan
#         freezeup_date[2] = np.nan
#         freezeup_Tw = np.nan
#         mask_freezeup[istart] = False

#     return freezeup_date, freezeup_Tw, mask_freezeup


# date_start = dt.timedelta(days=int(time[0])) + date_ref
# if date_start.month < 3:
#     year = date_start.year-1
# else:
#     year = date_start.year


# mask_threshold = tf.math.less_equal(Twater_in, thresh_T)
# mask_freezeup = tf.constant(False, dtype=bool,shape=tf.shape(mask_threshold))

# # tf.where(mask_threshold)

# #%%

# freezeup_date=np.zeros((3))*np.nan
# isample = 0
# # Loop on sample time steps
# for im in range(mask_freezeup.shape[1]):

#     if (im == 0): # First time step cannot be detected as freeze-up
#         sum_m = 0
#         istart = -1 # This ensures that a freeze-up is not detected if the time series started already below the freezing temp.
#     else:
#         if (tf.equal(tf.reduce_sum(tf.cast(mask_freezeup,'int32'), axis=1)[isample],0)):
#         # if (np.sum(mask_freezeup) == 0): # Only continue while no prior freeze-up was detected for the sequence
#             if (~mask_threshold[im-1]):
#                 sum_m = 0
#                 if ~mask_threshold[im]:
#                     sum_m = 0
#                 else:
#                     # start new group
#                     sum_m +=1
#                     istart = im
#                     # Below will only occur if ndays is set to 1, e.g. first day of freezing temp.
#                     if (sum_m >= ndays):
#                         freezeup_date,freezeup_Tw,mask_freezeup = record_event_tf(istart,time,year,Twater_in,freezeup_date,mask_freezeup,date_ref)
#             else:
#                 if (mask_threshold[im]) & (istart > 0):
#                     sum_m += 1
#                     if (sum_m >= ndays):
#                         freezeup_date,freezeup_Tw,mask_freezeup = record_event_tf(istart,time,year,Twater_in,freezeup_date,mask_freezeup,date_ref)










#%%
# # BELOW IS JUST A SPACE TO TEST CUSTOM LOSS IMPLEMENTATION
# mse = tf.keras.losses.MeanSquaredError()
# # mse2 = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

# # [batchsize,pred_len,1]
# y_pred = np.expand_dims(y_pred_test[0:512,:],axis=-1)
# y_true = np.expand_dims(y_test[0:512,:],axis=-1)

# # penalize_FN = True
# penalize_FN = False

# # use_exp_decay_loss = True
# use_exp_decay_loss = False

# if penalize_FN:
#     # Check for each sample in targets and predictions if Tw reaches below the freeze-up threshold
#     obs_FU_batch = tf.cast(tf.reduce_any(tf.math.less(y_true,0.75),axis=1),y_true.dtype)
#     pred_FU_batch = tf.cast(tf.reduce_any(tf.math.less(y_pred,0.75),axis=1),y_true.dtype)

#     # One weight per sample, depending if sample is a false positive,
#     # false negative, true positive or true negative for detected freeze-up
#     w_batch = tf.constant(1,y_true.dtype,shape=obs_FU_batch.shape) + 2*tf.math.squared_difference(obs_FU_batch,pred_FU_batch) + tf.math.subtract(obs_FU_batch,pred_FU_batch)
#     w_batch = tf.squeeze(w_batch)

#     if use_exp_decay_loss:
#         # Forcing an exponential weight decay per time step in
#         # forecast horizon, i.e. put more weight on the error
#         # of the first forecast steps.
#         obs_FU_all_steps = tf.cast(tf.reduce_any(tf.math.less(y_true,0.75),axis=-1),y_true.dtype)
#         pred_FU_all_steps = tf.cast(tf.reduce_any(tf.math.less(y_pred,0.75),axis=-1),y_true.dtype)

#         wtmp = tf.cast(tf.fill(tf.shape(obs_FU_all_steps),1.0),y_true.dtype)
#         exp_tmp = 1/tf.exp(tf.range(1,pred_len+1,dtype=y_true.dtype))
#         exp_w = tf.multiply(wtmp,exp_tmp)

#         w = tf.multiply(exp_w,tf.expand_dims(w_batch,axis=-1))
#     else:
#         w = w_batch

# else:
#     if use_exp_decay_loss:
#         # Forcing an exponential weight decay per time step in
#         # forecast horizon, i.e. put more weight on the error
#         # of the first forecast steps.
#         obs_FU_all_steps = tf.cast(tf.reduce_any(tf.math.less(y_true,0.75),axis=-1),y_true.dtype)
#         pred_FU_all_steps = tf.cast(tf.reduce_any(tf.math.less(y_pred,0.75),axis=-1),y_true.dtype)

#         wtmp = tf.cast(tf.fill(tf.shape(tf.squeeze(y_true)),1.0),y_true.dtype)
#         exp_tmp = 1/tf.exp(tf.range(1,pred_len+1,dtype=y_true.dtype))
#         exp_w = tf.multiply(wtmp,exp_tmp)

#         w = exp_w
#     else:
#         w = 1

# out = mse(y_true,y_pred,sample_weight=w)


# exp_tmp = 1/tf.exp(tf.range(1,pred_len+1,dtype=y_true.dtype))
# tmp = tf.reduce_sum(tf.math.log(tf.reduce_mean(tf.math.square(y_true-y_pred),axis=0)),axis=0)

# tmp = tf.math.log(tf.reduce_mean(tf.math.square(y_true-y_pred),axis=0))
