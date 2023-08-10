#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:27:49 2022

@author: amelie
"""
#%%
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from keras.layers import Lambda

import os
import numpy as np
import pandas as pd
import datetime as dt

from matplotlib import pyplot as plt

from functions import rolling_climo
from functions_ML import regression_metrics, plot_prediction_timeseries

#%%

def exp_scheduler(epoch,lr):
    # This function decreases the learning rate exponentially.
    decay=0.05
    lr_min=None
    # lr_min=1e-5

    if lr_min is None:
        return lr * tf.math.exp(-decay)
    else:
        return np.max((lr * tf.math.exp(-decay),lr_min))




def lin_scheduler(epoch,lr):
    # This function decreases the learning rate linearly.
    decay=0.00016
    return lr - decay



def loss_wraper(loss_name,use_exp_decay_loss,tau,weights_on_timesteps,added_weight,scaled_FU_threshold,pred_len):
    # !!!!! NOTE:
    # 'penalize_FN', 'penalize_recall', AND
    # THE 'weights' OPTIONS WILL ONLY WORK
    # IF anomaly_target = False ...
    # NEED TO ADD THIS AS A CONDITION FOR THE CUSTOM LOSS USE.

    def custom_loss(y_true,y_pred):
        mse = tf.keras.losses.MeanSquaredError()
        # bce = tf.keras.losses.BinaryCrossentropy()

        if loss_name == 'MSETw':
            if use_exp_decay_loss:
                # Forcing an exponential weight decay per time step in
                # forecast horizon, i.e. put more weight on the error
                # of the first forecast steps.
                wtmp = tf.cast(tf.fill(tf.shape(tf.squeeze(y_true)),1.0),y_true.dtype)
                exp_tmp = 1/tf.exp(tf.range(1,pred_len+1,dtype=y_true.dtype)*(1./tau))
                exp_w = tf.multiply(wtmp,exp_tmp)

                w = exp_w
            elif weights_on_timesteps:
                w = (1.+added_weight*tf.cast(tf.reduce_any(tf.math.less_equal(y_true, scaled_FU_threshold),axis=2),'float'))
            else:
                w = 1

            return mse(y_true,y_pred,sample_weight=w)


        if loss_name == 'MSETw_MSEdTwdt':
            dy_true = y_true[:,1:,:]-y_true[:,:-1,:]
            dy_pred = y_pred[:,1:,:]-y_pred[:,:-1,:]

            if use_exp_decay_loss:
                # Forcing an exponential weight decay per time step in
                # forecast horizon, i.e. put more weight on the error
                # of the first forecast steps.
                wtmp = tf.cast(tf.fill(tf.shape(tf.squeeze(y_true)),1.0),y_true.dtype)
                exp_tmp = 1/tf.exp(tf.range(1,pred_len+1,dtype=y_true.dtype)*(1./tau))
                exp_w = tf.multiply(wtmp,exp_tmp)

                w = exp_w
                wd = w[:,:-1]
            elif weights_on_timesteps:
                w = (1.+added_weight*tf.cast(tf.reduce_any(tf.math.less_equal(y_true, scaled_FU_threshold),axis=2),'float'))
                wd = w[:,:-1]
            else:
                w = 1
                wd = 1

            return mse(y_true,y_pred,w)+mse(dy_true,dy_pred,wd)


        if loss_name == 'log_loss':
            return tf.reduce_sum(tf.math.log(tf.reduce_mean(tf.math.square(y_true-y_pred),axis=0)),axis=0)


        if loss_name == 'MSETw_penalize_FN':
            # We compute the weight for how the sample loss will
            # contribute to the overall batch loss, to penalize cases of
            # False Negative (FN) freeze-up events (weight times 4),
            # False Positive (FP) freeze-up events (weight times 2),
            # while True Positive (TP) and True Negative (TN) are
            # weighted as usual (weight = 1)

            # Check if Tw reaches below the freeze-up threshold in each sample.
            # This is 1 if there is a freeze-up in the target or prediction,
            #      or 0 if there is no freeze-up.
            # (The freeze-up is defined as Tw < scaled_FU_threshold)
            obs_FU_batch = tf.cast(tf.reduce_any(tf.math.less(y_true,scaled_FU_threshold),axis=1),y_true.dtype)
            pred_FU_batch = tf.cast(tf.reduce_any(tf.math.less(y_pred,scaled_FU_threshold),axis=1),y_true.dtype)

            # Compute the weight to assign to that sample.
            #     # w = 1 for true positive and true negative (i.e. freeze-up or no freeze-up is detected in both the prediction and observations)
            #     # w = 2 for false positive (i.e. freeze-up is detected in the prediction, but not in observations)
            #     # w = 4 for false negative (i.e. no freeze-up is detected in the prediction, but there was a freeze-up in the observations)
            w_batch = tf.fill(tf.shape(obs_FU_batch),1.0) + 2*tf.math.squared_difference(obs_FU_batch,pred_FU_batch) + tf.math.subtract(obs_FU_batch,pred_FU_batch)
            w_batch = tf.squeeze(w_batch)

            if use_exp_decay_loss:
                # Forcing an exponential weight decay per time step in
                # forecast horizon, i.e. put more weight on the error
                # of the first forecast steps.
                wtmp = tf.cast(tf.fill(tf.shape(tf.squeeze(y_true)),1.0),y_true.dtype)
                exp_tmp = 1/tf.exp(tf.range(1,pred_len+1,dtype=y_true.dtype)*(1./tau))
                exp_w = tf.multiply(wtmp,exp_tmp)

                w = tf.multiply(exp_w,tf.expand_dims(w_batch,axis=-1))
            else:
                w = w_batch

            return mse(y_true,y_pred,sample_weight=w)


        if loss_name == 'MSETw_penalize_recall':
            mask_threshold = tf.cast(tf.math.less_equal(y_true, scaled_FU_threshold), 'float')
            mask_threshold_pred = tf.cast(tf.math.less_equal(y_pred, scaled_FU_threshold), 'float')

            tp = tf.reduce_sum(tf.cast(tf.math.multiply(mask_threshold,mask_threshold_pred), 'float'), axis=1)
            tn = tf.reduce_sum(tf.cast(tf.math.multiply(1-mask_threshold,1-mask_threshold_pred), 'float'), axis=1)
            fp = tf.reduce_sum(tf.cast(tf.math.multiply(1-mask_threshold,mask_threshold_pred), 'float'), axis=1)
            fn = tf.reduce_sum(tf.cast(tf.math.multiply(mask_threshold,1-mask_threshold_pred), 'float'), axis=1)

            precision = tp / (tp + fp + K.epsilon())
            recall = tp / (tp + fn + K.epsilon())
            FNR = fn / (tp + fn + K.epsilon())

            # return mse(y_true,y_pred)+ mse(dy_true,dy_pred)+tf.reduce_sum(FNR)
            return mse(y_true,y_pred,1+fn) + mse(dy_true,dy_pred,1+fn)



    return custom_loss




def plot_lr(lr0, nepochs, scheduler_type, decay=None, lr_min=None, ax_in=None):
    lr =[]
    for it in range(nepochs):
        if it == 0:
            lr.append(lr0)
        else:
            if scheduler_type == 'exponential':
                lr.append(exp_scheduler(it, lr[it-1]))
            if scheduler_type == 'linear':
                lr.append(lin_scheduler(it, lr[it-1]))

    if ax_in is None:
        plt.figure()
        plt.plot(lr)
    else:
        ax_in.plot(lr)




def save_SEAS5_predictor_arrays(local_path,time_rep='daily',region='D'):
    filename = 'SEAS5_SEAS51_forecast_'+time_rep+'_predictors'

    def get_all_SEAS5_predictors(var,month_start_arr,lead,time_rep,yrs,region,r_dir,base,local_path):

        from analysis.SEAS5.SEAS5_forecast_class import SEAS5frcst
        from functions import K_to_C

        # Array format is [frcst_start_month,years,lead_timemonths,all_ensemble_members]
        if (lead == 'all'):
            if (time_rep == 'monthly'):
                vout = np.zeros((12,len(yrs),8,51))*np.nan
            if (time_rep == 'daily'):
                vout = np.zeros((12,len(yrs),215,51))*np.nan
        else:
            vout = np.zeros((12,len(yrs),len(lead),51))*np.nan


        for month_lead0 in month_start_arr:
            extension = "{}{}.nc".format(2008, str(month_lead0).rjust(2, '0'))
            path = r_dir +"region"+ region + "/{}-{}/".format(2008, str(month_lead0).rjust(2, '0'))

            # Get data:
            s = SEAS5frcst(path + base + '_' + var + '_' + extension)
            vall = s.get_all_years(spatial_avg=True,
                                   lead = lead,
                                   time_rep=time_rep
                                   )

            if 'temperature' in var:
                vall = K_to_C(vall)

            if (time_rep == 'monthly'):
                for l in range(vout.shape[2]):
                    vout[month_lead0-1+l-12*((month_lead0-1+l)//12),:,l,:] = vall[:,l,:]
            if (time_rep == 'daily'):
                vout[month_lead0-1,:,:,:] = vall

        return vout

    SEAS5_years = np.arange(1981,2022+1)
    r_dir5 = local_path + 'slice/data/raw/SEAS5/'
    r_dir51 = local_path + 'slice/data/raw/SEAS51/'
    base5 = 'SEAS5'
    base51 = 'SEAS51'
    month_start_arr = [1,2,3,4,5,6,7,8,9,10,11,12]
    lead = 'all'

    print('Computing SEAS5 monthly predictors...')
    SEAS5_Ta_mean = get_all_SEAS5_predictors('2m_temperature',month_start_arr,lead,time_rep,SEAS5_years,region,r_dir5,base5,local_path)
    SEAS5_snowfall = get_all_SEAS5_predictors('snowfall_processed',month_start_arr,lead,time_rep,SEAS5_years,region,r_dir5,base5,local_path)

    print('Computing SEAS51 monthly predictors...')
    month_start_arr = [11,12]
    SEAS51_Ta_mean = get_all_SEAS5_predictors('2m_temperature',month_start_arr,lead,time_rep,SEAS5_years,region,r_dir51,base51,local_path)
    SEAS51_snowfall = get_all_SEAS5_predictors('snowfall_processed',month_start_arr,lead,time_rep,SEAS5_years,region,r_dir51,base51,local_path)

    # Save variables:
    print('Saving SEAS5 monthly predictors...')
    np.savez(local_path+'/slice/data/processed/SEAS5/'+filename,
            SEAS5_Ta_mean = SEAS5_Ta_mean,
            SEAS51_Ta_mean = SEAS51_Ta_mean,
            SEAS5_snowfall = SEAS5_snowfall,
            SEAS51_snowfall = SEAS51_snowfall
            )

    return SEAS5_Ta_mean,SEAS51_Ta_mean,SEAS5_snowfall,SEAS51_snowfall




def SEAS5_dailyarray_to_ts(a,years_a,month_lead0,time_in,date_ref = dt.date(1900,1,1)):
    ts_out = np.zeros(len(time_in))*np.nan
    for iyr,yr in enumerate(years_a):

        date_0 = dt.date(yr,month_lead0,1)
        it0 = np.where(time_in == (date_0-date_ref).days)[0][0]

        if (len(a[month_lead0-1,iyr,:]) == 215) & (len(ts_out[it0:it0+215]) == 215):
            ts_out[it0:it0+215] = a[month_lead0-1,iyr,:]

    return ts_out




def obs_dailyts_to_forecast_ts(ts_in,month_lead0,time_in,date_ref = dt.date(1900,1,1)):
    ts_out = np.zeros(len(time_in))*np.nan
    for it,t in enumerate(time_in):
        if ( (date_ref+dt.timedelta(days=int(time_in[it]))).month == month_lead0 ) & ((date_ref+dt.timedelta(days=int(time_in[it]))).day == 1 ):
            ts_out[it:it+215] = ts_in[it:it+215]

    return ts_out




def fit_scaler(df_train_in,norm_type='MinMax'):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler

    if norm_type == 'None':
        return None

    else:
        if norm_type =='MinMax': scaler = MinMaxScaler()
        if norm_type =='Standard': scaler = StandardScaler()

        return scaler.fit(df_train_in)




def normalize_df(df_in,scaler):
    df_out = df_in.copy()
    scaled_values = scaler.transform(df_out.values)

    for ip, p in enumerate(df_in.columns):
        df_out.iloc[:,ip] = scaled_values[:,ip]

    return df_out




def get_predictor_clim_ensemble_fcst(df_tr,df_va,df_te,
                                       ind_tr,ind_va,ind_te,
                                       time_tr,tr_years,time_in,
                                       df_is_frcst = False,
                                       perfect_fcst = False,
                                       nb_members = 25,
                                       nw=1,
                                       verbose = True):

    df_tr_clim = np.zeros((df_tr.shape))*np.nan
    df_va_clim = np.zeros((df_va.shape))*np.nan
    df_te_clim = np.zeros((df_te.shape))*np.nan

    if df_is_frcst:

        if perfect_fcst:
            n_forecast_vars = int(df_tr.shape[1]/12)
            for v in range(n_forecast_vars):
                for month in range(12):
                    if verbose: print(df_tr.iloc[:,(12*v)+month].name+' - month '+str(month+1))
                    f_ts_clim_m, _, _ = rolling_climo(nw,df_tr.iloc[:,(12*v)+month].values,'other',time_tr,tr_years,time_other=time_in)
                    df_tr_clim[:,(12*v)+month] = f_ts_clim_m[ind_tr]
                    df_va_clim[:,(12*v)+month] = f_ts_clim_m[ind_va]
                    df_te_clim[:,(12*v)+month] = f_ts_clim_m[ind_te]
        else:
            em_tr_tmp = np.zeros((df_tr.shape[0],int(df_tr.shape[1]/nb_members)))*np.nan
            n_forecast_vars = int(df_tr.shape[1]/(12*nb_members))
            for v in range(n_forecast_vars):
                vtmp =  df_tr.iloc[:,v*12*nb_members:(v+1)*12*nb_members]
                for month in range(12):
                    # First, get ensemble mean during training to compute climatology:
                    em_tr_tmp[:,(12*v)+month] = np.nanmean(vtmp.iloc[:,month::12],axis=1)
                    if verbose: print(df_tr.iloc[:,(12*v*nb_members)+month].name+' - month '+str(month+1))
                    f_ts_clim_m, _, _ = rolling_climo(nw,em_tr_tmp[:,(12*v)+month],'other',time_tr,tr_years,time_other=time_in)
                    for im in range(nb_members):
                        df_tr_clim[:,12*(im+v*nb_members)+month] = f_ts_clim_m[ind_tr]
                        df_va_clim[:,12*(im+v*nb_members)+month] = f_ts_clim_m[ind_va]
                        df_te_clim[:,12*(im+v*nb_members)+month] = f_ts_clim_m[ind_te]

    else:
        for ip, p in enumerate(df_tr.columns):
            if verbose: print(p, df_tr.iloc[:,ip].name)
            p_clim_mean, p_clim_std, _ = rolling_climo(nw, df_tr.iloc[:,ip].values,'other',time_tr,tr_years,time_other=time_in)
            df_tr_clim[:,ip] = p_clim_mean[ind_tr]
            df_va_clim[:,ip] = p_clim_mean[ind_va]
            df_te_clim[:,ip] = p_clim_mean[ind_te]

    df_tr_clim = pd.DataFrame(df_tr_clim,columns= df_tr.columns,index=df_tr.index)
    df_va_clim = pd.DataFrame(df_va_clim,columns= df_va.columns,index=df_va.index)
    df_te_clim = pd.DataFrame(df_te_clim,columns= df_te.columns,index=df_te.index)

    return df_tr_clim, df_va_clim, df_te_clim




def replace_nan_with_clim(df_tr,df_va,df_te,
                          df_tr_clim,df_va_clim,df_te_clim,
                          verbose = True):

    for ip, p in enumerate(df_tr.columns):
        df_tr.iloc[np.where(np.isnan(df_tr.iloc[:,ip]))[0],ip] = df_tr_clim.iloc[np.where(np.isnan(df_tr.iloc[:,ip]))[0],ip]
        df_va.iloc[np.where(np.isnan(df_va.iloc[:,ip]))[0],ip] = df_va_clim.iloc[np.where(np.isnan(df_va.iloc[:,ip]))[0],ip]
        df_te.iloc[np.where(np.isnan(df_te.iloc[:,ip]))[0],ip] = df_te_clim.iloc[np.where(np.isnan(df_te.iloc[:,ip]))[0],ip]
        # If there are still NaNs in the climatology because of missing data, we will just put a zero.
        # !!! THIS IS A QUICK FIX AND COULD BE IMPROVED !!! But it only happens for certain climate indices.
        # !!! CHECK THIS FOR MASKING THE NANS DURING MODEL TRAINING/TESTING: https://keras.io/api/layers/core_layers/masking/
        df_tr.iloc[np.where(np.isnan(df_tr.iloc[:,ip]))[0],ip] = 0
        df_va.iloc[np.where(np.isnan(df_va.iloc[:,ip]))[0],ip] = 0
        df_te.iloc[np.where(np.isnan(df_te.iloc[:,ip]))[0],ip] = 0

    if verbose:
        # Check if there are remaining nan values.
        print('Nans in train set?' , np.any(np.sum(np.isnan(df_tr[df_tr.columns[:]])) > 0 ))
        print('Nans in valid set?' ,np.any(np.sum(np.isnan(df_va[df_va.columns[:]])) > 0 ))
        print('Nans in test set?' ,np.any(np.sum(np.isnan(df_te[df_te.columns[:]])) > 0 ))

    return df_tr,df_va,df_te




def create_dataset_tpf(df_t_in,df_p_in,df_f_in,
                        df_t_clim_in,df_p_clim_in, df_f_clim_in,
                        time_in,month_in,
                        n_forecasts,window_size,forecast_size,batch_size,
                        shuffle = False,
                        shuffle_buffer_size = 0):

    # Total size of window is given by the number of steps to be considered
    # before prediction time + steps that we want to forecast
    total_size = window_size + forecast_size

    # # Buffer size for shuffle:
    # shuffle_buffer_size = np.min([9500*5,len(df_t_in)]) # This number should be equal to or larger than the number of samples.

    # Selecting windows
    data_p = tf.data.Dataset.from_tensor_slices(df_p_in.values)
    data_p = data_p.window(total_size, shift=1, drop_remainder=True)
    data_p = data_p.flat_map(lambda k: k.batch(total_size))

    data_t = tf.data.Dataset.from_tensor_slices(df_t_in.values)
    data_t = data_t.window(total_size, shift=1, drop_remainder=True)
    data_t = data_t.flat_map(lambda k: k.batch(total_size))

    data_f = tf.data.Dataset.from_tensor_slices(df_f_in.values)
    data_f = data_f.window(total_size, shift=1, drop_remainder=True)
    data_f = data_f.flat_map(lambda k: k.batch(total_size))

    data_p_clim = tf.data.Dataset.from_tensor_slices(df_p_clim_in.values)
    data_p_clim = data_p_clim.window(total_size, shift=1, drop_remainder=True)
    data_p_clim = data_p_clim.flat_map(lambda k: k.batch(total_size))

    data_t_clim = tf.data.Dataset.from_tensor_slices(df_t_clim_in.values)
    data_t_clim = data_t_clim.window(total_size, shift=1, drop_remainder=True)
    data_t_clim = data_t_clim.flat_map(lambda k: k.batch(total_size))

    data_f_clim = tf.data.Dataset.from_tensor_slices(df_f_clim_in.values)
    data_f_clim = data_f_clim.window(total_size, shift=1, drop_remainder=True)
    data_f_clim = data_f_clim.flat_map(lambda k: k.batch(total_size))

    time = tf.data.Dataset.from_tensor_slices(time_in)
    time = time.window(total_size, shift=1, drop_remainder=True)
    time = time.flat_map(lambda k: k.batch(total_size))

    month = tf.data.Dataset.from_tensor_slices(month_in)
    month = month.window(total_size, shift=1, drop_remainder=True)
    month = month.flat_map(lambda k: k.batch(total_size))

    # Zip all datasets together so that we can filter out the samples
    # that are discontinuous in time due to cross-validation splits.
    all_ds_p = tf.data.Dataset.zip((data_p, data_p_clim, time, month))
    all_ds_t = tf.data.Dataset.zip((data_t, data_t_clim, time, month))
    all_ds_f = tf.data.Dataset.zip((data_f, data_f_clim, time, month))

    # Filter discontinuous samples:
    all_ds_p_filtered =  all_ds_p.filter(lambda d,dc,t,m: tf.math.equal(t[-1]-t[0]+1,total_size))
    all_ds_t_filtered =  all_ds_t.filter(lambda d,dc,t,m: tf.math.equal(t[-1]-t[0]+1,total_size))
    all_ds_f_filtered =  all_ds_f.filter(lambda d,dc,t,m: tf.math.equal(t[-1]-t[0]+1,total_size))

    # Filter to keep only the lead time corresponding to the forecast start month.
    all_ds_f_mfiltered =  all_ds_f_filtered.map(lambda d,dc,t,m: [d[:,m[-forecast_size]-1::12], dc[:,m[-forecast_size]-1::12], t, m])

    # Then extract the separate data sets
    data_p_filtered = all_ds_p_filtered.map(lambda d,dc,t,m: d)
    data_p_clim_filtered =  all_ds_p_filtered.map(lambda d,dc,t,m: dc)
    time_p_filtered =  all_ds_p_filtered.map(lambda d,dc,t,m: t)

    data_t_filtered = all_ds_t_filtered.map(lambda d,dc,t,m: d)
    data_t_clim_filtered =  all_ds_t_filtered.map(lambda d,dc,t,m: dc)
    time_t_filtered =  all_ds_t_filtered.map(lambda d,dc,t,m: t)

    data_f_filtered = all_ds_f_mfiltered.map(lambda d,dc,t,m: d)
    data_f_clim_filtered =  all_ds_f_mfiltered.map(lambda d,dc,t,m: dc)
    time_f_filtered =  all_ds_f_mfiltered.map(lambda d,dc,t,m: t)


    # Shuffling data
    # !!!!! NOT SURE HOW TO DEAL WITH SHUFFLE AND RECONSTRUCT THE SHUFFLED TIME SERIES...
    # so we keep shuffle to False for now...
    # shuffle = False
    if shuffle:
        data_p_filtered = data_p_filtered.shuffle(shuffle_buffer_size, seed=42)
        data_p_clim_filtered =  data_p_clim_filtered.shuffle(shuffle_buffer_size, seed=42)
        time_p_filtered =  time_p_filtered.shuffle(shuffle_buffer_size, seed=42)

        data_t_filtered = data_t_filtered.shuffle(shuffle_buffer_size, seed=42)
        data_t_clim_filtered =  data_t_clim_filtered.shuffle(shuffle_buffer_size, seed=42)
        time_t_filtered =  time_t_filtered.shuffle(shuffle_buffer_size, seed=42)

        data_f_filtered = data_f_filtered.shuffle(shuffle_buffer_size, seed=42)
        data_f_clim_filtered =  data_f_clim_filtered.shuffle(shuffle_buffer_size, seed=42)
        time_f_filtered =  time_f_filtered.shuffle(shuffle_buffer_size, seed=42)


    # Now put into a single one to map to the output:
    data_filtered = tf.data.Dataset.zip((data_t_filtered, data_p_filtered, data_f_filtered))
    data_clim_filtered = tf.data.Dataset.zip((data_t_clim_filtered, data_p_clim_filtered, data_f_clim_filtered))
    time_filtered = time_p_filtered # Here time_p_filtered = time_f_filtered = time_t_filtered


    # Extracting (past features, forecasts, decoder initial recurrent input) + targets
    # NOTE : the initial decoder input is set as the last value of the target.
    if n_forecasts > 0:
        data_filtered = data_filtered.map(lambda t,p,f: ((p[:-forecast_size,:], # Past predictors samples
                                                          f[-forecast_size:,:], # Future forecasts samples
                                                          t[-forecast_size-1:-forecast_size,:] # Decoder input: last time step of target before prediction time starts
                                                          ),
                                                         t[-forecast_size:, :])) # Target samples during prediction time

        data_clim_filtered = data_clim_filtered.map(lambda t,p,f: (p[:-forecast_size,:], # Past predictor climatology samples
                                                                   t[-forecast_size:,:])) # Target climatology samples during prediction time



    else:
        data_filtered = data_filtered.map(lambda t,p,f: ((p[:-forecast_size,:],  # Past predictors samples
                                                          t[-forecast_size-1:-forecast_size,:] # Decoder input: last time step of target before prediction time starts
                                                          ),
                                                         t[-forecast_size:, :])) # Target samples during prediction time

        data_clim_filtered = data_clim_filtered.map(lambda t,p,f: (p[:-forecast_size,:], # Past predictor climatology samples
                                                                   t[-forecast_size:,:])) # Target climatology samples during prediction time

    time_filtered = time_filtered.map(lambda k: (k[:-forecast_size], # Time for past predictors samples
                                                  k[-forecast_size:]))    # Time for prediction samples


    return data_filtered.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE), data_clim_filtered.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE), time_filtered.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)




def create_dataset_tpf_nonrecursive(df_t_in,df_p_in,df_f_in,
                                    df_t_clim_in,df_p_clim_in, df_f_clim_in,
                                    time_in,month_in,
                                    n_forecasts,window_size,forecast_size,batch_size,
                                    shuffle = False,
                                    shuffle_buffer_size = 0):

    # Total size of window is given by the number of steps to be considered
    # before prediction time + steps that we want to forecast
    total_size = window_size + forecast_size

    # # Buffer size for shuffle:
    # shuffle_buffer_size = np.min([9500*5,len(df_t_in)]) # This number should be equal to or larger than the number of samples.

    # Selecting windows
    data_p = tf.data.Dataset.from_tensor_slices(df_p_in.values)
    data_p = data_p.window(total_size, shift=1, drop_remainder=True)
    data_p = data_p.flat_map(lambda k: k.batch(total_size))

    data_t = tf.data.Dataset.from_tensor_slices(df_t_in.values)
    data_t = data_t.window(total_size, shift=1, drop_remainder=True)
    data_t = data_t.flat_map(lambda k: k.batch(total_size))

    data_f = tf.data.Dataset.from_tensor_slices(df_f_in.values)
    data_f = data_f.window(total_size, shift=1, drop_remainder=True)
    data_f = data_f.flat_map(lambda k: k.batch(total_size))

    data_p_clim = tf.data.Dataset.from_tensor_slices(df_p_clim_in.values)
    data_p_clim = data_p_clim.window(total_size, shift=1, drop_remainder=True)
    data_p_clim = data_p_clim.flat_map(lambda k: k.batch(total_size))

    data_t_clim = tf.data.Dataset.from_tensor_slices(df_t_clim_in.values)
    data_t_clim = data_t_clim.window(total_size, shift=1, drop_remainder=True)
    data_t_clim = data_t_clim.flat_map(lambda k: k.batch(total_size))

    data_f_clim = tf.data.Dataset.from_tensor_slices(df_f_clim_in.values)
    data_f_clim = data_f_clim.window(total_size, shift=1, drop_remainder=True)
    data_f_clim = data_f_clim.flat_map(lambda k: k.batch(total_size))

    time = tf.data.Dataset.from_tensor_slices(time_in)
    time = time.window(total_size, shift=1, drop_remainder=True)
    time = time.flat_map(lambda k: k.batch(total_size))

    month = tf.data.Dataset.from_tensor_slices(month_in)
    month = month.window(total_size, shift=1, drop_remainder=True)
    month = month.flat_map(lambda k: k.batch(total_size))

    # Zip all datasets together so that we can filter out the samples
    # that are discontinuous in time due to cross-validation splits.
    all_ds_p = tf.data.Dataset.zip((data_p, data_p_clim, time, month))
    all_ds_t = tf.data.Dataset.zip((data_t, data_t_clim, time, month))
    all_ds_f = tf.data.Dataset.zip((data_f, data_f_clim, time, month))

    # Filter discontinuous samples:
    all_ds_p_filtered =  all_ds_p.filter(lambda d,dc,t,m: tf.math.equal(t[-1]-t[0]+1,total_size))
    all_ds_t_filtered =  all_ds_t.filter(lambda d,dc,t,m: tf.math.equal(t[-1]-t[0]+1,total_size))
    all_ds_f_filtered =  all_ds_f.filter(lambda d,dc,t,m: tf.math.equal(t[-1]-t[0]+1,total_size))

    # Filter to keep only the lead time corresponding to the forecast start month.
    all_ds_f_mfiltered =  all_ds_f_filtered.map(lambda d,dc,t,m: [d[:,m[-forecast_size]-1::12], dc[:,m[-forecast_size]-1::12], t, m])

    # Then extract the separate data sets
    data_p_filtered = all_ds_p_filtered.map(lambda d,dc,t,m: d)
    data_p_clim_filtered =  all_ds_p_filtered.map(lambda d,dc,t,m: dc)
    time_p_filtered =  all_ds_p_filtered.map(lambda d,dc,t,m: t)

    data_t_filtered = all_ds_t_filtered.map(lambda d,dc,t,m: d)
    data_t_clim_filtered =  all_ds_t_filtered.map(lambda d,dc,t,m: dc)
    time_t_filtered =  all_ds_t_filtered.map(lambda d,dc,t,m: t)

    data_f_filtered = all_ds_f_mfiltered.map(lambda d,dc,t,m: d)
    data_f_clim_filtered =  all_ds_f_mfiltered.map(lambda d,dc,t,m: dc)
    time_f_filtered =  all_ds_f_mfiltered.map(lambda d,dc,t,m: t)


    # Shuffling data
    # !!!!! NOT SURE HOW TO DEAL WITH SHUFFLE AND RECONSTRUCT THE SHUFFLED TIME SERIES...
    # so we keep shuffle to False for now...
    # shuffle = False
    if shuffle:
        data_p_filtered = data_p_filtered.shuffle(shuffle_buffer_size, seed=42)
        data_p_clim_filtered =  data_p_clim_filtered.shuffle(shuffle_buffer_size, seed=42)
        time_p_filtered =  time_p_filtered.shuffle(shuffle_buffer_size, seed=42)

        data_t_filtered = data_t_filtered.shuffle(shuffle_buffer_size, seed=42)
        data_t_clim_filtered =  data_t_clim_filtered.shuffle(shuffle_buffer_size, seed=42)
        time_t_filtered =  time_t_filtered.shuffle(shuffle_buffer_size, seed=42)

        data_f_filtered = data_f_filtered.shuffle(shuffle_buffer_size, seed=42)
        data_f_clim_filtered =  data_f_clim_filtered.shuffle(shuffle_buffer_size, seed=42)
        time_f_filtered =  time_f_filtered.shuffle(shuffle_buffer_size, seed=42)


    # Now put into a single one to map to the output:
    data_filtered = tf.data.Dataset.zip((data_t_filtered, data_p_filtered, data_f_filtered))
    data_clim_filtered = tf.data.Dataset.zip((data_t_clim_filtered, data_p_clim_filtered, data_f_clim_filtered))
    time_filtered = time_p_filtered # Here time_p_filtered = time_f_filtered = time_t_filtered


    # Extracting (past features, forecasts, decoder initial recurrent input) + targets
    # NOTE : the initial decoder input is set as the last value of the target.
    if n_forecasts > 0:
        data_filtered = data_filtered.map(lambda t,p,f: ((p[:-forecast_size,:], # Past predictors samples
                                                          f[-forecast_size:,:]  # Future forecasts samples
                                                          ),
                                                         t[-forecast_size:, :])) # Target samples during prediction time

        data_clim_filtered = data_clim_filtered.map(lambda t,p,f: (p[:-forecast_size,:], # Past predictor climatology samples
                                                                   t[-forecast_size:,:])) # Target climatology samples during prediction time


    time_filtered = time_filtered.map(lambda k: (k[:-forecast_size], # Time for past predictors samples
                                                  k[-forecast_size:]))    # Time for prediction samples


    return data_filtered.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE), data_clim_filtered.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE), time_filtered.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)




def reconstruct_ysamples_tpf(model,ds_w,ds_clim_w,time_w,sample_len,n_frcst_vars,is_anomaly,norm_type='None',scaler=None):
    # Get targets and predictions for all samples. These are the scaled
    # values of the samples that we got after the scaler.
    y_scaled = np.concatenate([y for x, y in ds_w], axis=0)
    y_pred_scaled = model.predict(ds_w)
    # print(y_pred_scaled.shape)
    time_y = np.concatenate([t_tar for t_pred, t_tar in time_w], axis=0)

    # All data must be retransformed back using the scaler.
    # The format of the data that was passed to the scaler was
    #     (n_timesteps, 1 (target)+ n_predictors columns+ n_forecast_vars):

    # We will reconstruct the target samples as:
    #    (nsamples, pred_len, 1)
    y_pred = np.zeros(y_pred_scaled.shape)
    y = np.zeros(y_scaled.shape)

    for i in range(sample_len):
        if norm_type == 'None':
            y_pred[:,i,0] = y_pred_scaled[:,i,:][:,0] # Here, 0 selects the target column.
            y[:,i,0] = y_scaled[:,i,:][:,0]
        else:
            y_pred[:,i,0] = scaler.inverse_transform(y_pred_scaled[:,i,:])[:,0] # Here, 0 selects the target column.
            y[:,i,0] = scaler.inverse_transform(y_scaled[:,i,:])[:,0]

    # The climatolgy was not scaled with the scaler, so we can
    # directly use the values
    y_clim = np.concatenate([tar_clim for pred_clim, tar_clim in ds_clim_w], axis=0)

    if is_anomaly:
        # Add the climatology back to the anomaly samples:
        y += y_clim
        y_pred += y_clim

    return y, y_pred, y_clim, time_y




def encoder_decoder_recursive(input_len,pred_len,npredictors,nfuturevars,latent_dim,nb_layers,inp_dropout,rec_dropout,dense_act_func):

    # ENCODER:
    past_inputs = keras.layers.Input(shape=(input_len, npredictors), name='past_inputs')
    encoder = keras.layers.LSTM(latent_dim, return_state=True,
                                dropout = inp_dropout,
                                recurrent_dropout = rec_dropout, name='encoder')
    encoder_outputs, encoder_state_h, encoder_state_c = encoder(past_inputs)
    # Discard encoder outputs and only keep the cell states and hidden states.
    encoder_states = [encoder_state_h, encoder_state_c]


    # DECODER: Process only one step at a time, reinjecting the output at step t as input to step t+1
    decoder_lstm_0 = keras.layers.LSTM(latent_dim, return_sequences=True,
                                     return_state=True,
                                     dropout = inp_dropout,
                                     recurrent_dropout = rec_dropout,
                                     input_shape=[None,1,nfuturevars+1], name='recursive_decoder_layer0')

    # decoder_lstm_1 = keras.layers.LSTM(latent_dim, return_sequences=True,
    #                                  return_state=True,
    #                                  dropout = inp_dropout,
    #                                  recurrent_dropout = rec_dropout,
    #                                  input_shape=[None,1,latent_dim], name='recursive_decoder_layer1')


    # The output of the dense layer is fixed at one to return only the predicted water temperature.
    # The sigmoid activation function ensures that the predicted values are positive and scaled between 0 and 1.
    # decoder_dense = keras.layers.Dense(1,activation=dense_act_func,name='Dense')
    decoder_dense = keras.layers.Dense(1,activation=None,name='Dense')

    # Set the initial state of the decoder to be the ouput state of the encoder
    states = encoder_states

    # Initalize the recursive and forecast inputs
    first_recursive_input = keras.layers.Input(shape=(1,1), name='first_recursive_input')

    if nfuturevars > 0 :
        it = -1
        future_inputs = keras.layers.Input(shape=(pred_len, nfuturevars), name='future_inputs')
        first_future_input = tf.keras.layers.Cropping1D(cropping=(0,pred_len-1), name='future_input_'+str(it+1))(future_inputs)
        inputs = tf.keras.layers.Concatenate(name='concat_input_'+str(it+1))([first_future_input, first_recursive_input])
    else:
        inputs = first_recursive_input

    all_outputs = []
    # print('Inputs shape: ', tf.shape(inputs))
    # print('States shape: ', tf.shape(states))
    for it in range(pred_len):
        # Run the decoder on one timestep
        outputs, state_h, state_c = decoder_lstm_0(inputs,initial_state=states)
        # print('outputs shape: ', tf.shape(outputs))
        # if nb_layers > 1:
        #     for l in range(nb_layers-1):
        #         outputs, state_h, state_c = decoder_lstm_1(outputs,initial_state=[state_h, state_c])
        #         # outputs, state_h, state_c = decoder_lstm_1(outputs,initial_state=states)

        outputs = decoder_dense(outputs)
        # print('outputs shape, after dense: ', tf.shape(outputs))
        # Store the current prediction (we will concatenate all predictions later)
        all_outputs.append(outputs)
        # print('all outputs shape (in loop): ', tf.shape(all_outputs))
        # Reinject the outputs as inputs for the next loop iteration and concatenate
        # with the forecast for the next time step
        # + update the states
        if nfuturevars > 0 :
            if it < pred_len - 1:
              inputs = tf.keras.layers.Concatenate(name='concat_input_'+str(it+1))([tf.keras.layers.Cropping1D(cropping=(it+1,pred_len-(it+2)),name='future_input_'+str(it+1))(future_inputs), outputs])
        else:
            inputs = outputs

        states = [state_h, state_c]

    # Concatenate all predictions
    # print('all outputs shape (final): ', tf.shape(all_outputs))
    # decoder_outputs_test = Lambda(lambda x: K.concatenate(x, axis=1), name='concat_outputs')(all_outputs)
    # decoder_outputs_test = tf.reshape(decoder_outputs_test,[tf.shape(decoder_outputs_test)[0],pred_len])
    decoder_outputs = tf.reshape(Lambda(lambda x: K.concatenate(x, axis=1), name='concat_outputs')(all_outputs),[tf.shape(future_inputs)[0],pred_len] )
    # print('all outputs shape (reshape): ', tf.shape(decoder_outputs))
    # print('Decoder output shape: ',decoder_outputs.shape)
    Tw_outputs = tf.expand_dims(decoder_outputs,axis=2,name='Tw_out')
    # print('FINAL OUTPUT shape: ', tf.shape(Tw_outputs))

    # densetw = keras.layers.Dense(pred_len,activation=dense_act_func,input_shape=[None,pred_len],name='Dense_Tw')
    # Tw_outputs = tf.expand_dims(densetw(decoder_outputs),axis=2,name='Tw_out')
    # print('Tw output shape: ',Tw_outputs.shape)

    # DEFINE MODEL:
    if nfuturevars > 0 :
        model = tf.keras.models.Model(inputs=[past_inputs, future_inputs, first_recursive_input], outputs=Tw_outputs)
    else:
        model = tf.keras.models.Model(inputs=[past_inputs, first_recursive_input], outputs=Tw_outputs)

    return model




def encoder_decoder(input_len,pred_len,npredictors,nfuturevars,latent_dim,nb_layers,inp_dropout,rec_dropout,dense_act_func):

    # INITIALIZE THE INPUT LAYERS:
    # Input shape: [None/Nsamples,window_size,nfeatures]
    past_inputs = keras.layers.Input(shape=(input_len, npredictors), name='past_inputs')
    future_inputs = keras.layers.Input(shape=(pred_len, nfuturevars), name='future_inputs')



    # ENCODER:
    encoder = keras.layers.LSTM(latent_dim, return_state=True,
                                dropout = inp_dropout,
                                recurrent_dropout = rec_dropout, name='encoder')
    encoder_outputs, encoder_state_h, encoder_state_c = encoder(past_inputs)
    # Discard encoder outputs and only keep the cell states and hidden states.
    encoder_states = [encoder_state_h, encoder_state_c]
    # encoder_tates shape: [2 states, None/Nsamples, latent_dim]



    # DECODER: Process only one step at a time, reinjecting the output at step t as input to step t+1
    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True,
                                     return_state=True,
                                     dropout = inp_dropout,
                                     recurrent_dropout = rec_dropout,
                                     input_shape=[None,pred_len,nfuturevars], name='recursive_decoder_layer0')

    # The output of the dense layer is fixed at one to return only the predicted water temperature. (i.e. 1 variable)
    # The sigmoid activation function ensures that the predicted values are positive and scaled between 0 and 1.
    # decoder_dense = keras.layers.Dense(1,activation=dense_act_func,name='Dense')
    decoder_dense = keras.layers.Dense(1,activation=None,name='Dense')

    # Set the initial state of the decoder to be the ouput state of the encoder
    # LSTM outputs shape (before dense):
    # [None/Nsamples, pred_len, latent_dim]
    outputs, _, _ = decoder_lstm(future_inputs,initial_state=encoder_states)

    decoder_outputs = decoder_dense(outputs) # [None, pred_len, 1]
    # print('outputs shape, after dense: ', tf.shape(decoder_outputs))
    # Dense outputs shape:
    # [None/Nsamples, pred_len, 1]


    # DEFINE MODEL:
    model = tf.keras.models.Model(inputs=[past_inputs, future_inputs], outputs=decoder_outputs)

    return model




def execute_fold(df_t_train_in, df_t_valid_in, df_t_test_in,
                 df_p_train_in, df_p_valid_in, df_p_test_in,
                 df_f_train_in, df_f_valid_in, df_f_test_in,
                 ind_train_in, ind_valid_in, ind_test_in,
                 train_years,time,
                 time_train,time_valid,time_test,
                 time_train_plot,time_valid_plot,time_test_plot,
                 month_train,month_valid,month_test,
                 latent_dim,nb_layers,inp_dropout,rec_dropout,dense_act_func,
                 n_pred_vars,n_frcst_vars,
                 input_len,pred_len,
                 norm_type,anomaly_target,anomaly_past,anomaly_frcst,
                 batch_size,lr_in,
                 loss_name,use_exp_decay_loss,tau,weights_on_timesteps,added_weight,Tw_thresh,
                 n_epochs,seed,
                 perfect_fcst,
                 ensemble_mean_train,ensemble_mean_fcst,nb_members,
                 model_name,
                 save_model = True,
                 nonrecursive = False,
                 continuous_training = False,
                 shuffle_training = False,
                 fixed_seed = False,
                 plot_loss=True,
                 plot_predictions=True,
                 plot_targets=False,
                 show_modelgraph=False,
                 show_weights=False,
                 verbose=True):


    # GET TRAINING CLIMATOLOGIES:
    df_t_train_clim, df_t_valid_clim, df_t_test_clim = get_predictor_clim_ensemble_fcst(df_t_train_in,df_t_valid_in,df_t_test_in,
                                                                                          ind_train_in,ind_valid_in,ind_test_in,
                                                                                          time_train,train_years,time,
                                                                                          df_is_frcst=False,
                                                                                          perfect_fcst=perfect_fcst,
                                                                                          nb_members=nb_members,
                                                                                          nw=1)

    df_p_train_clim, df_p_valid_clim, df_p_test_clim = get_predictor_clim_ensemble_fcst(df_p_train_in,df_p_valid_in,df_p_test_in,
                                                                                          ind_train_in,ind_valid_in,ind_test_in,
                                                                                          time_train,train_years,time,
                                                                                          df_is_frcst = False,
                                                                                          perfect_fcst = perfect_fcst,
                                                                                          nb_members=nb_members,
                                                                                          nw=1)

    df_f_train_clim, df_f_valid_clim, df_f_test_clim = get_predictor_clim_ensemble_fcst(df_f_train_in, df_f_valid_in, df_f_test_in,
                                                                                          ind_train_in,ind_valid_in,ind_test_in,
                                                                                          time_train,train_years,time,
                                                                                          df_is_frcst = True,
                                                                                          perfect_fcst = perfect_fcst,
                                                                                          nb_members=nb_members,
                                                                                          nw=1)

    # print('Nans in train_clim?' , np.any(np.sum(np.isnan(df_t_train_clim[df_t_train_clim.columns[:]])) > 0 ), np.any(np.sum(np.isnan(df_p_train_clim[df_p_train_clim.columns[:]])) > 0 ), np.any(np.sum(np.isnan(df_f_train_clim[df_f_train_clim.columns[:]])) > 0 ))
    # print('Nans in valid_clim?' ,np.any(np.sum(np.isnan(df_t_valid_clim[df_t_valid_clim.columns[:]])) > 0 ),np.any(np.sum(np.isnan(df_p_valid_clim[df_p_valid_clim.columns[:]])) > 0 ),np.any(np.sum(np.isnan(df_f_valid_clim[df_f_valid_clim.columns[:]])) > 0 ))
    # print('Nans in test_clim?' ,np.any(np.sum(np.isnan(df_t_test_clim[df_t_test_clim.columns[:]])) > 0 ),np.any(np.sum(np.isnan(df_p_test_clim[df_p_test_clim.columns[:]])) > 0 ),np.any(np.sum(np.isnan(df_f_test_clim[df_f_test_clim.columns[:]])) > 0 ))

    # print('Nans in train?' , np.any(np.sum(np.isnan(df_t_train_in[df_t_train_in.columns[:]])) > 0 ), np.any(np.sum(np.isnan(df_p_train_in[df_p_train_in.columns[:]])) > 0 ), np.any(np.sum(np.isnan(df_f_train_in[df_f_train_in.columns[:]])) > 0 ))
    # print('Nans in valid?' ,np.any(np.sum(np.isnan(df_t_valid_in[df_t_valid_in.columns[:]])) > 0 ),np.any(np.sum(np.isnan(df_p_valid_in[df_p_valid_in.columns[:]])) > 0 ),np.any(np.sum(np.isnan(df_f_valid_in[df_f_valid_in.columns[:]])) > 0 ))
    # print('Nans in test?' ,np.any(np.sum(np.isnan(df_t_test_in[df_t_test_in.columns[:]])) > 0 ),np.any(np.sum(np.isnan(df_p_test_in[df_p_test_in.columns[:]])) > 0 ),np.any(np.sum(np.isnan(df_f_test_in[df_f_test_in.columns[:]])) > 0 ))



    # REPLACE NAN WITH CLIMATOLOGICAL VALUES:
    df_t_train_in, df_t_valid_in, df_t_test_in = replace_nan_with_clim(df_t_train_in,df_t_valid_in,df_t_test_in,
                                                                        df_t_train_clim,df_t_valid_clim,df_t_test_clim,
                                                                        verbose = True)

    df_p_train_in, df_p_valid_in, df_p_test_in = replace_nan_with_clim(df_p_train_in,df_p_valid_in,df_p_test_in,
                                                                        df_p_train_clim,df_p_valid_clim,df_p_test_clim,
                                                                        verbose = True)

    # print('Nans in train_clim?' , np.any(np.sum(np.isnan(df_t_train_clim[df_t_train_clim.columns[:]])) > 0 ), np.any(np.sum(np.isnan(df_p_train_clim[df_p_train_clim.columns[:]])) > 0 ), np.any(np.sum(np.isnan(df_f_train_clim[df_f_train_clim.columns[:]])) > 0 ))
    # print('Nans in valid_clim?' ,np.any(np.sum(np.isnan(df_t_valid_clim[df_t_valid_clim.columns[:]])) > 0 ),np.any(np.sum(np.isnan(df_p_valid_clim[df_p_valid_clim.columns[:]])) > 0 ),np.any(np.sum(np.isnan(df_f_valid_clim[df_f_valid_clim.columns[:]])) > 0 ))
    # print('Nans in test_clim?' ,np.any(np.sum(np.isnan(df_t_test_clim[df_t_test_clim.columns[:]])) > 0 ),np.any(np.sum(np.isnan(df_p_test_clim[df_p_test_clim.columns[:]])) > 0 ),np.any(np.sum(np.isnan(df_f_test_clim[df_f_test_clim.columns[:]])) > 0 ))

    # print('Nans in train?' , np.any(np.sum(np.isnan(df_t_train_in[df_t_train_in.columns[:]])) > 0 ), np.any(np.sum(np.isnan(df_p_train_in[df_p_train_in.columns[:]])) > 0 ), np.any(np.sum(np.isnan(df_f_train_in[df_f_train_in.columns[:]])) > 0 ))
    # print('Nans in valid?' ,np.any(np.sum(np.isnan(df_t_valid_in[df_t_valid_in.columns[:]])) > 0 ),np.any(np.sum(np.isnan(df_p_valid_in[df_p_valid_in.columns[:]])) > 0 ),np.any(np.sum(np.isnan(df_f_valid_in[df_f_valid_in.columns[:]])) > 0 ))
    # print('Nans in test?' ,np.any(np.sum(np.isnan(df_t_test_in[df_t_test_in.columns[:]])) > 0 ),np.any(np.sum(np.isnan(df_p_test_in[df_p_test_in.columns[:]])) > 0 ),np.any(np.sum(np.isnan(df_f_test_in[df_f_test_in.columns[:]])) > 0 ))



    # REMOVE CLIMATOLOGY TO GET ANOMALIES:
    if anomaly_target:
        df_t_train_in.iloc[:,:] = df_t_train_in.iloc[:,:].values-df_t_train_clim.iloc[:,:].values
        df_t_valid_in.iloc[:,:] = df_t_valid_in.iloc[:,:].values-df_t_valid_clim.iloc[:,:].values
        df_t_test_in.iloc[:,:] = df_t_test_in.iloc[:,:].values-df_t_test_clim.iloc[:,:].values

    if anomaly_past:
        df_p_train_in.iloc[:,:] = df_p_train_in.iloc[:,:].values-df_p_train_clim.iloc[:,:].values
        df_p_valid_in.iloc[:,:] = df_p_valid_in.iloc[:,:].values-df_p_valid_clim.iloc[:,:].values
        df_p_test_in.iloc[:,:] = df_p_test_in.iloc[:,:].values-df_p_test_clim.iloc[:,:].values

    if anomaly_frcst:
        df_f_train_in.iloc[:,:] = df_f_train_in.iloc[:,:].values-df_f_train_clim.iloc[:,:].values
        df_f_valid_in.iloc[:,:] = df_f_valid_in.iloc[:,:].values-df_f_valid_clim.iloc[:,:].values
        df_f_test_in.iloc[:,:] = df_f_test_in.iloc[:,:].values-df_f_test_clim.iloc[:,:].values


    if plot_targets:
        plt.figure()
        plt.plot(time_train_plot,df_t_train_in.iloc[:,0:1], color='blue')
        plt.plot(time_valid_plot,df_t_valid_in.iloc[:,0:1],color='green')
        plt.plot(time_test_plot,df_t_test_in.iloc[:,0:1],color='red')
        if not anomaly_target:
            plt.plot(time_train_plot, df_t_train_clim.iloc[:,0:1], ':',color='cyan')
            plt.plot(time_valid_plot, df_t_valid_clim.iloc[:,0:1], ':',color='brown')
            plt.plot(time_test_plot, df_t_test_clim.iloc[:,0:1], ':',color='orange')


    # RESHAPE AND GET ENSEMBLE MEAN IF ensmeble_mean_train = True OR if ensemble_mean_fcst = True
    def get_ensemble_mean_df(df_in,nb_members):
        em = np.zeros((df_in.shape[0],int(df_in.shape[1]/nb_members)))*np.nan
        n_forecast_vars = int(df_in.shape[1]/(12*nb_members))
        for v in range(n_forecast_vars):
            vtmp =  df_in.iloc[:,v*12*nb_members:(v+1)*12*nb_members]
            for month in range(12):
                em[:,(12*v)+month] = np.nanmean(vtmp.iloc[:,month::12],axis=1)

        # return em
        return pd.DataFrame(em)

    def append_df_into_continuous(df_t_in,df_p_in,df_f_in,n_frcst_vars,time_in,month_in,nb_members):
        for m in range(nb_members):
            if m == 0:
                df_t_in_all = df_t_in
                df_p_in_all = df_p_in
                time_all = time_in
                month_all = month_in

                for v in range(n_frcst_vars):
                    if v == 0:
                        df_f_in_all = df_f_in.iloc[:,12*(m+v*nb_members):12*(m+v*nb_members+1)]
                    else:
                        df_f_in_all = pd.concat([df_f_in_all,df_f_in.iloc[:,12*(m+v*nb_members):12*(m+v*nb_members+1)]],axis=1)

            else:
                df_t_in_all = pd.concat([df_t_in_all,df_t_in])
                df_p_in_all = pd.concat([df_p_in_all,df_p_in])
                time_all = np.concatenate([time_all,time_in])
                month_all = np.concatenate([month_all,month_in])

                for v in range(n_frcst_vars):
                    if v == 0:
                        df_f_in_all_tmp = df_f_in.iloc[:,12*(m+v*nb_members):12*(m+v*nb_members+1)]
                    else:
                        df_f_in_all_tmp = pd.concat([df_f_in_all_tmp,df_f_in.iloc[:,12*(m+v*nb_members):12*(m+v*nb_members+1)]],axis=1)

                df_f_in_all = pd.concat([df_f_in_all,df_f_in_all_tmp])

        df_t_out = df_t_in_all
        df_p_out = df_p_in_all
        df_f_out = df_f_in_all
        time_out = time_all
        month_out = month_all

        return df_t_out, df_p_out, df_f_out, time_out, month_out



    if (not perfect_fcst):
        if ensemble_mean_train:
            df_f_train_in = get_ensemble_mean_df(df_f_train_in,nb_members)
            df_f_train_clim = get_ensemble_mean_df(df_f_train_clim,nb_members)

            if ensemble_mean_fcst:
                df_f_valid_in = get_ensemble_mean_df(df_f_valid_in,nb_members)
                df_f_valid_clim = get_ensemble_mean_df(df_f_valid_clim,nb_members)
                df_f_test_in = get_ensemble_mean_df(df_f_test_in,nb_members)
                df_f_test_clim = get_ensemble_mean_df(df_f_test_clim,nb_members)
            else:
                df_t_valid_in, df_p_valid_in, df_f_valid_in, time_valid, month_valid = append_df_into_continuous(df_t_valid_in,df_p_valid_in,df_f_valid_in,n_frcst_vars,time_valid,month_valid,nb_members)
                df_t_valid_clim, df_p_valid_clim, df_f_valid_clim, _, _ = append_df_into_continuous(df_t_valid_clim,df_p_valid_clim,df_f_valid_clim,n_frcst_vars,time_valid,month_valid,nb_members)
                df_t_test_in, df_p_test_in, df_f_test_in, time_test, month_test = append_df_into_continuous(df_t_test_in,df_p_test_in,df_f_test_in,n_frcst_vars,time_test,month_test,nb_members)
                df_t_test_clim, df_p_test_clim, df_f_test_clim, _, _ = append_df_into_continuous(df_t_test_clim,df_p_test_clim,df_f_test_clim,n_frcst_vars,time_test,month_test,nb_members)
        else:
            if continuous_training:
                df_t_train_in, df_p_train_in, df_f_train_in, time_train, month_train = append_df_into_continuous(df_t_train_in,df_p_train_in,df_f_train_in,n_frcst_vars,time_train,month_train,nb_members)
                df_t_train_clim, df_p_train_clim, df_f_train_clim, _, _ = append_df_into_continuous(df_t_train_clim,df_p_train_clim,df_f_train_clim,n_frcst_vars,time_train,month_train,nb_members)

                if ensemble_mean_fcst:
                    df_f_valid_in = get_ensemble_mean_df(df_f_valid_in,nb_members)
                    df_f_valid_clim = get_ensemble_mean_df(df_f_valid_clim,nb_members)
                    df_f_test_in = get_ensemble_mean_df(df_f_test_in,nb_members)
                    df_f_test_clim = get_ensemble_mean_df(df_f_test_clim,nb_members)
                else:
                    df_t_valid_in, df_p_valid_in, df_f_valid_in, time_valid, month_valid = append_df_into_continuous(df_t_valid_in,df_p_valid_in,df_f_valid_in,n_frcst_vars,time_valid,month_valid,nb_members)
                    df_t_valid_clim, df_p_valid_clim, df_f_valid_clim, _, _ = append_df_into_continuous(df_t_valid_clim,df_p_valid_clim,df_f_valid_clim,n_frcst_vars,time_valid,month_valid,nb_members)
                    df_t_test_in, df_p_test_in, df_f_test_in, time_test, month_test = append_df_into_continuous(df_t_test_in,df_p_test_in,df_f_test_in,n_frcst_vars,time_test,month_test,nb_members)
                    df_t_test_clim, df_p_test_clim, df_f_test_clim, _, _ = append_df_into_continuous(df_t_test_clim,df_p_test_clim,df_f_test_clim,n_frcst_vars,time_test,month_test,nb_members)

            # else:
            #     Training and testing with all members, all at once
            #     NOTHING TO DO!


    # DATA NORMALIZATION
    # Normalize all predictors, forecasts, and targets using only the training data
    if norm_type != 'None':
        scaler_t = fit_scaler(df_t_train_in,norm_type=norm_type)
        df_t_train_scaled = normalize_df(df_t_train_in,scaler_t)
        df_t_valid_scaled = normalize_df(df_t_valid_in,scaler_t)
        df_t_test_scaled = normalize_df(df_t_test_in,scaler_t)

        scaler_p = fit_scaler(df_p_train_in,norm_type=norm_type)
        df_p_train_scaled = normalize_df(df_p_train_in,scaler_p)
        df_p_valid_scaled = normalize_df(df_p_valid_in,scaler_p)
        df_p_test_scaled = normalize_df(df_p_test_in,scaler_p)

        scaler_f = fit_scaler(df_f_train_in,norm_type=norm_type)
        df_f_train_scaled = normalize_df(df_f_train_in,scaler_f)
        df_f_valid_scaled = normalize_df(df_f_valid_in,scaler_f)
        df_f_test_scaled = normalize_df(df_f_test_in,scaler_f)

        FU_threshold_t = np.ones((len(df_t_train_in),1))*Tw_thresh
        scaled_FU_threshold = scaler_t.transform(FU_threshold_t)[0,0]

    else:
        df_t_train_scaled = df_t_train_in
        df_t_valid_scaled = df_t_valid_in
        df_t_test_scaled = df_t_test_in

        df_p_train_scaled = df_p_train_in
        df_p_valid_scaled = df_p_valid_in
        df_p_test_scaled = df_p_test_in

        df_f_train_scaled = df_f_train_in
        df_f_valid_scaled = df_f_valid_in
        df_f_test_scaled = df_f_test_in

        FU_threshold = np.ones((len(df_t_train_in),1))*Tw_thresh
        scaled_FU_threshold = FU_threshold[0,0]




    # GET WINDOWED DATA SETS
    # Now we get training, validation, and test as tf.data.Dataset objects
    # The 'create_dataset' function returns batched datasets ('batch_size')
    # using a rolling window shifted by 1-day.
    # Buffer size for shuffle:
    if nonrecursive:
        train_windowed, train_clim_windowed, time_train_windowed = create_dataset_tpf_nonrecursive(df_t_train_scaled,df_p_train_scaled,df_f_train_scaled,
                                                                                                    df_t_train_clim,df_p_train_clim, df_f_train_clim,
                                                                                                    time_train,month_train,
                                                                                                    n_frcst_vars,
                                                                                                    input_len,pred_len,
                                                                                                    batch_size,
                                                                                                    shuffle = shuffle_training,
                                                                                                    shuffle_buffer_size = np.min([9500*5,len(df_t_train_scaled)]))

        valid_windowed, valid_clim_windowed, time_valid_windowed  = create_dataset_tpf_nonrecursive(df_t_valid_scaled,df_p_valid_scaled,df_f_valid_scaled,
                                                                                                    df_t_valid_clim,df_p_valid_clim, df_f_valid_clim,
                                                                                                    time_valid,month_valid,
                                                                                                    n_frcst_vars,
                                                                                                    input_len,pred_len,
                                                                                                    batch_size,
                                                                                                    shuffle = False,
                                                                                                    shuffle_buffer_size = np.min([9500*5,len(df_t_valid_scaled)]))

        test_windowed, test_clim_windowed, time_test_windowed  = create_dataset_tpf_nonrecursive(df_t_test_scaled,df_p_test_scaled,df_f_test_scaled,
                                                                                                  df_t_test_clim,df_p_test_clim, df_f_test_clim,
                                                                                                  time_test,month_test,
                                                                                                  n_frcst_vars,
                                                                                                  input_len,pred_len,
                                                                                                  batch_size,
                                                                                                  shuffle = False,
                                                                                                  shuffle_buffer_size = np.min([9500*5,len(df_t_test_scaled)]))


    else:
        train_windowed, train_clim_windowed, time_train_windowed = create_dataset_tpf(df_t_train_scaled,df_p_train_scaled,df_f_train_scaled,
                                                                                        df_t_train_clim,df_p_train_clim, df_f_train_clim,
                                                                                        time_train,month_train,
                                                                                        n_frcst_vars,
                                                                                        input_len,pred_len,
                                                                                        batch_size,
                                                                                        shuffle = shuffle_training,
                                                                                        shuffle_buffer_size = np.min([9500*5,len(df_t_train_scaled)]))

        valid_windowed, valid_clim_windowed, time_valid_windowed  = create_dataset_tpf(df_t_valid_scaled,df_p_valid_scaled,df_f_valid_scaled,
                                                                                        df_t_valid_clim,df_p_valid_clim, df_f_valid_clim,
                                                                                        time_valid,month_valid,
                                                                                        n_frcst_vars,
                                                                                        input_len,pred_len,
                                                                                        batch_size,
                                                                                        shuffle = False,
                                                                                        shuffle_buffer_size = np.min([9500*5,len(df_t_valid_scaled)]))

        test_windowed, test_clim_windowed, time_test_windowed  = create_dataset_tpf(df_t_test_scaled,df_p_test_scaled,df_f_test_scaled,
                                                                                      df_t_test_clim,df_p_test_clim, df_f_test_clim,
                                                                                      time_test,month_test,
                                                                                      n_frcst_vars,
                                                                                      input_len,pred_len,
                                                                                      batch_size,
                                                                                      # batch_size=1,
                                                                                      shuffle = False,
                                                                                      shuffle_buffer_size = np.min([9500*5,len(df_t_test_scaled)]))

    # print('TRAIN')
    # for t in train_windowed:
    #     print(np.sum(np.isnan(t[0][0])),np.sum(np.isnan(t[0][1])),np.sum(np.isnan(t[0][2])),np.sum(np.isnan(t[1])))
    # print('VALID')
    # for t in valid_windowed:
    #     print(np.sum(np.isnan(t[0][0])),np.sum(np.isnan(t[0][1])),np.sum(np.isnan(t[0][2])),np.sum(np.isnan(t[1])))
    # print('TEST')
    # for t in test_windowed:
    #     print(np.sum(np.isnan(t[0][0])),np.sum(np.isnan(t[0][1])),np.sum(np.isnan(t[0][2])),np.sum(np.isnan(t[1])))


    # FIX SEED:
    if fixed_seed:
        tf.keras.utils.set_random_seed(seed)


    # DEFINE CALLBACKS FOR TRAINING & SAVING:
    early_stop = tf.keras.callbacks.EarlyStopping(
                                                monitor="val_loss",
                                                # patience=5, # Set to 5 when using forecasts.
                                                patience=10,# Set to 8 or 10 when not using forecasts.
                                                min_delta=0.0001,
                                                verbose=1
                                            )
    lr_plateau =  tf.keras.callbacks.ReduceLROnPlateau(
                                                        monitor="val_loss",
                                                        factor=0.5,
                                                        patience=3,
                                                        verbose=1,
                                                        min_delta=0.0001,
                                                        min_lr=0.00001
                                                    )

    checkpoint_path = model_name+'_cp{epoch:04d}.ckpt'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    if save_model:
        training_callbacks = [tf.keras.callbacks.LearningRateScheduler(exp_scheduler),cp_callback]
    else:
        training_callbacks = [tf.keras.callbacks.LearningRateScheduler(exp_scheduler)]


    # IF MODEL ALREADY EXISTS LOAD IT, OTHERWISE BUILD MODEL:
    if os.path.exists(model_name+'.keras'):
        print('Loading already existing model.')
        model = keras.models.load_model(model_name+'.keras',compile=False)
        h = model.history
    else:
        if nonrecursive:
            model = encoder_decoder(input_len,pred_len,
                                    n_pred_vars,
                                    n_frcst_vars,
                                    latent_dim,
                                    nb_layers,
                                    inp_dropout,rec_dropout,
                                    dense_act_func)
        else:
            model = encoder_decoder_recursive(input_len,pred_len,
                                              n_pred_vars,
                                              n_frcst_vars,
                                              latent_dim,
                                              nb_layers,
                                              inp_dropout,rec_dropout,
                                              dense_act_func)

        optimizer = keras.optimizers.Adam(learning_rate=lr_in)
        # optimizer = keras.optimizers.SGD(learning_rate=lr_in)

        # COMPILE MODEL WITH CUSTOM LOSS FUNCTION:
        model.compile(optimizer=optimizer,
                      loss=loss_wraper(loss_name,use_exp_decay_loss,tau,weights_on_timesteps,added_weight,scaled_FU_threshold,pred_len),
                      metrics=["mae"])

        # IF CHECKPOINT ALREADY EXISTS, LOAD IT TO CONTINUE TRAINING FROM THERE:
        # latest = tf.train.latest_checkpoint(os.path.dirname(model_name))
        # if (latest is not None):
        #     model.load_weights(latest)

        # FIT/TRAIN MODEL:
        h = model.fit(train_windowed,
                      epochs = n_epochs,
                      validation_data = valid_windowed,
                      verbose = 2,
                      callbacks = training_callbacks)

        # SHOW TRAINING AND VALIDATION LOSS:
        # Compare the training and validation loss to diagnose overfitting
        # plot_loss = True
        if plot_loss:
            fig, ax = plt.subplots(figsize=[12, 6])
            ax.plot(h.history['loss'], 'o-')
            ax.plot(h.history['val_loss'], 'o-')
            date_ref = dt.date(1900,1,1)
            ax.set_title('Loss - ' + str((date_ref+dt.timedelta(days=int(time_test[30]))).year))
            plt.grid(True)
            plt.legend(['Loss', 'Val loss'])
            plt.xlabel('Number of epochs')
            plt.ylabel('MSE')

    # SHOW MODEL STRUCTURE:
    # show_modelgraph = True
    if show_modelgraph:
        plot_model(
            model,
            to_file='model_plot.png',
            show_shapes=True,
            show_layer_names=True
            )

    # SHOW MODEL WEIGHTS:
    # show_weights = False
    if show_weights:
        model.summary()


    # EVALUATE THE MODEL PERFORMANCE
    # The model performance is evaluated using typical regression metrics (i.e. MAE, RMSE, R$^2$)

    # Reconstrcut predictions and targets, by scaling them back to their
    # original scales, and in the format (nsamples, pred_len):
    if norm_type == 'None':
        y_train,y_pred_train,y_clim_train,target_time_train = reconstruct_ysamples_tpf(model,train_windowed,train_clim_windowed,time_train_windowed,pred_len,n_frcst_vars,anomaly_target,norm_type=norm_type)
        y_valid,y_pred_valid,y_clim_valid,target_time_valid = reconstruct_ysamples_tpf(model,valid_windowed,valid_clim_windowed,time_valid_windowed,pred_len,n_frcst_vars,anomaly_target,norm_type=norm_type)
        y_test,y_pred_test,y_clim_test,target_time_test = reconstruct_ysamples_tpf(model,test_windowed,test_clim_windowed,time_test_windowed,pred_len,n_frcst_vars,anomaly_target,norm_type=norm_type)
    else:
        y_train,y_pred_train,y_clim_train,target_time_train = reconstruct_ysamples_tpf(model,train_windowed,train_clim_windowed,time_train_windowed,pred_len,n_frcst_vars,anomaly_target,norm_type=norm_type,scaler=scaler_t)
        y_valid,y_pred_valid,y_clim_valid,target_time_valid = reconstruct_ysamples_tpf(model,valid_windowed,valid_clim_windowed,time_valid_windowed,pred_len,n_frcst_vars,anomaly_target,norm_type=norm_type,scaler=scaler_t)
        y_test,y_pred_test,y_clim_test,target_time_test = reconstruct_ysamples_tpf(model,test_windowed,test_clim_windowed,time_test_windowed,pred_len,n_frcst_vars,anomaly_target,norm_type=norm_type,scaler=scaler_t)

    # Get the regression metrics:
    y_train = np.squeeze(y_train); y_clim_train = np.squeeze(y_clim_train)
    y_valid = np.squeeze(y_valid); y_clim_valid = np.squeeze(y_clim_valid)
    y_test = np.squeeze(y_test); y_clim_test = np.squeeze(y_clim_test)

    y_pred_train = np.squeeze(y_pred_train)
    y_pred_valid = np.squeeze(y_pred_valid)
    y_pred_test = np.squeeze(y_pred_test)

    rsqr_train, mae_train, rmse_train =  regression_metrics(y_train,y_pred_train)
    rsqr_valid, mae_valid, rmse_valid =  regression_metrics(y_valid,y_pred_valid)
    rsqr_test, mae_test, rmse_test =  regression_metrics(y_test,y_pred_test)

    if verbose:
        print('TRAINING ---')
        print('Rsqr = '+ str(np.round(rsqr_train, 2)))
        print('MAE = '+ str(np.round(mae_train, 2)))
        print('RMSE = '+ str(np.round(rmse_train, 2)))
        print(' ')
        print('VALIDATION ---')
        print('Rsqr = '+ str(np.round(rsqr_valid, 2)))
        print('MAE = '+ str(np.round(mae_valid, 2)))
        print('RMSE = '+ str(np.round(rmse_valid, 2)))
        print(' ')
        print('TEST ---')
        print('Rsqr = '+ str(np.round(rsqr_test, 2)))
        print('MAE = '+ str(np.round(mae_test, 2)))
        print('RMSE = '+ str(np.round(rmse_test, 2)))
        print(' ')
        print('============================================')
        print(' ')
        print('TRAINING ---')
        print('Rsqr = '+ str(np.round(np.mean(rsqr_train), 4)))
        print('MAE = '+ str(np.round(np.mean(mae_train), 4)))
        print('RMSE = '+ str(np.round(np.mean(rmse_train), 4)))
        print(' ')
        print('VALIDATION ---')
        print('Rsqr = '+ str(np.round(np.mean(rsqr_valid), 4)))
        print('MAE = '+ str(np.round(np.mean(mae_valid), 4)))
        print('RMSE = '+ str(np.round(np.mean(rmse_valid), 4)))
        print(' ')
        print('TEST ---')
        print('Rsqr = '+ str(np.round(np.mean(rsqr_test), 4)))
        print('MAE = '+ str(np.round(np.mean(mae_test), 4)))
        print('RMSE = '+ str(np.round(np.mean(rmse_test), 4)))


    if plot_predictions:
        # Plot predictions - TRAINING
        plot_prediction_timeseries(y_pred_train,y_train,y_clim_train,target_time_train, pred_type = 'training', lead=0, nyrs_plot= 2)
        plot_prediction_timeseries(y_pred_train,y_train,y_clim_train,target_time_train, pred_type = 'training', lead=50, nyrs_plot= 2)

        # Plot predictions - TESTING
        plot_prediction_timeseries(y_pred_test,y_test,y_clim_test,target_time_test,pred_type='testing', lead=0, nyrs_plot= 2)
        plot_prediction_timeseries(y_pred_test,y_test,y_clim_test,target_time_test,pred_type='testing', lead=50, nyrs_plot= 2)


    if (not os.path.exists(model_name+'.keras')):
        if save_model:
            model.save(model_name+'.keras',overwrite=False,save_format='keras')#,custom_objects={"loss_wraper": loss_wraper})
            # model.save(model_name+'.keras',overwrite=True,save_format='keras')#,custom_objects={"custom_loss": loss_wraper(loss_name,use_exp_decay_loss,tau,weights_on_timesteps,added_weight,scaled_FU_threshold,pred_len)})

        print('RETURNING HISTORY')
        return model,h.history['loss'],h.history['val_loss'],target_time_train,target_time_valid,target_time_test,y_pred_train,y_pred_valid,y_pred_test,y_train,y_valid,y_test,y_clim_train,y_clim_valid,y_clim_test,train_windowed,valid_windowed,test_windowed,train_clim_windowed,valid_clim_windowed,test_clim_windowed,time_train_windowed,time_valid_windowed,time_test_windowed,scaler_t,scaled_FU_threshold
    else:
        if save_model:
            print('NOT SAVING MODEL: A saved model already exists with the same name.')

        print('NOT RETURNING HISTORY')
        return model,h                ,h                    ,target_time_train,target_time_valid,target_time_test,y_pred_train,y_pred_valid,y_pred_test,y_train,y_valid,y_test,y_clim_train,y_clim_valid,y_clim_test,train_windowed,valid_windowed,test_windowed,train_clim_windowed,valid_clim_windowed,test_clim_windowed,time_train_windowed,time_valid_windowed,time_test_windowed,scaler_t,scaled_FU_threshold


