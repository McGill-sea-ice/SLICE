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

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from functions import rolling_climo
from functions_ML import regression_metrics, plot_prediction_timeseries

#%%

def fit_scaler(df_train_in,norm_type='MinMax'):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler

    if norm_type =='MinMax': scaler = MinMaxScaler()
    if norm_type =='Standard': scaler = StandardScaler()

    return scaler.fit(df_train_in)




def normalize_df(df_in,scaler):
    scaled_values = scaler.transform(df_in.values)

    for ip, p in enumerate(df_in.columns):
        df_in.iloc[:,ip] = scaled_values[:,ip]

    return df_in



def get_predictor_clim(df_tr,df_va,df_te,
                       ind_tr,ind_va,ind_te,
                       time_tr,tr_years,time_in,
                       nw=1,
                       verbose = True):

    df_tr_clim = np.zeros((df_tr.shape))*np.nan
    df_va_clim = np.zeros((df_va.shape))*np.nan
    df_te_clim = np.zeros((df_te.shape))*np.nan

    for ip, p in enumerate(df_tr.columns):
        if verbose: print(p, df_tr.iloc[:,ip].name)
        p_clim_mean, p_clim_std, _ = rolling_climo(nw, df_tr.iloc[:,ip].values,'other',time_tr,tr_years,time_other=time_in)
        df_tr_clim[:,ip] = p_clim_mean[ind_tr]
        df_va_clim[:,ip] = p_clim_mean[ind_va]
        df_te_clim[:,ip] = p_clim_mean[ind_te]

    df_tr_clim = pd.DataFrame(df_tr_clim,columns= df_tr.columns)
    df_va_clim = pd.DataFrame(df_va_clim,columns= df_va.columns)
    df_te_clim = pd.DataFrame(df_te_clim,columns= df_te.columns)
    
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




def create_dataset(df, df_clim, time_in,
                   n_forecasts,
                   window_size, forecast_size,
                   batch_size,
                   shuffle=False):
    """
    SEE WEB EXAMPLE:
    (https://www.angioi.com/time-series-encoder-decoder-tensorflow/)
    """

    # Total size of window is given by the number of steps to be considered
    # before prediction time + steps that we want to forecast
    total_size = window_size + forecast_size

    # Selecting windows
    data = tf.data.Dataset.from_tensor_slices(df.values)
    data = data.window(total_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda k: k.batch(total_size))

    data_clim = tf.data.Dataset.from_tensor_slices(df_clim.values)
    data_clim = data_clim.window(total_size, shift=1, drop_remainder=True)
    data_clim = data_clim.flat_map(lambda k: k.batch(total_size))

    time = tf.data.Dataset.from_tensor_slices(time_in)
    time = time.window(total_size, shift=1, drop_remainder=True)
    time = time.flat_map(lambda k: k.batch(total_size))

    # Zip all datasets together so that we can filter out the samples
    # that are discontinuous in time due to cross-validation splits.
    all_ds = tf.data.Dataset.zip((data, data_clim, time))
    all_ds_filtered =  all_ds.filter(lambda d,dc,t: tf.math.equal(t[-1]-t[0]+1,total_size))

    # Then extract the separate data sets
    data_filtered = all_ds_filtered.map(lambda d,dc,t: d)
    data_clim_filtered =  all_ds_filtered.map(lambda d,dc,t: dc)
    time_filtered =  all_ds_filtered.map(lambda d,dc,t: t)

    # Shuffling data
    # !!!!! NOT SURE HOW TO DEAL WITH SHUFFLE AND RECONSTRUCT THE SHUFFLED TIME SERIES...
    # so we keep shuffle to False for now...
    shuffle = False
    if shuffle:
        shuffle_buffer_size = len(data_filtered) # This number can be changed
    #     data = data.shuffle(shuffle_buffer_size, seed=42)
        data_filtered = data_filtered.shuffle(shuffle_buffer_size, seed=42)
        data_clim_filtered =  data_clim_filtered.shuffle(shuffle_buffer_size, seed=42)
        time_filtered =  time_filtered.shuffle(shuffle_buffer_size, seed=42)

    # Extracting (past features, forecasts, decoder initial recurrent input) + targets
    # NOTE : the initial decoder input is set as the last value of the target.
    if n_forecasts > 0:
        data_filtered = data_filtered.map(lambda k: ((k[:-forecast_size,1:-n_forecasts], # Past predictors samples
                                                      k[-forecast_size:, -n_forecasts:], # Future forecasts samples
                                                      k[-forecast_size-1:-forecast_size,0:1] # Decoder input: last time step of target before prediction time starts
                                                     ),
                                                 k[-forecast_size:, 0:1])) # Target samples during prediction time

        data_clim_filtered = data_clim_filtered.map(lambda k: (k[:-forecast_size,1:-n_forecasts], # Past predictor climatology samples
                                                               k[-forecast_size:,0:1])) # Target climatology samples during prediction time



    else:
        data_filtered = data_filtered.map(lambda k: ((k[:-forecast_size,1:],  # Past predictors samples
                                                      k[-forecast_size-1:-forecast_size,0:1] # Decoder input: last time step of target before prediction time starts
                                                     ),
                                                 k[-forecast_size:, 0:1])) # Target samples during prediction time

        data_clim_filtered = data_clim_filtered.map(lambda k: (k[:-forecast_size,1:], # Past predictor climatology samples
                                                               k[-forecast_size:,0:1])) # Target climatology samples during prediction time

    time_filtered = time_filtered.map(lambda k: (k[:-forecast_size], # Time for past predictors samples
                                                 k[-forecast_size:]))    # Time for prediction samples



    return data_filtered.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE), data_clim_filtered.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE), time_filtered.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)




def reconstruct_ysamples(model,ds_w,ds_clim_w,time_w,scaler,sample_len,n_frcst_vars,is_anomaly):
    # Get targets and predictions for all samples. These are the scaled
    # values of the samples that we got after the scaler.
    y_scaled = np.concatenate([y for x, y in ds_w], axis=0)
    y_pred_scaled = model.predict(ds_w)
    time_y = np.concatenate([t_tar for t_pred, t_tar in time_w], axis=0)

    # Get past predictors and forecasts as well because
    # they will be needed to reverse the scaler.
    X = np.concatenate([x[0] for x, y in ds_w], axis=0)
    if n_frcst_vars > 0:
        F = np.concatenate([x[1] for x, y in ds_w], axis=0)

    # All data must be retransformed back using the scaler.
    # The format of the data that was passed to the scaler was
    #     (n_timesteps, 1 (target)+ n_predictors columns+ n_forecast_vars):

    # We will reconstruct the target samples as:
    #    (nsamples, pred_len, 1)
    y_pred = np.zeros(y_pred_scaled.shape)
    y = np.zeros(y_scaled.shape)

    for i in range(sample_len):
        if n_frcst_vars > 0:
            y_pred[:,i,0] = scaler.inverse_transform(np.concatenate((y_pred_scaled[:,i,:], X[:,i,:], F[:,i,:]), axis=1))[:,0] # Here, 0 selects the target column.
            y[:,i,0] = scaler.inverse_transform(np.concatenate((y_scaled[:,i,:], X[:,i,:], F[:,i,:]), axis=1))[:,0]
        else:
            y_pred[:,i,0] = scaler.inverse_transform(np.concatenate((y_pred_scaled[:,i,:], X[:,i,:]), axis=1))[:,0] # Here, 0 selects the target column.
            y[:,i,0] = scaler.inverse_transform(np.concatenate((y_scaled[:,i,:], X[:,i,:]), axis=1))[:,0]

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

    decoder_lstm_1 = keras.layers.LSTM(latent_dim, return_sequences=True,
                                     return_state=True,
                                     dropout = inp_dropout,
                                     recurrent_dropout = rec_dropout,
                                     input_shape=[None,1,latent_dim], name='recursive_decoder_layer1')


    # The output of the dense layer is fixed at one to return only the predicted water temperature.
    # The sigmoid activation function ensures that the predicted values are positive and scaled between 0 and 1.
    decoder_dense = keras.layers.Dense(1,activation=dense_act_func,name='Dense')

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

    for it in range(pred_len):
        # Run the decoder on one timestep
        outputs, state_h, state_c = decoder_lstm_0(inputs,initial_state=states)
        
        if nb_layers > 1:
            for l in range(nb_layers-1):
                outputs, state_h, state_c = decoder_lstm_1(outputs,initial_state=[state_h, state_c])
                # outputs, state_h, state_c = decoder_lstm_1(outputs,initial_state=states)
                
        
        outputs = decoder_dense(outputs)

        # Store the current prediction (we will concatenate all predictions later)
        all_outputs.append(outputs)

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
    # decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1), name='concat_outputs')(all_outputs)
    # decoder_outputs = tf.reshape(decoder_outputs,[tf.shape(decoder_outputs)[0],pred_len])
    decoder_outputs = tf.reshape(Lambda(lambda x: K.concatenate(x, axis=1), name='concat_outputs')(all_outputs),[tf.shape(past_inputs)[0],pred_len] )
    
    densetw = keras.layers.Dense(pred_len,activation=dense_act_func,input_shape=[None,pred_len],name='Dense_Tw')
    Tw_outputs = tf.expand_dims(densetw(decoder_outputs),axis=2,name='Tw_out')

    # DEFINE MODEL:
    if nfuturevars > 0 :
        model = tf.keras.models.Model(inputs=[past_inputs, future_inputs, first_recursive_input], outputs=Tw_outputs)
    else:
        model = tf.keras.models.Model(inputs=[past_inputs, first_recursive_input], outputs=Tw_outputs)

    return model




def execute_fold(df_train_in, df_valid_in, df_test_in,
                  ind_train_in, ind_valid_in, ind_test_in,
                  train_years,time,
                  time_train,time_valid,time_test,
                  time_train_plot,time_valid_plot,time_test_plot,
                  latent_dim,nb_layers,inp_dropout,rec_dropout,dense_act_func,
                  n_pred_vars,n_frcst_vars,
                  input_len,pred_len,
                  norm_type,anomaly_target,anomaly_past,anomaly_frcst,
                  batch_size,lr_in,
                  loss_name,use_exp_decay_loss,tau,weights_on_timesteps,added_weight,Tw_thresh,
                  n_epochs,
                  plot_loss=True,plot_predictions=True,
                  plot_targets=False,show_modelgraph=False,show_weights=False,
                  verbose=True):

    
    # GET TRAINING CLIMATOLOGIES:
    df_train_clim, df_valid_clim, df_test_clim = get_predictor_clim(df_train_in,df_valid_in,df_test_in,
                                                                    ind_train_in,ind_valid_in,ind_test_in,
                                                                    time_train,train_years,time,nw=1,verbose = True)
    
    print(df_train_in.shape, df_train_clim.shape)
    # REPLACE NAN WITH CLIMATOLOGICAL VALUES:
    df_train_in, df_valid_in, df_test_in = replace_nan_with_clim(df_train_in,df_valid_in,df_test_in,
                                                        df_train_clim,df_valid_clim,df_test_clim,
                                                        verbose = True)


    # print(df_train_in.shape, df_train_clim.shape)
    # REMOVE CLIMATOLOGY TO GET ANOMALIES:
    if anomaly_target:
        # print(df_train_in.shape, df_train_clim.shape)
        # print(df_train_in.iloc[:,0:1].shape, df_train_clim.iloc[:,0:1].shape)
        df_train_in.iloc[:,0:1] = df_train_in.iloc[:,0:1].values-df_train_clim.iloc[:,0:1].values
        df_valid_in.iloc[:,0:1] = df_valid_in.iloc[:,0:1].values-df_valid_clim.iloc[:,0:1].values
        df_test_in.iloc[:,0:1] = df_test_in.iloc[:,0:1].values-df_test_clim.iloc[:,0:1].values

    if anomaly_past:
        if n_frcst_vars > 0:
            df_train_in.iloc[:,1:-n_frcst_vars] = df_train_in.iloc[:,1:-n_frcst_vars].values-df_train_clim.iloc[:,1:-n_frcst_vars].values
            df_valid_in.iloc[:,1:-n_frcst_vars] = df_valid_in.iloc[:,1:-n_frcst_vars].values-df_valid_clim.iloc[:,1:-n_frcst_vars].values
            df_test_in.iloc[:,1:-n_frcst_vars] = df_test_in.iloc[:,1:-n_frcst_vars].values-df_test_clim.iloc[:,1:-n_frcst_vars].values
        else:
            df_train_in.iloc[:,1:] = df_train_in.iloc[:,1:].values-df_train_clim.iloc[:,1:].values
            df_valid_in.iloc[:,1:] = df_valid_in.iloc[:,1:].values-df_valid_clim.iloc[:,1:].values
            df_test_in.iloc[:,1:] = df_test_in.iloc[:,1:].values-df_test_clim.iloc[:,1:].values

    if anomaly_frcst:
        if n_frcst_vars > 0:
            # !!!! WHEN USING REAL FORECAST, THIS IS WHERE I CAN COMPUTE THE FORECAST ANOMALY.
            # OR I SIMPLY KEEP THIS AS FALSE AND PASS IN THE FORECAST VARIABLES AS ANOMALIES ALREADY...
            df_train_in.iloc[:,-n_frcst_vars:] = df_train_in.iloc[:,-n_frcst_vars:].values-df_train_clim.iloc[:,-n_frcst_vars:].values
            df_valid_in.iloc[:,-n_frcst_vars:] = df_valid_in.iloc[:,-n_frcst_vars:].values-df_valid_clim.iloc[:,-n_frcst_vars:].values
            df_test_in.iloc[:,-n_frcst_vars:] = df_test_in.iloc[:,-n_frcst_vars:].values-df_test_clim.iloc[:,-n_frcst_vars:].values

    # plot_targets = True
    if plot_targets:
        plt.figure()
        plt.plot(time_train_plot,df_train_in.iloc[:,0:1], color='blue')
        plt.plot(time_valid_plot,df_valid_in.iloc[:,0:1],color='green')
        plt.plot(time_test_plot, df_test_in.iloc[:,0:1],color='red')
        if not anomaly_target:
            plt.plot(time_train_plot, df_train_clim.iloc[:,0:1], ':',color='cyan')
            plt.plot(time_valid_plot, df_valid_clim.iloc[:,0:1], ':',color='brown')
            plt.plot(time_test_plot, df_test_clim.iloc[:,0:1], ':',color='orange')


    # DATA NORMALIZATION
    # Normalize all predictors, forecasts, and targets using only the training data
    scaler = fit_scaler(df_train_in,norm_type=norm_type)
    df_train_scaled = normalize_df(df_train_in,scaler)
    df_valid_scaled = normalize_df(df_valid_in,scaler)
    df_test_scaled = normalize_df(df_test_in,scaler)

    target_scaler = fit_scaler(df_train_in.iloc[:,0:1],norm_type=norm_type)
    FU_threshold = np.ones((len(df_train_in),1))*Tw_thresh
    scaled_FU_threshold = target_scaler.transform(FU_threshold)[0,0]
    # print(scaled_FU_threshold)


    # GET WINDOWED DATA SETS
    # Now we get training, validation, and test as tf.data.Dataset objects
    # The 'create_dataset' function returns batched datasets ('batch_size')
    # using a rolling window shifted by 1-day.
    train_windowed, train_clim_windowed, time_train_windowed = create_dataset(df_train_scaled, df_train_clim, time_train,
                                                                              n_frcst_vars,
                                                                              input_len, pred_len,
                                                                              batch_size,
                                                                              shuffle = False)

    valid_windowed, valid_clim_windowed, time_valid_windowed  = create_dataset(df_valid_scaled, df_valid_clim, time_valid,
                                                                                n_frcst_vars,
                                                                                input_len, pred_len,
                                                                                batch_size,
                                                                                shuffle = False)

    test_windowed, test_clim_windowed, time_test_windowed  = create_dataset(df_test_scaled, df_test_clim, time_test,
                                                                            n_frcst_vars,
                                                                            input_len, pred_len,
                                                                            batch_size=1,
                                                                            shuffle = False)

    # BUILD MODEL:
    model = encoder_decoder_recursive(input_len,pred_len,
                                      n_pred_vars,
                                      n_frcst_vars,
                                      latent_dim,
                                      nb_layers,
                                      inp_dropout,rec_dropout,
                                      dense_act_func)

    optimizer = keras.optimizers.Adam(learning_rate=lr_in)

    # DEFINE CALLBACKS FOR TRAINING:
    early_stop = tf.keras.callbacks.EarlyStopping(
                                                monitor="val_loss",
                                                # patience=5, # Set to 5 when using forecasts.
                                                patience=8,# Set to 8 or 10 when not using forecasts.
                                                min_delta=0,
                                                verbose=1
                                            )
    lr_plateau =  tf.keras.callbacks.ReduceLROnPlateau(
                                                        monitor="val_loss",
                                                        factor=0.5,
                                                        patience=5,
                                                        verbose=1,
                                                        min_delta=0.0001,
                                                        min_lr=0.001
                                                    )
    training_callbacks = [lr_plateau,early_stop]


    def loss_wraper(loss_name,use_exp_decay_loss,tau,weights_on_timesteps,added_weight,scaled_FU_threshold):
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

    # COMPILE MODEL WITH CUSTOM LOSS FUNCTION:
    model.compile(optimizer=optimizer, loss=loss_wraper(loss_name,use_exp_decay_loss,tau,weights_on_timesteps,added_weight,scaled_FU_threshold),
                    metrics=["mae"]
                    )


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
    y_train,y_pred_train,y_clim_train,target_time_train = reconstruct_ysamples(model,train_windowed,train_clim_windowed,time_train_windowed,scaler,pred_len,n_frcst_vars,anomaly_target)
    y_valid,y_pred_valid,y_clim_valid,target_time_valid = reconstruct_ysamples(model,valid_windowed,valid_clim_windowed,time_valid_windowed,scaler,pred_len,n_frcst_vars,anomaly_target)
    y_test,y_pred_test,y_clim_test,target_time_test = reconstruct_ysamples(model,test_windowed,test_clim_windowed,time_test_windowed,scaler,pred_len,n_frcst_vars,anomaly_target)

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

    # plot_predictions = True
    if plot_predictions:
        # Plot predictions - TRAINING
        plot_prediction_timeseries(y_pred_train,y_train,y_clim_train,target_time_train, pred_type = 'training', lead=0, nyrs_plot= 2)
        plot_prediction_timeseries(y_pred_train,y_train,y_clim_train,target_time_train, pred_type = 'training', lead=50, nyrs_plot= 2)

        # Plot predictions - TESTING
        plot_prediction_timeseries(y_pred_test,y_test,y_clim_test,target_time_test,pred_type='testing', lead=0, nyrs_plot= 2)
        plot_prediction_timeseries(y_pred_test,y_test,y_clim_test,target_time_test,pred_type='testing', lead=50, nyrs_plot= 2)

    return model,h.history['loss'],h.history['val_loss'],target_time_train,target_time_valid,target_time_test,y_pred_train,y_pred_valid,y_pred_test,y_train,y_valid,y_test,y_clim_train,y_clim_valid,y_clim_test
    


