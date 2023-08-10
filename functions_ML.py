#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:56:01 2022

@author: Amelie
"""

import sklearn.metrics as metrics
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import calendar
import scipy as sp
import pandas as pd
import statsmodels.api as sm
from netCDF4 import Dataset
from matplotlib import dates as mdates

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#%%%==============================================================

###
def plot_prediction_timeseries(ypred_ts_in,yobs_ts_in,yclim_in,time_in,pred_type, lead=0, nyrs_plot= 1,show_clim=True, year_in = None):
    fig, ax = plt.subplots(figsize=[5, 6])
    date_ref = dt.date(1900,1,1)
    first_day = (date_ref+dt.timedelta(days=int(time_in[0:nyrs_plot*365,lead][0]))).toordinal()
    last_day = (date_ref+dt.timedelta(days=int(time_in[0:nyrs_plot*365,lead][-1]))).toordinal()
    # time_plot = np.arange(first_day, last_day + 1)
    time_plot = [(date_ref+dt.timedelta(days=int(time_in[k][lead]))) for k in range(time_in.shape[0])]

    ax.plot(time_plot,yobs_ts_in[0:nyrs_plot*365,lead], color='black', label='Observed T$_w$')

    if show_clim:
            ax.plot(time_plot,yclim_in[0:nyrs_plot*365,lead], color='C1', label='Daily climatology')

    if pred_type == 'train':
        ax.plot(time_plot,ypred_ts_in[0:nyrs_plot*365,lead], color='C0', label='Predicted T$_w$')
    if pred_type == 'valid':
        ax.plot(time_plot,ypred_ts_in[0:nyrs_plot*365,lead], color='C0', label='Predicted T$_w$')
    if pred_type == 'test':
        ax.plot(time_plot,ypred_ts_in[0:nyrs_plot*365,lead], color='C0', label='Predicted T$_w$')

    ax.plot(time_plot,np.ones(len(ypred_ts_in[0:nyrs_plot*365,lead]))*0.75, '--',color='gray' )

    ax.set_xticks(np.arange(time_plot[0], time_plot[-1], step=120))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    ax.set_xlabel('Time')
    ax.set_ylabel('Water temperature $[^{\circ}C]$')
    ax.legend(loc='best')
    ax.grid(True)
    ax.set_title(str(year_in)+' LSTM model in '+pred_type+'. Lead = ' + str(lead))
    fig.show()
    plt.draw()
    plt.show()


def plot_sample(ypred_ts_in,yobs_ts_in,yclim_in,time_in,it,pred_type,axes_in=None,show_obs=True,show_clim=True,show_legend=True,show_label=True,date_ref = dt.date(1900,1,1)):
    if axes_in is None:
        fig, ax = plt.subplots(figsize=[4, 6])
    else:
        ax = axes_in

    time_plot = [(date_ref+dt.timedelta(days=int(time_in[it][k]))) for k in range(time_in.shape[-1])]

    # time_plot=time_in[it]
    if show_obs:
        if show_label:
            ax.plot(time_plot,yobs_ts_in[it], color='black', label='Observed T$_w$')
        else:
            ax.plot(time_plot,yobs_ts_in[it], color='black')

    if pred_type == 'train':
        if show_label:
            ax.plot(time_plot,ypred_ts_in[it], color='C0', label='Predicted T$_w$')
        else:
            ax.plot(time_plot,ypred_ts_in[it], color='C0')

    if pred_type == 'test':
        if show_label:
            ax.plot(time_plot,ypred_ts_in[it], color='C0', label='Predicted T$_w$')
        else:
            ax.plot(time_plot,ypred_ts_in[it], color='C0')

    if pred_type == 'valid':
        if show_label:
            ax.plot(time_plot,ypred_ts_in[it], color='C0', label='Predicted T$_w$')
        else:
            ax.plot(time_plot,ypred_ts_in[it], color='C0')


    if show_label: ax.plot(time_plot,np.ones(len(ypred_ts_in[it]))*0.75, '--',color='gray' )


    if show_clim:
        if show_label:
            ax.plot(time_plot,yclim_in[it], color='C1', label='Daily climatology')
        else:
            ax.plot(time_plot,yclim_in[it], color='C1')

    ax.set_xticks(np.arange(time_plot[0], time_plot[-1], step=14))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    ax.set_xlabel('Time')
    ax.set_ylabel('Water temperature $[^{\circ}C]$')
    if show_legend: ax.legend(loc='best')
    ax.grid(True)
    ax.set_title('LSTM model in '+pred_type+'. ')
    plt.show()


###
def regression_metrics(y_true, y_pred, output_opt = 'raw_values',verbose=False):
    """
    This function computes regression metrics, i.e.
     - Mean Absolute Error (MAE)
     - Root Mean Square Error (RMSE)
     - Coefficient of determination (R^2)

    :param y_true: Observed traget
    :param y_pred: Predicted target

    :return: mae, rmse, rsqr:
    """
    # explained_variance = metrics.explained_variance_score(y_true, y_pred,multioutput=output_opt)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred,multioutput=output_opt)
    mse = metrics.mean_squared_error(y_true, y_pred,multioutput=output_opt)
    # mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred,multioutput=output_opt)
    # median_absolute_error=metrics.median_absolute_error(y_true, y_pred,multioutput=output_opt)
    r2 = metrics.r2_score(y_true, y_pred,multioutput=output_opt)

    if verbose:
        # print('explained_variance: ', round(explained_variance,2))
        print('r2: ', np.round(r2,2))
        print('MAE: ', np.round(mean_absolute_error,2))
        print('RMSE: ', np.round(np.sqrt(mse),2))

    return r2, mean_absolute_error, np.sqrt(mse)


###
def torch_train_model(epoch, model, train_loader, optimizer, loss_fct, device, verbose=True):
    # activate the training mode
    model.train()
    torch.set_grad_enabled(True)
    total_loss = 0

    # iteration over the mini-batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # transfer the data on the chosen device
        data, target = data.to(device), target.to(device)
        # reinitialize the gradients to zero
        # optimizer.zero_grad()
        # Apparently, the is better than optimizer.zero_grad() for memory:
        for param in model.parameters():
            param.grad = None
        # forward propagation on the data
        prediction = model(data)
        # compute the cost function w.r.t. the targets
        loss = (loss_fct(prediction, target))
        # execute the backpropagation
        loss.backward()
        # execute an optimization step
        optimizer.step()
        # accumulate the loss
        total_loss += loss.item()*len(data)

    # compute the average cost per epoch
    mean_loss = total_loss/len(train_loader.dataset)

    if verbose:
        print('Train Epoch: {}   Avg_Loss: {:.5f}  '.format(epoch, mean_loss))
    return mean_loss

###
def torch_train_model_early_stopping(nepochs, model, train_loader, eval_loader, patience, optimizer, loss_fct, device, verbose=True):

    # Initialize loss and trigger count for early stoppping
    last_loss = 100
    trigger_count = 0

    # Initialize loss arrays
    train_loss = np.zeros((nepochs))*np.nan
    valid_loss = np.zeros((nepochs))*np.nan

    for epoch in range(0, nepochs):

        # activate the training mode
        model.train()
        torch.set_grad_enabled(True)
        total_loss = 0

        # iteration over the mini-batches
        for batch_idx, (data, target) in enumerate(train_loader):
            # transfer the data on the chosen device
            data, target = data.to(device), target.to(device)
            # reinitialize the gradients to zero
            # optimizer.zero_grad()
            # Apparently, the is better than optimizer.zero_grad() for memory:
            for param in model.parameters():
                param.grad = None
            # forward propagation on the data
            prediction = model(data)
            # compute the cost function w.r.t. the targets
            loss = (loss_fct(prediction, target))
            # execute the backpropagation
            loss.backward()
            # execute an optimization step
            optimizer.step()
            # accumulate the loss
            total_loss += loss.item()*len(data)

        # compute the average cost per epoch
        train_loss[epoch] = total_loss/len(train_loader.dataset)

        if verbose:
            print('Train Epoch: {}   Avg_Loss: {:.5f}  '.format(epoch+1, train_loss[epoch]))


        # Early stopping
        current_loss = torch_evaluate_model(model, eval_loader, loss_fct, device, verbose=True)
        valid_loss[epoch] = current_loss

        if current_loss > last_loss:
            trigger_count += 1
            print('trigger:', trigger_count)

            if trigger_count >= patience:
                print('Early stopping!')
                return train_loss, valid_loss, model

        else:
            trigger_count = 0

        last_loss = current_loss

    return train_loss, valid_loss, model


###
def torch_evaluate_model(model, eval_loader, loss_fct, device, verbose=True):
    # activate the evaluation mode
    model.eval()
    total_loss = 0

    with torch.no_grad():
        # iterate over the batches
        for batch_idx, (data, target) in enumerate(eval_loader):
            # transfer the data on the chosen device
            data, target = data.to(device), target.to(device)
            # forward propagation on the data
            prediction = model(data)
            # compute the cost function w.r.t. the targets
            loss = (loss_fct(prediction, target))
            # accumulate the loss
            total_loss += loss.item()*len(data)

    # compute the average cost per epoch
    mean_loss = total_loss/len(eval_loader.dataset)

    if verbose:
        print('Eval:  Avg_Loss: {:.5f}  '.format(mean_loss))
    return mean_loss


###
def plot_losses(t_loss,v_loss):
    xplot = list(range(len(t_loss)))
    plt.figure()
    plt.subplot(111)
    plt.plot(xplot, t_loss, 'r', label="Train")
    plt.plot(xplot, v_loss, 'm', label="Validation")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=False, fancybox=False)
    leg.get_frame().set_alpha(0.99)
    plt.grid()


###
def make_dataset_from_numpy(x_window,y_window,batch_size,shuffle_opt):
    x = torch.from_numpy(x_window).float()
    y = torch.from_numpy(y_window).float()
    data_loader = DataLoader(TensorDataset(x, y), batch_size, shuffle=shuffle_opt)

    return data_loader


###
def sliding_window_samples(data,time_in,input_width,label_width,shift,nslide,input_columns,label_columns):
    """
                                                       SHIFT
                                           <------------------------->
                                           |                         |
                                           |                         |

    sample 1: |____________________________|           |_____________|
                        input_width                     prediction window

    sample 2:         |____________________________|           |_____________|
                                input_width                     prediction window
              |       |
    sample 3: |       |       |____________________________|           |_____________|
              |       |       |         input_width                     prediction window
              |       |       |
              |       |       |
              <-------><------>
                NSLIDE  NSLIDE
    etc.

    input_width: width of the context window
    label_width: width of the prediction window
    nslide: by how much the input windows slides forward between each samples
    shift: delay between the end of the context window and the end of the prediction window.

    if shift == label_width,
        then the prediction window starts right after the context window.
    """

    # data shape is (time, features)
    input_data = data[:,input_columns]
    label_data = data[:,label_columns]

    # Only complete samples are created, remainder of inout and output arrays aere ignored.
    nsamples = np.floor((data.shape[0]-(input_width+shift))/float(nslide)).astype(int)+1

    Xout = np.zeros((nsamples,input_width,len(input_columns)))*np.nan
    Yout = np.zeros((nsamples,label_width,len(label_columns)))*np.nan

    input_time = np.zeros((nsamples,input_width))*np.nan
    label_time = np.zeros((nsamples,label_width))*np.nan

    iw_n = 0
    for iw in range(nsamples):
        istart = (iw*nslide)
        iend   = istart+input_width

        if (np.sum(np.isnan(input_data[istart:iend])) > 0 ) | (np.sum(np.isnan(label_data[iend+shift-label_width:iend+shift])) > 0 ):
            # Do not consider example if there is one or more "nan" in it.
            continue
        else:
            Xout[iw_n] = input_data[istart:iend]
            Yout[iw_n] = label_data[iend+shift-label_width:iend+shift]
            input_time[iw_n] = time_in[istart:iend]
            label_time[iw_n] = time_in[iend+shift-label_width:iend+shift]

            iw_n += 1

    Xout = Xout[0:iw_n]
    Yout = Yout[0:iw_n]
    input_time = input_time[0:iw_n]
    label_time = label_time[0:iw_n]

    return Xout, Yout, input_time, label_time

###
def plot_series(time, series, format="-", start=0, end=None, ax_labels=['Time','Value'],linecolor=''):
    plt.plot(time[start:end], series[start:end], format, color=linecolor)
    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])
    plt.grid(True)

###
def plot_series_1step(time, series, format="-", ax_labels=['Time','Value'],linecolor=''):
    plt.plot(time, series, format, color=linecolor)
    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])
    plt.grid(True)


###
def reconstruct_ts(targets,predictions,yr_input,time_in,Tw_climatology_mean,t_range,t_offset,input_width, shift, label_width, nslide, normalize_target, normalize_predictors, ytransform, train_range_pred, train_offset_pred, predictors,plot_pred_ts = False):
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
            print(targets_recons.shape,input_width,shift,label_width)
            plot_series_1step(time_in[:,0],np.squeeze(clim_recons[:,0]),'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(0))
            for s in np.arange(0,targets_recons.shape[0]-(input_width+shift),label_width):
                plot_series(time_in[s],np.array(targets_recons)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor='black')
                plot_series(time_in[s],np.array(predictions_recons)[s,:],'-',ax_labels=['Time', 'T$_{w}$'],linecolor=plt.get_cmap('tab20')(2))

    return predictions_recons, targets_recons, clim_recons


###
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=15, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


###
class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)







