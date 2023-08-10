#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:23:12 2022

@author: Amelie
"""

#%%
local_path = '/storage/amelie/'
# local_path = '/Volumes/SeagateUSB/McGill/Postdoc/'
#%%
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath(local_path +'slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import datetime as dt
import itertools
import calendar

import statsmodels.api as sm
from netCDF4 import Dataset

from functions import K_to_C
from functions import detect_FUD_from_Tw, detrend_ts, linear_fit
from functions_MLR import get_monthly_vars_from_daily, remove_collinear_features
from functions_MLR import find_models,eval_accuracy_multiple_models,make_metric_df
from functions_MLR import find_all_column_combinations
from functions_ML import regression_metrics
from functions_encoderdecoder import save_SEAS5_predictor_arrays
from analysis.SEAS5.SEAS5_forecast_class import SEAS5frcst

import sklearn.metrics as metrics
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


#%%
def loo_eval_new(y_in,model_pred,frcst_start_month,years_loo,years_y,years_SEAS5,years_pred,df_in,SEAS5_pred,SEAS51_pred,model,model_type,em='mean',anomaly=False,anomaly_type='mean',normalize=True,X_frcst=None,show_coeff=False,plot=True,verbose=True):

    def pred_str_to_month(pname):
        if pname.split()[0] == 'Jan.' : month = 1
        if pname.split()[0] == 'Feb.' : month = 2
        if pname.split()[0] == 'Mar.' : month = 3
        if pname.split()[0] == 'Apr.' : month = 4
        if pname.split()[0] == 'May.' : month = 5
        if pname.split()[0] == 'Jun.' : month = 6
        if pname.split()[0] == 'Jul.' : month = 7
        if pname.split()[0] == 'Aug.' : month = 8
        if pname.split()[0] == 'Sep.' : month = 9
        if pname.split()[0] == 'Oct.' : month = 10
        if pname.split()[0] == 'Nov.' : month = 11
        if pname.split()[0] == 'Dec.' : month = 12

        return month


    def get_X_train_test(train_yrs,test_yr):

        def get_df_data():
            # Use ERA5 (past observations) predictors
            if verbose: print('Using ERA5 for',p )
            if test_yr in years_pred:
                v_train = df_in[p].iloc[[np.where(years_pred == year)[0][0] for year in train_yrs if year in years_pred]]
                v_test = df_in[p].iloc[[np.where(years_pred == year)[0][0] for year in test_yr if year in years_pred]]
                if anomaly:
                    if anomaly_type == 'linear':
                        var_in = v_train
                        [m,b],_ = linear_fit(train_yrs, var_in)
                        v_train = v_train-(m*train_yrs + b)
                        v_test = v_test-(m*test_yr + b)
                    if anomaly_type == 'mean':
                        v_clim = np.nanmean(v_train)
                        v_train = v_train - v_clim
                        v_test = v_test - v_clim
                return v_train, v_test
            else:
                raise Exception('No predictor data for test year: ', test_yr)
            return None


        Xtrain = np.zeros((len(train_yrs),len(model_pred)))*np.nan
        Xtest = np.zeros((1,len(model_pred)))*np.nan

        for i,p in enumerate(model_pred):
            m = pred_str_to_month(p)

            if frcst_start_month == 13:
                # Perfect forecast exepriment - Use only osb.
                v_train, v_test = get_df_data()
                Xtrain[[np.where(train_yrs == year)[0][0] for year in train_yrs if year in years_pred],i] = v_train
                Xtest[:,i] = v_test
            else:
                # Real-world forecast exepriment - Use obs. + SEAS5 predictors
                if m >= frcst_start_month:
                    # Use SEAS5 (forecast) predictors
                    lead = m-frcst_start_month
                    if (p.split()[-1] == 'Ta_mean'):
                        if test_yr < 2022:
                            if verbose: print('Using SEAS5 frcst for', p)
                            vf_train = SEAS5_Ta_mean_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in train_yrs if year in years_SEAS5] ,lead,:]
                            vf_test = SEAS5_Ta_mean_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in test_yr if year in years_SEAS5] ,lead,:]
                            if anomaly:
                                if anomaly_type == 'linear':
                                    var_in = np.nanmean(SEAS5_Ta_mean_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in clim_years_SEAS5 if year != test_yr],lead,:],axis=1)
                                    [m,b],_ = linear_fit(clim_years_SEAS5[clim_years_SEAS5 != test_yr], var_in)
                                    for mnb in range(Nmembers):
                                        vf_train[:,mnb] = vf_train[:,mnb]-(m*train_yrs + b)
                                    vf_test = vf_test-(m*test_yr + b)
                                if anomaly_type == 'mean':
                                    vf_clim = np.nanmean(SEAS5_Ta_mean_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in clim_years_SEAS5 if year != test_yr],lead,:])
                                    vf_train = vf_train - vf_clim
                                    vf_test = vf_test - vf_clim
                            if em == 'mean':
                                Xtrain[[np.where(train_yrs == year)[0][0] for year in train_yrs if year in years_SEAS5],i] = np.nanmean(vf_train, axis=1)
                                Xtest[:,i] = np.nanmean(vf_test,axis=1)
                            else:
                                Xtrain[[np.where(train_yrs == year)[0][0] for year in train_yrs if year in years_SEAS5],i] = vf_train[:,int(em)]
                                Xtest[:,i] = vf_test[:,int(em)]
                        else:
                            if verbose: print('Using SEAS5.1 frcst for', p)
                            vf_train = SEAS51_Ta_mean_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in train_yrs if year in years_SEAS5] ,lead,:]
                            vf_test = SEAS51_Ta_mean_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in test_yr if year in years_SEAS5] ,lead,:]
                            if anomaly:
                                if anomaly_type == 'linear':
                                    var_in = np.nanmean(SEAS51_Ta_mean_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in clim_years_SEAS51 if year != test_yr],lead,:],axis=1)
                                    [m,b],_ = linear_fit(clim_years_SEAS51[clim_years_SEAS51 != test_yr], var_in)
                                    for mnb in range(Nmembers):
                                        vf_train[:,mnb] = vf_train[:,mnb]-(m*train_yrs + b)
                                    vf_test = vf_test-(m*test_yr + b)
                                if anomaly_type == 'mean':
                                    vf_clim = np.nanmean(SEAS51_Ta_mean_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in clim_years_SEAS51 if year != test_yr],lead,:])
                                    vf_train = vf_train - vf_clim
                                    vf_test = vf_test - vf_clim
                            if em == 'mean':
                                Xtrain[[np.where(train_yrs == year)[0][0] for year in train_yrs if year in years_SEAS5],i] = np.nanmean(vf_train, axis=1)
                                Xtest[:,i] = np.nanmean(vf_test,axis=1)
                            else:
                                Xtrain[[np.where(train_yrs == year)[0][0] for year in train_yrs if year in years_SEAS5],i] = vf_train[:,int(em)]
                                Xtest[:,i] = vf_test[:,int(em)]

                    elif (p.split()[-1] == 'snowfall') :
                        if test_yr < 2022:
                            if verbose: print('Using SEAS5 frcst for', p)
                            vf_train = SEAS5_snowfall_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in train_yrs if year in years_SEAS5] ,lead,:]
                            vf_test = SEAS5_snowfall_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in test_yr if year in years_SEAS5] ,lead,:]
                            if anomaly:
                                if anomaly_type == 'linear':
                                    var_in = np.nanmean(SEAS5_snowfall_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in clim_years_SEAS5 if year != test_yr],lead,:],axis=1)
                                    [m,b],_ = linear_fit(clim_years_SEAS5[clim_years_SEAS5 != test_yr], var_in)
                                    for mnb in range(Nmembers):
                                        vf_train[:,mnb] = vf_train[:,mnb]-(m*train_yrs + b)
                                    vf_test = vf_test-(m*test_yr + b)
                                if anomaly_type == 'mean':
                                    vf_clim = np.nanmean(SEAS5_snowfall_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in clim_years_SEAS5 if year != test_yr],lead,:])
                                    vf_train = vf_train - vf_clim
                                    vf_test = vf_test - vf_clim
                            if em == 'mean':
                                Xtrain[[np.where(train_yrs == year)[0][0] for year in train_yrs if year in years_SEAS5],i] = np.nanmean(vf_train, axis=1)
                                Xtest[:,i] = np.nanmean(vf_test,axis=1)
                            else:
                                Xtrain[[np.where(train_yrs == year)[0][0] for year in train_yrs if year in years_SEAS5],i] = vf_train[:,int(em)]
                                Xtest[:,i] = vf_test[:,int(em)]
                        else:
                            if verbose: print('Using SEAS5.1 frcst for', p)
                            vf_train = SEAS51_snowfall_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in train_yrs if year in years_SEAS5] ,lead,:]
                            vf_test = SEAS51_snowfall_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in test_yr if year in years_SEAS5] ,lead,:]
                            if anomaly:
                                if anomaly_type == 'linear':
                                    var_in = np.nanmean(SEAS51_snowfall_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in clim_years_SEAS51 if year != test_yr],lead,:],axis=1)
                                    [m,b],_ = linear_fit(clim_years_SEAS51[clim_years_SEAS51 != test_yr], var_in)
                                    for mnb in range(Nmembers):
                                        vf_train[:,mnb] = vf_train[:,mnb]-(m*train_yrs + b)
                                    vf_test = vf_test-(m*test_yr + b)
                                if anomaly_type == 'mean':
                                    vf_clim = np.nanmean(SEAS51_snowfall_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in clim_years_SEAS51 if year != test_yr],lead,:])
                                    vf_train = vf_train - vf_clim
                                    vf_test = vf_test - vf_clim
                            if em == 'mean':
                                Xtrain[[np.where(train_yrs == year)[0][0] for year in train_yrs if year in years_SEAS5],i] = np.nanmean(vf_train, axis=1)
                                Xtest[:,i] = np.nanmean(vf_test,axis=1)
                            else:
                                Xtrain[[np.where(train_yrs == year)[0][0] for year in train_yrs if year in years_SEAS5],i] = vf_train[:,int(em)]
                                Xtest[:,i] = vf_test[:,int(em)]

                    else:
                        raise Exception('Error: SEAS5 predictor variable does not exist.')

                else:
                    v_train, v_test = get_df_data()
                    Xtrain[[np.where(train_yrs == year)[0][0] for year in train_yrs if year in years_pred],i] = v_train
                    Xtest[:,i] = v_test

        return Xtrain, Xtest


    # Extract SEAS5 predictors
    SEAS5_Ta_mean_frcst = SEAS5_pred.get('SEAS5_Ta_mean')
    SEAS51_Ta_mean_frcst = SEAS51_pred.get('SEAS51_Ta_mean')
    SEAS5_snowfall_frcst = SEAS5_pred.get('SEAS5_snowfall')
    SEAS51_snowfall_frcst = SEAS51_pred.get('SEAS51_snowfall')
    clim_years_SEAS5 = np.arange(1993,2017)
    clim_years_SEAS51 = np.arange(1981,2017)

    # Prepare variables to hold model predictions
    Nmembers = 25
    y_pred_test = np.zeros((len(years_loo)))
    y_plot = np.zeros((len(years_loo)))

    # Sample predictand on LOO years only
    y_loo = y_in.iloc[[np.where(years_y == year)[0][0] for year in years_loo]]

    # Prepare plot for regression coefficients if needed
    if show_coeff:
        fig_c,ax_c = plt.subplots(nrows = int(np.ceil((len(years_loo)/4))), ncols = 4,sharex=True,sharey=True)

    # Begin LOO model evaluation:
    loo = LeaveOneOut()

    for i,[train_index, test_index] in enumerate(loo.split(years_loo)):
        if verbose:
            print('------------')
            print('TEST YEAR = ', years_loo[i])

        # Get X/y train/test:
        test_year = years_loo[test_index]
        train_years = years_loo[train_index]

        X_train, X_test = get_X_train_test(train_years,test_year)
        y_train, y_test = y_loo.iloc[train_index], y_loo.iloc[test_index]
        if (int(np.sum(np.isnan(X_train))+np.sum(np.isnan(y_train))+np.sum(np.isnan(X_test))+np.sum(np.isnan(y_test))) != 0):
            y_pred_test[i] = np.nan
            continue

        # Standardize predictors:
        Xscaler = StandardScaler()
        yscaler = StandardScaler()
        Xscaler = Xscaler.fit(X_train)
        yscaler = yscaler.fit(y_train)
        X_train_scaled = Xscaler.transform(X_train)
        X_test_scaled = Xscaler.transform(X_test)

        # Get Prediction:
        if normalize:
            X_train_fit = X_train_scaled
            X_test_fit = X_test_scaled
        else:
            X_train_fit = X_train
            X_test_fit = X_test
        y_pred_test[i] = model.fit(X_train_fit,y_train).predict(X_test_fit)
        y_plot[i] = y_test.values

        if not np.all(model.coef_ == 0):
            if verbose:
                print('Reg. coeff.: ', model.coef_[model.coef_ != 0])
                print('Stand. coeff.: ', model.coef_[model.coef_ != 0]*(np.nanstd(X_train_scaled,axis=0)/np.nanstd(y_train)))

            if model_type == 'lasso':
                x = np.where(model.coef_)[0]
            else:
                x = np.arange((len(model_pred)))

            if show_coeff:
                if i/len(years_loo) < 0.25:
                    ax_c[i,0].stem(
                    x,
                    model.coef_[model.coef_ != 0],
                    markerfmt="x")
                elif i/len(years_loo) < 0.5:
                    ax_c[i-int(np.ceil((len(years_loo)/4))),1].stem(
                    x,
                    model.coef_[model.coef_ != 0],
                    markerfmt="x")
                elif i/len(years_loo) < 0.75:
                    ax_c[i-2*int(np.ceil((len(years_loo)/4))),2].stem(
                    x,
                    model.coef_[model.coef_ != 0],
                    markerfmt="x")
                else:
                    ax_c[i-3*int(np.ceil((len(years_loo)/4))),3].stem(
                    x,
                    model.coef_[model.coef_ != 0],
                    markerfmt="x")

    if plot:
        plt.figure()
        plt.plot(years_loo,y_plot,'-o',color='k')
        plt.plot(years_loo,y_pred_test,'-o',color=plt.get_cmap('tab20c')(0))

    mae = np.nanmean(np.abs(y_plot[:]-y_pred_test[:]))
    acc7days = np.sum(np.abs(y_plot[:]-y_pred_test[:]) <= 7)/(np.sum(~np.isnan(y_loo)))

    if verbose:
        print('MAE:', mae)
        print('7-day accuracy: ', float(acc7days))


    # And finally, fit model with all data:
    # X_train = X_in
    # y_train = y_loo
    # # Xscaler = MinMaxScaler()
    # Xscaler = StandardScaler()
    # Xscaler = Xscaler.fit(X_train)
    # X_train_scaled = Xscaler.transform(X_train)

    # if normalize:
    #     X_train_fit = X_train_scaled
    # else:
    #     X_train_fit = X_train
    # # X_train_fit = X_train_scaled
    # model = model.fit(X_train_fit,y_train)

    return y_pred_test#, model, Xscaler, mae, acc7days


def loo_eval_train_mean_eval_all(y_in,model_pred,frcst_start_month,years_loo,years_y,years_SEAS5,years_pred,df_in,SEAS5_pred,SEAS51_pred,model,model_type,em='mean',anomaly=False,anomaly_type='mean',normalize=True,X_frcst=None,show_coeff=False,plot=True,verbose=True):

    def pred_str_to_month(pname):
        if pname.split()[0] == 'Jan.' : month = 1
        if pname.split()[0] == 'Feb.' : month = 2
        if pname.split()[0] == 'Mar.' : month = 3
        if pname.split()[0] == 'Apr.' : month = 4
        if pname.split()[0] == 'May.' : month = 5
        if pname.split()[0] == 'Jun.' : month = 6
        if pname.split()[0] == 'Jul.' : month = 7
        if pname.split()[0] == 'Aug.' : month = 8
        if pname.split()[0] == 'Sep.' : month = 9
        if pname.split()[0] == 'Oct.' : month = 10
        if pname.split()[0] == 'Nov.' : month = 11
        if pname.split()[0] == 'Dec.' : month = 12

        return month


    def get_X_train_test(train_yrs,test_yr):

        def get_df_data():
            # Use ERA5 (past observations) predictors
            if verbose: print('Using ERA5 for',p )
            if test_yr in years_pred:
                v_train = df_in[p].iloc[[np.where(years_pred == year)[0][0] for year in train_yrs if year in years_pred]]
                v_test = df_in[p].iloc[[np.where(years_pred == year)[0][0] for year in test_yr if year in years_pred]]
                if anomaly:
                    if anomaly_type == 'linear':
                        var_in = v_train
                        [m,b],_ = linear_fit(train_yrs, var_in)
                        v_train = v_train-(m*train_yrs + b)
                        v_test = v_test-(m*test_yr + b)
                    if anomaly_type == 'mean':
                        v_clim = np.nanmean(v_train)
                        v_train = v_train - v_clim
                        v_test = v_test - v_clim
                return v_train, v_test
            else:
                raise Exception('No predictor data for test year: ', test_yr)
            return None

        Nmembers = 25
        Xtrain = np.zeros((len(train_yrs),len(model_pred)))*np.nan
        Xtest = np.zeros((1,len(model_pred),Nmembers))*np.nan

        for i,p in enumerate(model_pred):
            m = pred_str_to_month(p)

            # Real-world forecast exepriment - Use obs. + SEAS5 predictors
            if m >= frcst_start_month:
                # Use SEAS5 (forecast) predictors
                lead = m-frcst_start_month
                if (p.split()[-1] == 'Ta_mean'):
                    if test_yr < 2022:
                        if verbose: print('Using SEAS5 frcst for', p)
                        vf_train = SEAS5_Ta_mean_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in train_yrs if year in years_SEAS5] ,lead,0:Nmembers]
                        vf_test = SEAS5_Ta_mean_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in test_yr if year in years_SEAS5] ,lead,0:Nmembers]
                        if anomaly:
                            if anomaly_type == 'linear':
                                var_in = np.nanmean(SEAS5_Ta_mean_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in clim_years_SEAS5 if year != test_yr],lead,0:Nmembers],axis=1)
                                [m,b],_ = linear_fit(clim_years_SEAS5[clim_years_SEAS5 != test_yr], var_in)
                                for mnb in range(Nmembers):
                                    vf_train[:,mnb] = vf_train[:,mnb]-(m*train_yrs + b)
                                vf_test = vf_test-(m*test_yr + b)
                            if anomaly_type == 'mean':
                                vf_clim = np.nanmean(SEAS5_Ta_mean_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in clim_years_SEAS5 if year != test_yr],lead,0:Nmembers])
                                vf_train = vf_train - vf_clim
                                vf_test = vf_test - vf_clim

                        Xtrain[[np.where(train_yrs == year)[0][0] for year in train_yrs if year in years_SEAS5],i] = np.nanmean(vf_train, axis=1)
                        Xtest[:,i,:] = vf_test

                    else:
                        if verbose: print('Using SEAS5.1 frcst for', p)
                        vf_train = SEAS51_Ta_mean_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in train_yrs if year in years_SEAS5] ,lead,0:Nmembers]
                        vf_test = SEAS51_Ta_mean_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in test_yr if year in years_SEAS5] ,lead,0:Nmembers]
                        if anomaly:
                            if anomaly_type == 'linear':
                                var_in = np.nanmean(SEAS51_Ta_mean_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in clim_years_SEAS51 if year != test_yr],lead,0:Nmembers],axis=1)
                                [m,b],_ = linear_fit(clim_years_SEAS51[clim_years_SEAS51 != test_yr], var_in)
                                for mnb in range(Nmembers):
                                    vf_train[:,mnb] = vf_train[:,mnb]-(m*train_yrs + b)
                                vf_test = vf_test-(m*test_yr + b)
                            if anomaly_type == 'mean':
                                vf_clim = np.nanmean(SEAS51_Ta_mean_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in clim_years_SEAS51 if year != test_yr],lead,0:Nmembers])
                                vf_train = vf_train - vf_clim
                                vf_test = vf_test - vf_clim

                        Xtrain[[np.where(train_yrs == year)[0][0] for year in train_yrs if year in years_SEAS5],i] = np.nanmean(vf_train, axis=1)
                        Xtest[:,i,:] = vf_test

                elif (p.split()[-1] == 'snowfall') :
                    if test_yr < 2022:
                        if verbose: print('Using SEAS5 frcst for', p)
                        vf_train = SEAS5_snowfall_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in train_yrs if year in years_SEAS5] ,lead,0:Nmembers]
                        vf_test = SEAS5_snowfall_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in test_yr if year in years_SEAS5] ,lead,0:Nmembers]
                        if anomaly:
                            if anomaly_type == 'linear':
                                var_in = np.nanmean(SEAS5_snowfall_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in clim_years_SEAS5 if year != test_yr],lead,0:Nmembers],axis=1)
                                [m,b],_ = linear_fit(clim_years_SEAS5[clim_years_SEAS5 != test_yr], var_in)
                                for mnb in range(Nmembers):
                                    vf_train[:,mnb] = vf_train[:,mnb]-(m*train_yrs + b)
                                vf_test = vf_test-(m*test_yr + b)
                            if anomaly_type == 'mean':
                                vf_clim = np.nanmean(SEAS5_snowfall_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in clim_years_SEAS5 if year != test_yr],lead,0:Nmembers])
                                vf_train = vf_train - vf_clim
                                vf_test = vf_test - vf_clim

                        Xtrain[[np.where(train_yrs == year)[0][0] for year in train_yrs if year in years_SEAS5],i] = np.nanmean(vf_train, axis=1)
                        Xtest[:,i,:] = vf_test
                    else:
                        if verbose: print('Using SEAS5.1 frcst for', p)
                        vf_train = SEAS51_snowfall_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in train_yrs if year in years_SEAS5] ,lead,0:Nmembers]
                        vf_test = SEAS51_snowfall_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in test_yr if year in years_SEAS5] ,lead,0:Nmembers]
                        if anomaly:
                            if anomaly_type == 'linear':
                                var_in = np.nanmean(SEAS51_snowfall_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in clim_years_SEAS51 if year != test_yr],lead,0:Nmembers],axis=1)
                                [m,b],_ = linear_fit(clim_years_SEAS51[clim_years_SEAS51 != test_yr], var_in)
                                for mnb in range(Nmembers):
                                    vf_train[:,mnb] = vf_train[:,mnb]-(m*train_yrs + b)
                                vf_test = vf_test-(m*test_yr + b)
                            if anomaly_type == 'mean':
                                vf_clim = np.nanmean(SEAS51_snowfall_frcst[m-1,[np.where(years_SEAS5 == year)[0][0] for year in clim_years_SEAS51 if year != test_yr],lead,0:Nmembers])
                                vf_train = vf_train - vf_clim
                                vf_test = vf_test - vf_clim

                        Xtrain[[np.where(train_yrs == year)[0][0] for year in train_yrs if year in years_SEAS5],i] = np.nanmean(vf_train, axis=1)
                        Xtest[:,i,:] = vf_test

                else:
                    raise Exception('Error: SEAS5 predictor variable does not exist.')

            else:
                v_train, v_test = get_df_data()
                Xtrain[[np.where(train_yrs == year)[0][0] for year in train_yrs if year in years_pred],i] = v_train
                Xtest[:,i,:] = np.vstack([v_test]*Nmembers).T

        return Xtrain, Xtest


    # Extract SEAS5 predictors
    SEAS5_Ta_mean_frcst = SEAS5_pred.get('SEAS5_Ta_mean')
    SEAS51_Ta_mean_frcst = SEAS51_pred.get('SEAS51_Ta_mean')
    SEAS5_snowfall_frcst = SEAS5_pred.get('SEAS5_snowfall')
    SEAS51_snowfall_frcst = SEAS51_pred.get('SEAS51_snowfall')
    clim_years_SEAS5 = np.arange(1993,2017)
    clim_years_SEAS51 = np.arange(1981,2017)

    # Prepare variables to hold model predictions
    Nmembers = 25
    X_test = np.zeros((len(years_loo),len(model_pred),Nmembers))*np.nan
    y_pred_test = np.zeros((len(years_loo),Nmembers))
    y_plot = np.zeros((len(years_loo)))

    # Sample predictand on LOO years only
    y_loo = y_in.iloc[[np.where(years_y == year)[0][0] for year in years_loo]]

    # Prepare plot for regression coefficients if needed
    if show_coeff:
        fig_c,ax_c = plt.subplots(nrows = int(np.ceil((len(years_loo)/4))), ncols = 4,sharex=True,sharey=True)

    # Begin LOO model evaluation:
    loo = LeaveOneOut()

    for i,[train_index, test_index] in enumerate(loo.split(years_loo)):
        if verbose:
            print('------------')
            print('TEST YEAR = ', years_loo[i])

        # Get X/y train/test:
        test_year = years_loo[test_index]
        train_years = years_loo[train_index]

        X_train, X_test[i:i+1,:,:] = get_X_train_test(train_years,test_year)
        y_train, y_test = y_loo.iloc[train_index], y_loo.iloc[test_index]
        if (int(np.sum(np.isnan(X_train))+np.sum(np.isnan(y_train))+np.sum(np.isnan(X_test[i:i+1,:,:]))+np.sum(np.isnan(y_test))) != 0):
            y_pred_test[i,:] = np.nan
            continue

        # print('X shape (train, test): ',X_train.shape, X_test[i:i+1,:,:].shape)
        # print(X_test[:,:,3].shape,X_test[0:1,:,3].shape,X_test[0,:,3].shape, X_test[0,:,3].reshape(1,-1).shape)
        # Standardize predictors:
        Xscaler = StandardScaler()
        yscaler = StandardScaler()
        Xscaler = Xscaler.fit(X_train)
        yscaler = yscaler.fit(y_train)
        X_train_scaled = Xscaler.transform(X_train)
        X_test_scaled = X_test.copy()
        for im in range(Nmembers):
            X_test_scaled[i:i+1,:,im] = Xscaler.transform(X_test[i:i+1,:,im])

        # Get Prediction:
        if normalize:
            X_train_fit = X_train_scaled
            X_test_fit = X_test_scaled[i:i+1,:,:]
        else:
            X_train_fit = X_train
            X_test_fit = X_test[i:i+1,:,:]
        for im in range(Nmembers):
            y_pred_test[i,im] = model.fit(X_train_fit,y_train).predict(X_test_fit[:,:,im])
        y_plot[i] = y_test.values

        if not np.all(model.coef_ == 0):
            if verbose:
                print('Reg. coeff.: ', model.coef_[model.coef_ != 0])
                print('Stand. coeff.: ', model.coef_[model.coef_ != 0]*(np.nanstd(X_train_scaled,axis=0)/np.nanstd(y_train)))

            if model_type == 'lasso':
                x = np.where(model.coef_)[0]
            else:
                x = np.arange((len(model_pred)))

            if show_coeff:
                if i/len(years_loo) < 0.25:
                    ax_c[i,0].stem(
                    x,
                    model.coef_[model.coef_ != 0],
                    markerfmt="x")
                elif i/len(years_loo) < 0.5:
                    ax_c[i-int(np.ceil((len(years_loo)/4))),1].stem(
                    x,
                    model.coef_[model.coef_ != 0],
                    markerfmt="x")
                elif i/len(years_loo) < 0.75:
                    ax_c[i-2*int(np.ceil((len(years_loo)/4))),2].stem(
                    x,
                    model.coef_[model.coef_ != 0],
                    markerfmt="x")
                else:
                    ax_c[i-3*int(np.ceil((len(years_loo)/4))),3].stem(
                    x,
                    model.coef_[model.coef_ != 0],
                    markerfmt="x")

    if plot:
        plt.figure()
        plt.plot(years_loo,y_plot,'-o',color='k')
        plt.plot(years_loo,y_pred_test,'-o',color=plt.get_cmap('tab20c')(0))

    # mae = np.nanmean(np.abs(y_plot[:]-y_pred_test[:]))
    # acc7days = np.sum(np.abs(y_plot[:]-y_pred_test[:]) <= 7)/(np.sum(~np.isnan(y_loo)))

    # if verbose:
    #     print('MAE:', mae)
    #     print('7-day accuracy: ', float(acc7days))


    # And finally, fit model with all data:
    # X_train = X_in
    # y_train = y_loo
    # # Xscaler = MinMaxScaler()
    # Xscaler = StandardScaler()
    # Xscaler = Xscaler.fit(X_train)
    # X_train_scaled = Xscaler.transform(X_train)

    # if normalize:
    #     X_train_fit = X_train_scaled
    # else:
    #     X_train_fit = X_train
    # # X_train_fit = X_train_scaled
    # model = model.fit(X_train_fit,y_train)

    return y_pred_test, X_test#, model, Xscaler, mae, acc7days


#%%
if __name__ == "__main__":

    date_ref = dt.date(1900,1,1)
    date_start = dt.date(1980,1,1)
    date_end = dt.date(2021,12,31)
    ndays = (date_end-date_start).days + 1
    time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

    years = np.array([1991,1992,1993,1994,1995,1996,
              1997,1998,1999,2000,2001,
              2002,2003,2004,2005,2006,
              2007,2008,2009,2010,2011,
              2012,2013,2014,2015,2016,2017,
              2018,2019,2020,2021])

    ignore_warnings = True
    if ignore_warnings:
        import warnings
        warnings.filterwarnings("ignore")


    # -------------------
    # OPTIONS

    # Choose start and end test years for the LOO-k evaluation
    start_yr = 1992
    end_yr = 2019

    train_yr_start = start_yr # [1992 - 2007] = 16 years
    valid_yr_start = 2008 # [2008 - 2013] = 6 years
    test_yr_start = 2014  # [2014 - 2019] = 6 years
    nsplits = end_yr-valid_yr_start+1


    # Choose how many k-folds for cross-validation
    nfolds = 5


    # Choose Forecast start date:
        # 11: November
        # 12: December
    start_month = 11
    # start_month = 12

    # Choose if using anomalies to make forecasts:
    # anomaly = True
    anomaly = False

    # Choose if normalizaing the regression coefficients:
    norm_MLR = True
    # norm_MLR = False

    # Choose FUD definition
    freezeup_opt = 1

    # Plot options
    verbose = False
    plot_reliability_diagram = False
    savefig = False



    #%% Load Twater and FUD data
    fp_p_Twater = local_path+'slice/data/processed/'
    Twater_loc_list = ['Longueuil_updated']
    station_type = 'cities'
    freezeup_doy, Twater = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False)
    freezeup_doy[np.where(years > end_yr)] = np.nan

    Twater_mean = np.nanmean(Twater,axis=1)
    Twater_mean = np.expand_dims(Twater_mean, axis=1)
    Twater_mean[14269:14329] = 0.

    # Average (and round) FUD from all locations:
    avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
    avg_freezeup_doy = np.round(avg_freezeup_doy)

    # Get FUD categories for accuracy measure:
    it_1992 = np.where(years == 1992)[0][0]
    it_2008= np.where(years == 2008)[0][0]
    mean_FUD = np.nanmean(avg_freezeup_doy[it_1992:it_2008])
    std_FUD = np.nanstd(avg_freezeup_doy[it_1992:it_2008])
    tercile1_FUD = np.nanpercentile(avg_freezeup_doy[it_1992:it_2008],(1/3.)*100)
    tercile2_FUD = np.nanpercentile(avg_freezeup_doy[it_1992:it_2008],(2/3.)*100)

    #%% Load all monthly predictor data
    fpath_mp = local_path+'slice/data/monthly_predictors/'
    monthly_pred = np.load(fpath_mp+'monthly_vars'+'_'+'Longueuil'+'.npz')
    monthly_pred_data = monthly_pred['data']
    pred_names = monthly_pred['labels']

    # Keep only starr_yr to end_yr (inclusive)
    it_start = np.where(years == start_yr)[0][0]
    it_end = np.where(years == end_yr)[0][0]

    years = years[it_start:it_end+1]
    avg_freezeup_doy = avg_freezeup_doy[it_start:it_end+1]
    monthly_pred_data = monthly_pred_data[:,it_start:it_end+1,:]

    p_list = ['Avg. cloud cover','Avg. cloud cover','Avg. cloud cover','Avg. cloud cover','Avg. cloud cover','Avg. level Ottawa River','Avg. level Ottawa River','Avg. level Ottawa River','Avg. level Ottawa River','Avg. level Ottawa River','NAO','NAO','NAO','NAO','NAO','PDO','PDO','PDO','PDO','PDO','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. Ta_mean','Avg. Ta_mean','Avg. Ta_mean','Avg. Ta_mean','Avg. Ta_mean','Tot. snowfall','Tot. snowfall','Tot. snowfall', 'Avg. windspeed', 'Avg. windspeed','Avg. windspeed','Avg. windspeed','Avg. windspeed','Avg. SW down (sfc)','Avg. SW down (sfc)','Avg. SW down (sfc)','Avg. SW down (sfc)','Avg. SW down (sfc)', 'Avg. LH (sfc)', 'Avg. LH (sfc)','Avg. LH (sfc)','Avg. LH (sfc)','Avg. LH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)' , 'Avg. SLP', 'Avg. SLP', 'Avg. SLP', 'Avg. SLP', 'Avg. SLP']
    m_list = [12,                11,                 10,               9,                8,                   12,                       11,                       10,                      9,                        8,                        12,   11,    10,     9,  8,    12,    11,   10,    9, 8,     12,                         11,                           10,                         9,                           8,                            12,           11,            10,           9,              8,            12,               11,              10,             8,                9,                10,             11,             12,              8,                   9 ,                  10,                  11,                  12,                   8,                9,              10,           11,              12,              8,              9,              10,            11,             12,               8,           9,          10,         11,        12  ]

    # Make dataframe with all predictors
    month_str = ['Jan. ', 'Feb. ', 'Mar. ', 'Apr. ', 'May ', 'Jun. ','Jul. ', 'Aug. ', 'Sep. ', 'Oct. ', 'Nov. ','Dec. ']
    pred_arr = np.zeros((monthly_pred_data.shape[1],len(p_list)))*np.nan
    col = []
    for i in range(len(p_list)):
        ipred = np.where(pred_names == p_list[i])[0][0]
        pred_arr[:,i] = monthly_pred_data[ipred,:,m_list[i]-1]
        col.append(month_str[m_list[i]-1]+p_list[i])
    pred_df =  pd.DataFrame(pred_arr,columns=col)


    #%% Load or compute/save SEAS5 forecasts
    # Computing the SEAS5 monthly predictors takes a while. It it preferred to do it only once and then load the saved file.
    load = True
    # load = False
    filename = 'SEAS5_SEAS51_forecast_monthly_predictors'

    if load:
        dtmp = np.load(local_path+'slice/data/processed/SEAS5/'+filename+'.npz', allow_pickle = True)
        SEAS5_Ta_mean = dtmp['SEAS5_Ta_mean']
        SEAS5_snowfall = dtmp['SEAS5_snowfall']
        SEAS51_Ta_mean = dtmp['SEAS51_Ta_mean']
        SEAS51_snowfall = dtmp['SEAS51_snowfall']
        SEAS5_years = np.arange(1981,2022+1)
    else:
        [SEAS5_Ta_mean,SEAS51_Ta_mean,SEAS5_snowfall,SEAS51_snowfall] = save_SEAS5_predictor_arrays(local_path,time_rep='monthly',region='D')

#%%

# model_pred = ['Dec. Avg. Ta_mean']
# model_pred = ['Dec. Tot. snowfall']
model_pred = ['Dec. Avg. Ta_mean','Nov. Tot. snowfall','Sep. Avg. cloud cover']


# model_pred = ['Nov. Tot. snowfall']
# model_pred = ['Dec. Avg. Ta_mean','Nov. Tot. snowfall','Sep. Avg. SW down (sfc)']
# model_pred = ['Sep. Avg. level Ottawa River','Dec. Avg. Ta_mean','Sep. Avg. SW down (sfc)','Oct. Avg. SH (sfc)']
# model_pred = ['Sep. Avg. level Ottawa River','Dec. Avg. Ta_mean','Nov. Avg. Ta_mean','Sep. Avg. SW down (sfc)','Oct. Avg. SH (sfc)']
# model_pred = ['Dec. Avg. Ta_mean','Nov. Tot. snowfall','Dec. Tot. snowfall']
SEAS5_pred={'SEAS5_Ta_mean':SEAS5_Ta_mean, 'SEAS5_snowfall':SEAS5_snowfall}
SEAS51_pred={'SEAS51_Ta_mean':SEAS51_Ta_mean, 'SEAS51_snowfall':SEAS51_snowfall}
years_loo = np.arange(1993,years[-1]+1)

verbose = False
atype = 'linear'

FUD_arr = pd.DataFrame(avg_freezeup_doy,columns=['Avg. FUD DOY'])

print('Processing ensemble mean')
pred_em = loo_eval_new(y_in=FUD_arr,
                      frcst_start_month=start_month,
                      model_pred=model_pred,
                      df_in=pred_df,
                      SEAS5_pred=SEAS5_pred,
                      SEAS51_pred=SEAS51_pred,
                      years_loo=years_loo,
                      years_y=years,
                      years_SEAS5=SEAS5_years,
                      years_pred=years,
                      model=LinearRegression(),
                      model_type='mlr',
                      em = 'mean',
                      anomaly=anomaly,
                      anomaly_type=atype,
                      normalize=norm_MLR,
                      show_coeff=False,
                      plot=False,
                      verbose=verbose)

r2_em,mae_em,rmse_em = regression_metrics(avg_freezeup_doy[[np.where(years == year)[0][0] for year in years_loo]], pred_em)
print(' ----  Ensemble Mean forecast ----')
print('MAE: ', float(mae_em))
print('RMSE: ', float(rmse_em))
print('Rsqr: ', float(r2_em))

print('Processing all members')
pred_em_all, X_test_all = loo_eval_train_mean_eval_all(y_in=FUD_arr,
                                                      frcst_start_month=start_month,
                                                      model_pred=model_pred,
                                                      df_in=pred_df,
                                                      SEAS5_pred=SEAS5_pred,
                                                      SEAS51_pred=SEAS51_pred,
                                                      years_loo=years_loo,
                                                      years_y=years,
                                                      years_SEAS5=SEAS5_years,
                                                      years_pred=years,
                                                      model=LinearRegression(),
                                                      model_type='mlr',
                                                      em = 'mean',
                                                      anomaly=anomaly,
                                                      anomaly_type=atype,
                                                      normalize=norm_MLR,
                                                      show_coeff=False,
                                                      plot=False,
                                                      verbose=verbose)

f, ax = plt.subplots(figsize=[12,5])
ax.plot(years_loo,avg_freezeup_doy[1:],'-o',color='k', label='Observed FUD')
ax.plot(years_loo,pred_em,'-o',color=plt.get_cmap('tab20c')(8), label='MLR - Ensemble mean forecast')#' (MAE:' +str(round(float(mae_em),2))+', RMSE:'+str(round(float(rmse_em),2))+', Rsqr:'+str(round(float(r2_em),2))+')')
ax.plot(years_loo,np.nanmean(pred_em_all,axis=1),'-o',color=plt.get_cmap('tab20c')(8), label='Ensemble mean MLR forecasts')#' (MAE:' +str(round(float(mae_mean),2))+', RMSE:'+str(round(float(rmse_mean),2))+', Rsqr:'+str(round(float(r2_mean),2))+')')

Nmembers = 25
for im in range(Nmembers):
    if im ==0:
        ax.plot(years_loo,pred_em_all[:,im],'-',color=plt.get_cmap('tab20c')(10), linewidth=0.86, label='MLR - Individual member forecasts')
    else:
        ax.plot(years_loo,pred_em_all[:,im],'-',color=plt.get_cmap('tab20c')(10), linewidth=0.86)
ax.plot(years_loo,avg_freezeup_doy[1:],'-o',color='k')
ax.plot(years_loo,pred_em,'-o',color=plt.get_cmap('tab20c')(8))#' (MAE:' +str(round(float(mae_em),2))+', RMSE:'+str(round(float(rmse_em),2))+', Rsqr:'+str(round(float(r2_em),2))+')')

ax.legend()
ax.set_ylabel('FUD')
ax.set_xlabel('Year')
plt.tight_layout()
# plt.minorticks_on()
ax.grid(linestyle=':', which='major')
ax.grid(linestyle=':', which='minor')

if savefig:
    save_folder = './'
    model_name = 'DecTa_NovSnow_SepClouds'
    f.savefig(save_folder+'MLR_RWE_forecast_timeseries_'+model_name+anomaly*('_anomaly_'+atype+'_')+'startmonth_'+str(start_month)+'.png', dpi=600)

#%%
# print('Training ensemble members individually')
# FUD_ensemble = np.zeros((len(years_loo),SEAS5_Ta_mean.shape[-1]))*np.nan
# pred_mean = np.zeros((len(years_loo)))
# n = 0
# for i in tqdm(range(SEAS5_Ta_mean.shape[-1]), position=0, leave=True):
#     pred = loo_eval_new(y_in=FUD_arr,
#                           frcst_start_month=start_month,
#                           model_pred=model_pred,
#                           df_in=pred_df,
#                           SEAS5_pred=SEAS5_pred,
#                           SEAS51_pred=SEAS51_pred,
#                           years_loo=years_loo,
#                           years_y=years,
#                           years_SEAS5=SEAS5_years,
#                           years_pred=years,
#                           model=LinearRegression(),
#                           model_type='mlr',
#                           em = i,
#                           anomaly=anomaly,
#                           anomaly_type=atype,
#                           normalize=norm_MLR,
#                           show_coeff=False,
#                           plot=False,
#                           verbose=verbose)
#     FUD_ensemble[:,i] = pred
#     if ~np.any(np.isnan(pred)):
#         n += 1
#         pred_mean += pred
#         r2_test,mae_test,rmse_test = regression_metrics(avg_freezeup_doy[[np.where(years == year)[0][0] for year in years_loo]], pred)

#     # if i ==0:
#     #     ax.plot(years_loo,pred,'-',color=plt.get_cmap('tab20c')(10), linewidth=0.6, label='with predictors from individual members')
#     # else:
#     #     ax.plot(years_loo,pred,'-',color=plt.get_cmap('tab20c')(10), linewidth=0.6)

# pred_mean = pred_mean/n
# r2_mean,mae_mean,rmse_mean = regression_metrics(avg_freezeup_doy[[np.where(years == year)[0][0] for year in years_loo]], pred_mean)
# print(' ----  Mean of all member forecasts ----')
# print('MAE: ', float(mae_mean))
# print('RMSE: ', float(rmse_mean))
# print('Rsqr: ', float(r2_mean))

#%% PREDICTOR PLOTS
# f,ax = plt.subplots(nrows = 4, ncols = 1, sharex = True)
# ax[0].plot(years, pred_df['Dec. Avg. Ta_mean'],'*-')
# ax[3].plot(years, pred_df['Sep. Avg. cloud cover'],'*-')
# # ax[0].plot(years, pred_df['Dec. NAO'],'*-')
# ax[1].plot(years, pred_df['Dec. Avg. Ta_mean'],'*-')
# ax[2].plot(years, pred_df['Nov. Tot. snowfall'],'*-')
# # ax[0].plot(years, pred_df['Nov. Avg. SLP'],'*-')
# # ax[1].plot(years, avg_freezeup_doy,'o-')

#%% RELIABILITY DIAGRAMS: 3 CATEGORIES
if plot_reliability_diagram:
    # "Thus, for example, when the forecast states an event will occur
    # with a probability of 25% then for perfect reliability,
    # the event should occur on 25% of occasions on which the
    # statement is made."
    # Source: https://www.metoffice.gov.uk/research/climate/seasonal-to-decadal/gpc-outlooks/user-guide/interpret-reliability

    plot_dist = True

    prob_early = np.zeros((len(years_loo)))*np.nan
    prob_normal = np.zeros((len(years_loo)))*np.nan
    prob_late = np.zeros((len(years_loo)))*np.nan

    obs_early = np.zeros((len(years_loo)))*np.nan
    obs_normal = np.zeros((len(years_loo)))*np.nan
    obs_late = np.zeros((len(years_loo)))*np.nan
    frcst_early = np.zeros((len(years_loo)))*np.nan
    frcst_normal = np.zeros((len(years_loo)))*np.nan
    frcst_late = np.zeros((len(years_loo)))*np.nan

    BS_early = np.zeros((len(years_loo)))*np.nan
    BS_normal = np.zeros((len(years_loo)))*np.nan
    BS_late = np.zeros((len(years_loo)))*np.nan

    for iyr in range(len(years_loo)):
        p33_obs = np.nanpercentile(FUD_arr.iloc[1:,0],33.33)
        p66_obs = np.nanpercentile(FUD_arr.iloc[1:,0],66.66)

        if plot_dist:
            fig, ax = plt.subplots()
            ax.hist(FUD_arr.iloc[1:,0],bins=30,range=(330,390),alpha=0.3,color='gray',density=True)
            ax.hist(pred_em_all[iyr,:],bins=30,range=(330,390),alpha=0.3,density=True)
            ax.plot(np.ones((10))*p33_obs,np.arange(0,10),':',color='black')
            ax.plot(np.ones((10))*p66_obs,np.arange(0,10),':',color='black')
            ax.plot(np.ones((10))*FUD_arr.iloc[1:,0].values[iyr],np.arange(0,10),'-',color='red')
            ax.plot(np.ones((10))*np.nanmean(pred_em_all[iyr,:]),np.arange(0,10),'-',color='orange')
            ax.plot(np.ones((10))*np.nanmean(pred_em[iyr]),np.arange(0,10),'--',color='orange')


        if np.all(np.isnan(pred_em_all[iyr,25:])):
            N_members = 25
        else:
            N_members = 51
        prob_early[iyr] = np.sum(pred_em_all[iyr,0:N_members]<= p33_obs)/N_members
        prob_normal[iyr] = np.sum((pred_em_all[iyr,0:N_members] > p33_obs) & (pred_em_all[iyr,0:N_members]<= p66_obs))/N_members
        prob_late[iyr] = np.sum(pred_em_all[iyr,0:N_members]> p66_obs)/N_members

        obs_early[iyr] = np.sum(FUD_arr.iloc[1:,0].values[iyr]<= p33_obs)
        obs_normal[iyr] = np.sum((FUD_arr.iloc[1:,0].values[iyr] > p33_obs) & (FUD_arr.iloc[1:,0].values[iyr]<= p66_obs))
        obs_late[iyr] = np.sum(FUD_arr.iloc[1:,0].values[iyr]> p66_obs)

        BS_early[iyr] = (prob_early[iyr] - obs_early[iyr])**2.
        BS_normal[iyr] = (prob_normal[iyr] - obs_normal[iyr])**2.
        BS_late[iyr] = (prob_late[iyr] - obs_late[iyr])**2.

        if (prob_early[iyr] > prob_normal[iyr]) & (prob_early[iyr] > prob_late[iyr]):
            frcst_early[iyr] = 1
        elif (prob_normal[iyr] > prob_early[iyr]) & (prob_normal[iyr] > prob_late[iyr]):
            frcst_normal[iyr] = 1
        elif  (prob_late[iyr] > prob_early[iyr]) & (prob_late[iyr] > prob_normal[iyr]):
            frcst_late[iyr] = 1
        else:
            print("Ambiguous forecast!!!", years_loo[iyr], prob_early[iyr], prob_normal[iyr],prob_late[iyr])

        if plot_dist:
            ax.text(335,0.18,str(np.round(prob_early[iyr],2))+'%')
            ax.text(355,0.18,str(np.round(prob_normal[iyr],2))+'%')
            ax.text(375,0.18,str(np.round(prob_late[iyr],2))+'%')
            ax.text(375,0.08,str(iyr)+'YEAr')
            ax.set_ylim(0,0.25)
            ax.set_xlim(325,395)


    d = np.concatenate((np.expand_dims(obs_early,axis=1),np.expand_dims(obs_normal,axis=1),np.expand_dims(obs_late,axis=1)),axis=1)
    r = np.concatenate((np.expand_dims(prob_early,axis=1),np.expand_dims(prob_normal,axis=1),np.expand_dims(prob_late,axis=1)),axis=1)

    BS_early = np.nanmean(BS_early.copy())
    BS_normal = np.nanmean(BS_normal.copy())
    BS_late = np.nanmean(BS_late.copy())

    print(BS_early)
    print(BS_normal)
    print(BS_late)
    print(np.mean((r-d)**2.,axis=0))

    nbins = 3
    bin_lims = [np.round(i*(1/nbins),2) for i in range(nbins+1)]

    if plot_dist:
        fig,ax = plt.subplots(figsize=(4,5))
        ax.hist(prob_early,bins=bin_lims, label= 'Early',color='pink')
        # n,b,_=ax.hist(prob_early,bins=5,range=(0,1), label= 'Early',color='pink')
        ax.legend()
        ax.set_xlabel('Forecast probability')
        ax.set_ylabel('Count')
        ax.set_ylim(0,20)
        ax.set_yticks(np.arange(0,20,2))

        fig,ax = plt.subplots(figsize=(4,5))
        ax.hist(prob_normal,bins=bin_lims, label= 'Normal',color='green')
        ax.legend()
        ax.set_xlabel('Forecast probability')
        ax.set_ylabel('Count')
        ax.set_ylim(0,20)
        ax.set_yticks(np.arange(0,20,2))

        fig,ax = plt.subplots(figsize=(4,5))
        ax.hist(prob_late,bins=bin_lims, label= 'Late',color='blue')
        ax.legend()
        ax.set_xlabel('Forecast probability')
        ax.set_ylabel('Count')
        ax.set_ylim(0,20)
        ax.set_yticks(np.arange(0,20,2))


    freq_early = np.zeros((nbins))*np.nan
    freq_normal = np.zeros((nbins))*np.nan
    freq_late = np.zeros((nbins))*np.nan
    n_early = np.zeros((nbins))*np.nan
    n_normal = np.zeros((nbins))*np.nan
    n_late = np.zeros((nbins))*np.nan
    bin_center = np.zeros((nbins))*np.nan

    for i in range(nbins):
        bin_center[i] = 0.5*(bin_lims[i]+bin_lims[i+1])

        if i == nbins-1:
            freq_early[i] = np.sum(obs_early[(prob_early <= (bin_lims[i+1])) & (prob_early >= bin_lims[i])])/np.nansum((prob_early <= (bin_lims[i+1])) & (prob_early >= bin_lims[i]))
            freq_normal[i] = np.sum(obs_normal[(prob_normal <= (bin_lims[i+1])) & (prob_normal >= bin_lims[i])])/np.nansum((prob_normal <= (bin_lims[i+1])) & (prob_normal >= bin_lims[i]))
            freq_late[i] = np.sum(obs_late[(prob_late <= (bin_lims[i+1])) & (prob_late >= bin_lims[i])])/np.nansum((prob_late <= (bin_lims[i+1])) & (prob_late >= bin_lims[i]))

            n_early[i] = np.nansum((prob_early <= bin_lims[i+1]) & (prob_early >= bin_lims[i]))
            n_normal[i]= np.nansum((prob_normal <= bin_lims[i+1]) & (prob_normal >= bin_lims[i]))
            n_late[i]= np.nansum((prob_late <= bin_lims[i+1]) & (prob_late >= bin_lims[i]))
            # print(np.sum(obs_early[(prob_early <= (bin_lims[i+1])) & (prob_early >= bin_lims[i])]))
        else:
            freq_early[i] = np.sum(obs_early[(prob_early < (bin_lims[i+1])) & (prob_early >= bin_lims[i])])/np.nansum((prob_early < (bin_lims[i+1])) & (prob_early >= bin_lims[i]))
            freq_normal[i] = np.sum(obs_normal[(prob_normal < (bin_lims[i+1])) & (prob_normal >= bin_lims[i])])/np.nansum((prob_normal < (bin_lims[i+1])) & (prob_normal >= bin_lims[i]))
            freq_late[i] = np.sum(obs_late[(prob_late < (bin_lims[i+1])) & (prob_late >= bin_lims[i])])/np.nansum((prob_late < (bin_lims[i+1])) & (prob_late >= bin_lims[i]))

            n_early[i] = np.nansum((prob_early < bin_lims[i+1]) & (prob_early >= bin_lims[i]))
            n_normal[i]= np.nansum((prob_normal < bin_lims[i+1]) & (prob_normal >= bin_lims[i]))
            n_late[i]= np.nansum((prob_late < bin_lims[i+1]) & (prob_late >= bin_lims[i]))
            # print(np.sum(obs_early[(prob_early < (bin_lims[i+1])) & (prob_early >= bin_lims[i])]))


    fig,ax = plt.subplots()
    ax.plot(bin_center,freq_early,'-',color='pink')
    ax.plot(bin_center,freq_normal,'-',color='green')
    ax.plot(bin_center,freq_late,'-',color='blue')
    ax.scatter(bin_center,freq_normal,s=n_normal**2,color='green')
    ax.scatter(bin_center,freq_late,s=n_late**2,color='blue')
    ax.scatter(bin_center,freq_early,s=n_early**2,color='pink')
    ax.plot(np.arange(0,10,1),np.arange(0,10,1),':',color='black')
    ax.set_xlim(-0.05,1.05)
    ax.set_ylim(-0.05,1.05)
    ax.set_ylabel('Observed frequency')
    ax.set_xlabel('Forecast probability')
    ax.grid()

    # rk = np.zeros(r.shape)*np.nan
    # for s in range(nbins):
    #     for n in range(3):
    #         for k in range(len(years_loo)):
    #             if s == (nbins-1):
    #                 if (r[k,n] >= bin_lims[s]) & (r[k,n] <= bin_lims[s+1]) :
    #                     # rk[k,n] = bin_center[s]
    #                     rk[k,n] = bin_lims[s]
    #             else:
    #                 if (r[k,n] >= bin_lims[s]) & (r[k,n] < bin_lims[s+1]) :
    #                     # rk[k,n] = bin_center[s]
    #                     rk[k,n] = bin_lims[s]
    # rt = np.unique(rk.copy(), axis=0)

    # Kt = np.zeros((rt.shape[0]))
    # for t in range(rt.shape[0]):
    #     for k in range(len(years_loo)):
    #         if np.all(rk[k] == rt[t]): Kt[t] += 1


    # dt_bar = np.zeros((rt.shape[0],d.shape[1]))
    # for t in range(rt.shape[0]):
    #     for k in range(len(years_loo)):
    #         if np.all(rk[k] == rt[t]):
    #             dt_bar[t,:] += d[k]/Kt[t]

    # u = np.ones(rt.shape)

    # dbar = np.zeros((rt.shape[0],d.shape[1]))
    # for t in range(rt.shape[0]):
    #     dbar[t,:] = (1/len(years_loo))*np.sum(np.expand_dims(Kt,axis=1)*dt_bar,axis=0)

    # BS_unc = np.nanmean(dbar*(u-dbar),axis=0)
    # BS_rel = (1/len(years_loo))*np.sum((np.expand_dims(Kt,axis=1)*(rt-dt_bar)**2.),axis=0)
    # BS_res = (1/len(years_loo))*np.sum((np.expand_dims(Kt,axis=1)*(dt_bar-dbar)**2.),axis=0)
    # BS = BS_unc + BS_rel - BS_res

    # print(BS)
    # print(BS_early, BS_normal, BS_late)

#%% RELIABILITY DIAGRAMS: 2 CATEGORIES
if plot_reliability_diagram:
    # "Thus, for example, when the forecast states an event will occur
    # with a probability of 25% then for perfect reliability,
    # the event should occur on 25% of occasions on which the
    # statement is made."
    # Source: https://www.metoffice.gov.uk/research/climate/seasonal-to-decadal/gpc-outlooks/user-guide/interpret-reliability

    # FUD_ensemble = pred_em_all.copy()
    plot_dist = False

    prob_early = np.zeros((len(years_loo)))*np.nan
    prob_late = np.zeros((len(years_loo)))*np.nan

    obs_early = np.zeros((len(years_loo)))*np.nan
    obs_late = np.zeros((len(years_loo)))*np.nan
    frcst_early = np.zeros((len(years_loo)))*np.nan
    frcst_late = np.zeros((len(years_loo)))*np.nan

    BS_early = np.zeros((len(years_loo)))*np.nan
    BS_late = np.zeros((len(years_loo)))*np.nan

    for iyr in range(len(years_loo)):
        p50_obs = np.nanpercentile(FUD_arr.iloc[1:,0],50)

        if plot_dist:
            fig, ax = plt.subplots()
            ax.hist(FUD_arr.iloc[1:,0],bins=30,range=(330,390),alpha=0.3,color='gray',density=True)
            ax.hist(pred_em_all[iyr,:],bins=30,range=(330,390),alpha=0.3,density=True)
            ax.plot(np.ones((10))*p50_obs,np.arange(0,10),':',color='black')

        if np.all(np.isnan(pred_em_all[iyr,25:])):
            N_members = 25
        else:
            N_members = 51
        prob_early[iyr] = np.sum(pred_em_all[iyr,0:N_members]<= p50_obs)/N_members
        prob_late[iyr] = np.sum(pred_em_all[iyr,0:N_members]> p50_obs)/N_members

        obs_early[iyr] = np.sum(FUD_arr.iloc[1:,0].values[iyr]<= p50_obs)
        obs_late[iyr] = np.sum(FUD_arr.iloc[1:,0].values[iyr]> p50_obs)

        BS_early[iyr] = (prob_early[iyr] - obs_early[iyr])**2.
        BS_late[iyr] = (prob_late[iyr] - obs_late[iyr])**2.

        if (prob_early[iyr] > prob_late[iyr]):
            frcst_early[iyr] = 1
        elif  (prob_late[iyr] > prob_early[iyr]):
            frcst_late[iyr] = 1
        else:
            print("Ambiguous forecast!!!", years_loo[iyr], prob_early[iyr], prob_normal[iyr],prob_late[iyr])

        if plot_dist:
            ax.text(335,0.18,str(np.round(prob_early[iyr],2))+'%')
            ax.text(375,0.18,str(np.round(prob_late[iyr],2))+'%')
            ax.text(375,0.08,str(iyr)+'YEAr')
            ax.set_ylim(0,0.25)
            ax.set_xlim(325,395)

    d = np.concatenate((np.expand_dims(obs_early,axis=1),np.expand_dims(obs_late,axis=1)),axis=1)
    r = np.concatenate((np.expand_dims(prob_early,axis=1),np.expand_dims(prob_late,axis=1)),axis=1)

    BS_early = np.nanmean(BS_early.copy())
    BS_late = np.nanmean(BS_late.copy())

    print(BS_early)
    print(BS_late)
    print(np.mean((r-d)**2.,axis=0))

    nbins = 3
    bin_lims = [np.round(i*(1/nbins),2) for i in range(nbins+1)]

    if plot_dist:
        fig,ax = plt.subplots(figsize=(4,5))
        ax.hist(prob_early,bins=bin_lims, label= 'Early',color='pink')
        # n,b,_=ax.hist(prob_early,bins=5,range=(0,1), label= 'Early',color='pink')
        ax.legend()
        ax.set_xlabel('Forecast probability')
        ax.set_ylabel('Count')
        ax.set_ylim(0,20)
        ax.set_yticks(np.arange(0,20,2))

        fig,ax = plt.subplots(figsize=(4,5))
        ax.hist(prob_late,bins=bin_lims, label= 'Late',color='blue')
        ax.legend()
        ax.set_xlabel('Forecast probability')
        ax.set_ylabel('Count')
        ax.set_ylim(0,20)
        ax.set_yticks(np.arange(0,20,2))


    freq_early = np.zeros((nbins))*np.nan
    freq_late = np.zeros((nbins))*np.nan
    n_early = np.zeros((nbins))*np.nan
    n_late = np.zeros((nbins))*np.nan
    bin_center = np.zeros((nbins))*np.nan
    for i in range(nbins):
        bin_center[i] = 0.5*(bin_lims[i]+bin_lims[i+1])

        if i == nbins-1:
            freq_early[i] = np.sum(obs_early[(prob_early <= (bin_lims[i+1])) & (prob_early >= bin_lims[i])])/np.nansum((prob_early <= (bin_lims[i+1])) & (prob_early >= bin_lims[i]))
            freq_late[i] = np.sum(obs_late[(prob_late <= (bin_lims[i+1])) & (prob_late >= bin_lims[i])])/np.nansum((prob_late <= (bin_lims[i+1])) & (prob_late >= bin_lims[i]))

            n_early[i] = np.nansum((prob_early <= bin_lims[i+1]) & (prob_early >= bin_lims[i]))
            n_late[i]= np.nansum((prob_late <= bin_lims[i+1]) & (prob_late >= bin_lims[i]))
            # print(np.sum(obs_early[(prob_early <= (bin_lims[i+1])) & (prob_early >= bin_lims[i])]))
        else:
            freq_early[i] = np.sum(obs_early[(prob_early < (bin_lims[i+1])) & (prob_early >= bin_lims[i])])/np.nansum((prob_early < (bin_lims[i+1])) & (prob_early >= bin_lims[i]))
            freq_late[i] = np.sum(obs_late[(prob_late < (bin_lims[i+1])) & (prob_late >= bin_lims[i])])/np.nansum((prob_late < (bin_lims[i+1])) & (prob_late >= bin_lims[i]))

            n_early[i] = np.nansum((prob_early < bin_lims[i+1]) & (prob_early >= bin_lims[i]))
            n_late[i]= np.nansum((prob_late < bin_lims[i+1]) & (prob_late >= bin_lims[i]))
            # print(np.sum(obs_early[(prob_early < (bin_lims[i+1])) & (prob_early >= bin_lims[i])]))

    fig,ax = plt.subplots()
    ax.plot(bin_center,freq_early,'-',color='pink')
    ax.plot(bin_center,freq_late,'-',color='blue')
    ax.scatter(bin_center,freq_late,s=n_late**2,color='blue')
    ax.scatter(bin_center,freq_early,s=n_early**2,color='pink')
    ax.plot(np.arange(0,10,1),np.arange(0,10,1),':',color='black')
    ax.set_xlim(-0.05,1.05)
    ax.set_ylim(-0.05,1.05)
    ax.set_ylabel('Observed frequency')
    ax.set_xlabel('Forecast probability')
    ax.grid()
#%% PROBABILISTIC CATEGORICAL FORECAST VALUES
prob_early = np.zeros((len(years_loo)))*np.nan
prob_normal = np.zeros((len(years_loo)))*np.nan
prob_late = np.zeros((len(years_loo)))*np.nan

obs_early = np.zeros((len(years_loo)))*np.nan
obs_normal = np.zeros((len(years_loo)))*np.nan
obs_late = np.zeros((len(years_loo)))*np.nan

N_members = 25
for iyr in range(len(years_loo)):
    p33_obs = np.nanpercentile(FUD_arr.iloc[1:,0],33.33)
    p66_obs = np.nanpercentile(FUD_arr.iloc[1:,0],66.66)

    prob_early[iyr] = np.sum(pred_em_all[iyr,0:N_members]<= p33_obs)/N_members
    prob_normal[iyr] = np.sum((pred_em_all[iyr,0:N_members] > p33_obs) & (pred_em_all[iyr,0:N_members]<= p66_obs))/N_members
    prob_late[iyr] = np.sum(pred_em_all[iyr,0:N_members]> p66_obs)/N_members

    obs_early[iyr] = np.sum(FUD_arr.iloc[1:,0].values[iyr]<= p33_obs)
    obs_normal[iyr] = np.sum((FUD_arr.iloc[1:,0].values[iyr] > p33_obs) & (FUD_arr.iloc[1:,0].values[iyr]<= p66_obs))
    obs_late[iyr] = np.sum(FUD_arr.iloc[1:,0].values[iyr]> p66_obs)

d = np.concatenate((np.expand_dims(obs_early,axis=1),np.expand_dims(obs_normal,axis=1),np.expand_dims(obs_late,axis=1)),axis=1)
r = np.concatenate((np.expand_dims(prob_early,axis=1),np.expand_dims(prob_normal,axis=1),np.expand_dims(prob_late,axis=1)),axis=1)

cat_obs = np.zeros(len(years_loo))*np.nan
cat_mlr_probabilistic = np.zeros(len(years_loo))*np.nan

for iyr in range(len(years_loo)):
    if np.where(r[iyr] == np.max(r[iyr]))[0][0] == 0:
        cat_mlr_probabilistic[iyr] = -1
    elif np.where(r[iyr] == np.max(r[iyr]))[0][0] == 2:
        cat_mlr_probabilistic[iyr] = 1
    else:
        cat_mlr_probabilistic[iyr] = 0

    if np.where(d[iyr] == np.max(d[iyr]))[0][0] == 0:
        cat_obs[iyr] = -1
    elif np.where(d[iyr] == np.max(d[iyr]))[0][0] == 2:
        cat_obs[iyr] = 1
    else:
        cat_obs[iyr] = 0

cat_mlr = np.zeros(len(years_loo))*np.nan
FUD_obs = FUD_arr.values[1:,0]
for iyr,year in enumerate(years_loo):
    if pred_em[iyr] <= p33_obs:
        cat_mlr[iyr] = -1
    elif pred_em[iyr] > p66_obs:
        cat_mlr[iyr] = 1
    else:
        cat_mlr[iyr] = 0
mlr_acc = (np.sum(cat_mlr == cat_obs)/(np.sum(~np.isnan(FUD_obs))))*100

# ax.fill_between(years_loo,np.ones(len(years_loo))*p33_obs,np.ones(len(years_loo))*p66_obs,alpha=0.14, color='gray')


#%% MCRPSS CALCULATION FOR FUD

def ecdf(x, data):
        r'''
        For computing the empirical cumulative distribution function (ecdf) for a
        data sample at values x.

        Args:
            x (float or ndarray):
                The value(s) at which the ecdf is evaluated

            data (float or ndarray):
                A sample for which to compute the ecdf.

        Returns: ecdf_vals (ndarray):
            The ecdf for data, evaluated at x.
        '''

        if isinstance(x,float):
            #if x comes in as float, turn it into a numpy array
            x = np.array([x])


        if isinstance(data,float):
            #if data comes in as float, turn it into a numpy array
            data = np.array([data])


        # sort the values of data from smallest to largest
        xs = np.sort(data[~np.isnan(data)])

        # get the sample size of xs satisfying xs<=x for each x
        def func(vals):
            return len(xs[xs<=vals])

        ys = [len(xs[xs<=vals]) for vals in x]

        return np.array(ys)/float(len(xs))

FUD_obs = avg_freezeup_doy[1:]
CRPS_MLR = np.zeros(len(years_loo))*np.nan
CRPS_clim = np.zeros(len(years_loo))*np.nan
for iyr,yr in enumerate(years_loo):
    ecdf_MLR = ecdf(np.arange(325,440,1),pred_em_all[iyr,0:25])
    ecdf_clim = ecdf(np.arange(325,440,1),FUD_obs[np.where(years_loo != years_loo[iyr])])
    ecdf_obs = np.zeros(len(np.arange(325,440,1)))
    ecdf_obs[int(np.where(np.arange(325,440,1) == FUD_obs[iyr])[0]):] = 1
    # plt.figure()
    # plt.plot(np.arange(325,385,1),ecdf_obs, color='k')
    # plt.plot(np.arange(325,385,1),ecdf_clim,color='gray')
    # plt.plot(np.arange(325,385,1),ecdf_MLR)
    CRPS_MLR[iyr] = np.sum((ecdf_obs-ecdf_MLR)**2.)
    CRPS_clim[iyr] = np.sum((ecdf_obs-ecdf_clim)**2.)
print('--------------------------------------------')
print('Mean CRPS MLR: ', np.mean(CRPS_MLR))
print('Mean CRPS clim: ', np.mean(CRPS_clim))
print('MCRPSS MLR: ',(1-(np.mean(CRPS_MLR)/np.mean(CRPS_clim)))*100, '%')


#%% SPREAD-ERROR RELATIONSHIP
N_cases = len(years_loo)
N_ens = Nmembers

var_x = np.nanvar(pred_em_all,axis=1, ddof=1)
avg_var_x = np.nanmean(var_x)
mean_x = np.nanmean(pred_em_all,axis=1)
MSE = np.nanmean((mean_x-FUD_obs)**2.)
MSE_fair = MSE - (avg_var_x/N_ens)
alpha_fair = MSE_fair/avg_var_x
print('--------------------------------------------')
print('Spread-Error alpha_fair: ', alpha_fair)


#%% MCRPSS CALCULATION - SEAS5 Dec. Ta_mean
x = X_test_all[:,0,:]
y_obs = pred_df['Dec. Avg. Ta_mean'][1:].values
CRPS_x = np.zeros(len(years_loo))*np.nan
CRPS_clim = np.zeros(len(years_loo))*np.nan
for iyr,yr in enumerate(years_loo):
    ecdf_x = ecdf(np.arange(-12,5,0.01),x[iyr,0:25])
    ecdf_clim = ecdf(np.arange(-12,5,0.01),y_obs[np.where(years_loo != years_loo[iyr])])
    ecdf_obs = np.zeros(len(np.arange(-12,5,0.01)))
    ecdf_obs[int(np.where(np.round(np.arange(-12,5,0.01),2) == np.round(y_obs[iyr],2))[0]):] = 1
    # plt.figure()
    # plt.plot(np.arange(-12,5,0.01),ecdf_obs, color='k')
    # plt.plot(np.arange(-12,5,0.01),ecdf_clim,color='gray')
    # plt.plot(np.arange(-12,5,0.01),ecdf_x)
    CRPS_x[iyr] = np.sum((ecdf_obs-ecdf_x)**2.)
    CRPS_clim[iyr] = np.sum((ecdf_obs-ecdf_clim)**2.)
print('--------------------------------------------')
print('Mean CRPS - SEAS5 Dec. Ta_mean: ', np.mean(CRPS_x))
print('Mean CRPS clim: ', np.mean(CRPS_clim))
print('MCRPSS - SEAS5 Dec. Ta_mean: ',(1-(np.mean(CRPS_x)/np.mean(CRPS_clim)))*100, '%')

fig_SEAS5_DecTa,ax_SEAS5_DecTa = plt.subplots(figsize=[12,5])
ax_SEAS5_DecTa.plot(years_loo,y_obs,'o-',color='k',label='ERA5 reanalysis')
ax_SEAS5_DecTa.plot(years_loo,np.nanmean(x,axis=1),'o-',color=plt.get_cmap('tab20c')(4),label='SEAS5 - Ensemble mean forecasts')
for im in range(Nmembers):
    if im == 0:
        ax_SEAS5_DecTa.plot(years_loo,x[:,im],color=plt.get_cmap('tab20c')(7),linewidth ='0.86',label='SEAS5 - Individual member forecasts')
    else:
        ax_SEAS5_DecTa.plot(years_loo,x[:,im],color=plt.get_cmap('tab20c')(7),linewidth ='0.86')
ax_SEAS5_DecTa.plot(years_loo,y_obs,'o-',color='k')
ax_SEAS5_DecTa.plot(years_loo,np.nanmean(x,axis=1),'o-',color=plt.get_cmap('tab20c')(4))
ax_SEAS5_DecTa.legend()
ax_SEAS5_DecTa.set_ylabel('December Avg. Air Temperature ($^{\circ}$C)')
ax_SEAS5_DecTa.set_xlabel('Year')
plt.tight_layout()
ax_SEAS5_DecTa.grid(linestyle=':')
if savefig:
    save_folder = './'
    model_name = 'SEAS5_DecTa'
    fig_SEAS5_DecTa.savefig(save_folder+model_name+'_timeseries_'+'startmonth_'+str(start_month)+'.png', dpi=600)

var_x = np.nanvar(x,axis=1, ddof=1)
avg_var_x = np.nanmean(var_x)
mean_x = np.nanmean(x,axis=1)
MSE = np.nanmean((mean_x-y_obs)**2.)
MSE_fair = MSE - (avg_var_x/N_ens)
alpha_fair = MSE_fair/avg_var_x
print('--------------------------------------------')
print('Spread-Error alpha_fair- SEAS5 Dec. Ta_mean: ', alpha_fair)



#%% CATEGORICAL BASELINE FOR SEAS5 - DEC. TAIR
x = X_test_all[:,0,:]

x_ensemble_mean = np.nanmean(x,axis=1)
[m,b],_ = linear_fit(years_loo[0:np.where(years_loo == 2016)[0][0]], x_ensemble_mean[0:np.where(years_loo == 2016)[0][0]])
x_lin_detrended_anomaly = np.zeros(x.shape)
for im in range(Nmembers):
    x_lin_detrended_anomaly[:,im] = x[:,im]-(m*years_loo + b)

# ---- USING RAW FORECAST VALUES ----

p33_ensemble_mean_SEAS5 = np.nanpercentile(x[0:np.where(years_loo==2016)[0][0]],33.33)
p66_ensemble_mean_SEAS5 = np.nanpercentile(x[0:np.where(years_loo==2016)[0][0]],66.66)
cat_SEAS5_ensemble_mean = np.zeros((len(years_loo)))*np.nan
cat_SEAS5 = np.zeros((len(years_loo)))*np.nan
for iyr in range(len(years_loo)):
    if x_ensemble_mean[iyr] <= p33_ensemble_mean_SEAS5:
        cat_SEAS5_ensemble_mean[iyr] = -1
    elif x_ensemble_mean[iyr] > p66_ensemble_mean_SEAS5:
        cat_SEAS5_ensemble_mean[iyr] = 1
    else:
        cat_SEAS5_ensemble_mean[iyr] = 0

    n_late = np.sum(x[iyr]>p66_ensemble_mean_SEAS5)
    n_early = np.sum(x[iyr]<=p33_ensemble_mean_SEAS5)
    n_normal = len(years_loo)-n_early-n_late
    if (n_early > n_late) & (n_early > n_normal):
        cat_SEAS5[iyr] = -1
    elif (n_late > n_early) & (n_late > n_normal):
        cat_SEAS5[iyr] = 1
    elif (n_normal > n_early) & (n_normal > n_late):
        cat_SEAS5[iyr] = 0
    else:
        cat_SEAS5[iyr] = 0
        print('Ambiguous year...',n_early,n_normal,n_late)

acc_SEAS5_ensemble_mean = (np.sum(cat_SEAS5_ensemble_mean == cat_obs)/(np.sum(~np.isnan(FUD_obs))))*100
acc_SEAS5 = (np.sum(cat_SEAS5 == cat_obs)/(np.sum(~np.isnan(FUD_obs))))*100
# print('Baseline Categorical Accuracy - SEAS5 ensemble_mean: ',acc_SEAS5_ensemble_mean)
print('Baseline Categorical Accuracy - SEAS5 probabilisitic category: ',acc_SEAS5)
# f,a = plt.subplots(nrows=2)
# a[0].plot(years_loo, FUD_obs, 'o-', color='k')
# a[1].plot(years_loo, x, '-',color=plt.get_cmap('tab20')(0),linewidth='0.6')
# a[1].plot(years_loo, x_ensemble_mean, 'o-',color=plt.get_cmap('tab20')(2))
# a[1].plot(years_loo, np.ones(len(years_loo))*p33_ensemble_mean_SEAS5,'-', color='k')
# a[1].plot(years_loo, np.ones(len(years_loo))*p66_ensemble_mean_SEAS5,'-', color='k')
# ax_SEAS5_DecTa.fill_between(years_loo,np.ones(len(years_loo))*p33_ensemble_mean_SEAS5,np.ones(len(years_loo))*p66_ensemble_mean_SEAS5,alpha=0.4, color=plt.get_cmap('tab20c')(4))
# ax_SEAS5_DecTa.fill_between(years_loo,np.ones(len(years_loo))*np.nanpercentile(pred_df['Dec. Avg. Ta_mean'][1:].values,33.33),np.ones(len(years_loo))*np.nanpercentile(pred_df['Dec. Avg. Ta_mean'][1:].values,66.66),alpha=0.4, color='gray')




# ---- OR USING LINEARLY DETRENDED FORECAST VALUES ----

p33_lin_detrended_ensemble_mean_SEAS5 = np.nanpercentile(x_lin_detrended_anomaly[0:np.where(years_loo==2016)[0][0]],33.33)
p66_lin_detrended_ensemble_mean_SEAS5 = np.nanpercentile(x_lin_detrended_anomaly[0:np.where(years_loo==2016)[0][0]],66.66)
cat_SEAS5_lin_detrended_ensemble_mean = np.zeros((len(years_loo)))*np.nan
cat_SEAS5_lin_detrended = np.zeros((len(years_loo)))*np.nan
for iyr in range(len(years_loo)):
    if np.nanmean(x_lin_detrended_anomaly,axis=1)[iyr] <= p33_lin_detrended_ensemble_mean_SEAS5:
        cat_SEAS5_lin_detrended_ensemble_mean[iyr] = -1
    elif np.nanmean(x_lin_detrended_anomaly,axis=1)[iyr] > p66_lin_detrended_ensemble_mean_SEAS5:
        cat_SEAS5_lin_detrended_ensemble_mean[iyr] = 1
    else:
        cat_SEAS5_lin_detrended_ensemble_mean[iyr] = 0

    n_late = np.sum(x_lin_detrended_anomaly[iyr]>p66_lin_detrended_ensemble_mean_SEAS5)
    n_early = np.sum(x_lin_detrended_anomaly[iyr]<=p33_lin_detrended_ensemble_mean_SEAS5)
    n_normal = len(years_loo)-n_early-n_late
    if (n_early > n_late) & (n_early > n_normal):
        cat_SEAS5_lin_detrended[iyr] = -1
    elif (n_late > n_early) & (n_late > n_normal):
        cat_SEAS5_lin_detrended[iyr] = 1
    elif (n_normal > n_early) & (n_normal > n_late):
        cat_SEAS5_lin_detrended[iyr] = 0
    else:
        cat_SEAS5_lin_detrended[iyr] = 0
        print('Ambiguous year...',n_early,n_normal,n_late)

acc_SEAS5_lin_detrended_ensemble_mean = (np.sum(cat_SEAS5_lin_detrended_ensemble_mean == cat_obs)/(np.sum(~np.isnan(FUD_obs))))*100
acc_SEAS5_lin_detrended = (np.sum(cat_SEAS5_lin_detrended == cat_obs)/(np.sum(~np.isnan(FUD_obs))))*100
# print('Baseline Categorical Accuracy - SEAS5 linearly detrended ensemble_mean: ',acc_SEAS5_lin_detrended_ensemble_mean)
print('Baseline Categorical Accuracy - SEAS5 linearly detrended anomalies - probabilistic category: ',acc_SEAS5_lin_detrended)
# f,a = plt.subplots(nrows=2)
# a[0].plot(years_loo, FUD_obs, 'o-', color='k')
# a[1].plot(years_loo, x_lin_detrended_anomaly, '-',color=plt.get_cmap('tab20')(0),linewidth='0.6')
# a[1].plot(years_loo, np.nanmean(x_lin_detrended_anomaly,axis=1), 'o-',color=plt.get_cmap('tab20')(2))
# a[1].plot(years_loo, np.ones(len(years_loo))*p33_lin_detrended_ensemble_mean_SEAS5,'-', color='k')
# a[1].plot(years_loo, np.ones(len(years_loo))*p66_lin_detrended_ensemble_mean_SEAS5,'-', color='k')





#%% MCRPSS CALCULATION - SEAS5 Nov. snowfall
if start_month < 12:
    x = X_test_all[:,1,:]
    y_obs = pred_df['Nov. Tot. snowfall'][1:].values
    CRPS_x = np.zeros(len(years_loo))*np.nan
    CRPS_clim = np.zeros(len(years_loo))*np.nan
    for iyr,yr in enumerate(years_loo):
        ecdf_x = ecdf(np.arange(0,0.12,0.0001),x[iyr,0:25])
        ecdf_clim = ecdf(np.arange(0,0.12,0.0001),y_obs[np.where(years_loo != years_loo[iyr])])
        ecdf_obs = np.zeros(len(np.arange(0,0.12,0.0001)))
        ecdf_obs[int(np.where(np.round(np.arange(0,0.12,0.0001),4) == np.round(y_obs[iyr],4))[0]):] = 1
        # plt.figure()
        # plt.plot(np.arange(0,0.12,0.0001),ecdf_obs, color='k')
        # plt.plot(np.arange(0,0.12,0.0001),ecdf_clim,color='gray')
        # plt.plot(np.arange(0,0.12,0.0001),ecdf_x)
        CRPS_x[iyr] = np.sum((ecdf_obs-ecdf_x)**2.)
        CRPS_clim[iyr] = np.sum((ecdf_obs-ecdf_clim)**2.)
    print('--------------------------------------------')
    print('Mean CRPS - SEAS5 Nov. snowfall: ', np.mean(CRPS_x))
    print('Mean CRPS clim: ', np.mean(CRPS_clim))
    print('MCRPSS - SEAS5 Nov. snowfall: ',(1-(np.mean(CRPS_x)/np.mean(CRPS_clim)))*100, '%')

    fig_SEAS5_Novsnow,ax_SEAS5_Novsnow = plt.subplots(figsize=[12,5])
    ax_SEAS5_Novsnow.plot(years_loo,y_obs,'o-',color='k',label='ERA5 reanalysis')
    ax_SEAS5_Novsnow.plot(years_loo,np.nanmean(x,axis=1),'o-',color=plt.get_cmap('tab20c')(0),label='SEAS5 - Ensemble mean forecast')
    for im in range(Nmembers):
        if im == 0:
            ax_SEAS5_Novsnow.plot(years_loo,x[:,im],color=plt.get_cmap('tab20c')(3),linewidth ='0.86',label='SEAS5 - Individual member forecasts')
        else:
            ax_SEAS5_Novsnow.plot(years_loo,x[:,im],color=plt.get_cmap('tab20c')(3),linewidth ='0.86')
    ax_SEAS5_Novsnow.plot(years_loo,y_obs,'o-',color='k')
    ax_SEAS5_Novsnow.plot(years_loo,np.nanmean(x,axis=1),'o-',color=plt.get_cmap('tab20c')(0))
    ax_SEAS5_Novsnow.legend()
    ax_SEAS5_Novsnow.set_ylabel('November Tot. snowfall (m of equivalent water)')
    ax_SEAS5_Novsnow.set_xlabel('Year')
    plt.tight_layout()
    ax_SEAS5_Novsnow.grid(linestyle=':')
    if savefig:
        save_folder = './'
        model_name = 'SEAS5_Novsnow'
        fig_SEAS5_Novsnow.savefig(save_folder+model_name+'_timeseries_'+'startmonth_'+str(start_month)+'.png', dpi=600)


    var_x = np.nanvar(x,axis=1, ddof=1)
    avg_var_x = np.nanmean(var_x)
    mean_x = np.nanmean(x,axis=1)
    MSE = np.nanmean((mean_x-y_obs)**2.)
    MSE_fair = MSE - (avg_var_x/N_ens)
    alpha_fair = MSE_fair/avg_var_x
    print('--------------------------------------------')
    print('Spread-Error alpha_fair - SEAS5 Nov. snowfall: ', alpha_fair)



#%%

# x = X_test_all[:,0,:]
# y_obs = pred_df['Dec. Avg. Ta_mean'][1:].values
# plt.figure()
# plt.plot(y_obs,np.nanmean(x,axis=1),'o')
# plt.plot(np.arange(np.nanmin((np.nanmin(y_obs),np.nanmin(x))),np.nanmax((np.nanmax(y_obs),np.nanmax(x)))),np.arange(np.nanmin((np.nanmin(y_obs),np.nanmin(x))),np.nanmax((np.nanmax(y_obs),np.nanmax(x)))))


# x = X_test_all[:,1,:]
# y_obs = pred_df['Nov. Tot. snowfall'][1:].values
# plt.figure()
# plt.plot(y_obs,np.nanmean(x,axis=1),'o')
# plt.plot(np.arange(0,0.05,0.001),np.arange(0,0.05,0.001))


