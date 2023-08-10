#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:23:12 2022

@author: Amelie
"""

#%%
# local_path = '/storage/amelie/'
local_path = '/Volumes/SeagateUSB/McGill/Postdoc/'

import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath(local_path +'slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime as dt
import itertools
import calendar

import statsmodels.api as sm
from netCDF4 import Dataset

from functions import K_to_C
from functions import detect_FUD_from_Tw, detrend_ts
from functions_MLR import get_monthly_vars_from_daily, remove_collinear_features
from functions_MLR import find_models,eval_accuracy_multiple_models,make_metric_df
from functions_MLR import find_all_column_combinations

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
def loo_eval(model,model_type,X_in,y_in,years,normalize=True,X_frcst=None,show_coeff=False,plot=True,verbose=True):

    loo = LeaveOneOut()

    y_pred_test = np.zeros(y_in.shape)
    y_plot = np.zeros(y_in.shape)

    if show_coeff:
        fig_c,ax_c = plt.subplots(nrows = int(len(years)/4), ncols = 4,sharex=True,sharey=True)

    for i,[train_index, test_index] in enumerate(loo.split(X_in)):
        if verbose:
            print('------------')
            print('TEST YEAR = ', years[i])
        X_train = X_in.iloc[train_index]
        if X_frcst is None:
            X_test = X_in.iloc[test_index]
        else:
            X_train_frcst = X_frcst.iloc[train_index]
            X_test = X_frcst.iloc[test_index]
        y_train, y_test = y_in.iloc[train_index], y_in.iloc[test_index]
        print('YYYYYYYYYY',X_train)
        # Standardize predictors:
        if X_frcst is None:
            # Xscaler = MinMaxScaler()
            Xscaler = StandardScaler()
            Xscaler = Xscaler.fit(X_train)
            X_train_scaled = Xscaler.transform(X_train)
            X_test_scaled = Xscaler.transform(X_test)
        else:
            # Xscaler = MinMaxScaler()
            Xscaler = StandardScaler()
            Xscaler_frcst = StandardScaler()
            Xscaler = Xscaler.fit(X_train)
            Xscaler_frcst = Xscaler_frcst.fit(X_train_frcst)
            X_train_scaled = Xscaler.transform(X_train)
            X_test_scaled = Xscaler_frcst.transform(X_test)

        # yscaler = MinMaxScaler()
        yscaler = StandardScaler()
        yscaler = yscaler.fit(y_train)
        # y_train_scaled = yscaler.transform(y_train)
        # y_test_scaled = yscaler.transform(y_test)

        # Get Prediction:
        if normalize:
            X_train_fit = X_train_scaled
            X_test_fit = X_test_scaled
        else:
            X_train_fit = X_train
            X_test_fit = X_test
        # X_train_fit = X_train_scaled
        # X_test_fit = X_test_scaled
        y_pred_test[i] = model.fit(X_train_fit,y_train).predict(X_test_fit)
        y_plot[i] = y_test
        print('Reg. coeff.: ', model.coef_[model.coef_ != 0])

        if not np.all(model.coef_ == 0):
            if verbose:
                print('Reg. coeff.: ', model.coef_[model.coef_ != 0])
                print('Stand. coeff.: ', model.coef_[model.coef_ != 0]*(np.nanstd(X_train_scaled,axis=0)/np.nanstd(y_train)))

            if model_type == 'lasso':
                x = np.where(model.coef_)[0]
            else:
                x = np.arange(X_in.shape[1])

            if show_coeff:
                if i/len(years) < 0.25:
                    ax_c[i,0].stem(
                    x,
                    model.coef_[model.coef_ != 0],
                    markerfmt="x")
                elif i/len(years) < 0.5:
                    ax_c[i-int(len(years)/4),1].stem(
                    x,
                    model.coef_[model.coef_ != 0],
                    markerfmt="x")
                elif i/len(years) < 0.75:
                    ax_c[i-2*int(len(years)/4),2].stem(
                    x,
                    model.coef_[model.coef_ != 0],
                    markerfmt="x")
                else:
                    ax_c[i-3*int(len(years)/4),3].stem(
                    x,
                    model.coef_[model.coef_ != 0],
                    markerfmt="x")

    if plot:
        plt.figure()
        plt.plot(years,y_plot,'-o',color='k')
        plt.plot(years,y_pred_test,'-o',color=plt.get_cmap('tab20c')(0))

    mae = np.nanmean(np.abs(y_plot[:]-y_pred_test[:]))
    acc7days = np.sum(np.abs(y_plot[:]-y_pred_test[:]) <= 7)/(np.sum(~np.isnan(y_in)))

    if verbose:
        print('MAE:', mae)
        print('7-day accuracy: ', acc7days)


    # And finally, fit model with all data:
    if X_frcst is None:
        X_train = X_in
    else:
        X_train = X_in
        X_train_frcst = X_frcst
    y_train = y_in

    if X_frcst is None:
        # Xscaler = MinMaxScaler()
        Xscaler = StandardScaler()
        Xscaler = Xscaler.fit(X_train)
        X_train_scaled = Xscaler.transform(X_train)
    else:
        # Xscaler = MinMaxScaler()
        Xscaler = StandardScaler()
        Xscaler_frcst = StandardScaler()
        Xscaler = Xscaler.fit(X_train)
        Xscaler_frcst = Xscaler_frcst.fit(X_train_frcst)
        X_train_scaled = Xscaler.transform(X_train)

    if normalize:
        X_train_fit = X_train_scaled
    else:
        X_train_fit = X_train
    # X_train_fit = X_train_scaled
    model = model.fit(X_train_fit,y_train)

    return y_pred_test, model, Xscaler, mae, acc7days


#%%
if __name__ == "__main__":
    p_critical = 0.05

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

    start_yr = 1992
    end_yr = 2019

    train_yr_start = start_yr # [1992 - 2007] = 16 years
    valid_yr_start = 2008 # [2008 - 2013] = 6 years
    test_yr_start = 2014  # [2014 - 2019] = 6 years
    nsplits = end_yr-valid_yr_start+1
    nfolds = 5

    anomaly = True
    # anomaly = False

    # norm_MLR = True
    norm_MLR = False

    freezeup_opt = 1

    verbose = False

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

    p_list = ['Avg. cloud cover','Avg. cloud cover','Avg. cloud cover','Avg. cloud cover','Avg. cloud cover','Avg. level Ottawa River','Avg. level Ottawa River','Avg. level Ottawa River','Avg. level Ottawa River','Avg. level Ottawa River','NAO','NAO','NAO','NAO','NAO','PDO','PDO','PDO','PDO','PDO','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. discharge St-L. River','Avg. Ta_mean','Avg. Ta_mean','Avg. Ta_mean','Avg. Ta_mean','Avg. Ta_mean','Tot. snowfall','Tot. snowfall','Tot. snowfall', 'Avg. windspeed', 'Avg. windspeed','Avg. windspeed','Avg. windspeed','Avg. windspeed','Avg. SW down (sfc)','Avg. SW down (sfc)','Avg. SW down (sfc)','Avg. SW down (sfc)','Avg. SW down (sfc)', 'Avg. LH (sfc)', 'Avg. LH (sfc)','Avg. LH (sfc)','Avg. LH (sfc)','Avg. LH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)','Avg. SH (sfc)' ]
    m_list = [12,                11,                 10,               9,                8,                   12,                       11,                       10,                      9,                        8,                        12,   11,    10,     9,  8,    12,    11,   10,    9, 8,     12,                         11,                           10,                         9,                           8,                            12,           11,            10,           9,              8,            12,               11,              10,             8,                9,                10,             11,             12,              8,                   9 ,                  10,                  11,                  12,                   8,                9,              10,           11,              12,              8,              9,              10,            11,             12,             ]
    # p_list = ['Avg. Ta_mean','Tot. snowfall','Tot. snowfall']
    # m_list = [12,             11,             12            ]

    # Make dataframe with all predictors
    month_str = ['Jan. ', 'Feb. ', 'Mar. ', 'Apr. ', 'May ', 'Jun. ','Jul. ', 'Aug. ', 'Sep. ', 'Oct. ', 'Nov. ','Dec. ']
    pred_arr = np.zeros((monthly_pred_data.shape[1],len(p_list)))*np.nan
    col = []
    for i in range(len(p_list)):
        ipred = np.where(pred_names == p_list[i])[0][0]
        pred_arr[:,i] = monthly_pred_data[ipred,:,m_list[i]-1]
        col.append(month_str[m_list[i]-1]+p_list[i])
    pred_df =  pd.DataFrame(pred_arr,columns=col)

    pred_df_clim = pd.DataFrame(np.expand_dims(np.nanmean(pred_df,axis=0),axis=0),columns=col)
    if anomaly:
        for c in range(pred_df.shape[1]):
            pred_df.iloc[:,c] = pred_df.iloc[:,c] - pred_df_clim.iloc[0,c]

    #%% Get CanSIPS forecasts
    # month_lead0 = 11
    # # Select only 1 feature for now:
    # feature = 'TMP_TGL_2m'
    # # feature = 'PRATE_SFC_0'

    # def get_monthly_ensemble_mean(feature,region,month,year_start=1981,year_end=2010):
    #     from functions import ncdump

    #     def ensemble_avg(var_in):
    #         if len(var_in.shape) > 1:
    #             var_out = np.zeros((12,var_in.shape[1],var_in.shape[2]))*np.nan
    #             for imonth in range(12):
    #                 var_out[imonth,:,:] = np.nanmean(var_in[imonth:240:12,:,:],axis=0)
    #         else:
    #             var_out = np.zeros((12))*np.nan
    #             for imonth in range(12):
    #                 var_out[imonth] = np.nanmean(var_in[imonth:240:12],axis=0)

    #         return var_out


    #     ftype = '5months'
    #     data = np.load(p_dir+base+res+'ensemble_vars_sep_dec_f'+ftype+'.npz')
    #     lat = data['lat'][:]
    #     lon = data['lon'][:]

    #     if region == 'D':
    #         rlon1, rlat1 = 360-77.5, 43.5
    #         rlon2, rlat2 = 360-73.5, 45.5
    #     if region == 'YUL':
    #         rlon1, rlat1 = 360-74.5, 45.5
    #         rlon2, rlat2 = 360-73.5, 45.5
    #     if region == 'all':
    #         rlon1, rlat1 = 0.5, -89.5
    #         rlon2, rlat2 = 359.5,  89.5
    #     if region == 'Dplus':
    #         rlon1, rlat1 = 360-84.5, 42.5
    #         rlon2, rlat2 = 360-72.5, 47.5
    #     if region == 'test':
    #         rlon1, rlat1 = 360-78.5, 31.5
    #         rlon2, rlat2 = 360-73.5, 37.5

    #     ilat1 = np.where(lat == rlat1)[0][0]
    #     ilat2 = np.where(lat == rlat2)[0][0]+1
    #     ilon1 = np.where(lon == rlon1)[0][0]
    #     ilon2 = np.where(lon == rlon2)[0][0]+1

    #     lat_select = lat[ilat1:ilat2+1]
    #     lon_select = lon[ilon1:ilon2+1]

    #     if region == 'all':
    #         var_out = np.zeros((year_end-year_start+1,12,len(lat_select),len(lon_select)))*np.nan
    #     else:
    #         var_out = np.zeros((year_end-year_start+1,12))*np.nan

    #     for iyr,year in enumerate(np.arange(year_start,year_end+1)):
    #         extension = "_{}-{}_allmembers.grib2.nc".format(year, str(month).rjust(2, '0'))
    #         filename = base + res + feature + extension
    #         path = r_dir + "{}-{}/".format(year, str(month).rjust(2, '0'))
    #         ncid = Dataset(path+filename,'r')
    #         ndump = ncdump(ncid, verb=False)
    #         v = ndump[2][-1]
    #         var = np.squeeze(ncid[v][:])
    #         # Average in selected region before making climatology
    #         if region == 'all':
    #             var_ensm = ensemble_avg(var)
    #             var_out[iyr,:,:,:] = var_ensm
    #         else:
    #             var = np.nanmean(var[:,ilat1:ilat2,ilon1:ilon2],axis=(1,2))
    #             var_ensm = ensemble_avg(var)
    #             var_out[iyr,:] = var_ensm

    #     p33 = np.squeeze(np.nanpercentile(var_out,100/3.,axis=0))
    #     p66 = np.squeeze(np.nanpercentile(var_out,200/3.,axis=0))

    #     return var_out, np.squeeze(np.nanmean(var_out,axis=0)), np.squeeze(np.nanstd(var_out,axis=0)),p33, p66

    # feature_list = ['TMP_TGL_2m','PRATE_SFC_0']
    # base = "cansips_hindcast_raw_"
    # res = "latlon1.0x1.0_"
    # r_dir = local_path+'slice/data/raw/CanSIPS/hindcast/raw/'
    # p_dir = local_path+'slice/data/processed/CanSIPS/hindcast/'

    # ys = 1992
    # ye = 2019
    # years_cansips = np.arange(ys,ye+1)

    # region = 'D'

    # if region == 'all':
    #     cansips_clim = np.zeros( (  3, 180, 360 )  )*np.nan
    #     cansips_clim_std = np.zeros( ( 3, 180, 360 )  )*np.nan
    #     cansips_clim_p33 = np.zeros( ( 3, 180, 360 )  )*np.nan
    #     cansips_clim_p66 = np.zeros( ( 3, 180, 360 )  )*np.nan
    #     cansips_cat_frcst = np.zeros((len(years_cansips),3, 180, 360))*np.nan
    #     cansips_frcst = np.zeros((len(years_cansips),3, 180, 360))*np.nan
    # else:
    #     cansips_clim = np.zeros(3)*np.nan
    #     cansips_clim_std = np.zeros(3)*np.nan
    #     cansips_clim_p33 = np.zeros(3)*np.nan
    #     cansips_clim_p66 = np.zeros(3)*np.nan
    #     cansips_cat_frcst = np.zeros((len(years_cansips),3))*np.nan
    #     cansips_frcst = np.zeros((len(years_cansips),3))*np.nan

    # # Get November (lead = 0) & December (lead =1) Forecast Climatology and Std Deviation
    # var_ensm_avg_monthly,var_ensm_clim,var_ensm_std,var_ensm_p33,var_ensm_p66 = get_monthly_ensemble_mean(feature,region,month_lead0,year_start=ys,year_end=ye)

    # if feature[0:3] == 'TMP':
    #     cansips_clim[0:2] = np.squeeze(var_ensm_clim[0:2]-273.15) # Keep only 0:2 to have only December (lead = 0) & January (lead =1)
    #     cansips_clim_std[0:2] = np.squeeze(var_ensm_std[0:2]-273.15)
    #     cansips_clim_p33[0:2] = np.squeeze(var_ensm_p33[0:2]-273.15)
    #     cansips_clim_p66[0:2] = np.squeeze(var_ensm_p66[0:2]-273.15)
    #     cansips_frcst[:,0:2] = var_ensm_avg_monthly[:,0:2] -273.15

    # else:
    #     cansips_clim[0:2] = np.squeeze(var_ensm_clim[0:2]) # Keep only 0:2 to have only December (lead = 0) & January (lead =1)
    #     cansips_clim_std[0:2] = np.squeeze(var_ensm_std[0:2])
    #     cansips_clim_p33[0:2] = np.squeeze(var_ensm_p33[0:2])
    #     cansips_clim_p66[0:2] = np.squeeze(var_ensm_p66[0:2])
    #     cansips_frcst[:,0:2] = var_ensm_avg_monthly[:,0:2]

    # if anomaly:
    #     cansips_frcst = cansips_frcst - cansips_clim

    #%% Load or compute/save SEAS5 forecasts
    load = True
    save = False
    # filename = 'SEAS5_MLR_forecast_predictors_newsnow_testclim'
    filename = 'SEAS5_MLR_forecast_predictors_newsnow'
    # filename = 'SEAS5_MLR_forecast_predictors'
    # filename2 = 'SEAS5_MLR_NovTa'

    if load:
        dtmp = np.load(filename+'.npz', allow_pickle = True)
        SEAS5_Dec_Ta_mean_em = dtmp['SEAS5_Dec_Ta_mean_em']
        SEAS5_Dec_snowfall_em = dtmp['SEAS5_Dec_snowfall_em']
        SEAS5_Nov_snowfall_em = dtmp['SEAS5_Nov_snowfall_em']

        SEAS5_Dec_Ta_mean_clim = dtmp['SEAS5_Dec_Ta_mean_clim']
        SEAS5_Dec_snowfall_clim = dtmp['SEAS5_Dec_snowfall_clim']
        SEAS5_Nov_snowfall_clim = dtmp['SEAS5_Nov_snowfall_clim']

        SEAS5_Dec_Ta_mean_all = dtmp['SEAS5_Dec_Ta_mean_all']
        SEAS5_Dec_snowfall_all = dtmp['SEAS5_Dec_snowfall_all']
        SEAS5_Nov_snowfall_all = dtmp['SEAS5_Nov_snowfall_all']

        # dtmp = np.load(filename2+'.npz', allow_pickle = True)
        SEAS5_Nov_Ta_mean_em = dtmp['SEAS5_Nov_Ta_mean_em']
        SEAS5_Nov_Ta_mean_clim = dtmp['SEAS5_Nov_Ta_mean_clim']
        SEAS5_Nov_Ta_mean_all = dtmp['SEAS5_Nov_Ta_mean_all']

        # Remove climatology to make forecast anomalies (if desired):
        SEAS5_Dec_Ta_mean_em = SEAS5_Dec_Ta_mean_em - anomaly*(SEAS5_Dec_Ta_mean_clim)
        SEAS5_Dec_snowfall_em = SEAS5_Dec_snowfall_em -anomaly*(SEAS5_Dec_snowfall_clim)
        SEAS5_Nov_snowfall_em = SEAS5_Nov_snowfall_em-anomaly*(SEAS5_Nov_snowfall_clim)
        SEAS5_Nov_Ta_mean_em = SEAS5_Nov_Ta_mean_em-anomaly*(SEAS5_Nov_Ta_mean_clim)
        SEAS5_Dec_Ta_mean_all = SEAS5_Dec_Ta_mean_all-anomaly*(np.repeat(SEAS5_Dec_Ta_mean_clim[:, :, np.newaxis],SEAS5_Dec_Ta_mean_all.shape[2], axis=2))
        SEAS5_Dec_snowfall_all = SEAS5_Dec_snowfall_all-anomaly*(np.repeat(SEAS5_Dec_snowfall_clim[:, :, np.newaxis],SEAS5_Dec_snowfall_all.shape[2], axis=2))
        SEAS5_Nov_snowfall_all = SEAS5_Nov_snowfall_all-anomaly*(np.repeat(SEAS5_Nov_snowfall_clim[:, :, np.newaxis],SEAS5_Nov_snowfall_all.shape[2], axis=2))
        SEAS5_Nov_Ta_mean_all = SEAS5_Nov_Ta_mean_all-anomaly*(np.repeat(SEAS5_Nov_Ta_mean_clim[:, :, np.newaxis],SEAS5_Nov_Ta_mean_all.shape[2], axis=2))

        # Put nan in 1992 because we do not have SEAS5
        SEAS5_Dec_Ta_mean_em[:,0] = np.zeros(SEAS5_Dec_Ta_mean_em[:,0].shape)*np.nan
        SEAS5_Dec_Ta_mean_clim[:,0] = np.zeros(SEAS5_Dec_Ta_mean_clim[:,0].shape)*np.nan
        SEAS5_Dec_Ta_mean_all[:,0,:] = np.zeros(SEAS5_Dec_Ta_mean_all[:,0,:].shape)*np.nan

        SEAS5_Nov_Ta_mean_em[:,0] = np.zeros(SEAS5_Nov_Ta_mean_em[:,0].shape)*np.nan
        SEAS5_Nov_Ta_mean_clim[:,0] = np.zeros(SEAS5_Nov_Ta_mean_clim[:,0].shape)*np.nan
        SEAS5_Nov_Ta_mean_all[:,0,:] = np.zeros(SEAS5_Nov_Ta_mean_all[:,0,:].shape)*np.nan

        SEAS5_Dec_snowfall_em[:,0] = np.zeros(SEAS5_Dec_snowfall_em[:,0].shape)*np.nan
        SEAS5_Dec_snowfall_clim[:,0] = np.zeros(SEAS5_Dec_snowfall_clim[:,0].shape)*np.nan
        SEAS5_Dec_snowfall_all[:,0,:] = np.zeros(SEAS5_Dec_snowfall_all[:,0,:].shape)*np.nan

        SEAS5_Nov_snowfall_em[:,0] = np.zeros(SEAS5_Nov_snowfall_em[:,0].shape)*np.nan
        SEAS5_Nov_snowfall_clim[:,0] = np.zeros(SEAS5_Nov_snowfall_clim[:,0].shape)*np.nan
        SEAS5_Nov_snowfall_all[:,0,:] = np.zeros(SEAS5_Nov_snowfall_all[:,0,:].shape)*np.nan


    else:
        features = ['snowfall_processed','2m_temperature']

        r_dir = local_path + 'slice/data/raw/SEAS5/'
        region = 'D'
        base = 'SEAS5'

        SEAS5_years = np.arange(1993,2021+1)

        SEAS5_Nov_Ta_mean_em = np.zeros((1,len(years)))*np.nan
        SEAS5_Nov_Ta_mean_clim = np.zeros((1,len(years)))*np.nan
        SEAS5_Nov_Ta_mean_all = np.zeros((1,len(years),51))*np.nan

        SEAS5_Dec_Ta_mean_em = np.zeros((2,len(years)))*np.nan
        SEAS5_Dec_snowfall_em = np.zeros((2,len(years)))*np.nan
        SEAS5_Nov_snowfall_em = np.zeros((1,len(years)))*np.nan

        SEAS5_Dec_Ta_mean_clim = np.zeros((2,len(years)))*np.nan
        SEAS5_Dec_snowfall_clim = np.zeros((2,len(years)))*np.nan
        SEAS5_Nov_snowfall_clim = np.zeros((1,len(years)))*np.nan

        SEAS5_Dec_Ta_mean_all = np.zeros((2,len(years),51))*np.nan
        SEAS5_Dec_snowfall_all = np.zeros((2,len(years),51))*np.nan
        SEAS5_Nov_snowfall_all = np.zeros((1,len(years),51))*np.nan

        # Get SEAS5 forecasts - Nov. 1st
        month_lead0 = 11
        for feature in features:
            print(((month_lead0 == 11) & ((feature == 'snowfall')| (feature == 'snowfall_processed'))))
            for iyr,year in enumerate(years):
                if year in SEAS5_years:
                    print(year, ' - ', feature)
                    extension = "{}{}.nc".format(year, str(month_lead0).rjust(2, '0'))

                    path = r_dir +"region"+ region + "/{}-{}/".format(year, str(month_lead0).rjust(2, '0'))
                    fname = base + '_' + feature + '_' + extension

                    # Initialize forecast class
                    s = SEAS5frcst(path+fname)

                    # Get ensemble mean
                    var_em, time = s.read_vars([s.var,'time'],
                                                spatial_avg=True,
                                                ensemble_avg=True,
                                                lead = [1,2],
                                                time_rep='monthly',
                                                time_format='plot'
                                                )
                    if 'temperature' in feature: var_em = K_to_C(var_em)

                    # Get all members
                    var, time = s.read_vars([s.var,'time'],
                                              spatial_avg=True,
                                              lead = [1,2],
                                              time_rep='monthly',
                                              time_format='plot'
                                            )
                    if 'temperature' in feature: var = K_to_C(var)

                    # Get climatology
                    v_yr = s.get_climatology(spatial_avg=True,
                                             lead = [1,2],
                                             time_rep='monthly'
                                             )
                    var_clim_mean = v_yr[0]

                    if 'temperature' in feature: var_clim_mean = K_to_C(var_clim_mean)

                    if ((month_lead0 == 11) & (feature == '2m_temperature')):
                        SEAS5_Nov_Ta_mean_em[0,iyr] = var_em[0]
                        SEAS5_Nov_Ta_mean_all[0,iyr,0:var.shape[1]] = var[0,:]
                        SEAS5_Nov_Ta_mean_clim[0,iyr] = var_clim_mean[0]
                        SEAS5_Dec_Ta_mean_em[1,iyr] = var_em[1]
                        SEAS5_Dec_Ta_mean_all[1,iyr,0:var.shape[1]] = var[1,:]
                        SEAS5_Dec_Ta_mean_clim[1,iyr] = var_clim_mean[1]
                    if ((month_lead0 == 12) & (feature == '2m_temperature')):
                        SEAS5_Dec_Ta_mean_em[0,iyr] = var_em[0]
                        SEAS5_Dec_Ta_mean_all[0,iyr,0:var.shape[1]] = var[0,:]
                        SEAS5_Dec_Ta_mean_clim[0,iyr] = var_clim_mean[0]

                    if ((month_lead0 == 11) & ((feature == 'snowfall')| (feature == 'snowfall_processed'))):
                        SEAS5_Nov_snowfall_em[0,iyr] = var_em[0]
                        SEAS5_Nov_snowfall_all[0,iyr,0:var.shape[1]] = var[0,:]
                        SEAS5_Nov_snowfall_clim[0,iyr] = var_clim_mean[0]
                        SEAS5_Dec_snowfall_em[1,iyr] = var_em[1]
                        SEAS5_Dec_snowfall_all[1,iyr,0:var.shape[1]] = var[1,:]
                        SEAS5_Dec_snowfall_clim[1,iyr] = var_clim_mean[1]
                    if ((month_lead0 == 12) & ((feature == 'snowfall') | (feature == 'snowfall_processed'))):
                        SEAS5_Dec_snowfall_em[0,iyr] = var_em[0]
                        SEAS5_Dec_snowfall_all[0,iyr,0:var.shape[1]] = var[0,:]
                        SEAS5_Dec_snowfall_clim[0,iyr] = var_clim_mean[0]


        # Get SEAS5 forecasts - Dec. 1
        month_lead0 = 12
        for feature in features:
            for iyr,year in enumerate(years):
                if year in SEAS5_years:
                    print(year, ' - ', feature)
                    extension = "{}{}.nc".format(year, str(month_lead0).rjust(2, '0'))

                    path = r_dir +"region"+ region + "/{}-{}/".format(year, str(month_lead0).rjust(2, '0'))
                    fname = base + '_' + feature + '_' + extension

                    # Initialize forecast class
                    s = SEAS5frcst(path+fname)

                    # Get ensemble mean
                    var_em, time = s.read_vars([s.var,'time'],
                                                spatial_avg=True,
                                                ensemble_avg=True,
                                                lead = [1,2],
                                                time_rep='monthly',
                                                time_format='plot'
                                                )
                    if 'temperature' in feature: var_em = K_to_C(var_em)

                    # Get all members
                    var, time = s.read_vars([s.var,'time'],
                                              spatial_avg=True,
                                              lead = [1,2],
                                              time_rep='monthly',
                                              time_format='plot'
                                            )
                    if 'temperature' in feature: var = K_to_C(var)

                    # Get climatology
                    v_yr = s.get_climatology(spatial_avg=True,
                                             lead = [1,2],
                                             time_rep='monthly'
                                             )
                    var_clim_mean = v_yr[0]

                    if 'temperature' in feature: var_clim_mean = K_to_C(var_clim_mean)

                    if ((month_lead0 == 11) & (feature == '2m_temperature')):
                        SEAS5_Nov_Ta_mean_em[0,iyr] = var_em[0]
                        SEAS5_Nov_Ta_mean_all[0,iyr,0:var.shape[1]] = var[0,:]
                        SEAS5_Nov_Ta_mean_clim[0,iyr] = var_clim_mean[0]
                        SEAS5_Dec_Ta_mean_em[1,iyr] = var_em[1]
                        SEAS5_Dec_Ta_mean_all[1,iyr,0:var.shape[1]] = var[1,:]
                        SEAS5_Dec_Ta_mean_clim[1,iyr] = var_clim_mean[1]
                    if ((month_lead0 == 12) & (feature == '2m_temperature')):
                        SEAS5_Dec_Ta_mean_em[0,iyr] = var_em[0]
                        SEAS5_Dec_Ta_mean_all[0,iyr,0:var.shape[1]] = var[0,:]
                        SEAS5_Dec_Ta_mean_clim[0,iyr] = var_clim_mean[0]

                    if ((month_lead0 == 11) & ((feature == 'snowfall')| (feature == 'snowfall_processed'))):
                        SEAS5_Nov_snowfall_em[0,iyr] = var_em[0]
                        SEAS5_Nov_snowfall_all[0,iyr,0:var.shape[1]] = var[0,:]
                        SEAS5_Nov_snowfall_clim[0,iyr] = var_clim_mean[0]
                        SEAS5_Dec_snowfall_em[1,iyr] = var_em[1]
                        SEAS5_Dec_snowfall_all[1,iyr,0:var.shape[1]] = var[1,:]
                        SEAS5_Dec_snowfall_clim[1,iyr] = var_clim_mean[1]
                    if ((month_lead0 == 12) & ((feature == 'snowfall') | (feature == 'snowfall_processed'))):
                        SEAS5_Dec_snowfall_em[0,iyr] = var_em[0]
                        SEAS5_Dec_snowfall_all[0,iyr,0:var.shape[1]] = var[0,:]
                        SEAS5_Dec_snowfall_clim[0,iyr] = var_clim_mean[0]


        # Save or load SEAS5 variables:
        if save:
            np.savez(filename,
                    SEAS5_Dec_Ta_mean_em = SEAS5_Dec_Ta_mean_em,
                    SEAS5_Dec_snowfall_em = SEAS5_Dec_snowfall_em,
                    SEAS5_Nov_snowfall_em = SEAS5_Nov_snowfall_em,
                    SEAS5_Dec_Ta_mean_clim = SEAS5_Dec_Ta_mean_clim,
                    SEAS5_Dec_snowfall_clim = SEAS5_Dec_snowfall_clim,
                    SEAS5_Nov_snowfall_clim = SEAS5_Nov_snowfall_clim,
                    SEAS5_Dec_Ta_mean_all = SEAS5_Dec_Ta_mean_all,
                    SEAS5_Dec_snowfall_all = SEAS5_Dec_snowfall_all,
                    SEAS5_Nov_snowfall_all = SEAS5_Nov_snowfall_all,
                    SEAS5_Nov_Ta_mean_em = SEAS5_Nov_Ta_mean_em,
                    SEAS5_Nov_Ta_mean_clim = SEAS5_Nov_Ta_mean_clim,
                    SEAS5_Nov_Ta_mean_all = SEAS5_Nov_Ta_mean_all,
                    )

        # Remove climatology to make forecast anomalies (if desired):
        SEAS5_Dec_Ta_mean_em = SEAS5_Dec_Ta_mean_em - anomaly*(SEAS5_Dec_Ta_mean_clim)
        SEAS5_Dec_snowfall_em = SEAS5_Dec_snowfall_em -anomaly*(SEAS5_Dec_snowfall_clim)
        SEAS5_Nov_snowfall_em = SEAS5_Nov_snowfall_em-anomaly*(SEAS5_Nov_snowfall_clim)
        SEAS5_Nov_Ta_mean_em = SEAS5_Nov_Ta_mean_em-anomaly*(SEAS5_Nov_Ta_mean_clim)
        SEAS5_Dec_Ta_mean_all = SEAS5_Dec_Ta_mean_all-anomaly*(np.repeat(SEAS5_Dec_Ta_mean_clim[:, :, np.newaxis],SEAS5_Dec_Ta_mean_all.shape[2], axis=2))
        SEAS5_Dec_snowfall_all = SEAS5_Dec_snowfall_all-anomaly*(np.repeat(SEAS5_Dec_snowfall_clim[:, :, np.newaxis],SEAS5_Dec_snowfall_all.shape[2], axis=2))
        SEAS5_Nov_snowfall_all = SEAS5_Nov_snowfall_all-anomaly*(np.repeat(SEAS5_Nov_snowfall_clim[:, :, np.newaxis],SEAS5_Nov_snowfall_all.shape[2], axis=2))
        SEAS5_Nov_Ta_mean_all = SEAS5_Nov_Ta_mean_all-anomaly*(np.repeat(SEAS5_Nov_Ta_mean_clim[:, :, np.newaxis],SEAS5_Nov_Ta_mean_all.shape[2], axis=2))


    #%% Plot predictor time series

    # fig, ax = plt.subplots()
    # ax.set_title('Dec. Tair')
    # ax.plot(years, pred_df['Dec. Avg. Ta_mean'], label = 'Obs.')
    # ax.plot(years, SEAS5_Dec_Ta_mean_em[1,:], label = 'SEAS5 - Lead = 1 month')
    # # ax.plot(years, cansips_frcst[:,1], label = 'CanSIPS - Lead = 1 month')
    # ax.legend()

    # fig, ax = plt.subplots()
    # ax.set_title('Nov. snowfall')
    # ax.plot(years, (pred_df['Nov. Tot. snowfall']), label = 'Obs.')
    # ax.plot(years, SEAS5_Nov_snowfall_em[0,:], label = 'SEAS5 - Lead = 0 month')
    # ax.legend()

    # fig, ax = plt.subplots()
    # ax.set_title('Dec. snowfall')
    # ax.plot(years, pred_df['Dec. Tot. snowfall'], label = 'Obs.')
    # ax.plot(years, SEAS5_Dec_snowfall_em[1,:], label = 'SEAS5 - Lead = 1 month')
    # ax.legend()

    # fig, ax = plt.subplots()
    # ax.set_title('Dec. Tair')
    # ax.plot(years, pred_df['Dec. Avg. Ta_mean'], label = 'Obs.')
    # ax.plot(years, SEAS5_Dec_Ta_mean_em[0,:], label = 'SEAS5 - Lead = 0 month')
    # ax.legend()

    # fig, ax = plt.subplots()
    # ax.set_title('Dec. snowfall')
    # ax.plot(years, (pred_df['Dec. Tot. snowfall']) , label = 'Obs.')
    # ax.plot(years, SEAS5_Dec_snowfall_em[0,:], label = 'SEAS5 - Lead = 0 month')
    # ax.legend()

    #%% Load predictors for 2022:
    SeptNAO2022 = -0.702 - anomaly*(pred_df_clim['Sep. NAO'])
    SeptSH2022 = -76754.74194477081- anomaly*(pred_df_clim['Sep. Avg. SH (sfc)'])
    OctSH2022 = -34120.33740253003 - anomaly*(pred_df_clim['Oct. Avg. SH (sfc)'])
    SeptCloudCover2022 = 0.66412- anomaly*(pred_df_clim['Sep. Avg. cloud cover'])
    OctCloudCover2022 = 0.557011- anomaly*(pred_df_clim['Oct. Avg. cloud cover'])
    SeptORlevel2022 = 21.673 - anomaly*(pred_df_clim['Sep. Avg. level Ottawa River'])
    OctSLRdischarge2022 = 7554.41- anomaly*(pred_df_clim['Oct. Avg. discharge St-L. River'])
    NovTamean2022 = 4.275217950635346 - anomaly*(pred_df_clim['Nov. Avg. Ta_mean'])
    Novsnowfall2022 = 0.0003518938100026512 - anomaly*(pred_df_clim['Nov. Tot. snowfall'])
    SeptSW2022 = 544177.479283211 - anomaly*(pred_df_clim['Sep. Avg. SW down (sfc)'])
    AugSW2022 = 748539.3214844235 - anomaly*(pred_df_clim['Aug. Avg. SW down (sfc)'])

    #%% Get SEAS5 predictions for Nov. 2022:
    year = 2022
    month_lead0 = 11
    r_dir = local_path + 'slice/data/raw/SEAS51/'
    region = 'D'
    base = 'SEAS51'

    feature = '2m_temperature'
    extension = "{}{}.nc".format(year, str(month_lead0).rjust(2, '0'))
    path = r_dir +"region"+ region + "/{}-{}/".format(year, str(month_lead0).rjust(2, '0'))
    fname = base + '_' + feature + '_' + extension
    s = SEAS5frcst(path+fname)
    var_em, _ = s.read_vars([s.var,'time'],
                                        spatial_avg=True,
                                        ensemble_avg=True,
                                        lead = [1,2],
                                        time_rep='monthly',
                                        time_format='plot'
                                    )
    if 'temperature' in feature: var_em = K_to_C(var_em)
    var, _ = s.read_vars([s.var,'time'],
                              spatial_avg=True,
                              lead = [1,2],
                              time_rep='monthly',
                              time_format='plot'
                            )
    if 'temperature' in feature: var = K_to_C(var)
    v_yr = s.get_climatology(spatial_avg=True,
                             lead = [1,2],
                             time_rep='monthly'
                             )
    var_clim_mean = v_yr[0]
    if 'temperature' in feature: var_clim_mean = K_to_C(var_clim_mean)


    SEAS5NovTa2022_lead0 = var_em[0]-anomaly*(var_clim_mean[0])
    SEAS5DecTa2022_lead1 = var_em[1]-anomaly*(var_clim_mean[1])

    feature = 'snowfall_processed'
    extension = "{}{}.nc".format(year, str(month_lead0).rjust(2, '0'))
    path = r_dir +"region"+ region + "/{}-{}/".format(year, str(month_lead0).rjust(2, '0'))
    fname = base + '_' + feature + '_' + extension
    s = SEAS5frcst(path+fname)
    var_em, _ = s.read_vars([s.var,'time'],
                                        spatial_avg=True,
                                        ensemble_avg=True,
                                        lead = [1,2],
                                        time_rep='monthly',
                                        time_format='plot'
                                    )
    if 'temperature' in feature: var_em = K_to_C(var_em)
    var, _ = s.read_vars([s.var,'time'],
                              spatial_avg=True,
                              lead = [1,2],
                              time_rep='monthly',
                              time_format='plot'
                            )
    if 'temperature' in feature: var = K_to_C(var)
    v_yr = s.get_climatology(spatial_avg=True,
                             lead = [1,2],
                             time_rep='monthly'
                             )
    var_clim_mean = v_yr[0]
    if 'temperature' in feature: var_clim_mean = K_to_C(var_clim_mean)

    SEAS5Novsnowfall2022_lead0 = var_em[0]-anomaly*(var_clim_mean[0])
    SEAS5Decsnowfall2022_lead1 = var_em[1]-anomaly*(var_clim_mean[1])



    #%% Get SEAS5 predictions for Dec. 2022:
    year = 2022
    month_lead0 = 12
    r_dir = local_path + 'slice/data/raw/SEAS51/'
    region = 'D'
    base = 'SEAS51'

    feature = '2m_temperature'
    extension = "{}{}.nc".format(year, str(month_lead0).rjust(2, '0'))
    path = r_dir +"region"+ region + "/{}-{}/".format(year, str(month_lead0).rjust(2, '0'))
    fname = base + '_' + feature + '_' + extension
    s = SEAS5frcst(path+fname)
    var_em, _ = s.read_vars([s.var,'time'],
                                        spatial_avg=True,
                                        ensemble_avg=True,
                                        lead = [1,2],
                                        time_rep='monthly',
                                        time_format='plot'
                                    )
    if 'temperature' in feature: var_em = K_to_C(var_em)
    var, _ = s.read_vars([s.var,'time'],
                              spatial_avg=True,
                              lead = [1,2],
                              time_rep='monthly',
                              time_format='plot'
                            )
    if 'temperature' in feature: var = K_to_C(var)
    v_yr = s.get_climatology(spatial_avg=True,
                             lead = [1,2],
                             time_rep='monthly'
                             )
    var_clim_mean = v_yr[0]
    if 'temperature' in feature: var_clim_mean = K_to_C(var_clim_mean)

    SEAS5DecTa2022_lead0 = var_em[0] - anomaly*(var_clim_mean[0])

    feature = 'snowfall_processed'
    extension = "{}{}.nc".format(year, str(month_lead0).rjust(2, '0'))
    path = r_dir +"region"+ region + "/{}-{}/".format(year, str(month_lead0).rjust(2, '0'))
    fname = base + '_' + feature + '_' + extension
    s = SEAS5frcst(path+fname)
    var_em, _ = s.read_vars([s.var,'time'],
                                        spatial_avg=True,
                                        ensemble_avg=True,
                                        lead = [1,2],
                                        time_rep='monthly',
                                        time_format='plot'
                                    )
    if 'temperature' in feature: var_em = K_to_C(var_em)
    var, _ = s.read_vars([s.var,'time'],
                              spatial_avg=True,
                              lead = [1,2],
                              time_rep='monthly',
                              time_format='plot'
                            )
    if 'temperature' in feature: var = K_to_C(var)
    v_yr = s.get_climatology(spatial_avg=True,
                             lead = [1,2],
                             time_rep='monthly'
                             )
    var_clim_mean = v_yr[0]
    if 'temperature' in feature: var_clim_mean = K_to_C(var_clim_mean)

    SEAS5Decsnowfall2022_lead0 = var_em[0] - anomaly*(var_clim_mean[0])

    #%% Make predictions: Dec. Ta

    # Select model:
    pred = ['Dec. Avg. Ta_mean']
    pred_nov2022 = [SEAS5DecTa2022_lead1]
    pred_dec2022 = [SEAS5DecTa2022_lead0]

    # Perfect forecast model:
    X, y = pred_df[pred], pd.DataFrame(avg_freezeup_doy,columns=['Avg. FUD DOY'])
    y1,_,_,mae_perfect_test1_nov1,acc7days_perfect_test1_nov1 = loo_eval(LinearRegression(),'mlr',X,y,years,normalize=norm_MLR,show_coeff=False,plot=False,verbose=verbose)

    # Train models with past forecasts:
    pred_SEAS5_Nov1 = np.zeros((len(years),1))
    pred_SEAS5_Nov1[:,0] = SEAS5_Dec_Ta_mean_em[1,:] # Dec. Ta - Lead = 1
    pred_SEAS5_Nov1_df = pd.DataFrame(pred_SEAS5_Nov1)
    X_nov1 = pred_SEAS5_Nov1_df
    y_pred_test1_nov1, model1_nov1, Xscaler1_nov1, mae_test1_nov1, acc7days_test1_nov1 = loo_eval(LinearRegression(),'mlr',X_nov1.iloc[1:,:],y.iloc[1:,:],years[1:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)
    # #Train MLR model with obs. (ERA5) for Nov. and Dec., then test using the SEAS5 forecasts
    # y_pred_test_nov1, model_nov1, Xscaler_nov1, mae_test_nov1, acc7days_test_nov1 = loo_eval(LinearRegression(),'mlr',X.iloc[1:,:],y.iloc[1:,:],years[1:],X_frcst=X_nov1.iloc[1:,:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)

    pred_SEAS5_Dec1 = np.zeros((len(years),1))
    pred_SEAS5_Dec1[:,0] = SEAS5_Dec_Ta_mean_em[0,:]  # Dec. Ta - Lead = 0
    pred_SEAS5_Dec1_df = pd.DataFrame(pred_SEAS5_Dec1)
    X_dec1 = pred_SEAS5_Dec1_df
    y_pred_test1_dec1, model1_dec1, Xscaler1_dec1, mae_test1_dec1, acc7days_test1_dec1 = loo_eval(LinearRegression(),'mlr',X_dec1.iloc[1:,:],y.iloc[1:,:],years[1:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)
    # #Train MLR model with obs. (ERA5) for Nov. and Dec., then test using the SEAS5 forecasts
    # y_pred_test_dec1, model_dec1, Xscaler_dec1, mae_test_dec1, acc7days_test_dec1 = loo_eval(LinearRegression(),'mlr',X.iloc[1:,:],y.iloc[1:,:],years[1:],X_frcst=X_dec1.iloc[1:,:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)

    # Make prediction with trained model:
    if norm_MLR:
        y_pred1_nov2022 = model1_nov1.predict(Xscaler1_nov1.transform(pd.DataFrame(np.expand_dims(np.array(pred_nov2022),0))))
        y_pred1_dec2022 = model1_dec1.predict(Xscaler1_dec1.transform(pd.DataFrame(np.expand_dims(np.array(pred_dec2022),0))))
    else:
        y_pred1_nov2022 = model1_nov1.predict(pd.DataFrame(np.expand_dims(np.array(pred_nov2022),0)))
        y_pred1_dec2022 = model1_dec1.predict(pd.DataFrame(np.expand_dims(np.array(pred_dec2022),0)))

    print(y_pred1_nov2022)
    print(y_pred1_dec2022)

    #%% Make predictions: Dec. Ta + Dec. snow + Nov. snow

    # Select model:
    pred = ['Dec. Avg. Ta_mean','Nov. Tot. snowfall','Dec. Tot. snowfall']
    pred_nov2022 = [SEAS5DecTa2022_lead1, SEAS5Novsnowfall2022_lead0, SEAS5Decsnowfall2022_lead1]
    pred_dec2022 = [SEAS5DecTa2022_lead0, Novsnowfall2022, SEAS5Decsnowfall2022_lead0]

    # Perfect forecast model:
    X, y = pred_df[pred], pd.DataFrame(avg_freezeup_doy,columns=['Avg. FUD DOY'])
    y_intuition,_,_,mae_perfect_test_nov1,acc7days_perfect_test_nov1 = loo_eval(LinearRegression(),'mlr',X,y,years,normalize=norm_MLR,show_coeff=False,plot=False,verbose=verbose)

    # Train models with past forecasts:
    pred_SEAS5_Nov1 = np.zeros((len(years),3))
    pred_SEAS5_Nov1[:,0] = SEAS5_Dec_Ta_mean_em[1,:] # Dec. Ta - Lead = 1
    pred_SEAS5_Nov1[:,1] = SEAS5_Nov_snowfall_em[0,:] # Nov. snowfall - Lead = 0
    pred_SEAS5_Nov1[:,2] = SEAS5_Dec_snowfall_em[1,:] # Dec. snowfall - Lead = 1
    pred_SEAS5_Nov1_df = pd.DataFrame(pred_SEAS5_Nov1)
    X_nov1 = pred_SEAS5_Nov1_df
    y_pred_test_nov1, model_nov1, Xscaler_nov1, mae_test_nov1, acc7days_test_nov1 = loo_eval(LinearRegression(),'mlr',X_nov1.iloc[1:,:],y.iloc[1:,:],years[1:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)
    # #Train MLR model with obs. (ERA5) for Nov. and Dec., then test using the SEAS5 forecasts
    # y_pred_test_nov1, model_nov1, Xscaler_nov1, mae_test_nov1, acc7days_test_nov1 = loo_eval(LinearRegression(),'mlr',X.iloc[1:,:],y.iloc[1:,:],years[1:],X_frcst=X_nov1.iloc[1:,:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)

    pred_SEAS5_Dec1 = np.zeros((len(years),3))
    pred_SEAS5_Dec1[:,0] = SEAS5_Dec_Ta_mean_em[0,:]  # Dec. Ta - Lead = 0
    pred_SEAS5_Dec1[:,1] = pred_df['Nov. Tot. snowfall'] # Nov. snowfall - Obs.
    pred_SEAS5_Dec1[:,2] = SEAS5_Dec_snowfall_em[0,:]# Dec. snowfall - Lead = 0
    pred_SEAS5_Dec1_df = pd.DataFrame(pred_SEAS5_Dec1)
    X_dec1 = pred_SEAS5_Dec1_df
    y_pred_test_dec1, model_dec1, Xscaler_dec1, mae_test_dec1, acc7days_test_dec1 = loo_eval(LinearRegression(),'mlr',X_dec1.iloc[1:,:],y.iloc[1:,:],years[1:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)
    # #Train MLR model with obs. (ERA5) for Nov. and Dec., then test using the SEAS5 forecasts
    # y_pred_test_dec1, model_dec1, Xscaler_dec1, mae_test_dec1, acc7days_test_dec1 = loo_eval(LinearRegression(),'mlr',X.iloc[1:,:],y.iloc[1:,:],years[1:],X_frcst=X_dec1.iloc[1:,:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)

    # Make prediction with trained model:
    if norm_MLR:
        y_pred_nov2022 = model_nov1.predict(Xscaler_nov1.transform(pd.DataFrame(np.expand_dims(np.array(pred_nov2022),0))))
        y_pred_dec2022 = model_dec1.predict(Xscaler_dec1.transform(pd.DataFrame(np.expand_dims(np.array(pred_dec2022),0))))
    else:
        y_pred_nov2022 = model_nov1.predict(pd.DataFrame(np.expand_dims(np.array(pred_nov2022),0)))
        y_pred_dec2022 = model_dec1.predict(pd.DataFrame(np.expand_dims(np.array(pred_dec2022),0)))

    print(y_pred_nov2022)
    print(y_pred_dec2022)

    #%% Make predictions: Dec. Ta + Nov. snow + Sep. SW

    # Select model:
    pred = ['Dec. Avg. Ta_mean','Nov. Tot. snowfall','Sep. Avg. SW down (sfc)']
    pred_nov2022 = [SEAS5DecTa2022_lead1, SEAS5Novsnowfall2022_lead0, SeptSW2022]
    pred_dec2022 = [SEAS5DecTa2022_lead0, Novsnowfall2022, SeptSW2022]

    # Perfect forecast model:
    X, y = pred_df[pred], pd.DataFrame(avg_freezeup_doy,columns=['Avg. FUD DOY'])
    y_perf3,_,_,mae_perfect_test3_nov1,acc7days_perfect_test3_nov1 = loo_eval(LinearRegression(),'mlr',X,y,years,normalize=norm_MLR,show_coeff=False,plot=False,verbose=False)

    # Train model with past forecasts:
    pred_SEAS5_Nov1 = np.zeros((len(years),3))
    pred_SEAS5_Nov1[:,0] = SEAS5_Dec_Ta_mean_em[1,:] # Dec. Ta - Lead = 1
    pred_SEAS5_Nov1[:,1] = SEAS5_Nov_snowfall_em[0,:] # Nov. snowfall - Lead = 0
    pred_SEAS5_Nov1[:,2] = pred_df['Sep. Avg. SW down (sfc)'] # Sep. SW - Obs.
    pred_SEAS5_Nov1_df = pd.DataFrame(pred_SEAS5_Nov1)
    X_nov1 = pred_SEAS5_Nov1_df
    y_pred_test3_nov1, model3_nov1, Xscaler3_nov1, mae_test3_nov1, acc7days_test3_nov1 = loo_eval(LinearRegression(),'mlr',X_nov1.iloc[1:,:],y.iloc[1:,:],years[1:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)
    # #Train MLR model with obs. (ERA5) for Nov. and Dec., then test using the SEAS5 forecasts
    # y_pred_test3_nov1, model3_nov1, Xscaler3_nov1, mae_test3_nov1, acc7days_test3_nov1 = loo_eval(LinearRegression(),'mlr',X.iloc[1:,:],y.iloc[1:,:],years[1:],X_frcst=X_nov1.iloc[1:,:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)

    pred_SEAS5_Dec1 = np.zeros((len(years),3))
    pred_SEAS5_Dec1[:,0] = SEAS5_Dec_Ta_mean_em[0,:] # Dec. Ta - Lead = 0
    pred_SEAS5_Dec1[:,1] = pred_df['Nov. Tot. snowfall'] # Nov. snowfall - Obs.
    pred_SEAS5_Dec1[:,2] = pred_df['Sep. Avg. SW down (sfc)'] # Sep. SW - Obs.
    pred_SEAS5_Dec1_df = pd.DataFrame(pred_SEAS5_Dec1)
    X_dec1 = pred_SEAS5_Dec1_df
    y_pred_test3_dec1, model3_dec1, Xscaler3_dec1, mae_test3_dec1, acc7days_test3_dec1 = loo_eval(LinearRegression(),'mlr',X_dec1.iloc[1:,:],y.iloc[1:,:],years[1:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)
    # #Train MLR model with obs. (ERA5) for Nov. and Dec., then test using the SEAS5 forecasts
    # y_pred_test3_dec1, model3_dec1, Xscaler3_dec1, mae_test3_dec1, acc7days_test3_dec1 = loo_eval(LinearRegression(),'mlr',X.iloc[1:,:],y.iloc[1:,:],years[1:],X_frcst=X_dec1.iloc[1:,:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)

    # Make prediction with trained model:
    if norm_MLR:
        y_pred3_nov2022 = model3_nov1.predict(Xscaler3_nov1.transform(pd.DataFrame(np.expand_dims(np.array(pred_nov2022),0))))
        y_pred3_dec2022 = model3_dec1.predict(Xscaler3_dec1.transform(pd.DataFrame(np.expand_dims(np.array(pred_dec2022),0))))
    else:
        y_pred3_nov2022 = model3_nov1.predict(pd.DataFrame(np.expand_dims(np.array(pred_nov2022),0)))
        y_pred3_dec2022 = model3_dec1.predict(pd.DataFrame(np.expand_dims(np.array(pred_dec2022),0)))

    print(y_pred3_nov2022)
    print(y_pred3_dec2022)


    #%% Make predictions: Sep. OR Level + Dec. Ta + Nov. Ta + Sep. SW + Oct. SH
    pred = ['Sep. Avg. level Ottawa River','Dec. Avg. Ta_mean','Nov. Avg. Ta_mean','Sep. Avg. SW down (sfc)','Oct. Avg. SH (sfc)']
    pred_nov2022 = [SeptORlevel2022, SEAS5DecTa2022_lead1, SEAS5NovTa2022_lead0, SeptSW2022, OctSH2022]
    pred_dec2022 = [SeptORlevel2022, SEAS5DecTa2022_lead0, NovTamean2022, SeptSW2022, OctSH2022]

    # Perfect forecast model:
    X, y = pred_df[pred], pd.DataFrame(avg_freezeup_doy,columns=['Avg. FUD DOY'])
    y_perf5,_,_,mae_perfect_test5_nov1,acc7days_perfect_test5_nov1= loo_eval(LinearRegression(),'mlr',X,y,years,normalize=norm_MLR,show_coeff=False,plot=False,verbose=False)

    # Train model with past forecasts for Nov. and Dec.:
    pred_Nov1 = np.zeros((len(years),5))
    pred_Nov1[:,0] = pred_df['Sep. Avg. level Ottawa River'] # Sep. Level Ottawa River - Obs.
    pred_Nov1[:,1] = SEAS5_Dec_Ta_mean_em[1,:]  # Dec. Ta - Lead = 1
    pred_Nov1[:,2] = SEAS5_Nov_Ta_mean_em[0,:]  # Nov. Ta - Lead = 0
    pred_Nov1[:,3] = pred_df['Sep. Avg. SW down (sfc)'] # Sep. SW - Obs.
    pred_Nov1[:,4] = pred_df['Oct. Avg. SH (sfc)'] # Oct. SH - Obs.
    pred_Nov1_df = pd.DataFrame(pred_Nov1)
    X_nov1 = pred_Nov1_df
    y_pred_test5a_nov1, model5a_nov1, Xscaler5a_nov1,mae_test5a_nov1,acc7days_test5a_nov1 = loo_eval(LinearRegression(),'mlr',X_nov1.iloc[1:,:],y.iloc[1:,:],years[1:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)
    # #Train MLR model with obs. (ERA5) for Nov. and Dec., then test using the SEAS5 forecasts
    # y_pred_test5a_nov1, model5a_nov1, Xscaler5a_nov1, mae_test5a_nov1, acc7days_test5a_nov1 = loo_eval(LinearRegression(),'mlr',X.iloc[1:,:],y.iloc[1:,:],years[1:],X_frcst=X_nov1.iloc[1:,:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)

    pred_Dec1 = np.zeros((len(years),5))
    pred_Dec1[:,0] = pred_df['Sep. Avg. level Ottawa River'] # Sep. Level Ottawa River - Obs.
    pred_Dec1[:,1] = SEAS5_Dec_Ta_mean_em[0,:]  # Dec. Ta - Lead = 0
    pred_Dec1[:,2] = pred_df['Nov. Avg. Ta_mean']  # Nov. Ta - Obs.
    pred_Dec1[:,3] = pred_df['Sep. Avg. SW down (sfc)'] # Sep. SW - Obs.
    pred_Dec1[:,4] = pred_df['Oct. Avg. SH (sfc)'] # Oct. SH - Obs.
    pred_Dec1_df = pd.DataFrame(pred_Dec1)
    X_dec1 = pred_Dec1_df
    y_pred_test5a_dec1, model5a_dec1, Xscaler5a_dec1,mae_test5a_dec1,acc7days_test5a_dec1 = loo_eval(LinearRegression(),'mlr',X_dec1.iloc[1:,:],y.iloc[1:,:],years[1:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)
    # #Train MLR model with obs. (ERA5) for Nov. and Dec., then test using the SEAS5 forecasts
    # y_pred_test5a_dec1, model5a_dec1, Xscaler5a_dec1, mae_test5a_dec1, acc7days_testa_dec1 = loo_eval(LinearRegression(),'mlr',X.iloc[1:,:],y.iloc[1:,:],years[1:],X_frcst=X_dec1.iloc[1:,:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)


    # Make prediction with trained model:
    if norm_MLR:
        y_pred5a_nov2022 = model5a_nov1.predict(Xscaler5a_nov1.transform(pd.DataFrame(np.expand_dims(np.array(pred_nov2022),0))))
        y_pred5a_dec2022 = model5a_dec1.predict(Xscaler5a_dec1.transform(pd.DataFrame(np.expand_dims(np.array(pred_dec2022),0))))
    else:
        y_pred5a_nov2022 = model5a_nov1.predict(pd.DataFrame(np.expand_dims(np.array(pred_nov2022),0)))
        y_pred5a_dec2022 = model5a_dec1.predict(pd.DataFrame(np.expand_dims(np.array(pred_dec2022),0)))

    print(y_pred5a_nov2022)
    print(y_pred5a_dec2022)


    #%% Make predictions: Sep. OR Level + Dec. Ta + Sep. SW + Oct. SH
    pred = ['Sep. Avg. level Ottawa River','Dec. Avg. Ta_mean','Sep. Avg. SW down (sfc)','Oct. Avg. SH (sfc)']
    pred_nov2022 = [SeptORlevel2022, SEAS5DecTa2022_lead1, SeptSW2022, OctSH2022]
    pred_dec2022 = [SeptORlevel2022, SEAS5DecTa2022_lead0, SeptSW2022, OctSH2022]

    # Perfect forecast model:
    X, y = pred_df[pred], pd.DataFrame(avg_freezeup_doy,columns=['Avg. FUD DOY'])
    y_perf4,_,_,mae_perfect_test4_nov1,acc7days_perfect_test4_nov1 = loo_eval(LinearRegression(),'mlr',X,y,years,normalize=norm_MLR,show_coeff=False,plot=False,verbose=False)

    # Train model with past forecasts for Nov. and Dec.:
    pred_Nov1 = np.zeros((len(years),4))
    pred_Nov1[:,0] = pred_df['Sep. Avg. level Ottawa River'] # Sep. Level Ottawa River - Obs.
    pred_Nov1[:,1] = SEAS5_Dec_Ta_mean_em[1,:]  # Dec. Ta - Lead = 1
    pred_Nov1[:,2] = pred_df['Sep. Avg. SW down (sfc)'] # Sep. SW - Obs.
    pred_Nov1[:,3] = pred_df['Oct. Avg. SH (sfc)'] # Oct. SH - Obs.
    pred_Nov1_df = pd.DataFrame(pred_Nov1)
    X_nov1 = pred_Nov1_df
    y_pred_test4_nov1, model4_nov1, Xscaler4_nov1,mae_test4_nov1,acc7days_test4_nov1 = loo_eval(LinearRegression(),'mlr',X_nov1.iloc[1:,:],y.iloc[1:,:],years[1:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)
    # #Train MLR model with obs. (ERA5) for Nov. and Dec., then test using the SEAS5 forecasts
    # y_pred_test4_nov1, model4_nov1, Xscaler4_nov1, mae_test4_nov1, acc7days_test4_nov1 = loo_eval(LinearRegression(),'mlr',X.iloc[1:,:],y.iloc[1:,:],years[1:],X_frcst=X_nov1.iloc[1:,:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)

    pred_Dec1 = np.zeros((len(years),4))
    pred_Dec1[:,0] = pred_df['Sep. Avg. level Ottawa River'] # Sep. Level Ottawa River - Obs.
    pred_Dec1[:,1] = SEAS5_Dec_Ta_mean_em[0,:]  # Dec. Ta - Lead = 0
    pred_Dec1[:,2] = pred_df['Sep. Avg. SW down (sfc)'] # Sep. SW - Obs.
    pred_Dec1[:,3] = pred_df['Oct. Avg. SH (sfc)'] # Oct. SH - Obs.
    pred_Dec1_df = pd.DataFrame(pred_Dec1)
    X_dec1 = pred_Dec1_df
    y_pred_test4_dec1, model4_dec1, Xscaler4_dec1,mae_test4_dec1,acc7days_test4_dec1 = loo_eval(LinearRegression(),'mlr',X_dec1.iloc[1:,:],y.iloc[1:,:],years[1:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)
    # #Train MLR model with obs. (ERA5) for Nov. and Dec., then test using the SEAS5 forecasts
    # y_pred_test4_dec1, model4_dec1, Xscaler4_dec1, mae_test4_dec1, acc7days_test_dec1 = loo_eval(LinearRegression(),'mlr',X.iloc[1:,:],y.iloc[1:,:],years[1:],X_frcst=X_dec1.iloc[1:,:],normalize=norm_MLR,show_coeff=False,plot=False,verbose = verbose)

    # # Make prediction with trained model:
    if norm_MLR:
        y_pred4_nov2022 = model4_nov1.predict(Xscaler4_nov1.transform(pd.DataFrame(np.expand_dims(np.array(pred_nov2022),0))))
        y_pred4_dec2022 = model4_dec1.predict(Xscaler4_dec1.transform(pd.DataFrame(np.expand_dims(np.array(pred_dec2022),0))))
    else:
        y_pred4_nov2022 = model4_nov1.predict(pd.DataFrame(np.expand_dims(np.array(pred_nov2022),0)))
        y_pred4_dec2022 = model4_dec1.predict(pd.DataFrame(np.expand_dims(np.array(pred_dec2022),0)))

    print(y_pred4_nov2022)
    print(y_pred4_dec2022)


    #%%
    fig,ax = plt.subplots()
    ax.plot(years[:],y, 'o-', color='k')
    ax.plot(years[:],y_perf5, 'o-', label = 'Best 5 pred. (MAE = '+ str(np.round(mae_perfect_test5_nov1,1)) +' days, 7-d Acc. = '+ str(np.int(np.round(acc7days_perfect_test5_nov1*100))) +'%)')
    ax.plot(years[:],y_perf4, 'o-', label = 'Best 4 pred. (MAE = '+ str(np.round(mae_perfect_test4_nov1,1)) +' days, 7-d Acc. = '+ str(np.int(np.round(acc7days_perfect_test4_nov1*100))) +'%)')
    ax.plot(years[:],y_perf3, 'o-', label = 'Best 3 pred. (MAE = '+ str(np.round(mae_perfect_test3_nov1,1)) +' days, 7-d Acc. = '+ str(np.int(np.round(acc7days_perfect_test3_nov1*100))) +'%)')
    ax.plot(years[:],y_intuition, 'o-', label = 'Dec Ta, Dec. snow, Nov. snow (MAE = '+ str(np.round(mae_perfect_test_nov1,1)) +' days, 7-d Acc. = '+ str(np.int(np.round(acc7days_perfect_test_nov1*100))) +'%)')
    ax.plot(years[:],y1, 'o-', label = 'Dec Ta (MAE = '+ str(np.round(mae_perfect_test1_nov1,1)) +' days, 7-d Acc. = '+ str(np.int(np.round(acc7days_perfect_test1_nov1*100))) +'%)')

    ax.plot(years[:],np.ones(len(years))*(355), '--', color='gray')
    ax.fill_between(years[:],np.ones(len(years))*(350),np.ones(len(years))*(360), color='gray', alpha=0.1)

    ax.set_ylabel('FUD')
    ax.set_xlabel('Year')
    ax.set_title('Perfect forecast')
    ax.legend()

#%%
    fig,ax = plt.subplots()
    ax.plot(years[:],y, 'o-', color='k')

    plt.plot(years[1:],y_pred_test5a_nov1, 'o-', color = plt.get_cmap('tab20')(0), label='Best 5 pred. (MAE = '+ str(np.round(mae_test5a_nov1,1)) +' days, 7-d Acc. = '+ str(np.int(np.round(acc7days_test5a_nov1*100))) +'%)')
    plt.plot(2022,y_pred5a_nov2022,'*', markersize = 15,color = plt.get_cmap('tab20')(1))

    plt.plot(years[1:],y_pred_test4_nov1, 'o-', color = plt.get_cmap('tab20')(2), label='Best 4 pred. (MAE = '+ str(np.round(mae_test4_nov1,1)) +' days, 7-d Acc. = '+ str(np.int(np.round(acc7days_test4_nov1*100))) +'%)')
    plt.plot(2022,y_pred4_nov2022,'*', markersize = 15,color = plt.get_cmap('tab20')(3))

    plt.plot(years[1:],y_pred_test3_nov1, 'o-', color = plt.get_cmap('tab20')(4), label='Best 3 pred. (MAE = '+ str(np.round(mae_test3_nov1,1)) +' days, 7-d Acc. = '+ str(np.int(np.round(acc7days_test3_nov1*100))) +'%)')
    plt.plot(2022,y_pred3_nov2022,'*', markersize = 15,color = plt.get_cmap('tab20')(5))

    ax.plot(years[1:],y_pred_test_nov1, 'o-', color = plt.get_cmap('tab20')(6), label= 'Dec Ta, Dec. snow, Nov. snow (MAE = '+ str(np.round(mae_test_nov1,1)) +' days, 7-d Acc. = '+ str(np.int(np.round(acc7days_test_nov1*100))) +'%)')
    ax.plot(2022,y_pred_nov2022,'*', markersize = 15,color = plt.get_cmap('tab20')(7))

    ax.plot(years[1:],y_pred_test1_nov1, 'o-', color = plt.get_cmap('tab20')(8), label= 'Dec Ta (MAE = '+ str(np.round(mae_test1_nov1,1)) +' days, 7-d Acc. = '+ str(np.int(np.round(acc7days_test1_nov1*100))) +'%)')
    ax.plot(2022,y_pred1_nov2022,'*', markersize = 15,color = plt.get_cmap('tab20')(9))

    ax.plot(np.arange(years[0],2024),np.ones(len(np.arange(years[0],2024)))*(355), '--', color='gray')
    ax.fill_between(np.arange(years[0],2024),np.ones(len(np.arange(years[0],2024)))*(350),np.ones(len(np.arange(years[0],2024)))*(360), color='gray', alpha=0.1)

    ax.set_ylabel('FUD')
    ax.set_xlabel('Year')
    ax.set_title('Nov. 1st')
    ax.legend()

    #%%
    fig,ax = plt.subplots()
    ax.plot(years[:],y, 'o-', color='k')

    plt.plot(years[1:],y_pred_test5a_dec1, 'o-', color = plt.get_cmap('tab20')(0), label='Best 5 pred. (MAE = '+ str(np.round(mae_test5a_dec1,1)) +' days, 7-d Acc. = '+ str(np.int(np.round(acc7days_test5a_dec1*100))) +'%)')
    plt.plot(2022,y_pred5a_dec2022,'*', markersize = 15,color = plt.get_cmap('tab20')(1))

    plt.plot(years[1:],y_pred_test4_dec1, 'o-', color = plt.get_cmap('tab20')(2), label='Best 4 pred. (MAE = '+ str(np.round(mae_test4_dec1,1)) +' days, 7-d Acc. = '+ str(np.int(np.round(acc7days_test4_dec1*100))) +'%)')
    plt.plot(2022,y_pred4_dec2022,'*', markersize = 15,color = plt.get_cmap('tab20')(3))

    plt.plot(years[1:],y_pred_test3_dec1, 'o-', color = plt.get_cmap('tab20')(4), label='Best 3 pred. (MAE = '+ str(np.round(mae_test3_dec1,1)) +' days, 7-d Acc. = '+ str(np.int(np.round(acc7days_test3_dec1*100))) +'%)')
    plt.plot(2022,y_pred3_dec2022,'*', markersize = 15,color = plt.get_cmap('tab20')(5))

    ax.plot(years[1:],y_pred_test_dec1, 'o-', color = plt.get_cmap('tab20')(6), label= 'Dec Ta, Dec. snow, Nov. snow (MAE = '+ str(np.round(mae_test_dec1,1)) +' days, 7-d Acc. = '+ str(np.int(np.round(acc7days_test_dec1*100))) +'%)')
    ax.plot(2022,y_pred_dec2022,'*', markersize = 15,color = plt.get_cmap('tab20')(7))

    ax.plot(years[1:],y_pred_test1_dec1, 'o-', color = plt.get_cmap('tab20')(8), label= 'Dec Ta (MAE = '+ str(np.round(mae_test1_dec1,1)) +' days, 7-d Acc. = '+ str(np.int(np.round(acc7days_test1_dec1*100))) +'%)')
    ax.plot(2022,y_pred1_dec2022,'*', markersize = 15,color = plt.get_cmap('tab20')(9))

    ax.plot(np.arange(years[0],2024),np.ones(len(np.arange(years[0],2024)))*(355), '--', color='gray')
    ax.fill_between(np.arange(years[0],2024),np.ones(len(np.arange(years[0],2024)))*(350),np.ones(len(np.arange(years[0],2024)))*(360), color='gray', alpha=0.1)

    ax.set_ylabel('FUD')
    ax.set_xlabel('Year')
    ax.set_title('Dec. 1st')
    ax.legend()











