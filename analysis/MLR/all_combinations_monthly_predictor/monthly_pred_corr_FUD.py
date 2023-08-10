#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:36:46 2022

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
import datetime as dt
import calendar
import matplotlib.pyplot as plt
import statsmodels.api as sm
import cartopy.crs as ccrs
import cartopy as cartopy
from functions import detect_FUD_from_Tw, detrend_ts
from functions_MLR import get_monthly_vars_from_daily

#%%
verbose = False
p_critical = 0.01

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


#%%
# Load Twater and FUD data
fp_p_Twater = local_path+'slice/data/processed/'
Twater_loc_list = ['Longueuil_updated']
station_type = 'cities'
freezeup_opt = 1
freezeup_doy, Twater = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False)
freezeup_doy[np.where(years == 2020)] = np.nan

Twater_mean = np.nanmean(Twater,axis=1)
Twater_mean = np.expand_dims(Twater_mean, axis=1)

# Average (and round) FUD from all locations:
avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
avg_freezeup_doy = np.round(avg_freezeup_doy)
# years = np.array(years[:-1])
# avg_freezeup_doy = avg_freezeup_doy[:-1]

#%%
# Get monthly mean Twater data
monthly_avg_Twater = get_monthly_vars_from_daily(Twater_mean,['Avg. Twater'],years,time,replace_with_nan=False)
monthly_avg_Twater = np.squeeze(monthly_avg_Twater)

Dec_Tw = monthly_avg_Twater[:,11]

#%%
# Load monthly predictor data
fpath_mp = local_path+'slice/data/monthly_predictors/'
monthly_pred = np.load(fpath_mp+'monthly_vars_Longueuil.npz')
monthly_pred_data = monthly_pred['data']
pred_names = monthly_pred['labels'].tolist()

# Replace zeros with nan for snowfall, FDD, TDD:
monthly_pred_data[5,:,:][monthly_pred_data[5,:,:] == 0] = np.nan
monthly_pred_data[10,:,:][monthly_pred_data[10,:,:] == 0] = np.nan
monthly_pred_data[11,:,:][monthly_pred_data[11,:,:] == 0] = np.nan

# New variable: Twater- Tair
ivar_Tamean = np.where(np.array(pred_names) == 'Avg. Ta_mean')[0][0]
Tw_minus_Tair =  monthly_avg_Twater[:,:] - monthly_pred_data[ivar_Tamean,:,:]
pred_names.append( 'Avg. Twater - Tair')

monthly_pred_data_tmp = monthly_pred_data.copy()
monthly_pred_data = np.zeros((len(pred_names),monthly_pred_data.shape[1],monthly_pred_data.shape[2]))*np.nan
monthly_pred_data[:-1,:,:] = monthly_pred_data_tmp
monthly_pred_data[-1,:,:] = Tw_minus_Tair

#%%
# Find correlation coefficient and pvalue for all monthly variables
# with FUD time series
nvars = monthly_pred_data.shape[0]
corr_coeff = np.zeros((nvars, 12))*np.nan
pvals = np.zeros((nvars, 12))*np.nan

for ivar in range(nvars):
    for imonth in range(12):
        if np.sum(np.isnan(monthly_pred_data[ivar,:,imonth])) < 0.25*(np.sum(~np.isnan(avg_freezeup_doy))):
            lm = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(monthly_pred_data[ivar,:,imonth]),has_constant='skip'), missing='drop').fit()
            if lm.params[1] < 0 :
                corr_coeff[ivar,imonth] = -1*np.sqrt(lm.rsquared)
            else:
                corr_coeff[ivar,imonth] = np.sqrt(lm.rsquared)
            pvals[ivar,imonth] = lm.f_pvalue


corrFUD_df = pd.DataFrame(corr_coeff.T,columns=pred_names)
pvalFUD_df = pd.DataFrame(pvals.T,columns=pred_names)


#%%
# Find correlation coefficient and pvalue for all monthly variables
# with average Twater in December
nvars = monthly_pred_data.shape[0]
corr_coeff_Tw = np.zeros((nvars, 12))*np.nan
pvals_Tw = np.zeros((nvars, 12))*np.nan

for ivar in range(nvars):
    for imonth in range(12):
        if np.sum(np.isnan(monthly_pred_data[ivar,:,imonth])) < 0.25*(np.sum(~np.isnan(Dec_Tw))):
            lm = sm.OLS(Dec_Tw,sm.add_constant(np.squeeze(monthly_pred_data[ivar,:,imonth]),has_constant='skip'), missing='drop').fit()
            if lm.params[1] < 0 :
                corr_coeff_Tw[ivar,imonth] = -1*np.sqrt(lm.rsquared)
            else:
                corr_coeff_Tw[ivar,imonth] = np.sqrt(lm.rsquared)
            pvals_Tw[ivar,imonth] = lm.f_pvalue


corrTw_df = pd.DataFrame(corr_coeff_Tw.T,columns=pred_names)
pvalTw_df = pd.DataFrame(pvals_Tw.T,columns=pred_names)

signif_01_df = corrFUD_df.copy()
signif_01_df[pvalFUD_df > 0.01] = np.nan
signif_05_df = corrFUD_df.copy()
signif_05_df[pvalFUD_df > 0.05] = np.nan


#%%
#TEST: Plot autocorrelation function for FUD and Dec. Twater time series
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.tsa.stattools import acf

# plot_acf(avg_freezeup_doy,missing='drop',alpha=0.01)
# tacf_doy = acf(avg_freezeup_doy,missing='drop')

# plot_acf(Dec_Tw,missing='drop',alpha=0.01)
# tacf_doy = acf(Dec_Tw,missing='drop')







