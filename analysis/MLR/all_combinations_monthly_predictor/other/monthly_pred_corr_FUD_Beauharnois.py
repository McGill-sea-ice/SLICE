#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:36:46 2022

@author: Amelie
"""
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
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
fdir_r = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/CMC_GHRSST/'
fdir_p = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/CMC_GHRSST/'

verbose = False
p_critical = 0.01

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1
time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)
years = np.arange(1980,2022)

freezeup_type = 'first_ice'
# freezeup_type = 'stable_ice'
detrend = False
anomaly = 'linear'

#%%
# Load FUD data from Hydro-Quebec (Beauharnois)
data = np.load('/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/freezeup_dates_HQ/freezeup_HQ_BeauharnoisCanal.npz')
fi = data['freezeup_fi'][:]
si = data['freezeup_si'][:]

fi = fi[~np.isnan(fi)]
si = si[~np.isnan(si)]

# Convert days since ... into DOY
years_HQ = np.arange(1960,2020)
doy_fi_HQ = np.zeros((len(fi)))*np.nan
doy_si_HQ = np.zeros((len(si)))*np.nan
avg_freezeup_doy = np.zeros((len(years)))*np.nan
for i in range(len(fi)):
    date_FUD_fi = date_ref + dt.timedelta(days=int(fi[i]))
    if date_FUD_fi.year == years_HQ[i]:
        doy_FUD_fi = (date_FUD_fi-dt.date(years_HQ[i],1,1)).days + 1
    else:
        doy_FUD_fi = (365 + calendar.isleap(years_HQ[i]) +
                      (date_FUD_fi-dt.date(years_HQ[i]+1,1,1)).days + 1)
    doy_fi_HQ[i] = doy_FUD_fi

    date_FUD_si = date_ref + dt.timedelta(days=int(si[i]))
    if date_FUD_si.year == years_HQ[i]:
        doy_FUD_si = (date_FUD_si-dt.date(years_HQ[i],1,1)).days + 1
    else:
        doy_FUD_si = (365 + calendar.isleap(years_HQ[i]) +
                      (date_FUD_si-dt.date(years_HQ[i]+1,1,1)).days + 1)
    doy_si_HQ[i] = doy_FUD_si

    if years_HQ[i] in years:
        if freezeup_type == 'first_ice':
            avg_freezeup_doy[np.where(years==years_HQ[i])[0][0]] = doy_fi_HQ[i]
        if freezeup_type == 'stable_ice':
            avg_freezeup_doy[np.where(years==years_HQ[i])[0][0]] = doy_si_HQ[i]

# Check if trend is significant:
trend_model = sm.OLS(avg_freezeup_doy, sm.add_constant(years,has_constant='skip'), missing='drop').fit()
if trend_model.pvalues[1] <= 0.05:
    print('*** Warning ***\n Trend for FUD is significant. Consider detrending.')

if detrend:
    if anomaly == 'linear':
        avg_freezeup_doy, [m,b] = detrend_ts(avg_freezeup_doy,years,anomaly)
    if anomaly == 'mean':
        avg_freezeup_doy, mean = detrend_ts(avg_freezeup_doy,years,anomaly)

#TEST: Plot autocorrelation function for FUD and Dec. Twater time series
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.tsa.stattools import acf

# plot_acf(avg_freezeup_doy,missing='drop',alpha=0.01)
# tacf_doy = acf(avg_freezeup_doy,missing='drop')


#%%
# Load monthly predictor data
fpath_mp = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/monthly_predictors/'
monthly_pred = np.load(fpath_mp+'monthly_vars_Beauharnois'+detrend*'_detrended'+'.npz')
monthly_pred_data = monthly_pred['data']
pred_names = monthly_pred['labels']

# Replace zeros with nan for snowfall, FDD, TDD:
ivar_snowfall = np.where(pred_names == 'Tot. snowfall')[0][0]
monthly_pred_data[ivar_snowfall,:,:][monthly_pred_data[ivar_snowfall,:,:] == 0] = np.nan

ivar_FDD = np.where(pred_names == 'Tot. FDD')[0][0]
monthly_pred_data[ivar_FDD,:,:][monthly_pred_data[ivar_FDD,:,:] == 0] = np.nan

ivar_TDD = np.where(pred_names == 'Tot. TDD')[0][0]
monthly_pred_data[ivar_TDD,:,:][monthly_pred_data[ivar_TDD,:,:] == 0] = np.nan

#%%
# Find correlation coefficient and pvalue for all monthly variables
# with FUD time series
nvars = monthly_pred_data.shape[0]
corr_coeff = np.zeros((nvars, 12))*np.nan
pvals = np.zeros((nvars, 12))*np.nan

for ivar in range(nvars):
    for imonth in range(12):
        if np.sum(np.isnan(monthly_pred_data[ivar,:,imonth])) < 0.3*(np.sum(~np.isnan(avg_freezeup_doy))):
            lm = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(monthly_pred_data[ivar,:,imonth]),has_constant='skip'), missing='drop').fit()
            if lm.params[1] < 0 :
                corr_coeff[ivar,imonth] = -1*np.sqrt(lm.rsquared)
            else:
                corr_coeff[ivar,imonth] = np.sqrt(lm.rsquared)
            pvals[ivar,imonth] = lm.f_pvalue


corrFUD_df = pd.DataFrame(corr_coeff.T,columns=pred_names)
pvalFUD_df = pd.DataFrame(pvals.T,columns=pred_names)

signif_01_df = corrFUD_df.copy()
signif_01_df[pvalFUD_df > 0.01] = np.nan
signif_05_df = corrFUD_df.copy()
signif_05_df[pvalFUD_df > 0.05] = np.nan
