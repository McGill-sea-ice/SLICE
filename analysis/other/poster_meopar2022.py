#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 18:57:30 2022

@author: Amelie
"""
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import calendar
from netCDF4 import Dataset
import statsmodels.api as sm
#%%

years = np. array([1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002,
                   2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
                   2014, 2015, 2016, 2017, 2018, 2019])

# OBSERVATIONS (LONGUEUIL 1992-2019)
avg_freezeup_doy = np.array([360., 358., 364., 342., 365., 350., 365., 360., 343., 367., 339.,
                               348., 354., 350., 381., 341., 347., 352., 357., 364., 358., 347.,
                               365., 371., 351., 348., 356., 354.])
FUD_clim = 355.0
cat_obs = np.array([0., 0.,  1., -1.,  1., -1.,  1.,  0., -1.,  1., -1., -1.,  0., -1.,1.,
                   -1., -1.,  0.,  0.,  1.,  0., -1.,  1.,  1.,  0., -1.,  0., 0.,])
p66_obs = 360.0
p33_obs = 350.0

# CATEGORICAL BASELINE
cb_acc = 35.71


# CLIM FUD BASELINE
clim_pred = np.zeros(len(years))*np.nan
cat_clim = np.zeros(len(years))*np.nan
for iyr,year in enumerate(years[:]):
    if ~np.isnan(avg_freezeup_doy[iyr]):
        # REMOVE THE YEAR BEING FORECASTED FROM DATA FOR MAKING CLIMATOLOGY
        FUD_in = avg_freezeup_doy.copy()
        FUD_in[iyr] = np.nan
        # COMPUTE TW CLIMATOLOGY AND AVERAGE OBSERVED FUD FOR ALL OTHER YEARS
        mean_obs_FUD = (np.nanmean(FUD_in))
        std_obs_FUD = (np.nanstd(FUD_in))
        clim_pred[iyr] = np.floor(mean_obs_FUD)
        if clim_pred[iyr] <= p33_obs:
            cat_clim[iyr] = -1
        elif clim_pred[iyr] > p66_obs:
            cat_clim[iyr] = 1
        else:
            cat_clim[iyr] = 0
clim_mae = np.nanmean(np.abs(clim_pred-avg_freezeup_doy))
clim_rmse = np.sqrt(np.nanmean( (clim_pred-avg_freezeup_doy)**2.))
model = sm.OLS(avg_freezeup_doy, sm.add_constant(clim_pred,has_constant='skip'), missing='drop').fit()
clim_rsqr = model.rsquared
clim_acc7 = ((np.sum(np.abs(avg_freezeup_doy-clim_pred) <= 7)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
clim_acc = (np.sum(cat_clim == cat_obs)/(np.sum(~np.isnan(avg_freezeup_doy))))*100



# MLR
mlr_predictors = ['Dec. Avg. Ta_mean', 'Nov. Tot. snowfall', 'Dec. Tot. snowfall']
mlr_pred = np.array([359.40951606, 352.67324834, 362.18307155, 336.60483673,
                       360.51048258, 350.0055085 , 368.2963757 , 364.13916396,
                       344.47439593, 367.64450276, 353.20552084, 354.95435474,
                       352.06631262, 352.02763068, 369.13267482, 344.45558016,
                       344.44778823, 355.96208168, 356.47388826, 364.49966777,
                       356.52550888, 346.64313222, 357.01375152, 379.37235905,
                       353.31352317, 348.71312248, 349.37760737, 354.40419873])

cat_mlr = np.zeros(len(years))*np.nan
for iyr in range(len(years)):
    if ~np.isnan(mlr_pred[iyr]):
        if mlr_pred[iyr] <= p33_obs:
            cat_mlr[iyr] = -1
        elif mlr_pred[iyr] > p66_obs:
            cat_mlr[iyr] = 1
        else:
            cat_mlr[iyr] = 0
mlr_mae = np.nanmean(np.abs(mlr_pred-avg_freezeup_doy))
mlr_rmse = np.sqrt(np.nanmean( (mlr_pred-avg_freezeup_doy)**2.))
model = sm.OLS(avg_freezeup_doy, sm.add_constant(mlr_pred,has_constant='skip'), missing='drop').fit()
mlr_rsqr = model.rsquared
mlr_acc7 = ((np.sum(np.abs(avg_freezeup_doy-mlr_pred) <= 7)/(np.sum(~np.isnan(avg_freezeup_doy)))))*100
mlr_acc = (np.sum(cat_mlr == cat_obs)/(np.sum(~np.isnan(avg_freezeup_doy))))*100

# ML - IDEAL FORECAST WITH SEAS5 TA_MEAN ONLY
ml_pred = np.array([[360., 362., 366., 343.,  np.nan, 358., 366., 361., 345.,  np.nan, 366.,
                     np.nan, 354., 351.,  np.nan, 346., 351., 354., 353.,  np.nan, 359., 348.,
                     np.nan,  np.nan, 366., 359., 348., 355.],
                   [359., 361., 366., 342., 364., 357., 363., 361., 347., 372., 365.,
                    373., 354., 349.,  np.nan, 347., 346., 355., 355., 368., 360., 348.,
                     np.nan,  np.nan, 367., 364., 348., 353.],
                   [359., 360., 365., 343., 365., 358., 363., 363., 347., 368., 367.,
                    372., 354., 350.,  np.nan, 348., 351., 358., 352., 369., 361., 350.,
                    371., 380., 367., 377., 359., 355.],
                   [359., 358., 365., 342., 365., 358., 362., 364., 344., 368., 370.,
                    372., 354., 348., 383., 346., 346., 356., 354., 367., 362., 350.,
                    370., 380., 366., 358., 348., 353.],
                   [359., 359., 365., 342., 367., 356., 363., 365., 346., 370., 346.,
                    372., 353., 350., 383., 343., 353., 357., 352., 369., 362., 349.,
                    370., 380., 370., 353., 358., 353.]])

ml_pred[np.isnan(ml_pred)] = FUD_clim

ml_mae = np.array([5.4       , 6.07407407, 6.37037037, 5.32142857, 4.57142857])
ml_rmse = np.array([ 8.06225775,  9.45946518, 10.28123065,  8.90224691,  6.94879229])
ml_rsqr = np.array([0.18700706, 0.18364894, 0.29678957, 0.47252545, 0.70432206])
ml_acc = np.array([0.48      , 0.62962963, 0.7037037 , 0.71428571, 0.75      ])*100
ml_acc7 = np. array([0.68      , 0.74074074, 0.77777778, 0.75      , 0.89285714])*100


#%%

istart_label = ['Nov. 3', 'Nov. 10', 'Nov. 17', 'Nov. 24', 'Dec. 1' ]
istart_plot = [0,1,2,3,4]

# fig_mae_ts,ax_mae_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
# fig_rsqr_ts,ax_rsqr_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
# fig_acc_ts,ax_acc_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
# fig_acc7_ts,ax_acc7_ts = plt.subplots(nrows = 1, ncols = 1,figsize=(4,3))
fig_metrics,ax_metrics = plt.subplots(nrows = 1, ncols = 4,figsize=(16,3), sharex = True)

ax_mae = ax_metrics[0] ;  #ax_mae.grid(axis='y')
ax_rsqr = ax_metrics[1] ; #ax_rsqr.grid(axis='y')
ax_acc = ax_metrics[2];   #ax_acc.grid(axis='y')
ax_acc7 = ax_metrics[3];  #ax_acc7.grid(axis='y')

color_ml = plt.get_cmap('tab20c')(0)
color_mlr = plt.get_cmap('Dark2')(5)
color_clim = 'k'
color_cb = 'gray'


ax_mae.plot(np.arange(len(istart_plot)),ml_mae[istart_plot],'o-',label='E-D', color=color_ml)
# ax_mae.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*mlr_mae,'o-',label='MLR', color=color_mlr)
ax_mae.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*clim_mae,'o-',color=color_clim,label='Climatology')
ax_mae.set_ylabel('MAE (days)')
ax_mae.set_xlabel('Forecast start date')
ax_mae.set_xticks(np.arange(len(istart_plot)))
ax_mae.set_xticklabels([istart_label[k] for k in istart_plot])
# ax_mae.legend()

ax_rsqr.plot(np.arange(len(istart_plot)),ml_rsqr[istart_plot],'o-',label='E-D', color=color_ml)
# ax_rsqr.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*mlr_rsqr,'o-',label='MLR', color=color_mlr)
ax_rsqr.set_ylabel('R$^{2}$')
ax_rsqr.set_xlabel('Forecast start date')
ax_rsqr.set_xticks(np.arange(len(istart_plot)))
ax_rsqr.set_xticklabels([istart_label[k] for k in istart_plot])
# ax_rsqr.legend()

ax_acc.plot(np.arange(len(istart_plot)),ml_acc[istart_plot],'o-',label='E-D', color=color_ml)
# ax_acc.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*mlr_acc,'o-',label='MLR', color=color_mlr)
ax_acc.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*cb_acc,'o-',color=color_cb,label='Categorical baseline')
ax_acc.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*clim_acc,'o-',color=color_clim,label='Climatology')
ax_acc.set_ylabel('Categorical Accuracy (%)')
ax_acc.set_xlabel('Forecast start date')
ax_acc.set_xticks(np.arange(len(istart_plot)))
ax_acc.set_xticklabels([istart_label[k] for k in istart_plot])
# ax_acc.legend()
# ax_acc.plot(np.arange(len(istart_plot)),ml_rmse[istart_plot],'o-',label='E-D', color=color_ml)
# ax_acc.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*mlr_rmse,'o-',label='MLR', color=color_mlr)
# ax_acc.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*clim_rmse,'o-',color=color_clim,label='Climatology')
# ax_acc.set_ylabel('RMSE (days)')
# ax_acc.set_xlabel('Forecast start date')
# ax_acc.set_xticks(np.arange(len(istart_plot)))
# ax_acc.set_xticklabels([istart_label[k] for k in istart_plot])

ax_acc7.plot(np.arange(len(istart_plot)),ml_acc7[istart_plot],'o-',label='E-D', color=color_ml)
# ax_acc7.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*mlr_acc7,'o-',label='MLR', color=color_mlr)
ax_acc7.plot(np.arange(len(istart_plot)),np.ones(len(istart_plot))*clim_acc7,'o-',color=color_clim,label='Climatology')
ax_acc7.set_ylabel('7-day Accuracy (%)')
ax_acc7.set_xlabel('Forecast start date')
ax_acc7.set_xticks(np.arange(len(istart_plot)))
ax_acc7.set_xticklabels([istart_label[k] for k in istart_plot])
# ax_acc7.legend()


fig_metrics.subplots_adjust(wspace=0.25,right=0.95,left=0.06,top=0.9,bottom =0.2)
# plt.savefig('/Users/Amelie/Dropbox/Postdoc/Projet_Fednav/metrics_fcst.png',transparent=True, dpi=900)

#%%

fig_ts,ax_ts = plt.subplots(nrows = 1, ncols = 1)

color_ml_dec1 = plt.get_cmap('tab20c')(0)
color_ml_nov1 = plt.get_cmap('tab20c')(2)
color_mlr = plt.get_cmap('Dark2')(5)

ax_ts.plot(years,avg_freezeup_doy,'o-',color='black', label = 'Observed')
ax_ts.plot(years,np.ones(len(years))*FUD_clim ,'--',color = 'gray',linewidth=1, label='Climatology')

# ax_ts.plot(years,mlr_pred,'o-',color=color_mlr,label='MLR (Dec. T$_{air}$, Dec. snowfall, Nov. snowfall)')
ax_ts.plot(years,ml_pred[0,:],'o-',color=color_ml_nov1,label='Encoder-Decoder - Nov. 1')
ax_ts.plot(years,ml_pred[4,:],'o-',color=color_ml_dec1,label='Encoder-Decoder - Dec. 1')
ax_ts.legend()
ax_ts.set_ylabel('FUD (day of year)')
ax_ts.set_xlabel('Year')
ax_ts.grid(linestyle=':')
ax_ts.set_ylim(334,386)
ax_ts.set_xlim(1989,2021)