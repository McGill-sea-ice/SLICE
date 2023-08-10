#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:37:41 2022

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
import datetime as dt
import calendar
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from functions import ncdump
import os
import cartopy.crs as ccrs
from functions import detect_FUD_from_Tw,detect_FUD_from_Tw_clim
import statsmodels.api as sm

from functions import rolling_climo
#%%
fdir_r = local_path+'slice/data/raw/CMC_GHRSST/'
fdir_p = local_path+'slice/data/processed/CMC_GHRSST/'

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
# Twater_loc_list = ['Longueuil','Candiac','Atwater']
Twater_loc_list = ['Longueuil_updated']
station_type = 'cities'
freezeup_opt = 1
freezeup_doy, Twater = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False)
freezeup_doy[np.where(years == 2020)[0][0]] = np.nan

# Average (and round) FUD from all locations:
avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
avg_freezeup_doy = np.round(avg_freezeup_doy)
# avg_freezeup_doy = freezeup_doy[:,0]

# Average Twater from all locations:
avg_Twater = np.nanmean(Twater,axis=1)
avg_Twater_vars = np.expand_dims(avg_Twater, axis=1)
# avg_Twater = Twater[:,0]
# avg_Twater_vars = np.expand_dims(avg_Twater, axis=1)
avg_Twater_varnames = ['Avg. Twater']

#%%
# TEST USE KINGSTON WATER TEMP INSTEAD
# Tw_Kingston = np.load(local_path+'slice/data/processed/Twater_SLSMC/Twater_SLSMC_Kingston.npz', allow_pickle=True)
# Tw_Kingston = Tw_Kingston['Twater'][:,1]
# Tw_Kingston[14755:] = np.nan
# avg_Twater_vars = np.expand_dims(Tw_Kingston, axis=1)
# avg_Twater_varnames = ['Avg. Twater - Kingston']

# Tw_Iroquois = np.load(local_path+'slice/data/processed/Twater_SLSMC/Twater_SLSMC_Iroquois.npz', allow_pickle=True)
# Tw_Iroquois = Tw_Iroquois['Twater'][:,1]
# Tw_Iroquois[14755:] = np.nan
# avg_Twater_vars = np.expand_dims(Tw_Iroquois, axis=1)
# avg_Twater_varnames = ['Avg. Twater - Iroquois']

# Tw_Cornwall = np.load(local_path+'slice/data/processed/Twater_SLSMC/Twater_SLSMC_Cornwall.npz', allow_pickle=True)
# Tw_Cornwall = Tw_Cornwall['Twater'][:,1]
# Tw_Cornwall[14755:] = np.nan
# avg_Twater_vars = np.expand_dims(Tw_Cornwall, axis=1)
# avg_Twater_varnames = ['Avg. Twater - Cornwall']

# Tw_StLambert = np.load(local_path+'slice/data/processed/Twater_SLSMC/Twater_SLSMC_StLambert.npz', allow_pickle=True)
# Tw_StLambert = Tw_StLambert['Twater'][:,1]
# Tw_StLambert[14755:] = np.nan
# avg_Twater_vars = np.expand_dims(Tw_StLambert, axis=1)
# avg_Twater_varnames = ['Avg. Twater - StLambert']

#%%
# start_doy_arr    = [300,         307,       314,         321,         328,         335,         342,       349]
# start_doy_labels = ['Oct. 27th', 'Nov. 3rd', 'Nov. 10th', 'Nov. 17th', 'Nov. 24th', 'Dec. 1st', 'Dec. 8th', 'Dec. 15th']
# start_doy_arr    = [335,         342,       349]
# start_doy_labels = ['Dec. 1st', 'Dec. 8th', 'Dec. 15th']
start_doy_arr    = [314,         321,         328,         335,         342,       349]
start_doy_labels = ['Nov. 10th', 'Nov. 17th', 'Nov. 24th', 'Dec. 1st', 'Dec. 8th', 'Dec. 15th']

# Get FUD categories for accuracy measure:
it_1992 = np.where(years == 1992)[0][0]
it_2008= np.where(years == 2008)[0][0]
mean_FUD = np.nanmean(avg_freezeup_doy[it_1992:it_2008])
std_FUD = np.nanstd(avg_freezeup_doy[it_1992:it_2008])
tercile1_FUD = np.nanpercentile(avg_freezeup_doy[it_1992:it_2008],(1/3.)*100)
tercile2_FUD = np.nanpercentile(avg_freezeup_doy[it_1992:it_2008],(2/3.)*100)

obs_FUD_cat = np.zeros((len(years)))*np.nan
for iyr,year in enumerate(years):
    if avg_freezeup_doy[iyr] <= tercile1_FUD:
        obs_FUD_cat[iyr] = -1
    elif avg_freezeup_doy[iyr] > tercile2_FUD:
        obs_FUD_cat[iyr] = 1
    else:
        obs_FUD_cat[iyr] = 0


# MAKE PERSISTENCE OF ANOMALY FORECAST USING
# LEAVE-ONE-OUT CROSS-VALIDATION, I.E. THE TW CLIMATOLOGY
# (FROM WHICH THE ANOMALY IS CALCULATED) IS COMPUTED WITH
# ALL YEARS EXCEPT THE ONE BEING FORECASTED.

tw_anomaly = np.zeros((len(start_doy_arr),len(years)))*np.nan
fud_anomaly = np.zeros((len(start_doy_arr),len(years)))*np.nan
fud_anomaly_observed = np.zeros((len(years)))*np.nan
FUD_clim_forecasts = np.zeros((len(years)))*np.nan
FUD_detected = np.zeros((len(start_doy_arr),len(years)))*np.nan
FUD_obs = avg_freezeup_doy.copy()

plot_all_yrs = False
plot_all_yrs = True
anomaly_type = 'time'
# anomaly_type = 'Tw_zscore'

T_test = np.zeros((len(years),len(start_doy_arr)))*np.nan
T_test_anomaly = np.zeros((len(years),len(start_doy_arr)))*np.nan

for iyr,year in enumerate(years[:]):
    if ~np.isnan(avg_freezeup_doy[iyr]):

        itJan1 = np.where(time == (dt.date(year,1,1)-date_ref).days)[0][0]
        itDec31 = np.where(time == (dt.date(year,12,31)-date_ref).days)[0][0]+1

        # REMOVE THE YEAR BEING FORECASTED FROM DATA FOR MAKING CLIMATOLOGY
        Tw = avg_Twater_vars.copy()
        Tw_in = avg_Twater_vars.copy()
        Tw_in[itJan1:itDec31] = np.nan
        FUD_in = avg_freezeup_doy.copy()
        FUD_in[iyr] = np.nan

        # COMPUTE TW CLIMATOLOGY AND AVERAGE OBSERVED FUD FOR ALL OTHER YEARS
        nw = 31
        freezeup_opt_clim = freezeup_opt
        Tw_clim, Tw_clim_std, _ = rolling_climo(nw,np.squeeze(Tw_in),'all_time',time,years)
        freezeup_doy_clim = detect_FUD_from_Tw_clim(Tw_clim,freezeup_opt_clim,years,time,show=False)
        mean_obs_FUD = (np.nanmean(FUD_in))
        std_obs_FUD = (np.nanstd(FUD_in))
        FUD_clim_forecasts[iyr] = np.floor(mean_obs_FUD)
        # FUD_clim_forecasts[iyr] =(mean_obs_FUD)

        # PLOT TWATER AND TW CLIMATOLOGY FOR THE FORECAST YEAR:
        dit=100
        if plot_all_yrs:
            fig,ax=plt.subplots()
            plt.title(str(year))
            ax.plot(time[itJan1+dit:itDec31+dit],Tw[itJan1+dit:itDec31+dit],'-',color='k')
            ax.plot(time[itJan1+dit:itDec31+dit],Tw_clim[itJan1+dit:itDec31+dit],'-',color=plt.get_cmap('tab20')(0))
            ax.fill_between(time[itJan1+dit:itDec31+dit],Tw_clim[itJan1+dit:itDec31+dit]-Tw_clim_std[itJan1+dit:itDec31+dit],Tw_clim[itJan1+dit:itDec31+dit]+Tw_clim_std[itJan1+dit:itDec31+dit],color=plt.get_cmap('tab20')(1),alpha=0.4)

            # ADD CLIMATOLOGICAL FUD DETECTED FROM TW CLIM:
            # clim_start_doy = dt.date(int(year),1,1)+dt.timedelta(days=int(np.nanmean(freezeup_doy_clim)-1+1*calendar.isleap(year)))
            # clim_it_start = np.where(time == (clim_start_doy-date_ref).days)[0][0]
            # Tw_clim_doy = Tw_clim.copy()
            # Tw_clim_doy[:] = np.nan
            # Tw_clim_doy[clim_it_start] = Tw_clim[clim_it_start].copy()
            # Tw_clim_doy_plot=Tw_clim_doy[itJan1+dit:itDec31+dit]
            # ax.plot(time[itJan1+dit:itDec31+dit],Tw_clim_doy_plot,'o',color=plt.get_cmap('tab20')(0))

        # ADD AVERAGE OBSERVED FUD:
        climobs_start_doy = dt.date(int(year),1,1)+dt.timedelta(days=int(np.nanmean(FUD_in)-1+1*calendar.isleap(year)))
        climobs_it_start = np.where(time == (climobs_start_doy-date_ref).days)[0][0]
        climobs_doy = Tw.copy()
        climobs_doy[:] = np.nan
        climobs_doy[climobs_it_start] = Tw[climobs_it_start].copy()
        climobs_doy_plot=climobs_doy[itJan1+dit:itDec31+dit]
        if plot_all_yrs:
            # ax.plot(time[itJan1+dit:itDec31+dit],climobs_doy_plot,'d',color='gray')
            ax.fill_betweenx(np.arange(-1,25),np.ones(len(np.arange(-1,25)))*(time[climobs_it_start]-std_obs_FUD),np.ones(len(np.arange(-1,25)))*(time[climobs_it_start]+std_obs_FUD),color=[0.92, 0.92, 0.92],alpha=0.4)
            ax.plot(np.ones(len(np.arange(-1,25)))*(time[climobs_it_start]),np.arange(-1,25),color=[0.62, 0.62, 0.62])

        # ADD OBSERVED FUD FOR FORECAST YEAR:
        FUD_yr_doy = avg_freezeup_doy[iyr]
        FUD_start_doy = dt.date(int(year),1,1)+dt.timedelta(days=int(FUD_yr_doy-1+1*calendar.isleap(year)))
        FUD_it_start = np.where(time == (FUD_start_doy-date_ref).days)[0][0]
        FUD_yr = Tw.copy()
        FUD_yr[:] = np.nan
        FUD_yr[FUD_it_start] = Tw[FUD_it_start].copy()
        FUD_yr_plot=FUD_yr[itJan1+dit:itDec31+dit]
        if plot_all_yrs:
            ax.plot(time[itJan1+dit:itDec31+dit],FUD_yr_plot,'o',color='k')

        # ADD MARKER AT START_DATES USED TO MAKE FORECASTS
        # AND COMPUTE ANOMALY IN DAYS TO GET BACK TO TW CLIMATOLOGY
        ref_val = np.floor(mean_obs_FUD)
        # ref_val = np.nanmean(freezeup_doy_clim)
        for istart in range(len(start_doy_arr)):
            if anomaly_type == 'time':
                start_doy = dt.date(int(year),1,1)+dt.timedelta(days=int(start_doy_arr[istart]-1+1*calendar.isleap(year)))
                it_start = np.where(time == (start_doy-date_ref).days)[0][0]
                anomaly = Tw[it_start]-Tw_clim[it_start]
                fud_detected = False
                T_test[iyr,istart] = Tw[it_start]
                T_test_anomaly[iyr,istart] = anomaly

                for it in range(it_start-90,np.min([it_start+120,len(time)-1])):
                    if (Tw_clim[it] > Tw[it_start]) & (Tw_clim[it+1] <= Tw[it_start]):
                        fud_detected = True
                        FUD_detected[istart,iyr] = ref_val+int(it_start-it)
                    if fud_detected:
                        break

                # PLOT LINE TO SHOW ANOMALY FROM TW TO TW CLIMATOLOGY
                # AND ADD MARKER AT START DATE:
                if plot_all_yrs:
                    t_doy = Tw.copy()
                    t_doy[:]=np.nan
                    t_doy[it_start] = Tw[it_start].copy()
                    time_anom_start = time[it_start]
                    time_anom_end = time[it_start-it_start+it]
                    t_doy_plot = t_doy[itJan1+dit:itDec31+dit]
                    ax.plot([time_anom_start,time_anom_end],[Tw[it_start],Tw[it_start]],'-',color=plt.get_cmap('tab20')(istart*2+1),linewidth=2)
                    ax.plot(time[itJan1+dit:itDec31+dit],t_doy_plot,'o',color=plt.get_cmap('tab20')(istart*2))

                    # ADD MARKER AND LINES TO SHOW WHERE THE NEW FORECAST IS:
                    # t_clim_FUD_start = time[climobs_it_start]
                    # t_clim_FUD_end = time[climobs_it_start-it_start+it]
                    # ax.plot([t_clim_FUD_start,t_clim_FUD_end],[Tw[climobs_it_start],Tw[climobs_it_start]],':',color='gray',linewidth=2)
                    # ax.plot([t_clim_FUD_end,t_clim_FUD_end],[Tw[climobs_it_start],Tw[climobs_it_start-it_start+it]],':',color='gray',linewidth=2)

                fud_anomaly[istart,iyr] = int(it_start-it)

            if anomaly_type == 'Tw_zscore':
                start_doy = dt.date(int(year),1,1)+dt.timedelta(days=int(start_doy_arr[istart]-1+1*calendar.isleap(year)))
                it_start = np.where(time == (start_doy-date_ref).days)[0][0]
                # tw_anomaly[istart,iyr] = (np.nanmean(Tw[it_start-31:it_start])-Tw_clim[it_start-16])
                # z_tw_anomaly = tw_anomaly[istart,iyr]/Tw_clim_std[it_start-16]
                # tw_anomaly[istart,iyr] = (np.nanmean(Tw[it_start-7:it_start])-Tw_clim[it_start-4])
                # z_tw_anomaly = tw_anomaly[istart,iyr]/Tw_clim_std[it_start-4]
                tw_anomaly[istart,iyr] = (np.nanmean(Tw[it_start])-Tw_clim[it_start])
                z_tw_anomaly = tw_anomaly[istart,iyr]/Tw_clim_std[it_start]

                fud_anomaly_observed[iyr] = FUD_yr_doy-mean_obs_FUD
                fud_anomaly[istart,iyr] = z_tw_anomaly*std_obs_FUD
                FUD_detected[istart,iyr] = ref_val+fud_anomaly[istart,iyr]

                # PLOT LINE TO SHOW ANOMALY FROM TW TO TW CLIMATOLOGY
                # AND ADD MARKER AT START DATE:
                t_doy = Tw.copy()
                t_doy[:]=np.nan
                t_doy[it_start] = Tw[it_start].copy()

                if plot_all_yrs:
                    t_doy_plot = t_doy[itJan1+dit:itDec31+dit]
                    ax.plot(time[itJan1+dit:itDec31+dit],t_doy_plot,'o',color=plt.get_cmap('tab20')(istart*2))

                    # ADD MARKER AND LINES TO SHOW WHERE THE NEW FORECAST IS:
                    # t_clim_FUD_start = time[climobs_it_start]
                    # t_clim_FUD_end = time[climobs_it_start-it_start+it]
                    # ax.plot([t_clim_FUD_start,t_clim_FUD_end],[Tw[climobs_it_start],Tw[climobs_it_start]],':',color='gray',linewidth=2)
                    # ax.plot([t_clim_FUD_end,t_clim_FUD_end],[Tw[climobs_it_start],Tw[climobs_it_start-it_start+it]],':',color='gray',linewidth=2)

#%%
from functions import linear_fit
print('Correlation with Tw:')
for istart in range(len(start_doy_arr)):
    y = avg_freezeup_doy
    x = T_test[:,istart]
    mask_obs = ~np.isnan(y)
    mask_mod = ~np.isnan(x)
    mask = mask_mod & mask_obs
    m = linear_fit(x[mask],y[mask])
    c = np.sqrt(m[1])
    print(c)

print('Correlation with Tw anomaly:')
for istart in range(len(start_doy_arr)):
    y = avg_freezeup_doy
    x = T_test_anomaly[:,istart]
    mask_obs = ~np.isnan(y)
    mask_mod = ~np.isnan(x)
    mask = mask_mod & mask_obs
    m = linear_fit(x[mask],y[mask])
    c = np.sqrt(m[1])
    print(c)

#%%
MAE_CV = np.zeros(len(start_doy_arr))*np.nan
RMSE_CV = np.zeros(len(start_doy_arr))*np.nan
Rsqr_CV = np.zeros(len(start_doy_arr))*np.nan
pval_CV = np.zeros(len(start_doy_arr))*np.nan
one_week_accuracy_CV = np.zeros(len(start_doy_arr))*np.nan
accuracy_CV = np.zeros(len(start_doy_arr))*np.nan

fig_ts,ax_ts = plt.subplots(nrows=len(start_doy_arr),ncols=1,figsize=(8,10),sharex=True,sharey=True)
for istart in range(len(start_doy_arr)):

    for year in years:
        ax_ts[istart].plot([year,year],[FUD_detected[istart,np.where(years==year)[0][0]],FUD_clim_forecasts[np.where(years==year)[0][0]]],'-',color=plt.get_cmap('tab20')(istart*2+1))

    ax_ts[istart].plot(years,FUD_clim_forecasts,'-',color=[0.8,0.8,0.8])
    ax_ts[istart].plot(years,avg_freezeup_doy,'o-',color='black')
    ax_ts[istart].plot(years,FUD_detected[istart,:],'o-',label=start_doy_labels[istart],color=plt.get_cmap('tab20')(2*istart))
    ax_ts[istart].legend()

    mask_obs = ~np.isnan(avg_freezeup_doy)
    mask_mod = ~np.isnan(FUD_detected[istart,:])
    mask = mask_mod & mask_obs
    MAE_CV[istart] = np.nanmean(np.abs(FUD_detected[istart,:]-avg_freezeup_doy))
    RMSE_CV[istart] = np.sqrt(np.nanmean((FUD_detected[istart,:]-avg_freezeup_doy)**2.))
    one_week_accuracy_CV[istart] = np.sum(np.abs(FUD_detected[istart,:]-avg_freezeup_doy) <= 7)/np.sum(~np.isnan(avg_freezeup_doy+FUD_detected[istart,:]))
    model_CV = sm.OLS(np.squeeze(avg_freezeup_doy[mask]), sm.add_constant(np.squeeze(FUD_detected[istart,:][mask]),has_constant='skip'), missing='drop').fit()
    Rsqr_CV[istart] = model_CV.rsquared
    pval_CV[istart] = model_CV.f_pvalue

    # Get prediction category for accuracy
    pred_FUD_cat = np.zeros((len(years)))*np.nan
    for iyr,year in enumerate(years):
        if FUD_detected[istart,iyr] <= tercile1_FUD:
            pred_FUD_cat[iyr] = -1
        elif FUD_detected[istart,iyr] > tercile2_FUD:
            pred_FUD_cat[iyr] = 1
        else:
            pred_FUD_cat[iyr] = 0
    accuracy_CV[istart] = (np.nansum(pred_FUD_cat == obs_FUD_cat))/np.sum(~np.isnan(obs_FUD_cat))

    ax_ts[-1].set_xlabel('Year')

    print('=========================')
    print(start_doy_labels[istart])
    print('MAE_CV: ',MAE_CV[istart])
    print('RMSE_CV: ',RMSE_CV[istart])
    print('Rsqr_CV: ',Rsqr_CV[istart])
    print('One Week Accuracy_CV: ',one_week_accuracy_CV[istart])
    print('Accuracy_CV: ', accuracy_CV[istart])
    print('p-value_CV: ',pval_CV[istart])


#%%
# MAE_clim = np.zeros(len(start_doy_arr))*np.nan
# RMSE_clim = np.zeros(len(start_doy_arr))*np.nan
# Rsqr_clim = np.zeros(len(start_doy_arr))*np.nan
# pval_clim = np.zeros(len(start_doy_arr))*np.nan
# one_week_accuracy_clim = np.zeros(len(start_doy_arr))*np.nan
# accuracy_clim = np.zeros(len(start_doy_arr))*np.nan

# fig_tsclim,ax_tsclim = plt.subplots(nrows=len(start_doy_arr),ncols=1,figsize=(8,10),sharex=True,sharey=True)
# for istart in range(len(start_doy_arr)):

#     for year in years:
#         ax_tsclim[istart].plot([year,year],[FUD_clim_forecasts[np.where(years==year)[0][0]],FUD_clim_forecasts[np.where(years==year)[0][0]]],'-',color=plt.get_cmap('tab20')(istart*2+1))

#     ax_tsclim[istart].plot(years,FUD_clim_forecasts,'-',color=[0.8,0.8,0.8])
#     ax_tsclim[istart].plot(years,avg_freezeup_doy,'o-',color='black')
#     ax_tsclim[istart].plot(years,FUD_clim_forecasts[:],'o-',label=start_doy_labels[istart],color=plt.get_cmap('tab20')(2*istart))
#     ax_tsclim[istart].legend()

#     MAE_clim[istart] = np.nanmean(np.abs(FUD_clim_forecasts[:]-avg_freezeup_doy))
#     RMSE_clim[istart] = np.sqrt(np.nanmean((FUD_clim_forecasts[:]-avg_freezeup_doy)**2.))
#     one_week_accuracy_clim[istart] = np.sum(np.abs(FUD_clim_forecasts[:]-avg_freezeup_doy) <= 7)/np.sum(~np.isnan(avg_freezeup_doy+FUD_clim_forecasts[:]))
#     model_clim = sm.OLS(np.squeeze(avg_freezeup_doy), sm.add_constant(np.squeeze(FUD_clim_forecasts[:]),has_constant='skip'), missing='drop').fit()
#     Rsqr_clim[istart] = model_clim.rsquared
#     pval_clim[istart] = model_clim.f_pvalue

#     # Get prediction category for accuracy
#     clim_FUD_cat = np.zeros((len(years)))*np.nan
#     for iyr,year in enumerate(years):
#         if FUD_clim_forecasts[iyr] <= tercile1_FUD:
#             clim_FUD_cat[iyr] = -1
#         elif FUD_clim_forecasts[iyr] > tercile2_FUD:
#             clim_FUD_cat[iyr] = 1
#         else:
#             clim_FUD_cat[iyr] = 0
#     accuracy_clim[istart] = (np.nansum(clim_FUD_cat == obs_FUD_cat))/np.sum(~np.isnan(obs_FUD_cat))

#     ax_tsclim[-1].set_xlabel('Year')

#     print('=========================')
#     print(start_doy_labels[istart])
#     print('Clim - MAE_CV: ',MAE_clim[istart])
#     print('Clim - RMSE_CV: ',RMSE_clim[istart])
#     print('Clim - Rsqr_CV: ',Rsqr_clim[istart])
#     print('Clim - One Week Accuracy_CV: ',one_week_accuracy_clim[istart])
#     print('Clim - Accuracy_CV: ',accuracy_clim[istart])
#     print('Clim - p-value_CV: ',pval_clim[istart])

# #%%
# # PLOT FORECAST METRICS

# fig,ax = plt.subplots(nrows=1,ncols=4,figsize=(10,3.5), sharex=True)

# ax[0].plot(np.arange(len(start_doy_arr)),RMSE_CV,'o-', color = 'gray')
# ax[0].plot(np.arange(len(start_doy_arr)),RMSE_clim,'o-', color = 'k')
# ax[0].set_ylabel('RMSE (days)')

# ax[1].plot(np.arange(len(start_doy_arr)),MAE_CV,'o-', color = 'gray')
# ax[1].plot(np.arange(len(start_doy_arr)),MAE_clim,'o-', color = 'k')
# ax[1].set_ylabel('MAE (days)')

# ax[2].plot(np.arange(len(start_doy_arr)),100*(accuracy_CV),'o-', color = 'gray')
# ax[2].plot(np.arange(len(start_doy_arr)),100*(accuracy_clim),'o-', color = 'k')
# ax[2].set_ylabel('Accuracy (%)')

# ax[3].plot(np.arange(len(start_doy_arr)),Rsqr_CV,'o-', color = 'gray')
# ax[3].plot(np.arange(len(start_doy_arr)),Rsqr_clim,'o-', color = 'k')
# ax[3].set_ylabel('Rsqr')

# for im in range(4):
#     plt.sca(ax[im])
#     plt.xticks(np.arange(len(start_doy_arr)), start_doy_labels, rotation=90)
#     ax[im].set_xlabel('Forecast start')

# plt.tight_layout()

# #%%
# # TEST TO USE LINEAR REGRESSION OF ANOMALY INSTEAD OF Z-SCORE:

# # for istart in range(len(start_doy_arr)):
# #     for iyr, year in enumerate(years):
# #         x_reg = np.delete(tw_anomaly[istart,:],iyr)
# #         y_reg = np.delete(fud_anomaly_observed,iyr)
# #         mreg = sm.OLS(y_reg,sm.add_constant(x_reg,has_constant='skip'),missing='drop').fit()

# #         x_fit = tw_anomaly[istart,iyr]
# #         y_fit = mreg.params[1]*x_fit + mreg.params[0]

# #         ax_ts[istart].plot(year,FUD_clim_forecasts[iyr]+y_fit,'*',color=plt.get_cmap('tab20')(2*istart+1))







# #%%
# # # TEST TO MAKE ANOMALY FORECAST BASED ON DECAYING WEIGHTS FROM PAST TW ANOMALIES
# # weights = 1./(np.floor(np.nanmean(FUD_clim_forecasts))-start_doy_arr.copy())

# # MAE_CV_WA = np.zeros(len(start_doy_arr))*np.nan
# # RMSE_CV_WA = np.zeros(len(start_doy_arr))*np.nan
# # Rsqr_CV_WA = np.zeros(len(start_doy_arr))*np.nan
# # pval_CV_WA = np.zeros(len(start_doy_arr))*np.nan
# # accuracy_CV_WA = np.zeros(len(start_doy_arr))*np.nan

# # fig_ts,ax_ts = plt.subplots(nrows=len(start_doy_arr),ncols=1,figsize=(8,10),sharex=True,sharey=True)
# # for istart in range(len(start_doy_arr)):

# #     test_f = np.zeros((len(years)))*np.nan

# #     for iyr,year in enumerate(years):
# #         a = np.sum(weights[0:istart+1]*fud_anomaly[0:istart+1,iyr])/(np.sum(weights[0:istart+1]))
# #         test_f[iyr] = a

# #     ax_ts[istart].plot(years,FUD_clim_forecasts,'-',color=[0.8,0.8,0.8])
# #     ax_ts[istart].plot(years,avg_freezeup_doy,'o-',color='black')
# #     ax_ts[istart].plot(years,test_f+FUD_clim_forecasts,'o-',label=start_doy_labels[istart]+' - Weighted Avg.',color=plt.get_cmap('tab20')(2*istart))
# #     ax_ts[istart].legend()
# #     ax_ts[istart].set_xlabel('Year')


# #     MAE_CV_WA[istart] = np.nanmean(np.abs(test_f+FUD_clim_forecasts-avg_freezeup_doy))
# #     RMSE_CV_WA[istart] = np.sqrt(np.nanmean((test_f+FUD_clim_forecasts-avg_freezeup_doy)**2.))
# #     accuracy_CV_WA[istart] = np.sum(np.abs(test_f+FUD_clim_forecasts-avg_freezeup_doy) <= 7)/np.sum(~np.isnan(avg_freezeup_doy+test_f+FUD_clim_forecasts))
# #     model_CV_WA = sm.OLS(np.squeeze(avg_freezeup_doy), sm.add_constant(np.squeeze(test_f+FUD_clim_forecasts),has_constant='skip'), missing='drop').fit()
# #     Rsqr_CV_WA[istart] = model_CV_WA.rsquared
# #     pval_CV_WA[istart] = model_CV_WA.f_pvalue

# #     ax_ts[-1].set_xlabel('Year')

# #     print('=========================')
# #     print(start_doy_labels[istart])
# #     print('MAE_CV_WA: ',MAE_CV_WA[istart])
# #     print('RMSE_CV_WA: ',RMSE_CV_WA[istart])
# #     print('Rsqr_CV_WA: ',Rsqr_CV_WA[istart])
# #     print('Accuracy_CV_WA: ',accuracy_CV_WA[istart])
# #     print('p-value_CV_WA: ',pval_CV_WA[istart])


