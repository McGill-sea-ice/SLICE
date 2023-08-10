#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:51:47 2022

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
import matplotlib.pyplot as plt
import calendar
from netCDF4 import Dataset
from functions import ncdump
from cdo import Cdo
cdo = Cdo()
cdo = Cdo(tempdir=local_path+'slice/prog/temp_files/') #python

from functions import detect_FUD_from_Tw
from sklearn.metrics import accuracy_score
#%%
def ensemble_avg(var_in):
    if len(var_in.shape) > 1:
        var_out = np.zeros((12,var_in.shape[1],var_in.shape[2]))*np.nan
        for imonth in range(12):
            var_out[imonth,:,:] = np.nanmean(var_in[imonth:240:12,:,:],axis=0)
    else:
        var_out = np.zeros((12))*np.nan
        for imonth in range(12):
            var_out[imonth] = np.nanmean(var_in[imonth:240:12],axis=0)

    return var_out


def ensemble_clim_monthly(feature,region,month,year_start=1981,year_end=2010):

    ftype = '5months'
    data = np.load(p_dir+base+res+'ensemble_vars_sep_dec_f'+ftype+'.npz')
    lat = data['lat'][:]
    lon = data['lon'][:]

    if region == 'D':
        rlon1, rlat1 = 360-77.5, 43.5
        rlon2, rlat2 = 360-73.5, 45.5
    if region == 'YUL':
        rlon1, rlat1 = 360-74.5, 45.5
        rlon2, rlat2 = 360-73.5, 45.5
    if region == 'all':
        rlon1, rlat1 = 0.5, -89.5
        rlon2, rlat2 = 359.5,  89.5
    if region == 'Dplus':
        rlon1, rlat1 = 360-84.5, 42.5
        rlon2, rlat2 = 360-72.5, 47.5
    if region == 'test':
        rlon1, rlat1 = 360-78.5, 31.5
        rlon2, rlat2 = 360-73.5, 37.5

    ilat1 = np.where(lat == rlat1)[0][0]
    ilat2 = np.where(lat == rlat2)[0][0]+1
    ilon1 = np.where(lon == rlon1)[0][0]
    ilon2 = np.where(lon == rlon2)[0][0]+1

    lat_select = lat[ilat1:ilat2+1]
    lon_select = lon[ilon1:ilon2+1]

    if region == 'all':
        var_out = np.zeros((year_end-year_start+1,12,len(lat_select),len(lon_select)))*np.nan
    else:
        var_out = np.zeros((year_end-year_start+1,12))*np.nan

    for iyr,year in enumerate(np.arange(year_start,year_end+1)):
        extension = "_{}-{}_allmembers.grib2.nc".format(year, str(month).rjust(2, '0'))
        filename = base + res + feature + extension
        path = r_dir + "{}-{}/".format(year, str(month).rjust(2, '0'))
        ncid = Dataset(path+filename,'r')
        ndump = ncdump(ncid, verb=False)
        v = ndump[2][-1]
        var = np.squeeze(ncid[v][:])
        # Average in selected region before making climatology
        if region == 'all':
            var_ensm = ensemble_avg(var)
            var_out[iyr,:,:,:] = var_ensm
        else:
            var = np.nanmean(var[:,ilat1:ilat2,ilon1:ilon2],axis=(1,2))
            var_ensm = ensemble_avg(var)
            var_out[iyr,:] = var_ensm

    p33 = np.squeeze(np.nanpercentile(var_out,100/3.,axis=0))
    p66 = np.squeeze(np.nanpercentile(var_out,200/3.,axis=0))

    return var_out, np.squeeze(np.nanmean(var_out,axis=0)), np.squeeze(np.nanstd(var_out,axis=0)),p33, p66


def get_monthly_forecast(feature,region,month,years_in):

    ftype = '5months'
    data = np.load(p_dir+base+res+'ensemble_vars_sep_dec_f'+ftype+'.npz')
    lat = data['lat'][:]
    lon = data['lon'][:]

    if region == 'D':
        rlon1, rlat1 = 360-77.5, 43.5
        rlon2, rlat2 = 360-73.5, 45.5
    if region == 'YUL':
        rlon1, rlat1 = 360-74.5, 45.5
        rlon2, rlat2 = 360-73.5, 45.5
    if region == 'all':
        rlon1, rlat1 = 0.5, -89.5
        rlon2, rlat2 = 359.5,  89.5
    if region == 'Dplus':
        rlon1, rlat1 = 360-84.5, 42.5
        rlon2, rlat2 = 360-72.5, 47.5
    if region == 'test':
        rlon1, rlat1 = 360-78.5, 31.5
        rlon2, rlat2 = 360-73.5, 37.5

    ilat1 = np.where(lat == rlat1)[0][0]
    ilat2 = np.where(lat == rlat2)[0][0]+1
    ilon1 = np.where(lon == rlon1)[0][0]
    ilon2 = np.where(lon == rlon2)[0][0]+1

    lat_select = lat[ilat1:ilat2+1]
    lon_select = lon[ilon1:ilon2+1]

    if region == 'all':
        var_out = np.zeros((len(years_in),12,len(lat_select),len(lon_select)))*np.nan
    else:
        var_out = np.zeros((len(years_in),12))*np.nan

    for iyr,year in enumerate(years_in):
        extension = "_{}-{}_allmembers.grib2.nc".format(year, str(month).rjust(2, '0'))
        filename = base + res + feature + extension
        path = r_dir + "{}-{}/".format(year, str(month).rjust(2, '0'))
        ncid = Dataset(path+filename,'r')
        ndump = ncdump(ncid, verb=False)
        v = ndump[2][-1]
        var = np.squeeze(ncid[v][:])
        # Average in selected region before making climatology
        if region == 'all':
            var_ensm = ensemble_avg(var)
            var_out[iyr,:,:,:] = var_ensm
        else:
            var = np.nanmean(var[:,ilat1:ilat2,ilon1:ilon2],axis=(1,2))
            var_ensm = ensemble_avg(var)
            var_out[iyr,:] = var_ensm

    return var_out


#%%

feature_list = ['TMP_TGL_2m']
base = "cansips_hindcast_raw_"
res = "latlon1.0x1.0_"
r_dir = local_path+'slice/data/raw/CanSIPS/hindcast/raw/'
p_dir = local_path+'slice/data/processed/CanSIPS/hindcast/'

# ys = 1981
# ye = 2010
# ys = 1992
# ye = 2007
ys = 1990
ye = 2020

years_cansips = np.arange(1980,2021)
years = np.arange(1979,2022)

region = 'D'

if region == 'all':
    cansips_clim = np.zeros( (  3, 180, 360 )  )*np.nan
    cansips_clim_std = np.zeros( ( 3, 180, 360 )  )*np.nan
    cansips_clim_p33 = np.zeros( ( 3, 180, 360 )  )*np.nan
    cansips_clim_p66 = np.zeros( ( 3, 180, 360 )  )*np.nan
    cansips_cat_frcst = np.zeros((len(years_cansips),3, 180, 360))*np.nan
    cansips_frcst = np.zeros((len(years_cansips),3, 180, 360))*np.nan
else:
    cansips_clim = np.zeros(3)*np.nan
    cansips_clim_std = np.zeros(3)*np.nan
    cansips_clim_p33 = np.zeros(3)*np.nan
    cansips_clim_p66 = np.zeros(3)*np.nan
    cansips_cat_frcst = np.zeros((len(years_cansips),3))*np.nan
    cansips_frcst = np.zeros((len(years_cansips),3))*np.nan

# Select only 1 feature for now:
feature = 'TMP_TGL_2m'

# Get November (lead = 0) & December (lead =1) Forecast Climatology and Std Deviation
month = 11 # Month start = November
# month = 12 # Month start = December
test,var_ensm_clim,var_ensm_std,var_ensm_p33,var_ensm_p66 = ensemble_clim_monthly(feature,region,month,year_start=ys,year_end=ye)
cansips_clim[0:2] = np.squeeze(var_ensm_clim[0:2]-273.15) # Keep only 0:2 to have only November (lead = 0) & December (lead =1)
cansips_clim_std[0:2] = np.squeeze(var_ensm_std[0:2]-273.15)
cansips_clim_p33[0:2] = np.squeeze(var_ensm_p33[0:2]-273.15)
cansips_clim_p66[0:2] = np.squeeze(var_ensm_p66[0:2]-273.15)

# Now get November (lead = 0) & December (lead =1) Forecast
cansips_frcst_tmp = get_monthly_forecast(feature,region,month,years_cansips)
cansips_frcst[:,0:2] = cansips_frcst_tmp[:,0:2] -273.15

#%%
# NOW ADD NDJ FORECASTS:

# Get Forecast Ensemble Mean starting Nov.
startmonth = 11
# startmonth = 12
ensm_avg_monthly,_,_,_,_ = ensemble_clim_monthly(feature,region,startmonth,year_start=ys,year_end=ye)
# Get NDJ forecast climatology by averaging November (lead = 0) & December (lead =1)
# & January (lead = 2) during training period
ensm_avg_NDJ = np.nanmean(ensm_avg_monthly[:,0:3],axis=1)
# Get average, std, and percentile of NDJ forecasts over the training period
cansips_clim[2] = np.nanmean(ensm_avg_NDJ-273.15)
cansips_clim_std[2] = np.nanstd(ensm_avg_NDJ-273.15)
cansips_clim_p33[2] = np.nanpercentile(ensm_avg_NDJ-273.15,(1/3.)*100)
cansips_clim_p66[2] = np.nanpercentile(ensm_avg_NDJ-273.15,(2/3.)*100)

# Now get NDJ forecasts for all years
cansips_monthly_frcst_tmp = get_monthly_forecast(feature,region,startmonth,years_cansips)
cansips_frcst[:,2] = np.nanmean(cansips_monthly_frcst_tmp[:,0:3],axis=1)-273.15

#%%
# Get categorical forecast based on three tercile (i.e. +/- 0.43 std)
for im in range(3):
    for iyr in range(len(years_cansips)):
        if (cansips_frcst[iyr,im] > cansips_clim_p66[im]):
            cansips_cat_frcst[iyr,im] = 1
        elif (cansips_frcst[iyr,im] <= cansips_clim_p33[im]):
            cansips_cat_frcst[iyr,im] = -1
        else:
            cansips_cat_frcst[iyr,im] = 0


#%%
# Load Twater and FUD data
date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

fp_p_Twater = local_path+'slice/data/processed/'
Twater_loc_list = ['Longueuil_updated']
station_type = 'cities'
freezeup_opt = 1
freezeup_doy, Twater = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False)
avg_freezeup_doy = freezeup_doy[:,0]
avg_freezeup_doy[years >= 2020] = np.nan

#%%
# Select same years for FUD as for CanSIPS
y0 = np.max((years[0],years_cansips[0]))
y1 = np.min((years[-1],years_cansips[-1]))
FUD = avg_freezeup_doy[np.where(years == y0)[0][0]:np.where(years == y1)[0][0]+1]
cansips_cat_frcst = cansips_cat_frcst[np.where(years_cansips == y0)[0][0]:np.where(years_cansips == y1)[0][0]+1]
years_all = np.arange(y0,y1+1)

#%%
# Get FUD climatology
# ys_FUD = 1992
# ye_FUD = 2007
# ys_FUD = 1981
# ye_FUD = 2010
ys_FUD = 1990
ye_FUD = 2020
FUD_clim = np.nanmean(FUD[np.where(years_all == ys_FUD)[0][0]:np.where(years_all == ye_FUD)[0][0]+1])
FUD_clim_std = np.nanstd(FUD[np.where(years_all == ys_FUD)[0][0]:np.where(years_all == ye_FUD)[0][0]+1])
FUD_clim_p66 = np.nanpercentile(FUD[np.where(years_all == ys_FUD)[0][0]:np.where(years_all == ye_FUD)[0][0]+1],200/3.)
FUD_clim_p33 = np.nanpercentile(FUD[np.where(years_all == ys_FUD)[0][0]:np.where(years_all == ye_FUD)[0][0]+1],100/3.)

# Get categorical FUD
FUD_cat = np.zeros(FUD.shape)*np.nan
for iyr in range(len(years_all)):
    if ~np.isnan(FUD[iyr]):
        if (FUD[iyr] > FUD_clim_p66 ):
            FUD_cat[iyr] = 1
        elif (FUD[iyr] <= FUD_clim_p33):
            FUD_cat[iyr] = -1
        else:
            FUD_cat[iyr] = 0

#%%
fig,ax = plt.subplots(nrows=4,ncols=1,sharex=True)

color_minus = plt.get_cmap('tab20')(6)
color_plus = plt.get_cmap('tab20')(4)
# color_train = plt.get_cmap('tab20c')(0)
color_train = [0.5, 0.5, 0.5]

ax[0].fill_betweenx(np.arange(330,400,(400-330)/10.),np.ones(10)*ys_FUD,np.ones(10)*ye_FUD,
                color=color_train , alpha=0.1)
ax[0].plot(years_all,np.ones(len(years_all))*FUD_clim,'--',color = 'gray',linewidth=1)
ax[0].fill_between(years_all,np.ones(len(years_all))*(FUD_clim_p33),np.ones(len(years_all))*(FUD_clim_p66),color='gray',alpha=0.2)
ax[0].plot(years_all,FUD,'-o',color='k')
FUD_plus = FUD.copy()
FUD_minus = FUD.copy()
FUD_plus[FUD_cat < 1] = np.nan
FUD_minus[FUD_cat >=0] = np.nan
ax[0].plot(years_all,FUD_plus,'o',color=color_plus)
ax[0].plot(years_all,FUD_minus,'o',color=color_minus)
ax[0].set_ylabel('FUD (DOY)')
ax[0].grid(linestyle=':')

im = 0
ax[1].fill_betweenx(np.arange(-1.5,6.,(6+1.5)/10.),np.ones(10)*ys,np.ones(10)*ye,
                color=color_train, alpha=0.1)
ax[1].plot(years_cansips,np.ones(len(years_cansips))*cansips_clim[im],'--',color = 'gray',linewidth=1)
ax[1].fill_between(years_cansips,np.ones(len(years_cansips))*(cansips_clim_p33[im]),np.ones(len(years_cansips))*(cansips_clim_p66[im]),color='gray',alpha=0.2)
ax[1].plot(years_cansips,cansips_frcst[:,im],'-o',color='k')
plus = cansips_frcst[:,im].copy()
minus = cansips_frcst[:,im].copy()
plus[cansips_cat_frcst[:,im] < 1] = np.nan
minus[cansips_cat_frcst[:,im] >=0] = np.nan
ax[1].plot(years_all,plus,'o',color=color_plus)
ax[1].plot(years_all,minus,'o',color=color_minus)
ax[1].set_ylabel('Nov. T$_a$ ($^{\circ}$C)\n(lead = 0 month)')
ax[1].grid(linestyle=':')

im = 1
ax[2].fill_betweenx(np.arange(-5.5,-1.,(-1.+5.5)/10.),np.ones(10)*ys,np.ones(10)*ye,
                color=color_train , alpha=0.1)
ax[2].plot(years_cansips,np.ones(len(years_cansips))*cansips_clim[im],'--',color = 'gray',linewidth=1)
ax[2].fill_between(years_cansips,np.ones(len(years_cansips))*(cansips_clim_p33[im]),np.ones(len(years_cansips))*(cansips_clim_p66[im]),color='gray',alpha=0.2)
ax[2].plot(years_cansips,cansips_frcst[:,im],'-o',color='k')
plus = cansips_frcst[:,im].copy()
minus = cansips_frcst[:,im].copy()
plus[cansips_cat_frcst[:,im] < 1] = np.nan
minus[cansips_cat_frcst[:,im] >=0] = np.nan
ax[2].plot(years_all,plus,'o',color=color_plus)
ax[2].plot(years_all,minus,'o',color=color_minus)
ax[2].set_ylabel('Dec. T$_a$ ($^{\circ}$C)\n(lead = 1 month)')
ax[2].grid(linestyle=':')

im = 2
ax[3].fill_betweenx(np.arange(-5,-1,(4)/10.),np.ones(10)*ys,np.ones(10)*ye,
                color=color_train , alpha=0.1)
ax[3].plot(years_cansips,np.ones(len(years_cansips))*cansips_clim[im],'--',color = 'gray',linewidth=1)
ax[3].fill_between(years_cansips,np.ones(len(years_cansips))*(cansips_clim_p33[im]),np.ones(len(years_cansips))*(cansips_clim_p66[im]),color='gray',alpha=0.2)
ax[3].plot(years_cansips,cansips_frcst[:,im],'-o',color='k')
plus = cansips_frcst[:,im].copy()
minus = cansips_frcst[:,im].copy()
plus[cansips_cat_frcst[:,im] < 1] = np.nan
minus[cansips_cat_frcst[:,im] >=0] = np.nan
ax[3].plot(years_all,plus,'o',color=color_plus)
ax[3].plot(years_all,minus,'o',color=color_minus)
ax[3].set_ylabel('NDJ. T$_a$ ($^{\circ}$C)\n(lead = 0-2 month)')
ax[3].grid(linestyle=':')
ax[3].set_xlabel('Years')

cansips_cat_frcst_accuracy = np.zeros(len(cansips_clim))*np.nan
cansips_cat_frcst_accuracy_valid = np.zeros(len(cansips_clim))*np.nan
cansips_cat_frcst_accuracy_test = np.zeros(len(cansips_clim))*np.nan
for im in range(len(cansips_clim)):
    y_true = FUD_cat[12:-1]
    y_pred = cansips_cat_frcst[12:-1,im]
    cansips_cat_frcst_accuracy[im] = accuracy_score(y_true, y_pred)

    ax[0].text(2022.5,310-(im*60),'Total Accuracy (28 years): {}%'.format(int(cansips_cat_frcst_accuracy[im]*100)))

    y_true = FUD_cat[28:28+6]
    y_pred = cansips_cat_frcst[28:28+6,im]
    cansips_cat_frcst_accuracy_valid[im] = accuracy_score(y_true, y_pred)

    ax[0].text(2022.5,303-(im*60),'Valid Accuracy (6 years): {}%'.format(int(cansips_cat_frcst_accuracy_valid[im]*100)))

    y_true = FUD_cat[28+6:28+6+6]
    y_pred = cansips_cat_frcst[28+6:28+6+6,im]
    cansips_cat_frcst_accuracy_test[im] = accuracy_score(y_true, y_pred)

    ax[0].text(2022.5,296-(im*60),'Test Accuracy (6 years): {}%'.format(int(cansips_cat_frcst_accuracy_test[im]*100)))

fig.subplots_adjust(right=0.75,left=0.1,top=0.95)

#%%


# plt.savefig('categorical_forecasts', dpi=900)

#%%
fig,ax = plt.subplots(nrows=1,ncols=1,sharex=True)

color_minus = plt.get_cmap('tab20')(6)
color_plus = plt.get_cmap('tab20')(4)
# color_train = plt.get_cmap('tab20c')(0)
color_train = [0.5, 0.5, 0.5]

# ax.fill_betweenx(np.arange(330,400,(400-330)/10.),np.ones(10)*1992,np.ones(10)*2007,
#                 color=color_train , alpha=0.1)
ax.plot(years_all,np.ones(len(years_all))*FUD_clim,'--',color = 'gray',linewidth=1)
ax.fill_between(years_all,np.ones(len(years_all))*(FUD_clim_p33),np.ones(len(years_all))*(FUD_clim_p66),color='gray',alpha=0.2)
ax.plot(years_all,FUD,'-o',color='k')
FUD_plus = FUD.copy()
FUD_minus = FUD.copy()
FUD_plus[FUD_cat < 1] = np.nan
FUD_minus[FUD_cat >=0] = np.nan
# ax.plot(years_all,FUD_plus,'o',color=color_plus)
# ax.plot(years_all,FUD_minus,'o',color=color_minus)
ax.set_ylabel('FUD (DOY)')
ax.set_xlabel('Years')
ax.set_xlim([1991,2020])
ax.set_ylim([330,385])
ax.grid(linestyle=':')


#%%
from functions import detrend_ts

# im = 2
# ax[3].fill_betweenx(np.arange(-5,-1,(4)/10.),np.ones(10)*ys,np.ones(10)*ye,
#                 color=color_train , alpha=0.1)
# ax[3].plot(years_cansips,np.ones(len(years_cansips))*cansips_clim[im],'--',color = 'gray',linewidth=1)
# ax[3].fill_between(years_cansips,np.ones(len(years_cansips))*(cansips_clim_p33[im]),np.ones(len(years_cansips))*(cansips_clim_p66[im]),color='gray',alpha=0.2)
# ax[3].plot(years_cansips,cansips_frcst[:,im],'-o',color='k')
# plus = cansips_frcst[:,im].copy()
# minus = cansips_frcst[:,im].copy()
# plus[cansips_cat_frcst[:,im] < 1] = np.nan
# minus[cansips_cat_frcst[:,im] >=0] = np.nan
# ax[3].plot(years_all,plus,'o',color=color_plus)
# ax[3].plot(years_all,minus,'o',color=color_minus)
# ax[3].set_ylabel('NDJ. T$_a$ ($^{\circ}$C)\n(lead = 0-2 month)')
# ax[3].grid(linestyle=':')
# ax[3].set_xlabel('Years')

fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True)


im = 2
ax[0].fill_betweenx(np.arange(-5,-1,(4)/10.),np.ones(10)*ys,np.ones(10)*ye,
                color=color_train , alpha=0.1)
ax[0].plot(years_cansips,np.ones(len(years_cansips))*cansips_clim[im],'--',color = 'gray',linewidth=1)
ax[0].fill_between(years_cansips,np.ones(len(years_cansips))*(cansips_clim_p33[im]),np.ones(len(years_cansips))*(cansips_clim_p66[im]),color='gray',alpha=0.2)
ax[0].plot(years_cansips,cansips_frcst[:,im],'-o',color='k')
plus = cansips_frcst[:,im].copy()
minus = cansips_frcst[:,im].copy()
plus[cansips_cat_frcst[:,im] < 1] = np.nan
minus[cansips_cat_frcst[:,im] >=0] = np.nan
ax[0].plot(years_all,plus,'o',color=color_plus)
ax[0].plot(years_all,minus,'o',color=color_minus)
ax[0].set_ylabel('NDJ. T$_a$ ($^{\circ}$C)\n(lead = 0-2 month)')
ax[0].grid(linestyle=':')
ax[0].set_xlabel('Years')

# detrend NDJ forecast
it_ys = np.where(years_cansips == ys)[0][0]
it_ye = np.where(years_cansips == ye)[0][0]
yd,[m,b] = detrend_ts(cansips_frcst[it_ys:it_ye,2],years_cansips[it_ys:it_ye],anomaly_type='linear')

(cansips_frcst[:,2] - (m*years_cansips + b))

# Get detrended NDJ Climatology and Std Deviation
yd_clim = np.nanmean(yd)
yd_std = np.nanstd(yd)
yd_p33 = np.nanpercentile(yd,(1/3.)*100)
yd_p66 = np.nanpercentile(yd,(2/3.)*100)
# Get categorical forecast based on three tercile (i.e. +/- 0.43 std)
yd_cat_frcst = np.zeros(len(years_cansips))*np.nan
for iyr in range(len(years_cansips)):
    if ((cansips_frcst[:,2] - (m*years_cansips + b))[iyr] > yd_p66):
        yd_cat_frcst[iyr] = 1
    elif ((cansips_frcst[:,2] - (m*years_cansips + b))[iyr] <= yd_p33):
        yd_cat_frcst[iyr] = -1
    else:
        yd_cat_frcst[iyr] = 0

ax[1].fill_betweenx(np.arange(-2,2,(4)/10.),np.ones(10)*ys,np.ones(10)*ye,
                color=color_train , alpha=0.1)
ax[1].plot(years_cansips,np.ones(len(years_cansips))*yd_clim,'--',color = 'gray',linewidth=1)
ax[1].fill_between(years_cansips,np.ones(len(years_cansips))*(yd_p33),np.ones(len(years_cansips))*(yd_p66),color='gray',alpha=0.2)
ax[1].plot(years_cansips,(cansips_frcst[:,2] - (m*years_cansips + b)),'-o',color='k')
plus = (cansips_frcst[:,2] - (m*years_cansips + b)).copy()
minus = (cansips_frcst[:,2] - (m*years_cansips + b)).copy()
plus[yd_cat_frcst < 1] = np.nan
minus[yd_cat_frcst>=0] = np.nan
ax[1].plot(years_all,plus,'o',color=color_plus)
ax[1].plot(years_all,minus,'o',color=color_minus)
ax[1].set_ylabel('Detrended NDJ. T$_a$ ($^{\circ}$C)\n(lead = 0-2 month)')
ax[1].grid(linestyle=':')
ax[1].set_xlabel('Years')

#%%

fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True)

color_minus = plt.get_cmap('Accent')(4)
color_plus = plt.get_cmap('Dark2')(5)
color_train = [0.1, 0.1, 0.5]


years_plot = np.arange(ys-1,ye+2)
y_plot = np.zeros(len(years_plot))*np.nan
y_plot[1:-2] = FUD[np.where(years_all == ys)[0][0]:np.where(years_all == ye)[0][0]]

ax[0].plot(years_plot,np.ones(len(years_plot))*FUD_clim,'--',color = 'gray',linewidth=1)
ax[0].fill_between(years_plot,np.ones(len(years_plot))*(FUD_clim_p33),np.ones(len(years_plot))*(FUD_clim_p66),color='gray',alpha=0.2)
ax[0].plot(years_plot,y_plot,'-o',color='k')
FUD_plus = FUD.copy()
FUD_minus = FUD.copy()
FUD_plus[FUD_cat < 1] = np.nan
FUD_minus[FUD_cat >=0] = np.nan
ax[0].plot(years_all,FUD_plus,'o',color=color_plus)
ax[0].plot(years_all,FUD_minus,'o',color=color_minus)
ax[0].set_ylabel('FUD\n(day of year)')
ax[0].grid(linestyle=':')
ax[0].set_ylim(334,386)


y_d = (cansips_frcst[:,2] - (m*years_cansips + b))
# y_d[0:it_ys] = np.nan
y_plot = np.zeros(len(years_plot))*np.nan
y_plot[1:-1] = y_d[it_ys:it_ye+1]


ax[1].plot(years_plot,np.ones(len(years_plot))*yd_clim,'--',color = 'gray',linewidth=1)
ax[1].fill_between(years_plot,np.ones(len(years_plot))*(yd_p33),np.ones(len(years_plot))*(yd_p66),color='gray',alpha=0.2)
ax[1].plot(years_plot,y_plot,'-o',color='k')
plus = ((cansips_frcst[:,2] - (m*years_cansips + b))).copy()
minus = ((cansips_frcst[:,2] - (m*years_cansips + b))).copy()
plus[yd_cat_frcst < 1] = np.nan
minus[yd_cat_frcst>=0] = np.nan
ax[1].plot(years_all,plus,'o',color=color_plus)
ax[1].plot(years_all,minus,'o',color=color_minus)
ax[1].set_ylabel('Detrended Ensemble Mean\n NDJ T$_a$ anomaly ($^{\circ}$C)\n(lead = 0-2 month)')
ax[1].grid(linestyle=':')
ax[1].set_xlabel('Years')
ax[1].set_xlim(1989,2021)
ax[1].set_ylim(-1.4,1.4)

fig.subplots_adjust(right=0.93,left=0.18,top=0.95)



y_true = FUD_cat[12:-1]
y_pred = cansips_cat_frcst[12:-1,2]
cat_acc = accuracy_score(y_true, y_pred)

print(cat_acc)
# plt.savefig('baseline_categorical_forecast_NDJ_Nov1', dpi=900)

