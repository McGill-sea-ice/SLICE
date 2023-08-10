#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 18:15:03 2022

@author: Amelie
"""
#%%
local_path = '/storage/amelie/'
# local_path = '/Volumes/SeagateUSB/McGill/Postdoc/'

import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath(local_path +'slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

from functions import detect_FUD_from_Tw

#%%

save = False

#%%
# Make daily timeseries of FUD in the next XX days
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

# Load Twater and FUD data
fp_p_Twater = local_path+'slice/data/processed/'
Twater_loc_list = ['Longueuil_updated']
station_type = 'cities'
freezeup_opt = 2
freezeup_doy, Twater, fud_dates = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False,return_FUD_dates = True)
freezeup_doy[np.where(years == 2020)] = np.nan

Twater_mean = np.nanmean(Twater,axis=1)
Twater_mean = np.expand_dims(Twater_mean, axis=1)
Twater_mean[14269:14329] = 0.

# Average (and round) FUD from all locations:
avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
avg_freezeup_doy = np.round(avg_freezeup_doy)

# Rearrange to get same time series as the one saved by 'save_predictor_daily_timeseries'
yr_start = 1992
date_ref = dt.date(1900,1,1)
it_start = np.where(time == (dt.date(yr_start,1,1)-date_ref).days)[0][0]
Twater_mean = Twater_mean[it_start:]
time = time[it_start:]
avg_freezeup_doy = avg_freezeup_doy[np.where(years==yr_start)[0][0]:]
fud_dates = fud_dates[np.where(years==yr_start)[0][0]:]
years = years[np.where(years==yr_start)[0][0]:]

# Initialize time series variables
fud_0d_10d = np.zeros(time.shape)
fud_0d_15d = np.zeros(time.shape)
fud_0d_20d = np.zeros(time.shape)
fud_0d_30d = np.zeros(time.shape)
fud_0d_40d = np.zeros(time.shape)
fud_0d_45d = np.zeros(time.shape)
fud_0d_50d = np.zeros(time.shape)
fud_0d_60d = np.zeros(time.shape)

# Find if FUD happen sin the next XX days:
for iyr,year in enumerate(years):
    fud_yr = fud_dates[iyr][0]
    fud_mn = fud_dates[iyr][1]
    fud_dy = fud_dates[iyr][2]
    if np.all(~np.isnan(fud_dates[iyr])):
        it_fud = np.where(time == (dt.date(int(fud_yr),int(fud_mn),int(fud_dy))-date_ref).days)[0][0]

        fud_0d_10d[it_fud-10:it_fud+1] = 1
        fud_0d_15d[it_fud-15:it_fud+1] = 1
        fud_0d_20d[it_fud-20:it_fud+1] = 1
        fud_0d_30d[it_fud-30:it_fud+1] = 1
        fud_0d_40d[it_fud-40:it_fud+1] = 1
        fud_0d_45d[it_fud-45:it_fud+1] = 1
        fud_0d_50d[it_fud-50:it_fud+1] = 1
        fud_0d_60d[it_fud-60:it_fud+1] = 1


#%%
fpath = local_path+'slice/data/daily_predictors/ML_timeseries/'
fname = 'ML_dataset_with_cansips.npz'
# fname = 'ML_dataset.npz'

with np.load(fpath+fname, allow_pickle='TRUE') as data:
    ds = data['data']
    # date_ref = data['date_ref']
    date_ref = dt.date(1900,1,1)
    region_ERA5 = data['region_ERA5']
    region_cansips = data['region_cansips']
    loc_Twater = data['Twater_loc_list']
    loc_discharge = data['loc_discharge']
    loc_level = data['loc_level']
    loc_levelO = data['loc_levelO']
    labels = [k.decode('UTF-8') for k in data['labels']]


#%%
df = pd.DataFrame(ds,columns=labels)
time = df['Days since 1900-01-01'].values
yr_start = 1992
# yr_end = 2020
date_ref = dt.date(1900,1,1)
it_start = np.where(time == (dt.date(yr_start,1,1)-date_ref).days)[0][0]
# it_end = np.where(time == (dt.date(yr_end+1,1,1)-date_ref).days)[0][0]

# df = df[df.columns][it_start:it_end]
df = df[df.columns][it_start:]

# Add new variables with FUD in the next XX days:
df['FUD-0010'] = fud_0d_10d
df['FUD-0015'] = fud_0d_15d
df['FUD-0020'] = fud_0d_20d
df['FUD-0030'] = fud_0d_30d
df['FUD-0040'] = fud_0d_40d
df['FUD-0045'] = fud_0d_45d
df['FUD-0050'] = fud_0d_50d
df['FUD-0060'] = fud_0d_60d



#%%
savepath = local_path+'slice/data/daily_predictors/'
if save:
    df.to_excel(savepath+'/predictor_data_daily_timeseries.xlsx', index = False)


#%%

# def compute_ice_sfc_temp(Ta_k,F_sw_k,F_lwd_k,h_km1,Ti_km1=None,linearize='temp'):
#     if Ta_k >= 273.15:
#         # Ice is melting. Fix surface temperature to 0 deg. C.
#         T_k = 273.15
#     else:
#         # Ice is growing from the bottom. Surface temperature adjusts with the heat flux balance.
#         if linearize == 'time':
#             # i.e. Chen (1984)
#             T_k = 273.15

#         if linearize == 'temp':
#             # i.e. my notes
#             T_k = 273.15

#     return T_k


# # Compute equivalent temperature, with ice thickness estimated from accumulated FDD
#     #   i) Detect FUD
#     #  ii) Detect IFD (Ice-Free Date)
#     # iii) Accumulate FDD from FUD to IFD
#     #  iv) Convert aFDD time series to ice thickness time series.
#     #   v) Compute T_eq =

# k = 2.03 # W m^-1 K^-1
# rho_i = 917.0 # kg m^3
# L = 3.335 * 10e5 # J kg^-1
# a = 1.0

# Ta = df['Avg. Ta_mean'].values + 273.15
# Tw = df['Avg. Twater'].values + 273.15

# time_int = df['Days since 1900-01-01'].values
# hi = np.zeros(len(time_int))
# Ti_sfc = np.zeros(len(time_int)) * np.nan
# E_eq = np.zeros(len(time_int)) *np.nan

# for it in range(len(time_int)):
#     date_it = date_ref+ dt.timedelta(days=int(time_int[it]))

#     if date_it.year in years:

#         if date_it.month > 10:
#             # This is leading up to freeze-up.
#             # We start with no ice (h = 0) until freeze-up is detected.
#             # Then ice can grow (if Ta < 0 C ) or melt (if Ta >= 0 C)
#             iyr = np.where(years == date_it.year)[0][0]
#             fud_yr = avg_freezeup_doy[iyr]
#             doy_it = (date_it-dt.date(date_it.year,1,1)).days + 1

#             if doy_it == fud_yr:
#                 # This is freeze-up. We set 1 mm of ice to start with.
#                 hi[it] = 0.001
#                 Ti_sfc[it] = compute_ice_sfc_temp(doy_it,Ta[it])
#                 E_eq[it] = compute_internal_energy(Ti[it],Tw[it],hi[it])
#             if doy_it > fud_yr:
#                 # Freeze-up has occured, so changes in thickness can be recorded.
#                 h0 = hi[it-1]

#                 if Ta[it] >= 273.15:
#                     # Ice is melting.
#                     # There is no conduction heat flux and ice sfc temperature is set to 0 C.
#                     Ti_sfc[it] = 273.15
#                     # Then find thickness:
#                     # dh = ( (k_i * ()/h0 ) - ( )  ) / (rho_i * L)
#                     # hi[it] =
#                     # Then get equivalent internal energy:
#                     # E_eq[it] = compute_internal_energy(Ti[it],Tw[it], hi[it])

#                 else:
#                     # Ice is growing.
#                     # Find ice temp. first:
#                     # Ti[it] = compute_ice_sfc_temp(doy_it,Ta[it])
#                     # Then find thickness:
#                     # dh = ( (k_i * ()/h0 ) - ( )  ) / (rho_i * L)
#                     # hi[it] =
#                     hi[it] = np.sqrt( hi[it-1]**2. +  86400.* (  ((2*k*a)/(rho_i*L)) * ( 273.15-Ta[it] )  )   )
#                     # Then get equivalent internal energy:
#                     # E_eq[it] = compute_internal_energy(Ti[it],Tw[it], hi[it])


#         elif date_it.month < 5:
#             # This is after freeze-up, up to breakup.
#             # Ice can be growing (if Ta < 0 C ) or melting (if Ta >= 0 C)
#             if hi[it-1] > 0:
#                 if Ta[it] >= 273.15:
#                     # Ice is melting.
#                     # There is no conduction heat flux and ice sfc temperature is set to 0 C.
#                     Ti_sfc[it] = 273.15
#                     # Then find thickness:
#                     # dh = ( (k_i * ()/h0 ) - ( )  ) / (rho_i * L)
#                     # hi[it] =
#                     # Then get equivalent internal energy:
#                     # E_eq[it] = compute_internal_energy(Ti[it],Tw[it], hi[it])

#                 else:
#                     # Ice is growing.
#                     # Find ice temp. first:
#                     # Ti[it] = compute_ice_sfc_temp(doy_it,Ta[it])
#                     # Then find thickness:
#                     # dh = ( (k_i * ()/h0 ) - ( )  ) / (rho_i * L)
#                     # hi[it] =
#                     hi[it] = np.sqrt( hi[it-1]**2. +  86400.* (  ((2*k*a)/(rho_i*L)) * ( 273.15-Ta[it] )  )   )
#                     # Then get equivalent internal energy:
#                     # E_eq[it] = compute_internal_energy(Ti[it],Tw[it], hi[it])

#             else:
#                 # There is no more ice (h =0), but if temperatures get cold again new ice can form.
#                 if Ta[it] >= 273.15:# No ice grows.
#                     hi[it] = 0
#                 else: # Ice can grow.
#                 # !!!!! NOT SURE HOW TO HANDLE THIS!!!! NEED TO DETECT NEW FREEZUP?!?
#                     hi[it] = 0
#                     # hi[it] = np.sqrt( hi[it-1]**2. +  86400.* (  ((2*k*a)/(rho_i*L)) * ( 273.15-Ta[it] )  )   )


# fig,ax= plt.subplots(nrows = 3, ncols = 1,figsize=(4,5), sharex=True)
# ax[0].plot(time_int,Tw)
# ax[1].plot(time_int,Ta)
# ax[2].plot(time_int,hi)

# ax[0].grid()
# ax[1].grid()
# ax[2].grid()