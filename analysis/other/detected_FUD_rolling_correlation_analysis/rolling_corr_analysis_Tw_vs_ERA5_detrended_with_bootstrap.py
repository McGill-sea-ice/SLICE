#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:45:58 2021

@author: Amelie
"""
import numpy as np
import scipy
from scipy import ndimage

import pandas
from statsmodels.formula.api import ols

import datetime as dt
import calendar

import matplotlib.pyplot as plt

from functions import running_nanmean,find_freezeup_Tw_all_yrs,season_mask
from functions import linear_fit, rolling_climo, r_confidence_interval

#%%
def get_window_vars(vars_in,names_in,end_dates,window_arr,years,time,month_start_day=1):
    nvars = len(names_in)
    vars_out = np.zeros((nvars,len(years),len(window_arr),2))*np.nan
    window_size = window_arr[1]-window_arr[0]

    for iyr, year in enumerate(years):
        # print(iyr,year)
        i0 = (dt.date(int(year),1,month_start_day)-date_ref).days
        i0 = np.where(time == i0)[0][0]

        i1 = (dt.date(int(year)+1,3,month_start_day)-date_ref).days
        try:
            i1 = np.where(time == i1)[0][0]
        except:
            i1 = len(time)-1

        doy0 = (dt.date(int(year),1,month_start_day)-(dt.date(int(year),1,1))).days + 1
        doy_arr = np.arange(doy0, doy0+(i1-i0))

        if ~np.isnan(end_dates[iyr]):
            for iw,w in enumerate(window_arr):
                # window_type == 'moving':
                ied = np.where(doy_arr == end_dates[iyr])[0][0]
                ifd = ied-(iw)*(window_size)
                iw0 = ied-(iw+1)*(window_size)

                for ivar in range(nvars):
                    var_year = vars_in[i0:i1,ivar]
                    varname = names_in[ivar]

                    if (varname[0:3] == 'Avg'):
                        vars_out[ivar,iyr,iw,0] = np.nanmean(var_year[iw0:ifd])

                    if (varname[0:3] == 'Tot'):
                        vars_out[ivar,iyr,iw,0] = np.nansum(var_year[iw0:ifd])
                        # if np.nansum(var_year[iw0:ifd]) == 0:
                        #     vars_out[ivar,iyr,iw,0] = np.nan
                        # else:
                        #     vars_out[ivar,iyr,iw,0] = np.nansum(var_year[iw0:ifd])

                # window_type == 'increasing':
                ifd = np.where(doy_arr == end_dates[iyr])[0][0]
                iw0 = ifd-w

                for ivar in range(nvars):
                    var_year = vars_in[i0:i1,ivar]
                    varname = names_in[ivar]

                    if (varname[0:3] == 'Avg'):
                        vars_out[ivar,iyr,iw,1] = np.nanmean(var_year[iw0:ifd])

                    if (varname[0:3] == 'Tot'):
                        vars_out[ivar,iyr,iw,1] = np.nansum(var_year[iw0:ifd])
                        # if np.nansum(var_year[iw0:ifd]) == 0:
                        #     vars_out[ivar,iyr,iw,1] = np.nan
                        # else:
                        #     vars_out[ivar,iyr,iw,1] = np.nansum(var_year[iw0:ifd])

    return vars_out


def deasonalize_ts(Nwindow,vars_in,varnames,time_spec,time,years):
    vars_out = np.zeros(vars_in.shape)*np.nan

    for ivar in range(len(varnames)):
        var_mean, var_std, weather_window = rolling_climo(Nwindow,vars_in[:,ivar],time_spec,time,years)
        # if weather_varnames[ivar][0:3] == 'Tot' :
        #     weather_vars[:,ivar] = weather_vars[:,ivar]
        # else:
        #     weather_vars[:,ivar] = weather_vars[:,ivar]-var_mean
        vars_out[:,ivar] = vars_in[:,ivar]-var_mean

    return vars_out


def bootstrap(xvar_in, yvar_in, nboot=1000):

    nyears = len(xvar_in)
    r_out = np.zeros((nboot))*np.nan

    for n in range(nboot):
        if nboot >1:
            boot_indx = np.random.choice(nyears,size=nyears,replace=True)
        else:
            boot_indx = np.random.choice(nyears,size=nyears,replace=False)


        xvar_boot = xvar_in[boot_indx].copy()
        yvar_boot = yvar_in[boot_indx].copy()

        lincoeff, Rsqr = linear_fit(xvar_boot,yvar_boot)

        r_out[n] = np.sqrt(Rsqr)
        if (lincoeff[0]< 0):
            r_out[n] *= -1

    return r_out


def detrend_ts(xvar_in,yvar_in,years,anomaly_type):

    if anomaly_type == 'linear':
        [mx,bx],_ = linear_fit(years, xvar_in)
        [my,by],_ = linear_fit(years, yvar_in)
        x_trend = mx*years + bx
        y_trend = my*years + by

        xvar_out = xvar_in-x_trend
        yvar_out = yvar_in-y_trend

    if anomaly_type == 'mean':
        x_mean = np.nanmean(xvar_in)
        y_mean = np.nanmean(yvar_in)

        xvar_out = xvar_in-x_mean
        yvar_out = yvar_in-y_mean

    return xvar_out, yvar_out

#%%
years = [1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019]

fp = '../../../data/processed/'

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2020,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)
years = np.array(years)

#%%
end_dates_arr = np.zeros((len(years),5))*np.nan
for iyear,year in enumerate(years):
    doy_sep1 = (dt.date(int(year),9,1)-(dt.date(int(year),1,1))).days + 1
    doy_oct1 = (dt.date(int(year),10,1)-(dt.date(int(year),1,1))).days + 1
    doy_nov1 = (dt.date(int(year),11,1)-(dt.date(int(year),1,1))).days + 1
    doy_dec1 = (dt.date(int(year),12,1)-(dt.date(int(year),1,1))).days + 1
    doy_dec15 = (dt.date(int(year),12,15)-(dt.date(int(year),1,1))).days + 1
    doy_jan1 = (dt.date(int(year+1),1,1)-(dt.date(int(year),1,1))).days + 1
    # end_dates_arr[iyear,0] = doy_dec15
    end_dates_arr[iyear,0] = doy_jan1
    end_dates_arr[iyear,1] = doy_dec1
    end_dates_arr[iyear,2] = doy_nov1
    end_dates_arr[iyear,3] = doy_oct1
    end_dates_arr[iyear,4] = doy_sep1
enddate_labels = ['Jan. 1st','Dec. 1st', 'Nov. 1st', 'Oct. 1st', 'Sept. 1st']

p_critical = 0.05

deseasonalize = False
detrend = True
anomaly = 'linear'

nboot = 1

#window_arr = 2*2**np.arange(0,8) # For powers of 2
# window_arr = np.arange(1,39)*7 # For weeks, up to Jan 1st
# window_arr = np.arange(1,3)*7
window_arr = np.arange(1,9)*30 # For months

#%%
# LOAD FREEZEUP DATES OR FIND FROM TWATER TIME SERIES
water_name_list = ['Longueuil_cleaned_filled']
station_labels = ['Longueuil']
station_type = 'cities'

# water_name_list = ['Atwater_cleaned_filled','DesBaillets_cleaned_filled','Longueuil_cleaned_filled','Candiac_cleaned_filled']
# station_labels = ['Atwater','DesBaillets','Longueuil','Candiac']
# station_type = 'cities'

load_freezeup = False
freezeup_opt = 2
month_start_day = 1

# OPTION 1
if freezeup_opt == 1:
    def_opt = 1
    smooth_T =False; N_smooth = 3; mean_type='centered'
    round_T = False; round_type= 'half_unit'
    Gauss_filter = False
    T_thresh = 0.75
    dTdt_thresh = 0.25
    d2Tdt2_thresh = 0.25
    nd = 1

# OPTION 2
if freezeup_opt == 2:
    def_opt = 3
    smooth_T =False; N_smooth = 3; mean_type='centered'
    round_T = False; round_type= 'half_unit'
    Gauss_filter = True
    sig_dog = 3.5
    T_thresh = 3.
    dTdt_thresh = 0.15
    d2Tdt2_thresh = 0.15
    # dTdt_thresh = 0.20 # using 0.20 here instead of 0.15 can achieve even lower difference between stations, but then some years do not match the charts timing anymore...
    # d2Tdt2_thresh = 0.20
    nd = 30

if load_freezeup:
    print('ERROR: STILL NEED TO DEFINE THIS FROM SAVED ARRAYS...')
else:
    freezeup_dates = np.zeros((len(years),3,len(water_name_list)))*np.nan
    freezeup_doy = np.zeros((len(years),len(water_name_list)))*np.nan
    freezeup_temp = np.zeros((len(years),len(water_name_list)))*np.nan

    Twater = np.zeros((len(time),len(water_name_list)))*np.nan
    Twater_dTdt = np.zeros((len(time),len(water_name_list)))*np.nan
    Twater_d2Tdt2 = np.zeros((len(time),len(water_name_list)))*np.nan
    Twater_DoG1 = np.zeros((len(time),len(water_name_list)))*np.nan
    Twater_DoG2 = np.zeros((len(time),len(water_name_list)))*np.nan

    for iloc,loc in enumerate(water_name_list):
        loc_water_loc = water_name_list[iloc]
        water_loc_data = np.load(fp+'Twater_'+station_type+'/Twater_'+station_type+'_'+loc_water_loc+'.npz',allow_pickle='TRUE')
        Twater_tmp = water_loc_data['Twater'][:,1]

        # APPLY WINTER OFFSET TO WATER TEMPERATURE TIME SERIES FIRST
        Twater[:,iloc] = Twater_tmp
        if loc == 'Candiac_cleaned_filled':
            Twater[:,iloc] = Twater_tmp-0.8
        if (loc == 'Atwater_cleaned_filled'):
            Twater[0:12490,iloc] = Twater_tmp[0:12490]-0.7


        # THEN FIND DTDt, D2TDt2, etc.
        Twater_tmp = Twater[:,iloc].copy()
        if round_T:
            if round_type == 'unit':
                Twater_tmp = np.round(Twater_tmp.copy())
            if round_type == 'half_unit':
                Twater_tmp = np.round(Twater_tmp.copy()* 2) / 2.
        if smooth_T:
            Twater_tmp = running_nanmean(Twater_tmp.copy(),N_smooth,mean_type=mean_type)

        dTdt_tmp = np.zeros((Twater_tmp.shape[0],3))*np.nan

        dTdt_tmp[0:-1,0]= Twater_tmp[1:]- Twater_tmp[0:-1] # Forwards
        dTdt_tmp[1:,1] = Twater_tmp[1:] - Twater_tmp[0:-1] # Backwards
        dTdt_tmp[0:-1,2]= Twater_tmp[0:-1]-Twater_tmp[1:]  # -1*Forwards

        Twater_dTdt[:,iloc] = np.nanmean(dTdt_tmp[:,0:2],axis=1)
        Twater_d2Tdt2[:,iloc] = -1*np.nanmean(dTdt_tmp[:,1:3],axis=1)

        if Gauss_filter:
            Twater_DoG1[:,iloc] = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=1)
            Twater_DoG2[:,iloc] = scipy.ndimage.gaussian_filter1d(Twater_tmp.copy(),sigma=sig_dog,order=2)

        # THEN FIND FREEZEUP ACCORDING TO CHOSEN OPTION:
        if def_opt == 3:
            fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw_all_yrs(def_opt,Twater_tmp,Twater_DoG1[:,iloc],Twater_DoG2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
            freezeup_dates[:,:,iloc] = fd
            freezeup_temp[:,iloc] = ftw
        else:
            fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw_all_yrs(def_opt,Twater_tmp,Twater_dTdt[:,iloc],Twater_d2Tdt2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
            freezeup_dates[:,:,iloc] = fd
            freezeup_temp[:,iloc] = ftw

        # FINALLY, TRANSFORM FREEZEUP FROM DATE FORMAT TO DOY FORMAT:
        for iyr,year in enumerate(years):
            if ~np.isnan(freezeup_dates[iyr,0,iloc]):
                fd_yy = int(freezeup_dates[iyr,0,iloc])
                fd_mm = int(freezeup_dates[iyr,1,iloc])
                fd_dd = int(freezeup_dates[iyr,2,iloc])

                fd_doy = (dt.date(fd_yy,fd_mm,fd_dd)-dt.date(fd_yy,1,1)).days + 1
                if fd_doy < 60: fd_doy += 365

                freezeup_doy[iyr,iloc]=fd_doy

# Average all stations to get mean freezeup DOY for each year
avg_freezeup_doy = np.round(np.nanmean(freezeup_doy,axis=1))

# end_dates_arr = np.zeros((len(years),1))*np.nan
# end_dates_arr[:,0] = avg_freezeup_doy


#%%
# MAKE TWATER INTO AN EXPLANATORY VARIABLE
Twater_varnames = ['Avg. water temp.']
Twater_vars = np.zeros((len(time),len(Twater_varnames)))*np.nan
Twater_vars[:,0] = np.nanmean(Twater,axis=1)
Twater_vars = np.squeeze(Twater_vars)

if deseasonalize:
    Nwindow = 31
    Twater_vars = deasonalize_ts(Nwindow,Twater_vars,['Twater'],'all_time',time,years)

# weather_vars2_all = np.zeros((len(weather_varnames2),len(years),len(window_arr),end_dates_arr.shape[1],2,len(weather_loc_list)))*np.nan
Twater_vars_all = np.zeros((len(years),len(window_arr),end_dates_arr.shape[1],2))*np.nan
for iend in range(end_dates_arr.shape[1]):
    Twater_vars_all[:,:,iend,:] = get_window_vars(np.expand_dims(Twater_vars,axis=1),['Avg. Twater'],np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)


#%%
# LOAD WEATHER VARIABLES OR COMPUTE THEM
load_weather = False
save_weather = False

if load_weather:

    weather_data = np.load('weather_vars_all.npz',allow_pickle='TRUE')
    # weather_data = np.load('weather_vars2_all.npz',allow_pickle='TRUE')

    vars_all = weather_data['weather_vars']
    avg_freezeup_doy = weather_data['avg_freezeup_doy']
    varnames = weather_data['varnames']
    locnames = weather_data['locnames']
    years = weather_data['years']
    window_arr = weather_data['window_arr']
    deseasonalize = weather_data['deseasonalize']
    end_dates_arr = weather_data['end_dates_arr']
    enddate_labels = weather_data['enddate_labels']

else:
    # weather_loc_list = ['D','A','B','E']
    weather_loc_list = ['D']

    weather_varnames = ['Avg. Ta_max',
                        'Avg. Ta_min',
                        'Avg. Ta_mean',
                        'Tot. TDD',
                        'Tot. FDD',
                        'Tot. CDD',
                        'Tot. precip.',
                        'Avg. SLP',
                        'Avg. wind speed',
                        'Avg. u-wind',
                        'Avg. v-wind'
                        ]

    weather_varnames2 = ['Tot. snowfall',
                          'Avg. cloud cover',
                          'Avg. spec. hum.',
                          'Avg. rel. hum.'
                          ]

    weather_vars_all = np.zeros((len(weather_varnames),len(years),len(window_arr),end_dates_arr.shape[1],2,len(weather_loc_list)))*np.nan
    weather_vars2_all = np.zeros((len(weather_varnames2),len(years),len(window_arr),end_dates_arr.shape[1],2,len(weather_loc_list)))*np.nan

    # ADD YUL STATION TO WEATHER_VARS FIRST
    weather_loc = 'MontrealDorvalMontrealPETMontrealMcTavishmerged'
    # weather_data_YUL = np.load(fp+'weather_NCEI/weather_NCEI_'+weather_loc+'.npz',allow_pickle='TRUE')
    # weather_YUL = weather_data_YUL['weather_data']

    # # weather_vars = ['TIME','MAX','MIN','TEMP','DEWP','PRCP','SLP','WDSP']
    #     # MAX - Maximum temperature reported during the day in Fahrenheit to tenths. Missing = 9999.9
    #     # MIN - Minimum temperature reported during the day in Fahrenheit to tenths. Missing = 9999.9
    #     # TEMP - Mean temperature for the day in degrees Fahrenheit to tenths. Missing = 9999.9
    #     # DEWP - Mean dew point for the day in degrees Fahrenheit to tenths. Missing = 9999.9
    #     # PRCP - Total precipitation (rain and/or melted snow) reported during the day in inches
    #     #       and hundredths; will usually not end with the midnight observation (i.e. may include
    #     #       latter part of previous day). “0” indicates no measurable precipitation (includes a trace).Missing = 99.99
    #     # SLP - Mean sea level pressure for the day in millibars to tenths. Missing = 9999.9
    #     # WDSP - Mean wind speed for the day in knots to tenths.  Missing = 999.9
    # max_Ta_YUL = weather_YUL[:,1]
    # min_Ta_YUL = weather_YUL[:,2]
    # avg_Ta_YUL = weather_YUL[:,3]
    # precip_YUL = weather_YUL[:,5]
    # slp_YUL = weather_YUL[:,6]
    # windspeed_YUL = weather_YUL[:,7]

    # # Convert Farenheits to Celsius:
    # max_Ta_YUL  = (max_Ta_YUL- 32) * (5/9.)
    # min_Ta_YUL  = (min_Ta_YUL- 32) * (5/9.)
    # avg_Ta_YUL  = (avg_Ta_YUL- 32) * (5/9.)

    # mask_FDD_YUL = (avg_Ta_YUL <= 0)
    # FDD_YUL = avg_Ta_YUL.copy()
    # FDD_YUL[~mask_FDD_YUL] = np.nan

    # mask_TDD_YUL = (avg_Ta_YUL > 0)
    # TDD_YUL = avg_Ta_YUL.copy()
    # TDD_YUL[~mask_TDD_YUL] = np.nan

    # CDD_YUL = avg_Ta_YUL.copy()

    # weather_vars_YUL = np.zeros((len(time),len(weather_varnames)))*np.nan
    # weather_vars_YUL[:,0] = max_Ta_YUL
    # weather_vars_YUL[:,1] = min_Ta_YUL
    # weather_vars_YUL[:,2] = avg_Ta_YUL
    # weather_vars_YUL[:,3] = TDD_YUL
    # weather_vars_YUL[:,4] = -1*FDD_YUL
    # weather_vars_YUL[:,5] = CDD_YUL
    # weather_vars_YUL[:,6] = precip_YUL
    # weather_vars_YUL[:,7] = slp_YUL
    # weather_vars_YUL[:,8] = windspeed_YUL


    # if deseasonalize:
    #     Nwindow = 31
    #     weather_vars_YUL = deasonalize_ts(Nwindow,weather_vars_YUL,weather_varnames,'all_time',time,years)

    # for iend in range(end_dates_arr.shape[1]):
    #     weather_vars_all[:,:,:,iend,:,0]= get_window_vars(weather_vars_YUL,weather_varnames,np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)


    # THEN ADD THE OTHER LOCATIONS FROM ERA5:
    for iloc,weather_loc in enumerate(weather_loc_list):

        weather_data = np.load(fp+'weather_ERA5/weather_ERA5_region'+weather_loc+'.npz',allow_pickle='TRUE')
        weather = weather_data['weather_data']
        weather_vars = weather_data['select_vars']

        max_Ta = weather[:,1] # Daily max. Ta [K]
        min_Ta = weather[:,2] # Daily min. Ta [K]
        avg_Ta = weather[:,3] # Daily avg. Ta [K]
        precip = weather[:,4] # Daily total precip. [m]
        slp = weather[:,5] # Sea level pressure [Pa]
        uwind = weather[:,6] # U-velocity of 10m wind [m/s]
        vwind = weather[:,7] # V-velocity of 10m wind [m/s]
        max_Td = weather[:,9] # Daily max. dew point [K]
        min_Td = weather[:,10] # Daily min. dew point [K]
        avg_Td = weather[:,11] # Daily avg. dew point [K]
        snow = weather[:,12] # Snowfall [m of water equivalent]
        clouds = weather[:,13] # Total cloud cover [%]

        # Convert to kPa:
        slp = slp/1000.
        # Convert Kelvins to Celsius:
        max_Ta  = (max_Ta-273.15)
        min_Ta  = (min_Ta-273.15)
        avg_Ta  = (avg_Ta-273.15)
        max_Td  = (max_Td-273.15)
        min_Td  = (min_Td-273.15)
        avg_Td  = (avg_Td-273.15)

        # Derive new variables:
        windspeed = np.sqrt(uwind**2 + vwind**2)
        e_sat =0.61094*np.exp((17.625*avg_Ta)/(avg_Ta +243.04)) # Saturation vapor pressure (August–Roche–Magnus formula)
        avg_SH = 0.622*e_sat/(slp-0.378*e_sat) # Daily avg. specific humidity
        avg_RH = (np.exp((17.625*avg_Td)/(243.04+avg_Td))/np.exp((17.625*avg_Ta)/(243.04+avg_Ta))) # Daily avg. relative humidity

        mask_FDD = (avg_Ta <= 0)
        FDD = avg_Ta.copy()
        FDD[~mask_FDD] = np.nan

        mask_TDD = (avg_Ta > 0)
        TDD = avg_Ta.copy()
        TDD[~mask_TDD] = np.nan

        CDD = avg_Ta.copy()

        weather_vars = np.zeros((len(time),len(weather_varnames)))*np.nan
        weather_vars[:,0] = max_Ta
        weather_vars[:,1] = min_Ta
        weather_vars[:,2] = avg_Ta
        weather_vars[:,3] = TDD
        weather_vars[:,4] = -1*FDD
        weather_vars[:,5] = CDD
        weather_vars[:,6] = precip
        weather_vars[:,7] = slp
        weather_vars[:,8] = windspeed
        weather_vars[:,9] = uwind
        weather_vars[:,10] = vwind

        weather_vars2 = np.zeros((len(time),len(weather_varnames2)))*np.nan
        weather_vars2[:,0] = snow
        weather_vars2[:,1] = clouds
        weather_vars2[:,2] = avg_SH
        weather_vars2[:,3] = avg_RH

        if deseasonalize:
            Nwindow = 31
            weather_vars = deasonalize_ts(Nwindow,weather_vars,weather_varnames,'all_time',time,years)
            weather_vars2 = deasonalize_ts(Nwindow,weather_vars2,weather_varnames2,'all_time',time,years)

        # Separate in different windows with different end dates
        for iend in range(end_dates_arr.shape[1]):
            weather_vars_all[:,:,:,iend,:,iloc] = get_window_vars(weather_vars,weather_varnames,np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)
            weather_vars2_all[:,:,:,iend,:,iloc] = get_window_vars(weather_vars2,weather_varnames2,np.squeeze(end_dates_arr[:,iend]),window_arr,years,time,month_start_day)

    if save_weather:
        save_path = './'
        savename = 'weather_vars_all_monthly'
        locnames = ['NCEI\nYUL','ERA5\nMLO+OR','ERA5\nMontreal','ERA5\nLake Ontario','ERA5\nOttawa River']
        np.savez(save_path+savename,
                  avg_freezeup_doy = avg_freezeup_doy,
                  weather_vars= weather_vars_all,
                  varnames = weather_varnames,
                  locnames = locnames,
                  years = years,
                  window_arr = window_arr,
                  deseasonalize = deseasonalize,
                  end_dates_arr = end_dates_arr,
                  enddate_labels = enddate_labels)


        save_path = './'
        savename2 = 'weather_vars2_all_monthly'
        locnames2 = ['ERA5\nMLO+OR','ERA5\nMontreal','ERA5\nLake Ontario','ERA5\nOttawa River']
        np.savez(save_path+savename2,
                  avg_freezeup_doy = avg_freezeup_doy,
                  weather_vars=weather_vars2_all,
                  varnames = weather_varnames2,
                  locnames = locnames2,
                  years = years,
                  window_arr = window_arr,
                  deseasonalize = deseasonalize,
                  end_dates_arr = end_dates_arr,
                  enddate_labels = enddate_labels)


#%%
# CHOOSE X VARS
vars_all = np.expand_dims(Twater_vars_all.copy(),axis=0)
vars_all = np.expand_dims(vars_all.copy(),axis=-1)
varnames = Twater_varnames
# locnames = ['NCEI\nYUL','ERA5\nMLO+OR','ERA5\nMontreal','ERA5\nLake Ontario','ERA5\nOttawa River']
locnames = ['Mean Tw. near Montreal']

vars_all = weather_vars_all
varnames = weather_varnames
# locnames = ['NCEI\nYUL','ERA5\nMLO+OR','ERA5\nMontreal','ERA5\nLake Ontario','ERA5\nOttawa River']
locnames = ['ERA5\nMLO+OR','ERA5\nMontreal','ERA5\nLake Ontario','ERA5\nOttawa River']

# vars_all = weather_vars2_all
# varnames = weather_varnames2
# locnames = ['ERA5\nMLO+OR','ERA5\nMontreal','ERA5\nLake Ontario','ERA5\nOttawa River']

#%%
# CHOOSE Y VAR
# i) freeze-up
# yvars_all = avg_freezeup_doy

# ii) Twater in December
tmp_vars_all = np.expand_dims(Twater_vars_all.copy(),axis=0)
tmp_vars_all = np.expand_dims(tmp_vars_all.copy(),axis=-1)
yvars_all = np.squeeze(tmp_vars_all[0,:,0,0,0,0].copy())

#%%

nvars = vars_all.shape[0]
nyears = vars_all.shape[1]
nwindows = vars_all.shape[2]
nend = vars_all.shape[3]
nwindowtype = vars_all.shape[4]
nlocs = vars_all.shape[5]


if detrend:
    vars_all_detrended = np.zeros(vars_all.shape)*np.nan
    for ivar in range(nvars):
        for iw in range(nwindows):
            for iend in range(nend):
                for ip in range(nwindowtype):
                    for iloc in range(nlocs):
                        xvar = vars_all[ivar,:,iw,iend,ip,iloc]
                        yvar = yvars_all

                        vars_all_detrended[ivar,:,iw,iend,ip,iloc], yvars_all_detrended = detrend_ts(xvar,yvar,years,anomaly)
else:
    vars_all_detrended = vars_all.copy()
    yvars_all_detrended = yvars_all.copy()


r_mean = np.zeros((nvars,nwindows,nend,nwindowtype,nlocs))*np.nan
r_std = np.zeros((nvars,nwindows,nend,nwindowtype,nlocs))*np.nan
for ivar in range(nvars):
    for iw in range(nwindows):
        for iend in range(nend):
            for ip in range(nwindowtype):
                for iloc in range(nlocs):

                    print('Bootstrap '+ varnames[ivar]+' - %3i'%(iw)+'/%3i'%(nwindows)+', %3i'%(iend)+'/%3i'%(nend)+', %3i'%(ip)+'/%3i'%(nwindowtype)+', %3i'%(iloc)+'/%3i'%(nlocs))

                    xvar = vars_all_detrended[ivar,:,iw,iend,ip,iloc]
                    yvar = yvars_all_detrended

                    r = bootstrap(xvar,yvar,nboot)
                    r_mean[ivar,iw,iend,ip,iloc] = np.nanmean(r)
                    r_std[ivar,iw,iend,ip,iloc] = np.nanstd(r)


#%%
pc = p_critical

rc_m1, rc_p1 = r_confidence_interval(0,pc,nyears,tailed='one')
rc_m2, rc_p2 = r_confidence_interval(0,pc,nyears,tailed='two')
plot_colors = [plt.get_cmap('tab20b')(0),plt.get_cmap('tab20b')(4),plt.get_cmap('tab20b')(8),plt.get_cmap('tab20b')(12)]

# for iend in range(0,1):
for iend in range(1,2):
# for iend in range(2,3):
# for iend in range(3,4):
# for iend in range(nend):
    enddate_str = enddate_labels[iend]

    for ivar in range(nvars):
        var1 = varnames[ivar]
        nrows = nlocs
        ncols = 1
        fig,ax = plt.subplots(nrows,figsize=(5,(nlocs)*(10/5.)),sharex=True,sharey=True,squeeze=False)
        if (nrows == 1) | (ncols == 1) :
            ax = ax.reshape(-1)
        plt.suptitle(var1)

        for ip in range(nwindowtype):
            for iloc in range(nlocs):
                r_mean_plot = r_mean[ivar,:,iend,ip,iloc]
                r_std_plot = r_std[ivar,:,iend,ip,iloc]

                ax[iloc].plot(window_arr,r_mean_plot,'.-',color=plot_colors[ip])
                ax[iloc].fill_between(window_arr,r_mean_plot+r_std_plot,r_mean_plot-r_std_plot,color=plot_colors[ip],alpha=0.15)

                ax[iloc].plot(window_arr,np.ones(len(window_arr))*rc_p2,':', color='gray')
                ax[iloc].plot(window_arr,np.ones(len(window_arr))*rc_m2,':', color='gray')

                plt.subplots_adjust(left=0.2,right=0.9,bottom=0.23)
                ax[iloc].set_xlim(0,np.nanmax(window_arr)+np.nanmax(window_arr)/10.)
                ax[iloc].set_ylim(-1,1)
                ax[iloc].set_ylabel('r\n'+locnames[iloc],fontsize=10)
                ax[iloc].grid()

                if iloc == nlocs-1:
                    if (window_arr[1]-window_arr[0]) == 7:
                        ax[iloc].set_xlabel('Previous X weeks (XW) from '+enddate_str,fontsize=10)
                        labels_list = [str(np.arange(1,len(window_arr)+1,2)[i])+'W' for i in range(len(np.arange(1,len(window_arr)+1,2)))]
                        labels_list = np.insert(labels_list, np.arange(1,len(labels_list)+1) ,'')

                    if (window_arr[1]-window_arr[0]) == 30:
                        ax[iloc].set_xlabel('Previous X months (XM) from '+enddate_str,fontsize=10)
                        labels_list = [str(np.arange(1,len(window_arr)+1,2)[i])+'M' for i in range(len(np.arange(1,len(window_arr)+1,2)))]
                        labels_list = np.insert(labels_list, np.arange(1,len(labels_list)+1) ,'')

                    ax[iloc].set_xticks(window_arr)
                    ax[iloc].set_xticklabels(labels_list)







#%%
ip = 0
iend = 1
iloc = 0
ivar = 0

vars_monthly = vars_all_detrended[ivar,:,:,iend,ip,iloc].copy()

x = vars_monthly[:,0] # Nov Tw
# x = vars_monthly[:,6] # May Tw
y = yvars_all_detrended[:]
years_plot = years[:]

fig, ax = plt.subplots()
ax.plot(years_plot,y,'o-', color= plt.get_cmap('tab10')(0))
ax.set_xlabel('Year')
ax.set_ylabel('Freezeup Anomaly', color= plt.get_cmap('tab10')(0))
ax.grid()
ax2=ax.twinx()
ax2.plot(years_plot,x,'o--', color= plt.get_cmap('tab10')(1))
ax2.set_ylabel('Detrended Tw Anomaly in Nov.', color= plt.get_cmap('tab10')(1))
lincoeff, Rsqr = linear_fit(x,y)
r =np.sqrt(Rsqr)
if lincoeff[0]<0: r*=-1
ax.text(2000,21,'r = %3.2f'%(r))

#%%
ip = 0
iend = 0
iloc = 0
ivar = 0

vars_monthly = vars_all_detrended[ivar,:,:,iend,ip,iloc].copy()

x = vars_monthly[:,0] # Dec Tw
y = yvars_all_detrended[:]
years_plot = years[:]

fig, ax = plt.subplots()
ax.plot(years_plot,y,'o-', color= plt.get_cmap('tab10')(0))
ax.set_xlabel('Year')
ax.set_ylabel('Freezeup Anomaly', color= plt.get_cmap('tab10')(0))
ax.grid()
ax2=ax.twinx()
ax2.plot(years_plot,x,'o--', color= plt.get_cmap('tab10')(1))
ax2.set_ylabel('Detrended Tw Anomaly in Dec.', color= plt.get_cmap('tab10')(1))
lincoeff, Rsqr = linear_fit(x,y)
r =np.sqrt(Rsqr)
if lincoeff[0]<0: r*=-1
ax.text(2000,21,'r = %3.2f'%(r))


#%%
# ip = 0
# iend = 0
# ivar = 4 #FDD
# iloc = 1


# x = -1*vars_all_detrended[4,:,1,iend,ip,1].copy()
# y = -1*vars_all_detrended[4,:,33,iend,ip,1].copy()
# varplot = vars_all[4,:,33,iend,ip,1].copy()
# fig, ax = plt.subplots()
# ax.plot(years,x,'o-', color= plt.get_cmap('tab10')(0))
# ax.set_xlabel('Year')
# ax.set_ylabel('Weekly accumulated FDD\nat -2W from Dec 1st', color= plt.get_cmap('tab10')(0))
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(years,y,'o--', color= plt.get_cmap('tab10')(1))
# ax2.set_ylabel('Weekly accumulated FDD\nat -34W from Dec 1st', color= plt.get_cmap('tab10')(1))
# lincoeff, Rsqr = linear_fit(x,y)
# r =np.sqrt(Rsqr)
# if lincoeff[0]<0: r*=-1
# ax.text(2016,-10,'r = %3.2f'%(r))

# varplot = -1*vars_all_detrended[4,:,1,iend,ip,1].copy()
# # varplot = vars_all[4,:,1,iend,ip,1].copy()
# # varplot[varplot==0] = np.nan
# nv = len(varplot[~np.isnan(varplot)])
# rc_m2, rc_p2 = r_confidence_interval(0,pc,nv,tailed='two')
# print(rc_m2,rc_p2)

# fig, ax = plt.subplots()
# ax.plot(years,yvars_all_detrended,'o-', color= plt.get_cmap('tab10')(0))
# ax.set_xlabel('Year')
# ax.set_ylabel('Freezeup Anomaly', color= plt.get_cmap('tab10')(0))
# ax.grid()
# ax.set_ylim([-30,30])
# ax2=ax.twinx()
# ax2.plot(years,varplot,'o--', color= plt.get_cmap('tab10')(1))
# ax2.set_ylabel('Weekly accumulated FDD\nat -2W from Dec 1st', color= plt.get_cmap('tab10')(1))
# lincoeff, Rsqr = linear_fit(varplot,yvars_all_detrended)
# r =np.sqrt(Rsqr)
# if lincoeff[0]<0: r*=-1
# ax.text(2016,21,'r = %3.2f'%(r))
# ax2.set_ylim([-30,30])

# varplot = vars_all_detrended[1,:,14,iend,ip,1].copy()
# # varplot = vars_all[1,:,15,iend,ip,1].copy()
# # varplot[varplot==0] = np.nan
# nv = len(varplot[~np.isnan(varplot)])
# rc_m2, rc_p2 = r_confidence_interval(0,pc,nv,tailed='two')
# print(rc_m2,rc_p2)

# fig, ax = plt.subplots()
# ax.plot(years,yvars_all_detrended,'o-', color= plt.get_cmap('tab10')(0))
# ax.set_xlabel('Year')
# ax.set_ylabel('Freezeup Anomaly', color= plt.get_cmap('tab10')(0))
# ax.grid()
# ax.set_ylim([-30,30])
# ax2=ax.twinx()
# ax2.plot(years,varplot,'o--', color= plt.get_cmap('tab10')(1))
# ax2.set_ylabel('Weekly accumulated FDD\nat -16W from Dec 1st', color= plt.get_cmap('tab10')(1))
# lincoeff, Rsqr = linear_fit(varplot,yvars_all_detrended)
# r =np.sqrt(Rsqr)
# if lincoeff[0]<0: r*=-1
# ax.text(2016,21,'r = %3.2f'%(r))
# ax2.set_ylim([-30,30])



#%%

# x = vars_all[ivar,:,:,iend,ip,iloc].copy()
# x = x[:,7]

# fig, ax = plt.subplots()
# ax.plot(years[:],yvars_all_detrended[:],'o-', color= plt.get_cmap('tab10')(0))
# ax.set_xlabel('Year')
# ax.set_ylabel('Freezeup Anomaly', color= plt.get_cmap('tab10')(0))
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(years[:],x,'o--', color= plt.get_cmap('tab10')(1))
# ax2.set_ylabel('FDD in April', color= plt.get_cmap('tab10')(1))
# lincoeff, Rsqr = linear_fit(x,yvars_all_detrended[:])
# r =np.sqrt(Rsqr)
# if lincoeff[0]<0: r*=-1
# ax.text(2000,21,'r = %3.2f'%(r))


#%%
# ip = 0
# iend = 0
# iloc = 0
# ivar = 0

# vars_monthly = vars_all_detrended[ivar,:,:,iend,ip,iloc].copy()
# # vars_monthly = vars_all[ivar,:,:,iend,ip,iloc]

# # x = vars_monthly[0:16,0] # Nov snowfall
# x = vars_monthly[0:16,7] # April snowfall
# fig, ax = plt.subplots()
# ax.plot(years[0:16],yvars_all_detrended[0:16],'o-', color= plt.get_cmap('tab10')(0))
# ax.set_xlabel('Year')
# ax.set_ylabel('Freezeup Anomaly', color= plt.get_cmap('tab10')(0))
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(years[0:16],x,'o--', color= plt.get_cmap('tab10')(1))
# ax2.set_ylabel('Detrended snowfall anomaly in April', color= plt.get_cmap('tab10')(1))
# # ax2.set_ylabel('Detrended snowfall anomaly in Nov', color= plt.get_cmap('tab10')(1))
# lincoeff, Rsqr = linear_fit(x,yvars_all_detrended[0:16])
# r =np.sqrt(Rsqr)
# if lincoeff[0]<0: r*=-1
# ax.text(2000,21,'r = %3.2f'%(r))


# x = vars_all[ivar,:,:,iend,ip,iloc].copy()
# # x = x[0:16,0]
# x = x[0:16,7]
# fig, ax = plt.subplots()
# ax.plot(years[0:16],yvars_all_detrended[0:16],'o-', color= plt.get_cmap('tab10')(0))
# ax.set_xlabel('Year')
# ax.set_ylabel('Freezeup Anomaly', color= plt.get_cmap('tab10')(0))
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(years[0:16],x,'o--', color= plt.get_cmap('tab10')(1))
# ax2.set_ylabel('Snowfall in April', color= plt.get_cmap('tab10')(1))
# # ax2.set_ylabel('Snowfall in Nov.', color= plt.get_cmap('tab10')(1))
# lincoeff, Rsqr = linear_fit(x,yvars_all_detrended[0:16])
# r =np.sqrt(Rsqr)
# if lincoeff[0]<0: r*=-1
# ax.text(2000,21,'r = %3.2f'%(r))

#%%

# ip = 1
# iend = 0
# iloc = 0
# ivar = 7

# vars_monthly = vars_all_detrended[ivar,:,:,iend,ip,iloc].copy()
# # vars_monthly = vars_all[ivar,:,:,iend,ip,iloc]

# # x = vars_monthly[0:16,0] # Nov snowfall
# x = vars_monthly[:,0] # April snowfall

# y = yvars_all_detrended[:]

# fig, ax = plt.subplots()
# ax.plot(years[:],y,'o-', color= plt.get_cmap('tab10')(0))
# ax.set_xlabel('Year')
# ax.set_ylabel('Freezeup Anomaly', color= plt.get_cmap('tab10')(0))
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(years[:],x,'o--', color= plt.get_cmap('tab10')(1))
# ax2.set_ylabel('Detrended SLP anomaly in Nov.', color= plt.get_cmap('tab10')(1))
# # ax2.set_ylabel('Detrended snowfall anomaly in Nov', color= plt.get_cmap('tab10')(1))
# lincoeff, Rsqr = linear_fit(x,y)
# r =np.sqrt(Rsqr)
# if lincoeff[0]<0: r*=-1
# ax.text(2000,21,'r = %3.2f'%(r))




#%%
# x = vars_all[ivar,:,:,iend,ip,iloc].copy()
# x = x[0:16,7]

# fig, ax = plt.subplots()
# ax.plot(years[0:16],yvars_all_detrended[0:16],'o-', color= plt.get_cmap('tab10')(0))
# ax.set_xlabel('Year')
# ax.set_ylabel('Freezeup Anomaly', color= plt.get_cmap('tab10')(0))
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(years[0:16],x,'o--', color= plt.get_cmap('tab10')(1))
# ax2.set_ylabel('FDD in April', color= plt.get_cmap('tab10')(1))
# lincoeff, Rsqr = linear_fit(x,yvars_all_detrended[0:16])
# r =np.sqrt(Rsqr)
# if lincoeff[0]<0: r*=-1
# ax.text(2000,21,'r = %3.2f'%(r))




#%%

# fig, ax = plt.subplots()
# plt.plot(years,vars_monthly[:,7],'.-')
# plt.xlabel('Year')
# plt.ylabel('FDD Anomaly')
# ax.grid()

# fig, ax = plt.subplots()
# ax.plot(varplot,yvars_all_detrended,'o')
# ax.set_ylabel('Freezeup Anomaly')
# ax.set_xlabel('Weekly accumulated FDD\nat -34W from Dec 1st')
# lincoeff, Rsqr = linear_fit(varplot,yvars_all_detrended)
# r =np.sqrt(Rsqr)
# if lincoeff[0]<0: r*=-1
# ax.text(9,0,'r = %3.2f'%(r))
# ax.plot(varplot,varplot*lincoeff[0]+lincoeff[1],'-',color='gray')


#%%
# x = vars_all[1,0:16,0,iend,ip,iloc].copy() # Precip in Nov.
# # x = vars_all[6,0:16,0,iend,ip,iloc].copy() # Precip in Nov.
# # y = vars_all[7,0:16,0,iend,ip,iloc].copy() # SLP in Nov.

# fig, ax = plt.subplots()
# ax.plot(years[0:16],x,'o-', color= plt.get_cmap('tab10')(0))
# ax.set_xlabel('Year')
# ax.set_ylabel('Clouds in Nov. ', color= plt.get_cmap('tab10')(0))
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(years[0:16],y,'o--', color= plt.get_cmap('tab10')(1))
# ax2.set_ylabel('SLP in Nov. ', color= plt.get_cmap('tab10')(1))
# # ax2.set_ylabel('Snowfall in Nov.', color= plt.get_cmap('tab10')(1))
# lincoeff, Rsqr = linear_fit(x,y)
# r =np.sqrt(Rsqr)
# if lincoeff[0]<0: r*=-1
# ax.text(2000,x.max(),'r = %3.2f'%(r))









#%%
# fig, ax = plt.subplots()
# ax.plot(varplot,yvars_all_detrended,'o')
# ax.set_ylabel('Freezeup Anomaly')
# ax.set_xlabel('Weekly accumulated FDD\nat -34W from Dec 1st')

# lincoeff, Rsqr = linear_fit(varplot,yvars_all_detrended)
# r =np.sqrt(Rsqr)
# if lincoeff[0]<0: r*=-1
# ax.text(9,0,'r = %3.2f'%(r))
# ax.plot(varplot,varplot*lincoeff[0]+lincoeff[1],'-',color='gray')

# #%%
# # fig, ax = plt.subplots()
# # plt.plot(years,yvars_all_detrended,'.-')
# # plt.xlabel('Year')
# # plt.ylabel('Freezeup Anomaly')
# # ax.grid()
