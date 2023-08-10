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

from functions import running_nanmean,find_freezeup_Tw,season_mask
from functions import linear_fit

#%%

def mask_season_vars(vars_in, names_in, years, time, month_start_day=1):
    nvars = len(names_in)
    vars_out = np.zeros((nvars,len(years),4))

    for iyr, year in enumerate(years):

        i0 = (dt.date(int(year),3,month_start_day)-date_ref).days
        i0 = np.where(time == i0)[0][0]

        i1 = (dt.date(int(year)+1,3,month_start_day)-date_ref).days
        try:
            i1 = np.where(time == i1)[0][0]
        except:
            i1 = len(time)-1


        time_year = time[i0:i1]

        mask_spring = season_mask(time_year,'spring',msd=month_start_day)
        mask_summer = season_mask(time_year,'summer',msd=month_start_day)
        mask_fall = season_mask(time_year,'fall',msd=month_start_day)
        mask_winter = season_mask(time_year,'winter',msd=month_start_day)

        for ivar in range(nvars):
            var_year = vars_in[i0:i1,ivar]
            varname = names_in[ivar]

            if (varname[0:3] == 'Avg'):
                vars_out[ivar,iyr,0] = np.nanmean(var_year[mask_spring])
                vars_out[ivar,iyr,1] = np.nanmean(var_year[mask_summer])
                vars_out[ivar,iyr,2] = np.nanmean(var_year[mask_fall])
                vars_out[ivar,iyr,3] = np.nanmean(var_year[mask_winter])

            if (varname[0:3] == 'Tot'):
                vars_out[ivar,iyr,0] = np.nansum(var_year[mask_spring])
                vars_out[ivar,iyr,1] = np.nansum(var_year[mask_summer])
                vars_out[ivar,iyr,2] = np.nansum(var_year[mask_fall])
                vars_out[ivar,iyr,3] = np.nansum(var_year[mask_winter])

            # if (varname[-3:] == 'FDD'):
            #     # if 'Avg. Ta_min' in names_in:
            #     #     ivar_minTa = names_in.index('Avg. Ta_min')
            #     if 'Avg. Ta_mean' in names_in:
            #         ivar_minTa = names_in.index('Avg. Ta_mean')
            #         min_Ta_year = vars_in[i0:i1,ivar_minTa]

            #         vars_out[ivar,iyr,0] = np.nansum(min_Ta_year[mask_spring][min_Ta_year[mask_spring]<=0])
            #         vars_out[ivar,iyr,1] = np.nansum(min_Ta_year[mask_summer][min_Ta_year[mask_summer]<=0])
            #         vars_out[ivar,iyr,2] = np.nansum(min_Ta_year[mask_fall][min_Ta_year[mask_fall]<=0])
            #         vars_out[ivar,iyr,3] = np.nansum(min_Ta_year[mask_winter][min_Ta_year[mask_winter]<=0])
            #     else:
            #         print('Cannot compute FDD... Need Ta_min in vars_in')
            #         continue

            # if (varname[-3:] == 'TDD'):
            #     # if 'Avg. Ta_max' in names_in:
            #     #     ivar_maxTa = names_in.index('Avg. Ta_max')
            #     if 'Avg. Ta_mean' in names_in:
            #         ivar_maxTa = names_in.index('Avg. Ta_mean')
            #         max_Ta_year = vars_in[i0:i1,ivar_maxTa]

            #         vars_out[ivar,iyr,0] = np.nansum(max_Ta_year[mask_spring][max_Ta_year[mask_spring]>0])
            #         vars_out[ivar,iyr,1] = np.nansum(max_Ta_year[mask_summer][max_Ta_year[mask_summer]>0])
            #         vars_out[ivar,iyr,2] = np.nansum(max_Ta_year[mask_fall][max_Ta_year[mask_fall]>0])
            #         vars_out[ivar,iyr,3] = np.nansum(max_Ta_year[mask_winter][max_Ta_year[mask_winter]>0])
            #     else:
            #         print('Cannot compute TDD... Need Ta_max in vars_in')
            #         continue



    return vars_out



def multivars_scatter_plot(iw,xvars,yvar,varnames,ystart,yend):

    season_colors = [plt.get_cmap('tab20')(4),plt.get_cmap('tab20')(6),plt.get_cmap('tab20')(2),plt.get_cmap('tab20')(0)]

    xvars = xvars[:,:,:,iw] # select weather region
    nvars = xvars.shape[0]
    yr_start = int(np.where(years == ystart)[0][0])
    yr_end = int(np.where(years == yend)[0][0])

    if len(yvar.shape) == 1:
        yvar = np.array([yvar,]*4).transpose() # Repeat the same values for all seasons

    fig,ax = plt.subplots(nrows=4,ncols=nvars,figsize=((nvars)*(8/5.),8),sharey='row')

    if nvars > 1:
        for ivar,var1 in enumerate(varnames):
            for iseason in range(4):
                ax[iseason,ivar].plot(xvars[ivar,yr_start:yr_end,iseason],yvar[yr_start:yr_end,iseason],'o',color=season_colors[iseason],alpha=0.5)
                xvar_min = np.nanmin(xvars[ivar,yr_start:yr_end,iseason])
                xvar_max = np.nanmax(xvars[ivar,yr_start:yr_end,iseason])
                xvar_range = xvar_max-xvar_min
                ax[iseason,ivar].set_xlim(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)
                yvar_min = np.nanmin(yvar[yr_start:yr_end,iseason])
                yvar_max = np.nanmax(yvar[yr_start:yr_end,iseason])
                yvar_range = yvar_max-yvar_min
                ax[iseason,ivar].set_ylim(yvar_min-yvar_range/5.,yvar_max+2*yvar_range/5.)

                x_fit = xvars[ivar,yr_start:yr_end,iseason]
                y_fit = yvar[yr_start:yr_end,iseason]
                mask_x = ~np.isnan(x_fit)
                mask_y = ~np.isnan(y_fit)
                x_fit = x_fit[mask_x&mask_y]
                y_fit = y_fit[mask_x&mask_y]
                lincoeff, Rsqr = linear_fit(x_fit,y_fit)
                ax[iseason,ivar].text(xvar_min-xvar_range/10.,yvar_max+yvar_range/6.,'$R^{2}$: '+'%.2f'%(Rsqr),fontsize=8,color=[0.5,0.5,0.5])

                xplot = np.arange(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.,xvar_range/10.)
                yplot = lincoeff[0]*xplot + lincoeff[1]
                ax[iseason,ivar].plot(xplot,yplot,'-',color=[0.5,0.5,0.5])

                if ivar ==0:
                    ax[iseason,ivar].set_ylabel(season_list[iseason],fontsize=10)
                if iseason == 0:
                    ax[iseason,ivar].xaxis.set_label_position('top')
                    ax[iseason,ivar].set_xlabel(varnames[ivar],fontsize=10)

        if (nvars)*(8/5.) > 12.75:
            plt.subplots_adjust(left=0.063,right=0.9508)

    else:
        for ivar,var1 in enumerate(varnames):
            for iseason in range(4):
                ax[iseason].plot(xvars[ivar,yr_start:yr_end,iseason],yvar[yr_start:yr_end,iseason],'o',color=season_colors[iseason],alpha=0.5)
                xvar_min = np.nanmin(xvars[ivar,yr_start:yr_end,iseason])
                xvar_max = np.nanmax(xvars[ivar,yr_start:yr_end,iseason])
                xvar_range = xvar_max-xvar_min
                ax[iseason].set_xlim(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)
                yvar_min = np.nanmin(yvar[yr_start:yr_end,iseason])
                yvar_max = np.nanmax(yvar[yr_start:yr_end,iseason])
                yvar_range = yvar_max-yvar_min
                ax[iseason].set_ylim(yvar_min-yvar_range/5.,yvar_max+2*yvar_range/5.)

                x_fit = xvars[ivar,yr_start:yr_end,iseason]
                y_fit = yvar[yr_start:yr_end,iseason]
                mask_x = ~np.isnan(x_fit)
                mask_y = ~np.isnan(y_fit)
                x_fit = x_fit[mask_x&mask_y]
                y_fit = y_fit[mask_x&mask_y]
                lincoeff, Rsqr = linear_fit(x_fit,y_fit)
                ax[iseason].text(xvar_min-xvar_range/10.,yvar_max+yvar_range/6.,'$R^{2}$: '+'%.2f'%(Rsqr),fontsize=8,color=[0.5,0.5,0.5])

                xplot = np.arange(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.,xvar_range/10.)
                yplot = lincoeff[0]*xplot + lincoeff[1]
                ax[iseason].plot(xplot,yplot,'-',color=[0.5,0.5,0.5])

                if ivar ==0:
                    ax[iseason].set_ylabel(season_list[iseason],fontsize=10)
                if iseason == 0:
                    ax[iseason].xaxis.set_label_position('top')
                    ax[iseason].set_xlabel(varnames[ivar],fontsize=10)



def multilocs_scatter_plot(xvars,yvar,varnames,locnames,ystart,yend):

    season_colors = [plt.get_cmap('tab20')(4),plt.get_cmap('tab20')(6),plt.get_cmap('tab20')(2),plt.get_cmap('tab20')(0)]

    nlocs = xvars.shape[-1]
    yr_start = int(np.where(years == ystart)[0][0])
    yr_end = int(np.where(years == yend)[0][0])

    if len(yvar.shape) == 1:
        yvar = np.array([yvar,]*4).transpose() # Repeat the same values for all seasons

    for ivar,var1 in enumerate(varnames):
        fig,ax = plt.subplots(nrows=4,ncols=nlocs,figsize=((nlocs)*(8/5.),8),sharey='row')
        plt.suptitle(var1)
        for iloc in range(nlocs):
            for iseason in range(4):
                ax[iseason,iloc].plot(xvars[ivar,yr_start:yr_end,iseason,iloc],yvar[yr_start:yr_end,iseason],'o',color=season_colors[iseason],alpha=0.5)
                xvar_min = np.nanmin(xvars[ivar,yr_start:yr_end,iseason,iloc])
                xvar_max = np.nanmax(xvars[ivar,yr_start:yr_end,iseason,iloc])
                xvar_range = xvar_max-xvar_min
                ax[iseason,iloc].set_xlim(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)
                yvar_min = np.nanmin(yvar[yr_start:yr_end,iseason])
                yvar_max = np.nanmax(yvar[yr_start:yr_end,iseason])
                yvar_range = yvar_max-yvar_min
                ax[iseason,iloc].set_ylim(yvar_min-yvar_range/5.,yvar_max+2*yvar_range/5.)

                x_fit = xvars[ivar,yr_start:yr_end,iseason,iloc]
                y_fit = yvar[yr_start:yr_end,iseason]
                mask_x = ~np.isnan(x_fit)
                mask_y = ~np.isnan(y_fit)
                x_fit = x_fit[mask_x&mask_y]
                y_fit = y_fit[mask_x&mask_y]
                lincoeff, Rsqr = linear_fit(x_fit,y_fit)

                data = pandas.DataFrame({'x': x_fit, 'y': y_fit})
                model = ols("y ~ x", data).fit() # Fit the model to get p-value from F-test

                if xvar_range > 0:
                    xplot = np.arange(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.,xvar_range/10.)
                    yplot = lincoeff[0]*xplot + lincoeff[1]
                # if np.round(Rsqr,2) >=0.1:
                if model.f_pvalue <= 0.01:
                    ax[iseason,iloc].text(xvar_min-xvar_range/10.,yvar_max+yvar_range/6.,'R$^{2}$: '+'%.2f'%(Rsqr),fontsize=8,color=[0.15,0.15,0.15], fontweight='bold')
                    ax[iseason,iloc].plot(xplot,yplot,'-',color=[0.15,0.15,0.15])
                else:
                    ax[iseason,iloc].text(xvar_min-xvar_range/10.,yvar_max+yvar_range/6.,'R$^{2}$: '+'%.2f'%(Rsqr),fontsize=8,color=[0.5,0.5,0.5])
                    ax[iseason,iloc].plot(xplot,yplot,'-',color=[0.5,0.5,0.5])

                if iloc ==0:
                    ax[iseason,iloc].set_ylabel(season_list[iseason],fontsize=10)
                if iseason == 0:
                    ax[iseason,iloc].xaxis.set_label_position('top')
                    ax[iseason,iloc].set_xlabel(locnames[iloc],fontsize=10)

        if (nlocs)*(8/5.) > 12.75:
            plt.subplots_adjust(left=0.063,right=0.9508)



#%%
years = [1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020]

season_list = ['spring','summer','fall','winter']
fp = '../../data/processed/'

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2020,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)
years = np.array(years)

#%%
# LOAD FREEZEUP DATES OR FIND FROM TWATER TIME SERIES

water_name_list = ['Atwater_cleaned_filled','DesBaillets_cleaned_filled','Longueuil_cleaned_filled','Candiac_cleaned_filled']
station_labels = ['Atwater','DesBaillets','Longueuil','Candiac']
station_type = 'cities'

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
            fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw(def_opt,Twater_tmp,Twater_DoG1[:,iloc],Twater_DoG2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
            freezeup_dates[:,:,iloc] = fd
            freezeup_temp[:,iloc] = ftw
        else:
            fd, ftw, T_freezeup, mask_freeze = find_freezeup_Tw(def_opt,Twater_tmp,Twater_dTdt[:,iloc],Twater_d2Tdt2[:,iloc],time,years,thresh_T = T_thresh,thresh_dTdt = dTdt_thresh,thresh_d2Tdt2 = d2Tdt2_thresh,ndays = nd)
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
# avg_freezeup_doy = np.nanmin(freezeup_doy,axis=1)
# avg_freezeup_doy = np.nanmax(freezeup_doy,axis=1)

mask_years_p18 = avg_freezeup_doy < np.nanpercentile(avg_freezeup_doy,18)
mask_years_p82 = avg_freezeup_doy > np.nanpercentile(avg_freezeup_doy,82)

# plt.figure()
# plt.plot(avg_freezeup_doy,'o')
# plt.fill_between(np.arange(len(years)),np.ones(len(years))*(np.nanpercentile(avg_freezeup_doy,18)),np.ones(len(years))*(np.nanpercentile(avg_freezeup_doy,82)),facecolor=[0.9, 0.9, 0.9], interpolate=True, alpha=0.65)
# plt.plot(np.arange(len(years)),np.ones(len(years))*np.nanmean(avg_freezeup_doy),'-',color='k')

Twater_varnames = ['Avg. water temp.']
Twater_vars = np.zeros((len(time),len(Twater_varnames)))*np.nan
Twater_vars[:,0] = np.nanmean(Twater,axis=1)
Twater_vars = mask_season_vars(Twater_vars, Twater_varnames, years, time, month_start_day)

#%%
# LOAD WEATHER DATA
weather_loc_list = ['D','A','B','E']
weather_loc_list = ['D']

weather_varnames = ['Avg. Ta_max',
                    'Avg. Ta_min',
                    'Avg. Ta_mean',
                    'Tot. TDD',
                    'Tot. FDD',
                    'Tot. CDD',
                    'Tot. precip.',
                    'Avg. SLP',
                    'Avg. wind speed'
                    ]
weather_varnames2 = ['Tot. snowfall',
                     'Avg. cloud cover',
                     'Avg. spec. hum.',
                     'Avg. rel. hum.'
                     ]

weather_vars_all = np.zeros((len(weather_varnames),len(years),4,len(weather_loc_list)+1))*np.nan
weather_vars2_all = np.zeros((len(weather_varnames2),len(years),4,len(weather_loc_list)))*np.nan


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
    weather_vars[:,4] = FDD
    weather_vars[:,5] = CDD
    weather_vars[:,6] = precip
    weather_vars[:,7] = slp
    weather_vars[:,8] = windspeed
    weather_vars = mask_season_vars(weather_vars, weather_varnames, years, time, month_start_day)
    weather_vars_all[:,:,:,iloc+1] = weather_vars

    weather_vars2 = np.zeros((len(time),len(weather_varnames2)))*np.nan
    weather_vars2[:,0] = snow
    weather_vars2[:,1] = clouds
    weather_vars2[:,2] = avg_SH
    weather_vars2[:,3] = avg_RH
    weather_vars2 = mask_season_vars(weather_vars2, weather_varnames2, years, time, month_start_day)
    weather_vars2_all[:,:,:,iloc] = weather_vars2

# ADD YUL STATION TO LOCATIONS FOR WEATHER_VARS1
weather_loc = 'MontrealDorvalMontrealPETMontrealMcTavishmerged'
weather_data_YUL = np.load(fp+'weather_NCEI/weather_NCEI_'+weather_loc+'.npz',allow_pickle='TRUE')
weather_YUL = weather_data_YUL['weather_data']

# weather_vars = ['TIME','MAX','MIN','TEMP','DEWP','PRCP','SLP','WDSP']
    # MAX - Maximum temperature reported during the day in Fahrenheit to tenths. Missing = 9999.9
    # MIN - Minimum temperature reported during the day in Fahrenheit to tenths. Missing = 9999.9
    # TEMP - Mean temperature for the day in degrees Fahrenheit to tenths. Missing = 9999.9
    # DEWP - Mean dew point for the day in degrees Fahrenheit to tenths. Missing = 9999.9
    # PRCP - Total precipitation (rain and/or melted snow) reported during the day in inches
    #       and hundredths; will usually not end with the midnight observation (i.e. may include
    #       latter part of previous day). “0” indicates no measurable precipitation (includes a trace).Missing = 99.99
    # SLP - Mean sea level pressure for the day in millibars to tenths. Missing = 9999.9
    # WDSP - Mean wind speed for the day in knots to tenths.  Missing = 999.9
max_Ta_YUL = weather_YUL[:,1]
min_Ta_YUL = weather_YUL[:,2]
avg_Ta_YUL = weather_YUL[:,3]
precip_YUL = weather_YUL[:,5]
slp_YUL = weather_YUL[:,6]
windspeed_YUL = weather_YUL[:,7]

# Convert Farenheits to Celsius:
max_Ta_YUL  = (max_Ta_YUL- 32) * (5/9.)
min_Ta_YUL  = (min_Ta_YUL- 32) * (5/9.)
avg_Ta_YUL  = (avg_Ta_YUL- 32) * (5/9.)


mask_FDD_YUL = (avg_Ta_YUL <= 0)
FDD_YUL = avg_Ta_YUL.copy()
FDD_YUL[~mask_FDD_YUL] = np.nan

mask_TDD_YUL = (avg_Ta_YUL > 0)
TDD_YUL = avg_Ta_YUL.copy()
TDD_YUL[~mask_TDD_YUL] = np.nan

CDD_YUL = avg_Ta_YUL.copy()

weather_vars_YUL = np.zeros((len(time),len(weather_varnames)))*np.nan
weather_vars_YUL[:,0] = max_Ta_YUL
weather_vars_YUL[:,1] = min_Ta_YUL
weather_vars_YUL[:,2] = avg_Ta_YUL
weather_vars_YUL[:,3] = TDD_YUL
weather_vars_YUL[:,4] = FDD_YUL
weather_vars_YUL[:,5] = CDD_YUL
weather_vars_YUL[:,6] = precip_YUL
weather_vars_YUL[:,7] = slp_YUL
weather_vars_YUL[:,8] = windspeed_YUL
weather_vars_YUL = mask_season_vars(weather_vars_YUL, weather_varnames, years, time, month_start_day)
weather_vars_all[:,:,:,0] = weather_vars_YUL

#%%
# PLOT ALL VARIABLES FOR ONE LOCATION ONLY

# # # Total period
# ystart = 1991
# yend = 2020
# # First half
# # ystart = 1991
# # yend = 2004
# # Second half
# # ystart = 2005
# # yend = 2020

# yvar = avg_freezeup_doy
# for iloc,weather_loc in enumerate(weather_loc_list):
#     multivars_scatter_plot(iloc,weather_vars_all,yvar,weather_varnames,ystart,yend)
#     multivars_scatter_plot(iloc,weather_vars2_all,yvar,weather_varnames2,ystart,yend)

# yvar = np.squeeze(Twater_vars)
# for iloc,weather_loc in enumerate(weather_loc_list):
#     multivars_scatter_plot(iloc,weather_vars_all,yvar,weather_varnames,ystart,yend)
#     multivars_scatter_plot(iloc,weather_vars2_all,yvar,weather_varnames2,ystart,yend)


#%%
# PLOT ALL LOCATIONS FOR ONE VARIABLE ONLY

# # Total period
ystart = 1991
yend = 2019
# First half
# ystart = 1991
# yend = 2005
# Second half
# ystart = 2006
# yend = 2020

loc_names=['NCEI\nYUL','ERA5\nMLO+OR','ERA5\nMontreal','ERA5\nLake Ontario','ERA5\nOttawa River']
loc_names2=['ERA5\nMLO+OR','ERA5\nMontreal','ERA5\nLake Ontario','ERA5\nOttawa River']


yvar = avg_freezeup_doy
multilocs_scatter_plot(weather_vars_all,yvar,weather_varnames,loc_names,ystart,yend)
multilocs_scatter_plot(weather_vars2_all,yvar,weather_varnames2,loc_names2,ystart,yend)

# yvar = np.squeeze(Twater_vars)
# multilocs_scatter_plot(weather_vars_all,yvar,weather_varnames,loc_names,ystart,yend)
# multilocs_scatter_plot(weather_vars2_all,yvar,weather_varnames2,loc_names2,ystart,yend)


