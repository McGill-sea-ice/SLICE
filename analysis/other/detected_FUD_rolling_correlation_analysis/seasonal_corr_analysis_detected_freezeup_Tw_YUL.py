#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:45:58 2021

@author: Amelie
"""
import numpy as np
import scipy
from scipy import ndimage

import datetime as dt
import calendar

import matplotlib.pyplot as plt

from functions import running_nanmean,find_freezeup_Tw,season_mask
from functions import linear_fit

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
avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
# avg_freezeup_doy = np.nanmin(freezeup_doy,axis=1)
# avg_freezeup_doy = np.nanmax(freezeup_doy,axis=1)

mask_years_p18 = avg_freezeup_doy < np.nanpercentile(avg_freezeup_doy,18)
mask_years_p82 = avg_freezeup_doy > np.nanpercentile(avg_freezeup_doy,82)

# plt.figure()
# plt.plot(avg_freezeup_doy,'o')
# plt.fill_between(np.arange(len(years)),np.ones(len(years))*(np.nanpercentile(avg_freezeup_doy,18)),np.ones(len(years))*(np.nanpercentile(avg_freezeup_doy,82)),facecolor=[0.9, 0.9, 0.9], interpolate=True, alpha=0.65)
# plt.plot(np.arange(len(years)),np.ones(len(years))*np.nanmean(avg_freezeup_doy),'-',color='k')

#%%
# LOAD WEATHER DATA
weather_loc = 'MontrealDorvalMontrealPETMontrealMcTavishmerged'
weather_data = np.load(fp+'weather_NCEI/weather_NCEI_'+weather_loc+'.npz',allow_pickle='TRUE')
weather = weather_data['weather_data']

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
max_Ta = weather[:,1]
min_Ta = weather[:,2]
avg_Ta = weather[:,3]
precip = weather[:,5]
slp = weather[:,6]
windspeed = weather[:,7]

# Convert Farenheits to Celsius:
max_Ta  = (max_Ta- 32) * (5/9.)
min_Ta  = (min_Ta- 32) * (5/9.)
avg_Ta  = (avg_Ta- 32) * (5/9.)


#%%
# LOAD LEVEL AND DISCHARGE DATA
level_SL_loc = 'PointeClaire'
level_OR_loc = 'SteAnnedeBellevue'
discharge_loc = 'Lasalle'

level_SL_data = np.load(fp+'water_levels_discharge_ECCC/water_levels_discharge_'+level_SL_loc+'.npz',allow_pickle='TRUE')
level_SL = level_SL_data['level'][:,1]

level_OR_data = np.load(fp+'water_levels_discharge_ECCC/water_levels_discharge_'+level_OR_loc+'.npz',allow_pickle='TRUE')
level_OR = level_OR_data['level'][:,1]

discharge_data = np.load(fp+'water_levels_discharge_ECCC/water_levels_discharge_'+discharge_loc+'.npz',allow_pickle='TRUE')
discharge = discharge_data['discharge'][:,1]

#%%
# DEFINE EXPLANATORY VARIABLES

explo_vars = np.zeros((13,len(years),4))

# explo_vars[0]: mean max_Ta
# explo_vars[1]: mean min_Ta
# explo_vars[2]: mean avg_Ta
# explo_vars[3]: total tdd (thawing degree days)
# explo_vars[4]: total fdd (freezing degree days)
# explo_vars[5]: mean precip
# explo_vars[6]: total precip
# explo_vars[7]: mean SLP
# explo_vars[8]: mean wind speed
# explo_vars[9]: mean St-Lawrence level
# explo_vars[10]: mean Ottawa River level
# explo_vars[11]: mean discharge
# explo_vars[12]: mean Twater


for iyr, year in enumerate(years):

    i0 = (dt.date(int(year),3,month_start_day)-date_ref).days
    i0 = np.where(time == i0)[0][0]

    i1 = (dt.date(int(year)+1,3,month_start_day)-date_ref).days
    try:
        i1 = np.where(time == i1)[0][0]
    except:
        i1 = len(time)-1


    time_year = time[i0:i1]
    Twater_year = np.nanmean(Twater[i0:i1],axis=1)
    max_Ta_year = max_Ta[i0:i1]
    min_Ta_year = min_Ta[i0:i1]
    avg_Ta_year = avg_Ta[i0:i1]
    precip_year = precip[i0:i1]
    slp_year = slp[i0:i1]
    windspeed_year = windspeed[i0:i1]
    level_SL_year = level_SL[i0:i1]
    level_OR_year = level_OR[i0:i1]
    discharge_year = discharge[i0:i1]

    mask_spring = season_mask(time_year,'spring',msd=month_start_day)
    mask_summer = season_mask(time_year,'summer',msd=month_start_day)
    mask_fall = season_mask(time_year,'fall',msd=month_start_day)
    mask_winter = season_mask(time_year,'winter',msd=month_start_day)

    # max_Ta
    explo_vars[0,iyr,0] = np.nanmean(max_Ta_year[mask_spring])
    explo_vars[0,iyr,1] = np.nanmean(max_Ta_year[mask_summer])
    explo_vars[0,iyr,2] = np.nanmean(max_Ta_year[mask_fall])
    explo_vars[0,iyr,3] = np.nanmean(max_Ta_year[mask_winter])
    # min_Ta
    explo_vars[1,iyr,0] = np.nanmean(min_Ta_year[mask_spring])
    explo_vars[1,iyr,1] = np.nanmean(min_Ta_year[mask_summer])
    explo_vars[1,iyr,2] = np.nanmean(min_Ta_year[mask_fall])
    explo_vars[1,iyr,3] = np.nanmean(min_Ta_year[mask_winter])
    # avg_Ta
    explo_vars[2,iyr,0] = np.nanmean(avg_Ta_year[mask_spring])
    explo_vars[2,iyr,1] = np.nanmean(avg_Ta_year[mask_summer])
    explo_vars[2,iyr,2] = np.nanmean(avg_Ta_year[mask_fall])
    explo_vars[2,iyr,3] = np.nanmean(avg_Ta_year[mask_winter])
    # TDD
    explo_vars[3,iyr,0] = np.nansum(max_Ta_year[mask_spring][max_Ta_year[mask_spring]>0])
    explo_vars[3,iyr,1] = np.nansum(max_Ta_year[mask_summer][max_Ta_year[mask_summer]>0])
    explo_vars[3,iyr,2] = np.nansum(max_Ta_year[mask_fall][max_Ta_year[mask_fall]>0])
    explo_vars[3,iyr,3] = np.nansum(max_Ta_year[mask_winter][max_Ta_year[mask_winter]>0])
    # FDD
    explo_vars[4,iyr,0] = np.nansum(min_Ta_year[mask_spring][min_Ta_year[mask_spring]<=0])
    explo_vars[4,iyr,1] = np.nansum(min_Ta_year[mask_summer][min_Ta_year[mask_summer]<=0])
    explo_vars[4,iyr,2] = np.nansum(min_Ta_year[mask_fall][min_Ta_year[mask_fall]<=0])
    explo_vars[4,iyr,3] = np.nansum(min_Ta_year[mask_winter][min_Ta_year[mask_winter]<=0])
    # Mean precip
    explo_vars[5,iyr,0] = np.nanmean(precip_year[mask_spring])
    explo_vars[5,iyr,1] = np.nanmean(precip_year[mask_summer])
    explo_vars[5,iyr,2] = np.nanmean(precip_year[mask_fall])
    explo_vars[5,iyr,3] = np.nanmean(precip_year[mask_winter])
    # Total precip
    explo_vars[6,iyr,0] = np.nansum(precip_year[mask_spring])
    explo_vars[6,iyr,1] = np.nansum(precip_year[mask_summer])
    explo_vars[6,iyr,2] = np.nansum(precip_year[mask_fall])
    explo_vars[6,iyr,3] = np.nansum(precip_year[mask_winter])
    # Mean SLP
    explo_vars[7,iyr,0] = np.nanmean(slp_year[mask_spring])
    explo_vars[7,iyr,1] = np.nanmean(slp_year[mask_summer])
    explo_vars[7,iyr,2] = np.nanmean(slp_year[mask_fall])
    explo_vars[7,iyr,3] = np.nanmean(slp_year[mask_winter])
    # Mean wind speed
    explo_vars[8,iyr,0] = np.nanmean(windspeed_year[mask_spring])
    explo_vars[8,iyr,1] = np.nanmean(windspeed_year[mask_summer])
    explo_vars[8,iyr,2] = np.nanmean(windspeed_year[mask_fall])
    explo_vars[8,iyr,3] = np.nanmean(windspeed_year[mask_winter])
    # Mean level St-Lawrence
    explo_vars[9,iyr,0] = np.nanmean(level_SL_year[mask_spring])
    explo_vars[9,iyr,1] = np.nanmean(level_SL_year[mask_summer])
    explo_vars[9,iyr,2] = np.nanmean(level_SL_year[mask_fall])
    explo_vars[9,iyr,3] = np.nanmean(level_SL_year[mask_winter])
    # Mean level Ottawa River
    explo_vars[10,iyr,0] = np.nanmean(level_OR_year[mask_spring])
    explo_vars[10,iyr,1] = np.nanmean(level_OR_year[mask_summer])
    explo_vars[10,iyr,2] = np.nanmean(level_OR_year[mask_fall])
    explo_vars[10,iyr,3] = np.nanmean(level_OR_year[mask_winter])
    # Mean discharge
    explo_vars[11,iyr,0] = np.nanmean(discharge_year[mask_spring])
    explo_vars[11,iyr,1] = np.nanmean(discharge_year[mask_summer])
    explo_vars[11,iyr,2] = np.nanmean(discharge_year[mask_fall])
    explo_vars[11,iyr,3] = np.nanmean(discharge_year[mask_winter])
    # Mean water_temperature
    explo_vars[12,iyr,0] = np.nanmean(Twater_year[mask_spring])
    explo_vars[12,iyr,1] = np.nanmean(Twater_year[mask_summer])
    explo_vars[12,iyr,2] = np.nanmean(Twater_year[mask_fall])
    explo_vars[12,iyr,3] = np.nanmean(Twater_year[mask_winter])


#%%
# Box plots and 5 earliest vs 5 latest freezeups
explo_var_list = ['max. Ta',
                  'min. Ta',
                  'avg. Ta',
                  'Tot. TDD',
                  'Tot. FDD',
                  'Mean precip.',
                  'Tot. precip.',
                  'Avg. SLP',
                  'Avg. wind speed',
                  'Avg. St-L. level',
                  'Avg. Ott. Riv. level',
                  'Avg. discharge',
                  'Avg. water temp.'
                  ]
plot_composite = False
if plot_composite:
    for ivar,var in enumerate(explo_var_list):
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,3.5))
        ax.boxplot(explo_vars[ivar,:-1,:],positions=[0,1,2,3],whis=[5, 95],widths=(0.3),showfliers=True)
        ax.plot([np.ones(np.sum(mask_years_p18))*0.25, np.ones(np.sum(mask_years_p18))*1.25, np.ones(np.sum(mask_years_p18))*2.25, np.ones(np.sum(mask_years_p18))*3.25],explo_vars[ivar][mask_years_p18].T,'o',markersize=5,alpha=0.6,color=plt.get_cmap('tab20')(0*2), label = 'Early freezeup')
        ax.plot([np.ones(np.sum(mask_years_p82))*0.5, np.ones(np.sum(mask_years_p82))*1.5, np.ones(np.sum(mask_years_p82))*2.5, np.ones(np.sum(mask_years_p82))*3.5],explo_vars[ivar][mask_years_p82].T,'o',markersize=5,alpha=0.6,color=plt.get_cmap('tab20')(1*2), label = 'Delayed freezeup')
        ax.boxplot(np.nanmean(explo_vars[ivar,:-1,:],axis=1),positions=[4],whis=[5, 95],widths=(0.3),showfliers=True)
        mean_allseasons = np.nanmean(explo_vars[ivar,:-1,:],axis=1)
        ax.plot(np.ones(np.sum(mask_years_p18[:-1]))*4.25,mean_allseasons[mask_years_p18[:-1]].T,'o',markersize=5,alpha=0.6,color=plt.get_cmap('tab20')(0*2), label = 'Early freezeup')
        ax.plot(np.ones(np.sum(mask_years_p82[:-1]))*4.5,mean_allseasons[mask_years_p82[:-1]].T,'o',markersize=5,alpha=0.6,color=plt.get_cmap('tab20')(1*2), label = 'Delayed freezeup')

        ax.set_ylabel(var)
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels(['spring','summer','fall','winter','All year'], rotation=45,fontsize=10)
        ax.set_xlim(-0.5,5)
        plt.subplots_adjust(bottom=0.24)

#%%
# Scatter plots for freezeup date vs explanatory variables

# yr_start = int(np.where(years == 1991)[0][0])
# yr_end = int(np.where(years == 2000)[0][0]+1)

# yr_start = int(np.where(years == 2001)[0][0])
# yr_end = int(np.where(years == 2010)[0][0]+1)

# yr_start = int(np.where(years == 2011)[0][0])
# yr_end = int(np.where(years == 2020)[0][0])



# First half
# yr_start = int(np.where(years == 1991)[0][0])
# yr_end = int(np.where(years == 2004)[0][0]+1)
# Second half
# yr_start = int(np.where(years == 2005)[0][0])
# yr_end = int(np.where(years == 2020)[0][0])


# # Total period
yr_start = int(np.where(years == 1991)[0][0])
yr_end = int(np.where(years == 2020)[0][0])


season_colors = [plt.get_cmap('tab20')(4),plt.get_cmap('tab20')(6),plt.get_cmap('tab20')(2),plt.get_cmap('tab20')(0)]

istart = 0
iend = 5
fig,ax = plt.subplots(nrows=4,ncols=iend-istart,figsize=((iend-istart)*(8/5.),8),sharey=True)
for ivar,var1 in enumerate(explo_var_list[istart:iend]):
    for iseason in range(4):
        ax[iseason,ivar].plot(explo_vars[ivar+istart,yr_start:yr_end,iseason],avg_freezeup_doy[yr_start:yr_end],'o',color=season_colors[iseason],alpha=0.5)
        xvar_min = np.nanmin(explo_vars[ivar+istart,yr_start:yr_end,iseason])
        xvar_max = np.nanmax(explo_vars[ivar+istart,yr_start:yr_end,iseason])
        xvar_range = xvar_max-xvar_min
        ax[iseason,ivar].set_xlim(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)
        yvar_min = np.nanmin(avg_freezeup_doy[yr_start:yr_end])
        yvar_max = np.nanmax(avg_freezeup_doy[yr_start:yr_end])
        yvar_range = yvar_max-yvar_min
        ax[iseason,ivar].set_ylim(yvar_min-yvar_range/5.,yvar_max+2*yvar_range/5.)

        x_fit = explo_vars[ivar+istart,yr_start:yr_end,iseason]
        y_fit = avg_freezeup_doy[yr_start:yr_end]
        mask_x = ~np.isnan(x_fit)
        mask_y = ~np.isnan(y_fit)
        x_fit = x_fit[mask_x&mask_y]
        y_fit = y_fit[mask_x&mask_y]
        lincoeff, Rsqr = linear_fit(x_fit,y_fit)
        ax[iseason,ivar].text(xvar_min-xvar_range/10.,yvar_max+yvar_range/6.,'$R^{2}$: '+'%.2f'%(Rsqr),fontsize=8,color=[0.5,0.5,0.5])

        xplot = np.arange(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)
        yplot = lincoeff[0]*xplot + lincoeff[1]
        ax[iseason,ivar].plot(xplot,yplot,'-',color=[0.5,0.5,0.5])

        if ivar ==0:
            ax[iseason,ivar].set_ylabel(season_list[iseason],fontsize=10)
        if iseason == 0:
            ax[iseason,ivar].xaxis.set_label_position('top')
            ax[iseason,ivar].set_xlabel(explo_var_list[ivar+istart],fontsize=10)


istart = 5
iend = 9
fig,ax = plt.subplots(nrows=4,ncols=iend-istart,figsize=((iend-istart)*(8/5.),8),sharey=True)
for ivar,var1 in enumerate(explo_var_list[istart:iend]):
    for iseason in range(4):
        ax[iseason,ivar].plot(explo_vars[ivar+istart,yr_start:yr_end,iseason],avg_freezeup_doy[yr_start:yr_end],'o',color=season_colors[iseason],alpha=0.5)
        xvar_min = np.nanmin(explo_vars[ivar+istart,yr_start:yr_end,iseason])
        xvar_max = np.nanmax(explo_vars[ivar+istart,yr_start:yr_end,iseason])
        xvar_range = xvar_max-xvar_min
        ax[iseason,ivar].set_xlim(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)
        yvar_min = np.nanmin(avg_freezeup_doy[yr_start:yr_end])
        yvar_max = np.nanmax(avg_freezeup_doy[yr_start:yr_end])
        yvar_range = yvar_max-yvar_min
        ax[iseason,ivar].set_ylim(yvar_min-yvar_range/5.,yvar_max+2*yvar_range/5.)

        x_fit = explo_vars[ivar+istart,yr_start:yr_end,iseason]
        y_fit = avg_freezeup_doy[yr_start:yr_end]
        mask_x = ~np.isnan(x_fit)
        mask_y = ~np.isnan(y_fit)
        x_fit = x_fit[mask_x&mask_y]
        y_fit = y_fit[mask_x&mask_y]
        lincoeff, Rsqr = linear_fit(x_fit,y_fit)
        ax[iseason,ivar].text(xvar_min-xvar_range/10.,yvar_max+yvar_range/6.,'$R^{2}$: '+'%.2f'%(Rsqr),fontsize=8,color=[0.5,0.5,0.5])

        xplot = np.arange(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)
        yplot = lincoeff[0]*xplot + lincoeff[1]
        ax[iseason,ivar].plot(xplot,yplot,'-',color=[0.5,0.5,0.5])

        if ivar ==0:
            ax[iseason,ivar].set_ylabel(season_list[iseason],fontsize=10)
        if iseason == 0:
            ax[iseason,ivar].xaxis.set_label_position('top')
            ax[iseason,ivar].set_xlabel(explo_var_list[ivar+istart],fontsize=10)


istart = 9
iend = 12
fig,ax = plt.subplots(nrows=4,ncols=iend-istart,figsize=((iend-istart)*(8/5.),8),sharey=True)
for ivar,var1 in enumerate(explo_var_list[istart:iend]):
    for iseason in range(4):
        ax[iseason,ivar].plot(explo_vars[ivar+istart,yr_start:yr_end,iseason],avg_freezeup_doy[yr_start:yr_end],'o',color=season_colors[iseason],alpha=0.5)
        xvar_min = np.nanmin(explo_vars[ivar+istart,yr_start:yr_end,iseason])
        xvar_max = np.nanmax(explo_vars[ivar+istart,yr_start:yr_end,iseason])
        xvar_range = xvar_max-xvar_min
        ax[iseason,ivar].set_xlim(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)
        yvar_min = np.nanmin(avg_freezeup_doy[yr_start:yr_end])
        yvar_max = np.nanmax(avg_freezeup_doy[yr_start:yr_end])
        yvar_range = yvar_max-yvar_min
        ax[iseason,ivar].set_ylim(yvar_min-yvar_range/5.,yvar_max+2*yvar_range/5.)

        x_fit = explo_vars[ivar+istart,yr_start:yr_end,iseason]
        y_fit = avg_freezeup_doy[yr_start:yr_end]
        mask_x = ~np.isnan(x_fit)
        mask_y = ~np.isnan(y_fit)
        x_fit = x_fit[mask_x&mask_y]
        y_fit = y_fit[mask_x&mask_y]
        lincoeff, Rsqr = linear_fit(x_fit,y_fit)
        ax[iseason,ivar].text(xvar_min-xvar_range/10.,yvar_max+yvar_range/6.,'$R^{2}$: '+'%.2f'%(Rsqr),fontsize=8,color=[0.5,0.5,0.5])

        xplot = np.arange(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)
        yplot = lincoeff[0]*xplot + lincoeff[1]
        ax[iseason,ivar].plot(xplot,yplot,'-',color=[0.5,0.5,0.5])

        if ivar ==0:
            ax[iseason,ivar].set_ylabel(season_list[iseason],fontsize=10)
        if iseason == 0:
            ax[iseason,ivar].xaxis.set_label_position('top')
            ax[iseason,ivar].set_xlabel(explo_var_list[ivar+istart],fontsize=10)


istart = 11
iend = 13
fig,ax = plt.subplots(nrows=4,ncols=iend-istart,figsize=((iend-istart)*(8/5.),8),sharey=True)
for ivar,var1 in enumerate(explo_var_list[istart:iend]):
    for iseason in range(4):
        ax[iseason,ivar].plot(explo_vars[ivar+istart,yr_start:yr_end,iseason],avg_freezeup_doy[yr_start:yr_end],'o',color=season_colors[iseason],alpha=0.5)
        xvar_min = np.nanmin(explo_vars[ivar+istart,yr_start:yr_end,iseason])
        xvar_max = np.nanmax(explo_vars[ivar+istart,yr_start:yr_end,iseason])
        xvar_range = xvar_max-xvar_min
        ax[iseason,ivar].set_xlim(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)
        yvar_min = np.nanmin(avg_freezeup_doy[yr_start:yr_end])
        yvar_max = np.nanmax(avg_freezeup_doy[yr_start:yr_end])
        yvar_range = yvar_max-yvar_min
        ax[iseason,ivar].set_ylim(yvar_min-yvar_range/5.,yvar_max+2*yvar_range/5.)

        x_fit = explo_vars[ivar+istart,yr_start:yr_end,iseason]
        y_fit = avg_freezeup_doy[yr_start:yr_end]
        mask_x = ~np.isnan(x_fit)
        mask_y = ~np.isnan(y_fit)
        x_fit = x_fit[mask_x&mask_y]
        y_fit = y_fit[mask_x&mask_y]
        lincoeff, Rsqr = linear_fit(x_fit,y_fit)
        ax[iseason,ivar].text(xvar_min-xvar_range/10.,yvar_max+yvar_range/6.,'$R^{2}$: '+'%.2f'%(Rsqr),fontsize=8,color=[0.5,0.5,0.5])

        xplot = np.arange(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)
        yplot = lincoeff[0]*xplot + lincoeff[1]
        ax[iseason,ivar].plot(xplot,yplot,'-',color=[0.5,0.5,0.5])

        if ivar ==0:
            ax[iseason,ivar].set_ylabel(season_list[iseason],fontsize=10)
        if iseason == 0:
            ax[iseason,ivar].xaxis.set_label_position('top')
            ax[iseason,ivar].set_xlabel(explo_var_list[ivar+istart],fontsize=10)

#%%
# Scatter plots for Twater vs other explanatory variables

# First half
# yr_start = int(np.where(years == 1991)[0][0])
# yr_end = int(np.where(years == 2004)[0][0]+1)
# Second half
# yr_start = int(np.where(years == 2005)[0][0])
# yr_end = int(np.where(years == 2020)[0][0])


# # Total period
yr_start = int(np.where(years == 1991)[0][0])
yr_end = int(np.where(years == 2020)[0][0])

season_colors = [plt.get_cmap('tab20')(4),plt.get_cmap('tab20')(6),plt.get_cmap('tab20')(2),plt.get_cmap('tab20')(0)]

istart = 0
iend = 5
fig,ax = plt.subplots(nrows=4,ncols=iend-istart,figsize=((iend-istart)*(8/5.),8),sharey='row')
for ivar,var1 in enumerate(explo_var_list[istart:iend]):
    for iseason in range(4):
        ax[iseason,ivar].plot(explo_vars[ivar+istart,yr_start:yr_end,iseason],explo_vars[12,yr_start:yr_end,iseason],'o',color=season_colors[iseason],alpha=0.5)
        yvar_min = np.nanmin(explo_vars[12,yr_start:yr_end,iseason])
        yvar_max = np.nanmax(explo_vars[12,yr_start:yr_end,iseason])
        yvar_range = yvar_max-yvar_min
        ax[iseason,ivar].set_ylim(yvar_min-yvar_range/5.,yvar_max+2*yvar_range/5.)
        xvar_min = np.nanmin(explo_vars[ivar+istart,yr_start:yr_end,iseason])
        xvar_max = np.nanmax(explo_vars[ivar+istart,yr_start:yr_end,iseason])
        xvar_range = xvar_max-xvar_min
        ax[iseason,ivar].set_xlim(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)

        x_fit = explo_vars[ivar+istart,yr_start:yr_end,iseason]
        y_fit = explo_vars[12,yr_start:yr_end,iseason]
        mask_x = ~np.isnan(x_fit)
        mask_y = ~np.isnan(y_fit)
        x_fit = x_fit[mask_x&mask_y]
        y_fit = y_fit[mask_x&mask_y]
        lincoeff, Rsqr = linear_fit(x_fit,y_fit)
        ax[iseason,ivar].text(xvar_min-xvar_range/10.,yvar_max+yvar_range/6.,'$R^{2}$: '+'%.2f'%(Rsqr),fontsize=8,color=[0.5,0.5,0.5])

        xplot = np.arange(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)
        yplot = lincoeff[0]*xplot + lincoeff[1]
        ax[iseason,ivar].plot(xplot,yplot,'-',color=[0.5,0.5,0.5])

        if ivar ==0:
            ax[iseason,ivar].set_ylabel(season_list[iseason],fontsize=10)
        if iseason == 0:
            ax[iseason,ivar].xaxis.set_label_position('top')
            ax[iseason,ivar].set_xlabel(explo_var_list[ivar+istart],fontsize=10)


istart = 5
iend = 9
fig,ax = plt.subplots(nrows=4,ncols=iend-istart,figsize=((iend-istart)*(8/5.),8),sharey='row')
for ivar,var1 in enumerate(explo_var_list[istart:iend]):
    for iseason in range(4):
        ax[iseason,ivar].plot(explo_vars[ivar+istart,yr_start:yr_end,iseason],explo_vars[12,yr_start:yr_end,iseason],'o',color=season_colors[iseason],alpha=0.5)
        yvar_min = np.nanmin(explo_vars[12,yr_start:yr_end,iseason])
        yvar_max = np.nanmax(explo_vars[12,yr_start:yr_end,iseason])
        yvar_range = yvar_max-yvar_min
        ax[iseason,ivar].set_ylim(yvar_min-yvar_range/5.,yvar_max+2*yvar_range/5.)
        xvar_min = np.nanmin(explo_vars[ivar+istart,yr_start:yr_end,iseason])
        xvar_max = np.nanmax(explo_vars[ivar+istart,yr_start:yr_end,iseason])
        xvar_range = xvar_max-xvar_min
        ax[iseason,ivar].set_xlim(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)

        x_fit = explo_vars[ivar+istart,yr_start:yr_end,iseason]
        y_fit = explo_vars[12,yr_start:yr_end,iseason]
        mask_x = ~np.isnan(x_fit)
        mask_y = ~np.isnan(y_fit)
        x_fit = x_fit[mask_x&mask_y]
        y_fit = y_fit[mask_x&mask_y]
        lincoeff, Rsqr = linear_fit(x_fit,y_fit)
        ax[iseason,ivar].text(xvar_min-xvar_range/10.,yvar_max+yvar_range/6.,'$R^{2}$: '+'%.2f'%(Rsqr),fontsize=8,color=[0.5,0.5,0.5])

        xplot = np.arange(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)
        yplot = lincoeff[0]*xplot + lincoeff[1]
        ax[iseason,ivar].plot(xplot,yplot,'-',color=[0.5,0.5,0.5])

        if ivar ==0:
            ax[iseason,ivar].set_ylabel(season_list[iseason],fontsize=10)
        if iseason == 0:
            ax[iseason,ivar].xaxis.set_label_position('top')
            ax[iseason,ivar].set_xlabel(explo_var_list[ivar+istart],fontsize=10)

istart = 9
iend = 12
fig,ax = plt.subplots(nrows=4,ncols=iend-istart,figsize=((iend-istart)*(8/5.),8),sharey='row')
for ivar,var1 in enumerate(explo_var_list[istart:iend]):
    for iseason in range(4):
        ax[iseason,ivar].plot(explo_vars[ivar+istart,yr_start:yr_end,iseason],explo_vars[12,yr_start:yr_end,iseason],'o',color=season_colors[iseason],alpha=0.5)
        yvar_min = np.nanmin(explo_vars[12,yr_start:yr_end,iseason])
        yvar_max = np.nanmax(explo_vars[12,yr_start:yr_end,iseason])
        yvar_range = yvar_max-yvar_min
        ax[iseason,ivar].set_ylim(yvar_min-yvar_range/5.,yvar_max+2*yvar_range/5.)
        xvar_min = np.nanmin(explo_vars[ivar+istart,yr_start:yr_end,iseason])
        xvar_max = np.nanmax(explo_vars[ivar+istart,yr_start:yr_end,iseason])
        xvar_range = xvar_max-xvar_min
        ax[iseason,ivar].set_xlim(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)

        x_fit = explo_vars[ivar+istart,yr_start:yr_end,iseason]
        y_fit = explo_vars[12,yr_start:yr_end,iseason]
        mask_x = ~np.isnan(x_fit)
        mask_y = ~np.isnan(y_fit)
        x_fit = x_fit[mask_x&mask_y]
        y_fit = y_fit[mask_x&mask_y]
        lincoeff, Rsqr = linear_fit(x_fit,y_fit)
        ax[iseason,ivar].text(xvar_min-xvar_range/10.,yvar_max+yvar_range/6.,'$R^{2}$: '+'%.2f'%(Rsqr),fontsize=8,color=[0.5,0.5,0.5])

        xplot = np.arange(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)
        yplot = lincoeff[0]*xplot + lincoeff[1]
        ax[iseason,ivar].plot(xplot,yplot,'-',color=[0.5,0.5,0.5])

        if ivar ==0:
            ax[iseason,ivar].set_ylabel(season_list[iseason],fontsize=10)
        if iseason == 0:
            ax[iseason,ivar].xaxis.set_label_position('top')
            ax[iseason,ivar].set_xlabel(explo_var_list[ivar+istart],fontsize=10)


istart = 11
iend = 13
fig,ax = plt.subplots(nrows=4,ncols=iend-istart,figsize=((iend-istart)*(8/5.),8),sharey='row')
for ivar,var1 in enumerate(explo_var_list[istart:iend]):
    for iseason in range(4):
        ax[iseason,ivar].plot(explo_vars[ivar+istart,yr_start:yr_end,iseason],explo_vars[12,yr_start:yr_end,iseason],'o',color=season_colors[iseason],alpha=0.5)
        yvar_min = np.nanmin(explo_vars[12,yr_start:yr_end,iseason])
        yvar_max = np.nanmax(explo_vars[12,yr_start:yr_end,iseason])
        yvar_range = yvar_max-yvar_min
        ax[iseason,ivar].set_ylim(yvar_min-yvar_range/5.,yvar_max+2*yvar_range/5.)
        xvar_min = np.nanmin(explo_vars[ivar+istart,yr_start:yr_end,iseason])
        xvar_max = np.nanmax(explo_vars[ivar+istart,yr_start:yr_end,iseason])
        xvar_range = xvar_max-xvar_min
        ax[iseason,ivar].set_xlim(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)

        x_fit = explo_vars[ivar+istart,yr_start:yr_end,iseason]
        y_fit = explo_vars[12,yr_start:yr_end,iseason]
        mask_x = ~np.isnan(x_fit)
        mask_y = ~np.isnan(y_fit)
        x_fit = x_fit[mask_x&mask_y]
        y_fit = y_fit[mask_x&mask_y]
        lincoeff, Rsqr = linear_fit(x_fit,y_fit)
        ax[iseason,ivar].text(xvar_min-xvar_range/10.,yvar_max+yvar_range/6.,'$R^{2}$: '+'%.2f'%(Rsqr),fontsize=8,color=[0.5,0.5,0.5])

        xplot = np.arange(xvar_min-xvar_range/5.,xvar_max+xvar_range/5.)
        yplot = lincoeff[0]*xplot + lincoeff[1]
        ax[iseason,ivar].plot(xplot,yplot,'-',color=[0.5,0.5,0.5])

        if ivar ==0:
            ax[iseason,ivar].set_ylabel(season_list[iseason],fontsize=10)
        if iseason == 0:
            ax[iseason,ivar].xaxis.set_label_position('top')
            ax[iseason,ivar].set_xlabel(explo_var_list[ivar+istart],fontsize=10)


#%%
# plt.figure()
# plt.plot(avg_spring_lvl_SL,avg_spring_dis, 'o',label='spring')
# plt.plot(avg_summer_lvl_SL,avg_summer_dis, '+',label='summer')
# plt.plot(avg_fall_lvl_SL,avg_fall_dis, '<',label='fall')
# plt.plot(avg_winter_lvl_SL,avg_winter_dis, '*',label='winter')
# plt.legend()

# plt.figure()
# plt.plot(avg_spring_lvl_SL,avg_spring_lvl_OR, 'o',label='spring')
# plt.plot(avg_summer_lvl_SL,avg_summer_lvl_OR, '+',label='summer')
# plt.plot(avg_fall_lvl_SL,avg_fall_lvl_OR, '<',label='fall')
# plt.plot(avg_winter_lvl_SL,avg_winter_lvl_OR, '*',label='winter')
# plt.legend()

# plt.figure()
# plt.plot(avg_spring_precip,avg_spring_dis, 'o',label='spring')

# #%%
# plt.figure()
# plt.plot(avg_spring_min_Ta,avg_spring_wdsp, 'o',label='spring')
# plt.plot(avg_summer_min_Ta,avg_summer_wdsp, 'o',label='summer')

# plt.figure()
# plt.plot(avg_spring_slp,avg_spring_wdsp, 'o',label='spring')
# plt.plot(avg_summer_slp,avg_summer_wdsp, 'o',label='summer')

# #%%
# plt.figure()
# plt.plot(avg_spring_slp,avg_summer_avg_Ta, 'o',label='spring/summer')
# plt.plot(avg_summer_slp,avg_fall_avg_Ta, '+',label='summer/fall')
# plt.plot(avg_fall_slp,avg_winter_avg_Ta, '>',label='fall/winter')

#%%

# plt.figure()
# plt.boxplot([avg_spring_min_Ta,avg_summer_min_Ta,avg_fall_min_Ta,avg_winter_min_Ta],positions=[0,1,2,3],whis=[5, 95],widths=(0.3),showfliers=True)
# # plt.boxplot([avg_spring_min_Ta[mask_years_p18],avg_summer_min_Ta[mask_years_p18],avg_fall_min_Ta[mask_years_p18],avg_winter_min_Ta[mask_years_p18]],positions=[0.25,1.25,2.25,3.25],whis=[5, 95],showmeans=True,showfliers=True)
# # plt.boxplot([avg_spring_min_Ta[mask_years_p82],avg_summer_min_Ta[mask_years_p82],avg_fall_min_Ta[mask_years_p82],avg_winter_min_Ta[mask_years_p82]],positions=[0.5,1.5,2.5,3.5],whis=[5, 95],showmeans=True,showfliers=True)

# plt.plot(np.ones(np.sum(mask_years_p18))*0.25,avg_spring_min_Ta[mask_years_p18],'o',markersize=5,alpha=0.6,color=plt.get_cmap('tab20')(0*2))
# plt.plot(np.ones(np.sum(mask_years_p82))*0.5,avg_spring_min_Ta[mask_years_p82],'o',markersize=5,alpha=0.6,color=plt.get_cmap('tab20')(1*2))

# plt.plot(np.ones(np.sum(mask_years_p18))*1.25,avg_summer_min_Ta[mask_years_p18],'o',markersize=5,alpha=0.6,color=plt.get_cmap('tab20')(0*2))
# plt.plot(np.ones(np.sum(mask_years_p82))*1.5,avg_summer_min_Ta[mask_years_p82],'o',markersize=5,alpha=0.6,color=plt.get_cmap('tab20')(1*2))

# plt.plot(np.ones(np.sum(mask_years_p18))*2.25,avg_fall_min_Ta[mask_years_p18],'o',markersize=5,alpha=0.6,color=plt.get_cmap('tab20')(0*2))
# plt.plot(np.ones(np.sum(mask_years_p82))*2.5,avg_fall_min_Ta[mask_years_p82],'o',markersize=5,alpha=0.6,color=plt.get_cmap('tab20')(1*2))

# plt.plot(np.ones(np.sum(mask_years_p18))*3.25,avg_winter_min_Ta[mask_years_p18],'o',markersize=5,alpha=0.6,color=plt.get_cmap('tab20')(0*2))
# plt.plot(np.ones(np.sum(mask_years_p82))*3.5,avg_winter_min_Ta[mask_years_p82],'o',markersize=5,alpha=0.6,color=plt.get_cmap('tab20')(1*2))



