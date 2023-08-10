#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:33:26 2021

@author: Amelie

"""

import numpy as np
import scipy as sp
import pandas as pd

from scipy.signal import medfilt

import datetime as dt
import calendar

import matplotlib.pyplot as plt

# =========================================================================
# years = [1994,1995,1996,1997,
#           1998,1999,2000,2001,2002,2003,2004] # Atwater
# years = [2011,2012,
#             2013,2014,2015,2016,2017,
#             2018,2019] # Atwater

years = [2006,2007,2008,
          2009,2010,2011,2012,
          2013,2014,2015,2016,2017,
          2018,2019,2020] # Des Baillets

# years = [2004,2005,2006,2007,2008,
#          2009,2010,2011,2012,
#          2013,2014,2015,2016,2017,
#          2018,2019] # Candiac

# years = [1992,1993,1994,1995,1996,
#           1997,1998,1999,2000,2001,
#           2002,2003,2004,2005,2006,
#           2007,2008,2009,2010,2011,
#           2012,2013,2014,2015,2016,2017,
#           2018,2019
#             ] # Longueuil


# years = [2004,2011,2013,2015,2017,2019]

# water_cities_name_list = 'DesBaillets_clean'
# water_cities_name_list = 'DesBaillets'
# water_cities_name_list = 'Atwater'
# water_cities_name_list = 'Candiac'
# water_cities_name_list = ['Longueuil','Atwater','Candiac','DesBaillets']
water_cities_name_list = ['Candiac','Longueuil','DesBaillets','Atwater']
# water_cities_name_list = ['Longueuil','Atwater']
# water_cities_name_list = ['Candiac','Longueuil','Atwater']
# water_cities_name_list = ['Candiac']
# water_cities_name_list = ['Longueuil']

weather_name_list = ['MontrealDorvalMontrealPETMontrealMcTavishmerged']

fp = '../../data/'

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2020,11,1)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

loc_weather = weather_name_list[0]
weather_data = np.load(fp+'weather_NCEI/weather_NCEI_'+loc_weather+'.npz',allow_pickle='TRUE')
weather_data = weather_data['weather_data']
Ta = weather_data[:,3]

#%%
# First put all data in one array
data_Tw = np.zeros((366,len(years),len(water_cities_name_list)))*np.nan
data_Ta = np.zeros((366,len(years)))*np.nan

t = np.zeros((366,len(years)))*np.nan

for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater_city = water_city_data['Twater']

    for iyear,year in enumerate(years):

        date=(dt.date(year,1,1)-date_ref).days
        i0 = np.where(time==date)[0][0]
        i1 = i0+365+calendar.isleap(year)

        data_Tw[0:365+calendar.isleap(year),iyear,icity] = Twater_city[i0:i1,1].copy()
        t[0:len(time[i0:i1]),iyear] = time[i0:i1]
        if not calendar.isleap(year):
            data_Tw[-1,iyear,icity] = np.nanmean([Twater_city[i0:i1,1][-1],Twater_city[i1,1]])

        data_Ta[0:365+calendar.isleap(year),iyear] = Ta[i0:i1].copy()
        if not calendar.isleap(year):
            data_Ta[-1,iyear] = np.nanmean([Ta[i0:i1][-1],Ta[i1]])


#%%
# Compute daily climatology
daily_clim_Tw = np.zeros((366,len(water_cities_name_list)))*np.nan
daily_clim_Ta = np.zeros((366))*np.nan

n_years_min = 10
for icity in range(data_Tw.shape[2]):
    data_city_Tw = data_Tw[:,:,icity].copy()

    # Check if at each day there is at least 10 years available:
    mask_daily_Tw = np.sum(~np.isnan(data_city_Tw),axis=1)
    print('------------')
    print(water_cities_name_list[icity])
    print('Mean nb. years available each day: {:4.2f} '.format(np.mean(mask_daily_Tw)))
    print('Days with less than {} years? --> {}'.format(n_years_min, np.any(mask_daily_Tw<n_years_min)))

    daily_clim_Tw[:,icity] = np.nanmean(data_city_Tw,axis=1)
    daily_clim_Ta[:] = np.nanmean(data_Ta,axis=1)


#%%
# Compute monthly climatology
data_md_Tw = np.zeros((12,31,len(years),len(water_cities_name_list)))*np.nan
data_md_Ta = np.zeros((12,31,len(years)))*np.nan

monthly_clim_Tw = np.zeros((12,len(water_cities_name_list)))*np.nan
monthly_std_Tw = np.zeros((12,len(water_cities_name_list)))*np.nan
monthly_p2_Tw = np.zeros((12,len(water_cities_name_list)))*np.nan
monthly_p5_Tw = np.zeros((12,len(water_cities_name_list)))*np.nan
monthly_p95_Tw = np.zeros((12,len(water_cities_name_list)))*np.nan
monthly_p98_Tw = np.zeros((12,len(water_cities_name_list)))*np.nan

monthly_clim_Ta = np.zeros((12))*np.nan
monthly_std_Ta = np.zeros((12))*np.nan
monthly_p2_Ta = np.zeros((12))*np.nan
monthly_p5_Ta = np.zeros((12))*np.nan
monthly_p95_Ta = np.zeros((12))*np.nan
monthly_p98_Ta = np.zeros((12))*np.nan

for icity in range(data_Tw.shape[2]):
    data_city_Tw = data_Tw[:,:,icity].copy()

    for iyear,year in enumerate(years):
        data_city_yr_Tw = data_city_Tw[:,iyear]
        data_yr_Ta = data_Ta[:,iyear]
        t_yr = t[:,iyear]
        for iday in range(t_yr.shape[0]):
            if ~np.isnan(t_yr[iday]):
                date_yr = date_ref+dt.timedelta(days=t_yr[iday])
                month = date_yr.month
                day = date_yr.day
                data_md_Tw[month-1,day-1,iyear,icity] = data_city_yr_Tw[iday]
                data_md_Ta[month-1,day-1,iyear] = data_yr_Ta[iday]
            else:
                continue

    for imonth in range(12):
        d_Tw = data_md_Tw[imonth,:,:,icity].copy()
        monthly_clim_Tw[imonth,icity] = np.nanmean(d_Tw)
        monthly_std_Tw[imonth,icity] = np.nanstd(d_Tw)
        monthly_p2_Tw[imonth,icity] = np.nanpercentile(d_Tw,2)
        monthly_p5_Tw[imonth,icity] = np.nanpercentile(d_Tw,5)
        monthly_p95_Tw[imonth,icity] = np.nanpercentile(d_Tw,95)
        monthly_p98_Tw[imonth,icity] = np.nanpercentile(d_Tw,98)

        d_Ta = data_md_Ta[imonth,:,:].copy()
        monthly_clim_Ta[imonth] = np.nanmean(d_Ta)
        monthly_std_Ta[imonth] = np.nanstd(d_Ta)
        monthly_p2_Ta[imonth] = np.nanpercentile(d_Ta,2)
        monthly_p5_Ta[imonth] = np.nanpercentile(d_Ta,5)
        monthly_p95_Ta[imonth] = np.nanpercentile(d_Ta,95)
        monthly_p98_Ta[imonth] = np.nanpercentile(d_Ta,98)


#%%
# mid_month = [15.5,45,74.5,105,135.5,166,196.5,227.5,258,288.5,319,349.5]

# for icity in range(len(water_cities_name_list)):
#     plt.figure()
#     # plt.plot(daily_clim[:,icity],color='gray')
#     plt.plot(data[:,:,icity],color='gray')
#     plt.plot(mid_month,monthly_clim[:,icity],color=plt.get_cmap('tab20')(icity*2))
#     plt.plot(mid_month,monthly_clim[:,icity]+2*monthly_std[:,icity],color=plt.get_cmap('tab20')(icity*2+1))
#     plt.plot(mid_month,monthly_clim[:,icity]-2*monthly_std[:,icity],color=plt.get_cmap('tab20')(icity*2+1))
#     plt.plot(mid_month,monthly_p5[:,icity],':',color=plt.get_cmap('tab20')(icity*2+1))
#     plt.plot(mid_month,monthly_p95[:,icity],':',color=plt.get_cmap('tab20')(icity*2+1))
#     plt.plot(mid_month,monthly_p2[:,icity],':',color=plt.get_cmap('tab20')(icity*2))
#     plt.plot(mid_month,monthly_p98[:,icity],':',color=plt.get_cmap('tab20')(icity*2))

#%%
days_per_month = [31,29,31,30,31,30,31,31,30,31,30,31]
month_doy = [1,32,61,92,122,153,183,214,245,275,306,336]

monthly_clim_y_Tw = np.zeros((366,len(water_cities_name_list)))*np.nan
monthly_std_y_Tw = np.zeros((366,len(water_cities_name_list)))*np.nan
monthly_p2_y_Tw = np.zeros((366,len(water_cities_name_list)))*np.nan
monthly_p5_y_Tw = np.zeros((366,len(water_cities_name_list)))*np.nan
monthly_p95_y_Tw = np.zeros((366,len(water_cities_name_list)))*np.nan
monthly_p98_y_Tw = np.zeros((366,len(water_cities_name_list)))*np.nan

monthly_clim_y_Ta = np.zeros((366))*np.nan
monthly_std_y_Ta = np.zeros((366))*np.nan
monthly_p2_y_Ta = np.zeros((366))*np.nan
monthly_p5_y_Ta = np.zeros((366))*np.nan
monthly_p95_y_Ta = np.zeros((366))*np.nan
monthly_p98_y_Ta = np.zeros((366))*np.nan

for icity in range(len(water_cities_name_list)):
    for imonth in range(12):
        i0 = month_doy[imonth]-1
        i1 = month_doy[imonth]+days_per_month[imonth]-1

        c = monthly_clim_Tw[imonth,icity]
        monthly_clim_y_Tw[i0:i1,icity] = np.ones(monthly_clim_y_Tw[i0:i1,icity].shape)*c

        c = monthly_std_Tw[imonth,icity]
        monthly_std_y_Tw[i0:i1,icity] = np.ones(monthly_std_y_Tw[i0:i1,icity].shape)*c

        c = monthly_p2_Tw[imonth,icity]
        monthly_p2_y_Tw[i0:i1,icity] = np.ones(monthly_p2_y_Tw[i0:i1,icity].shape)*c

        c = monthly_p5_Tw[imonth,icity]
        monthly_p5_y_Tw[i0:i1,icity] = np.ones(monthly_p5_y_Tw[i0:i1,icity].shape)*c

        c = monthly_p95_Tw[imonth,icity]
        monthly_p95_y_Tw[i0:i1,icity] = np.ones(monthly_p95_y_Tw[i0:i1,icity].shape)*c

        c = monthly_p98_Tw[imonth,icity]
        monthly_p98_y_Tw[i0:i1,icity] = np.ones(monthly_p98_y_Tw[i0:i1,icity].shape)*c


        c = monthly_clim_Ta[imonth]
        monthly_clim_y_Ta[i0:i1] = np.ones(monthly_clim_y_Ta[i0:i1].shape)*c

        c = monthly_std_Ta[imonth]
        monthly_std_y_Ta[i0:i1] = np.ones(monthly_std_y_Ta[i0:i1].shape)*c

        c = monthly_p2_Ta[imonth]
        monthly_p2_y_Ta[i0:i1] = np.ones(monthly_p2_y_Ta[i0:i1].shape)*c

        c = monthly_p5_Ta[imonth]
        monthly_p5_y_Ta[i0:i1,] = np.ones(monthly_p5_y_Ta[i0:i1].shape)*c

        c = monthly_p95_Ta[imonth]
        monthly_p95_y_Ta[i0:i1] = np.ones(monthly_p95_y_Ta[i0:i1].shape)*c

        c = monthly_p98_Ta[imonth]
        monthly_p98_y_Ta[i0:i1] = np.ones(monthly_p98_y_Ta[i0:i1].shape)*c


#%%

t0  = 2.5
t0a = 2.5
for iyear,year in enumerate(years):
    fig1,ax1 = plt.subplots(nrows=2,ncols=2,figsize=(12,10),sharex=True)
    plt.suptitle(years[iyear])

    Ta = data_Ta[:,iyear]
    clim_Ta = monthly_clim_y_Ta
    std_Ta = monthly_std_y_Ta

    ax1[1,0].plot(Ta)
    ax1[1,1].plot(Ta-clim_Ta,color=plt.get_cmap('tab20')(0))
    ax1[1,1].plot(std_Ta,color=plt.get_cmap('tab20')(1))
    ax1[1,1].plot(-std_Ta,color=plt.get_cmap('tab20')(1))
    ax1[1,1].plot(t0*std_Ta,color=plt.get_cmap('tab20')(1))
    ax1[1,1].plot(-t0*std_Ta,color=plt.get_cmap('tab20')(1))

    for icity,city in enumerate(water_cities_name_list):
        Twater = data_Tw[:,iyear,icity]
        clim_Tw = monthly_clim_y_Tw[:,icity]
        std_Tw = monthly_std_y_Tw[:,icity]

        Twater_filtered = Twater.copy()
        mask_Tw = np.abs(Twater-clim_Tw) > t0*std_Tw
        mask_Ta = np.abs(Ta-clim_Ta) <= t0a*std_Ta
        mask = mask_Tw & mask_Ta
        Twater_filtered[mask]=np.nan

        ax1[0,0].plot(Twater,color=plt.get_cmap('tab20')(icity*2))
        ax1[0,0].plot(Twater_filtered,color=plt.get_cmap('tab20')(icity*2+1))
        ax1[0,1].plot(Twater-clim_Tw,color=plt.get_cmap('tab20')(icity*2))
        # ax1[0,1].plot(std_Tw,color=plt.get_cmap('tab20')(icity*2+1))
        # ax1[0,1].plot(-std_Tw,color=plt.get_cmap('tab20')(icity*2+1))
        ax1[0,1].plot(t0*std_Tw,color=plt.get_cmap('tab20')(icity*2+1))
        ax1[0,1].plot(-t0*std_Tw,color=plt.get_cmap('tab20')(icity*2+1))





