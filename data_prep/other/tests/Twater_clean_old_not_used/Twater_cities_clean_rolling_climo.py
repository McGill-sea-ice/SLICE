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
#%% =========================================================================
# def running_nanmean0(x, N=3, window_mode='centered'):
#     xmean = np.ones(x.shape[0])*np.nan
#     temp = np.vstack([x[i:-(N-i)].copy() for i in range(N)].copy()) # stacks vertically the strided arrays
#     temp = np.nanmean(temp, axis=0)

#     if window_mode == 'centered':
#         if N%2 == 0:
#             print('ERROR! Window size should be odd when using the centered window mode.')
#         else:
#             xmean[int((N-1)/2):int(((N-1)/2)+temp.shape[0])]= temp
#     if window_mode == 'before':
#         xmean[N-1:N-1+temp.shape[0]] = temp

#     return xmean

# def running_nanmean(x, N=3, window_mode='centered'):
#     xmean = np.ones(x.shape[0])*np.nan
#     for i in range(x.shape[0]):
#         if window_mode == 'centered':
#             if N%2 == 0:
#                 print('ERROR! Window size should be odd when using the centered window mode.')
#             else:
#                 iw0 = np.max([0,i-int((N-1)/2)])
#                 iw1 = i+int((N-1)/2)+1
#         if window_mode == 'before':
#             iw0 = np.max([0,i+1-N])
#             iw1 = i+1

#         xmean[i] = np.nanmean(x[iw0:iw1])

#     return xmean


def running_nanmean(x, N=3):
    xmean = np.ones(x.shape[0])*np.nan
    temp = np.vstack([x[i:-(N-i)] for i in range(N)]) # stacks vertically the strided arrays
    temp = np.nanmean(temp, axis=0)
    xmean[N-1:N-1+temp.shape[0]] = temp

    return xmean



#%% =========================================================================
# years = [1992,1993,1994,1995,1996,
#           1997,1998,1999,2000,2001,
#            2002,2003,2004,2005,2006,
#           2007,2008,2009,2010,2011,
#           2012,2013,2014,2015,2016,2017,
#           2018,2019,2020
#             ]
years = [2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020]
# years = [2017,2018,2019,2020]
# water_cities_name_list = ['Longueuil','Atwater','Candiac','DesBaillets']
water_cities_name_list = ['Candiac','Longueuil','Atwater','DesBaillets']
# water_cities_name_list = ['Longueuil','Atwater']
# water_cities_name_list = ['Candiac','Longueuil','Atwater']
# water_cities_name_list = ['Candiac']
# water_cities_name_list = ['Longueuil']

weather_name_list = ['MontrealDorvalMontrealPETMontrealMcTavishmerged']

fp = '../../../../data/processed/'

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2020,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

loc_weather = weather_name_list[0]
weather_data = np.load(fp+'weather_NCEI/weather_NCEI_'+loc_weather+'.npz',allow_pickle='TRUE')
weather_data = weather_data['weather_data']
Ta = weather_data[:,3]

#%%
# Twater_rolling_mean = np.zeros((len(time),len(water_cities_name_list)))*np.nan

# for icity,city in enumerate(water_cities_name_list):
#     loc_water_city = water_cities_name_list[icity]
#     water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
#     Twater = water_city_data['Twater'][:,1]

#     Twater_rolling_mean[:,icity] = running_nanmean(Twater,31,'centered')
#     # plt.figure();plt.plot(Twater,'.-');plt.plot(Twater_rolling_mean[:,icity])

#%%
# First re-arrange data to have each 31-day window, for each date, each year, each city
Nwindow = 31 # Only odd window size are possible
# Nwindow = 91 # Only odd window size are possible
data_Tw = np.zeros((Nwindow,366,len(years),len(water_cities_name_list)))*np.nan
data_Ta = np.zeros((Nwindow,366,len(years),len(water_cities_name_list)))*np.nan

years = np.array(years)

for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][:,1]

    for it in range(Twater.shape[0]):

        iw0 = np.max([0,it-int((Nwindow-1)/2)])
        iw1 = it+int((Nwindow-1)/2)+1

        Twater_window = Twater[iw0:iw1]
        Ta_window = Ta[iw0:iw1]
        date_mid = date_ref+dt.timedelta(days=int(time[it]))
        year_mid = date_mid.year
        month_mid = date_mid.month
        day_mid = date_mid.day

        if len(np.where(years == year_mid)[0]) > 0:
            iyear = np.where(years == year_mid)[0][0]
            doy = (dt.date(year_mid,month_mid,day_mid)-dt.date(year_mid,1,1)).days

            data_Tw[0:len(Twater_window),doy,iyear,icity] = Twater_window
            data_Ta[0:len(Ta_window),doy,iyear,icity] = Ta_window

            if not calendar.isleap(year_mid) and (doy == 364):
                imid = int((Nwindow-1)/2)
                Twater_window_366 = np.zeros((Nwindow))*np.nan
                Twater_window_366[imid] = np.array(np.nanmean([Twater[it],Twater[it+1]]))
                Twater_window_366[0:imid] = Twater[int(it+1-((Nwindow-1)/2)):it+1]
                Twater_window_366[imid+1:Nwindow] = Twater[it+1:int(it+1+((Nwindow-1)/2))]
                data_Tw[:,365,iyear,icity] = Twater_window_366

                Ta_window_366 = np.zeros((Nwindow))*np.nan
                Ta_window_366[imid] = np.array(np.nanmean([Ta[it],Ta[it+1]]))
                Ta_window_366[0:imid] = Ta[int(it+1-((Nwindow-1)/2)):it+1]
                Ta_window_366[imid+1:Nwindow] = Ta[it+1:int(it+1+((Nwindow-1)/2))]
                data_Ta[:,365,iyear,icity] = Ta_window_366


# Then, find the 31-day climatological mean and std for each date
mean_clim_Tw = np.zeros((366,len(water_cities_name_list)))*np.nan
std_clim_Tw = np.zeros((366,len(water_cities_name_list)))*np.nan
std_clim_Tw_test = np.zeros((366,len(water_cities_name_list)))*np.nan

p98_clim_Tw = np.zeros((366,len(water_cities_name_list)))*np.nan
p2_clim_Tw = np.zeros((366,len(water_cities_name_list)))*np.nan

mean_clim_Ta = np.zeros((366,len(water_cities_name_list)))*np.nan
std_clim_Ta = np.zeros((366,len(water_cities_name_list)))*np.nan
p98_clim_Ta = np.zeros((366,len(water_cities_name_list)))*np.nan
p2_clim_Ta = np.zeros((366,len(water_cities_name_list)))*np.nan

for icity,city in enumerate(water_cities_name_list):

    data = data_Tw[:,:,:,icity]
    mean_clim_Tw[:,icity] = np.nanmean(data,axis=(0,2))
    std_clim_Tw[:,icity] = np.nanstd(data,axis=(0,2))
    # std_clim_Tw_test[:,icity] = np.nanstd(data-np.nanmean(data,axis=(0,2)),axis=(0,2))
    p98_clim_Tw[:,icity] = np.nanpercentile(data,98,axis=(0,2))
    p2_clim_Tw[:,icity] = np.nanpercentile(data,2,axis=(0,2))

    data = data_Ta[:,:,:,icity]
    mean_clim_Ta[:,icity] = np.nanmean(data,axis=(0,2))
    std_clim_Ta[:,icity] = np.nanstd(data,axis=(0,2))
    p98_clim_Ta[:,icity] = np.nanpercentile(data,98,axis=(0,2))
    p2_clim_Ta[:,icity] = np.nanpercentile(data,2,axis=(0,2))

    plt.figure()
    plt.plot(mean_clim_Tw[:,icity])
    plt.plot(mean_clim_Tw[:,icity]+std_clim_Tw[:,icity],':',color='gray')
    plt.plot(mean_clim_Tw[:,icity]-std_clim_Tw[:,icity],':',color='gray')

    # if icity == 0:
    #     plt.figure()
    #     plt.plot(mean_clim_Ta[:,icity])
    #     plt.plot(mean_clim_Ta[:,icity]+std_clim_Ta[:,icity],':',color='gray')
    #     plt.plot(mean_clim_Ta[:,icity]-std_clim_Ta[:,icity],':',color='gray')

#%%
for icity,city in enumerate(water_cities_name_list):
    plt.figure()
    Twater_rm = np.zeros((366,len(years)))
    for iyear, year in enumerate(years):
        imid = int((Nwindow-1)/2)
        Twater = data_Tw[imid,:,iyear,icity]
        clim_Tw = mean_clim_Tw[:,icity]
        std_Tw = std_clim_Tw[:,icity]
        # Twater_rm[:,iyear]=running_nanmean(data_Tw[-1,:,iyear,icity],N=31)
        plt.plot(Twater,color='gray',linewidth=0.5)

    plt.plot(clim_Tw)
    plt.plot(clim_Tw+std_Tw,'--',color='black')
    plt.plot(clim_Tw-std_Tw,'--',color='black')
    plt.plot(clim_Tw+2*std_Tw,'--',color='black')
    plt.plot(clim_Tw-2*std_Tw,'--',color='black')
    plt.plot(clim_Tw+3*std_Tw,'--',color='black')
    plt.plot(clim_Tw-3*std_Tw,'--',color='black')
    # Twater_rm = np.nanmean(Twater_rm,axis=1)
    # plt.plot(Twater_rm,'--')

#%%

# t0  = 2.
# t0a  = 2.
# for iyear,year in enumerate(years):
#     fig1,ax1 = plt.subplots(nrows=2,ncols=2,figsize=(12,10),sharex=True)
#     plt.suptitle(years[iyear])

#     Ta = data_Ta[imid,:,iyear,0]
#     clim_Ta = mean_clim_Ta[:,0]
#     std_Ta = std_clim_Ta[:,0]

#     mask_Ta = np.abs(Ta-clim_Ta) <= t0a*std_Ta

#     # mask_Ta = mask_Ta.astype('bool')
#     # for ia in range(Ta.shape[0]-4):
#     #     mask_Ta[ia+4] = ~np.any(~mask_Ta[ia:ia+4])


#     ax1[1,0].plot(Ta)
#     ax1[1,1].plot(Ta-clim_Ta,color=plt.get_cmap('tab20')(0))
#     ax1[1,1].plot(std_Ta,color=plt.get_cmap('tab20')(1))
#     ax1[1,1].plot(-std_Ta,color=plt.get_cmap('tab20')(1))
#     ax1[1,1].plot(t0a*std_Ta,color=plt.get_cmap('tab20')(1))
#     ax1[1,1].plot(-t0a*std_Ta,color=plt.get_cmap('tab20')(1))


#     for icity,city in enumerate(water_cities_name_list):
#         imid = int((Nwindow-1)/2)
#         Twater = data_Tw[imid,:,iyear,icity]
#         clim_Tw = mean_clim_Tw[:,icity]
#         std_Tw = std_clim_Tw[:,icity]

#         mask_Tw = np.abs(Twater-clim_Tw) > t0*std_Tw
#         mask = mask_Tw & mask_Ta
#         # mask = mask_Tw

#         Twater_filtered = Twater.copy()
#         Twater_filtered[mask]=np.nan

#         ax1[0,0].plot(Twater,color=plt.get_cmap('tab20')(icity*2),label=str(city))
#         ax1[0,0].plot(Twater_filtered,color=plt.get_cmap('tab20')(icity*2+1))
#         ax1[0,1].plot(Twater-clim_Tw,color=plt.get_cmap('tab20')(icity*2))
#         # ax1[0,1].plot(std_Tw,color=plt.get_cmap('tab20')(icity*2+1))
#         # ax1[0,1].plot(-std_Tw,color=plt.get_cmap('tab20')(icity*2+1))
#         ax1[0,1].plot(t0*std_Tw,color=plt.get_cmap('tab20')(icity*2+1))
#         ax1[0,1].plot(-t0*std_Tw,color=plt.get_cmap('tab20')(icity*2+1))


#     ax1[0,0].legend()

# #%%

# t0  = 3.
# t0a  = 3.
# for iyear,year in enumerate(years):
#     fig1,ax1 = plt.subplots(nrows=2,ncols=1,figsize=(6,10),sharex=True)
#     plt.suptitle(years[iyear])

#     for icity,city in enumerate(water_cities_name_list):
#         imid = int((Nwindow-1)/2)
#         Twater = data_Tw[imid,:,iyear,icity]
#         clim_Tw = mean_clim_Tw[:,icity]
#         std_Tw = std_clim_Tw[:,icity]
#         p98_Tw = p98_clim_Tw[:,icity]
#         p2_Tw = p2_clim_Tw[:,icity]

#         mask_std_Tw = np.abs(Twater-clim_Tw) > t0*std_Tw
#         mask_std = mask_std_Tw

#         mask_percentile98_Tw = Twater > p98_Tw
#         mask_percentile2_Tw = Twater < p2_Tw
#         mask_percentile = mask_percentile98_Tw | mask_percentile2_Tw

#         Twater_std_filtered = Twater.copy()
#         Twater_std_filtered[mask_std]=np.nan

#         Twater_percentile_filtered = Twater.copy()
#         Twater_percentile_filtered[mask_percentile]=np.nan

#         ax1[0].plot(Twater,color=plt.get_cmap('tab20')(icity*2),label=str(city))
#         ax1[0].plot(Twater_std_filtered,color=plt.get_cmap('tab20')(icity*2+1))

#         ax1[1].plot(Twater,color=plt.get_cmap('tab20')(icity*2),label=str(city))
#         ax1[1].plot(Twater_percentile_filtered,color=plt.get_cmap('tab20')(icity*2+1))



#     ax1[0].legend()


# #%%

# t0  = 2.
# t0a  = 2.
# for iyear,year in enumerate(years):
#     fig1,ax1 = plt.subplots(nrows=1,ncols=2,figsize=(10,5),sharex=True)
#     plt.suptitle(years[iyear])

#     Ta = data_Ta[imid,:,iyear,0]
#     clim_Ta = mean_clim_Ta[:,0]
#     std_Ta = std_clim_Ta[:,0]

#     mask_Ta = np.abs(Ta-clim_Ta) <= t0a*std_Ta

#     for icity,city in enumerate(water_cities_name_list):
#         imid = int((Nwindow-1)/2)
#         Twater = data_Tw[imid,:,iyear,icity]
#         clim_Tw = mean_clim_Tw[:,icity]
#         std_Tw = std_clim_Tw[:,icity]

#         mask_Tw = np.abs(Twater-clim_Tw) > t0*std_Tw
#         mask = mask_Tw & mask_Ta
#         # mask = mask_Tw

#         Twater_filtered = Twater.copy()
#         Twater_filtered[mask]=np.nan

#         ax1[0].plot(Twater,color=plt.get_cmap('tab20')(icity*2),label=str(city))
#         ax1[0].plot(Twater_filtered,color=plt.get_cmap('tab20')(icity*2+1))
#         ax1[1].plot(Twater-clim_Tw,color=plt.get_cmap('tab20')(icity*2))
#         # ax1[1].plot(std_Tw,color=plt.get_cmap('tab20')(icity*2+1))
#         # ax1[1].plot(-std_Tw,color=plt.get_cmap('tab20')(icity*2+1))
#         ax1[1].plot(t0*std_Tw,color=plt.get_cmap('tab20')(icity*2+1))
#         ax1[1].plot(-t0*std_Tw,color=plt.get_cmap('tab20')(icity*2+1))

#     ax1[0].legend()



