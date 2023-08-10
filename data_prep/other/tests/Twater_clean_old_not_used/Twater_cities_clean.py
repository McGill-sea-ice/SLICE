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


# ==========================================================================
# years = [1994,1995,1996,1997,
#           1998,1999,2000,2001,2002,2003,2004] # Atwater
# years = [2011,2012,
#             2013,2014,2015,2016,2017,
#             2018,2019] # Atwater

# years = [2006,2007,2008,
#           2009,2010,2011,2012,
#           2013,2014,2015,2016,2017,
#           2018,2019] # Des Baillets

# years = [2004,2005,2006,2007,2008,
#          2009,2010,2011,2012,
#          2013,2014,2015,2016,2017,
#          2018,2019] # Candiac

years = [1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019
            ] # Longueuil


# years = [2005,2015,2017,2018]
# years = [1999,2000,2001,2002]
# years = [2007,
#          2010,2012,
#          2016,2017]
# years = [2004,2011,2013,2015,2017,2019]

# water_cities_name_list = 'DesBaillets_clean'
# water_cities_name_list = 'DesBaillets'
# water_cities_name_list = 'Atwater'
# water_cities_name_list = 'Candiac'
# water_cities_name_list = ['Longueuil','Atwater','Candiac','DesBaillets']
# water_cities_name_list = ['Longueuil','DesBaillets','Atwater','Candiac']
# water_cities_name_list = ['Longueuil','Atwater']
water_cities_name_list = ['Candiac','Longueuil','Atwater','DesBaillets']
# water_cities_name_list = ['Candiac']
# water_cities_name_list = ['Longueuil']

fp = '../../data/'

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2020,11,1)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)



# for icity,city in enumerate(water_cities_name_list):
#     loc_water_city = water_cities_name_list[icity]
#     water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
#     Twater_city = water_city_data['Twater']
#     plt.figure()
#     plt.plot(Twater_city[:,1])

#%%

plt.figure()
climo = np.zeros((366,len(years),len(water_cities_name_list)))*np.nan

for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater_city = water_city_data['Twater']

    for iyear,year in enumerate(years):

        date=(dt.date(year,1,1)-date_ref).days
        i0 = np.where(time==date)[0][0]
        i1 = i0+365+calendar.isleap(year)

        climo[0:365+calendar.isleap(year),iyear,icity] = Twater_city[i0:i1,1].copy()
        if not calendar.isleap(year):
            climo[-1,iyear,icity] = np.nanmean([Twater_city[i0:i1,1][-1],Twater_city[i1,1]])

        plt.plot(climo[:,iyear,icity],color='gray',linewidth=1)

    daily_climo = np.nanmean(climo,1)
    climo_std = np.nanstd(climo,1)

plt.plot(daily_climo,color='black',linewidth=2)
plt.plot(daily_climo+climo_std,color='red',linewidth=1)
plt.plot(daily_climo-climo_std,color='red',linewidth=1)

#%%


for iyear,year in enumerate(years):
    fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(8,5),sharex=True)
    plt.title(years[iyear])
    for icity,city in enumerate(water_cities_name_list):
        loc_water_city = water_cities_name_list[icity]
        water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
        Twater_city = water_city_data['Twater']

        Tw_clean = np.zeros(Twater_city.shape)*np.nan

        date=(dt.date(year,1,1)-date_ref).days
        i0 = np.where(time==date)[0][0]
        i1 = i0+365+calendar.isleap(year)


        Tw_city_yr = Twater_city[i0:i1,1].copy()
        Date_yr = Twater_city[i0:i1,0].copy()

        Tw_clean[i0:i1,1]= Tw_city_yr-daily_climo[0:len(Tw_city_yr),icity]
        Tw_clean[i0:i1,0]= Date_yr

        ax[0].plot(climo[:,iyear,icity])
        ax[1].plot(Tw_clean[i0:i1,1])



# MAKE A MONTHLY CLIMATOLOGY
# COMPUTE MONTHLY STD
# COMPUTE MONTHLY 5TH AND 95 TH PERCENTILE


#%%
# # plt.figure()
# for iyear in range(climo.shape[1]):
#     plt.figure()
#     plt.plot(climo[:,iyear])
#     plt.title(str(years[iyear]))
#     # plt.plot(medfilt(climo[:,iyear],3))


#%%
def running_mean(x, N=3, mode='centered'):
    cumsum = np.nancumsum(np.insert(x, 0, 0))
    xmean_tmp = (cumsum[N:] - cumsum[:-N]) / float(N)
    if mode == 'centered':
        xmean = np.insert(xmean_tmp,0,np.zeros(int((N-1)/2.))*np.nan)
        xmean = np.insert(xmean,xmean.size, np.zeros(int((N-1)/2.))*np.nan)
    if mode == 'before':
        xmean = np.insert(xmean_tmp,0,np.zeros(int((N-1)))*np.nan)

    return xmean


def running_nanmean(x, N=3):
    xmean = np.ones(x.shape[0])*np.nan
    temp = np.vstack([x[i:-(N-i)] for i in range(N)]) # stacks vertically the strided arrays
    temp = np.nanmean(temp, axis=0)
    xmean[N-1:N-1+temp.shape[0]] = temp

    return xmean


m=3
for iyear,year in enumerate(years):
    # plt.figure()
    fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(8,5),sharex=True)
    plt.title(years[iyear])

    date=(dt.date(year,1,1)-date_ref).days
    i0 = np.where(time==date)[0][0]
    i1 = i0+365+calendar.isleap(year)

    for icity,city in enumerate(water_cities_name_list):
        loc_water_city = water_cities_name_list[icity]
        water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
        Twater_city = water_city_data['Twater']

        Tplot= Twater_city[i0:i1,1].copy()

        ax[0].plot(Tplot, color=plt.get_cmap('tab20')(icity*2))
        # ax[0].plot(running_nanmean(Tplot,m), color=plt.get_cmap('tab20')(icity*2+1))

        res = (Tplot-running_nanmean(Tplot,m))
        ax[1].plot(res,color=plt.get_cmap('tab20')(icity*2))
        ax[1].plot(np.arange(len(res)), 4.*np.nanstd(res)*np.ones(len(res)),'--',color=plt.get_cmap('tab20')(icity*2+1))
        # ax[1].plot(np.arange(len(res)),-3*np.nanstd(res)*np.ones(len(res)),'--',color=plt.get_cmap('tab20')(icity*2+1))

        outliersT = Tplot.copy()
        outliersR = Tplot.copy()*np.nan
        h_cr = 4.*np.nanstd(res)*np.ones(len(res))
        # l_cr = -3*np.nanstd(res)*np.ones(len(res))
        for i in range(len(res)):
            if (res[i] >= h_cr[i]):
                outliersR[i]=res[i]
                outliersT[i]=np.nan
        ax[0].plot(outliersT,color=plt.get_cmap('tab20')(icity*2+1))
        ax[1].plot(outliersR,'*',color=plt.get_cmap('tab20')(icity*2+1))

        # if icity == 0:
        #     # plt.plot(medfilt(Tplot,3), color=plt.get_cmap('tab20')(4))
        #     plt.plot(running_nanmean(Tplot,m), color=plt.get_cmap('tab20')(1))


#%%

# def hampel(x, k=5, t0=3, exclude_crrt_point=False, nan_substitution=False):
#     '''Perform hampel filtering, returning both filtered series and
#     mask of filtered points.
#     Input:
#         x: 1-d numpy array of numbers to be filtered
#         k: number of items in (window-1)/2, i.e. using k elems before and after
#             in addition to the current point
#         t0: number of standard deviations to use; 3 is default
#         exclude_crrt_point: if the current point being inspected should be
#             ignored when calculating variance.
#         nan_substitution: if True then invalid points should be replaced by a
#             NaN value. If False, interpolate with the median value.
#     Output:
#         y: 1-d numpy array obtained by filtering (this is a np.copy of x),
#             NaN where discarded
#         mask_modified: boolean mask, True if point has been discarded or
#             modified by the filter
#     '''
#     # NOTE: This code is from: https://github.com/scipy/scipy/issues/12809
#     # NOTE: adapted from hampel function in R package pracma
#     # NOTE: this is adapted from: https://stackoverflow.com/questions/46819260/
#     # filtering-outliers-how-to-make-median-based-hampel-function-faster
#     # NOTE: adapted from the issue by Jean Rabault jean.rblt@gmail.com 09-2020

#     if not isinstance(x, np.ndarray):
#         raise ValueError("x should be a numpy array")

#     if not len(np.squeeze(x).shape) == 1:
#         raise ValueError("x should be 1-dimensional")

#     if not isinstance(k, int):
#         raise ValueError("k should be an int")

#     if not isinstance(t0, int):
#         raise ValueError("t0 should be an int")

#     y = np.copy(x)  # y is the corrected series
#     y = np.squeeze(y)

#     mask_modified = np.zeros((y.shape[0]), dtype=bool)

#     n = y.shape[0]
#     L = 1.4826

#     # cannot apply the filter too close to the edges, as we need k points
#     # before and after
#     for i in range((k), (n - k)):

#         # follow user preference on using or not the current point for
#         # estimating statistical properties
#         if exclude_crrt_point:
#             array_neighborhood = np.concatenate((
#                 y[(i - k):(i)],
#                 y[(i + 1):(i + k + 1)]
#             ))
#         else:
#             array_neighborhood = y[i - k: i + k + 1]

#         # if all points around are already nans, cannot trust local point
#         if np.all(np.isnan(array_neighborhood)):
#             if not np.isnan(y[i]):
#                 y[i] = np.nan
#                 mask_modified[i] = True
#             continue

#         # if current point is already a nan, keep it so
#         if np.isnan(y[i]):
#             continue

#         # otherwise, should perform the filtering
#         x0 = np.nanmedian(array_neighborhood)
#         S0 = L * np.nanmedian(np.abs(array_neighborhood - x0))

#         if (np.abs(y[i] - x0) > t0 * S0):
#             if nan_substitution:
#                 y[i] = np.nan
#             else:
#                 y[i] = x0
#             mask_modified[i] = True

#     return (y, mask_modified)



# for iyear,year in enumerate(years):
#     fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(21,5),sharex=True)
#     plt.title(years[iyear])

#     date=(dt.date(year,1,1)-date_ref).days
#     i0 = np.where(time==date)[0][0]
#     i1 = i0+365+calendar.isleap(year)

#     filt_array0 = np.zeros((366,len(water_cities_name_list)))
#     filt_array1 = np.zeros((366,len(water_cities_name_list)))
#     corr_array = np.zeros((366,len(water_cities_name_list)))

#     for icity,city in enumerate(water_cities_name_list):
#         loc_water_city = water_cities_name_list[icity]
#         water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
#         Twater_city = water_city_data['Twater']

#         Tplot= Twater_city[i0:i1,1].copy()
#         Tfilter1,_ = hampel(Tplot,k=7,t0=3, exclude_crrt_point=False)
#         ax.plot(Tplot, color=plt.get_cmap('tab20')(icity*2),label=str(city))
#         ax.plot(Tfilter1, color=plt.get_cmap('tab20')(icity*2+1))
#         # filt_array0[0:365+calendar.isleap(year),icity] = Tfilter

#         # Tplot= running_nanmean(Twater_city[i0:i1,1].copy())
#         # Tfilter2,_ = hampel(Tplot,k=7,t0=2, exclude_crrt_point=True)
#         # ax[1].plot(Tplot, color=plt.get_cmap('tab20')(icity*2))
#         # ax[1].plot(Tfilter2, color=plt.get_cmap('tab20')(icity*2+1))
#         # # filt_array1[0:365+calendar.isleap(year),icity] = Tfilter

#         # ax[2].plot(Tfilter1, color=plt.get_cmap('tab20')(icity*2))
#         # ax[2].plot(Tfilter2, color=plt.get_cmap('tab20')(icity*2+1))

#     ax.legend()

#%%
# fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(16,5),sharex=True)
# ax[0].imshow(C0)
# ax[1].imshow(C1)
# print(np.nanmean(C0))
# print(np.nanmean(C1))
# print(np.nanmean(Cref))

#%%

# ts_out = medfilt(climo[:,3],3)
# plt.figure()
# plt.plot(climo[:,3])
# plt.plot(ts_out)

#%%

# years = [2011,2012,
#            2013,2014,2015,2016,2017] # Atwater
# water_cities_name_list = ['Longueuil']
# icity=0

# window = 15
# thresh = 3
# for iyear,year in enumerate(years):
#     loc_water_city = water_cities_name_list[icity]
#     water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
#     Twater_city = water_city_data['Twater']

#     date=(dt.date(year,1,1)-date_ref).days
#     i0 = np.where(time==date)[0][0]
#     i1 = i0+365+calendar.isleap(year)

#     Tplot = Twater_city[i0:i1,1].copy()
#     Date_yr = Twater_city[i0:i1,0].copy()

#     flag = np.zeros(Tplot.shape)*np.nan
#     for iw in range(len(Tplot)-(window-1)):
#         Tw_window = Tplot[iw:iw+window]
#         window_median = np.nanmedian(Tw_window)
#         window_std = np.nanstd(np.abs(Tw_window-window_median))

#         Tw_lowp = np.percentile(Tw_window,2)
#         Tw_highp= np.percentile(Tw_window,98)

#         # if (iw < 160) & (iw > 150):
#         #     plt.figure()
#         #     plt.plot(np.arange(window),Tw_window)
#         #     plt.plot(np.arange(window),np.ones(window)*thresh*window_std,'-',color='gray')
#         #     plt.plot(np.arange(window),np.ones(window)*(-thresh*window_std),'-',color='gray')

#         mid_window_id = int((window-1)/2)
#         if np.abs(Tw_window[mid_window_id]-window_median) > thresh*window_std:
#             flag[iw+mid_window_id]= Tw_window[mid_window_id]

#         # mid_window_id = int((window-1)/2)
#         # if (Tw_window[mid_window_id] > Tw_highp) | (Tw_window[mid_window_id] < Tw_lowp):
#         #     flag[iw+mid_window_id]= Tw_window[mid_window_id]

#         # if iw == 0:
#         #     # Tw_window[0:mid_window_id][np.abs(Tw_window[0:mid_window_id]) > thresh*window_std] = np.nan
#         #     flag[0:mid_window_id][np.abs(Tw_window[0:mid_window_id]) > thresh*window_std] = 1
#         # elif iw == (len(Tw_clean_yr)-(window-1)-1):
#         #      # Tw_window[mid_window_id:-1][np.abs(Tw_window[mid_window_id:-1]) > thresh*window_std] = np.nan
#         #     flag[mid_window_id:-1][np.abs(Tw_window[mid_window_id:-1]) > thresh*window_std] = 1
#         # else:
#         #     if np.abs(Tw_window[mid_window_id]) > thresh*window_std:
#         #         # Tw_window[mid_window_id] = np.nan
#         #         flag[mid_window_id]= Tw_window[mid_window_id]


#     fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,5),sharex=True)

#     ax.plot(Date_yr,Tplot)
#     ax.plot(Date_yr,flag,'*')
#     plt.title(str(year))


# # (window-1)/2

# # 0:window
# # 1:window+1
# # 2:window+2

# # -1 - (window-1)/2
# # len(Tw_city_yr)


# # | x x X x x | x x x x x x x x x 0
# #  x | x X x | x x x x x x x x 1
# #  x x | x X x | x x x x x x x 2
# #  x x x | x X x | x x x x x x 3
# #  x x x x | x X x | x x x x x 4
# #  x x x x x | x X x | x x x x 5
# #  x x x x x x | x X x | x x x 6
# #  x x x x x x x | x X x | x x 7
# #  x x x x x x x x |x X x| x 8
# #  x x x x x x x x x |x X x| 9


#%%
# years = [2006,2007,
#           2008,2009,2010,2011,2012,
#           2013,2014,2015,2016,2017,
#           2018,2019]
# # years = [2017]
# for iyear,year in enumerate(years):

#     date=(dt.date(year,1,1)-date_ref).days
#     i0 = np.where(time==date)[0][0]
#     i1 = i0+365+calendar.isleap(year)


#     Tw_city_yr = Twater_city[i0:i1,1].copy()
#     Date_yr = Twater_city[i0:i1,0].copy()

#     fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(8,5),sharex=True)

#     ax[0].plot(Date_yr,Tw_city_yr)
#     ax[0].plot(Date_yr,annual_climo[0:len(Tw_city_yr)],color='black')
#     ax[0].plot(Date_yr,annual_climo[0:len(Tw_city_yr)]+3.*climo_std[0:len(Tw_city_yr)],color='gray')
#     ax[0].plot(Date_yr,annual_climo[0:len(Tw_city_yr)]-3.*climo_std[0:len(Tw_city_yr)],color='gray')

#     outliers = Tw_city_yr.copy()
#     for i in range(len(outliers)):
#         if (outliers[i] <= (annual_climo[0:len(Tw_city_yr)][i]+3.*climo_std[0:len(Tw_city_yr)][i])) & (outliers[i] >= (annual_climo[0:len(Tw_city_yr)][i]-3.*climo_std[0:len(Tw_city_yr)][i])) :
#             outliers[i]=np.nan
#     ax[0].plot(Date_yr,outliers,'*')



#     # ax[0].plot(Date_yr,Tw_city_yr)
#     # ax[0].plot(Date_yr,np.nanpercentile(climo[0:len(Date_yr),:],50,axis=1),color='black')
#     # ax[0].plot(Date_yr,np.nanpercentile(climo[0:len(Date_yr),:],98,axis=1),color='gray')
#     # ax[0].plot(Date_yr,np.nanpercentile(climo[0:len(Date_yr),:],2,axis=1),color='gray')

#     # outliers = Tw_city_yr.copy()
#     # for i in range(len(outliers)):
#     #     if (outliers[i] <= np.percentile(climo[0:len(Date_yr),:],98,axis=1)[i]) & (outliers[i] >= np.percentile(climo[0:len(Date_yr),:],2,axis=1)[i]) :
#     #         outliers[i]=np.nan
#     # ax[0].plot(Date_yr,outliers,'*')


#     ax[1].plot(Date_yr,Tw_clean[i0:i1,1])
#     ax[1].plot(Date_yr,np.ones(Date_yr.shape)*(np.nanmean(Tw_clean,0)[1]),color='black')
#     ax[1].plot(Date_yr,np.ones(Date_yr.shape)*(np.nanmean(Tw_clean,0)[1]+2.5*np.nanstd(Tw_clean,0)[1]),color='gray')
#     ax[1].plot(Date_yr,np.ones(Date_yr.shape)*(np.nanmean(Tw_clean,0)[1]-2.5*np.nanstd(Tw_clean,0)[1]),color='gray')

#     outliers = Tw_clean[i0:i1,1].copy()
#     for i in range(len(outliers)):
#         if np.abs(outliers[i]) <= (np.nanmean(Tw_clean,0)[1]+2.5*np.nanstd(Tw_clean,0)[1]):
#             outliers[i]=np.nan

#     ax[1].plot(Date_yr,outliers,'*')
#     # ax[1].plot(Date_yr,flag,'*')

#     plt.title(str(year))







