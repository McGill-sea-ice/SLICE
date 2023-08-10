#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:33:26 2021

@author: Amelie

"""

import numpy as np

import scipy.stats as sp

import datetime as dt
import calendar

import matplotlib.pyplot as plt


# ==========================================================================
save = False
plot_years = True

years = [1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020
            ]
# water_cities_name_list = ['Longueuil_preclean','Atwater_preclean','DesBaillets_preclean','Candiac_preclean']
# water_eccc_name_list = ['Lasalle', 'LaPrairie']
water_cities_name_list = ['Candiac','Longueuil','Atwater','DesBaillets']
water_eccc_name_list = []

loc_weather = 'MontrealDorvalMontrealPETMontrealMcTavishmerged'

fp = '../../data/'

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2020,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

weather_data = np.load(fp+'weather_NCEI/weather_NCEI_'+loc_weather+'.npz',allow_pickle='TRUE')
weather_data = weather_data['weather_data']
Ta = weather_data[:,[0,3]]

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


def hampel(x, k=5, t0=3., exclude_crrt_point=False, nan_substitution=False, w=None, corr_lim=1.):
    '''Perform hampel filtering, returning both filtered series and
    mask of filtered points.
    Input:
        x: 1-d numpy array of numbers to be filtered
        k: number of items in (window-1)/2, i.e. using k elems before and after
            in addition to the current point
        t0: number of standard deviations to use; 3 is default
        exclude_crrt_point: if the current point being inspected should be
            ignored when calculating variance.
        nan_substitution: if True then invalid points should be replaced by a
            NaN value. If False, interpolate with the median value.
    Output:
        y: 1-d numpy array obtained by filtering (this is a np.copy of x),
            NaN where discarded
        mask_modified: boolean mask, True if point has been discarded or
            modified by the filter
    '''
    # NOTE: This code is from: https://github.com/scipy/scipy/issues/12809
    # NOTE: adapted from hampel function in R package pracma
    # NOTE: this is adapted from: https://stackoverflow.com/questions/46819260/
    # filtering-outliers-how-to-make-median-based-hampel-function-faster
    # NOTE: adapted from the issue by Jean Rabault jean.rblt@gmail.com 09-2020

    if not isinstance(x, np.ndarray):
        raise ValueError("x should be a numpy array")

    if not len(np.squeeze(x).shape) == 1:
        raise ValueError("x should be 1-dimensional")

    if not isinstance(k, int):
        raise ValueError("k should be an int")

    # if not isinstance(t0, int):
        # raise ValueError("t0 should be an int")

    y = np.copy(x)  # y is the corrected series
    y = np.squeeze(y)

    if w is not None: z = np.copy(w)

    mask_modified = np.zeros((y.shape[0]), dtype=bool)

    n = y.shape[0]
    L = 1.4826

    # cannot apply the filter too close to the edges, as we need k points
    # before and after
    for i in range((k), (n - k)):

        # follow user preference on using or not the current point for
        # estimating statistical properties
        if exclude_crrt_point:
            array_neighborhood = np.concatenate((
                x[(i - k):(i)].copy(),
                x[(i + 1):(i + k + 1)].copy()
            ))
        else:
            array_neighborhood = x[i - k: i + k + 1].copy()


        if w is not None:
            if exclude_crrt_point:
                corr_array_neighborhood = np.concatenate((
                    z[(i - k):(i)],
                    z[(i + 1):(i + k + 1)]
                ))
            else:
                corr_array_neighborhood = z[i - k: i + k + 1]

        # if all points around are already nans, cannot trust local point
        if np.all(np.isnan(array_neighborhood)):
            if not np.isnan(y[i]):
                y[i] = np.nan
                mask_modified[i] = True
            continue

        # if current point is already a nan, keep it so
        if np.isnan(y[i]):
            continue

        # otherwise, should perform the filtering
        x0 = np.nanmedian(array_neighborhood)
        S0 = L * np.nanmedian(np.abs(array_neighborhood - x0))

        # Check the local correlation with second array
        rsqr = 0
        if w is not None:
            if ~np.all(np.isnan(array_neighborhood)) & ~np.all(np.isnan(corr_array_neighborhood)):
                maskx = ~np.isnan(array_neighborhood)
                masky = ~np.isnan(corr_array_neighborhood)
                mask = maskx & masky
                x_w = array_neighborhood[mask]
                y_w = corr_array_neighborhood[mask]
                if (len(x_w) > 1) & (len(y_w) > 1):
                    rsqr = sp.pearsonr(x_w,y_w)[0]**2


        # if (i > 17) & (i < 29) :
        #     xplot = np.arange(len(array_neighborhood))
        #     yplot = np.ones(len(array_neighborhood))
        #     plt.figure()
        #     plt.plot(array_neighborhood)
        #     plt.plot(xplot,yplot*x0,':', color = 'blue')
        #     plt.plot(xplot,yplot*t0*S0,color = 'black')
        #     plt.plot(k,np.abs(y[i] - x0),'*',color = 'black')
        #     plt.plot(k,y[i],'o',color = 'black')


        if (np.abs(y[i] - x0) > t0 * S0):
            if rsqr < corr_lim:
                if nan_substitution:
                    y[i] = np.nan
                else:
                    y[i] = x0
                mask_modified[i] = True


    return (y, mask_modified)



def hampel_mean(x, k=5, t0=3., exclude_crrt_point=False, nan_substitution=False, w=None, corr_lim=1.):
    '''Perform hampel filtering, returning both filtered series and
    mask of filtered points.
    Input:
        x: 1-d numpy array of numbers to be filtered
        k: number of items in (window-1)/2, i.e. using k elems before and after
            in addition to the current point
        t0: number of standard deviations to use; 3 is default
        exclude_crrt_point: if the current point being inspected should be
            ignored when calculating variance.
        nan_substitution: if True then invalid points should be replaced by a
            NaN value. If False, interpolate with the median value.
    Output:
        y: 1-d numpy array obtained by filtering (this is a np.copy of x),
            NaN where discarded
        mask_modified: boolean mask, True if point has been discarded or
            modified by the filter
    '''
    # NOTE: This code is from: https://github.com/scipy/scipy/issues/12809
    # NOTE: adapted from hampel function in R package pracma
    # NOTE: this is adapted from: https://stackoverflow.com/questions/46819260/
    # filtering-outliers-how-to-make-median-based-hampel-function-faster
    # NOTE: adapted from the issue by Jean Rabault jean.rblt@gmail.com 09-2020

    if not isinstance(x, np.ndarray):
        raise ValueError("x should be a numpy array")

    if not len(np.squeeze(x).shape) == 1:
        raise ValueError("x should be 1-dimensional")

    if not isinstance(k, int):
        raise ValueError("k should be an int")

    # if not isinstance(t0, int):
        # raise ValueError("t0 should be an int")

    y = np.copy(x)  # y is the corrected series
    y = np.squeeze(y)

    if w is not None: z = np.copy(w)

    mask_modified = np.zeros((y.shape[0]), dtype=bool)

    n = y.shape[0]
    L = 1.4826

    # cannot apply the filter too close to the edges, as we need k points
    # before and after
    for i in range((k), (n - k)):

        # follow user preference on using or not the current point for
        # estimating statistical properties
        if exclude_crrt_point:
            array_neighborhood = np.concatenate((
                x[(i - k):(i)].copy(),
                x[(i + 1):(i + k + 1)].copy()
            ))
        else:
            array_neighborhood = x[i - k: i + k + 1].copy()


        if w is not None:
            if exclude_crrt_point:
                corr_array_neighborhood = np.concatenate((
                    z[(i - k):(i)],
                    z[(i + 1):(i + k + 1)]
                ))
            else:
                corr_array_neighborhood = z[i - k: i + k + 1]

        # if all points around are already nans, cannot trust local point
        if np.all(np.isnan(array_neighborhood)):
            if not np.isnan(y[i]):
                y[i] = np.nan
                mask_modified[i] = True
            continue

        # if current point is already a nan, keep it so
        if np.isnan(y[i]):
            continue

        # otherwise, should perform the filtering
        x0 = np.nanmean(array_neighborhood)
        S0 = np.nanstd(array_neighborhood)

        # Check the local correlation with second array
        rsqr = 0
        if w is not None:
            if ~np.all(np.isnan(array_neighborhood)) & ~np.all(np.isnan(corr_array_neighborhood)):
                maskx = ~np.isnan(array_neighborhood)
                masky = ~np.isnan(corr_array_neighborhood)
                mask = maskx & masky
                x_w = array_neighborhood[mask]
                y_w = corr_array_neighborhood[mask]
                if (len(x_w) > 1) & (len(y_w) > 1):
                    rsqr = sp.pearsonr(x_w,y_w)[0]**2

        if (np.abs(y[i] - x0) > t0 * S0):
            if rsqr < corr_lim:
                if nan_substitution:
                    y[i] = np.nan
                else:
                    y[i] = x0
                mask_modified[i] = True


    return (y, mask_modified)


#%%
# FILTER CITY DATA SETS

mask_1 = np.zeros(len(water_cities_name_list))
mask_2 = np.zeros(len(water_cities_name_list))
mask_tot = np.zeros(len(water_cities_name_list))
N = np.zeros(len(water_cities_name_list))

Twater_clean = np.zeros((Ta.shape[0],len(water_cities_name_list)))*np.nan

for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][:,1]

    Tfilter1,filter_mask1 = hampel(Twater,k=15,t0=2., exclude_crrt_point=False,w=Ta[:,1],corr_lim=1,nan_substitution=True)
    Tfilter2,filter_mask2 = hampel(Twater,k=15,t0=2., exclude_crrt_point=False,w=Ta[:,1],corr_lim=0.35,nan_substitution=True)

    # Tfilter1[Tfilter1 < 0] = 0
    # Tfilter2[Tfilter2 < 0] = 0

    Twater_clean[:,icity] = Tfilter1

    fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(6,10),sharex=True)
    plt.title(city)

    ax[0].plot(Twater, color=plt.get_cmap('tab20')(icity*2),label=str(city))
    ax[0].plot(Tfilter1,'.-', color=plt.get_cmap('tab20')(icity*2+1),markersize=2)
    ax[1].plot(Twater, color=plt.get_cmap('tab20')(icity*2),label=str(city))
    ax[1].plot(Tfilter2,'.-', color=plt.get_cmap('tab20')(icity*2+1),markersize=2)
    ax[2].plot(Ta[:,1], color=plt.get_cmap('tab20')(icity*2+1))
    ax[2].plot(running_nanmean(Ta[:,1],N=7), color=plt.get_cmap('tab20')(icity*2))
    # Tfiltera,filter_maska = hampel(Ta[:,1],k=15,t0=2., exclude_crrt_point=False,w=Ta[:,1],corr_lim=1,nan_substitution=True)
    # ax[2].plot(filter_maska,'.-', color=plt.get_cmap('tab20')(icity*2),markersize=2)


    mask_1[icity] += np.sum(filter_mask1)
    mask_2[icity] += np.sum(filter_mask2)
    mask_tot[icity] += np.sum(filter_mask1)+np.sum(filter_mask2)
    N[icity] += np.sum(~np.isnan(Twater))


# percentage of all data points that are filtered after the first filter
print(mask_1/N)
# percentage of all data points that are filtered after the second filter
print(mask_2/N)
# percentage of all data points that are filtered after two passes of the filter
print(mask_tot/N)


if save:
    for icity in range(len(water_cities_name_list)):
        Twater_city = np.zeros((Twater_clean.shape[0],2))*np.nan
        Twater_city[:,0] = time
        Twater_city[:,1] = Twater_clean[:,icity]
        save_name = water_cities_name_list[icity][:-9]
        np.savez('../../data/Twater_cities/Twater_cities_'+save_name+'_clean',
               Twater=Twater,date_ref=date_ref)


#%%
# plt.figure()
# plt.plot(Twater_clean[:,0],Twater_clean[:,1],'.')

# plt.figure()
# plt.plot(Twater_clean[:10000,0],Twater_clean[:10000,1],'.')

# plt.figure()
# plt.plot(Twater_clean[10000:,0],Twater_clean[10000:,1],'.')

#%%
# FILTER ECCC DATA SETS

mask_eccc1 = np.zeros(len(water_eccc_name_list))
mask_eccc2 = np.zeros(len(water_eccc_name_list))
mask_eccctot = np.zeros(len(water_eccc_name_list))
Neccc = np.zeros(len(water_eccc_name_list))

Twater_clean_eccc = np.zeros((Ta.shape[0],len(water_eccc_name_list)))*np.nan

for icity,city in enumerate(water_eccc_name_list):
    loc_water_eccc = water_eccc_name_list[icity]
    water_eccc_data = np.load(fp+'Twater_ECCC/Twater_ECCC_'+loc_water_eccc+'.npz',allow_pickle='TRUE')
    Twater_eccc = water_eccc_data['Twater'][:,1]

    Tfilter1_eccc,filter_mask1_eccc = hampel(Twater_eccc,k=15,t0=2., exclude_crrt_point=False,w=Ta[:,1],corr_lim=1,nan_substitution=True)
    Tfilter2_eccc,filter_mask2_eccc = hampel(Twater_eccc,k=15,t0=2., exclude_crrt_point=False,w=Ta[:,1],corr_lim=0.35,nan_substitution=True)

    # Tfilter1_eccc[Tfilter1_eccc < 0] = 0
    # Tfilter2_eccc[Tfilter2_eccc < 0] = 0

    Twater_clean_eccc[:,icity] = Tfilter1_eccc

    fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(6,10),sharex=True)
    plt.title(city)

    ax[0].plot(Twater_eccc, color=plt.get_cmap('tab20')(icity*2),label=str(city))
    ax[0].plot(Tfilter1_eccc,'.-', color=plt.get_cmap('tab20')(icity*2+1),markersize=2)
    ax[1].plot(Twater_eccc, color=plt.get_cmap('tab20')(icity*2),label=str(city))
    ax[1].plot(Tfilter2_eccc,'.-', color=plt.get_cmap('tab20')(icity*2+1),markersize=2)
    ax[2].plot(Ta[:,1], color=plt.get_cmap('tab20')(icity*2+1))
    ax[2].plot(running_nanmean(Ta[:,1],N=7), color=plt.get_cmap('tab20')(icity*2))
    # Tfiltera,filter_maska = hampel(Ta[:,1],k=15,t0=2., exclude_crrt_point=False,w=Ta[:,1],corr_lim=1,nan_substitution=True)
    # ax[2].plot(filter_maska,'.-', color=plt.get_cmap('tab20')(icity*2),markersize=2)


    mask_eccc1[icity] += np.sum(filter_mask1_eccc)
    mask_eccc2[icity] += np.sum(filter_mask2_eccc)
    mask_eccctot[icity] += np.sum(filter_mask1_eccc)+np.sum(filter_mask2_eccc)
    Neccc[icity] += np.sum(~np.isnan(Twater_eccc))

# percentage of all data points that are filtered after the first filter
print(mask_eccc1/Neccc)
# percentage of all data points that are filtered after the second filter
print(mask_eccc2/Neccc)
# percentage of all data points that are filtered after two passes of the filter
print(mask_eccctot/Neccc)


if save:
    for icity in range(len(water_eccc_name_list)):
        Twater_city_eccc = np.zeros((Twater_clean_eccc.shape[0],2))*np.nan
        Twater_city_eccc[:,0] = time
        Twater_city_eccc[:,1] = Twater_clean_eccc[:,icity]
        save_name = water_eccc_name_list[icity]
        np.savez('../../data/Twater_ECCC/Twater_ECCC_'+save_name+'_clean',
               Twater=Twater_eccc,date_ref=date_ref)


#%%
if plot_years:
    for iyear,year in enumerate(years):
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
        plt.title(years[iyear])

        date=(dt.date(year,1,1)-date_ref).days
        i0 = np.where(time==date)[0][0]
        i1 = i0+365+calendar.isleap(year)

        for icity,city in enumerate(water_cities_name_list):
            loc_water_city = water_cities_name_list[icity]
            water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
            Twater = water_city_data['Twater'][:,1]

            ax.plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            ax.plot(Twater_clean[i0:i1,icity], color=plt.get_cmap('tab20')(icity*2+1))

        for icity,city in enumerate(water_eccc_name_list):
            loc_water_eccc = water_eccc_name_list[icity]
            water_eccc_data = np.load(fp+'Twater_ECCC/Twater_ECCC_'+loc_water_eccc+'.npz',allow_pickle='TRUE')
            Twater_eccc = water_eccc_data['Twater'][:,1]

            ax.plot(Twater_eccc[i0:i1], '-', color=plt.get_cmap('tab20')((icity+len(water_cities_name_list))*2),label=water_eccc_name_list[icity])
            ax.plot(Twater_clean_eccc[i0:i1,icity], '-', color=plt.get_cmap('tab20')((icity+len(water_cities_name_list))*2+1),label=water_eccc_name_list[icity])


        ax.legend()

#%%

plt.figure()
plt.hist(Twater_clean[:,0],bins=30,range=(-2,30),label='Longueuil',density=True,alpha=0.3)
plt.hist(Twater_clean[:,1],bins=30,range=(-2,30),label='Atwater',density=True,alpha=0.3)
plt.hist(Twater_clean[:,2],bins=30,range=(-2,30),label='DesBaillets',density=True,alpha=0.3)
plt.hist(Twater_clean[:,3],bins=30,range=(-2,30),label='Candiac',density=True,alpha=0.3)

print("Longueuil: ",np.nanmean(Twater_clean[:,0]),np.nanstd(Twater_clean[:,0]),np.nanpercentile(Twater_clean[:,0],50))
print("Atwater: ",np.nanmean(Twater_clean[:,1]),np.nanstd(Twater_clean[:,1]),np.nanpercentile(Twater_clean[:,1],50))
print("DesBaillets: ",np.nanmean(Twater_clean[:,2]),np.nanstd(Twater_clean[:,2]),np.nanpercentile(Twater_clean[:,2],50))
print("Candiac: ",np.nanmean(Twater_clean[:,3]),np.nanstd(Twater_clean[:,3]),np.nanpercentile(Twater_clean[:,3],50))

plt.hist(Twater_clean_eccc[:,0],bins=30,range=(-2,30),label='Lasalle',density=True,alpha=0.3)
plt.hist(Twater_clean_eccc[:,1],bins=30,range=(-2,30),label='Laprairie',density=True,alpha=0.3)

print("Lasalle: ",np.nanmean(Twater_clean_eccc[:,0]),np.nanstd(Twater_clean_eccc[:,0]),np.nanpercentile(Twater_clean_eccc[:,0],50))
print("LaPrairie: ",np.nanmean(Twater_clean_eccc[:,1]),np.nanstd(Twater_clean_eccc[:,1]),np.nanpercentile(Twater_clean_eccc[:,1],50))

#%%

Twater_all = np.zeros((Twater_clean.shape[0]*len(water_cities_name_list),4))*np.nan

# 0: Twater
# 1: plant
# 2: year
# 3: season

for icity in range(len(water_cities_name_list)):
    year_tmp = np.zeros((Twater_clean.shape[0]))*np.nan
    season_tmp = np.zeros((Twater_clean.shape[0]))*np.nan

    for it in range(Twater_clean.shape[0]):
        date_it = date_ref+dt.timedelta(days=int(time[it]))
        year_tmp[it] = int(date_it.year)

        if (((date_it - dt.date(int(date_it.year),3,21)).days > 0) &
           ((date_it - dt.date(int(date_it.year),6,21)).days <= 0) ):
               season_tmp[it] = 0 # Spring

        if (((date_it - dt.date(int(date_it.year),6,21)).days > 0) &
           ((date_it - dt.date(int(date_it.year),9,21)).days <= 0) ):
               season_tmp[it] = 1 # Summer

        if (((date_it - dt.date(int(date_it.year),9,21)).days > 0) &
           ((date_it - dt.date(int(date_it.year),12,21)).days <= 0) ):
               season_tmp[it] = 2 # Fall

        if (((date_it - dt.date(int(date_it.year),12,21)).days > 0)):
             season_tmp[it] = 3 # Winter

        if (((date_it - dt.date(int(date_it.year),3,21)).days <= 0)):
             season_tmp[it] = 3 # Winter

    Twater_tmp = Twater_clean[:,icity].copy()
    # Remove data prior to end of 2010 to compare
    # only the same period (i.e. 2010-2020)
    # Twater_tmp[:11220] = np.nan

    # Remove data prior to end of 2004 to compare
    # only the same period (i.e. 2004-2020)
    # Twater_tmp[:8800] = np.nan


    # Twater_tmp[:12500] = np.nan

    # Twater_tmp[13600:] = np.nan

    # Twater_tmp[:12670] = np.nan
    # Twater_tmp[14676:] = np.nan

    Twater_all[icity*Twater_clean.shape[0]:Twater_clean.shape[0]*(icity+1),0] = Twater_tmp
    Twater_all[icity*Twater_clean.shape[0]:Twater_clean.shape[0]*(icity+1),1] = icity
    Twater_all[icity*Twater_clean.shape[0]:Twater_clean.shape[0]*(icity+1),2] = year_tmp
    Twater_all[icity*Twater_clean.shape[0]:Twater_clean.shape[0]*(icity+1),3] = season_tmp

# Twater_all = Twater_all[~np.isnan(Twater_all[:,0])]

if save:
    np.savez('../../data/Twater_cities/Twater_cities_all_clean',
           Twater_all=Twater_all,date_ref=date_ref)



#%%

Twater_all_eccc = np.zeros((Twater_clean_eccc.shape[0]*len(water_eccc_name_list),4))*np.nan

# 0: Twater
# 1: plant
# 2: year
# 3: season

for icity in range(len(water_eccc_name_list)):
    year_tmp = np.zeros((Twater_clean_eccc.shape[0]))*np.nan
    season_tmp = np.zeros((Twater_clean_eccc.shape[0]))*np.nan

    for it in range(Twater_clean_eccc.shape[0]):
        date_it = date_ref+dt.timedelta(days=int(time[it]))
        year_tmp[it] = int(date_it.year)

        if (((date_it - dt.date(int(date_it.year),3,21)).days > 0) &
           ((date_it - dt.date(int(date_it.year),6,21)).days <= 0) ):
               season_tmp[it] = 0 # Spring

        if (((date_it - dt.date(int(date_it.year),6,21)).days > 0) &
           ((date_it - dt.date(int(date_it.year),9,21)).days <= 0) ):
               season_tmp[it] = 1 # Summer

        if (((date_it - dt.date(int(date_it.year),9,21)).days > 0) &
           ((date_it - dt.date(int(date_it.year),12,21)).days <= 0) ):
               season_tmp[it] = 2 # Fall

        if (((date_it - dt.date(int(date_it.year),12,21)).days > 0)):
             season_tmp[it] = 3 # Winter

        if (((date_it - dt.date(int(date_it.year),3,21)).days <= 0)):
             season_tmp[it] = 3 # Winter

    Twater_tmp_eccc = Twater_clean_eccc[:,icity].copy()
    # Remove data prior to end of 2010 to compare
    # only the same period (i.e. 2010-2020)
    # Twater_tmp_eccc[:11220] = np.nan

    # Remove data prior to end of 2004 to compare
    # only the same period (i.e. 2004-2020)
    # Twater_tmp_eccc[:8800] = np.nan


    # Twater_tmp_eccc[:12670] = np.nan
    # Twater_tmp_eccc[14676:] = np.nan
    # Twater_tmp_eccc[:11220] = np.nan
    # Twater_tmp_eccc[:12500] = np.nan
    # Twater_tmp_eccc[13600:] = np.nan


    Twater_all_eccc[icity*Twater_clean_eccc.shape[0]:Twater_clean_eccc.shape[0]*(icity+1),0] = Twater_tmp_eccc
    Twater_all_eccc[icity*Twater_clean_eccc.shape[0]:Twater_clean_eccc.shape[0]*(icity+1),1] = icity
    Twater_all_eccc[icity*Twater_clean_eccc.shape[0]:Twater_clean_eccc.shape[0]*(icity+1),2] = year_tmp
    Twater_all_eccc[icity*Twater_clean_eccc.shape[0]:Twater_clean_eccc.shape[0]*(icity+1),3] = season_tmp

# Twater_all_eccc = Twater_all_eccc[~np.isnan(Twater_all_eccc[:,0])]

if save:
    np.savez('../../data/Twater_ECCC/Twater_ECCC_all_clean',
           Twater_all=Twater_all_eccc,date_ref=date_ref)



