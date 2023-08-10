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

#%%

save = False
plot_years = True
plot_dTdt = False

years = [1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005]
# years = [2006,
#           2007,2008,2009,2010,2011,
#           2012,2013,2014,2015,2016,2017,
#           2018,2019,2020]
# years = [1992,1993,1994,1995,1996,
#           1997,1998,1999,2000,2001,
#           2002,2003,2004,2005,2006,
#           2007,2008,2009,2010,2011,
#           2012,2013,2014,2015,2016,2017,
#           2018,2019,2020]
# water_cities_name_list = ['Longueuil_preclean','Atwater_preclean','DesBaillets_preclean','Candiac_preclean']
# water_eccc_name_list = ['Lasalle', 'LaPrairie']
water_cities_name_list = ['Candiac','Longueuil','Atwater','DesBaillets']
# water_cities_name_list = ['Candiac_preclean','Longueuil_preclean','Atwater_preclean','DesBaillets_preclean']
# water_cities_name_list = ['Atwater']

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
# MAKE DTair/Dt TIME SERIES

Ta_dTdt = np.zeros((len(time),1))*np.nan
dTdt_tmp = np.zeros((Ta.shape[0],2))*np.nan
dTdt_tmp[1:,0] = Ta[1:,1] - Ta[0:-1,1] # Backwards
dTdt_tmp[0:-1,1]= Ta[0:-1,1] - Ta[1:,1] # Forwards
Ta_dTdt = np.nanmean(dTdt_tmp,axis=1)


#%%
# FILTER BASED ON DT/DT +/- t0*STD FOR THE WHOLE RECORD

# Comput dTwater/Dt
Twater_dTdt = np.zeros((len(time),len(water_cities_name_list)))*np.nan
Twater_dTdt_f = np.zeros((len(time),len(water_cities_name_list)))*np.nan
Twater_dTdt_b = np.zeros((len(time),len(water_cities_name_list)))*np.nan

for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][:,1]
    dTdt_tmp = np.zeros((Twater.shape[0],2))*np.nan

    dTdt_tmp[1:,0] = Twater[1:] - Twater[0:-1] # Backwards
    dTdt_tmp[0:-1,1]= Twater[1:]- Twater[0:-1] # Forwards

    Twater_dTdt[:,icity] = np.nanmean(dTdt_tmp,axis=1)
    Twater_dTdt_f[:,icity] = dTdt_tmp[:,1]
    Twater_dTdt_b[:,icity] = dTdt_tmp[:,0]


# Filter records +/- t0*std away from the mean dTwater/dt
Twater_clean_f = np.zeros((len(time),len(water_cities_name_list)))*np.nan
Twater_clean_b = np.zeros((len(time),len(water_cities_name_list)))*np.nan
Twater_clean = np.zeros((len(time),len(water_cities_name_list)))*np.nan
Twater_clean_fb = np.zeros((len(time),len(water_cities_name_list)))*np.nan
Twater_clean_fnb = np.zeros((len(time),len(water_cities_name_list)))*np.nan

t0 = 4.25
for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][:,1]

    Tw_dTdt = Twater_dTdt[:,icity]
    # filter_mask_std = (Tw_dTdt > np.nanmean(Tw_dTdt)+t0*np.nanstd(Tw_dTdt)) | (Tw_dTdt < np.nanmean(Tw_dTdt)-t0*np.nanstd(Tw_dTdt))
    filter_mask_std = ( np.abs(np.abs(Tw_dTdt) - np.nanmean(np.abs(Tw_dTdt)))> t0*np.nanstd(np.abs(Tw_dTdt)))
    Tfilter1 = Twater.copy()
    Tfilter1[filter_mask_std] = np.nan

    Tw_dTdt_f = Twater_dTdt_f[:,icity]
    # filter_mask_std_f = (Tw_dTdt_f > np.nanmean(Tw_dTdt_f)+t0*np.nanstd(Tw_dTdt_f)) | (Tw_dTdt_f < np.nanmean(Tw_dTdt_f)-t0*np.nanstd(Tw_dTdt_f))
    # #filter_mask_std_f = (np.abs(Tw_dTdt_f) > np.nanmean(np.abs(Tw_dTdt_f))+t0*np.nanstd(np.abs(Tw_dTdt_f)))
    filter_mask_std_f = ( np.abs(np.abs(Tw_dTdt_f) - np.nanmean(np.abs(Tw_dTdt_f)))> t0*np.nanstd(np.abs(Tw_dTdt_f)))
    Tfilter_f = Twater.copy()
    Tfilter_f[filter_mask_std_f] = np.nan

    Tw_dTdt_b = Twater_dTdt_b[:,icity]
    # filter_mask_std_b = (Tw_dTdt_b > np.nanmean(Tw_dTdt_b)+t0*np.nanstd(Tw_dTdt_b)) | (Tw_dTdt_b < np.nanmean(Tw_dTdt_b)-t0*np.nanstd(Tw_dTdt_b))
    # #filter_mask_std_b = (np.abs(Tw_dTdt_b) > np.nanmean(np.abs(Tw_dTdt_b))+t0*np.nanstd(np.abs(Tw_dTdt_b)))
    filter_mask_std_b = ( np.abs(np.abs(Tw_dTdt_b) - np.nanmean(np.abs(Tw_dTdt_b))) > t0*np.nanstd(np.abs(Tw_dTdt_b)))
    Tfilter_b = Twater.copy()
    Tfilter_b[filter_mask_std_b] = np.nan

    filter_mask_std_fb = filter_mask_std_f | filter_mask_std_b #|filter_mask_std
    Tfilter_fb = Twater.copy()
    Tfilter_fb[filter_mask_std_fb] = np.nan


    Twater_clean[:,icity] = Tfilter1
    Twater_clean_f[:,icity] = Tfilter_f
    Twater_clean_b[:,icity] = Tfilter_b
    Twater_clean_fb[:,icity] = Tfilter_fb

    filter_mask_std_Ta = (Ta_dTdt > np.nanmean(Ta_dTdt)+t0*np.nanstd(Ta_dTdt)) | (Ta_dTdt < np.nanmean(Ta_dTdt)-t0*np.nanstd(Ta_dTdt))



t0 = 3.5
for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][:,1]

    Tw_dTdt_f = Twater_dTdt_f[:,icity]
    # filter_mask_std_f = (Tw_dTdt_f > np.nanmean(Tw_dTdt_f)+t0*np.nanstd(Tw_dTdt_f)) | (Tw_dTdt_f < np.nanmean(Tw_dTdt_f)-t0*np.nanstd(Tw_dTdt_f))
    # #filter_mask_std_f = (np.abs(Tw_dTdt_f) > np.nanmean(np.abs(Tw_dTdt_f))+t0*np.nanstd(np.abs(Tw_dTdt_f)))
    filter_mask_std_f = ( np.abs(np.abs(Tw_dTdt_f) - np.nanmean(np.abs(Tw_dTdt_f)))> t0*np.nanstd(np.abs(Tw_dTdt_f)))

    Tw_dTdt_b = Twater_dTdt_b[:,icity]
    # filter_mask_std_b = (Tw_dTdt_b > np.nanmean(Tw_dTdt_b)+t0*np.nanstd(Tw_dTdt_b)) | (Tw_dTdt_b < np.nanmean(Tw_dTdt_b)-t0*np.nanstd(Tw_dTdt_b))
    # #filter_mask_std_b = (np.abs(Tw_dTdt_b) > np.nanmean(np.abs(Tw_dTdt_b))+t0*np.nanstd(np.abs(Tw_dTdt_b)))
    filter_mask_std_b = ( np.abs(np.abs(Tw_dTdt_b) - np.nanmean(np.abs(Tw_dTdt_b))) > t0*np.nanstd(np.abs(Tw_dTdt_b)))

    filter_mask_std_fnb = filter_mask_std_f & filter_mask_std_b #|filter_mask_std
    Tfilter_fnb = Twater.copy()
    Tfilter_fnb[filter_mask_std_fnb] = np.nan

    Twater_clean_fnb[:,icity] = Tfilter_fnb


#%%
# FILTER BASED ON DT/DT +/- t0*STD FOR THE WHOLE RECORD
# ** THIS IS A WRONG VERSION THAT I DID FIRST BY MISTAKE,
# BUT IN FACT SHOWS REALLY GOOD RESULTS...

# Comput dTwater/Dt
Twater_dTdt_wrong = np.zeros((len(time),len(water_cities_name_list)))*np.nan

for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][:,1]
    dTdt_tmp = np.zeros((Twater.shape[0],2))*np.nan

    dTdt_tmp[1:,0] = Twater[1:] - Twater[0:-1] # Backwards
    dTdt_tmp[0:-1,1]= Twater[0:-1] - Twater[1:] # Forwards

    Twater_dTdt_wrong[:,icity] = np.nanmean(dTdt_tmp,axis=1)


# Filter records +/- t0*std away from the mean dTwater/dt
Twater_clean_wrong = np.zeros((len(time),len(water_cities_name_list)))*np.nan
Twater_clean_wrong2 = np.zeros((len(time),len(water_cities_name_list)))*np.nan

t0 = 3.0
for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][:,1]

    Tw_dTdt_wrong = Twater_dTdt_wrong[:,icity].copy()
    filter_mask_std_wrong = (Tw_dTdt_wrong > np.nanmean(Tw_dTdt_wrong)+t0*np.nanstd(Tw_dTdt_wrong)) | (Tw_dTdt_wrong < np.nanmean(Tw_dTdt_wrong)-t0*np.nanstd(Tw_dTdt_wrong))
    Tfilter_wrong = Twater.copy()
    Tfilter_wrong[filter_mask_std_wrong] = np.nan
    Twater_clean_wrong[:,icity] = Tfilter_wrong

    Tw_dTdt_wrong2 = -1*Twater_dTdt_wrong[:,icity].copy()
    filter_mask_std_wrong2 = np.abs(Tw_dTdt_wrong2 -np.nanmean(Tw_dTdt_wrong2))  > t0*np.nanstd(Tw_dTdt_wrong2)
    Tfilter_wrong2 = Twater.copy()
    Tfilter_wrong2[filter_mask_std_wrong2] = np.nan
    Twater_clean_wrong2[:,icity] = Tfilter_wrong2



#%%
from scipy import ndimage

Twater_clean_LoG = np.zeros((len(time),len(water_cities_name_list)))*np.nan

t0 = 3.
for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][:,1]
    Tw_dTdt_wrong = -1*Twater_dTdt_wrong[:,icity]
    Tw_dTdt = Twater_dTdt[:,icity]

    Tplot = Twater
    LoG = ndimage.gaussian_laplace(Tplot, sigma=0.95)


    # thres_mean = np.nanmean(np.absolute(LoG))
    # thres_std = np.nanstd(np.absolute(LoG))*t0
    thres_mean_abs = np.nanmean(np.absolute(Tw_dTdt_wrong))
    thres_std_abs = np.nanstd(np.absolute(Tw_dTdt_wrong))*t0
    thres_abs = thres_std_abs
    # thres = 1.

    thres_mean = np.nanmean((Tw_dTdt_wrong))
    thres_std = np.nanstd((Tw_dTdt_wrong))
    thres = thres_std



    output_LoG = np.zeros(Tplot.shape)
    output_LoG2 = np.zeros(Tplot.shape)
    output_LoG3 = np.zeros(Tplot.shape)
    output_LoG4 = np.zeros(Tplot.shape)
    cross = np.zeros(Tplot.shape)

    for y in range(1, output_LoG.shape[0] - 1):
        # patch = LoG[y-1:y+2]
        # p = LoG[y]
        patch = Tw_dTdt_wrong[y-1:y+2]
        p = Tw_dTdt_wrong[y]
        maxP = patch.max()
        minP = patch.min()
        if (p > 0):
            zeroCross = True if minP < 0 else False
        else:
            zeroCross = True if maxP > 0 else False

        if np.abs(p) < 1e-4:
            cross[y] = p
        else:
            cross[y] = np.nan

        if (np.abs(p-thres_mean) > t0*thres_std) and zeroCross:
            output_LoG4[y] = 1
            #Modify so that we only fiter if the point in the middle is the min or max.
            # This will treatt the crossing only once

        dpatch = Tw_dTdt[y-1:y+2]
        if ( (np.abs(dpatch[1]-np.nanmean(Tw_dTdt)) > 4*np.nanstd(Tw_dTdt)) & zeroCross):
            output_LoG2[y] = 1

        if np.abs(p-thres_mean) > t0*thres_std:
            output_LoG3[y] = 1 # This one is the same as Twater_dTdt_wrong


        # if ((maxP - minP) > thres_abs) and zeroCross:
        if ((maxP - minP) > 2) and zeroCross:
            output_LoG[y] = 1
            #Modify so that we only fiter if the point in the middle is the min or max.
            # This will treatt the crossing only once


    Tfilter_LoG = Twater.copy()
    Tfilter_LoG[output_LoG == 1] = np.nan
    # Tfilter_LoG[output_LoG3 == 1] = np.nan

    fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(6,10),sharex=True)
    ax[0].plot(Twater, color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
    ax[0].plot(Tfilter_LoG, '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))
    ax[1].plot((Tw_dTdt_wrong))
    ax[1].plot((Tw_dTdt))
    # ax[1].plot(cross, '*')
    ax[1].plot(np.zeros(Tplot.shape),'-', color='gray',linewidth=0.5)

    ax[2].plot(output_LoG3,'x')
    ax[2].plot(output_LoG4,'o')
    ax[2].plot(output_LoG,'*')
    Twater_clean_LoG[:,icity] = Tfilter_LoG

#%%
# TRY HAMPEL FILTER INSTEAD TO FILTER RECORDS WHEN dTwater/dt
# IN A GIVEN WINDOW SIZE IS LARGER THAN T0*STD IN THAT SAME WINDOW

Twater_clean_hampel_dTdt = np.zeros((len(time),len(water_cities_name_list)))*np.nan
Twater_clean_hampel_dTdt_f = np.zeros((len(time),len(water_cities_name_list)))*np.nan
Twater_clean_hampel_dTdt_b = np.zeros((len(time),len(water_cities_name_list)))*np.nan
Twater_clean_hampel_dTdt_forb = np.zeros((len(time),len(water_cities_name_list)))*np.nan

Twater_clean_hampel = np.zeros((len(time),len(water_cities_name_list)))*np.nan


for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][:,1]

    T_filter,filter_mask= hampel(Twater,k=45,t0=5,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)
    # T_filter,filter_mask= hampel(Twater,k=15,t0=2.,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)

    Tfilter_hampel = Twater.copy()
    Tfilter_hampel[filter_mask] = np.nan
    Twater_clean_hampel[:,icity] = Tfilter_hampel


    dTdt_tmp,filter_mask_dTdt_f= hampel(np.abs(Twater_dTdt_f[:,icity]),k=45,t0=6,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)
    dTdt_tmp,filter_mask_dTdt_b= hampel(np.abs(Twater_dTdt_b[:,icity]),k=45,t0=6,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)
    dTdt_tmp,filter_mask_dTdt= hampel(np.abs(Twater_dTdt[:,icity]),k=45,t0=6,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)

    # dTdt_tmp,filter_mask_dTdt_f= hampel(np.abs(Twater_dTdt_f[:,icity]),k=15,t0=2,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)
    # dTdt_tmp,filter_mask_dTdt_b= hampel(np.abs(Twater_dTdt_b[:,icity]),k=15,t0=2,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)
    # dTdt_tmp,filter_mask_dTdt= hampel(np.abs(Twater_dTdt[:,icity]),k=15,t0=2,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)

    Tfilter_hampel_dTdt = Twater.copy()
    Tfilter_hampel_dTdt[filter_mask_dTdt] = np.nan
    Twater_clean_hampel_dTdt[:,icity] = Tfilter_hampel_dTdt

    Tfilter_hampel_dTdt_f = Twater.copy()
    Tfilter_hampel_dTdt_f[filter_mask_dTdt_f] = np.nan
    Twater_clean_hampel_dTdt_f[:,icity] = Tfilter_hampel_dTdt_f

    Tfilter_hampel_dTdt_b = Twater.copy()
    Tfilter_hampel_dTdt_b[filter_mask_dTdt_b] = np.nan
    Twater_clean_hampel_dTdt_b[:,icity] = Tfilter_hampel_dTdt_b

    Tfilter_hampel_dTdt_forb = Twater.copy()
    filter_mask_dTdt_forb = filter_mask_dTdt_f | filter_mask_dTdt_b
    Tfilter_hampel_dTdt_forb[filter_mask_dTdt_forb] = np.nan
    Twater_clean_hampel_dTdt_forb[:,icity] = Tfilter_hampel_dTdt_forb


#%%
# FOR THIS FILTER, WE COMPUTE A ROLLING CLIMATOLOGY, AND
# COMPARE THE DEVIATION FROM THE CLIMATOLOGICAL MEAN IN
# A TIME WINDOW AROUND THE RECORD TO THE CLIMATOLOGICAL
# STANDARD DEVIATIONIN THAT WINDOW, I.E. +/- T0*STD_CLIM


# First re-arrange data to have each 31-day window, for each date, each year, each city
Nwindow = 31 # Only odd window size are possible
# Nwindow = 91 # Only odd window size are possible
# Nwindow = 101 # Only odd window size are possible
# Nwindow = 365 # Only odd window size are possible
# Nwindow = 7 # Only odd window size are possible
data_Tw = np.zeros((Nwindow,366,len(years),len(water_cities_name_list)))*np.nan
data_dTwdt = np.zeros((Nwindow,366,len(years),len(water_cities_name_list)))*np.nan
data_dTwdt_f = np.zeros((Nwindow,366,len(years),len(water_cities_name_list)))*np.nan
data_dTwdt_b = np.zeros((Nwindow,366,len(years),len(water_cities_name_list)))*np.nan

years = np.array(years)

for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][:,1]

    # Twater_dTdt_tmp = Twater_dTdt[:,icity]
    # Twater_dTdt_f_tmp = Twater_dTdt_f[:,icity]
    # Twater_dTdt_b_tmp = Twater_dTdt_b[:,icity]
    Twater_dTdt_tmp = np.abs(Twater_dTdt[:,icity])
    Twater_dTdt_f_tmp = np.abs(Twater_dTdt_f[:,icity])
    Twater_dTdt_b_tmp = np.abs(Twater_dTdt_b[:,icity])

    for it in range(Twater.shape[0]):

        iw0 = np.max([0,it-int((Nwindow-1)/2)])
        iw1 = it+int((Nwindow-1)/2)+1

        Twater_window = np.zeros(Nwindow)*np.nan
        Twater_window[0:len(Twater[iw0:iw1])] = Twater[iw0:iw1]

        dTdt_window = np.zeros(Nwindow)*np.nan
        dTdt_window[0:len(Twater_dTdt_tmp[iw0:iw1])] = Twater_dTdt_tmp[iw0:iw1]

        dTdt_window_f = np.zeros(Nwindow)*np.nan
        dTdt_window_f[0:len(Twater_dTdt_f_tmp[iw0:iw1])] = Twater_dTdt_f_tmp[iw0:iw1]

        dTdt_window_b = np.zeros(Nwindow)*np.nan
        dTdt_window_b[0:len(Twater_dTdt_b_tmp[iw0:iw1])] = Twater_dTdt_b_tmp[iw0:iw1]

        date_mid = date_ref+dt.timedelta(days=int(time[it]))
        year_mid = date_mid.year
        month_mid = date_mid.month
        day_mid = date_mid.day

        if len(np.where(years == year_mid)[0]) > 0:
            iyear = np.where(years == year_mid)[0][0]
            doy = (dt.date(year_mid,month_mid,day_mid)-dt.date(year_mid,1,1)).days

            data_Tw[:,doy,iyear,icity] = Twater_window
            data_dTwdt[:,doy,iyear,icity] = dTdt_window
            data_dTwdt_f[:,doy,iyear,icity] = dTdt_window_f
            data_dTwdt_b[:,doy,iyear,icity] = dTdt_window_b

            if not calendar.isleap(year_mid) and (doy == 364):
                imid = int((Nwindow-1)/2)
                Twater_window_366 = np.zeros((Nwindow))*np.nan
                Twater_window_366[imid] = np.array(np.nanmean([Twater[it],Twater[it+1]]))
                Twater_window_366[0:imid] = Twater[int(it+1-((Nwindow-1)/2)):it+1]
                Twater_window_366[imid+1:Nwindow] = Twater[it+1:int(it+1+((Nwindow-1)/2))]
                data_Tw[:,365,iyear,icity] = Twater_window_366

                dTdt_window_366 = np.zeros((Nwindow))*np.nan
                dTdt_window_366[imid] = np.array(np.nanmean([Twater_dTdt_tmp[it],Twater_dTdt_tmp[it+1]]))
                dTdt_window_366[0:imid] = Twater_dTdt_tmp[int(it+1-((Nwindow-1)/2)):it+1]
                dTdt_window_366[imid+1:Nwindow] = Twater_dTdt_tmp[it+1:int(it+1+((Nwindow-1)/2))]
                data_dTwdt[:,365,iyear,icity] = dTdt_window_366

                dTdt_window_f_366 = np.zeros((Nwindow))*np.nan
                dTdt_window_f_366[imid] = np.array(np.nanmean([Twater_dTdt_f_tmp[it],Twater_dTdt_f_tmp[it+1]]))
                dTdt_window_f_366[0:imid] = Twater_dTdt_f_tmp[int(it+1-((Nwindow-1)/2)):it+1]
                dTdt_window_f_366[imid+1:Nwindow] = Twater_dTdt_f_tmp[it+1:int(it+1+((Nwindow-1)/2))]
                data_dTwdt_f[:,365,iyear,icity] = dTdt_window_f_366

                dTdt_window_b_366 = np.zeros((Nwindow))*np.nan
                dTdt_window_b_366[imid] = np.array(np.nanmean([Twater_dTdt_b_tmp[it],Twater_dTdt_b_tmp[it+1]]))
                dTdt_window_b_366[0:imid] = Twater_dTdt_b_tmp[int(it+1-((Nwindow-1)/2)):it+1]
                dTdt_window_b_366[imid+1:Nwindow] = Twater_dTdt_b_tmp[it+1:int(it+1+((Nwindow-1)/2))]
                data_dTwdt_b[:,365,iyear,icity] = dTdt_window_b_366





# Then, find the 31-day climatological mean and std for each date
mean_clim_Tw = np.zeros((366,len(water_cities_name_list)))*np.nan
std_clim_Tw = np.zeros((366,len(water_cities_name_list)))*np.nan

mean_clim_dTwdt = np.zeros((366,len(water_cities_name_list)))*np.nan
std_clim_dTwdt = np.zeros((366,len(water_cities_name_list)))*np.nan

mean_clim_dTwdt_f = np.zeros((366,len(water_cities_name_list)))*np.nan
std_clim_dTwdt_f = np.zeros((366,len(water_cities_name_list)))*np.nan

mean_clim_dTwdt_b = np.zeros((366,len(water_cities_name_list)))*np.nan
std_clim_dTwdt_b = np.zeros((366,len(water_cities_name_list)))*np.nan

n_mean = np.zeros((366,len(water_cities_name_list)))*np.nan

for icity,city in enumerate(water_cities_name_list):

    data = data_Tw[:,:,:,icity]
    mean_clim_Tw[:,icity] = np.nanmean(data,axis=(0,2))
    std_clim_Tw[:,icity] = np.nanstd(data,axis=(0,2))

    data = data_dTwdt[:,:,:,icity]
    mean_clim_dTwdt[:,icity] = np.nanmean(data,axis=(0,2))
    std_clim_dTwdt[:,icity] = np.nanstd(data,axis=(0,2))

    data = data_dTwdt_f[:,:,:,icity]
    n_mean[:,icity]=np.sum(~np.isnan(data),axis=(0,2))
    mean_clim_dTwdt_f[:,icity] = np.nanmean(data,axis=(0,2))
    std_clim_dTwdt_f[:,icity] = np.nanstd(data,axis=(0,2))


    data = data_dTwdt_b[:,:,:,icity]
    mean_clim_dTwdt_b[:,icity] = np.nanmean(data,axis=(0,2))
    std_clim_dTwdt_b[:,icity] = np.nanstd(data,axis=(0,2))



Twater_clean_climo = np.zeros((366,len(years),len(water_cities_name_list)))*np.nan
Twater_clean_climo_dTwdt = np.zeros((366,len(years),len(water_cities_name_list)))*np.nan
Twater_clean_climo_dTwdt_f = np.zeros((366,len(years),len(water_cities_name_list)))*np.nan
Twater_clean_climo_dTwdt_b = np.zeros((366,len(years),len(water_cities_name_list)))*np.nan
Twater_clean_climo_dTwdt_forb = np.zeros((366,len(years),len(water_cities_name_list)))*np.nan


t0  = 5.
for iyear,year in enumerate(years):
    for icity,city in enumerate(water_cities_name_list):

        imid = int((Nwindow-1)/2)

        Twater = data_Tw[imid,:,iyear,icity]
        clim_Tw = mean_clim_Tw[:,icity]
        std_Tw = std_clim_Tw[:,icity]

        dTwater = data_dTwdt[imid,:,iyear,icity]
        clim_dTwdt = mean_clim_dTwdt[:,icity]
        std_dTwdt = std_clim_dTwdt[:,icity]

        dTwater_f = data_dTwdt_f[imid,:,iyear,icity]
        clim_dTwdt_f = mean_clim_dTwdt_f[:,icity]
        std_dTwdt_f = std_clim_dTwdt_f[:,icity]

        dTwater_b = data_dTwdt_b[imid,:,iyear,icity]
        clim_dTwdt_b = mean_clim_dTwdt_b[:,icity]
        std_dTwdt_b = std_clim_dTwdt_b[:,icity]

        mask_Tw = np.abs(Twater-clim_Tw) > t0*std_Tw
        mask_clim = mask_Tw
        Twater_filtered = Twater.copy()
        Twater_filtered[mask_clim]=np.nan
        Twater_clean_climo[:,iyear,icity] = Twater_filtered

        mask_dTwdt = np.abs(dTwater-clim_dTwdt) > t0*std_dTwdt
        mask_clim_dTwdt = mask_dTwdt
        Twater_filtered_dTwdt = Twater.copy()
        Twater_filtered_dTwdt[mask_clim_dTwdt]=np.nan
        Twater_clean_climo_dTwdt[:,iyear,icity] = Twater_filtered_dTwdt

        mask_dTwdt_f = np.abs(dTwater_f-clim_dTwdt_f) > t0*std_dTwdt_f
        mask_clim_dTwdt_f = mask_dTwdt_f
        Twater_filtered_dTwdt_f = Twater.copy()
        Twater_filtered_dTwdt_f[mask_clim_dTwdt_f]=np.nan
        Twater_clean_climo_dTwdt_f[:,iyear,icity] = Twater_filtered_dTwdt_f

        mask_dTwdt_b = np.abs(dTwater_b-clim_dTwdt_b) > t0*std_dTwdt_b
        mask_clim_dTwdt_b = mask_dTwdt_b
        Twater_filtered_dTwdt_b = Twater.copy()
        Twater_filtered_dTwdt_b[mask_clim_dTwdt_b]=np.nan
        Twater_clean_climo_dTwdt_b[:,iyear,icity] = Twater_filtered_dTwdt_b

        mask_clim_dTwdt_forb = mask_dTwdt_f |mask_dTwdt_b
        Twater_filtered_dTwdt_forb = Twater.copy()
        Twater_filtered_dTwdt_forb[mask_clim_dTwdt_forb]=np.nan
        Twater_clean_climo_dTwdt_forb[:,iyear,icity] = Twater_filtered_dTwdt_forb


#%%
# APPLY CLIMATOLOGICAL FILTER ON WATER TEMP, AFTER
# HAVING FIRST FILTERED BASED ON F & B DT/DT FILTER.

# First re-arrange data to have each 31-day window, for each date, each year, each city
Nwindow = 31 # Only odd window size are possible
# Nwindow = 101 # Only odd window size are possible
# Nwindow = 365 # Only odd window size are possible
# Nwindow = 7 # Only odd window size are possible
data_Twater_clean_fnb = np.zeros((Nwindow,366,len(years),len(water_cities_name_list)))*np.nan
# Twater_clean_fnb = np.zeros((len(time),len(water_cities_name_list)))*np.nan

years = np.array(years)

for icity,city in enumerate(water_cities_name_list):
    Twater = Twater_clean_fnb[:,icity]

    for it in range(Twater.shape[0]):

        iw0 = np.max([0,it-int((Nwindow-1)/2)])
        iw1 = it+int((Nwindow-1)/2)+1

        Twater_window = np.zeros(Nwindow)*np.nan
        Twater_window[0:len(Twater[iw0:iw1])] = Twater[iw0:iw1]

        date_mid = date_ref+dt.timedelta(days=int(time[it]))
        year_mid = date_mid.year
        month_mid = date_mid.month
        day_mid = date_mid.day

        if len(np.where(years == year_mid)[0]) > 0:
            iyear = np.where(years == year_mid)[0][0]
            doy = (dt.date(year_mid,month_mid,day_mid)-dt.date(year_mid,1,1)).days

            data_Twater_clean_fnb[:,doy,iyear,icity] = Twater_window

            if not calendar.isleap(year_mid) and (doy == 364):
                imid = int((Nwindow-1)/2)
                Twater_window_366 = np.zeros((Nwindow))*np.nan
                Twater_window_366[imid] = np.array(np.nanmean([Twater[it],Twater[it+1]]))
                Twater_window_366[0:imid] = Twater[int(it+1-((Nwindow-1)/2)):it+1]
                Twater_window_366[imid+1:Nwindow] = Twater[it+1:int(it+1+((Nwindow-1)/2))]
                data_Twater_clean_fnb[:,365,iyear,icity] = Twater_window_366




# Then, find the 31-day climatological mean and std for each date
mean_clim_Twater_clean_fnb = np.zeros((366,len(water_cities_name_list)))*np.nan
std_clim_Twater_clean_fnb = np.zeros((366,len(water_cities_name_list)))*np.nan

for icity,city in enumerate(water_cities_name_list):

    data = data_Twater_clean_fnb[:,:,:,icity]
    mean_clim_Twater_clean_fnb[:,icity] = np.nanmean(data,axis=(0,2))
    std_clim_Twater_clean_fnb[:,icity] = np.nanstd(data,axis=(0,2))



Twater_clean_climo_fnb = np.zeros((366,len(years),len(water_cities_name_list)))*np.nan

t0  = 3.5
for iyear,year in enumerate(years):
    for icity,city in enumerate(water_cities_name_list):

        imid = int((Nwindow-1)/2)

        Twater = data_Twater_clean_fnb[imid,:,iyear,icity]
        clim_Tw = mean_clim_Twater_clean_fnb[:,icity]
        std_Tw = std_clim_Twater_clean_fnb[:,icity]

        mask_Tw = np.abs(Twater-clim_Tw) > t0*std_Tw
        mask_clim = mask_Tw
        Twater_filtered = Twater.copy()
        Twater_filtered[mask_clim]=np.nan
        Twater_clean_climo_fnb[:,iyear,icity] = Twater_filtered


#%%
# APPLY HAMPEL FILTER ON TWATER AFTER HAVING FIRST
# FILTERED BASED ON F & B DT/DT

Twater_clean_hampel_dTdt_fnb = np.zeros((len(time),len(water_cities_name_list)))*np.nan
Twater_clean_hampel_T_fnb = np.zeros((len(time),len(water_cities_name_list)))*np.nan


for icity,city in enumerate(water_cities_name_list):

    Twater = Twater_clean_fnb[:,icity]

    T_filter,filter_mask= hampel(Twater,k=45,t0=5,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)

    Tfilter_hampel = Twater.copy()
    Tfilter_hampel[filter_mask] = np.nan
    Twater_clean_hampel_T_fnb[:,icity] = Tfilter_hampel

    dTdt_tmp,filter_mask_dTdt_f= hampel(np.abs(Twater_dTdt_f[:,icity]),k=45,t0=6,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)
    dTdt_tmp,filter_mask_dTdt_b= hampel(np.abs(Twater_dTdt_b[:,icity]),k=45,t0=6,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)
    dTdt_tmp,filter_mask_dTdt= hampel(np.abs(Twater_dTdt[:,icity]),k=45,t0=6,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)

    Tfilter_hampel_dTdt = Twater.copy()
    Tfilter_hampel_dTdt[filter_mask_dTdt] = np.nan
    Twater_clean_hampel_dTdt_fnb[:,icity] = Tfilter_hampel_dTdt

    # Tfilter_hampel_dTdt_f = Twater.copy()
    # Tfilter_hampel_dTdt_f[filter_mask_dTdt_f] = np.nan
    # Twater_clean_hampel_dTdt_fnb[:,icity] = Tfilter_hampel_dTdt_f

    # Tfilter_hampel_dTdt_b = Twater.copy()
    # Tfilter_hampel_dTdt_b[filter_mask_dTdt_b] = np.nan
    # Twater_clean_hampel_dTdt_fnb[:,icity] = Tfilter_hampel_dTdt_b

#%%
# COMPARE SIMPLE MEAN/STD FILTER ON D2TDt2 VS.
# ROLLING CLIMO MEAN/STD FILTER ON D2TDt2

# First re-arrange data to have each 31-day window, for each date, each year, each city
Nwindow = 31 # Only odd window size are possible
# Nwindow = 91 # Only odd window size are possible
# Nwindow = 101 # Only odd window size are possible
# Nwindow = 365 # Only odd window size are possible
# Nwindow = 7 # Only odd window size are possible
data_d2Tdt2 = np.zeros((Nwindow,366,len(years),len(water_cities_name_list)))*np.nan
data_Tw = np.zeros((Nwindow,366,len(years),len(water_cities_name_list)))*np.nan

years = np.array(years)

for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][:,1]

    Twater_d2Tdt2_tmp = (Twater_dTdt_wrong[:,icity])

    for it in range(Twater.shape[0]):

        iw0 = np.max([0,it-int((Nwindow-1)/2)])
        iw1 = it+int((Nwindow-1)/2)+1

        Twater_window = np.zeros(Nwindow)*np.nan
        Twater_window[0:len(Twater[iw0:iw1])] = Twater[iw0:iw1]

        d2Tdt2_window = np.zeros(Nwindow)*np.nan
        d2Tdt2_window[0:len(Twater_d2Tdt2_tmp[iw0:iw1])] = Twater_d2Tdt2_tmp[iw0:iw1]

        date_mid = date_ref+dt.timedelta(days=int(time[it]))
        year_mid = date_mid.year
        month_mid = date_mid.month
        day_mid = date_mid.day

        if len(np.where(years == year_mid)[0]) > 0:
            iyear = np.where(years == year_mid)[0][0]
            doy = (dt.date(year_mid,month_mid,day_mid)-dt.date(year_mid,1,1)).days

            data_Tw[:,doy,iyear,icity] = Twater_window
            data_d2Tdt2[:,doy,iyear,icity] = d2Tdt2_window

            if not calendar.isleap(year_mid) and (doy == 364):
                imid = int((Nwindow-1)/2)
                Twater_window_366 = np.zeros((Nwindow))*np.nan
                Twater_window_366[imid] = np.array(np.nanmean([Twater[it],Twater[it+1]]))
                Twater_window_366[0:imid] = Twater[int(it+1-((Nwindow-1)/2)):it+1]
                Twater_window_366[imid+1:Nwindow] = Twater[it+1:int(it+1+((Nwindow-1)/2))]
                data_Tw[:,365,iyear,icity] = Twater_window_366

                d2Tdt2_window_366 = np.zeros((Nwindow))*np.nan
                d2Tdt2_window_366[imid] = np.array(np.nanmean([Twater_d2Tdt2_tmp[it],Twater_d2Tdt2_tmp[it+1]]))
                d2Tdt2_window_366[0:imid] = Twater_d2Tdt2_tmp[int(it+1-((Nwindow-1)/2)):it+1]
                d2Tdt2_window_366[imid+1:Nwindow] = Twater_d2Tdt2_tmp[it+1:int(it+1+((Nwindow-1)/2))]
                data_d2Tdt2[:,365,iyear,icity] = d2Tdt2_window_366





# Then, find the 31-day climatological mean and std for each date
mean_clim_d2Tdt2 = np.zeros((366,len(water_cities_name_list)))*np.nan
std_clim_d2Tdt2 = np.zeros((366,len(water_cities_name_list)))*np.nan
n_mean = np.zeros((366,len(water_cities_name_list)))*np.nan

for icity,city in enumerate(water_cities_name_list):

    data = data_d2Tdt2[:,:,:,icity]
    mean_clim_d2Tdt2[:,icity] = np.nanmean(data,axis=(0,2))
    std_clim_d2Tdt2[:,icity] = np.nanstd(data,axis=(0,2))


Twater_clean_climo_d2Tdt2 = np.zeros((366,len(years),len(water_cities_name_list)))*np.nan

t0  = 4.
for iyear,year in enumerate(years):
    for icity,city in enumerate(water_cities_name_list):

        imid = int((Nwindow-1)/2)
        Twater = data_Tw[imid,:,iyear,icity]

        d2Twater = data_d2Tdt2[imid,:,iyear,icity]
        clim_d2Tdt2 = mean_clim_d2Tdt2[:,icity]
        std_d2Tdt2 = std_clim_d2Tdt2[:,icity]

        mask_d2Tdt2 = np.abs(d2Twater-clim_d2Tdt2) > t0*std_d2Tdt2
        mask_clim_d2Tdt2 = mask_d2Tdt2
        Twater_filtered_d2Tdt2 = Twater.copy()
        Twater_filtered_d2Tdt2[mask_clim_d2Tdt2]=np.nan
        Twater_clean_climo_d2Tdt2[:,iyear,icity] = Twater_filtered_d2Tdt2

#%%
if plot_years:
    for iyear,year in enumerate(years):
        fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(6,10),sharex=True)
        plt.title(years[iyear])

        date=(dt.date(year,1,1)-date_ref).days
        i0 = np.where(time==date)[0][0]
        i1 = i0+365+calendar.isleap(year)

        for icity,city in enumerate(water_cities_name_list):
            loc_water_city = water_cities_name_list[icity]
            water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
            Twater = water_city_data['Twater'][:,1]

            # ax[0].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            # ax[0].plot(Twater_clean_fnb[i0:i1,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))
            # ax[1].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            # ax[1].plot(Twater_clean_climo_fnb[:-1,iyear,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))
            # ax[2].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            # ax[2].plot(Twater_clean_wrong[i0:i1,icity],  '.-', markersize=3, color=plt.get_cmap('tab20')(icity*2+1))

            ax[0].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            ax[0].plot(Twater_clean_wrong[i0:i1,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))
            ax[1].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            ax[1].plot(Twater_clean_climo_d2Tdt2[:-1,iyear,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))
            ax[2].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            ax[2].plot(Twater_clean_climo_dTwdt[:-1,iyear,icity],  '.-', markersize=3, color=plt.get_cmap('tab20')(icity*2+1))


            # ax[1].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            # ax[1].plot(Twater_clean_fnb[i0:i1,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))

            # ax[1].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            # ax[1].plot(Twater_clean_climo_dTwdt_f[:-1,iyear,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))

            # ax[2].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            # ax[2].plot(Twater_clean_climo_dTwdt_b[:-1,iyear,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))

            # ax[1].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            # ax[1].plot(Twater_clean_climo_fnb[:-1,iyear,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))


            # ax[1].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            # ax[1].plot(Twater_clean_climo[:-1,iyear,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))

            # ax[1].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            # ax[1].plot(Twater_clean_hampel_dTdt[i0:i1,icity], '.-', markersize=3, color=plt.get_cmap('tab20')(icity*2+1))

            # ax[2].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            # ax[2].plot(Twater_clean_hampel_dTdt_f[i0:i1,icity], '.-', markersize=3, color=plt.get_cmap('tab20')(icity*2+1))


            # ax[2].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            # ax[2].plot(Twater_clean_fb[i0:i1,icity],  '.-', markersize=3, color=plt.get_cmap('tab20')(icity*2+1))

            # ax[2].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            # ax[2].plot(Twater_clean_wrong[i0:i1,icity],  '.-', markersize=3, color=plt.get_cmap('tab20')(icity*2+1))


        ax[0].legend()

#********
# NOTES:
# Doing f or b + clim filter
# Or doing f+b  + clim filter is very similar.
# Maybe the last one is simpler to explain.


# Also Doing f or b + clim filteris very similar
# to doing only f or b filter...
# but it removes the big blob in 2017.

# So overall, choose f+b first, and then apply other clim filter.

# Next try: f+b first, and then Hampel filter.

# Doing f+b first, and then Hampel filter on T
# does not change much from doing only f+b

# Doing f+b first, and then Hampel filter on dTdt
# is pretty much the same as only doing Hampel filter on dTdt






 #%%
# if plot_years:
#     for iyear,year in enumerate(years):
#         fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(6,10),sharex=True)
#         plt.title(years[iyear])

#         date=(dt.date(year,1,1)-date_ref).days
#         i0 = np.where(time==date)[0][0]
#         i1 = i0+365+calendar.isleap(year)

#         for icity,city in enumerate(water_cities_name_list):
#             loc_water_city = water_cities_name_list[icity]
#             water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
#             Twater = water_city_data['Twater'][:,1]

#             # ax[0].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
#             # ax[0].plot(Twater_clean_hampel[i0:i1,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))

#             # ax[1].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
#             # ax[1].plot(Twater_clean_fb[i0:i1,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))

#             ax[0].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
#             ax[0].plot(Twater_clean_climo_dTwdt[:-1,iyear,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))

#             ax[1].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
#             ax[1].plot(Twater_clean_climo[:-1,iyear,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))

#             # ax[1].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
#             # ax[1].plot(Twater_clean_hampel_dTdt_f[i0:i1,icity], '.-', markersize=3, color=plt.get_cmap('tab20')(icity*2+1))

#             ax[2].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
#             ax[2].plot(Twater_clean[i0:i1,icity],  '.-', markersize=3, color=plt.get_cmap('tab20')(icity*2+1))


#         ax[0].legend()


# #%%
# if plot_dTdt:
#     for icity,city in enumerate(water_cities_name_list):
#         loc_water_city = water_cities_name_list[icity]
#         water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
#         Twater = water_city_data['Twater'][:,1]

#         fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(6,10),sharex=True)
#         plt.title(city)

#         ax[0].plot(Twater, color=plt.get_cmap('tab20')(icity*2),label=str(city))
#         ax[1].plot((Twater_dTdt[:,icity]), color=plt.get_cmap('tab20')(icity*2+1),label=str(city))

#         ax[1].plot((np.nanmean((Twater_dTdt[:,icity]))+3*np.nanstd((Twater_dTdt[:,icity])))*np.ones(len(time)),'-',color='gray')
#         ax[1].plot((np.nanmean((Twater_dTdt[:,icity]))-3*np.nanstd((Twater_dTdt[:,icity])))*np.ones(len(time)),'-',color='gray')

#         ax[2].plot((Ta_dTdt), color=plt.get_cmap('tab20')(1))
#         ax[2].plot((np.nanmean((Ta_dTdt))+3*np.nanstd((Ta_dTdt)))*np.ones(len(time)),'-',color='gray')
#         ax[2].plot((np.nanmean((Ta_dTdt))-3*np.nanstd((Ta_dTdt)))*np.ones(len(time)),'-',color='gray')


#%%






















#%%

# plt.figure()
# plt.hist(Twater_clean[:,0],bins=30,range=(-2,30),label='Longueuil',density=True,alpha=0.3)
# plt.hist(Twater_clean[:,1],bins=30,range=(-2,30),label='Atwater',density=True,alpha=0.3)
# plt.hist(Twater_clean[:,2],bins=30,range=(-2,30),label='DesBaillets',density=True,alpha=0.3)
# plt.hist(Twater_clean[:,3],bins=30,range=(-2,30),label='Candiac',density=True,alpha=0.3)

# print("Longueuil: ",np.nanmean(Twater_clean[:,0]),np.nanstd(Twater_clean[:,0]),np.nanpercentile(Twater_clean[:,0],50))
# print("Atwater: ",np.nanmean(Twater_clean[:,1]),np.nanstd(Twater_clean[:,1]),np.nanpercentile(Twater_clean[:,1],50))
# print("DesBaillets: ",np.nanmean(Twater_clean[:,2]),np.nanstd(Twater_clean[:,2]),np.nanpercentile(Twater_clean[:,2],50))
# print("Candiac: ",np.nanmean(Twater_clean[:,3]),np.nanstd(Twater_clean[:,3]),np.nanpercentile(Twater_clean[:,3],50))

# plt.hist(Twater_clean_eccc[:,0],bins=30,range=(-2,30),label='Lasalle',density=True,alpha=0.3)
# plt.hist(Twater_clean_eccc[:,1],bins=30,range=(-2,30),label='Laprairie',density=True,alpha=0.3)

# print("Lasalle: ",np.nanmean(Twater_clean_eccc[:,0]),np.nanstd(Twater_clean_eccc[:,0]),np.nanpercentile(Twater_clean_eccc[:,0],50))
# print("LaPrairie: ",np.nanmean(Twater_clean_eccc[:,1]),np.nanstd(Twater_clean_eccc[:,1]),np.nanpercentile(Twater_clean_eccc[:,1],50))

# #%%

# Twater_all = np.zeros((Twater_clean.shape[0]*len(water_cities_name_list),4))*np.nan

# # 0: Twater
# # 1: plant
# # 2: year
# # 3: season

# for icity in range(len(water_cities_name_list)):
#     year_tmp = np.zeros((Twater_clean.shape[0]))*np.nan
#     season_tmp = np.zeros((Twater_clean.shape[0]))*np.nan

#     for it in range(Twater_clean.shape[0]):
#         date_it = date_ref+dt.timedelta(days=int(time[it]))
#         year_tmp[it] = int(date_it.year)

#         if (((date_it - dt.date(int(date_it.year),3,21)).days > 0) &
#            ((date_it - dt.date(int(date_it.year),6,21)).days <= 0) ):
#                season_tmp[it] = 0 # Spring

#         if (((date_it - dt.date(int(date_it.year),6,21)).days > 0) &
#            ((date_it - dt.date(int(date_it.year),9,21)).days <= 0) ):
#                season_tmp[it] = 1 # Summer

#         if (((date_it - dt.date(int(date_it.year),9,21)).days > 0) &
#            ((date_it - dt.date(int(date_it.year),12,21)).days <= 0) ):
#                season_tmp[it] = 2 # Fall

#         if (((date_it - dt.date(int(date_it.year),12,21)).days > 0)):
#              season_tmp[it] = 3 # Winter

#         if (((date_it - dt.date(int(date_it.year),3,21)).days <= 0)):
#              season_tmp[it] = 3 # Winter

#     Twater_tmp = Twater_clean[:,icity].copy()
#     # Remove data prior to end of 2010 to compare
#     # only the same period (i.e. 2010-2020)
#     # Twater_tmp[:11220] = np.nan

#     # Remove data prior to end of 2004 to compare
#     # only the same period (i.e. 2004-2020)
#     # Twater_tmp[:8800] = np.nan


#     # Twater_tmp[:12500] = np.nan

#     # Twater_tmp[13600:] = np.nan

#     # Twater_tmp[:12670] = np.nan
#     # Twater_tmp[14676:] = np.nan

#     Twater_all[icity*Twater_clean.shape[0]:Twater_clean.shape[0]*(icity+1),0] = Twater_tmp
#     Twater_all[icity*Twater_clean.shape[0]:Twater_clean.shape[0]*(icity+1),1] = icity
#     Twater_all[icity*Twater_clean.shape[0]:Twater_clean.shape[0]*(icity+1),2] = year_tmp
#     Twater_all[icity*Twater_clean.shape[0]:Twater_clean.shape[0]*(icity+1),3] = season_tmp

# # Twater_all = Twater_all[~np.isnan(Twater_all[:,0])]

# if save:
#     np.savez('../../data/Twater_cities/Twater_cities_all_clean',
#            Twater_all=Twater_all,date_ref=date_ref)



# #%%

# Twater_all_eccc = np.zeros((Twater_clean_eccc.shape[0]*len(water_eccc_name_list),4))*np.nan

# # 0: Twater
# # 1: plant
# # 2: year
# # 3: season

# for icity in range(len(water_eccc_name_list)):
#     year_tmp = np.zeros((Twater_clean_eccc.shape[0]))*np.nan
#     season_tmp = np.zeros((Twater_clean_eccc.shape[0]))*np.nan

#     for it in range(Twater_clean_eccc.shape[0]):
#         date_it = date_ref+dt.timedelta(days=int(time[it]))
#         year_tmp[it] = int(date_it.year)

#         if (((date_it - dt.date(int(date_it.year),3,21)).days > 0) &
#            ((date_it - dt.date(int(date_it.year),6,21)).days <= 0) ):
#                season_tmp[it] = 0 # Spring

#         if (((date_it - dt.date(int(date_it.year),6,21)).days > 0) &
#            ((date_it - dt.date(int(date_it.year),9,21)).days <= 0) ):
#                season_tmp[it] = 1 # Summer

#         if (((date_it - dt.date(int(date_it.year),9,21)).days > 0) &
#            ((date_it - dt.date(int(date_it.year),12,21)).days <= 0) ):
#                season_tmp[it] = 2 # Fall

#         if (((date_it - dt.date(int(date_it.year),12,21)).days > 0)):
#              season_tmp[it] = 3 # Winter

#         if (((date_it - dt.date(int(date_it.year),3,21)).days <= 0)):
#              season_tmp[it] = 3 # Winter

#     Twater_tmp_eccc = Twater_clean_eccc[:,icity].copy()
#     # Remove data prior to end of 2010 to compare
#     # only the same period (i.e. 2010-2020)
#     # Twater_tmp_eccc[:11220] = np.nan

#     # Remove data prior to end of 2004 to compare
#     # only the same period (i.e. 2004-2020)
#     # Twater_tmp_eccc[:8800] = np.nan


#     # Twater_tmp_eccc[:12670] = np.nan
#     # Twater_tmp_eccc[14676:] = np.nan
#     # Twater_tmp_eccc[:11220] = np.nan
#     # Twater_tmp_eccc[:12500] = np.nan
#     # Twater_tmp_eccc[13600:] = np.nan


#     Twater_all_eccc[icity*Twater_clean_eccc.shape[0]:Twater_clean_eccc.shape[0]*(icity+1),0] = Twater_tmp_eccc
#     Twater_all_eccc[icity*Twater_clean_eccc.shape[0]:Twater_clean_eccc.shape[0]*(icity+1),1] = icity
#     Twater_all_eccc[icity*Twater_clean_eccc.shape[0]:Twater_clean_eccc.shape[0]*(icity+1),2] = year_tmp
#     Twater_all_eccc[icity*Twater_clean_eccc.shape[0]:Twater_clean_eccc.shape[0]*(icity+1),3] = season_tmp

# # Twater_all_eccc = Twater_all_eccc[~np.isnan(Twater_all_eccc[:,0])]

# if save:
#     np.savez('../../data/Twater_ECCC/Twater_ECCC_all_clean',
#            Twater_all=Twater_all_eccc,date_ref=date_ref)



