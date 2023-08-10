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


def hampel_back(x, k=5, t0=3., exclude_crrt_point=False, nan_substitution=False, w=None, corr_lim=1.):
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
    for i in range(k, n):

        # follow user preference on using or not the current point for
        # estimating statistical properties
        if exclude_crrt_point:
            array_neighborhood = x[i - k: i].copy()

        else:
            array_neighborhood = x[i - k: i+1].copy()


        if w is not None:
            if exclude_crrt_point:
                corr_array_neighborhood = z[i - k:i]
            else:
                corr_array_neighborhood = z[i - k: i+ 1]

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

# years = [1992,1993,1994,1995,1996,
#           1997,1998,1999,2000,2001,
#           2002,2003,2004,2005]
# years = [2006,
#           2007,2008,2009,2010,2011,
#           2012,2013,2014,2015,2016,2017,
#           2018,2019,2020]

years = [1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020]

water_cities_name_list = ['Candiac','Longueuil','Atwater','DesBaillets']
# water_cities_name_list = ['Candiac','Longueuil','DesBaillets']

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


yr_start = 0 # 1992
yr_end = len(years) -1 #2020
# yr_start = 10 # 1992
# yr_end = 13 #2005

istart = np.where(time == (dt.date(years[yr_start],1,1)-date_ref).days)[0][0]
iend = np.where(time == (dt.date(years[yr_end],12,31)-date_ref).days)[0][0]
iend += 1
if istart < 5000: istart = 0

time_select = time[istart:iend]
years = np.array(years)

#%%
# FILTER BASED ON DT/DT +/- t0*STD FOR THE WHOLE RECORD

# Comput dTwater/Dt
Twater_dTdt = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
Twater_dTdt_f = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
Twater_dTdt_b = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan

for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][istart:iend,1]
    dTdt_tmp = np.zeros((Twater.shape[0],2))*np.nan

    dTdt_tmp[1:,0] = Twater[1:] - Twater[0:-1] # Backwards
    dTdt_tmp[0:-1,1]= Twater[1:]- Twater[0:-1] # Forwards

    Twater_dTdt[:,icity] = np.nanmean(dTdt_tmp,axis=1)
    Twater_dTdt_f[:,icity] = dTdt_tmp[:,1]
    Twater_dTdt_b[:,icity] = dTdt_tmp[:,0]


# Filter records +/- t0*std away from the mean dTwater/dt
Twater_clean_f = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
Twater_clean_b = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
Twater_clean = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
Twater_clean_fb = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
Twater_clean_fnb = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan

t0 = 4.25
for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][istart:iend,1]

    Tw_dTdt = Twater_dTdt[:,icity]
    filter_mask_std = ( np.abs(np.abs(Tw_dTdt) - np.nanmean(np.abs(Tw_dTdt)))> t0*np.nanstd(np.abs(Tw_dTdt)))
    Tfilter1 = Twater.copy()
    Tfilter1[filter_mask_std] = np.nan

    Tw_dTdt_f = Twater_dTdt_f[:,icity]
    filter_mask_std_f = ( np.abs(np.abs(Tw_dTdt_f) - np.nanmean(np.abs(Tw_dTdt_f)))> t0*np.nanstd(np.abs(Tw_dTdt_f)))
    Tfilter_f = Twater.copy()
    Tfilter_f[filter_mask_std_f] = np.nan

    Tw_dTdt_b = Twater_dTdt_b[:,icity]
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



t0 = 3.5
for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][istart:iend,1]

    Tw_dTdt_f = Twater_dTdt_f[:,icity]
    filter_mask_std_f = ( np.abs(np.abs(Tw_dTdt_f) - np.nanmean(np.abs(Tw_dTdt_f)))> t0*np.nanstd(np.abs(Tw_dTdt_f)))

    Tw_dTdt_b = Twater_dTdt_b[:,icity]
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
Twater_d2Tdt2 = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan

for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][istart:iend,1]
    dTdt_tmp = np.zeros((Twater.shape[0],2))*np.nan

    dTdt_tmp[1:,0] = Twater[1:] - Twater[0:-1] # Backwards
    dTdt_tmp[0:-1,1]= Twater[0:-1] - Twater[1:] # Forwards

    Twater_d2Tdt2[:,icity] = np.nanmean(dTdt_tmp,axis=1)


# Filter records +/- t0*std away from the mean dTwater/dt
Twater_clean_d2Tdt2 = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan

t0 = 3.0
for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][istart:iend,1]

    Tw_d2Tdt2 = -1*Twater_d2Tdt2[:,icity].copy()
    filter_mask_std = (Tw_d2Tdt2 > np.nanmean(Tw_d2Tdt2)+t0*np.nanstd(Tw_d2Tdt2)) | (Tw_d2Tdt2 < np.nanmean(Tw_d2Tdt2)-t0*np.nanstd(Tw_d2Tdt2))
    Tfilter_tmp = Twater.copy()
    Tfilter_tmp[filter_mask_std] = np.nan
    Twater_clean_d2Tdt2[:,icity] = Tfilter_tmp

    print(np.nanmean(Tw_d2Tdt2)+t0*np.nanstd(Tw_d2Tdt2))

"""
We filter points with very large second derivatives (larger than 3*std)
as a proxy for abrupt changes in the temporal evolution of temperature.
This is analogous to a Laplacian filter commonly used for edge detection.
While detected peaks or edges also correlate with large values of the first
derivative in the time series, we find that in practice the filter on the
second derivative performs better. This is because the filter dTdt tends to
identify events with large dTdt that are due to synoptic variations in the
shoulder seasons. It is rather when dTdt changes abruptly that it hints
at an outlier.

Actually, the filter on dTdt can give similar result to the filter on d2Tdt2
if we increase the threshold very high, e.g. 5*std. This is because it will
otherwise detect rapid changes in tempreature in the shoulder seasons that are
most likely associated with synoptic activity and the only way to avoid
detecting those is to increase the threshold. But then by increasing the
threshold you start loosing some obvious smaller peaks that are definitely
outliers but are to small to cut the 5*std threshold..... Maybe a solution
is then to use the seasonal rolling climatology std for the threshold to have
a lower threshold in the winter and larger threshold in the shoulder seasons
because the std of winter whould be smaller than std of shoulder seasons.



*** Update:
I tried using the rolling climatology instead of the total data set
mean and std, and you basically still have to increase the threshold to 5*std
to avoid filtering the synoptic variations in the shoulder seasons...
There are a couple times where it actually filters better in the winter months
than using the plain std and mean filter, but nothing dramatic. Maybe it makes
more sense to justify this mean because it takes into account the seasonality
of temperature variations.

I also need to check and re-do the filter on d2Tdt2 using the rolling
climatology to see if it changes anything. I guess in both cases, it makes
more sens to define the mean and std and threshold using this rolling
climatology to account for the seasonal vairation of temperature variations.

Then the optimal filter would be somthing like 3*std on d2Tdt2 and 5*std on
dTdt and justify the higher threshold on dTdt because of those synoptic
variations that we do not want to filter out.  And then after all this, also
apply climatology filter on T itself to remove any remanants of point that
were not filtered? A voir!


** OTHER THOUGHTS ON THE HAMPEL FILTER:
The Hampel filter could be revisited, using a window that "sees" only before
the current point. This will keep the chronology as if we were detecting the
outliers in "real-time". And then, to avoid large value sin the pas that should
be filtered from affecting he windo mean and sd, hen he code should be modified
back o is iniial version in which he window mean and std are calculaed wih the
filtered series and not the original one.
--> It is better to use the original time series for the window rather than the filtered one as suggested above. It gives better results.
--> And it is not better to use the backward window. The results are very similar, but the centered window is a bit better overall.


 --> SECOND UPDATE: I have tried the rolling climo filter with d2Tdt2, but it
doesn't give as good results as just using the regular mean/std filter using
the whole data set. So just stick to the 3*std filter on std calculated with
the whole data set.

NEXT:
Probably just use 3*std with d2Tdt2, and then try climatology on T.

"""



#%%
Twater_clean_hampel_T = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
Twater_clean_hampel_dTdt = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
Twater_clean_hampel_d2Tdt2 = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
Twater_clean_hampel_dTdt_forb = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan

for icity,city in enumerate(water_cities_name_list):

    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][istart:iend,1]
    Tw_d2Tdt2 = Twater_d2Tdt2[:,icity]
    Tw_dTdt = Twater_dTdt[:,icity]
    Tw_dTdt_f = Twater_dTdt_f[:,icity]
    Tw_dTdt_b = Twater_dTdt_b[:,icity]

    T_filter,filter_mask          = hampel(Twater,k=15,t0=5,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)
    dTdt_tmp,filter_mask_dTdt_f   = hampel(np.abs(Twater_dTdt_f[:,icity]),k=15,t0=5,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)
    dTdt_tmp,filter_mask_dTdt_b   = hampel(np.abs(Twater_dTdt_b[:,icity]),k=15,t0=5,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)
    dTdt_tmp,filter_mask_dTdt     = hampel(np.abs(Twater_dTdt[:,icity]),k=15,t0=5,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)
    d2Tdt2_tmp,filter_mask_d2Tdt2 = hampel(Twater_d2Tdt2[:,icity],k=15,t0=5,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)

    Tfilter_hampel = Twater.copy()
    Tfilter_hampel[filter_mask] = np.nan
    Twater_clean_hampel_T[:,icity] = Tfilter_hampel

    Tfilter_hampel_dTdt = Twater.copy()
    Tfilter_hampel_dTdt[filter_mask_dTdt] = np.nan
    Twater_clean_hampel_dTdt[:,icity] = Tfilter_hampel_dTdt

    Tfilter_hampel_dTdt_forb = Twater.copy()
    mask_forb = filter_mask_dTdt_f | filter_mask_dTdt_b
    Tfilter_hampel_dTdt_forb[mask_forb] = np.nan
    Twater_clean_hampel_dTdt_forb[:,icity] = Tfilter_hampel_dTdt_forb

    Tfilter_hampel_d2Tdt2 = Twater.copy()
    Tfilter_hampel_d2Tdt2[filter_mask_d2Tdt2] = np.nan
    Twater_clean_hampel_d2Tdt2[:,icity] = Tfilter_hampel_d2Tdt2

#%%
# FIRST FILTER ON D2TDt2 AND THEN
# APPLY ROLLING CLIMATOLOGY FILTER ON T

# First re-arrange data to have each 31-day window, for each date, each year, each city
Nwindow = 31 # Only odd window size are possible
data_Twater_clean_d2Tdt2_plus_climo = np.zeros((Nwindow,366,len(years[yr_start:yr_end+1]),len(water_cities_name_list)))*np.nan
data_Twater_clean_hampel_d2Tdt2_plus_climo = np.zeros((Nwindow,366,len(years[yr_start:yr_end+1]),len(water_cities_name_list)))*np.nan

for icity,city in enumerate(water_cities_name_list):
    Twater = Twater_clean_d2Tdt2[:,icity]
    Twater_hampel = Twater_clean_hampel_d2Tdt2[:,icity]

    for it in range(Twater.shape[0]):

        iw0 = np.max([0,it-int((Nwindow-1)/2)])
        iw1 = it+int((Nwindow-1)/2)+1

        Twater_window = np.zeros(Nwindow)*np.nan
        Twater_window[0:len(Twater[iw0:iw1])] = Twater[iw0:iw1]

        Twater_hampel_window = np.zeros(Nwindow)*np.nan
        Twater_hampel_window[0:len(Twater[iw0:iw1])] = Twater_hampel[iw0:iw1]

        date_mid = date_ref+dt.timedelta(days=int(time[it]))
        year_mid = date_mid.year
        month_mid = date_mid.month
        day_mid = date_mid.day

        if len(np.where(years[yr_start:yr_end+1] == year_mid)[0]) > 0:
            iyear = np.where(years[yr_start:yr_end+1] == year_mid)[0][0]
            doy = (dt.date(year_mid,month_mid,day_mid)-dt.date(year_mid,1,1)).days

            data_Twater_clean_d2Tdt2_plus_climo[:,doy,iyear,icity] = Twater_window
            data_Twater_clean_hampel_d2Tdt2_plus_climo[:,doy,iyear,icity] = Twater_hampel_window

            if not calendar.isleap(year_mid) and (doy == 364):
                imid = int((Nwindow-1)/2)
                Twater_window_366 = np.zeros((Nwindow))*np.nan
                Twater_window_366[imid] = np.array(np.nanmean([Twater[it],Twater[it+1]]))
                Twater_window_366[0:imid] = Twater[int(it+1-((Nwindow-1)/2)):it+1]
                Twater_window_366[imid+1:Nwindow] = Twater[it+1:int(it+1+((Nwindow-1)/2))]
                data_Twater_clean_d2Tdt2_plus_climo[:,365,iyear,icity] = Twater_window_366

                Twater_hampel_window_366 = np.zeros((Nwindow))*np.nan
                Twater_hampel_window_366[imid] = np.array(np.nanmean([Twater_hampel[it],Twater_hampel[it+1]]))
                Twater_hampel_window_366[0:imid] = Twater_hampel[int(it+1-((Nwindow-1)/2)):it+1]
                Twater_hampel_window_366[imid+1:Nwindow] = Twater_hampel[it+1:int(it+1+((Nwindow-1)/2))]
                data_Twater_clean_hampel_d2Tdt2_plus_climo[:,365,iyear,icity] = Twater_hampel_window_366


# Then, find the 31-day climatological mean and std for each date
mean_clim_Twater_clean_d2Tdt2 = np.zeros((366,len(water_cities_name_list)))*np.nan
std_clim_Twater_clean_d2Tdt2 = np.zeros((366,len(water_cities_name_list)))*np.nan
mean_clim_Twater_clean_hampel_d2Tdt2 = np.zeros((366,len(water_cities_name_list)))*np.nan
std_clim_Twater_clean_hampel_d2Tdt2 = np.zeros((366,len(water_cities_name_list)))*np.nan

for icity,city in enumerate(water_cities_name_list):

    data = data_Twater_clean_d2Tdt2_plus_climo[:,:,:,icity]
    mean_clim_Twater_clean_d2Tdt2[:,icity] = np.nanmean(data,axis=(0,2))
    std_clim_Twater_clean_d2Tdt2[:,icity] = np.nanstd(data,axis=(0,2))

    data = data_Twater_clean_hampel_d2Tdt2_plus_climo[:,:,:,icity]
    mean_clim_Twater_clean_hampel_d2Tdt2 [:,icity] = np.nanmean(data,axis=(0,2))
    std_clim_Twater_clean_hampel_d2Tdt2[:,icity] = np.nanstd(data,axis=(0,2))



Twater_clean_d2Tdt2_plus_climo = np.zeros((366,len(years[yr_start:yr_end+1]),len(water_cities_name_list)))*np.nan
Twater_clean_hampel_d2Tdt2_plus_climo = np.zeros((366,len(years[yr_start:yr_end+1]),len(water_cities_name_list)))*np.nan

t0  = 3.5
for iyear,year in enumerate(years[yr_start:yr_end+1]):
    for icity,city in enumerate(water_cities_name_list):

        imid = int((Nwindow-1)/2)
        # Twater = Twater_clean_d2Tdt2[:,icity]
        # Twater_hampel = Twater_clean_hampel_d2Tdt2[:,icity]

        Twater = data_Twater_clean_d2Tdt2_plus_climo[imid,:,iyear,icity]
        clim_Tw = mean_clim_Twater_clean_d2Tdt2[:,icity]
        std_Tw = std_clim_Twater_clean_d2Tdt2[:,icity]

        Twater_hampel = data_Twater_clean_hampel_d2Tdt2_plus_climo[imid,:,iyear,icity]
        clim_Tw_hampel = mean_clim_Twater_clean_hampel_d2Tdt2[:,icity]
        std_Tw_hampel = std_clim_Twater_clean_hampel_d2Tdt2[:,icity]

        mask_Tw = np.abs(Twater-clim_Tw) > t0*std_Tw
        mask_clim = mask_Tw
        Twater_filtered = Twater.copy()
        Twater_filtered[mask_clim]=np.nan
        Twater_clean_d2Tdt2_plus_climo[:,iyear,icity] = Twater_filtered

        mask_Tw_hampel = np.abs(Twater_hampel-clim_Tw_hampel) > t0*std_Tw_hampel
        mask_clim_hampel = mask_Tw_hampel
        Twater_hampel_filtered = Twater_hampel.copy()
        Twater_hampel_filtered[mask_clim_hampel]=np.nan
        Twater_clean_hampel_d2Tdt2_plus_climo[:,iyear,icity] = Twater_hampel_filtered

#%%
# FIRST FILTER ON D2TDt2 AND THEN
# APPLY ROLLING CLIMATOLOGY FILTER ON T

Twater_clean_d2Tdt2_plus_hampel = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
Twater_clean_hampel_d2Tdt2_plus_hampel = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan

for icity,city in enumerate(water_cities_name_list):

    Twater = Twater_clean_d2Tdt2[:,icity]
    Twater_hampel = Twater_clean_hampel_d2Tdt2[:,icity]


    T_filter,filter_mask          = hampel(Twater,k=15,t0=3.5,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)
    T_filter_hampel,filter_mask_hampel = hampel(Twater_hampel,k=15,t0=3.5,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)

    Tfilter = Twater.copy()
    Tfilter[filter_mask] = np.nan
    Twater_clean_d2Tdt2_plus_hampel[:,icity] = Tfilter

    Tfilter_hampel = Twater_hampel.copy()
    Tfilter_hampel[filter_mask_hampel] = np.nan
    Twater_clean_hampel_d2Tdt2_plus_hampel[:,icity] = Tfilter_hampel



#%%
# OK, SO IN THE END, THE BOTTOM 4 ONES ARE THE BEST OPTIONS, AND
#  PROBABLY ESPECIALLY THE MIDDLE ROW.

if plot_years:
    for iyear,year in enumerate(years[yr_start:yr_end+1]):
        fig,ax = plt.subplots(nrows=3,ncols=2,figsize=(12,10),sharex=True)
        plt.title(years[yr_start:yr_end+1][iyear])

        date=(dt.date(year,1,1)-date_ref).days
        i0 = np.where(time==date)[0][0]
        i1 = i0+365+calendar.isleap(year)

        i0_clean = np.where(time_select==date)[0][0]
        i1_clean = i0_clean+365+calendar.isleap(year)

        for icity,city in enumerate(water_cities_name_list):
        # for icity,city in enumerate(['Candiac']):

            loc_water_city = water_cities_name_list[icity]
            water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
            Twater = water_city_data['Twater'][:,1]

            ax[0,0].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            ax[0,0].plot(Twater_clean_d2Tdt2[i0_clean:i1_clean,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))

            #THIS ONE SEEMS TO BE THE BEST TO START WITH!!
            ax[1,0].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            ax[1,0].plot(Twater_clean_d2Tdt2_plus_climo[:-1, iyear,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))

            ax[2,0].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            ax[2,0].plot(Twater_clean_d2Tdt2_plus_hampel[i0_clean:i1_clean,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))


            ax[0,1].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            ax[0,1].plot(Twater_clean_hampel_d2Tdt2[i0_clean:i1_clean,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))

            ax[1,1].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            ax[1,1].plot(Twater_clean_hampel_d2Tdt2_plus_climo[:-1, iyear,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))

            ax[2,1].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
            ax[2,1].plot(Twater_clean_hampel_d2Tdt2_plus_hampel[i0_clean:i1_clean,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))


        ax[0,0].legend()


#%%

for iyear,year in enumerate(years[yr_start:yr_end+1]):
    fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(6,10),sharex=True)
    plt.title(years[yr_start:yr_end+1][iyear])

    date=(dt.date(year,1,1)-date_ref).days
    i0 = np.where(time==date)[0][0]
    i1 = i0+365+calendar.isleap(year)

    i0_clean = np.where(time_select==date)[0][0]
    i1_clean = i0_clean+365+calendar.isleap(year)

    for icity,city in enumerate(water_cities_name_list):
        loc_water_city = water_cities_name_list[icity]
        water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
        Twater = water_city_data['Twater'][i0:i1,1]
        Twater_clean = Twater_clean_d2Tdt2[i0_clean:i1_clean,icity]
        d2Tdt2 = Twater_d2Tdt2[i0_clean:i1_clean,icity]

        zero_d2Tdt2_mask = np.abs(d2Tdt2) < 0.04
        zero_T_mask = np.abs(Twater) < 2.0

        # T is above zero and d2Tdt2 is zero:
        mask_1 = ~zero_T_mask & zero_d2Tdt2_mask
        mask_line = mask_1.copy()
        mask_line[:] = False

        # THIS WILL MASK PERIODS OF 7 DAYS + OF CONSTANT DT/DT
        # INDICATIING EITHER THAT T IS CONSTANT FOR 7 DAYS OR MORE
        # WHICH WOULD INDICATE A PROBLEM WITH THE TEMPERATURE PROBE
        # OR IT WILL ALSO IDENTIFY LINEAR SEGMENTS SUCH AS IN 2010.
        # THE THRESHOLD OF 0.04 MIGHT NEED TO BE ADJUSTED FOR OTHER
        # DATA SETS...
        # FOR ATWATER, THIS FILTER MIGHT ALSO BE PROBLEMATIC BECAUSE
        # WHEN T IS ROUNDED TO THE UNIT ONLY, THERE ARE PERIODS WITH
        # SEVERAL DAYS MARKED AS THE SAME TEMPERATURE...

        # *** ALSO THE NUMBER OF DAYS (7 DAYS OR MORE) SHOULD BE
        # CHOSEN TO BE CONSISTENT WITH THE LARGEST INTERVAL I CHOOSE
        # TO PATCH WITH LINEAR INTERPOLATION.

        for im in range(1,mask_line.size):

            if (im == 1) | (~mask_1[im-1]):
                # start new grouping
                sum_m = 0
                if ~mask_1[im]:
                    sum_m = 0
                else:
                    sum_m +=1
                    istart = im

            else:
                if mask_1[im]:
                    sum_m += 1
                else:
                    # This is the end of the group,
                    # so count the total number of
                    # points in group, and remove group
                    # if total is larger than threshold
                    iend = im
                    if sum_m >= 7:
                        mask_line[istart:iend] = True

                    sum_m = 0 # Put back sum to zero


        ax[0].plot(Twater, color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
        ax[0].plot(Twater_clean, '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))

        Twater_clean_line = Twater.copy()
        Twater_clean_line[mask_line] = np.nan
        ax[1].plot(Twater_clean, color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
        ax[1].plot(Twater_clean_line, '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))

        ax[2].plot(d2Tdt2, color=plt.get_cmap('tab20')(0))
        ax[2].plot(np.ones(d2Tdt2.shape)*0.04, '-', color='gray')
        ax[2].plot(np.ones(d2Tdt2.shape)*(-0.04), '-', color='gray')


#%%

def mask_lines_and_steps(Twater_in, d2Tdt2_in, thresh_T = 2.0, thresh_d2Tdt2 = 0.04, ndays = 7):

    # THIS WILL MASK PERIODS OF 7 DAYS + OF CONSTANT DT/DT
    # INDICATIING EITHER THAT T IS CONSTANT FOR 7 DAYS OR MORE
    # WHICH WOULD INDICATE A PROBLEM WITH THE TEMPERATURE PROBE
    # OR IT WILL ALSO IDENTIFY LINEAR SEGMENTS SUCH AS IN 2010.
    # THE THRESHOLD OF 0.04 MIGHT NEED TO BE ADJUSTED FOR OTHER
    # DATA SETS...

    # FOR ATWATER, THIS FILTER MIGHT ALSO BE PROBLEMATIC BECAUSE
    # WHEN T IS ROUNDED TO THE UNIT ONLY, THERE ARE PERIODS WITH
    # SEVERAL DAYS MARKED AS THE SAME TEMPERATURE...

    # *** ALSO THE NUMBER OF DAYS (7 DAYS OR MORE) SHOULD BE
    # CHOSEN TO BE CONSISTENT WITH THE LARGEST INTERVAL I CHOOSE
    # TO PATCH WITH LINEAR INTERPOLATION.

    zero_d2Tdt2_mask = np.abs(d2Tdt2_in) < thresh_d2Tdt2
    zero_T_mask = np.abs(Twater_in) < thresh_T

    # T is above thresh_T and d2Tdt2 is below thresh_d2Tdt2:
    mask_tmp = ~zero_T_mask & zero_d2Tdt2_mask
    mask_line = mask_tmp.copy()
    mask_line[:] = False

    for im in range(1,mask_line.size):

        if (im == 1) | (~mask_tmp[im-1]):
            # start new group
            sum_m = 0
            if ~mask_tmp[im]:
                sum_m = 0
            else:
                sum_m +=1
                istart = im

        else:
            if mask_tmp[im]:
                sum_m += 1
            else:
                # This is the end of the group of constant dTdt,
                # so count the total number of points in group,
                # and remove whole group if total is larger than
                # ndays
                iend = im
                if sum_m >= ndays:
                    mask_line[istart:iend] = True

                sum_m = 0 # Put back sum to zero

    Twater_out = Twater_in.copy()
    Twater_out[mask_line] = np.nan

    return Twater_out, mask_line


for icity,city in enumerate(water_cities_name_list):
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
    plt.title(city)
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][:,1]
    dTdt   = Twater_dTdt[:,icity]
    dTdt_f = Twater_dTdt_f[:,icity]
    dTdt_b = Twater_dTdt_b[:,icity]
    d2Tdt2 = Twater_d2Tdt2[:,icity]


    Twater_mask_lines, _ = mask_lines_and_steps(Twater, d2Tdt2)

    ax.plot(Twater, color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
    ax.plot(Twater_mask_lines, '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))




for iyear,year in enumerate(years[yr_start:yr_end+1]):
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
    plt.title(years[yr_start:yr_end+1][iyear])

    date=(dt.date(year,1,1)-date_ref).days
    i0 = np.where(time==date)[0][0]
    i1 = i0+365+calendar.isleap(year)

    i0_clean = np.where(time_select==date)[0][0]
    i1_clean = i0_clean+365+calendar.isleap(year)

    for icity,city in enumerate(water_cities_name_list):
        loc_water_city = water_cities_name_list[icity]
        water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
        Twater = water_city_data['Twater'][i0:i1,1]
        dTdt   = Twater_dTdt[:,icity]
        dTdt_f = Twater_dTdt_f[:,icity]
        dTdt_b = Twater_dTdt_b[:,icity]
        d2Tdt2 = Twater_d2Tdt2[i0_clean:i1_clean,icity]


        Twater_mask_lines, _ = mask_lines_and_steps(Twater, d2Tdt2)

        ax.plot(Twater, color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
        ax.plot(Twater_mask_lines, '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))


#%%
# import torch
# import pandas as pd
# from pandas.plotting import autocorrelation_plot

# for icity,city in enumerate(water_cities_name_list):
#     plt.figure()
#     loc_water_city = water_cities_name_list[icity]
#     water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')

#     Twater = water_city_data['Twater'][:,1]
#     df = pd.DataFrame(Twater)
#     autocorrelation_plot(df,label=city)

#     plt.legend()


# fft = torch.rfft(torch.from_numpy(Twater[~np.isnan(Twater)]),1)
# f_per_dataset = np.arange(0, len(fft))

# days_per_year = 365.2524

# n_samples_Tw = len(Twater[~np.isnan(Twater)])
# years_per_dataset = n_samples_Tw/(days_per_year)

# f_per_year= f_per_dataset/years_per_dataset

# plt.figure()
# plt.step(f_per_year, np.abs(fft))
# plt.xscale('log')
# # plt.ylim(0, 27000)
# # plt.xlim([0.1, max(plt.xlim())])
# plt.xticks([1,365.2524], labels=['1/year','1/day'])
# _ = plt.xlabel('Frequency (log scale)')

#%%
# from scipy import ndimage

# Twater_clean_LoG = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan

# cross_dTdt = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
# cross_d2Tdt2 = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
# Twater_clean_peaks = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
# Twater_clean_LoG2 = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
# Twater_clean_LoG3 = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
# Twater_clean_LoG4 = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan


# t0 = 5
# for icity,city in enumerate(water_cities_name_list):
#     loc_water_city = water_cities_name_list[icity]
#     water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
#     Twater = water_city_data['Twater'][istart:iend,1]
#     Tw_d2Tdt2 = Twater_d2Tdt2[:,icity]
#     Tw_dTdt = Twater_dTdt[:,icity]

#     Tplot = Twater

#     # LoG = ndimage.gaussian_laplace(Tplot, sigma=0.95)
#     # thres_mean = np.nanmean(np.absolute(LoG))
#     # thres_std = np.nanstd(np.absolute(LoG))*t0

#     mean_d2Tdt2 = np.nanmean((Tw_d2Tdt2))
#     std_d2Tdt2 = np.nanstd((Tw_d2Tdt2))
#     thres_d2Tdt2 = 3.*std_d2Tdt2

#     mean_dTdt = np.nanmean((Tw_dTdt))
#     std_dTdt = np.nanstd((Tw_dTdt))
#     thres_dTdt = 3.*std_dTdt


#     output_LoG = np.zeros(Tplot.shape)
#     output_LoG2 = np.zeros(Tplot.shape)
#     output_LoG3 = np.zeros(Tplot.shape)
#     output_LoG4 = np.zeros(Tplot.shape)
#     filter_bothcross_plussize = np.zeros(Tw_d2Tdt2.shape)

#     for y in range(1, Tplot.shape[0] - 1):
#         patch_d2Tdt2 = Tw_d2Tdt2[y-1:y+2]
#         p_d2Tdt2 = Tw_d2Tdt2[y]
#         maxP_d2Tdt2 = np.nanmax(patch_d2Tdt2)
#         minP_d2Tdt2 = np.nanmin(patch_d2Tdt2)

#         if (p_d2Tdt2 > 0):
#             zeroCross_d2Tdt2 = True if minP_d2Tdt2 < 0 else False
#         else:
#             zeroCross_d2Tdt2 = True if maxP_d2Tdt2 > 0 else False

#         if zeroCross_d2Tdt2:
#             cross_d2Tdt2[y,icity] = p_d2Tdt2
#         else:
#             cross_d2Tdt2[y,icity] = np.nan


#         patch_dTdt = Tw_dTdt[y-1:y+2]
#         p_dTdt = Tw_dTdt[y]
#         maxP_dTdt = np.nanmax(patch_dTdt)
#         minP_dTdt = np.nanmin(patch_dTdt)

#         if (p_dTdt > 0):
#             zeroCross_dTdt = True if minP_dTdt < 0 else False
#         else:
#             zeroCross_dTdt = True if maxP_dTdt > 0 else False

#         if zeroCross_dTdt:
#             cross_dTdt[y,icity] = p_dTdt
#         else:
#             cross_dTdt[y,icity] = np.nan



#         if zeroCross_dTdt & zeroCross_d2Tdt2:
#             if ( np.abs(p_d2Tdt2 - np.nanmean(Tw_d2Tdt2)) > 3.*np.nanstd(Tw_d2Tdt2) ) | (np.abs(p_dTdt - np.nanmean(Tw_dTdt)) > 3.*np.nanstd(Tw_dTdt) ):
#                 filter_bothcross_plussize[y]=np.nan

#         Twater_filter_tmp = Twater.copy()
#         Twater_filter_tmp[np.isnan(filter_bothcross_plussize)] = np.nan
#         Twater_clean_peaks[:,icity] = Twater_filter_tmp


#         if (np.abs(p_d2Tdt2-mean_d2Tdt2) > t0*std_d2Tdt2) and zeroCross_d2Tdt2:
#             output_LoG4[y] = 1
#             #Modify so that we only fiter if the point in the middle is the min or max.
#             # This will treatt the crossing only once
#         Tfilter_LoG4 = Twater.copy()
#         Tfilter_LoG4[output_LoG4 == 1] = np.nan
#         Twater_clean_LoG4[:,icity] = Tfilter_LoG4

#         if np.abs(p_d2Tdt2-mean_d2Tdt2) > t0*std_d2Tdt2:
#             output_LoG3[y] = 1 # This one is the same as Twater_dTdt_wrong

#         Tfilter_LoG3 = Twater.copy()
#         Tfilter_LoG3[output_LoG3 == 1] = np.nan
#         Twater_clean_LoG3[:,icity] = Tfilter_LoG3


#         if (np.abs(p_dTdt-mean_dTdt) > t0*std_dTdt):
#             output_LoG2[y] = 1
#         Tfilter_LoG2 = Twater.copy()
#         Tfilter_LoG2[output_LoG2 == 1] = np.nan
#         Twater_clean_LoG2[:,icity] = Tfilter_LoG2



#%%

# Twater_clean_hampel_T_back = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
# Twater_clean_hampel_dTdt_back = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
# Twater_clean_hampel_d2Tdt2_back = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
# Twater_clean_hampel_dTdt_forb_back = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan

# for icity,city in enumerate(water_cities_name_list):

#     loc_water_city = water_cities_name_list[icity]
#     water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
#     Twater = water_city_data['Twater'][istart:iend,1]
#     Tw_d2Tdt2 = Twater_d2Tdt2[:,icity]
#     Tw_dTdt = Twater_dTdt[:,icity]
#     Tw_dTdt_f = Twater_dTdt_f[:,icity]
#     Tw_dTdt_b = Twater_dTdt_b[:,icity]



#     T_filter,filter_mask          = hampel_back(Twater,k=30,t0=5,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)
#     dTdt_tmp,filter_mask_dTdt_f   = hampel_back(np.abs(Twater_dTdt_f[:,icity]),k=30,t0=5,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)
#     dTdt_tmp,filter_mask_dTdt_b   = hampel_back(np.abs(Twater_dTdt_b[:,icity]),k=30,t0=5,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)
#     dTdt_tmp,filter_mask_dTdt     = hampel_back(np.abs(Twater_dTdt[:,icity]),k=30,t0=5,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)
#     d2Tdt2_tmp,filter_mask_d2Tdt2 = hampel_back(Twater_d2Tdt2[:,icity],k=30,t0=5,exclude_crrt_point=True,w=Ta[:,1],corr_lim=1,nan_substitution=True)

#     Tfilter_hampel = Twater.copy()
#     Tfilter_hampel[filter_mask] = np.nan
#     Twater_clean_hampel_T_back[:,icity] = Tfilter_hampel

#     Tfilter_hampel_dTdt = Twater.copy()
#     Tfilter_hampel_dTdt[filter_mask_dTdt] = np.nan
#     Twater_clean_hampel_dTdt_back[:,icity] = Tfilter_hampel_dTdt

#     Tfilter_hampel_dTdt_forb = Twater.copy()
#     mask_forb = filter_mask_dTdt_f | filter_mask_dTdt_b
#     Tfilter_hampel_dTdt_forb[mask_forb] = np.nan
#     Twater_clean_hampel_dTdt_forb_back[:,icity] = Tfilter_hampel_dTdt_forb

#     Tfilter_hampel_d2Tdt2 = Twater.copy()
#     Tfilter_hampel_d2Tdt2[filter_mask_d2Tdt2] = np.nan
#     Twater_clean_hampel_d2Tdt2_back[:,icity] = Tfilter_hampel_d2Tdt2


#%%
# if plot_years:
#     # for iyear,year in enumerate(years[-17:]):
#     for iyear,year in enumerate(years[:]):
#         fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(6,10),sharex=True)
#         plt.title(years[iyear])

#         date=(dt.date(year,1,1)-date_ref).days
#         i0 = np.where(time==date)[0][0]
#         i1 = i0+365+calendar.isleap(year)

#         i0_clean = np.where(time_select==date)[0][0]
#         i1_clean = i0_clean+365+calendar.isleap(year)

#         for icity,city in enumerate(water_cities_name_list):
#         # for icity,city in enumerate(['Candiac']):

#             # icity = 3
#             # city = 'DesBaillets'
#             icity = 0
#             city = 'Candiac'
#             loc_water_city = water_cities_name_list[icity]
#             water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
#             Twater = water_city_data['Twater'][:,1]

#             ax[0].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
#             ax[0].plot(Twater_clean_d2Tdt2[i0_clean:i1_clean,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))

#             ax[1].plot(Twater[i0:i1], color=plt.get_cmap('tab20')(icity*2),label=water_cities_name_list[icity])
#             ax[1].plot(Twater_clean_hampel_d2Tdt2[i0_clean:i1_clean,icity], '.-', markersize=3,  color=plt.get_cmap('tab20')(icity*2+1))

#             # Twater_clean_hampel_T = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
#             # Twater_clean_hampel_dTdt = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
#             # Twater_clean_hampel_d2Tdt2 = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan
#             # Twater_clean_hampel_dTdt_forb = np.zeros((len(time[istart:iend]),len(water_cities_name_list)))*np.nan

#             # Twater_clean_hampel_d2Tdt2_plus_climo
#             # Twater_clean_d2Tdt2_plus_climo




#             ax[2].plot(np.zeros(Twater_d2Tdt2[i0_clean:i1_clean,icity].shape),'-',color='gray')
#             ax[2].plot(Twater_d2Tdt2[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(0))
#             ax[2].plot(Twater_dTdt[i0_clean:i1_clean,icity], color=plt.get_cmap('tab20')(2))

#             ax[2].plot(np.ones(Twater_d2Tdt2[i0_clean:i1_clean,icity].shape)*np.nanmean(Twater_d2Tdt2[:,icity]), '-', color='gray')
#             ax[2].plot(np.ones(Twater_d2Tdt2[i0_clean:i1_clean,icity].shape)*(np.nanmean(Twater_d2Tdt2[:,icity])+3*np.nanstd(Twater_d2Tdt2[:,icity])), '--', color=plt.get_cmap('tab20')(1))
#             ax[2].plot(np.ones(Twater_d2Tdt2[i0_clean:i1_clean,icity].shape)*(np.nanmean(Twater_d2Tdt2[:,icity])-3*np.nanstd(Twater_d2Tdt2[:,icity])), '--', color=plt.get_cmap('tab20')(1))

#             ax[2].plot(np.ones(Twater_dTdt[i0_clean:i1_clean,icity].shape)*np.nanmean(Twater_dTdt[:,icity]), '-', color='gray')
#             ax[2].plot(np.ones(Twater_dTdt[i0_clean:i1_clean,icity].shape)*(np.nanmean(Twater_dTdt[:,icity])+3*np.nanstd(Twater_dTdt[:,icity])), '--', color=plt.get_cmap('tab20')(3))
#             ax[2].plot(np.ones(Twater_dTdt[i0_clean:i1_clean,icity].shape)*(np.nanmean(Twater_dTdt[:,icity])-3*np.nanstd(Twater_dTdt[:,icity])), '--', color=plt.get_cmap('tab20')(3))



#             # c1 = cross_dTdt[i0_clean:i1_clean,icity]
#             # c2 = cross_d2Tdt2[i0_clean:i1_clean,icity]

#             # c23 = np.zeros(c1.shape)*np.nan
#             # c13 = np.zeros(c1.shape)*np.nan
#             # mask1 = ~np.isnan(c1)
#             # mask2 = ~np.isnan(c2)
#             # mask = mask1 & mask2
#             # c23[mask] = c2[mask]
#             # c13[mask] = c1[mask]
#             # ax[2].plot(c23, '.', color=plt.get_cmap('tab20')(1))
#             # ax[2].plot(c13, '.', color=plt.get_cmap('tab20')(3))



#         ax[0].legend()
