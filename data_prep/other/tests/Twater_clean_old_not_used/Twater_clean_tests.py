#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:55:11 2021

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

# years = [1992,1993,1994,1995,1996,
#           1997,1998,1999,2000,2001,
#           2002,2003,2004,2005]
years = [2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020]
# years = [1992,1993,1994,1995,1996,
#           1997,1998,1999,2000,2001,
#           2002,2003,2004,2005,2006,
#           2007,2008,2009,2010,2011,
#           2012,2013,2014,2015,2016,2017,
#           2018,2019,2020]
# water_cities_name_list = ['Longueuil_preclean','Atwater_preclean','DesBaillets_preclean','Candiac_preclean']
# water_eccc_name_list = ['Lasalle', 'LaPrairie']
water_cities_name_list = ['Candiac','DesBaillets','Atwater','Longueuil']
# water_cities_name_list = ['Atwater']
# water_cities_name_list = ['Candiac']

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

# Compute dTwater/Dt
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

    # Twater_dTdt[:,icity] = np.abs(np.nanmean(dTdt_tmp,axis=1))
    # Twater_dTdt_f[:,icity] = np.abs( dTdt_tmp[:,1])
    # Twater_dTdt_b[:,icity] = np.abs(dTdt_tmp[:,0])

    Twater_dTdt[:,icity] = (np.nanmean(dTdt_tmp,axis=1))
    Twater_dTdt_f[:,icity] = (dTdt_tmp[:,1])
    Twater_dTdt_b[:,icity] = (dTdt_tmp[:,0])

# First re-arrange data to have each 31-day window, for each date, each year, each city
Nwindow = 31 # Only odd window size are possible
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

    Twater_dTdt_tmp = Twater_dTdt[:,icity]
    Twater_dTdt_f_tmp = Twater_dTdt_f[:,icity]
    Twater_dTdt_b_tmp = Twater_dTdt_b[:,icity]

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


mean_clim_dTwdt_all = np.zeros((366))*np.nan
std_clim_dTwdt_all = np.zeros((366))*np.nan

mean_clim_dTwdt_all_f = np.zeros((366))*np.nan
std_clim_dTwdt_all_f = np.zeros((366))*np.nan

mean_clim_dTwdt_all_b = np.zeros((366))*np.nan
std_clim_dTwdt_all_b = np.zeros((366))*np.nan


for icity,city in enumerate(water_cities_name_list):

    data = data_Tw[:,:,:,icity]
    mean_clim_Tw[:,icity] = np.nanmean(data,axis=(0,2))
    std_clim_Tw[:,icity] = np.nanstd(data,axis=(0,2))

    data = data_dTwdt[:,:,:,icity]
    mean_clim_dTwdt[:,icity] = np.nanmean(data,axis=(0,2))
    std_clim_dTwdt[:,icity] = np.nanstd(data,axis=(0,2))

    data = data_dTwdt_f[:,:,:,icity]
    mean_clim_dTwdt_f[:,icity] = np.nanmean(data,axis=(0,2))
    std_clim_dTwdt_f[:,icity] = np.nanstd(data,axis=(0,2))

    data = data_dTwdt_b[:,:,:,icity]
    mean_clim_dTwdt_b[:,icity] = np.nanmean(data,axis=(0,2))
    std_clim_dTwdt_b[:,icity] = np.nanstd(data,axis=(0,2))



    data = data_dTwdt
    mean_clim_dTwdt_all = np.nanmean(data,axis=(0,2,3))
    std_clim_dTwdt_all= np.nanstd(data,axis=(0,2,3))

    data = data_dTwdt_f
    mean_clim_dTwdt_all_f= np.nanmean(data,axis=(0,2,3))
    std_clim_dTwdt_all_f = np.nanstd(data,axis=(0,2,3))

    data = data_dTwdt_b
    mean_clim_dTwdt_all_b = np.nanmean(data,axis=(0,2,3))
    std_clim_dTwdt_all_b = np.nanstd(data,axis=(0,2,3))



Twater_clean_climo = np.zeros((366,len(years),len(water_cities_name_list)))*np.nan
Twater_clean_climo_dTwdt = np.zeros((366,len(years),len(water_cities_name_list)))*np.nan
Twater_clean_climo_dTwdt_f = np.zeros((366,len(years),len(water_cities_name_list)))*np.nan
Twater_clean_climo_dTwdt_b = np.zeros((366,len(years),len(water_cities_name_list)))*np.nan

Twater_clean_climo_dTwdt = np.zeros((366,len(years),len(water_cities_name_list)))*np.nan
Twater_clean_climo_dTwdt_f = np.zeros((366,len(years),len(water_cities_name_list)))*np.nan
Twater_clean_climo_dTwdt_b = np.zeros((366,len(years),len(water_cities_name_list)))*np.nan


t0  = 3.
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

#%%
for icity,city in enumerate(water_cities_name_list):
    loc_water_city = water_cities_name_list[icity]
    water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
    Twater = water_city_data['Twater'][:,1]

    fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(6,10),sharex=True)
    plt.title(city)

    ax[0].plot(Twater, color=plt.get_cmap('tab20')(icity*2),label=str(city))
    ax[1].plot((Twater_dTdt[:,icity]), color=plt.get_cmap('tab20')(icity*2+1),label=str(city))

    ax[1].plot((np.nanmean(Twater_dTdt[:,icity])+3*np.nanstd((Twater_dTdt[:,icity])))*np.ones(len(time)),'-',color='gray')

#%%

Twater_clean_climo_dTwdt_all = np.zeros((366,len(years),len(water_cities_name_list),3))*np.nan
Twater_clean_climo_dTwdt_all_f = np.zeros((366,len(years),len(water_cities_name_list),3))*np.nan
Twater_clean_climo_dTwdt_all_b = np.zeros((366,len(years),len(water_cities_name_list),3))*np.nan

outliers_climo_dTwdt_all = np.zeros((366,len(years),len(water_cities_name_list),3))*np.nan
outliers_climo_dTwdt_all_f = np.zeros((366,len(years),len(water_cities_name_list),3))*np.nan
outliers_climo_dTwdt_all_b = np.zeros((366,len(years),len(water_cities_name_list),3))*np.nan

outliers_sum = np.zeros((366,len(years),3))
outliers_f_sum = np.zeros((366,len(years),3))
outliers_b_sum = np.zeros((366,len(years),3))

for iyear,year in enumerate(years):
    for icity,city in enumerate(water_cities_name_list):

        imid = int((Nwindow-1)/2)

        Twater = data_Tw[imid,:,iyear,icity]

        dTwater = data_dTwdt[imid,:,iyear,icity]
        clim_dTwdt_all = mean_clim_dTwdt_all
        std_dTwdt_all = std_clim_dTwdt_all

        dTwater_f = data_dTwdt_f[imid,:,iyear,icity]
        clim_dTwdt_all_f = mean_clim_dTwdt_all_f
        std_dTwdt_all_f = std_clim_dTwdt_all_f

        dTwater_b = data_dTwdt_b[imid,:,iyear,icity]
        clim_dTwdt_all_b = mean_clim_dTwdt_all_b
        std_dTwdt_all_b = std_clim_dTwdt_all_b

        for it0,t0 in enumerate(np.arange(1.5,4.5)):
            mask_dTwdt = np.abs(dTwater-clim_dTwdt_all) > t0*std_dTwdt_all
            mask_dTwdt_f = np.abs(dTwater_f-clim_dTwdt_all_f) > t0*std_dTwdt_all_f
            mask_dTwdt_b = np.abs(dTwater_b-clim_dTwdt_all_b) > t0*std_dTwdt_all_b

            mask_clim_dTwdt = mask_dTwdt
            Twater_filtered_dTwdt = Twater.copy()
            Twater_filtered_dTwdt[mask_clim_dTwdt]=np.nan
            Twater_clean_climo_dTwdt_all[:,iyear,icity,it0] = Twater_filtered_dTwdt
            outliers_sum[:,iyear,it0] += (mask_clim_dTwdt.astype(int))

            mask_clim_dTwdt_f = mask_dTwdt_f
            Twater_filtered_dTwdt_f = Twater.copy()
            Twater_filtered_dTwdt_f[mask_clim_dTwdt_f]=np.nan
            Twater_clean_climo_dTwdt_all_f[:,iyear,icity,it0] = Twater_filtered_dTwdt_f
            outliers_f_sum[:,iyear,it0] += (mask_clim_dTwdt_f.astype(int))

            mask_clim_dTwdt_b = mask_dTwdt_b
            Twater_filtered_dTwdt_b = Twater.copy()
            Twater_filtered_dTwdt_b[mask_clim_dTwdt_b]=np.nan
            Twater_clean_climo_dTwdt_all_b[:,iyear,icity,it0] = Twater_filtered_dTwdt_b
            outliers_b_sum[:,iyear,it0] += (mask_clim_dTwdt_b.astype(int))




#%%
for iyear,year in enumerate(years):

    # fig,ax = plt.subplots(nrows=2,ncols=len(water_cities_name_list), gridspec_kw={'height_ratios': [3, 1]},figsize=(12,10),sharex=True,sharey='row')

    fig = plt.figure(figsize=(12,10))

    ax_tmp_x = plt.subplot2grid((3, len(water_cities_name_list)), (0, 0), rowspan=2, colspan=len(water_cities_name_list))
    ax_tmp_y = plt.subplot2grid((3, len(water_cities_name_list)), (2, 0))

    ax1 = plt.subplot2grid((3, len(water_cities_name_list)), (0, 0), rowspan=2, colspan=len(water_cities_name_list), sharex=ax_tmp_x)

    plt.title(years[iyear])

    date=(dt.date(year,1,1)-date_ref).days
    i0 = np.where(time==date)[0][0]
    i1 = i0+365+calendar.isleap(year)

    for icity,city in enumerate(water_cities_name_list):
        ax2 = plt.subplot2grid((3, len(water_cities_name_list)), (2, icity), sharey=ax_tmp_y, sharex=ax_tmp_x)

        loc_water_city = water_cities_name_list[icity]
        water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+loc_water_city+'.npz',allow_pickle='TRUE')
        Twater = water_city_data['Twater'][:,1]
        Twater_clean = Twater_clean_climo_dTwdt_all[:,iyear,icity,0]

        ax1.plot(Twater[i0:i1], color=plt.get_cmap('tab20c')(icity*4),label=water_cities_name_list[icity])
        ax1.plot(Twater_clean, color=plt.get_cmap('tab20c')(icity*4+3),label=water_cities_name_list[icity])


        ax2.plot((Twater_dTdt[i0:i1,icity]), color=plt.get_cmap('tab20c')(icity*4),label=str(city))
        ax2.plot(mean_clim_dTwdt_all,':', color=plt.get_cmap('tab20c')(icity*4+1))

        for it0,t0 in enumerate(np.arange(1.5,4.5)):
            ax2.plot(mean_clim_dTwdt_all+ t0*std_clim_dTwdt_all,'-', color=plt.get_cmap('tab20c')(icity*4+it0+1))
            ax2.plot(mean_clim_dTwdt_all- t0*std_clim_dTwdt_all,'-', color=plt.get_cmap('tab20c')(icity*4+it0+1))


        ax2.set_ylim([-2,2])



