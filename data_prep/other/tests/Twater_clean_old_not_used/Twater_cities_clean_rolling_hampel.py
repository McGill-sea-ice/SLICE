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
import scipy.stats as sp

import datetime as dt
import calendar

import matplotlib.pyplot as plt


# ==========================================================================
save = True

years = [1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020
            ]
water_cities_name_list = ['Longueuil_preclean','Atwater_preclean','DesBaillets_preclean','Candiac_preclean']

# water_cities_name_list = ['Candiac','Longueuil','Atwater','DesBaillets']
# water_cities_name_list = ['Longueuil','Atwater']
# water_cities_name_list = ['Candiac']
# water_cities_name_list = ['Longueuil']
# water_cities_name_list = ['DesBaillets']

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

plt.figure()
plt.plot(Twater_clean[:,0],Twater_clean[:,1],'.')

# percentage of all data points that are filtered after the first filter
print(mask_1/N)
# percentage of all data points that are filtered after the second filter
print(mask_2/N)
# percentage of all data points that are filtered after two passes of the filter
print(mask_tot/N)


#%%
if save:

    for icity in range(len(water_cities_name_list)):
        Twater_city = np.zeros((Twater_clean.shape[0],2))*np.nan
        Twater_city[:,0] = time
        Twater_city[:,1] = Twater_clean[:,icity]
        save_name = water_cities_name_list[icity][:-9]
        np.savez('../../data/Twater_cities/Twater_cities_'+save_name+'_clean',
               Twater=Twater,date_ref=date_ref)


#%%
plt.figure()
plt.plot(Twater_clean[:,0],Twater_clean[:,1],'.')

plt.figure()
plt.plot(Twater_clean[:10000,0],Twater_clean[:10000,1],'.')

plt.figure()
plt.plot(Twater_clean[10000:,0],Twater_clean[10000:,1],'.')

#%%

for iyear,year in enumerate(years):
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,5),sharex=True)
    plt.title(years[iyear])

    date=(dt.date(year,1,1)-date_ref).days
    i0 = np.where(time==date)[0][0]
    i1 = i0+365+calendar.isleap(year)

    ax.plot(Twater_clean[i0:i1,0], color=plt.get_cmap('tab20')(0*2),label='Longueuil')
    ax.plot(Twater_clean[i0:i1,1], color=plt.get_cmap('tab20')(1*2),label='Atwater')
    ax.plot(Twater_clean[i0:i1,2], color=plt.get_cmap('tab20')(2*2),label='Des Baillets')
    ax.plot(Twater_clean[i0:i1,3], color=plt.get_cmap('tab20')(3*2),label='Candiac')

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
    Twater_tmp[:8800] = np.nan


    # Twater_tmp[:11220] = np.nan
    # Twater_tmp[13600:] = np.nan


    Twater_all[icity*Twater_clean.shape[0]:Twater_clean.shape[0]*(icity+1),0] = Twater_tmp
    Twater_all[icity*Twater_clean.shape[0]:Twater_clean.shape[0]*(icity+1),1] = icity
    Twater_all[icity*Twater_clean.shape[0]:Twater_clean.shape[0]*(icity+1),2] = year_tmp
    Twater_all[icity*Twater_clean.shape[0]:Twater_clean.shape[0]*(icity+1),3] = season_tmp

Twater_all = Twater_all[~np.isnan(Twater_all[:,0])]


#%%
import scipy.stats as stats
import pandas as pd
import researchpy as rp

Tw_df = pd.DataFrame({'Twater':Twater_all[:,0],
                     'Plant':Twater_all[:,1],
                     'Year':Twater_all[:,2],
                     'Season':Twater_all[:,3]})

Tw_df['Plant'].replace({0: 'Longueuil', 1: 'Atwater', 2: 'DesBaillets', 3:'Candiac'}, inplace= True)
Tw_df['Season'].replace({0: 'Spring', 1: 'Summer', 2: 'Fall', 3:'Winter'}, inplace= True)

rp.summary_cont(Tw_df['Twater'].groupby(Tw_df['Plant']))


# TO CHECK NORMALITY OF DATA - ONE OF THE MAIN HYPOTHESES FOR USING THE F-TEST IN ANOVA
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# import scipy.stats as stats


# model = ols('Twater ~ C(Plant)', data=Tw_df).fit()
# aov_table = sm.stats.anova_lm(model, typ=2)
# print(aov_table)
# def anova_table(aov):
#     aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']

#     aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])

#     aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])

#     cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
#     aov = aov[cols]
#     return aov

# print(anova_table(aov_table))
# print(stats.shapiro(model.resid))
# print(stats.normaltest(model.resid))
# print(stats.kstest(model.resid, 'norm'))

# result = stats.anderson(model.resid)
# print(f'Test statistic: {result.statistic: .4f}')
# for i in range(len(result.critical_values)):
#     sig, crit = result.significance_level[i], result.critical_values[i]

#     if result.statistic < result.critical_values[i]:
#         print(f"At {sig}% significance,{result.statistic: .4f} <{result.critical_values[i]: .4f} data looks normal (fail to reject H0)")
#     else:
#         print(f"At {sig}% significance,{result.statistic: .4f} >{result.critical_values[i]: .4f} data does not look normal (reject H0)")




A = Tw_df['Twater'][Tw_df['Plant'] == 'Longueuil']
B = Tw_df['Twater'][Tw_df['Plant'] == 'Atwater']
C = Tw_df['Twater'][Tw_df['Plant'] == 'DesBaillets']
D = Tw_df['Twater'][Tw_df['Plant'] == 'Candiac']


print("Longueuil - Atwater - DesBaillets - Candiac")
F, pval_F = stats.f_oneway(A,B,C,D)
if pval_F < 0.05:
    print("F-TEST (1 WAY-ANOVA): "+ "%4.3f"%pval_F +"\n Reject NULL hypothesis - Significant differences exist between groups.")
if pval_F > 0.05:
    print("F-TEST (1 WAY-ANOVA): "+ "%4.3f"%pval_F +"\n Accept NULL hypothesis - No significant difference between groups.")
H, pval_H = stats.kruskal(A,B,C,D)
if pval_H < 0.05:
    print("KRUSKAL-WALLIS TEST: "+ "%4.3f"%pval_H +"\n Reject NULL hypothesis - Significant differences exist between groups.")
if pval_H > 0.05:
    print("KRUSKAL-WALLIS TEST: "+ "%4.3f"%pval_H +"\n Accept NULL hypothesis - No significant difference between groups.")


print(' ')
print("Longueuil - Atwater - DesBaillets")
F, pval_F = stats.f_oneway(A,B,C)
if pval_F < 0.05:
    print("F-TEST (1 WAY-ANOVA): "+ "%4.3f"%pval_F +"\n Reject NULL hypothesis - Significant differences exist between groups.")
if pval_F > 0.05:
    print("F-TEST (1 WAY-ANOVA): "+ "%4.3f"%pval_F +"\n Accept NULL hypothesis - No significant difference between groups.")
H, pval_H = stats.kruskal(A,B,C)
if pval_H < 0.05:
    print("KRUSKAL-WALLIS TEST: "+ "%4.3f"%pval_H +"\n Reject NULL hypothesis - Significant differences exist between groups.")
if pval_H > 0.05:
    print("KRUSKAL-WALLIS TEST: "+ "%4.3f"%pval_H +"\n Accept NULL hypothesis - No significant difference between groups.")


print(' ')
print("Longueuil - Atwater - Candiac")
F, pval_F = stats.f_oneway(A,B,D)
if pval_F < 0.05:
    print("F-TEST (1 WAY-ANOVA): "+ "%4.3f"%pval_F +"\n Reject NULL hypothesis - Significant differences exist between groups.")
if pval_F > 0.05:
    print("F-TEST (1 WAY-ANOVA): "+ "%4.3f"%pval_F +"\n Accept NULL hypothesis - No significant difference between groups.")
H, pval_H = stats.kruskal(A,B,D)
if pval_H < 0.05:
    print("KRUSKAL-WALLIS TEST: "+ "%4.3f"%pval_H +"\n Reject NULL hypothesis - Significant differences exist between groups.")
if pval_H > 0.05:
    print("KRUSKAL-WALLIS TEST: "+ "%4.3f"%pval_H +"\n Accept NULL hypothesis - No significant difference between groups.")


print(' ')
print("Longueuil - Atwater")
F, pval_F = stats.f_oneway(A,B)
if pval_F < 0.05:
    print("F-TEST (1 WAY-ANOVA): "+ "%4.3f"%pval_F +"\n Reject NULL hypothesis - Significant differences exist between groups.")
if pval_F > 0.05:
    print("F-TEST (1 WAY-ANOVA): "+ "%4.3f"%pval_F +"\n Accept NULL hypothesis - No significant difference between groups.")
H, pval_H = stats.kruskal(A,B)
if pval_H < 0.05:
    print("KRUSKAL-WALLIS TEST: "+ "%4.3f"%pval_H +"\n Reject NULL hypothesis - Significant differences exist between groups.")
if pval_H > 0.05:
    print("KRUSKAL-WALLIS TEST: "+ "%4.3f"%pval_H +"\n Accept NULL hypothesis - No significant difference between groups.")


print(' ')
print("DesBaillets - Atwater")
F, pval_F = stats.f_oneway(B,C)
if pval_F < 0.05:
    print("F-TEST (1 WAY-ANOVA): "+ "%4.3f"%pval_F +"\n Reject NULL hypothesis - Significant differences exist between groups.")
if pval_F > 0.05:
    print("F-TEST (1 WAY-ANOVA): "+ "%4.3f"%pval_F +"\n Accept NULL hypothesis - No significant difference between groups.")
H, pval_H = stats.kruskal(B,C)
if pval_H < 0.05:
    print("KRUSKAL-WALLIS TEST: "+ "%4.3f"%pval_H +"\n Reject NULL hypothesis - Significant differences exist between groups.")
if pval_H > 0.05:
    print("KRUSKAL-WALLIS TEST: "+ "%4.3f"%pval_H +"\n Accept NULL hypothesis - No significant difference between groups.")


print(' ')
print("Longueuil - DesBaillets")
F, pval_F = stats.f_oneway(A,C)
if pval_F < 0.05:
    print("F-TEST (1 WAY-ANOVA): "+ "%4.3f"%pval_F +"\n Reject NULL hypothesis - Significant differences exist between groups.")
if pval_F > 0.05:
    print("F-TEST (1 WAY-ANOVA): "+ "%4.3f"%pval_F +"\n Accept NULL hypothesis - No significant difference between groups.")
H, pval_H = stats.kruskal(A,C)
if pval_H < 0.05:
    print("KRUSKAL-WALLIS TEST: "+ "%4.3f"%pval_H +"\n Reject NULL hypothesis - Significant differences exist between groups.")
if pval_H > 0.05:
    print("KRUSKAL-WALLIS TEST: "+ "%4.3f"%pval_H +"\n Accept NULL hypothesis - No significant difference between groups.")




