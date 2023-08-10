#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 19:19:22 2021

@author: Amelie
"""
import numpy as np
import scipy.stats as sp

import datetime as dt
import calendar

import matplotlib.pyplot as plt
from functions import linear_fit,linear_fit_no_intercept

# ==========================================================================
# LOAD DATA
fp = '../../data/weather_NCEI/'

# This is the baseline data set:
loc_weather1 = 'MontrealDorval'
file_data1 = np.load(fp+'weather_NCEI_'+loc_weather1+'.npz',allow_pickle='TRUE')
weather_data1 = file_data1['weather_data']

# Patch missing data from baseline for vars in vars_indx
# using this data set:
var_indx2 = [1,2,3,4,6,7]
loc_weather2 = 'MontrealPET'
file_data2 = np.load(fp+'weather_NCEI_'+loc_weather2+'.npz',allow_pickle='TRUE')
weather_data2 = file_data2['weather_data']

# Patch missing data from baseline for vars in vars_indx
# using this data set:
var_indx3 = [4,5,6]
loc_weather3 = 'MontrealMcTavish'
file_data3 = np.load(fp+'weather_NCEI_'+loc_weather3+'.npz',allow_pickle='TRUE')
weather_data3 = file_data3['weather_data']

#%% ==========================================================================
# GET LINEAR RELATIONSHIP BETWEEN DATA SETS
date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2020,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

# Define time range to get linear fit:
year_start = 1997
year_end = 2020

date_start = (dt.date(year_start,1,1)-date_ref).days
date_end = (dt.date(year_end,12,31)-date_ref).days

istart = int(np.where(time == date_start)[0][0])
iend = int(np.where(time == date_end)[0][0])


for i in var_indx2:
    v1 = weather_data1[:,i]
    v2 = weather_data2[:,i]

    if i == 5:
        v1[5450:5551]=np.nan # This part of the data set seems erroneous... remove before forming interpolant
        istart_tmp = 5000 # Change start year temporarily to also patch this erroneous section.
        lincoeff, Rsqr = linear_fit_no_intercept(v2[istart_tmp:iend+1],v1[istart_tmp:iend+1])
        temp_fit = lincoeff[0]*v2
        print(lincoeff,Rsqr)
        temp_fit[:istart_tmp] = np.nan
        temp_fit[iend:] = np.nan
    else:
        lincoeff, Rsqr = linear_fit(v2[istart:iend+1],v1[istart:iend+1])
        temp_fit = lincoeff[0]*v2 + lincoeff[1]
        print(lincoeff,Rsqr)
        temp_fit[:istart] = np.nan
        temp_fit[iend:] = np.nan

    v1[np.isnan(v1)]=temp_fit[np.isnan(v1)]


for j in var_indx3:
    v1 = weather_data1[:,j]
    v3 = weather_data3[:,j]

    if j == 5:
        v1[9132:9497]=np.nan # This part of the data set seems erroneous...
        v1[5450:5551]=np.nan # This part of the data set seems erroneous...
        istart_tmp = 5000 # Change start year temporarily to also patch the erroneous sections.
        lincoeff, Rsqr = linear_fit_no_intercept(v3[istart_tmp:iend+1],v1[istart_tmp:iend+1])
        temp_fit = lincoeff[0]*v3
        print(lincoeff,Rsqr)
        temp_fit[:istart_tmp] = np.nan
        temp_fit[iend:] = np.nan
    else:
        lincoeff, Rsqr = linear_fit(v3[istart:iend+1],v1[istart:iend+1])
        temp_fit = lincoeff[0]*v3 + lincoeff[1]
        print(lincoeff,Rsqr)
        temp_fit[:istart] = np.nan
        temp_fit[iend:] = np.nan

    v1[np.isnan(v1)]=temp_fit[np.isnan(v1)]


#%%
j=1
plt.figure();
# weather_data2[9976:10235,j]=np.nan
plt.plot(weather_data1[:,j])
plt.plot(weather_data2[:,j])
xl=[]
xt=[]
for i in np.arange(1980,2020,2):
    ndays = (dt.date(i+1,1,1)-date_ref).days
    if (len(np.where(time==ndays)[0]) != 0):
        xl.append(np.where(time==ndays)[0][0])
        xt.append(str(i+1))
plt.xticks(xl,xt)


# lincoeff, Rsqr = linear_fit_no_intercept(weather_data3[istart:iend+1,j],weather_data1[istart:iend+1,j])
lincoeff, Rsqr = linear_fit(weather_data2[istart:iend+1,j],weather_data1[istart:iend+1,j])
print(lincoeff,Rsqr)
plt.figure()
plt.scatter(weather_data1[istart:iend+1,1],weather_data2[istart:iend+1,1])
plt.figure()
plt.scatter(weather_data1[:istart,1],weather_data2[:istart,1])
np.sum(np.isnan(weather_data1[:istart,j]))

#%%
j=3
plt.figure()
plt.plot(weather_data3[:,j])

plt.figure()
plt.plot(weather_data2[:,j])

plt.figure()
plt.plot(weather_data1[:,j])

#%% ==========================================================================
# np.savez(fp+'weather_NCEI_'+loc_weather1+loc_weather2+loc_weather3+'merged',
#           weather_data=weather_data1,select_vars = file_data1['select_vars'])



#%%



def fill_gaps(Twater_in, ndays = 7, fill_type = 'linear'):

    # mask_tmp is true if there is no data (i.e. Tw is nan):
    mask_tmp = np.isnan(Twater_in)

    mask_gap = mask_tmp.copy()
    mask_gap[:] = False

    Twater_out = Twater_in.copy()

    for im in range(1,mask_gap.size):

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
                if sum_m < ndays:
                    mask_gap[istart:iend] = True

                    if fill_type == 'linear':
                        # Fill small gap with linear interpolation
                        slope = (Twater_out[iend]-Twater_out[istart-1])/(sum_m+1)
                        Twater_out[istart:iend] = Twater_out[istart-1] + slope*(np.arange(sum_m)+1)

                    else:
                        print('Problem! ''fill_type'' not defined...')

                sum_m = 0 # Put back sum to zero


    return Twater_out, mask_gap



Ta_filled, mask_fill = fill_gaps(weather_data1[:,3])

