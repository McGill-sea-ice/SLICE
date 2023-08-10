#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:32:36 2022

@author: Amelie
"""
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from netCDF4 import Dataset

#%%
fdir_r = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/CMC_GHRSST/'
fdir_p = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/CMC_GHRSST/'

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1
time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

years = [1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020,2021]

# Daily or monthly time resolution of intial data:
# time_rep = 'daily'
time_rep = 'monthly'

# i,j location to extract Tw (on the 0.2 deg. grid)
i_0p2 = 220
j_0p2 = 518
location = 'Kingston'

# Convert Kelvins to Celsius?
T_in_Celsius = True

# Save data?
save = True

#%%
# Load SST data 1992-2015 (0.2 deg.)
data_0p2 = np.load(fdir_p+'monthly_SST_0p2.npz')
# data_0p2 = np.load(fdir_p+'daily_SST_0p2.npz')
data_0p2 = data_0p2['data']
years_0p2 = years[1:25]

# Get Lat/Lon for 0.2deg. grid
filename = '19920101120000-CMC-L4_GHRSST-SSTfnd-CMC0.2deg-GLOB-v02.0-fv02.0.nc'
ncid = Dataset(fdir_r+'1992/'+filename, 'r')
ncid.set_auto_mask(False)
lat_0p2 = np.squeeze(ncid.variables['lat'][:])
lon_0p2 = np.squeeze(ncid.variables['lon'][:])
mask_0p2 = np.squeeze(ncid.variables['mask'][:])
sel_lat_min, sel_lat_max = 0,70 # Select northern hemisphere
sel_lon_min, sel_lon_max = -180,0
ilatmin = np.where(lat_0p2 == sel_lat_min)[0][0]
ilatmax = np.where(lat_0p2 == sel_lat_max)[0][0]+1
ilonmin = np.where(lon_0p2 == sel_lon_min)[0][0]
ilonmax = np.where(lon_0p2 == sel_lon_max)[0][0]+1
lat_0p2 = lat_0p2[ilatmin:ilatmax]
lon_0p2 = lon_0p2[ilonmin:ilonmax]
mask_0p2 = mask_0p2[ilatmin:ilatmax,ilonmin:ilonmax]

#%%
# Load SST data 2016-2021 (0.1 deg.)
data_0p1 = np.load(fdir_p+'monthly_SST_0p1.npz')
# data_0p1 = np.load(fdir_p+'daily_SST_0p1.npz')
data_0p1 = data_0p1['data']
years_0p1 = years[25:]

# Get Lat/Lon for 0.1deg. grid
filename = '20170101120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc'
ncid = Dataset(fdir_r+'2017/'+filename, 'r')
ncid.set_auto_mask(False)
lat_0p1 = np.squeeze(ncid.variables['lat'][:])
lon_0p1 = np.squeeze(ncid.variables['lon'][:])
mask_0p1 = np.squeeze(ncid.variables['mask'][:])
sel_lat_min, sel_lat_max = 0,70 # Select northern hemisphere
sel_lon_min, sel_lon_max = -180,0
ilatmin = np.where(lat_0p1 == sel_lat_min)[0][0]
ilatmax = np.where(lat_0p1 == sel_lat_max)[0][0]+1
ilonmin = np.where(lon_0p1 == sel_lon_min)[0][0]
ilonmax = np.where(lon_0p1 == sel_lon_max)[0][0]+1
lat_0p1 = lat_0p1[ilatmin:ilatmax]
lon_0p1 = lon_0p1[ilonmin:ilonmax]
mask_0p1 = mask_0p1[ilatmin:ilatmax,ilonmin:ilonmax]

#%%
# Sample or average SSt at 0.1 deg to 0.2 deg. to have only
# one time series for 1992-2021.

sample = False

if sample:
    ind_lat = [np.where(lat_0p1 == lat_0p2[i])[0][0] for i in range(len(lat_0p2))]
    ind_lon = [np.where(lon_0p1 == lon_0p2[i])[0][0] for i in range(len(lon_0p2))]

    lat_0p1_s = lat_0p1[ind_lat]
    lon_0p1_s = lon_0p1[ind_lon]
    mask_0p1_s = mask_0p1[ind_lat,:]
    mask_0p1_s = mask_0p1[:,ind_lon]
    data_0p1_s = data_0p1[:,:,ind_lat,:]
    data_0p1_s = data_0p1_s[:,:,:,ind_lon]
else:
    ind_lat = [np.where(lat_0p1 == lat_0p2[i])[0][0] for i in range(len(lat_0p2))]
    ind_lon = [np.where(lon_0p1 == lon_0p2[i])[0][0] for i in range(len(lon_0p2))]

    data_0p1_s = np.zeros((data_0p1.shape[0],data_0p1.shape[1],data_0p2.shape[2],data_0p2.shape[3]))*np.nan

    for i,ii, in enumerate(ind_lat):
        for j,jj in enumerate(ind_lon):

            if (ii > 0) & (jj > 0) & (ii < (data_0p1.shape[2]-1)) & (jj < (data_0p1.shape[3]-1)):
                w_a = 0.05 *0.05
                w_b = 0.10 *0.05
                w_c = 0.05 *0.05
                w_d = 0.10 *0.05
                w_e = 0.10 *0.10
                w_f = 0.10 *0.05
                w_g = 0.05 *0.05
                w_h = 0.10 *0.05
                w_i = 0.05 *0.05

                w_tot =  w_a+w_b+w_c+w_d+w_e+w_f+w_g+w_h+w_i

                d_a = data_0p1[:,:,ii-1,jj-1]
                d_b = data_0p1[:,:,ii-1,jj  ]
                d_c = data_0p1[:,:,ii-1,jj+1]
                d_d = data_0p1[:,:,ii  ,jj-1]
                d_e = data_0p1[:,:,ii  ,jj  ]
                d_f = data_0p1[:,:,ii  ,jj+1]
                d_g = data_0p1[:,:,ii+1,jj-1]
                d_h = data_0p1[:,:,ii+1,jj  ]
                d_i = data_0p1[:,:,ii+1,jj+1]

                data_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
                                                    w_d*d_d + w_e*d_e + w_f*d_f +
                                                    w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot

            elif (ii == 0) & (jj > 0) & (jj < (data_0p1.shape[3]-1)):
                w_a = 0
                w_b = 0
                w_c = 0
                w_d = 0.10 *0.05
                w_e = 0.10 *0.10
                w_f = 0.10 *0.05
                w_g = 0.05 *0.05
                w_h = 0.10 *0.05
                w_i = 0.05 *0.05

                w_tot =  w_a+w_b+w_c+w_d+w_e+w_f+w_g+w_h+w_i

                d_a = 0
                d_b = 0
                d_c = 0
                d_d = data_0p1[:,:,ii  ,jj-1]
                d_e = data_0p1[:,:,ii  ,jj  ]
                d_f = data_0p1[:,:,ii  ,jj+1]
                d_g = data_0p1[:,:,ii+1,jj-1]
                d_h = data_0p1[:,:,ii+1,jj  ]
                d_i = data_0p1[:,:,ii+1,jj+1]

                data_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
                                                    w_d*d_d + w_e*d_e + w_f*d_f +
                                                    w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot


            elif (jj == 0) & (ii > 0) & (ii < (data_0p1.shape[2]-1)):
                w_a = 0
                w_b = 0.10 *0.05
                w_c = 0.05 *0.05
                w_d = 0
                w_e = 0.10 *0.10
                w_f = 0.10 *0.05
                w_g = 0
                w_h = 0.10 *0.05
                w_i = 0.05 *0.05

                w_tot =  w_a+w_b+w_c+w_d+w_e+w_f+w_g+w_h+w_i

                d_a = 0
                d_b = data_0p1[:,:,ii-1,jj  ]
                d_c = data_0p1[:,:,ii-1,jj+1]
                d_d = 0
                d_e = data_0p1[:,:,ii  ,jj  ]
                d_f = data_0p1[:,:,ii  ,jj+1]
                d_g = 0
                d_h = data_0p1[:,:,ii+1,jj  ]
                d_i = data_0p1[:,:,ii+1,jj+1]

                data_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
                                                    w_d*d_d + w_e*d_e + w_f*d_f +
                                                    w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot


            elif (ii == (data_0p1.shape[2]-1))  & (jj > 0) & (jj < (data_0p1.shape[3]-1)):
                w_a = 0.05 *0.05
                w_b = 0.10 *0.05
                w_c = 0.05 *0.05
                w_d = 0.10 *0.05
                w_e = 0.10 *0.10
                w_f = 0.10 *0.05
                w_g = 0
                w_h = 0
                w_i = 0

                w_tot =  w_a+w_b+w_c+w_d+w_e+w_f+w_g+w_h+w_i

                d_a = data_0p1[:,:,ii-1,jj-1]
                d_b = data_0p1[:,:,ii-1,jj  ]
                d_c = data_0p1[:,:,ii-1,jj+1]
                d_d = data_0p1[:,:,ii  ,jj-1]
                d_e = data_0p1[:,:,ii  ,jj  ]
                d_f = data_0p1[:,:,ii  ,jj+1]
                d_g = 0
                d_h = 0
                d_i = 0

                data_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
                                                    w_d*d_d + w_e*d_e + w_f*d_f +
                                                    w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot



            elif (jj == (data_0p1.shape[3]-1)) & (ii > 0) & (ii < (data_0p1.shape[2]-1)) :
                w_a = 0.05 *0.05
                w_b = 0.10 *0.05
                w_c = 0
                w_d = 0.10 *0.05
                w_e = 0.10 *0.10
                w_f = 0
                w_g = 0.05 *0.05
                w_h = 0.10 *0.05
                w_i = 0

                w_tot = w_a+w_b+w_c+w_d+w_e+w_f+w_g+w_h+w_i

                d_a = data_0p1[:,:,ii-1,jj-1]
                d_b = data_0p1[:,:,ii-1,jj  ]
                d_c = 0
                d_d = data_0p1[:,:,ii  ,jj-1]
                d_e = data_0p1[:,:,ii  ,jj  ]
                d_f = 0
                d_g = data_0p1[:,:,ii+1,jj-1]
                d_h = data_0p1[:,:,ii+1,jj  ]
                d_i = 0

                data_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
                                                    w_d*d_d + w_e*d_e + w_f*d_f +
                                                    w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot

            elif (ii == 0) & (jj == 0):
                w_a = 0
                w_b = 0
                w_c = 0
                w_d = 0
                w_e = 0.10 *0.10
                w_f = 0.10 *0.05
                w_g = 0
                w_h = 0.10 *0.05
                w_i = 0.05 *0.05

                w_tot =  w_a+w_b+w_c+w_d+w_e+w_f+w_g+w_h+w_i

                d_a = 0
                d_b = 0
                d_c = 0
                d_d = 0
                d_e = data_0p1[:,:,ii  ,jj  ]
                d_f = data_0p1[:,:,ii  ,jj+1]
                d_g = 0
                d_h = data_0p1[:,:,ii+1,jj  ]
                d_i = data_0p1[:,:,ii+1,jj+1]

                data_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
                                                    w_d*d_d + w_e*d_e + w_f*d_f +
                                                    w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot


            elif (ii == 0) &  (jj == (data_0p1.shape[3]-1)):
                w_a = 0
                w_b = 0
                w_c = 0
                w_d = 0.10 *0.05
                w_e = 0.10 *0.10
                w_f = 0
                w_g = 0.05 *0.05
                w_h = 0.10 *0.05
                w_i = 0

                w_tot =  w_a+w_b+w_c+w_d+w_e+w_f+w_g+w_h+w_i

                d_a = 0
                d_b = 0
                d_c = 0
                d_d = data_0p1[:,:,ii  ,jj-1]
                d_e = data_0p1[:,:,ii  ,jj  ]
                d_f = 0
                d_g = data_0p1[:,:,ii+1,jj-1]
                d_h = data_0p1[:,:,ii+1,jj  ]
                d_i = 0

                data_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
                                                    w_d*d_d + w_e*d_e + w_f*d_f +
                                                    w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot



            elif (jj == 0) & (ii == (data_0p1.shape[2]-1)) :
                w_a = 0
                w_b = 0.10 *0.05
                w_c = 0.05 *0.05
                w_d = 0
                w_e = 0.10 *0.10
                w_f = 0.10 *0.05
                w_g = 0
                w_h = 0
                w_i = 0

                w_tot =  w_a+w_b+w_c+w_d+w_e+w_f+w_g+w_h+w_i

                d_a = 0
                d_b = data_0p1[:,:,ii-1,jj  ]
                d_c = data_0p1[:,:,ii-1,jj+1]
                d_d = 0
                d_e = data_0p1[:,:,ii  ,jj  ]
                d_f = data_0p1[:,:,ii  ,jj+1]
                d_g = 0
                d_h = 0
                d_i = 0

                data_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
                                                    w_d*d_d + w_e*d_e + w_f*d_f +
                                                    w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot



            elif (ii == (data_0p1.shape[2]-1)) & (jj == (data_0p1.shape[3]-1)):
                w_a = 0.05 *0.05
                w_b = 0.10 *0.05
                w_c = 0
                w_d = 0.10 *0.05
                w_e = 0.10 *0.10
                w_f = 0
                w_g = 0
                w_h = 0
                w_i = 0

                w_tot =  w_a+w_b+w_c+w_d+w_e+w_f+w_g+w_h+w_i

                d_a = data_0p1[:,:,ii-1,jj-1]
                d_b = data_0p1[:,:,ii-1,jj  ]
                d_c = 0
                d_d = data_0p1[:,:,ii  ,jj-1]
                d_e = data_0p1[:,:,ii  ,jj  ]
                d_f = 0
                d_g = 0
                d_h = 0
                d_i = 0

                data_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
                                                    w_d*d_d + w_e*d_e + w_f*d_f +
                                                    w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot

            else:
                print('PROBLEM!!', ii, jj)

#%%
# Then, concatenate both series:
data = np.zeros((data_0p1_s.shape[0]+data_0p2.shape[0],12,data_0p2.shape[2],data_0p2.shape[3]))*np.nan

data[0:data_0p2.shape[0],:,:,:] = data_0p2
data[data_0p2.shape[0]:,:,:,:] = data_0p1_s
years = np.array(years_0p2+years_0p1)

#%%
# Extract Tw at selected location
if time_rep == 'monthly':
    ex_data = np.squeeze(data[:,:,i_0p2,j_0p2])
if time_rep == 'daily':
    ex_data = np.squeeze(data[:,:,:,i_0p2,j_0p2])

#%%
# Then make a daily time series for the same time array as all other predictor variables

daily_data = np.zeros(time.shape)*np.nan

if time_rep == 'monthly':
    for it in range(len(time)):
        date = date_ref + dt.timedelta(days=int(time[it]))
        if date.year in years:
            year_i = np.where(years == date.year)[0][0]
            month_i = int(date.month - 1)
            daily_data[it] = ex_data[year_i,month_i]


if time_rep == 'daily':
    for it in range(len(time)):
        date = date_ref + dt.timedelta(days=int(time[it]))
        if date.year in years:
            year_i = np.where(years == date.year)[0][0]
            month_i = int(date.month - 1)
            day_i = int(date.day - 1)
            daily_data[it] = ex_data[year_i,month_i,day_i]

#%%
# Conversion of Kelvins to Celsius
if T_in_Celsius:
    daily_data = daily_data - 273.15

#%%
# Save data
if save:
    savedir = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/CMC_GHRSST/'
    savename = time_rep+'_CMC_GHRSST_Tw_extracted_at_'+ location
    np.savez(savedir+savename, data = daily_data)











