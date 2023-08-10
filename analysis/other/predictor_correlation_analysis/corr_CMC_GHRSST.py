#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:21:23 2022

@author: Amelie
"""

import numpy as np
import datetime as dt
import calendar
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from functions import ncdump
import os
import cartopy.crs as ccrs
from functions import detect_FUD_from_Tw
import statsmodels.api as sm

#%%
def plot_pcolormesh_cartopy(var,gridlats,gridlons,proj=ccrs.PlateCarree()):
    plt.figure()
    ax = plt.axes(projection = proj)
    plt.pcolormesh(gridlons, gridlats, var)
    ax.coastlines()
    # ax.coastlines('110m', alpha=0.1)
    plt.show()

def plot_pcolormesh_cartopy_with_contours(var,contour_var,gridlats,gridlons,proj=ccrs.PlateCarree()):
    plt.figure(figsize=(6,2.5))
    ax = plt.axes(projection = proj)
    plt.pcolormesh(gridlons, gridlats, var, vmin=-0.6,vmax=0.6,cmap=plt.get_cmap('BrBG'))
    ax.coastlines()
    # ax.coastlines('110m', alpha=0.1)
    plt.show()
    plt.colorbar()

    # And black line contour where significant:
    line_c = ax.contour(gridlons, gridlats, contour_var,
                        colors=['black'],levels=[1],
                        transform=ccrs.PlateCarree())

    plt.setp(line_c.collections, visible=True)


def get_3month_vars_from_monthly(vars_in,vars_out,years,date_ref=dt.date(1900,1,1)):
    for iyr, year in enumerate(years):
        for month_3r in range(2,11):
            # -, -, JFM, FMA, MAM, AMJ, MJJ, JJA, JAS, ASO, SON, -
            if month_3r == 0:
                month1 = 11
                month3 = 1
            elif month_3r == 1:
                month1 = 12
                month3 = 2
            else:
                month1 = month_3r-1
                month3 = month1 + 2

            vars_out[:,month_3r,:,:] = np.nanmean(vars_in[:,month1-1:month3,:,:],axis=1)

    return vars_out


#%%
fdir_r = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/CMC_GHRSST/'
fdir_p = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/CMC_GHRSST/'

verbose = False
p_critical = 0.05

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

#%%
# Load SST data 1992-2015 and average every 3 months
data_0p2 = np.load(fdir_p+'monthly_SST_0p2.npz')
data_0p2 = data_0p2['data']
years_0p2 = years[1:25]
# data_threemonthly_0p2 = np.zeros(data_0p2.shape)*np.nan
# data_threemonthly_0p2 = get_3month_vars_from_monthly(data_0p2,data_threemonthly_0p2,years_0p2)
# np.savez(fdir_p+'threemonthly_SST_0p2.npz',data_threemonthly=data_threemonthly_0p2 )
data_threemonthly_0p2 = np.load(fdir_p+'threemonthly_SST_0p2.npz')
data_threemonthly_0p2 = data_threemonthly_0p2['data_threemonthly']

#%%

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
# # Load SST data 2016-2021 and average every 3 months
# data_0p1 = np.load(fdir_p+'monthly_SST_0p1.npz')
# data_0p1 = data_0p1['data']
# years_0p1 = years[25:]
# # data_threemonthly_0p1 = np.zeros(data_0p1.shape)*np.nan
# # data_threemonthly_0p1 = get_3month_vars_from_monthly(data_0p1,data_threemonthly_0p1,years_0p1)
# # np.savez(fdir_p+'threemonthly_SST_0p1.npz',data_threemonthly=data_threemonthly_0p1 )
# data_threemonthly_0p1 = np.load(fdir_p+'threemonthly_SST_0p1.npz')
# data_threemonthly_0p1 = data_threemonthly_0p1['data_threemonthly']

# filename = '20170101120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc'
# ncid = Dataset(fdir_r+'2017/'+filename, 'r')
# ncid.set_auto_mask(False)
# lat_0p1 = np.squeeze(ncid.variables['lat'][:])
# lon_0p1 = np.squeeze(ncid.variables['lon'][:])
# mask_0p1 = np.squeeze(ncid.variables['mask'][:])
# sel_lat_min, sel_lat_max = 0,70 # Select northern hemisphere
# sel_lon_min, sel_lon_max = -180,0
# ilatmin = np.where(lat_0p1 == sel_lat_min)[0][0]
# ilatmax = np.where(lat_0p1 == sel_lat_max)[0][0]+1
# ilonmin = np.where(lon_0p1 == sel_lon_min)[0][0]
# ilonmax = np.where(lon_0p1 == sel_lon_max)[0][0]+1
# lat_0p1 = lat_0p1[ilatmin:ilatmax]
# lon_0p1 = lon_0p1[ilonmin:ilonmax]
# mask_0p1 = mask_0p1[ilatmin:ilatmax,ilonmin:ilonmax]

#%%
# Sample or average SSt at 0.1 deg to 0.2 deg. to have only
# one time series for 1992-2021.

sample = False

# if sample:
#     ind_lat = [np.where(lat_0p1 == lat_0p2[i])[0][0] for i in range(len(lat_0p2))]
#     ind_lon = [np.where(lon_0p1 == lon_0p2[i])[0][0] for i in range(len(lon_0p2))]

#     lat_0p1_s = lat_0p1[ind_lat]
#     lon_0p1_s = lon_0p1[ind_lon]
#     mask_0p1_s = mask_0p1[ind_lat,:]
#     mask_0p1_s = mask_0p1[:,ind_lon]
#     data_threemonthly_0p1_s = data_threemonthly_0p1[:,:,ind_lat,:]
#     data_threemonthly_0p1_s = data_threemonthly_0p1_s[:,:,:,ind_lon]
#     data_0p1_s = data_0p1[:,:,ind_lat,:]
#     data_0p1_s = data_0p1_s[:,:,:,ind_lon]
# else:
#     ind_lat = [np.where(lat_0p1 == lat_0p2[i])[0][0] for i in range(len(lat_0p2))]
#     ind_lon = [np.where(lon_0p1 == lon_0p2[i])[0][0] for i in range(len(lon_0p2))]

#     data_threemonthly_0p1_s = np.zeros((data_threemonthly_0p1.shape[0],data_threemonthly_0p1.shape[1],data_threemonthly_0p2.shape[2],data_threemonthly_0p2.shape[3]))*np.nan
#     data_0p1_s = np.zeros((data_0p1.shape[0],data_0p1.shape[1],data_0p2.shape[2],data_0p2.shape[3]))*np.nan

#     for i,ii, in enumerate(ind_lat):
#         for j,jj in enumerate(ind_lon):

#             if (ii > 0) & (jj > 0) & (ii < (data_threemonthly_0p1.shape[2]-1)) & (jj < (data_threemonthly_0p1.shape[3]-1)):
#                 w_a = 0.05 *0.05
#                 w_b = 0.10 *0.05
#                 w_c = 0.05 *0.05
#                 w_d = 0.10 *0.05
#                 w_e = 0.10 *0.10
#                 w_f = 0.10 *0.05
#                 w_g = 0.05 *0.05
#                 w_h = 0.10 *0.05
#                 w_i = 0.05 *0.05

#                 w_tot =  w_a+w_b+w_c+w_d+w_e+w_f+w_g+w_h+w_i

#                 d_a = data_threemonthly_0p1[:,:,ii-1,jj-1]
#                 d_b = data_threemonthly_0p1[:,:,ii-1,jj  ]
#                 d_c = data_threemonthly_0p1[:,:,ii-1,jj+1]
#                 d_d = data_threemonthly_0p1[:,:,ii  ,jj-1]
#                 d_e = data_threemonthly_0p1[:,:,ii  ,jj  ]
#                 d_f = data_threemonthly_0p1[:,:,ii  ,jj+1]
#                 d_g = data_threemonthly_0p1[:,:,ii+1,jj-1]
#                 d_h = data_threemonthly_0p1[:,:,ii+1,jj  ]
#                 d_i = data_threemonthly_0p1[:,:,ii+1,jj+1]

#                 data_threemonthly_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
#                                                     w_d*d_d + w_e*d_e + w_f*d_f +
#                                                     w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot

#                 d_a = data_0p1[:,:,ii-1,jj-1]
#                 d_b = data_0p1[:,:,ii-1,jj  ]
#                 d_c = data_0p1[:,:,ii-1,jj+1]
#                 d_d = data_0p1[:,:,ii  ,jj-1]
#                 d_e = data_0p1[:,:,ii  ,jj  ]
#                 d_f = data_0p1[:,:,ii  ,jj+1]
#                 d_g = data_0p1[:,:,ii+1,jj-1]
#                 d_h = data_0p1[:,:,ii+1,jj  ]
#                 d_i = data_0p1[:,:,ii+1,jj+1]

#                 data_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
#                                                     w_d*d_d + w_e*d_e + w_f*d_f +
#                                                     w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot

#             elif (ii == 0) & (jj > 0) & (jj < (data_threemonthly_0p1.shape[3]-1)):
#                 w_a = 0
#                 w_b = 0
#                 w_c = 0
#                 w_d = 0.10 *0.05
#                 w_e = 0.10 *0.10
#                 w_f = 0.10 *0.05
#                 w_g = 0.05 *0.05
#                 w_h = 0.10 *0.05
#                 w_i = 0.05 *0.05

#                 w_tot =  w_a+w_b+w_c+w_d+w_e+w_f+w_g+w_h+w_i

#                 d_a = 0
#                 d_b = 0
#                 d_c = 0
#                 d_d = data_threemonthly_0p1[:,:,ii  ,jj-1]
#                 d_e = data_threemonthly_0p1[:,:,ii  ,jj  ]
#                 d_f = data_threemonthly_0p1[:,:,ii  ,jj+1]
#                 d_g = data_threemonthly_0p1[:,:,ii+1,jj-1]
#                 d_h = data_threemonthly_0p1[:,:,ii+1,jj  ]
#                 d_i = data_threemonthly_0p1[:,:,ii+1,jj+1]

#                 data_threemonthly_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
#                                                     w_d*d_d + w_e*d_e + w_f*d_f +
#                                                     w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot

#                 d_a = 0
#                 d_b = 0
#                 d_c = 0
#                 d_d = data_0p1[:,:,ii  ,jj-1]
#                 d_e = data_0p1[:,:,ii  ,jj  ]
#                 d_f = data_0p1[:,:,ii  ,jj+1]
#                 d_g = data_0p1[:,:,ii+1,jj-1]
#                 d_h = data_0p1[:,:,ii+1,jj  ]
#                 d_i = data_0p1[:,:,ii+1,jj+1]

#                 data_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
#                                                     w_d*d_d + w_e*d_e + w_f*d_f +
#                                                     w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot


#             elif (jj == 0) & (ii > 0) & (ii < (data_threemonthly_0p1.shape[2]-1)):
#                 w_a = 0
#                 w_b = 0.10 *0.05
#                 w_c = 0.05 *0.05
#                 w_d = 0
#                 w_e = 0.10 *0.10
#                 w_f = 0.10 *0.05
#                 w_g = 0
#                 w_h = 0.10 *0.05
#                 w_i = 0.05 *0.05

#                 w_tot =  w_a+w_b+w_c+w_d+w_e+w_f+w_g+w_h+w_i

#                 d_a = 0
#                 d_b = data_threemonthly_0p1[:,:,ii-1,jj  ]
#                 d_c = data_threemonthly_0p1[:,:,ii-1,jj+1]
#                 d_d = 0
#                 d_e = data_threemonthly_0p1[:,:,ii  ,jj  ]
#                 d_f = data_threemonthly_0p1[:,:,ii  ,jj+1]
#                 d_g = 0
#                 d_h = data_threemonthly_0p1[:,:,ii+1,jj  ]
#                 d_i = data_threemonthly_0p1[:,:,ii+1,jj+1]

#                 data_threemonthly_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
#                                                     w_d*d_d + w_e*d_e + w_f*d_f +
#                                                     w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot

#                 d_a = 0
#                 d_b = data_0p1[:,:,ii-1,jj  ]
#                 d_c = data_0p1[:,:,ii-1,jj+1]
#                 d_d = 0
#                 d_e = data_0p1[:,:,ii  ,jj  ]
#                 d_f = data_0p1[:,:,ii  ,jj+1]
#                 d_g = 0
#                 d_h = data_0p1[:,:,ii+1,jj  ]
#                 d_i = data_0p1[:,:,ii+1,jj+1]

#                 data_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
#                                                     w_d*d_d + w_e*d_e + w_f*d_f +
#                                                     w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot


#             elif (ii == (data_threemonthly_0p1.shape[2]-1))  & (jj > 0) & (jj < (data_threemonthly_0p1.shape[3]-1)):
#                 w_a = 0.05 *0.05
#                 w_b = 0.10 *0.05
#                 w_c = 0.05 *0.05
#                 w_d = 0.10 *0.05
#                 w_e = 0.10 *0.10
#                 w_f = 0.10 *0.05
#                 w_g = 0
#                 w_h = 0
#                 w_i = 0

#                 w_tot =  w_a+w_b+w_c+w_d+w_e+w_f+w_g+w_h+w_i

#                 d_a = data_threemonthly_0p1[:,:,ii-1,jj-1]
#                 d_b = data_threemonthly_0p1[:,:,ii-1,jj  ]
#                 d_c = data_threemonthly_0p1[:,:,ii-1,jj+1]
#                 d_d = data_threemonthly_0p1[:,:,ii  ,jj-1]
#                 d_e = data_threemonthly_0p1[:,:,ii  ,jj  ]
#                 d_f = data_threemonthly_0p1[:,:,ii  ,jj+1]
#                 d_g = 0
#                 d_h = 0
#                 d_i = 0

#                 data_threemonthly_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
#                                                     w_d*d_d + w_e*d_e + w_f*d_f +
#                                                     w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot

#                 d_a = data_0p1[:,:,ii-1,jj-1]
#                 d_b = data_0p1[:,:,ii-1,jj  ]
#                 d_c = data_0p1[:,:,ii-1,jj+1]
#                 d_d = data_0p1[:,:,ii  ,jj-1]
#                 d_e = data_0p1[:,:,ii  ,jj  ]
#                 d_f = data_0p1[:,:,ii  ,jj+1]
#                 d_g = 0
#                 d_h = 0
#                 d_i = 0

#                 data_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
#                                                     w_d*d_d + w_e*d_e + w_f*d_f +
#                                                     w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot



#             elif (jj == (data_threemonthly_0p1.shape[3]-1)) & (ii > 0) & (ii < (data_threemonthly_0p1.shape[2]-1)) :
#                 w_a = 0.05 *0.05
#                 w_b = 0.10 *0.05
#                 w_c = 0
#                 w_d = 0.10 *0.05
#                 w_e = 0.10 *0.10
#                 w_f = 0
#                 w_g = 0.05 *0.05
#                 w_h = 0.10 *0.05
#                 w_i = 0

#                 w_tot = w_a+w_b+w_c+w_d+w_e+w_f+w_g+w_h+w_i

#                 d_a = data_threemonthly_0p1[:,:,ii-1,jj-1]
#                 d_b = data_threemonthly_0p1[:,:,ii-1,jj  ]
#                 d_c = 0
#                 d_d = data_threemonthly_0p1[:,:,ii  ,jj-1]
#                 d_e = data_threemonthly_0p1[:,:,ii  ,jj  ]
#                 d_f = 0
#                 d_g = data_threemonthly_0p1[:,:,ii+1,jj-1]
#                 d_h = data_threemonthly_0p1[:,:,ii+1,jj  ]
#                 d_i = 0

#                 data_threemonthly_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
#                                                     w_d*d_d + w_e*d_e + w_f*d_f +
#                                                     w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot

#                 d_a = data_0p1[:,:,ii-1,jj-1]
#                 d_b = data_0p1[:,:,ii-1,jj  ]
#                 d_c = 0
#                 d_d = data_0p1[:,:,ii  ,jj-1]
#                 d_e = data_0p1[:,:,ii  ,jj  ]
#                 d_f = 0
#                 d_g = data_0p1[:,:,ii+1,jj-1]
#                 d_h = data_0p1[:,:,ii+1,jj  ]
#                 d_i = 0

#                 data_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
#                                                     w_d*d_d + w_e*d_e + w_f*d_f +
#                                                     w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot

#             elif (ii == 0) & (jj == 0):
#                 w_a = 0
#                 w_b = 0
#                 w_c = 0
#                 w_d = 0
#                 w_e = 0.10 *0.10
#                 w_f = 0.10 *0.05
#                 w_g = 0
#                 w_h = 0.10 *0.05
#                 w_i = 0.05 *0.05

#                 w_tot =  w_a+w_b+w_c+w_d+w_e+w_f+w_g+w_h+w_i

#                 d_a = 0
#                 d_b = 0
#                 d_c = 0
#                 d_d = 0
#                 d_e = data_threemonthly_0p1[:,:,ii  ,jj  ]
#                 d_f = data_threemonthly_0p1[:,:,ii  ,jj+1]
#                 d_g = 0
#                 d_h = data_threemonthly_0p1[:,:,ii+1,jj  ]
#                 d_i = data_threemonthly_0p1[:,:,ii+1,jj+1]

#                 data_threemonthly_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
#                                                     w_d*d_d + w_e*d_e + w_f*d_f +
#                                                     w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot

#                 d_a = 0
#                 d_b = 0
#                 d_c = 0
#                 d_d = 0
#                 d_e = data_0p1[:,:,ii  ,jj  ]
#                 d_f = data_0p1[:,:,ii  ,jj+1]
#                 d_g = 0
#                 d_h = data_0p1[:,:,ii+1,jj  ]
#                 d_i = data_0p1[:,:,ii+1,jj+1]

#                 data_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
#                                                     w_d*d_d + w_e*d_e + w_f*d_f +
#                                                     w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot


#             elif (ii == 0) &  (jj == (data_threemonthly_0p1.shape[3]-1)):
#                 w_a = 0
#                 w_b = 0
#                 w_c = 0
#                 w_d = 0.10 *0.05
#                 w_e = 0.10 *0.10
#                 w_f = 0
#                 w_g = 0.05 *0.05
#                 w_h = 0.10 *0.05
#                 w_i = 0

#                 w_tot =  w_a+w_b+w_c+w_d+w_e+w_f+w_g+w_h+w_i

#                 d_a = 0
#                 d_b = 0
#                 d_c = 0
#                 d_d = data_threemonthly_0p1[:,:,ii  ,jj-1]
#                 d_e = data_threemonthly_0p1[:,:,ii  ,jj  ]
#                 d_f = 0
#                 d_g = data_threemonthly_0p1[:,:,ii+1,jj-1]
#                 d_h = data_threemonthly_0p1[:,:,ii+1,jj  ]
#                 d_i = 0

#                 data_threemonthly_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
#                                                     w_d*d_d + w_e*d_e + w_f*d_f +
#                                                     w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot

#                 d_a = 0
#                 d_b = 0
#                 d_c = 0
#                 d_d = data_0p1[:,:,ii  ,jj-1]
#                 d_e = data_0p1[:,:,ii  ,jj  ]
#                 d_f = 0
#                 d_g = data_0p1[:,:,ii+1,jj-1]
#                 d_h = data_0p1[:,:,ii+1,jj  ]
#                 d_i = 0

#                 data_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
#                                                     w_d*d_d + w_e*d_e + w_f*d_f +
#                                                     w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot



#             elif (jj == 0) & (ii == (data_threemonthly_0p1.shape[2]-1)) :
#                 w_a = 0
#                 w_b = 0.10 *0.05
#                 w_c = 0.05 *0.05
#                 w_d = 0
#                 w_e = 0.10 *0.10
#                 w_f = 0.10 *0.05
#                 w_g = 0
#                 w_h = 0
#                 w_i = 0

#                 w_tot =  w_a+w_b+w_c+w_d+w_e+w_f+w_g+w_h+w_i

#                 d_a = 0
#                 d_b = data_threemonthly_0p1[:,:,ii-1,jj  ]
#                 d_c = data_threemonthly_0p1[:,:,ii-1,jj+1]
#                 d_d = 0
#                 d_e = data_threemonthly_0p1[:,:,ii  ,jj  ]
#                 d_f = data_threemonthly_0p1[:,:,ii  ,jj+1]
#                 d_g = 0
#                 d_h = 0
#                 d_i = 0

#                 data_threemonthly_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
#                                                     w_d*d_d + w_e*d_e + w_f*d_f +
#                                                     w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot

#                 d_a = 0
#                 d_b = data_0p1[:,:,ii-1,jj  ]
#                 d_c = data_0p1[:,:,ii-1,jj+1]
#                 d_d = 0
#                 d_e = data_0p1[:,:,ii  ,jj  ]
#                 d_f = data_0p1[:,:,ii  ,jj+1]
#                 d_g = 0
#                 d_h = 0
#                 d_i = 0

#                 data_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
#                                                     w_d*d_d + w_e*d_e + w_f*d_f +
#                                                     w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot



#             elif (ii == (data_threemonthly_0p1.shape[2]-1)) & (jj == (data_threemonthly_0p1.shape[3]-1)):
#                 w_a = 0.05 *0.05
#                 w_b = 0.10 *0.05
#                 w_c = 0
#                 w_d = 0.10 *0.05
#                 w_e = 0.10 *0.10
#                 w_f = 0
#                 w_g = 0
#                 w_h = 0
#                 w_i = 0

#                 w_tot =  w_a+w_b+w_c+w_d+w_e+w_f+w_g+w_h+w_i

#                 d_a = data_threemonthly_0p1[:,:,ii-1,jj-1]
#                 d_b = data_threemonthly_0p1[:,:,ii-1,jj  ]
#                 d_c = 0
#                 d_d = data_threemonthly_0p1[:,:,ii  ,jj-1]
#                 d_e = data_threemonthly_0p1[:,:,ii  ,jj  ]
#                 d_f = 0
#                 d_g = 0
#                 d_h = 0
#                 d_i = 0

#                 data_threemonthly_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
#                                                     w_d*d_d + w_e*d_e + w_f*d_f +
#                                                     w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot

#                 d_a = data_0p1[:,:,ii-1,jj-1]
#                 d_b = data_0p1[:,:,ii-1,jj  ]
#                 d_c = 0
#                 d_d = data_0p1[:,:,ii  ,jj-1]
#                 d_e = data_0p1[:,:,ii  ,jj  ]
#                 d_f = 0
#                 d_g = 0
#                 d_h = 0
#                 d_i = 0

#                 data_0p1_s[:,:,i,j] = (w_a*d_a + w_b*d_b + w_c*d_c +
#                                                     w_d*d_d + w_e*d_e + w_f*d_f +
#                                                     w_g*d_g + w_h*d_h + w_i*d_i   )/w_tot

#             else:
#                 print('PROBLEM!!', ii, jj)

#%%
# Then, concatenate both series:
# data = np.zeros((data_0p1_s.shape[0]+data_0p2.shape[0],12,data_0p2.shape[2],data_0p2.shape[3]))*np.nan
# data_threemonthly = np.zeros((data_threemonthly_0p1_s.shape[0]+data_threemonthly_0p2.shape[0],12,data_threemonthly_0p2.shape[2],data_threemonthly_0p2.shape[3]))*np.nan

# data[0:data_0p2.shape[0],:,:,:] = data_0p2
# data[data_0p2.shape[0]:,:,:,:] = data_0p1_s

# data_threemonthly[0:data_threemonthly_0p2.shape[0],:,:,:] = data_threemonthly_0p2
# data_threemonthly[data_threemonthly_0p2.shape[0]:,:,:,:] = data_threemonthly_0p1_s

#%%
# Load Twater and FUD data
fp_p_Twater = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/'
Twater_loc_list = ['Longueuil_updated']
station_type = 'cities'
freezeup_opt = 1
freezeup_doy, Twater = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False)
freezeup_doy[np.where(years == 2020)] = np.nan

# Average (and round) FUD from all locations:
avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
avg_freezeup_doy = np.round(avg_freezeup_doy)

# Average Twater from all locations:
avg_Twater = np.nanmean(Twater,axis=1)
avg_Twater[14269:14329] = 0.
avg_Twater_vars = np.expand_dims(avg_Twater, axis=1)
avg_Twater_varnames = ['Avg. Twater']

#%%
years = np.array(years[1:])
avg_freezeup_doy = avg_freezeup_doy[1:]

#%%
# COMPUTE THREE MONTHLY CORRELATIONS
# rsqr_sst_FUD = np.zeros((12,data_threemonthly.shape[2],data_threemonthly.shape[3]))*np.nan
# signif_sst_FUD = np.zeros((12,data_threemonthly.shape[2],data_threemonthly.shape[3]))
# corr_sst_FUD = np.zeros((12,data_threemonthly.shape[2],data_threemonthly.shape[3]))*np.nan

# yr_start = 0
# yr_end = 30

# for im in range(2,11):
#     print(im)
#     for i in range(data_threemonthly.shape[2]):
#         for j in range(data_threemonthly.shape[3]):

#             if ~np.all(np.isnan(data_threemonthly[yr_start:yr_end,im,i,j])):
#                 x = np.squeeze(data_threemonthly[yr_start:yr_end,im,i,j])
#                 model = sm.OLS(avg_freezeup_doy[yr_start:yr_end], sm.add_constant(x, has_constant='skip'), missing='drop').fit()
#                 rsqr_sst_FUD[im,i,j]= model.rsquared
#                 if model.f_pvalue <= p_critical:
#                     signif_sst_FUD[im,i,j] = 1
#                 if len(model.params) > 1:
#                     if model.params[1] < 0:
#                         corr_sst_FUD[im,i,j] = np.sqrt(model.rsquared)*-1
#                     if model.params[1] >= 0:
#                         corr_sst_FUD[im,i,j] = np.sqrt(model.rsquared)


# np.savez(fdir_p+'corr_SST_FUD_threemonthly_SST_0.1deg_' + 'sampled'*sample + 'averaged'*(~sample)+'_'+str(years[yr_start])+'_'+str(years[yr_end-1]),
#           rsqr_sst_FUD=rsqr_sst_FUD,
#           corr_sst_FUD=corr_sst_FUD,
#           signif_sst_FUD=signif_sst_FUD)

# # PLOT THREE MONTHLY CORRELATION MAPS
# data_tmp = np.load(fdir_p+'corr_SST_FUD_threemonthly_SST_0.1deg_' + 'sampled'*sample + 'averaged'*(not sample)+'_'+str(years[yr_start])+'_'+str(years[yr_end-1])+'.npz')
# rsqr_sst_FUD=data_tmp['rsqr_sst_FUD']
# signif_sst_FUD=data_tmp['signif_sst_FUD']
# corr_sst_FUD=data_tmp['corr_sst_FUD']

# fig,ax = plt.subplots(nrows=3,ncols=3,figsize=(18,8),sharex=True,sharey=True,subplot_kw={'projection': ccrs.PlateCarree()})
# title_str = ['-', '-', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', '-']
# iax = [np.nan, np.nan, 0, 1 ,2 , 0, 1, 2, 0, 1, 2, np.nan]
# jax = [np.nan, np.nan, 0, 0 ,0 , 1, 1, 1, 2, 2, 2, np.nan]
# proj = ccrs.PlateCarree()
# for im in range(2,11):
#     var = corr_sst_FUD[im,:,:]
#     c = signif_sst_FUD[im,:,:]
#     # plot_pcolormesh_cartopy_with_contours_with_axes(ax[iax[im],jax[im]],var,c,lat_0p2,lon_0p2)
#     ax[iax[im],jax[im]].pcolormesh(lon_0p2, lat_0p2, var, vmin=-0.6,vmax=0.6,cmap=plt.get_cmap('BrBG'))
#     ax[iax[im],jax[im]].coastlines()
#     # And black line contour where significant:
#     line_c = ax[iax[im],jax[im]].contour(lon_0p2, lat_0p2, c,
#                         colors=['black'],levels=[1],
#                         transform=ccrs.PlateCarree())

#     plt.setp(line_c.collections, visible=True)

#     ax[iax[im],jax[im]].set_title(title_str[im]+' '+str(years[yr_start])+'-'+str(years[yr_end-1]))
#     if sample:
#         plt.suptitle('0.1 deg Sampled every 0.2 deg')
#     else:
#         plt.suptitle('0.1 deg Averaged at 0.2 deg')


#%%
# COMPUTE MONTHLY CORRELATIONS
# rsqr_sst_FUD = np.zeros((12,data.shape[2],data.shape[3]))*np.nan
# signif_sst_FUD = np.zeros((12,data.shape[2],data.shape[3]))
# corr_sst_FUD = np.zeros((12,data.shape[2],data.shape[3]))*np.nan

yr_start = 0
yr_end = 30

# for im in range(12):
#     print(im)
#     for i in range(data.shape[2]):
#         for j in range(data.shape[3]):

#             if ~np.all(np.isnan(data[yr_start:yr_end,im,i,j])):
#                 x = np.squeeze(data[yr_start:yr_end,im,i,j])
#                 model = sm.OLS(avg_freezeup_doy[yr_start:yr_end], sm.add_constant(x, has_constant='skip'), missing='drop').fit()
#                 rsqr_sst_FUD[im,i,j]= model.rsquared
#                 if model.f_pvalue <= p_critical:
#                     signif_sst_FUD[im,i,j] = 1
#                 if len(model.params) > 1:
#                     if model.params[1] < 0:
#                         corr_sst_FUD[im,i,j] = np.sqrt(model.rsquared)*-1
#                     if model.params[1] >= 0:
#                         corr_sst_FUD[im,i,j] = np.sqrt(model.rsquared)

# np.savez(fdir_p+'corr_SST_FUD_monthly_SST_0.1deg_' + 'sampled'*sample + 'averaged'*(not sample)+'_'+str(years[yr_start])+'_'+str(years[yr_end-1]),
#           rsqr_sst_FUD=rsqr_sst_FUD,
#           corr_sst_FUD=corr_sst_FUD,
#           signif_sst_FUD=signif_sst_FUD)

#%%
# PLOT MONTHLY CORRELATION MAPS
data_tmp = np.load(fdir_p+'corr_SST_FUD_monthly_SST_0.1deg_' + 'sampled'*sample + 'averaged'*(not sample)+'_'+str(years[yr_start])+'_'+str(years[yr_end-1])+'.npz')
rsqr_sst_FUD=data_tmp['rsqr_sst_FUD']
signif_sst_FUD=data_tmp['signif_sst_FUD']
corr_sst_FUD=data_tmp['corr_sst_FUD']

# fig,ax = plt.subplots(nrows=3,ncols=3,figsize=(18,8),sharex=True,sharey=True,subplot_kw={'projection': ccrs.PlateCarree()})
# title_str = ['J', 'F', 'M', 'A', 'M', 'J', 'Jl', 'A', 'S', 'O', 'N', 'D']
# iax = [np.nan, np.nan, 0, 1 ,2 , 0, 1, 2, 0, 1, 2, np.nan]
# jax = [np.nan, np.nan, 0, 0 ,0 , 1, 1, 1, 2, 2, 2, np.nan]
# proj = ccrs.PlateCarree()
# for im in range(2,11):
#     var = corr_sst_FUD[im,:,:]
#     c = signif_sst_FUD[im,:,:]
#     # plot_pcolormesh_cartopy_with_contours_with_axes(ax[iax[im],jax[im]],var,c,lat_0p2,lon_0p2)
#     ax[iax[im],jax[im]].pcolormesh(lon_0p2, lat_0p2, var, vmin=-0.6,vmax=0.6,cmap=plt.get_cmap('BrBG'))
#     ax[iax[im],jax[im]].coastlines()
#     # And black line contour where significant:
#     line_c = ax[iax[im],jax[im]].contour(lon_0p2, lat_0p2, c,
#                         colors=['black'],levels=[1],
#                         transform=ccrs.PlateCarree())

#     plt.setp(line_c.collections, visible=True)

#     ax[iax[im],jax[im]].set_title(title_str[im]+' '+str(years[yr_start])+'-'+str(years[yr_end-1]))
#     if sample:
#         plt.suptitle('0.1 deg Sampled every 0.2 deg')
#     else:
#         plt.suptitle('0.1 deg Averaged at 0.2 deg')


fig,ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
title_str = 'May'
proj = ccrs.PlateCarree()
im = 4
var = corr_sst_FUD[im,:,:]
c = signif_sst_FUD[im,:,:]
# plot_pcolormesh_cartopy_with_contours_with_axes(ax[iax[im],jax[im]],var,c,lat_0p2,lon_0p2)
ax.pcolormesh(lon_0p2, lat_0p2, var, vmin=-0.6,vmax=0.6,cmap=plt.get_cmap('BrBG'))
ax.coastlines()
# And black line contour where significant:
line_c = ax.contour(lon_0p2, lat_0p2, c,
                    colors=['black'],levels=[1],
                    transform=ccrs.PlateCarree())

plt.setp(line_c.collections, visible=True)

ax.set_title(title_str+' '+str(years[yr_start])+'-'+str(years[yr_end-1]))
if sample:
    plt.suptitle('0.1 deg Sampled every 0.2 deg')
else:
    plt.suptitle('0.1 deg Averaged at 0.2 deg')




