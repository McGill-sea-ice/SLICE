#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:13:09 2021

@author: Amelie
"""
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from cdo import Cdo
cdo = Cdo()
cdo = Cdo(tempdir='/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/temp_files/temp/') #python

#%%
path = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/weather_ERA5/'
basename = 'ERA5_1991_2021_'
region = 'D'

output_dir = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/ERA5_test/region'+region+'/'

start_year = 1991
end_year = 2021

month_start = 1
month_end = 13

#%%
if region == 'A':
    # REGION A: ST_LAWRENCE
    rlon1, rlat1 = -75.25, 44.75
    rlon2, rlat2 = -73.25, 46.00
elif region == 'B':
    # REGION B: LAKE ONTARIO
    rlon1, rlat1 = -77.25, 43.25
    rlon2, rlat2 = -75.50, 44.50
elif region == 'C':
    # REGION C: OTTAWA RIVER + MONTREAL
    rlon1, rlat1 = -77.25, 44.75
    rlon2, rlat2 = -73.25, 46.00
elif region == 'D':
    # REGION D: ALL
    rlon1, rlat1 = -77.25, 43.25
    rlon2, rlat2 = -73.25, 46.00
elif region == 'E':
    # REGION E: OTTAWA RIVER ONLY
    rlon1, rlat1 = -77.25, 44.75
    rlon2, rlat2 = -75.25, 46.00


#%%
# Variable dictionaries

u10 = { 'var':'u10',
        'savename' :'10m_u_component_of_wind',
        'bundle': 'basics'
      }
v10 = { 'var':'v10',
        'savename' :'10m_v_component_of_wind',
        'bundle': 'basics'
      }
t2m = { 'var':'t2m',
        'savename' :'2m_temperature',
        'bundle': 'basics'
      }
d2m = { 'var':'d2m',
        'savename' :'2m_dewpoint_temperature',
        'bundle': 'precip'
      }
licd = { 'var':'licd',
         'savename' :'lake_ice_depth',
         'bundle': 'lakes'
      }
lict = { 'var':'lict',
         'savename' :'lake_ice_temperature',
         'bundle': 'lakes'
      }
ltlt = { 'var':'ltlt',
         'savename' :'lake_total_layer_temperature',
         'bundle': 'lakes'
      }
msl = { 'var':'msl',
        'savename' :'mean_sea_level_pressure',
        'bundle': 'basics'
      }
ro = { 'var':'ro',
       'savename' :'runoff',
       'bundle': 'precip'
      }
siconc = { 'var':'siconc',
           'savename' :'sea_ice_cover',
           'bundle': 'precip'
      }
sst = { 'var':'sst',
        'savename' :'sea_surface_temperature',
        'bundle': 'basics'
      }
sf = { 'var':'sf',
       'savename' :'snowfall',
       'bundle': 'precip'
      }
smlt = { 'var':'smlt',
         'savename' :'snowmelt',
         'bundle': 'precip'
      }
tcc = { 'var':'tcc',
        'savename' :'total_cloud_cover',
        'bundle': 'rates'
      }
tp = { 'var':'tp',
       'savename' :'total_precipitation',
       'bundle': 'basics'
      }

dics_list = [tcc, msl, sf, tp, ro, smlt, ltlt,
             siconc, sst,
             ]


#%%

for dic in dics_list:
    var = dic['var']
    bundle = dic['bundle']
    print(var, bundle, dic['savename'])


    for year in range(start_year, end_year):
        print(year)
        os.chdir(output_dir)

        for month in range(month_start,month_end):
            os.chdir(output_dir)

            print(month)
            # '01' instead of '1'
            month = str(month).rjust(2, '0')

            # eg. 1979-01
            subdirectory = "{}-{}".format(year, month)
            if not os.path.isdir(subdirectory):
                os.mkdir(subdirectory)

            os.chdir(subdirectory)

            if (int(month) == 12) | (int(month) < 3):
                season = 'DJF'
            elif (int(month) >= 3) & (int(month) < 6):
                season = 'MAM'
            elif (int(month) >= 6) & (int(month) < 9):
                season = 'JJA'
            else:
                season = 'SON_2'

            file = path+basename+dic['bundle']+'_'+season+'.nc'
            outfile = 'ERA5_'+dic['savename']+'_'+str(year)+month+'.nc'

            v = cdo.sellonlatbox(rlon1,rlon2,rlat1,rlat2,
                                     input=cdo.selyear(year,
                                         input=cdo.selmonth(int(month),
                                               input = cdo.selname(dic['var'],
                                                   input=file)))
                                  # , returnArray = dic['var'])
                                   , output = outfile)


#%%
# from functions import ncdump

# ncid = Dataset(outfile, 'r')
# v10_resave = ncid.variables['v10'][:]

# ncid2 = Dataset('/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/ERA5/regionD/1991-01/ERA5_10m_v_component_of_wind_199101.nc', 'r')
# v10_new = ncid2.variables['v10'][:]
# # ncdump(ncid2)

