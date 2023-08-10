#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:12:42 2020

@author: Amelie Bouchat (amelie.bouchat@mail.mcgill.ca)

ICE INFORMATION, IN ORDER OF QUALITY:
1) first_available_chart < last_water_on_chart < first_ice_on_chart, here uncertainty is nb. days between last_water_on_chart and first_ice_on_chart
1) first_available_chart = last_water_on_chart < first_ice_on_chart, here uncertainty is nb. days between last_water_on_chart and first_ice_on_chart

2) first_available_chart = first_ice_on_chart, last_water_on_chart = nan, here uncertainty is maximal

NO ICE INFORMATION:
first_available_chart = last_water_on_chart, first_ice_on_chart = nan
first_available_chart = first_ice_on_chart = last_water_on_chart = nan

"""
#%%
local_path = '/storage/amelie/'
# local_path = '/Volumes/SeagateUSB/McGill/Postdoc/'

#%%
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath(local_path +'slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)


import numpy as np
import datetime as dt
import calendar

import matplotlib.pyplot as plt

# ==========================================================================
def read_csv(file_path,skip=0,delimiter=','):

    f = open(file_path, 'r')
    data = np.genfromtxt(f,skip_header=skip,delimiter=',')
    f.close()

    return data

def clean_csv(arr,columns,nan_id,return_type=float):

    arr = arr[:,columns]
    arr[np.isin(arr,nan_id)] = np.nan

    clean_arr = arr

    return clean_arr.astype(return_type)


def day_of_year_arr(date_arr,flip_date_new_year):

    doy_arr=np.zeros((date_arr.shape[0]))*np.nan

    for i in range(date_arr.shape[0]):
        if np.all(~np.isnan(date_arr[i,:])):
            doy_arr[i] = (dt.datetime(int(date_arr[i,0]),int(date_arr[i,1]),int(date_arr[i,2])) - dt.datetime(int(date_arr[i,0]), 1, 1)).days + 1
            if calendar.isleap(int(date_arr[i,0])):
                doy_arr[i] -= 1 # Remove 1 for leap years so that
                                # e.g. Dec 1st is always DOY = 335

    if flip_date_new_year: doy_arr[doy_arr < 200] = doy_arr[doy_arr < 200]  + 365

    return doy_arr

# ==========================================================================

fp = local_path+'slice/data/raw/freezeup_charts_finer_analysis/'

loc_list = ['Candiac','LaPrairie','Lasalle','Longueuil','MontrealOldPort','MontrealPort','StLambert']

for iloc,loc in enumerate(loc_list):
    csv = read_csv(fp+loc+'_freezeup.csv',skip=1)

    first_available_chart = csv[:,0:3]
    last_water_on_chart = csv[:,3:6]
    first_ice_on_chart = csv[:,6:9]

    date_ref = dt.date(1900,1,1)
    date_start = dt.date(1980,1,1)
    date_end = dt.date(2021,12,31)
    ndays = (date_end-date_start).days + 1

    time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)
    freezeup_ci = np.zeros((ndays,2))*np.nan

    for iyr in range(csv.shape[0]):
        if ~np.isnan(first_available_chart[iyr,0]): # We have charts for this year
            if ~np.isnan(first_ice_on_chart[iyr,0]): # We have ice information on charts

                date_fc = dt.date(int(first_available_chart[iyr,2]),int(first_available_chart[iyr,1]),int(first_available_chart[iyr,0]))
                date_fi = dt.date(int(first_ice_on_chart[iyr,2]),int(first_ice_on_chart[iyr,1]),int(first_ice_on_chart[iyr,0]))

                if ~np.isnan(last_water_on_chart[iyr,0]): # We have a chart with water before we have a chart with ice
                    date_lw = dt.date(int(last_water_on_chart[iyr,2]),int(last_water_on_chart[iyr,1]),int(last_water_on_chart[iyr,0]))
                    freezeup_dt = (date_fi-date_lw).days
                    indx_ci = np.where(time == (date_fi-date_ref).days)[0]
                    freezeup_ci[indx_ci,0] = (date_fi-date_ref).days
                    freezeup_ci[indx_ci,1] = freezeup_dt

                else: # The first chart we have already has ice.
                    if (date_fi-date_fc).days == 0: # (this should be zero to confirm the above)
                        indx_ci = np.where(time == (date_fi-date_ref).days)[0]
                        freezeup_ci[indx_ci,0] = (date_fi-date_ref).days # freezeup timing uncertainty is left as nan because we have no estimate
                    else:
                        print('PROBLEM')


    # ==========================================================================
    np.savez(local_path+'slice/data/processed/freezeup_dates_charts/freezeup_charts_'+loc,
              freezeup_ci=freezeup_ci,date_ref=date_ref)

