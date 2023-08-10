#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 15:45:40 2020

@author: Amelie
"""
# from functions import read_csv
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

def clean_csv(arr,rows,columns,nan_id=np.nan,return_type=float):

    arr = arr[rows[0]:rows[1],columns[0]:columns[1]]
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

fp = '/Users/Amelie/Dropbox/Postdoc/Projet_Fednav/Data/Twater_cities/'


date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

station_list = ['DesBaillets']
years = [2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]


for l,loc_name in enumerate(station_list):
    print(loc_name)

    Twater = np.zeros((ndays,2))*np.nan

    for year in years:
        print(year)

        csv_tmp = read_csv(fp+loc_name+'/Tw_'+loc_name+'.csv',skip=1)
        csv_tmp = np.delete(csv_tmp,4,1)

        year_csv = csv_tmp[np.where(csv_tmp[:,0] == year)[0],:].copy()

        if calendar.isleap(year):
            day_arr = [31,29,31,30,31,30,31,31,30,31,30,31]
        else:
            day_arr = [31,28,31,30,31,30,31,31,30,31,30,31]


        for month in range(12):
            for day in range(day_arr[month]):

                month_csv = year_csv[np.where(year_csv[:,1] == month+1)[0],:].copy()
                day_csv = month_csv[np.where(month_csv[:,2] == day+1)[0],:].copy()

                date=(dt.date(year,month+1,day+1)-date_ref).days
                indx = np.where(time == date)[0]

                if len(day_csv) == 0:
                    Tw = np.nan
                    date = np.nan
                else:
                    Tw = np.nanmean(day_csv[:,4]).copy()

                Twater[indx,0] = date
                Twater[indx,1] = Tw

    # ==========================================================================
    np.savez('../../data/processed/Twater_cities/Twater_cities_'+loc_name,
              Twater=Twater,date_ref=date_ref)


#%%
# xplot = np.arange(len(csv_tmp[:,4]))

# plt.figure()
# plt.plot(xplot,csv_tmp[:,4],linestyle='-')
# plt.plot(xplot[0::4],csv_tmp[0::4,4],'.')
# plt.plot(xplot[1::4],csv_tmp[1::4,4],'.')
# plt.plot(xplot[2::4],csv_tmp[2::4,4],'.')
# plt.plot(xplot[3::4],csv_tmp[3::4,4],'.')



