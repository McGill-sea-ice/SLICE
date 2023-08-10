#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:18:47 2020

@author: Amelie
"""

import numpy as np

import datetime as dt
import calendar

import matplotlib.pyplot as plt
from functions import clean_csv

# ==========================================================================
fp = '/Users/Amelie/Dropbox/Postdoc/Projet_Fednav/Data/weather_NCEI/'
# name_list = ['Kingston','Massena', 'MontrealDorval', 'LacStPierre','Nicolet']
name_list = ['MontrealMcTavish','MontrealMirabel','MontrealPET','StHubert']

for loc_name in name_list:

    print(loc_name)

    # Read csv file, line by line
    f = open(fp+ loc_name +'.csv', "r")
    lines = f.readlines()
    var_keys = lines[0].split('\n')[0].split(',')
    f.close()


    # Select variables from csv file from the following list:
    # TEMP - Mean temperature for the day in degrees Fahrenheit to tenths. Missing = 9999.9
    # DEWP - Mean dew point for the day in degrees Fahrenheit to tenths. Missing = 9999.9
    # SLP - Mean sea level pressure for the day in millibars to tenths. Missing = 9999.9
    # STP - Mean station pressure for the day in millibars to tenths. Missing = 9999.9
    # VISIB - Mean visibility for the day in miles to tenths. Missing = 999.9
    # WDSP - Mean wind speed for the day in knots to tenths.  Missing = 999.9
    # MXSPD - Maximum sustained wind speed reported for the day in knots to tenths. Missing = 999.
    # GUST - Maximum wind gust reported for the day in knots to tenths.  Missing = 999.9
    # MAX - Maximum temperature reported during the day in Fahrenheit to tenths. Missing = 9999.9
    # MIN - Minimum temperature reported during the day in Fahrenheit to tenths. Missing = 9999.9
    # PRCP - Total precipitation (rain and/or melted snow) reported during the day in inches
    #       and hundredths; will usually not end with the midnight observation (i.e. may include
    #       latter part of previous day). “0” indicates no measurable precipitation (includes a trace).Missing = 99.99
    select_vars = ['MAX','MIN','TEMP','DEWP','PRCP','SLP','WDSP']
    missing_values = [9999.9,9999.9,9999.9,9999.9,99.99,9999.9,999.9]
    var_indx = [np.where([var_keys[i] == select_vars[j] for i in range(len(var_keys))])[0][0] for j in range(len(select_vars)) ]


    # Expand date in year, month, day values,
    # Select the chosen variables (columns),
    # And put in temporary array.
    csv_raw = np.zeros((len(lines)-1,len(select_vars)+7))*np.nan

    for i,line in enumerate(lines[1:]):
        l = lines[i+1].split('\n')[0].split(',')
        l.pop(2) # remove the extra column created by the state/country in the 'NAME' variable

        year = np.int(l[5].split('-')[0])
        month = np.int(l[5].split('-')[1])
        day = np.int(l[5].split('-')[2])

        csv_raw[i,0] = np.int(l[0]) # Station number
        csv_raw[i,1] = np.float(l[2]) # Latitude
        csv_raw[i,2] = np.float(l[3]) # Longitude
        csv_raw[i,3] = np.float(l[4]) # Elevation
        csv_raw[i,4] = year
        csv_raw[i,5] = month
        csv_raw[i,6] = day
        csv_raw[i,7:] = np.array([l[k] for k in var_indx]).astype(float)


    # Find the different station numbers and their index in the array.
    station_list = np.unique(csv_raw[:,0]).astype(int)
    station_indx = [np.where(csv_raw[:,0] == station_list[i])[0] for i in range(station_list.size)]


    # Find the minimum/maximum year covered by all stations.
    year_max = 0
    year_min = 2021
    for i in range(station_list.size):
            year_max = np.nanmax([year_max,np.nanmax(csv_raw[station_indx[i],4])])
            year_min = np.nanmin([year_min,np.nanmin(csv_raw[station_indx[i],4])])

    date_min = dt.date(int(year_min),1,1)
    date_max = dt.date(int(year_max),12,31)

    ndays = (date_max-date_min).days+1

    # date_ref = dt.date(1900,1,1)
    # date = dt.date(1980,1,1)
    # tdays = (date-date_ref).days
    # date_end = date_ref+dt.timedelta(days=tdays)
    # print(date_end)


    # Separate the original data per station,
    # Assign a position based on the number of days since the beginning of the minimum year
    # Cahnge the date format to days since 1900-01-01
    csv_all_stations = np.zeros((station_list.size,ndays,csv_raw.shape[1]-2))*np.nan
    date_ref_station = dt.date(int(year_min),1,1)
    date_ref = dt.date(1900,1,1)

    for i in range(len(station_list)):
        station_date = [dt.date(int(csv_raw[station_indx[i],:][j,4]),int(csv_raw[station_indx[i],:][j,5]),int(csv_raw[station_indx[i],:][j,6])) for j in range(csv_raw[station_indx[i],:].shape[0])]
        station_date_indx = [(station_date[k]-date_ref_station).days for k in range(len(station_date))]

        station_data = np.delete(csv_raw[station_indx[i],:],obj=[5,6],axis=1)
        station_data[:,4] = [(station_date[l]-date_ref).days for l in range(len(station_date))]

        csv_all_stations[i,station_date_indx,:] = station_data


    # Convert missing values to nan
    for i in range(len(station_list)):
        for j in range(len(select_vars)):
            mv_indx = np.where(csv_all_stations[i,:,j+5] == missing_values[j])[0]
            csv_all_stations[i,mv_indx,j+5] = np.nan


    # Merge data from all stations in the area
    # For each day in the span of the minimum/maximum year:
    #  - if only one station is available on that day, take that station
    #  - if multiple stations are available, average their data
    #  - if no data is available, leave no value.
    csv_merged = np.zeros((ndays,csv_all_stations.shape[2]-4))*np.nan

    for d in range(ndays):
        if np.sum(~np.isnan(csv_all_stations[:,d,0])) != 0:
            indx = np.where(~np.isnan(csv_all_stations[:,d,0]))[0]
            tmp = np.nanmean(csv_all_stations[indx,d,:],0)
            csv_merged[d,:] = tmp[4:]



    # Select data only for the time period of interest
    date_start = dt.date(1980,1,1)
    date_end = dt.date(2021,12,31)
    ndays = (date_end-date_start).days + 1

    time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)
    t_indx = [np.where(csv_merged[:,0] == time[i])[0] for i in range(time.shape[0])]

    weather_data = np.zeros((ndays,csv_merged.shape[1]))*np.nan

    for i in range(ndays):
        if np.where(csv_merged[:,0] == time[i])[0].size > 0:
            weather_data[i,:] = csv_merged[np.where(csv_merged[:,0] == time[i])[0][0],:]

    # and note any missing data...
    missing_indx = np.where(np.isnan(weather_data[:,0]))[0]
    missing_dates = [date_ref+dt.timedelta(days=int(time[missing_indx[i]])) for i in range(missing_indx.size)]


    # ==========================================================================
    np.savez('../../data/processed/weather_NCEI/weather_NCEI_'+loc_name,
             weather_data=weather_data,select_vars=select_vars,date_ref=date_ref,missing_dates = missing_dates)

