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

fp = '/Users/Amelie/Dropbox/Postdoc/Projet_Fednav/Data/SLSMC_Twater/'

name_list = ['StLambert','StLouisBridge', 'Cornwall', 'Iroquois','Kingston', 'PortColborneLK8', 'PortWellerLK1']

for loc_name in name_list:
    print(loc_name)

    csv_tmp = read_csv(fp+'Twater_'+loc_name+'.csv',skip=1)
    csv_tmp = clean_csv(csv_tmp,[0,120],[0,csv_tmp.shape[1]])

    # doy=np.zeros((csv_tmp.shape[0]))*np.nan
    # doy[0:61]=np.arange(305,365+1)
    # doy[61:]=np.arange(1,31+28+1)+365
    # csv_tmp[:,0] = doy

    date_ref = dt.date(1900,1,1)
    date_start = dt.date(1980,1,1)
    date_end = dt.date(2021,12,31)
    ndays = (date_end-date_start).days + 1

    years = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]
    time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

    Twater = np.zeros((ndays,2))*np.nan

    for iy,year_i in enumerate(years):
        season_data = csv_tmp[:,iy+1]

        for j in range(csv_tmp.shape[0]):

            if j < 61: year = year_i
            if j >= 61: year = year_i +1

            if j<30: month = 11
            if (j>=30) & (j<61): month = 12
            if (j>=61) & (j<92): month = 1
            if (j>=92): month = 2

            if month == 11: day = j+1
            if month == 12: day = j-29
            if month == 1: day = j-60
            if month == 2: day = j-91

            date=(dt.date(year,month,day)-date_ref).days

            indx = np.where(time == date)[0]

            Twater[indx,0] = date
            Twater[indx,1] = csv_tmp[j,iy+1]

    # ==========================================================================
    np.savez('../../data/processed/Twater_SLSMC/Twater_SLSMC_'+loc_name,
             Twater=Twater,date_ref=date_ref)


