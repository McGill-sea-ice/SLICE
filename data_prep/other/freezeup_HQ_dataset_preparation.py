#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:12:42 2020

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

fp = '/Users/Amelie/Dropbox/Postdoc/Projet_Fednav/Data/ice_HydroQuebec/'

beauharnois_csv = read_csv(fp+'freezeup_beauharnois_HQ.csv',skip=1)
csv_list = [beauharnois_csv ]
labels_list = ['BeauharnoisCanal']



for i in range(len(csv_list)):

    loc_name = labels_list[i]
    csv = csv_list[i]

    fi = clean_csv(csv,[0,1,2],[np.nan])
    si = clean_csv(csv,[4,5,6],[np.nan])
    li = clean_csv(csv,[8,9,10],[np.nan])

    date_ref = dt.date(1900,1,1)
    date_start = dt.date(1960,1,1)
    date_end = dt.date(2021,12,31)
    ndays = (date_end-date_start).days + 1

    time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

    freezeup_fi = np.zeros((ndays,1))*np.nan
    freezeup_si = np.zeros((ndays,1))*np.nan
    freezeup_li = np.zeros((ndays,1))*np.nan

    for j in range(fi.shape[0]):
        if np.all([~np.isnan(fi[j,:])]):
            year_fi  = int(fi[j,0])
            month_fi = int(fi[j,1])
            day_fi   = int(fi[j,2])
            date_fi = (dt.date(year_fi,month_fi,day_fi)-date_ref).days
            indx_fi = np.where(time == date_fi)[0]
            freezeup_fi[indx_fi,0] = date_fi

        if np.all([~np.isnan(si[j,:])]):
            year_si  = int(si[j,0])
            month_si = int(si[j,1])
            day_si   = int(si[j,2])
            date_si = (dt.date(year_si,month_si,day_si)-date_ref).days
            indx_si = np.where(time == date_si)[0]
            freezeup_si[indx_si,0] = date_si

        if np.all([~np.isnan(li[j,:])]):
            year_li  = int(li[j,0])
            month_li = int(li[j,1])
            day_li   = int(li[j,2])
            date_li = (dt.date(year_li,month_li,day_li)-date_ref).days
            indx_li = np.where(time == date_li)[0]
            freezeup_li[indx_li,0] = date_li


    # ==========================================================================
    np.savez('../../data/processed/freezeup_dates_HQ/freezeup_HQ_'+loc_name,
             freezeup_fi=freezeup_fi,freezeup_si=freezeup_si,
             freezeup_li=freezeup_li,date_ref=date_ref)





