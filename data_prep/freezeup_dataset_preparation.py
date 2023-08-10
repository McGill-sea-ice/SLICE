#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:12:42 2020

@author: Amelie
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

fp = local_path+'slice/data/raw/freezeup_dates_merged_from_SLSMC_and_Charts/'

iroquois_csv = read_csv(fp+'freezeup_iroquois.csv',skip=1)
summerstown_csv = read_csv(fp+'freezeup_summerstown.csv',skip=1)
lakestlawrence_csv = read_csv(fp+'freezeup_lakestlawrence.csv',skip=1)
lakestfrancis_csv = read_csv(fp+'freezeup_lakestfrancis.csv',skip=1)
beauharnois_csv = read_csv(fp+'freezeup_beauharnois.csv',skip=1)
lakestlouis_csv = read_csv(fp+'freezeup_lakestlouis.csv',skip=1)
southshore_csv = read_csv(fp+'freezeup_southshore.csv',skip=1)
montrealport_csv = read_csv(fp+'freezeup_montrealport.csv',skip=1)
varennes_contrecoeur_csv = read_csv(fp+'freezeup_varennes_contrecoeur.csv',skip=1)
contrecoeur_sorel_csv = read_csv(fp+'freezeup_contrecoeur_sorel.csv',skip=1)
lakestpierre_csv = read_csv(fp+'freezeup_lakestpierre.csv',skip=1)


csv_list = [iroquois_csv, summerstown_csv, lakestlawrence_csv ,
              lakestfrancis_csv, beauharnois_csv ,lakestlouis_csv,
              southshore_csv, montrealport_csv, varennes_contrecoeur_csv,
              contrecoeur_sorel_csv, lakestpierre_csv ]

labels_list = ['Iroquois','Summerstown','LakeStLawrence',
                'LakeStFrancisEAST','BeauharnoisCanal','LakeStLouis',
                'SouthShoreCanal','MontrealPort','VarennesContrecoeur',
                'ContrecoeurSorel','LakeStPierre']



for i in range(len(csv_list)):

    loc_name = labels_list[i]
    csv = csv_list[i]

    fi = clean_csv(csv,[0,1,2],[999,888])
    si = clean_csv(csv,[3,4,5],[999,888])
    ci = clean_csv(csv,[9,10,11],[999,888])

    date_ref = dt.date(1900,1,1)
    date_start = dt.date(1980,1,1)
    date_end = dt.date(2021,12,31)
    ndays = (date_end-date_start).days + 1

    time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

    freezeup_fi = np.zeros((ndays,1))*np.nan
    freezeup_si = np.zeros((ndays,1))*np.nan
    freezeup_ci = np.zeros((ndays,1))*np.nan

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

        if np.all([~np.isnan(ci[j,:])]):
            year_ci  = int(ci[j,0])
            month_ci = int(ci[j,1])
            day_ci   = int(ci[j,2])
            date_ci = (dt.date(year_ci,month_ci,day_ci)-date_ref).days
            indx_ci = np.where(time == date_ci)[0]
            freezeup_ci[indx_ci,0] = date_ci


    # ==========================================================================
    np.savez(local_path+'slice/data/processed/freezeup_dates_SLSMC/freezeup_SLSMC_'+loc_name,
             freezeup_fi=freezeup_fi,freezeup_si=freezeup_si,
             freezeup_ci=freezeup_ci,date_ref=date_ref)





