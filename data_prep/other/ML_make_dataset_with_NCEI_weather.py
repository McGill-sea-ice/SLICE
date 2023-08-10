#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 18:48:46 2021

@author: Amelie
"""

import numpy as np
import datetime as dt

# ==========================================================================

water_SLSMC = []
# water_ECCC_name_list = ['Lasallepatchlinear']
# water_city_list = []
water_ECCC_name_list = []
water_city_list = ['DesBaillets_cleaned_filled']
freezeup_name_list = ['SouthShoreCanal']
weather_name_list = ['MontrealDorvalMontrealPETMontrealMcTavishmerged']

fp = '../../data/'

n1=len(water_SLSMC)
n2=len(water_ECCC_name_list)
n3=len(water_city_list)
n4=len(freezeup_name_list)*3
n5=len(weather_name_list)*7

ntot = n1+n2+n3+n4+n5+1

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2020,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

# Initialize array for dataset
ds = np.zeros((ndays,ntot))
labels = np.chararray(ntot, itemsize=100)

ds[:,0] = time
labels[0] = 'Days since '+str(date_ref)

c = 1
save_name = ''

if n1 > 0:
    for i,iloc in enumerate(water_SLSMC):
        water_SLSMC_data = np.load(fp+'Twater_SLSMC/Twater_SLSMC_'+iloc+'.npz',allow_pickle='TRUE')
        Twater_SLSMC = water_SLSMC_data['Twater']

        ds[:,c] = Twater_SLSMC[:,1]
        labels[c]  ='Twater SLSMC [deg.C] - '+iloc
        c += 1
        save_name += iloc + '_'

if n2 > 0:
    for i,iloc in enumerate(water_ECCC_name_list):
        water_ECCC_data = np.load(fp+'Twater_ECCC/Twater_ECCC_'+iloc+'.npz',allow_pickle='TRUE')
        Twater_ECCC = water_ECCC_data['Twater']

        ds[:,c] = Twater_ECCC[:,1]
        labels[c]  ='Twater ECCC [deg.C] - '+iloc
        c += 1
        save_name += iloc + '_'

if n3 > 0:
    for i,iloc in enumerate(water_city_list):
        water_city_data = np.load(fp+'Twater_cities/Twater_cities_'+iloc+'.npz',allow_pickle='TRUE')
        Twater_city = water_city_data['Twater']

        ds[:,c] = Twater_city[:,1]
        labels[c]  ='Twater city [deg.C] - '+iloc
        c += 1
        save_name += iloc + '_'

if n4 > 0:
    for i,iloc in enumerate(freezeup_name_list):
        ice_data = np.load(fp+'freezeup_dates_SLSMC/freezeup_SLSMC_'+iloc+'.npz',allow_pickle='TRUE')
        freezeup_ci = ice_data['freezeup_ci']
        freezeup_fi = ice_data['freezeup_fi']
        freezeup_si = ice_data['freezeup_si']

        ds[:,c] = freezeup_ci[:,0]
        labels[c]  ='Freezeup date from charts - '+iloc
        c += 1
        ds[:,c] = freezeup_fi[:,0]
        labels[c]  ='First ice date from SLSMC - '+iloc
        c += 1
        ds[:,c] = freezeup_si[:,0]
        labels[c]  ='Stable ice date from SLSMC - '+iloc
        c += 1
        save_name += iloc + '_'

if n5 > 0:
    for i,iloc in enumerate(weather_name_list):
        file_data = np.load(fp+'weather_NCEI/weather_NCEI_'+iloc+'.npz',allow_pickle='TRUE')
        weather_data = file_data['weather_data']
        weather_vars = file_data['select_vars']
        # Convert Farenheit to Celsius
        weather_data[:,1] = (weather_data[:,1] - 32)*(5/9.)
        weather_data[:,2] = (weather_data[:,2] - 32)*(5/9.)
        weather_data[:,3] = (weather_data[:,3] - 32)*(5/9.)
        weather_data[:,4] = (weather_data[:,4] - 32)*(5/9.)
        # Convert millibars to kPa
        weather_data[:,6] = (weather_data[:,6])*(0.1)
        # Convert knots to km/h
        weather_data[:,7] = weather_data[:,7]*1.852

        ds[:,c:c+len(weather_vars)] = weather_data[:,1:]
        labels[c:c+len(weather_vars)]  = [weather_vars[k] + ' - '+ iloc for k in range(len(weather_vars))]

        c += len(weather_vars)
        save_name += iloc + '_'


# ==========================================================================
save_name = save_name.rstrip('_')
np.savez('../../data/ML_timeseries/ML_dataset_'+save_name,
          data=ds,labels=labels,date_ref=date_ref)



