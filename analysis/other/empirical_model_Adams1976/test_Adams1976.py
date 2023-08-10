#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:10:15 2022

@author: Amelie
"""
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

#%%
date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

#%%
# Load Tw data
Tw_Kingston = np.load('/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/Twater_SLSMC/Twater_SLSMC_Kingston.npz', allow_pickle=True)
Tw_Kingston = Tw_Kingston['Twater'][:,1]
Tw_Kingston[14755:] = np.nan

Tw_Iroquois = np.load('/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/Twater_SLSMC/Twater_SLSMC_Iroquois.npz', allow_pickle=True)
Tw_Iroquois = Tw_Iroquois['Twater'][:,1]
Tw_Iroquois[14755:] = np.nan

Tw_Cornwall = np.load('/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/Twater_SLSMC/Twater_SLSMC_Cornwall.npz', allow_pickle=True)
Tw_Cornwall = Tw_Cornwall['Twater'][:,1]
Tw_Cornwall[14755:] = np.nan

Tw_StLambert = np.load('/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/Twater_SLSMC/Twater_SLSMC_StLambert.npz', allow_pickle=True)
Tw_StLambert = Tw_StLambert['Twater'][:,1]
Tw_StLambert[14755:] = np.nan

# Find years available in data
year_0_Kingston = (date_ref+dt.timedelta(days=int(time[int(np.where(~np.isnan(Tw_Kingston))[0][0])]))).year
year_1_Kingston = (date_ref+dt.timedelta(days=int(time[int(np.where(~np.isnan(Tw_Kingston))[0][-1])]))).year

year_0_Cornwall = (date_ref+dt.timedelta(days=int(time[int(np.where(~np.isnan(Tw_Cornwall))[0][0])]))).year
year_1_Cornwall = (date_ref+dt.timedelta(days=int(time[int(np.where(~np.isnan(Tw_Cornwall))[0][-1])]))).year

year_0 = np.max([year_0_Kingston,year_0_Cornwall])
year_1 = np.min([year_1_Kingston,year_1_Cornwall])
years = np.arange(year_0, year_1)
#%%
plt.figure()
plt.plot(Tw_Kingston)
plt.plot(Tw_Iroquois)
plt.plot(Tw_Cornwall)
plt.plot(Tw_StLambert)

#%%
# Get forecast of Tw at Kingston on Dec. 1st and Dec. 15th:
a = [0.00935, 0.01750, 0.05180]
b = [0.0492,  0.0920,  0.2175 ]
c = [1.09886, 1.09886, 1.09886]

Tw_King_Oct1 = np.zeros((len(years)))*np.nan
Tw_King_Nov1 = np.zeros((len(years)))*np.nan
Tw_King_Dec1 = np.zeros((len(years)))*np.nan
Tw_King_Dec15 = np.zeros((len(years)))*np.nan

for iyr,year in enumerate(years):
    it_Oct1 = np.where(time == (dt.date(year,10,1)-date_ref).days)[0][0]
    it_Nov1 = np.where(time == (dt.date(year,11,1)-date_ref).days)[0][0]
    it_Dec1 = np.where(time == (dt.date(year,12,1)-date_ref).days)[0][0]
    it_Dec15 = np.where(time == (dt.date(year,12,15)-date_ref).days)[0][0]

    Tw_King_Oct1[iyr] = Tw_Kingston[it_Oct1]
    Tw_King_Nov1[iyr] = Tw_Kingston[it_Nov1]
    Tw_King_Dec1[iyr] = Tw_Kingston[it_Dec1]
    Tw_King_Dec15[iyr] = Tw_Kingston[it_Dec15]


Dec1_Tw_King_Oct1 = np.zeros((len(years)))*np.nan
Dec1_Tw_King_Nov1 = np.zeros((len(years)))*np.nan

Dec15_Tw_King_Oct1 = np.zeros((len(years)))*np.nan
Dec15_Tw_King_Nov1 = np.zeros((len(years)))*np.nan
Dec15_Tw_King_Dec1 = np.zeros((len(years)))*np.nan

for iyr,year in enumerate(years):
    Dec1_Tw_King_Oct1[iyr] = -(a[0]*Tw_King_Oct1[iyr] - b[0])*( (dt.date(int(year),12,1)-dt.date(int(year),10,1)).days**c[0]) + Tw_King_Oct1[iyr]
    Dec1_Tw_King_Nov1[iyr] = -(a[1]*Tw_King_Nov1[iyr] - b[1])*( (dt.date(int(year),12,1)-dt.date(int(year),11,1)).days**c[1]) + Tw_King_Nov1[iyr]

    Dec15_Tw_King_Oct1[iyr] = -(a[0]*Tw_King_Oct1[iyr] - b[0])*( (dt.date(int(year),12,15)-dt.date(int(year),10,1)).days**c[0]) + Tw_King_Oct1[iyr]
    Dec15_Tw_King_Nov1[iyr] = -(a[1]*Tw_King_Nov1[iyr] - b[1])*( (dt.date(int(year),12,15)-dt.date(int(year),11,1)).days**c[1]) + Tw_King_Nov1[iyr]
    Dec15_Tw_King_Dec1[iyr] = -(a[2]*Tw_King_Dec1[iyr] - b[2])*( (dt.date(int(year),12,15)-dt.date(int(year),12,1)).days**c[2]) + Tw_King_Dec1[iyr]


#%%
iyr = 0
rho = 997 # kg/m3
Cp = 4182 #J/kg°C

Dec1_Qt_Nov1 = np.zeros((len(years)))*np.nan
Dec1_Qt_Nov1[0] = -270 # ly == J/m2 ???

sec_per_day = 24*60*60 #[s/day]

# dist = 250e3 # Kingston - Cornwall [m]
# discharge_downstream = 7000 #[m^3/s]
# avg_river_width =  1.2e3 #[m]
# avg_river_depth = 19 #[m]

dist = 250e3 # Kingston - Montreal [m]
discharge_downstream = 8000 #[m^3/s]
avg_river_width =  2e3 #[m]
avg_river_depth = 12 #[m]

avg_cross_section = avg_river_width * avg_river_depth #[m^2]
flow_velocity = discharge_downstream / avg_cross_section #[m/s]
travel_time = (dist/flow_velocity)/sec_per_day #[day]

T0_Kingston = 0.5 + ((1/(rho*Cp*avg_river_depth))*(Dec1_Qt_Nov1[iyr])*(travel_time))

#%%
Tw_Mtl = Tw_Kingston - ((travel_time/(rho*Cp*avg_river_depth))*Cc*(Tw_Kingston-Tair))
