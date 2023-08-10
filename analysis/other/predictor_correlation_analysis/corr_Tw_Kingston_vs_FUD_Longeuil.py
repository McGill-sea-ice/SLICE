#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 12:40:49 2022

@author: Amelie
"""
#%%
local_path = '/storage/amelie/'
# local_path = '/Volumes/SeagateUSB/McGill/Postdoc/'
#%%

import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath(local_path+'slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from functions import detect_FUD_from_Tw
#%%
date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

years = np.array([1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020,2021])

#%%
# Load Tw data
Tw_Kingston = np.load(local_path+'slice/data/processed/Twater_SLSMC/Twater_SLSMC_Kingston.npz', allow_pickle=True)
Tw_Kingston = Tw_Kingston['Twater'][:,1]
Tw_Kingston[14755:] = np.nan

Tw_Iroquois = np.load(local_path+'slice/data/processed/Twater_SLSMC/Twater_SLSMC_Iroquois.npz', allow_pickle=True)
Tw_Iroquois = Tw_Iroquois['Twater'][:,1]
Tw_Iroquois[14755:] = np.nan

Tw_Cornwall = np.load(local_path+'slice/data/processed/Twater_SLSMC/Twater_SLSMC_Cornwall.npz', allow_pickle=True)
Tw_Cornwall = Tw_Cornwall['Twater'][:,1]
Tw_Cornwall[14755:] = np.nan

Tw_StLambert = np.load(local_path+'slice/data/processed/Twater_SLSMC/Twater_SLSMC_StLambert.npz', allow_pickle=True)
Tw_StLambert = Tw_StLambert['Twater'][:,1]
Tw_StLambert[14755:] = np.nan

# Find years available in data
year_0_Kingston = (date_ref+dt.timedelta(days=int(time[int(np.where(~np.isnan(Tw_Kingston))[0][0])]))).year
year_1_Kingston = (date_ref+dt.timedelta(days=int(time[int(np.where(~np.isnan(Tw_Kingston))[0][-1])]))).year

year_0_Cornwall = (date_ref+dt.timedelta(days=int(time[int(np.where(~np.isnan(Tw_Cornwall))[0][0])]))).year
year_1_Cornwall = (date_ref+dt.timedelta(days=int(time[int(np.where(~np.isnan(Tw_Cornwall))[0][-1])]))).year

year_0_Iroquois = (date_ref+dt.timedelta(days=int(time[int(np.where(~np.isnan(Tw_Iroquois))[0][0])]))).year
year_1_Iroquois = (date_ref+dt.timedelta(days=int(time[int(np.where(~np.isnan(Tw_Iroquois))[0][-1])]))).year

year_0_StLambert = (date_ref+dt.timedelta(days=int(time[int(np.where(~np.isnan(Tw_StLambert))[0][0])]))).year
year_1_StLambert = (date_ref+dt.timedelta(days=int(time[int(np.where(~np.isnan(Tw_StLambert))[0][-1])]))).year

year_0 = np.max([year_0_Kingston,year_0_Cornwall,year_0_Iroquois,year_0_StLambert])
year_1 = np.min([year_1_Kingston,year_1_Cornwall,year_1_Iroquois,year_1_StLambert])
years_SLSMC = np.arange(year_0, year_1)

#%%
# Load Twater and FUD data for Longueuil
fp_p_Twater = local_path+'slice/data/processed/'
Twater_loc_list = ['Longueuil_updated']
station_type = 'cities'
freezeup_opt = 1
freezeup_doy, Twater = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False)
freezeup_doy[np.where(years == 2020)] = np.nan

Twater_mean = np.nanmean(Twater,axis=1)
Twater_mean = np.expand_dims(Twater_mean, axis=1)
Twater_mean[14269:14329] = 0.

# Average (and round) FUD from all locations:
avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
avg_freezeup_doy = np.round(avg_freezeup_doy)
#%%
plt.figure()
plt.plot(Tw_Kingston,label='Kingston')
plt.plot(Tw_Iroquois,label='Iroquois')
plt.plot(Tw_Cornwall,label='Cornwall')
plt.plot(Tw_StLambert,label='St-Lambert')
plt.plot(Twater_mean,label='Longueuil')
plt.legend()

#%%
# Correlate whole Tw time series at Longueuil with Tw
# at upstream SLSMC stations - with lag between 0-9 days
import statsmodels.api as sm

for delay in range(10):
    print('---------------')
    if delay > 0:
        m_Long_King = sm.OLS(Twater_mean[delay:],sm.add_constant(Tw_Kingston[0:-delay],has_constant='skip'), missing='drop').fit()
        print('Tw Longueuil vs Tw Kingston - '+str(delay)+' days',m_Long_King.rsquared)
        m_Long_Iroq = sm.OLS(Twater_mean[delay:],sm.add_constant(Tw_Iroquois[0:-delay],has_constant='skip'), missing='drop').fit()
        print('Tw Longueuil vs Tw Iroquois - '+str(delay)+' days',m_Long_Iroq.rsquared)
        m_Long_Corn = sm.OLS(Twater_mean[delay:],sm.add_constant(Tw_Cornwall[0:-delay],has_constant='skip'), missing='drop').fit()
        print('Tw Longueuil vs Tw Cornwall - '+str(delay)+' days',m_Long_Corn.rsquared)
    else:
        m_Long_King = sm.OLS(Twater_mean,sm.add_constant(Tw_Kingston,has_constant='skip'), missing='drop').fit()
        print('Tw Longueuil vs Tw Kingston - '+str(delay)+' days',m_Long_King.rsquared)
        m_Long_Iroq = sm.OLS(Twater_mean,sm.add_constant(Tw_Iroquois,has_constant='skip'), missing='drop').fit()
        print('Tw Longueuil vs Tw Iroquois - '+str(delay)+' days',m_Long_Iroq.rsquared)
        m_Long_Corn = sm.OLS(Twater_mean,sm.add_constant(Tw_Cornwall,has_constant='skip'), missing='drop').fit()
        print('Tw Longueuil vs Tw Cornwall - '+str(delay)+' days',m_Long_Corn.rsquared)

#%%
# Get upstream water temperature at forecast start dates:
Tw_SLSMC_Oct1 = np.zeros((4,len(years)))*np.nan
Tw_SLSMC_Nov1 = np.zeros((4,len(years)))*np.nan
Tw_SLSMC_Dec1 = np.zeros((4,len(years)))*np.nan
Tw_SLSMC_Dec15 = np.zeros((4,len(years)))*np.nan

Tw_GHRSST_May1 = np.zeros((len(years)))*np.nan
Tw_GHRSST_Jun1 = np.zeros((len(years)))*np.nan
Tw_GHRSST_Jul1 = np.zeros((len(years)))*np.nan
Tw_GHRSST_Aug1 = np.zeros((len(years)))*np.nan

Tw_Long_Oct1 = np.zeros((len(years)))*np.nan
Tw_Long_Nov1 = np.zeros((len(years)))*np.nan
Tw_Long_Dec1 = np.zeros((len(years)))*np.nan
Tw_Long_Dec15 = np.zeros((len(years)))*np.nan


# Tw_GHRSST = np.load(local_path+'slice/data/processed/CMC_GHRSST/monthly_CMC_GHRSST_Tw_extracted_at_Kingston.npz', allow_pickle=True)
# Tw_GHRSST =Tw_GHRSST ['data']

for iyr,year in enumerate(years):
    it_May1 = np.where(time == (dt.date(year,5,1)-date_ref).days)[0][0]
    it_Jun1 = np.where(time == (dt.date(year,6,1)-date_ref).days)[0][0]
    it_Jul1 = np.where(time == (dt.date(year,7,1)-date_ref).days)[0][0]
    it_Aug1 = np.where(time == (dt.date(year,8,1)-date_ref).days)[0][0]

    it_Sep1 = np.where(time == (dt.date(year,9,1)-date_ref).days)[0][0]
    it_Oct1 = np.where(time == (dt.date(year,10,1)-date_ref).days)[0][0]
    it_Nov1 = np.where(time == (dt.date(year,11,1)-date_ref).days)[0][0]
    it_Dec1 = np.where(time == (dt.date(year,12,1)-date_ref).days)[0][0]
    it_Nov15 = np.where(time == (dt.date(year,11,15)-date_ref).days)[0][0]
    it_Dec15 = np.where(time == (dt.date(year,12,15)-date_ref).days)[0][0]

    Tw_SLSMC_Oct1[0,iyr] = Tw_Kingston[it_Oct1]
    Tw_SLSMC_Nov1[0,iyr] = Tw_Kingston[it_Nov1]
    Tw_SLSMC_Dec1[0,iyr] = Tw_Kingston[it_Dec1]
    Tw_SLSMC_Dec15[0,iyr] = Tw_Kingston[it_Dec15]

    Tw_SLSMC_Oct1[1,iyr] = Tw_Iroquois[it_Oct1]
    Tw_SLSMC_Nov1[1,iyr] = Tw_Iroquois[it_Nov1]
    Tw_SLSMC_Dec1[1,iyr] = Tw_Iroquois[it_Dec1]
    Tw_SLSMC_Dec15[1,iyr] = Tw_Iroquois[it_Dec15]

    Tw_SLSMC_Oct1[2,iyr] = Tw_Cornwall[it_Oct1]
    Tw_SLSMC_Nov1[2,iyr] = Tw_Cornwall[it_Nov1]
    Tw_SLSMC_Dec1[2,iyr] = Tw_Cornwall[it_Dec1]
    Tw_SLSMC_Dec15[2,iyr] = Tw_Cornwall[it_Dec15]

    Tw_SLSMC_Oct1[3,iyr] = np.nanmean(Twater_mean[it_Sep1 :it_Oct1])
    Tw_SLSMC_Nov1[3,iyr] = np.nanmean(Twater_mean[it_Oct1:it_Nov1])
    Tw_SLSMC_Dec1[3,iyr] = np.nanmean(Twater_mean[it_Nov1:it_Dec1])
    Tw_SLSMC_Dec15[3,iyr] = np.nanmean(Twater_mean[it_Nov15:it_Dec15])

    Tw_Long_Oct1[iyr] = Twater_mean[it_Oct1]
    Tw_Long_Nov1[iyr] = Twater_mean[it_Nov1]
    Tw_Long_Dec1[iyr] = Twater_mean[it_Dec1]
    Tw_Long_Dec15[iyr] = Twater_mean[it_Dec15]

    # Tw_GHRSST_May1[iyr] = Tw_GHRSST[it_May1]
    # Tw_GHRSST_Jun1[iyr] = Tw_GHRSST[it_Jun1]
    # Tw_GHRSST_Jul1[iyr] = Tw_GHRSST[it_Jul1]
    # Tw_GHRSST_Aug1[iyr] = Tw_GHRSST[it_Aug1]



#%%
# Correlate upstream water temperature at forecast start dates with FUD:

print('-------------------------')
m_Long_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(Tw_Long_Oct1,has_constant='skip'), missing='drop').fit()
print('Tw Longueuil vs FUD - Oct 1st: ',m_Long_FUD.rsquared,m_Long_FUD.f_pvalue)
m_Long_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(Tw_Long_Nov1,has_constant='skip'), missing='drop').fit()
print('Tw Longueuil vs FUD - Nov 1st: ',m_Long_FUD.rsquared,m_Long_FUD.f_pvalue)
m_Long_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(Tw_Long_Dec1,has_constant='skip'), missing='drop').fit()
print('Tw Longueuil vs FUD - Dec 1st: ',m_Long_FUD.rsquared,m_Long_FUD.f_pvalue)
m_Long_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(Tw_Long_Dec15,has_constant='skip'), missing='drop').fit()
print('Tw Longueuil vs FUD - Dec 15th: ',m_Long_FUD.rsquared,m_Long_FUD.f_pvalue)

print('-------------------------')
if ~np.all(np.isnan(Tw_SLSMC_Oct1[0,:])):
    m_King_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_SLSMC_Oct1[0,:]),has_constant='skip'), missing='drop').fit()
    print('Tw Kingston vs FUD - Oct 1st: ',m_King_FUD.rsquared,m_King_FUD.f_pvalue)
if ~np.all(np.isnan(Tw_SLSMC_Nov1[0,:])):
    m_King_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_SLSMC_Nov1[0,:]),has_constant='skip'), missing='drop').fit()
    print('Tw Kingston vs FUD - Nov 1st: ',m_King_FUD.rsquared,m_King_FUD.f_pvalue)
if ~np.all(np.isnan(Tw_SLSMC_Dec1[0,:])):
    m_King_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_SLSMC_Dec1[0,:]),has_constant='skip'), missing='drop').fit()
    print('Tw Kingston vs FUD - Dec 1st: ',m_King_FUD.rsquared,m_King_FUD.f_pvalue)
if ~np.all(np.isnan(Tw_SLSMC_Dec15[0,:])):
    m_King_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_SLSMC_Dec15[0,:]),has_constant='skip'), missing='drop').fit()
    print('Tw Kingston vs FUD - Dec 15th: ',m_King_FUD.rsquared,m_King_FUD.f_pvalue)

print('-------------------------')
if ~np.all(np.isnan(Tw_SLSMC_Oct1[1,:])):
    m_Iroq_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_SLSMC_Oct1[1,:]),has_constant='skip'), missing='drop').fit()
    print('Tw Iroquois vs FUD - Oct 1st: ',m_Iroq_FUD.rsquared,m_Iroq_FUD.f_pvalue)
if ~np.all(np.isnan(Tw_SLSMC_Nov1[1,:])):
    m_Iroq_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_SLSMC_Nov1[1,:]),has_constant='skip'), missing='drop').fit()
    print('Tw Iroquois vs FUD - Nov 1st: ',m_Iroq_FUD.rsquared,m_Iroq_FUD.f_pvalue)
if ~np.all(np.isnan(Tw_SLSMC_Dec1[1,:])):
    m_Iroq_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_SLSMC_Dec1[1,:]),has_constant='skip'), missing='drop').fit()
    print('Tw Iroquois vs FUD - Dec 1st: ',m_Iroq_FUD.rsquared,m_Iroq_FUD.f_pvalue)
if ~np.all(np.isnan(Tw_SLSMC_Dec15[1,:])):
    m_Iroq_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_SLSMC_Dec15[1,:]),has_constant='skip'), missing='drop').fit()
    print('Tw Iroquois vs FUD - Dec 15th: ',m_Iroq_FUD.rsquared,m_Iroq_FUD.f_pvalue)

print('-------------------------')
if ~np.all(np.isnan(Tw_SLSMC_Oct1[2,:])):
    m_Corn_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_SLSMC_Oct1[2,:]),has_constant='skip'), missing='drop').fit()
    print('Tw Cornwall vs FUD - Oct 1st: ',m_Corn_FUD.rsquared,m_Corn_FUD.f_pvalue)
if ~np.all(np.isnan(Tw_SLSMC_Nov1[2,:])):
    m_Corn_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_SLSMC_Nov1[2,:]),has_constant='skip'), missing='drop').fit()
    print('Tw Cornwall vs FUD - Nov 1st: ',m_Corn_FUD.rsquared,m_Corn_FUD.f_pvalue)
if ~np.all(np.isnan(Tw_SLSMC_Dec1[2,:])):
    m_Corn_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_SLSMC_Dec1[2,:]),has_constant='skip'), missing='drop').fit()
    print('Tw Cornwall vs FUD - Dec 1st: ',m_Corn_FUD.rsquared,m_Corn_FUD.f_pvalue)
if ~np.all(np.isnan(Tw_SLSMC_Dec15[2,:])):
    m_Corn_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_SLSMC_Dec15[2,:]),has_constant='skip'), missing='drop').fit()
    print('Tw Cornwall vs FUD - Dec 15th: ',m_Corn_FUD.rsquared,m_Corn_FUD.f_pvalue)

print('-------------------------')

#%%
print('-------------------------')
# if ~np.all(np.isnan(Tw_GHRSST_May1[:])):
#     m = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_GHRSST_May1),has_constant='skip'), missing='drop').fit()
#     print('Tw GHRSST vs FUD - Oct 1st: ',m.rsquared,m.f_pvalue)
# if ~np.all(np.isnan(Tw_GHRSST_Jun1[:])):
#     m = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_GHRSST_Jun1[:]),has_constant='skip'), missing='drop').fit()
#     print('Tw GHRSST vs FUD - Nov 1st: ',m.rsquared,m.f_pvalue)
# if ~np.all(np.isnan(Tw_GHRSST_Jul1[:])):
#     m_ = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_GHRSST_Jul1[:]),has_constant='skip'), missing='drop').fit()
#     print('Tw GHRSST vs FUD - Dec 1st: ',m.rsquared,m.f_pvalue)
# if ~np.all(np.isnan(Tw_GHRSST_Aug1[:])):
#     m = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_GHRSST_Aug1[:]),has_constant='skip'), missing='drop').fit()
#     print('Tw GHRSST vs FUD - Dec 15th: ',m.rsquared,m.f_pvalue)

if ~np.all(np.isnan(Tw_SLSMC_Oct1[3,:])):
    m_Corn_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_SLSMC_Oct1[3,:]),has_constant='skip'), missing='drop').fit()
    print('Tw Kingston GHRSST vs FUD - Oct 1st: ',m_Corn_FUD.rsquared,m_Corn_FUD.f_pvalue)
if ~np.all(np.isnan(Tw_SLSMC_Nov1[3,:])):
    m_Corn_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_SLSMC_Nov1[3,:]),has_constant='skip'), missing='drop').fit()
    print('Tw Kingston GHRSST vs FUD - Nov 1st: ',m_Corn_FUD.rsquared,m_Corn_FUD.f_pvalue)
if ~np.all(np.isnan(Tw_SLSMC_Dec1[3,:])):
    m_Corn_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_SLSMC_Dec1[3,:]),has_constant='skip'), missing='drop').fit()
    print('Tw Kingston GHRSST vs FUD - Dec 1st: ',m_Corn_FUD.rsquared,m_Corn_FUD.f_pvalue)
if ~np.all(np.isnan(Tw_SLSMC_Dec15[3,:])):
    m_Corn_FUD = sm.OLS(avg_freezeup_doy,sm.add_constant(np.squeeze(Tw_SLSMC_Dec15[3,:]),has_constant='skip'), missing='drop').fit()
    print('Tw Kingston GHRSST vs FUD - Dec 15th: ',m_Corn_FUD.rsquared,m_Corn_FUD.f_pvalue)



