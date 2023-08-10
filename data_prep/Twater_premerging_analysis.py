#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 15:28:03 2021

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
import pandas as pd

import datetime as dt
import calendar

import matplotlib.pyplot as plt

import scipy.stats as stats
import pandas as pd
# import researchpy as rp

#%%

def test_distributions(df_cities,df_eccc,loc_cities,loc_eccc,test='KW',show=False):

    if show:
        plt.ion()
    else:
        plt.ioff()

    fig,ax = plt.subplots(nrows=len(loc_cities)+len(loc_eccc),ncols=1,figsize=(5,6),sharex=True)#,tight_layout=True)

    vars_test = np.zeros((15341,len(loc_cities)+len(loc_eccc)))*np.nan
    mask = np.zeros((15341))
    labels=[]
    for iloc in range(len(loc_cities)):
        vars_test[:,iloc] = df_cities['Twater'][df_cities['Plant'] == loc_cities[iloc]]
        ax[iloc].plot(vars_test[:,iloc],color=plt.get_cmap('tab20')(iloc*2+1),label=loc_cities[iloc])
        ax[iloc].set_xlim(4000,15250)
        # if loc_cities[iloc] == 'Candiac':
        #     vars_test[11000:,iloc] = np.nan
        #     vars_test[:,iloc][vars_test[:,iloc] < 0] = 0.0
        v_arr = np.array(vars_test[:,iloc])
        mask += np.isnan(v_arr).astype(int)
        labels.append(loc_cities[iloc])

    for iloc in range(len(loc_cities),len(loc_eccc)+len(loc_cities)):
        vars_test[:,iloc] = df_eccc['Twater'][df_eccc['Plant'] == loc_eccc[iloc-len(loc_cities)]]
        ax[iloc].plot(vars_test[:,iloc],color=plt.get_cmap('tab20')(iloc*2+1),label=loc_eccc[iloc-len(loc_cities)])
        ax[iloc].set_xlim(4000,15250)
        v_arr = np.array(vars_test[:,iloc])
        mask += np.isnan(v_arr).astype(int)
        labels.append(loc_eccc[iloc-len(loc_cities)])

    for iloc in range(vars_test.shape[1]):
        var_plot = vars_test[:,iloc].copy()
        var_plot[mask > 0] = np.nan
        ax[iloc].plot(var_plot,color=plt.get_cmap('tab20')(iloc*2))
        ax[iloc].legend()
        if iloc == 0:
            twlongueuil = var_plot
        if iloc == 1:
            twlaprairie = var_plot


    plt.figure()
    plt.plot(twlongueuil,twlaprairie,'.')


    # Keep only where the data is available for all variables
    vars_test = vars_test[mask == 0,:]

    plt.figure()
    for iloc in range(vars_test.shape[1]):
        plt.hist(vars_test[:,iloc],bins=16,range=(-2,30),alpha=0.2,density=True,label=labels[iloc])
    plt.legend()

    if test == 'KW':
        H, pval = stats.kruskal(*[vars_test[:,ivar] for ivar in range(vars_test.shape[1])])

        print('\nKRUSKAL-WALLIS TEST\n')
        print("Locations: " + str([loc for loc in loc_cities]+[loc for loc in loc_eccc]))
        print("N values: "+'%i'%vars_test.shape[0])
        print("H statistic: " + "%5.3f"%H)
        print("p-value: "+ "%4.3f"%pval)
        print('')

        if pval < 0.05:
            print("Reject NULL hypothesis - Significant differences exist between groups.")
        if pval > 0.05:
            print("Accept NULL hypothesis - No significant difference between groups.")


    if test == 'ANOVA':
        F, pval = stats.f_oneway(*[vars_test[:,ivar] for ivar in range(vars_test.shape[1])])

        print('\nONE-WAY ANOVA TEST\n')
        print("Locations: " + str([loc for loc in loc_cities]+[loc for loc in loc_eccc]))
        print("N values: "+'%i'%vars_test.shape[0])
        print("F statistic: " + "%5.3f"%F)
        print("p-value: "+ "%4.3f"%pval)
        print('')

        if pval < 0.05:
            print("Reject NULL hypothesis - Significant differences exist between groups.")
        if pval > 0.05:
            print("Accept NULL hypothesis - No significant difference between groups.")


def test_distributions_by_season(df_cities,df_eccc,loc_cities,loc_eccc,test='KW',season='Fall',show=False):

    if show:
        plt.ion()
    else:
        plt.ioff()

    fig,ax = plt.subplots(nrows=len(loc_cities)+len(loc_eccc),ncols=1,figsize=(5,6),sharex=True)#,tight_layout=True)

    if len(loc_cities) != 0:
        ls = df_cities['Twater'][df_cities['Plant'] == loc_cities[0]][df_cities['Season']==season].shape[0]
    else:
        ls = df_eccc['Twater'][df_eccc['Plant'] == loc_eccc[0]][df_eccc['Season']==season].shape[0]


    vars_test = np.zeros((ls,len(loc_cities)+len(loc_eccc)))*np.nan
    mask = np.zeros((ls))
    labels=[]
    for iloc in range(len(loc_cities)):
        vars_test[:,iloc] = df_cities['Twater'][df_cities['Plant'] == loc_cities[iloc]][df_cities['Season'] == season]
        ax[iloc].plot(vars_test[:,iloc],color=plt.get_cmap('tab20')(iloc*2+1),label=loc_cities[iloc])
        # ax[iloc].set_xlim(4000,15250)
        v_arr = np.array(vars_test[:,iloc])
        mask += np.isnan(v_arr).astype(int)
        labels.append(loc_cities[iloc])

    for iloc in range(len(loc_cities),len(loc_eccc)+len(loc_cities)):
        vars_test[:,iloc] = df_eccc['Twater'][df_eccc['Plant'] == loc_eccc[iloc-len(loc_cities)]][df_eccc['Season'] == season]
        ax[iloc].plot(vars_test[:,iloc],color=plt.get_cmap('tab20')(iloc*2+1),label=loc_eccc[iloc-len(loc_cities)])
        # ax[iloc].set_xlim(4000,15250)
        v_arr = np.array(vars_test[:,iloc])
        mask += np.isnan(v_arr).astype(int)
        labels.append(loc_eccc[iloc-len(loc_cities)])

    for iloc in range(vars_test.shape[1]):
        var_plot = vars_test[:,iloc].copy()
        var_plot[mask > 0] = np.nan
        ax[iloc].plot(var_plot,color=plt.get_cmap('tab20')(iloc*2))
        ax[iloc].legend()
        if iloc == 0:
            twlongueuil = var_plot
        if iloc == 1:
            twlaprairie = var_plot


    plt.figure()
    plt.plot(twlongueuil,twlaprairie,'.')

    # Keep only where the data is available for all variables
    vars_test = vars_test[mask == 0,:]

    plt.figure()
    for iloc in range(vars_test.shape[1]):
        plt.hist(vars_test[:,iloc],bins=32,range=(-2,30),alpha=0.2,density=True,label=labels[iloc])
    plt.legend()

    if test == 'KW':
        H, pval = stats.kruskal(*[vars_test[:,ivar] for ivar in range(vars_test.shape[1])])

        print('\nKRUSKAL-WALLIS TEST\n')
        print("Locations: " + str([loc for loc in loc_cities]+[loc for loc in loc_eccc]))
        print("N values: "+'%i'%vars_test.shape[0])
        print("H statistic: " + "%5.3f"%H)
        print("p-value: "+ "%4.3f"%pval)
        print('')

        if pval < 0.05:
            print("Reject NULL hypothesis - Significant differences exist between groups.")
        if pval > 0.05:
            print("Accept NULL hypothesis - No significant difference between groups.")


    if test == 'ANOVA':
        F, pval = stats.f_oneway(*[vars_test[:,ivar] for ivar in range(vars_test.shape[1])])

        print('\nONE-WAY ANOVA TEST\n')
        print("Locations: " + str([loc for loc in loc_cities]+[loc for loc in loc_eccc]))
        print("N values: "+'%i'%vars_test.shape[0])
        print("F statistic: " + "%5.3f"%F)
        print("p-value: "+ "%4.3f"%pval)
        print('')

        if pval < 0.05:
            print("Reject NULL hypothesis - Significant differences exist between groups.")
        if pval > 0.05:
            print("Accept NULL hypothesis - No significant difference between groups.")




#%%
stats_verbose = True

fp = local_path+'slice/data/processed/'

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

cities_in = ['Longueuil_updated']
# cities_in = ['Atwater','Longueuil','DesBaillets','Candiac']
cities=''
for n in cities_in:
    cities += n
data_cities = np.load(fp+'Twater_cities/Twater_cities_all_'+cities+'_cleaned_filled.npz',allow_pickle='TRUE')
Twater_all = data_cities['Twater_all']

eccc_in = ['Lasalle','LaPrairie']
stations=''
for n in eccc_in:
    stations += n
data_eccc = np.load(fp+'Twater_ECCC/Twater_ECCC_all_'+stations+'_cleaned_filled.npz',allow_pickle='TRUE')
Twater_all_eccc = data_eccc['Twater_all']

slsmc_in = ['StLambert','StLouisBridge']
stations=''
for n in slsmc_in:
    stations += n
data_slsmc = np.load(fp+'Twater_SLSMC/Twater_SLSMC_all_'+stations+'_cleaned_filled.npz',allow_pickle='TRUE')
Twater_all_slsmc = data_slsmc['Twater_all']



# THIS IS NOT NECESSARY ANYMORE SINCE DATA PRIOR TO 2010 HAS
# NOW BEEN REMOVED FROM ATWATER RECORDS WHEN CLEANING DATA SET IN Twater_cleaning_filters.py

# For the purpose of this test, degrade the data
# prior to 2006 to match accuracy of Atwater's time series
# for il in range(Twater_all.shape[1]):
#     Tw = Twater_all[il*14976:(il+1)*14976,0]
#     Tw[0:9562] = np.round(Tw[0:9562])

# for il in range(Twater_all_eccc.shape[1]):
#     Tw = Twater_all_eccc[il*14976:(il+1)*14976,0]
#     Tw[0:9562] = np.round(Tw[0:9562])



# For the purpose of this test, round all data
# to nearest 0.5 deg. C:
# for il in range(Twater_all.shape[1]):
#     Tw = Twater_all[il*14976:(il+1)*14976,0]
#     Tw_tmp = np.round(Tw.copy() * 2) / 2.
#     Twater_all[il*14976:(il+1)*14976,0] = Tw_tmp

# for il in range(Twater_all_eccc.shape[1]):
#     Tw = Twater_all_eccc[il*14976:(il+1)*14976,0]
#     Tw_tmp = np.round(Tw.copy() * 2) / 2.
#     Twater_all_eccc[il*14976:(il+1)*14976,0] = Tw_tmp

# for il in range(Twater_all_slsmc.shape[1]):
#     Tw = Twater_all_slsmc[il*14976:(il+1)*14976,0]
#     Tw_tmp = np.round(Tw.copy() * 2) / 2.
#     Twater_all_slsmc[il*14976:(il+1)*14976,0] = Tw_tmp


# For the purpose of this test, round all data
# to nearest 0.1 deg. C:
# for il in range(Twater_all.shape[1]):
#     Tw = Twater_all[il*14976:(il+1)*14976,0]
#     Tw_tmp = np.round(Tw.copy() ,1)
#     Twater_all[il*14976:(il+1)*14976,0] = Tw_tmp

# for il in range(Twater_all_eccc.shape[1]):
#     Tw = Twater_all_eccc[il*14976:(il+1)*14976,0]
#     Tw_tmp = np.round(Tw.copy() ,1)
#     Twater_all_eccc[il*14976:(il+1)*14976,0] = Tw_tmp

# for il in range(Twater_all_slsmc.shape[1]):
#     Tw = Twater_all_slsmc[il*14976:(il+1)*14976,0]
#     Tw_tmp = np.round(Tw.copy() ,1)
#     Twater_all_slsmc[il*14976:(il+1)*14976,0] = Tw_tmp


# For the purpose of this test, put all negative Tw to zero:
for il in range(Twater_all.shape[1]):
    Tw_tmp = Twater_all[il*14976:(il+1)*14976,0].copy()
    Tw_tmp[Tw_tmp <= 0.5] = 0.0
    # Tw_tmp[Tw_tmp <= 2] = np.nan
    Twater_all[il*14976:(il+1)*14976,0] = Tw_tmp

for il in range(Twater_all_eccc.shape[1]):
    Tw_tmp = Twater_all_eccc[il*14976:(il+1)*14976,0].copy()
    Tw_tmp[Tw_tmp <= 0.5] = 0.0
    # Tw_tmp[Tw_tmp <= 2] = np.nan
    # Tw_tmp[14290:-1] = np.nan
    Twater_all_eccc[il*14976:(il+1)*14976,0] = Tw_tmp

for il in range(Twater_all_slsmc.shape[1]):
    Tw_tmp = Twater_all_slsmc[il*14976:(il+1)*14976,0].copy()
    Tw_tmp[Tw_tmp <= 0.5] = 0.0
    # Tw_tmp[Tw_tmp <= 2] = np.nan
    Twater_all_slsmc[il*14976:(il+1)*14976,0] = Tw_tmp



# # For the purpose of this test, put all negative Tw to zero:
# for il in range(Twater_all.shape[1]):
#     Tw_tmp = Twater_all[il*14976:(il+1)*14976,0].copy()
#     Tw_tmp[Tw_tmp <= 0.0] = 0.0
#     Twater_all[il*14976:(il+1)*14976,0] = Tw_tmp

# for il in range(Twater_all_eccc.shape[1]):
#     Tw_tmp = Twater_all_eccc[il*14976:(il+1)*14976,0].copy()
#     Tw_tmp[Tw_tmp <= 0.0] = 0.0
#     Twater_all_eccc[il*14976:(il+1)*14976,0] = Tw_tmp

# for il in range(Twater_all_slsmc.shape[1]):
#     Tw_tmp = Twater_all_slsmc[il*14976:(il+1)*14976,0].copy()
#     Tw_tmp[Tw_tmp <= 0.0] = 0.0
#     Twater_all_slsmc[il*14976:(il+1)*14976,0] = Tw_tmp



Tw_df = pd.DataFrame({'Twater':Twater_all[:,0],
                     'Plant':Twater_all[:,1],
                     'Year':Twater_all[:,2],
                     'Season':Twater_all[:,3]})

for ip,p in enumerate(cities_in):
    Tw_df['Plant'].replace({ip: p}, inplace= True)
Tw_df['Season'].replace({0: 'Spring', 1: 'Summer', 2: 'Fall', 3:'Winter'}, inplace= True)



Tw_eccc_df = pd.DataFrame({'Twater':Twater_all_eccc[:,0],
                     'Plant':Twater_all_eccc[:,1],
                     'Year':Twater_all_eccc[:,2],
                     'Season':Twater_all_eccc[:,3]})

for ip,p in enumerate(eccc_in):
    Tw_eccc_df['Plant'].replace({ip: p}, inplace= True)
Tw_eccc_df['Season'].replace({0: 'Spring', 1: 'Summer', 2: 'Fall', 3:'Winter'}, inplace= True)



Tw_slsmc_df = pd.DataFrame({'Twater':Twater_all_slsmc[:,0],
                     'Plant':Twater_all_slsmc[:,1],
                     'Year':Twater_all_slsmc[:,2],
                     'Season':Twater_all_slsmc[:,3]})

for ip,p in enumerate(slsmc_in):
    Tw_slsmc_df['Plant'].replace({ip: p}, inplace= True)
Tw_slsmc_df['Season'].replace({0: 'Spring', 1: 'Summer', 2: 'Fall', 3:'Winter'}, inplace= True)


# if stats_verbose: print(rp.summary_cont(Tw_df['Twater'].groupby(Tw_df['Plant'])))
# if stats_verbose: print(rp.summary_cont(Tw_eccc_df['Twater'].groupby(Tw_eccc_df['Plant'])))
# if stats_verbose: print(rp.summary_cont(Tw_slsmc_df['Twater'].groupby(Tw_slsmc_df['Plant'])))


#%%
# loc_cities = ['Longueuil_updated']
# loc_slsmc = ['StLambert']
# test_distributions(Tw_df,Tw_slsmc_df,loc_cities,loc_slsmc,test='KW',show=True)
# test_distributions(Tw_df,Tw_slsmc_df,loc_cities,loc_slsmc,test='ANOVA',show=True)

# #%%

# loc_slsmc = ['StLambert']
# loc_eccc = ['LaPrairie']
# test_distributions(Tw_eccc_df,Tw_slsmc_df,loc_eccc,loc_slsmc,test='KW',show=True)
# test_distributions(Tw_eccc_df,Tw_slsmc_df,loc_eccc,loc_slsmc,test='ANOVA',show=True)


#%%
loc_cities = ['Longueuil_updated']
loc_eccc = ['LaPrairie']
# loc_eccc = ['Lasalle']
# loc_eccc =[]
test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW',show=True)
test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='ANOVA',show=True)

plt.figure();
plt.plot(time,Tw_df['Twater'][Tw_df['Plant'] == 'Longueuil_updated'],label='Longueuil');
# plt.plot(time,Tw_eccc_df['Twater'][Tw_eccc_df['Plant'] == 'Lasalle'])
plt.plot(time,Tw_eccc_df['Twater'][Tw_eccc_df['Plant'] == 'LaPrairie'], label='LaPrairie')
plt.legend()

#%%
season_test = 'Summer'
print('\n')
print(season_test)
test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

season_test = 'Fall'
print('\n')
print(season_test)
test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW',show=True)

season_test = 'Winter'
print('\n')
print(season_test)
test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW',show=True)

season_test = 'Spring'
print('\n')
print(season_test)
test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')


#%%
plt.figure();
plt.plot(time,Tw_df['Twater'][Tw_df['Plant'] == 'Longueuil_updated'],label='Longueuil');
# plt.plot(time,Tw_eccc_df['Twater'][Tw_eccc_df['Plant'] == 'Lasalle'])
plt.plot(time,Tw_eccc_df['Twater'][Tw_eccc_df['Plant'] == 'LaPrairie'], label='LaPrairie')
plt.legend()


#%%
# plt.figure()
# plt.plot(Tw_df['Twater'][Tw_df['Plant'] == 'Longueuil_updated'],Tw_eccc_df['Twater'][Tw_eccc_df['Plant'] == 'LaPrairie'],'o')
# # plt.plot(Tw_df['Twater'][Tw_df['Plant'] == 'Longueuil_updated'],Tw_eccc_df['Twater'][Tw_eccc_df['Plant'] == 'Lasalle'],'.')
# plt.plot(Tw_df['Twater'][Tw_df['Plant'] == 'Longueuil_updated'],Tw_df['Twater'][Tw_df['Plant'] == 'Longueuil_updated'],'-')



#%%
# loc_cities = ['Atwater','Longueuil']
# loc_eccc = []
# test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW',show=True)

# loc_cities = ['Candiac','Longueuil']
# loc_eccc = []
# test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW',show=True)


# loc_cities = ['Candiac','Atwater']
# loc_eccc = []
# test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW',show=True)

#%%
# loc_cities = ['Atwater','Longueuil']
# loc_eccc = []
# season_test = 'Summer'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# season_test = 'Fall'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# season_test = 'Winter'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW',show=True)

# season_test = 'Spring'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')
#%%
# loc_cities = ['Candiac','Longueuil']
# loc_eccc = []
# season_test = 'Summer'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# season_test = 'Fall'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# season_test = 'Winter'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW',show=True)

# season_test = 'Spring'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')



#%%
# loc_cities = ['Atwater','Longueuil']
# loc_eccc = []
# season_test = 'Summer'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# season_test = 'Fall'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# season_test = 'Winter'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW',show=True)

# season_test = 'Spring'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')
# #%%
# loc_cities = ['Candiac','Atwater']
# loc_eccc = []
# season_test = 'Summer'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# season_test = 'Fall'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# season_test = 'Winter'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW',show=True)

# season_test = 'Spring'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')




#%%

# COMPARE PAIRED RESIDUALS:
# def test_distributions_by_season(df_cities,df_eccc,loc_cities,loc_eccc,test='KW',season='Fall',show=False):
df_cities = Tw_df
df_eccc= Tw_eccc_df
# loc_cities=['Longueuil','Candiac']
# loc_cities=['Longueuil','Atwater']
# loc_cities=['Atwater','Candiac']
# loc_cities=['Atwater','DesBaillets']
# loc_cities=['Longueuil','DesBaillets']
# loc_eccc=[]
loc_cities=['Longueuil_updated']
loc_eccc=['LaPrairie']
test='KW'
season='Summer'
show=True

if show:
    plt.ion()
else:
    plt.ioff()

season_list = ['Spring','Summer','Fall','Winter']
fig_res,ax_res = plt.subplots(nrows=1,ncols=1,figsize=(5,6))#,tight_layout=True)
plt.title(loc_cities)

for iseason,season in enumerate(season_list):

    if len(loc_cities) != 0:
        ls = df_cities['Twater'][df_cities['Plant'] == loc_cities[0]][df_cities['Season']==season].shape[0]
    else:
        ls = df_eccc['Twater'][df_eccc['Plant'] == loc_eccc[0]][df_eccc['Season']==season].shape[0]


    vars_test = np.zeros((ls,len(loc_cities)+len(loc_eccc)))*np.nan
    mask = np.zeros((ls))
    labels=[]
    for iloc in range(len(loc_cities)):
        vars_test[:,iloc] = df_cities['Twater'][df_cities['Plant'] == loc_cities[iloc]][df_cities['Season'] == season]
        v_arr = np.array(vars_test[:,iloc])
        mask += np.isnan(v_arr).astype(int)
        labels.append(loc_cities[iloc])

    for iloc in range(len(loc_cities),len(loc_eccc)+len(loc_cities)):
        vars_test[:,iloc] = df_eccc['Twater'][df_eccc['Plant'] == loc_eccc[iloc-len(loc_cities)]][df_eccc['Season'] == season]
        v_arr = np.array(vars_test[:,iloc])
        mask += np.isnan(v_arr).astype(int)
        labels.append(loc_eccc[iloc-len(loc_cities)])

    for iloc in range(vars_test.shape[1]):
        var_plot = vars_test[:,iloc].copy()
        var_plot[mask > 0] = np.nan


    # Keep only where the data is available for all variables
    vars_test = vars_test[mask == 0,:]

    var_ref = vars_test[:,0]
    var_rest = vars_test[:,1:]


    for ir in range(var_rest.shape[1]):
        paired_res = var_ref - var_rest[:,ir]
        ax_res.boxplot(paired_res,positions=[iseason],whis=[5,95],showmeans=True,showfliers=False,labels=[season])

#%%

















































# #%%
# loc_cities = ['Atwater','Longueuil','DesBaillets','Candiac']
# loc_eccc = []
# test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW',show=True)

# #%%
# loc_cities = ['Atwater','Longueuil','Candiac']
# loc_eccc = []
# test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW',show=True)

# #%%
# loc_cities = ['Candiac','Atwater']
# loc_eccc = []
# test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW',show=True)

# #%%
# loc_cities = ['Atwater','DesBaillets']
# loc_eccc = []
# test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW',show=True)
# #%%
# loc_cities = ['Longueuil','DesBaillets']
# loc_eccc = []
# test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW',show=True)
# #%%
# loc_cities = ['DesBaillets','Candiac']
# loc_eccc = []
# test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW')

# #%%
# loc_cities = []
# loc_eccc = ['Lasalle','LaPrairie']
# test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW',show=True)

# #%%
# loc_cities = ['Longueuil']
# loc_eccc = ['LaPrairie']
# test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW')

# loc_cities = ['Atwater']
# loc_eccc = ['LaPrairie']
# test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW')

# loc_cities = ['DesBaillets']
# loc_eccc = ['LaPrairie']
# test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW')

# loc_cities = ['Candiac']
# loc_eccc = ['LaPrairie']
# test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW')
# #%%
# loc_cities = ['Longueuil']
# loc_eccc = ['Lasalle']
# test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW')

# loc_cities = ['Atwater']
# loc_eccc = ['Lasalle']
# test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW')

# loc_cities = ['DesBaillets']
# loc_eccc = ['Lasalle']
# test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW')

# loc_cities = ['Candiac']
# loc_eccc = ['Lasalle']
# test_distributions(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,test='KW')


# #%%
# loc_cities = ['Longueuil']
# loc_slsmc = ['StLambert']
# test_distributions(Tw_df,Tw_slsmc_df,loc_cities,loc_slsmc,test='KW',show=True)

# loc_cities = ['Atwater']
# loc_slsmc = ['StLambert']
# test_distributions(Tw_df,Tw_slsmc_df,loc_cities,loc_slsmc,test='KW')

# loc_cities = ['DesBaillets']
# loc_slsmc = ['StLambert']
# test_distributions(Tw_df,Tw_slsmc_df,loc_cities,loc_slsmc,test='KW')

# loc_cities = ['Candiac']
# loc_slsmc = ['StLambert']
# test_distributions(Tw_df,Tw_slsmc_df,loc_cities,loc_slsmc,test='KW')

# #%%
# loc_cities = ['LaPrairie']
# loc_slsmc = ['StLambert']
# test_distributions(Tw_eccc_df,Tw_slsmc_df,loc_cities,loc_slsmc,test='KW',show=True)


# #%%
# # TO CHECK NORMALITY OF DATA - ONE OF THE MAIN HYPOTHESES FOR USING THE F-TEST IN ANOVA
# # import statsmodels.api as sm
# # from statsmodels.formula.api import ols
# # import scipy.stats as stats


# # model = ols('Twater ~ C(Plant)', data=Tw_df).fit()
# # aov_table = sm.stats.anova_lm(model, typ=2)
# # print(aov_table)
# # def anova_table(aov):
# #     aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']

# #     aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])

# #     aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])

# #     cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
# #     aov = aov[cols]
# #     return aov

# # print(anova_table(aov_table))
# # print(stats.shapiro(model.resid))
# # print(stats.normaltest(model.resid))
# # print(stats.kstest(model.resid, 'norm'))

# # result = stats.anderson(model.resid)
# # print(f'Test statistic: {result.statistic: .4f}')
# # for i in range(len(result.critical_values)):
# #     sig, crit = result.significance_level[i], result.critical_values[i]

# #     if result.statistic < result.critical_values[i]:
# #         print(f"At {sig}% significance,{result.statistic: .4f} <{result.critical_values[i]: .4f} data looks normal (fail to reject H0)")
# #     else:
# #         print(f"At {sig}% significance,{result.statistic: .4f} >{result.critical_values[i]: .4f} data does not look normal (reject H0)")







# #%%
# # TEST BY SEASON
# loc_cities = ['Atwater','Candiac','Longueuil']
# loc_eccc = []
# season_test = 'Summer'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# season_test = 'Fall'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# season_test = 'Winter'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW',show=True)

# season_test = 'Spring'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# #%%
# loc_cities = ['Atwater','Longueuil']
# loc_eccc = []
# season_test = 'Summer'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# season_test = 'Fall'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW',show=True)

# season_test = 'Winter'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# season_test = 'Spring'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW',show=True)
# #%%
# loc_cities = ['DesBaillets','Longueuil']
# loc_eccc = []
# season_test = 'Summer'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW',show=True)

# season_test = 'Fall'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW',show=True)

# season_test = 'Winter'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW',show=True)

# season_test = 'Spring'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW',show=True)


# #%%
# loc_cities = ['Candiac','Longueuil']
# loc_eccc = []
# season_test = 'Summer'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# season_test = 'Fall'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# season_test = 'Winter'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW',show=True)

# season_test = 'Spring'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')


# #%%
# loc_cities = ['Atwater','Candiac']
# loc_eccc = []
# season_test = 'Summer'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# season_test = 'Fall'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# season_test = 'Winter'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW',show=True)

# season_test = 'Spring'
# print('\n')
# print(season_test)
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# #%%
# season_test = 'Spring'

# loc_cities = ['Atwater','Candiac']
# loc_eccc = []
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')


# #%%
# season_test = 'Fall'

# loc_cities = ['Atwater']
# loc_eccc = ['LaPrairie']
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# loc_cities = ['Longueuil']
# loc_eccc = ['LaPrairie']
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# #%%
# season_test = 'Fall'

# loc_cities = ['Candiac']
# loc_eccc = ['Lasalle']
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# loc_cities = ['Longueuil']
# loc_eccc = ['Lasalle']
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW')

# #%%
# loc_cities = []
# loc_eccc = ['LaPrairie','Lasalle']
# season_test = 'Fall'
# test_distributions_by_season(Tw_df,Tw_eccc_df,loc_cities,loc_eccc,season=season_test,test='KW',show=True)


# #%%
# season_test = 'Fall'

# loc_cities = ['Atwater']
# loc_slsmc = ['StLambert']
# test_distributions_by_season(Tw_df,Tw_slsmc_df,loc_cities,loc_slsmc,season=season_test,test='KW',show=True)

# loc_cities = ['Longueuil']
# loc_slsmc = ['StLambert']
# test_distributions_by_season(Tw_df,Tw_slsmc_df,loc_cities,loc_slsmc,season=season_test,test='KW',show=True)

# loc_cities = ['DesBaillets']
# loc_slsmc = ['StLambert']
# test_distributions_by_season(Tw_df,Tw_slsmc_df,loc_cities,loc_slsmc,season=season_test,test='KW',show=True)

# loc_cities = ['Candiac']
# loc_slsmc = ['StLambert']
# test_distributions_by_season(Tw_df,Tw_slsmc_df,loc_cities,loc_slsmc,season=season_test,test='KW',show=True)


# #%%

# # COMPARE PAIRED RESIDUALS:
# # def test_distributions_by_season(df_cities,df_eccc,loc_cities,loc_eccc,test='KW',season='Fall',show=False):
# df_cities = Tw_df
# df_eccc= Tw_eccc_df
# loc_cities=['Longueuil','Candiac']
# loc_cities=['Longueuil','Atwater']
# # loc_cities=['Atwater','Candiac']
# # loc_cities=['Atwater','DesBaillets']
# # loc_cities=['Longueuil','DesBaillets']
# loc_eccc=[]
# test='KW'
# season='Summer'
# show=False

# season_list = ['Spring','Summer','Fall','Winter']
# fig_res,ax_res = plt.subplots(nrows=1,ncols=1,figsize=(5,6))#,tight_layout=True)
# plt.title(loc_cities)

# for iseason,season in enumerate(season_list):

#     if len(loc_cities) != 0:
#         ls = df_cities['Twater'][df_cities['Plant'] == loc_cities[0]][df_cities['Season']==season].shape[0]
#     else:
#         ls = df_eccc['Twater'][df_eccc['Plant'] == loc_eccc[0]][df_eccc['Season']==season].shape[0]



#     vars_test = np.zeros((ls,len(loc_cities)+len(loc_eccc)))*np.nan
#     mask = np.zeros((ls))
#     labels=[]
#     for iloc in range(len(loc_cities)):
#         vars_test[:,iloc] = df_cities['Twater'][df_cities['Plant'] == loc_cities[iloc]][df_cities['Season'] == season]

#         # For the purpose of this test, put all negative Tw to zero:
#         # vars_test[:,iloc][vars_test[:,iloc]<=0.5] = 0.0

#         v_arr = np.array(vars_test[:,iloc])
#         mask += np.isnan(v_arr).astype(int)
#         labels.append(loc_cities[iloc])

#     for iloc in range(len(loc_cities),len(loc_eccc)+len(loc_cities)):
#         vars_test[:,iloc] = df_eccc['Twater'][df_eccc['Plant'] == loc_eccc[iloc-len(loc_cities)]][df_eccc['Season'] == season]

#         # For the purpose of this test, put all negative Tw to zero:
#         # vars_test[:,iloc][vars_test[:,iloc]<=0.5] = 0.0

#         v_arr = np.array(vars_test[:,iloc])
#         mask += np.isnan(v_arr).astype(int)
#         labels.append(loc_eccc[iloc-len(loc_cities)])

#     for iloc in range(vars_test.shape[1]):
#         var_plot = vars_test[:,iloc].copy()
#         var_plot[mask > 0] = np.nan


#     # Keep only where the data is available for all variables
#     vars_test = vars_test[mask == 0,:]

#     var_ref = vars_test[:,0]
#     var_rest = vars_test[:,1:]


#     for ir in range(var_rest.shape[1]):
#         paired_res = var_ref - var_rest[:,ir]
#         ax_res.boxplot(paired_res,positions=[iseason],whis=[5,95],showmeans=True,showfliers=False,labels=[season])



