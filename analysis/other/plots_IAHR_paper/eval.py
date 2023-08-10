#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:43:28 2022

@author: Amelie
"""
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

import numpy as np
import pandas as pd
import datetime as dt
import calendar
import matplotlib.pyplot as plt
import statsmodels.api as sm
import cartopy.crs as ccrs
import cartopy as cartopy
from functions import detect_FUD_from_Tw, detrend_ts
from functions_MLR import get_monthly_vars_from_daily

#%%
start_doy_arr = [300,         307,       314,         321,         328,         335,       349]
istart_label = ['Oct. 27th', 'Nov. 3rd', 'Nov. 10th', 'Nov. 17th', 'Nov. 24th', 'Dec. 1st','Dec. 15th']
month_istart = [10,11,11,11,11,12,12]
day_istart   = [27, 3,10,17,24, 1,15]

fdir_r = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/raw/CMC_GHRSST/'
fdir_p = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/CMC_GHRSST/'

verbose = False
p_critical = 0.01
# p_critical = 0.05

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1
time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)

years = np.array([1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020,2021])

ignore_warnings = True
if ignore_warnings:
    import warnings
    warnings.filterwarnings("ignore")

train_yr_start = 1992 # [1992 - 2007] = 16 years
valid_yr_start = 2008 # [2008 - 2013] = 6 years
test_yr_start = 2014  # [2014 - 2019] = 6 years
nsplits = 2019-valid_yr_start+1

years_valid = np.arange(valid_yr_start,test_yr_start)
years_test = np.arange(test_yr_start,2020)

#%%
# Load Twater and FUD data
fp_p_Twater = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/processed/'
Twater_loc_list = ['Longueuil_updated']
station_type = 'cities'
freezeup_opt = 1
freezeup_doy, Twater = detect_FUD_from_Tw(fp_p_Twater,Twater_loc_list,station_type,freezeup_opt,years,time,show=False)
freezeup_doy[np.where(years == 2020)] = np.nan

Twater_mean = np.nanmean(Twater,axis=1)
Twater_mean = np.expand_dims(Twater_mean, axis=1)

# Average (and round) FUD from all locations:
avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
avg_freezeup_doy = np.round(avg_freezeup_doy)

# Get FUD categories for accuracy measure:
it_1992 = np.where(years == 1992)[0][0]
it_2008= np.where(years == 2008)[0][0]
it_valid_start= np.where(years == valid_yr_start)[0][0]
it_test_start= np.where(years == test_yr_start)[0][0]
it_2020= np.where(years == 2020)[0][0]
mean_FUD = np.nanmean(avg_freezeup_doy[it_1992:it_2008])
std_FUD = np.nanstd(avg_freezeup_doy[it_1992:it_2008])
tercile1_FUD = np.nanpercentile(avg_freezeup_doy[it_1992:it_2008],(1/3.)*100)
tercile2_FUD = np.nanpercentile(avg_freezeup_doy[it_1992:it_2008],(2/3.)*100)

FUD_valid = avg_freezeup_doy[it_valid_start:it_test_start]
FUD_test = avg_freezeup_doy[it_test_start:it_2020]

#%%
clim_Acc_valid = np.ones((len(start_doy_arr)))*0.5
clim_Acc_test = np.ones((len(start_doy_arr)))*0.5

mae_valid_tmp = 0
for iyr,yr in enumerate(years_valid):
    mae_valid_tmp += np.abs(FUD_valid[iyr]-mean_FUD )
clim_MAE_valid = np.ones((len(start_doy_arr)))* (mae_valid_tmp/len(years_valid))

mae_test_tmp = 0
for iyr,yr in enumerate(years_test):
    mae_test_tmp += np.abs(FUD_test[iyr]-mean_FUD )
clim_MAE_test = np.ones((len(start_doy_arr)))* (mae_test_tmp/len(years_test))

#%%
cat_Acc_valid = np.array([np.nan,0.1667 ,0.1667 ,0.1667 ,0.1667 , np.nan , np.nan])
cat_Acc_test = np.array([ np.nan,0.1667 ,0.1667 ,0.1667 ,0.1667 , np.nan , np.nan ])

#%%
inpw = 240
nlayers = 3
model_name = 'MLP'
norm_type = 'min_max'
use_softplus = False
lblw = 75
nneurons = 10
ne = 250

#%%
MLR_MAE_valid = np.array([np.nan,6.14421221346556 ,6.14421221346556 ,6.14421221346556 ,6.14421221346556 ,3.1864 , 3.1864])
MLR_Rsqr_valid = np.array([np.nan,0.12592275309127599 ,0.12592275309127599 ,0.12592275309127599 ,0.12592275309127599 ,0.7588 , 0.7588])
MLR_Acc_valid = np.array([np.nan,0.16666666666666666 ,0.16666666666666666 , 0.16666666666666666,0.16666666666666666 ,0.6667 ,0.6667 ])
MLR_MAE_test = np.array([np.nan,6.7823337734481015 ,6.7823337734481015 ,6.7823337734481015 ,6.7823337734481015 ,8.1605 , 8.1605])
MLR_Rsqr_test = np.array([np.nan,0.003938045974391513 , 0.003938045974391513,0.003938045974391513 ,0.003938045974391513 ,0.2215 ,0.2215 ])
MLR_Acc_test = np.array([np.nan,0.3333333333333333 ,0.3333333333333333 ,0.3333333333333333 ,0.3333333333333333 ,0.3333 ,0.3333 ])
MLR_valid_pred_Dec = np.array([342.8704411 , 346.82706737, 357.68778845, 364.50122904,353.87381742, 351.50069555])
MLR_test_pred_Dec = np.array([345.3578129 , 372.33264816, 352.41430559, 356.01098231,346.24746668, 345.18951458])
MLR_valid_pred_Nov = np.array([356.29142952, 348.34411886, 363.15220562, 357.99368043,352.09099827, 352.85043569])
MLR_test_pred_Nov = np.array([354.6893728 , 350.29415394, 349.48380237, 349.19231007,361.04849743, 355.92052426])

#%%
suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_mean'

eval_period = 'valid'
MLpath = '/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/analysis/plots_IAHR_paper/'
fname = MLpath+eval_period+'_'+model_name+'_'+'horizon'+str(lblw)+'_context'+str(inpw)+'_nlayers'+str(nlayers)+'_nneurons'+str(nneurons)+'_nepochs'+str(ne)+'_'+norm_type+'_'+suffix
ML_data_valid = np.load(fname+'.npz',allow_pickle='TRUE')
ML_MAE_valid = ML_data_valid['MAE_arr'][:]
ML_Rsqr_valid = ML_data_valid['Rsqr_arr'][:]
ML_Acc_valid = ML_data_valid['Acc_arr'][:]
ML_valid_pred = ML_data_valid['pred']

eval_period = 'test'
MLpath = '/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/analysis/plots_IAHR_paper/'
fname = MLpath+eval_period+'_'+model_name+'_'+'horizon'+str(lblw)+'_context'+str(inpw)+'_nlayers'+str(nlayers)+'_nneurons'+str(nneurons)+'_nepochs'+str(ne)+'_'+norm_type+'_'+suffix
ML_data_test = np.load(fname+'.npz',allow_pickle='TRUE')
ML_MAE_test = ML_data_test['MAE_arr'][:]
ML_Rsqr_test = ML_data_test['Rsqr_arr'][:]
ML_Acc_test = ML_data_test['Acc_arr'][:]
ML_test_pred = ML_data_test['pred']

istart_show = [1,2,3,4,5]
color_ML = plt.get_cmap('tab20c')(12)
color_MLR = plt.get_cmap('tab20')(16)

fig_valid,ax_valid = plt.subplots(nrows = 2, ncols = 3,figsize=(12,6),sharex=True,sharey='col')

# VALID
ax_valid[0,0].plot(istart_show,clim_MAE_valid[1:6],'o-',color='black')
# ax_valid[0,0].plot(istart_show,MLR_MAE_valid[1:6],'o-',color=color_MLR)
ax_valid[0,0].plot(istart_show,ML_MAE_valid[1:6],'o-',color=color_ML)
ax_valid[0,0].set_title('MAE (days)\n',fontsize=12,fontweight='bold')
ax_valid[0,0].grid(linestyle=':')
ax_valid[0,0].text(-0.9,5,'VALIDATION',fontsize=12,fontweight='bold',rotation =90)

# ax_valid[0,1].plot(istart_show,MLR_Rsqr_valid[1:6],'o-',color=color_MLR)
ax_valid[0,1].plot(istart_show,ML_Rsqr_valid[1:6],'o-',color=color_ML)
ax_valid[0,1].set_title('R$^{2}$\n',fontsize=12,fontweight='bold')
ax_valid[0,1].grid(linestyle=':')

ax_valid[0,2].plot(istart_show,clim_Acc_valid[1:6]*100,'o-',color='black',label='Climatology baseline')
# ax_valid[0,2].plot(istart_show,cat_Acc_valid[1:6]*100,'*:',color='black',label='Categorical baseline')
# ax_valid[0,2].plot(istart_show,MLR_Acc_valid[1:6]*100,'o-',color=color_MLR,label='MLR')
ax_valid[0,2].plot(istart_show,ML_Acc_valid[1:6]*100,'o-',color=color_ML,label='ML - (T$_{air}$)')
ax_valid[0,2].set_title('Categorical \nAccuracy (%)\n',fontsize=12,fontweight='bold')
ax_valid[0,2].grid(linestyle=':')
# ax_valid[0,2].legend()
ax_valid[0,2].set_ylim([0,102])

# ax_valid[0,2].text(-11.9,101,'a)',fontsize=12,fontweight='bold')
# ax_valid[0,2].text(-6,101,'b)',fontsize=12,fontweight='bold')
# ax_valid[0,2].text(-0.33,101,'c)',fontsize=12,fontweight='bold')

#TEST
ax_valid[1,0].plot(istart_show,clim_MAE_test[1:6],'o-',color='black')
# ax_valid[1,0].plot(istart_show,MLR_MAE_test[1:6],'o-',color=color_MLR)
ax_valid[1,0].plot(istart_show,ML_MAE_test[1:6],'o-',color=color_ML)
# ax_valid[1,0].set_title('MAE (days)',fontsize=12,fontweight='bold')
ax_valid[1,0].grid(linestyle=':')
ax_valid[1,0].text(-0.9,6.2,'TEST',fontsize=12,fontweight='bold',rotation =90)

# ax_valid[1,1].plot(istart_show,MLR_Rsqr_test[1:6],'o-',color=color_MLR)
ax_valid[1,1].plot(istart_show,ML_Rsqr_test[1:6],'o-',color=color_ML)
# ax_valid[1,1].set_title('R$^{2}$',fontsize=12,fontweight='bold')
ax_valid[1,1].grid(linestyle=':')

ax_valid[1,2].plot(istart_show,clim_Acc_test[1:6]*100,'o-',color='black',label='Climatology baseline')
# ax_valid[1,2].plot(istart_show,cat_Acc_test[1:6]*100,'*:',color='black',label='Categorical baseline')
# ax_valid[1,2].plot(istart_show,MLR_Acc_test[1:6]*100,'o-',color=color_MLR,label='MLR')
ax_valid[1,2].plot(istart_show,ML_Acc_test[1:6]*100,'o-',color=color_ML,label='ML - (T$_{air}$)')
# ax_valid[1,2].set_title('Cat. Accuracy (%)',fontsize=12,fontweight='bold')
ax_valid[1,2].grid(linestyle=':')
# ax_valid[1,2].legend()
ax_valid[1,2].set_ylim([0,102])

# ax_valid[1,2].text(-11.9,101,'d)',fontsize=12,fontweight='bold')
# ax_valid[1,2].text(-6.,101,'e)',fontsize=12,fontweight='bold')
# ax_valid[1,2].text(-0.33,101,'f)',fontsize=12,fontweight='bold')

for im in range(3):
    plt.sca(ax_valid[1,im])
    plt.xticks(np.arange(1,len(start_doy_arr[1:6])+1), istart_label[1:6] , rotation=90)
    ax_valid[1,im].set_xlabel('Forecast start',fontsize=12)#,fontweight='bold')

fig_valid.subplots_adjust(top=0.87,bottom=0.18,wspace=0.35,hspace=0.20,left=0.086,right=0.78)


# plt.tight_layout()

# ADD TOHER PREDICTORS FOR ML FORECASTS
# suffix_list = ['Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanNAOAOPDO',
#                'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanTot.FDDTot.snowfallAvg.SLP',
#                'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanNAOAOPDOTot.FDDTot.snowfallAvg.SLP']
suffix_list = [
                'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanNAOAOPDO',
                                # 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanTot.FDDTot.snowfallAvg.SLP',
               ]

for itest in range(len(suffix_list)):
    suffix = suffix_list[itest]
    eval_period = 'valid'
    MLpath = '/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/analysis/plots_IAHR_paper/'
    fname = MLpath+eval_period+'_'+model_name+'_'+'horizon'+str(lblw)+'_context'+str(inpw)+'_nlayers'+str(nlayers)+'_nneurons'+str(nneurons)+'_nepochs'+str(ne)+'_'+norm_type+'_'+suffix
    ML_data_valid = np.load(fname+'.npz',allow_pickle='TRUE')
    ML_MAE_valid = ML_data_valid['MAE_arr'][:]
    ML_Rsqr_valid = ML_data_valid['Rsqr_arr'][:]
    ML_Acc_valid = ML_data_valid['Acc_arr'][:]
    ML_valid_pred = ML_data_valid['pred']

    eval_period = 'test'
    MLpath = '/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/analysis/plots_IAHR_paper/'
    fname = MLpath+eval_period+'_'+model_name+'_'+'horizon'+str(lblw)+'_context'+str(inpw)+'_nlayers'+str(nlayers)+'_nneurons'+str(nneurons)+'_nepochs'+str(ne)+'_'+norm_type+'_'+suffix
    ML_data_test = np.load(fname+'.npz',allow_pickle='TRUE')
    ML_MAE_test = ML_data_test['MAE_arr'][:]
    ML_Rsqr_test = ML_data_test['Rsqr_arr'][:]
    ML_Acc_test = ML_data_test['Acc_arr'][:]
    ML_test_pred = ML_data_test['pred']

    #TEST
    color_ML = plt.get_cmap('tab20c')(12+(itest+1)+1)
    ax_valid[0,0].plot(istart_show,ML_MAE_valid[1:6],'o-',color=color_ML)
    ax_valid[0,1].plot(istart_show,ML_Rsqr_valid[1:6],'o-',color=color_ML)
    ax_valid[0,2].plot(istart_show,ML_Acc_valid[1:6]*100,'o-',color=color_ML,label='ML - (T$_{air}$ + NAO, AO, PDO)')

    ax_valid[1,0].plot(istart_show,ML_MAE_test[1:6],'o-',color=color_ML)
    ax_valid[1,1].plot(istart_show,ML_Rsqr_test[1:6],'o-',color=color_ML)
    ax_valid[1,2].plot(istart_show,ML_Acc_test[1:6]*100,'o-',color=color_ML,label='ML - (T$_{air}$ + NAO, AO, PDO)')


# ax_valid[1,2].plot(istart_show,cat_Acc_test[1:6]*100,'*:',color='black')
# ax_valid[0,2].plot(istart_show,cat_Acc_valid[1:6]*100,'*:',color='black')

ax_valid[0,2].legend(loc ='upper left', bbox_to_anchor=(1.05, -0.25, 0.5, 0.5))

fig_valid.subplots_adjust(top=0.87,bottom=0.18,wspace=0.35,hspace=0.20,left=0.086,right=0.78)


#%%

fig_ts,ax_ts = plt.subplots(nrows = 2, ncols = 1,figsize=(10,6),sharex=True)#,sharey=True)

ax_ts[0].plot(years,np.ones(len(years))*(mean_FUD),color=[0.7,0.7,0.7])
ax_ts[0].plot(years,avg_freezeup_doy,'o-',color='black')
ax_ts[1].plot(years,np.ones(len(years))*(mean_FUD),color=[0.7,0.7,0.7])
# ax_ts[1].plot(years,np.ones(len(years))*(365.),color=plt.get_cmap('tab20c')(1))
ax_ts[1].plot(years,avg_freezeup_doy,'o-',color='black')
ax_ts[0].fill_between(years,np.ones(len(years))*(tercile1_FUD),np.ones(len(years))*(tercile2_FUD),color='gray',alpha=0.1)
ax_ts[1].fill_between(years,np.ones(len(years))*(tercile1_FUD),np.ones(len(years))*(tercile2_FUD),color='gray',alpha=0.1)



ax_ts[0].plot(years_valid,MLR_valid_pred_Nov,'o-',color=plt.get_cmap('tab20c')(6))
ax_ts[0].plot(years_valid,MLR_valid_pred_Dec,'o-',color=plt.get_cmap('Set1')(6))
ax_ts[0].plot(years_test,MLR_test_pred_Nov,'o-',color=plt.get_cmap('tab20c')(10))
ax_ts[0].plot(years_test,MLR_test_pred_Dec,'o-',color=plt.get_cmap('Dark2')(0))
ax_ts[0].plot(years_test,MLR_test_pred_Dec,'o-',color=plt.get_cmap('tab20b')(5))

# suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_mean'
suffix= 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanTot.FDDTot.snowfallAvg.SLP'
# suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanNAOAOPDO'
eval_period = 'valid'
MLpath = '/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/analysis/plots_IAHR_paper/'
fname = MLpath+eval_period+'_'+model_name+'_'+'horizon'+str(lblw)+'_context'+str(inpw)+'_nlayers'+str(nlayers)+'_nneurons'+str(nneurons)+'_nepochs'+str(ne)+'_'+norm_type+'_'+suffix
ML_data_valid = np.load(fname+'.npz',allow_pickle='TRUE')
ML_MAE_valid = ML_data_valid['MAE_arr'][:]
ML_Rsqr_valid = ML_data_valid['Rsqr_arr'][:]
ML_Acc_valid = ML_data_valid['Acc_arr'][:]
ML_valid_pred = ML_data_valid['pred']
eval_period = 'test'
# MLpath = '/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/analysis/plots_IAHR_paper/'
fname = MLpath+eval_period+'_'+model_name+'_'+'horizon'+str(lblw)+'_context'+str(inpw)+'_nlayers'+str(nlayers)+'_nneurons'+str(nneurons)+'_nepochs'+str(ne)+'_'+norm_type+'_'+suffix
ML_data_test = np.load(fname+'.npz',allow_pickle='TRUE')
ML_MAE_test = ML_data_test['MAE_arr'][:]
ML_Rsqr_test = ML_data_test['Rsqr_arr'][:]
ML_Acc_test = ML_data_test['Acc_arr'][:]
ML_test_pred = ML_data_test['pred']

istart_show = [1,2,3,4,5]
for ic, istart in enumerate(istart_show):
    if istart== 5:
        ax_ts[1].plot(years_valid,ML_valid_pred[istart,:],'o-',color=plt.get_cmap('Set1')(6), label= 'MLP - '+istart_label[istart])
        # ax_ts[1].plot(years_test,ML_test_pred[istart,:],'o-',color=plt.get_cmap('Dark2')(0), label= 'MLP - '+istart_label[istart])
        ax_ts[1].plot(years_test,ML_test_pred[istart,:],'o-',color=plt.get_cmap('tab20b')(5), label= 'MLP - '+istart_label[istart])
    else:
        ax_ts[1].plot(years_valid,ML_valid_pred[istart,:],'o-',color=plt.get_cmap('tab20c')(7-ic), label= 'MLP - '+istart_label[istart])
        ax_ts[1].plot(years_test,ML_test_pred[istart,:],'o-',color=plt.get_cmap('tab20c')(11-ic), label= 'MLP - '+istart_label[istart])

    if istart== 5:
        ax_ts[1].plot(years_valid,ML_valid_pred[istart,:]+10,':',color=plt.get_cmap('Set1')(6), label= 'MLP - '+istart_label[istart])
        # ax_ts[1].plot(years_test,ML_test_pred[istart,:],'o-',color=plt.get_cmap('Dark2')(0), label= 'MLP - '+istart_label[istart])
        ax_ts[1].plot(years_test,ML_test_pred[istart,:]+10,':',color=plt.get_cmap('tab20b')(5), label= 'MLP - '+istart_label[istart])
    else:
        ax_ts[1].plot(years_valid,ML_valid_pred[istart,:]+10,':',color=plt.get_cmap('tab20c')(7-ic), label= 'MLP - '+istart_label[istart])
        ax_ts[1].plot(years_test,ML_test_pred[istart,:]+10,':',color=plt.get_cmap('tab20c')(11-ic), label= 'MLP - '+istart_label[istart])

    # if istart== 5:
    #     ax_ts[1].plot(years_valid,ML_valid_pred[istart,:]+10,'o-',color=plt.get_cmap('Set1')(6), label= 'MLP - '+istart_label[istart])
    #     ax_ts[1].plot(years_test,ML_test_pred[istart,:]+10,'o-',color=plt.get_cmap('tab20b')(5), label= 'MLP - '+istart_label[istart])
    # else:
    #     ax_ts[1].plot(years_valid,ML_valid_pred[istart,:]+10,'o-',color=plt.get_cmap('tab20c')(7-ic), label= 'MLP - '+istart_label[istart])
    #     ax_ts[1].plot(years_test,ML_test_pred[istart,:]+10,'o-',color=plt.get_cmap('tab20c')(11-ic), label= 'MLP - '+istart_label[istart])


ax_ts[1].set_xlabel('Years')
ax_ts[0].set_ylabel('FUD (DOY)')
ax_ts[1].set_ylabel('FUD (DOY)')
ax_ts[0].set_ylim([337,375])
# ax_ts[1].set_ylim([337,375])
ax_ts[1].set_ylim([337,383])
ax_ts[1].set_xlim([2006.3,2019.8])

fig_ts.subplots_adjust(top=0.92,bottom=0.08,wspace=0.35,hspace=0.27,left=0.086,right=0.78)
ax_ts[0].text(2005.,378,'a) Multiple Linear Regression Forecasts',fontsize=12,fontweight='bold')
# ax_ts[1].text(2005.,378,'b) Machine Learning Forecasts',fontsize=12,fontweight='bold')
# ax_ts[0].text(2005.,385,'a) Multiple Linear Regression Forecasts',fontsize=12,fontweight='bold')
ax_ts[1].text(2005.,387,'b) Machine Learning Forecasts',fontsize=12,fontweight='bold')

ax_ts[1].text(2021.2,378,' Forecast\nstart date',fontsize=10)
ax_ts[1].text(2020.1,376,'valid.',fontsize=10,style='italic')
ax_ts[1].text(2023.,376,'test',fontsize=10,style='italic')
ax_ts[1].text(2021.3,373-3.5,'Nov. 3rd',fontsize=10)
ax_ts[1].text(2021.3,369-3.,'Nov. 10th',fontsize=10)
ax_ts[1].text(2021.3,365-2.5,'Nov. 17th',fontsize=10)
ax_ts[1].text(2021.3,361-2.,'Nov. 24th',fontsize=10)
ax_ts[1].text(2021.3,357-1.5,'Dec. 1st',fontsize=10)

ax_ts[0].text(2021.2,370,' Forecast\nstart date',fontsize=10)
ax_ts[0].text(2020.1,368,'valid.',fontsize=10,style='italic')
ax_ts[0].text(2023.,368,'test',fontsize=10,style='italic')
ax_ts[0].text(2021.3,367-4,'Nov. 1st',fontsize=10)
ax_ts[0].text(2021.3,364-4,'Dec. 1st',fontsize=10)
# ax_ts[1].legend(loc ='upper left', bbox_to_anchor=(1.05, 0.25, 0.5, 0.5))


# fig_ts.savefig('ML_and_MLR_predictions', dpi=700)



#%%

fig_ts,ax_ts = plt.subplots(nrows = 2, ncols = 1,figsize=(10,6),sharex=True)#,sharey=True)

ax_ts[0].plot(years,np.ones(len(years))*(mean_FUD),color=[0.7,0.7,0.7])
ax_ts[0].plot(years,avg_freezeup_doy,'o-',color='black')
ax_ts[1].plot(years,np.ones(len(years))*(mean_FUD),color=[0.7,0.7,0.7])
# ax_ts[1].plot(years,np.ones(len(years))*(365.),color=plt.get_cmap('tab20c')(1))
ax_ts[1].plot(years,avg_freezeup_doy,'o-',color='black')
ax_ts[0].fill_between(years,np.ones(len(years))*(tercile1_FUD),np.ones(len(years))*(tercile2_FUD),color='gray',alpha=0.1)
ax_ts[1].fill_between(years,np.ones(len(years))*(tercile1_FUD),np.ones(len(years))*(tercile2_FUD),color='gray',alpha=0.1)



ax_ts[0].plot(years_valid,MLR_valid_pred_Nov,'o-',color=plt.get_cmap('tab20c')(6))
ax_ts[0].plot(years_valid,MLR_valid_pred_Dec,'o-',color=plt.get_cmap('Set1')(6))
ax_ts[0].plot(years_test,MLR_test_pred_Nov,'o-',color=plt.get_cmap('tab20c')(10))
ax_ts[0].plot(years_test,MLR_test_pred_Dec,'o-',color=plt.get_cmap('Dark2')(0))
ax_ts[0].plot(years_test,MLR_test_pred_Dec,'o-',color=plt.get_cmap('tab20b')(5))

# suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_mean'
# suffix= 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanTot.FDDTot.snowfallAvg.SLP'
suffix = 'Avg.Twatersin(DOY)cos(DOY)Avg.Ta_meanNAOAOPDO'
eval_period = 'valid'
MLpath = '/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/analysis/plots_IAHR_paper/'
fname = MLpath+eval_period+'_'+model_name+'_'+'horizon'+str(lblw)+'_context'+str(inpw)+'_nlayers'+str(nlayers)+'_nneurons'+str(nneurons)+'_nepochs'+str(ne)+'_'+norm_type+'_'+suffix
ML_data_valid = np.load(fname+'.npz',allow_pickle='TRUE')
ML_MAE_valid = ML_data_valid['MAE_arr'][:]
ML_Rsqr_valid = ML_data_valid['Rsqr_arr'][:]
ML_Acc_valid = ML_data_valid['Acc_arr'][:]
ML_valid_pred = ML_data_valid['pred']
eval_period = 'test'
# MLpath = '/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/analysis/plots_IAHR_paper/'
fname = MLpath+eval_period+'_'+model_name+'_'+'horizon'+str(lblw)+'_context'+str(inpw)+'_nlayers'+str(nlayers)+'_nneurons'+str(nneurons)+'_nepochs'+str(ne)+'_'+norm_type+'_'+suffix
ML_data_test = np.load(fname+'.npz',allow_pickle='TRUE')
ML_MAE_test = ML_data_test['MAE_arr'][:]
ML_Rsqr_test = ML_data_test['Rsqr_arr'][:]
ML_Acc_test = ML_data_test['Acc_arr'][:]
ML_test_pred = ML_data_test['pred']

istart_show = [1,2,3,4,5]
for ic, istart in enumerate(istart_show):
    if istart== 5:
        ax_ts[1].plot(years_valid,ML_valid_pred[istart,:],'o-',color=plt.get_cmap('Set1')(6), label= 'MLP - '+istart_label[istart])
        # ax_ts[1].plot(years_test,ML_test_pred[istart,:],'o-',color=plt.get_cmap('Dark2')(0), label= 'MLP - '+istart_label[istart])
        ax_ts[1].plot(years_test,ML_test_pred[istart,:],'o-',color=plt.get_cmap('tab20b')(5), label= 'MLP - '+istart_label[istart])
    else:
        ax_ts[1].plot(years_valid,ML_valid_pred[istart,:],'o-',color=plt.get_cmap('tab20c')(7-ic), label= 'MLP - '+istart_label[istart])
        ax_ts[1].plot(years_test,ML_test_pred[istart,:],'o-',color=plt.get_cmap('tab20c')(11-ic), label= 'MLP - '+istart_label[istart])

    # if istart== 5:
    #     ax_ts[1].plot(years_valid,ML_valid_pred[istart,:]+10,':',color=plt.get_cmap('Set1')(6), label= 'MLP - '+istart_label[istart])
    #     # ax_ts[1].plot(years_test,ML_test_pred[istart,:],'o-',color=plt.get_cmap('Dark2')(0), label= 'MLP - '+istart_label[istart])
    #     ax_ts[1].plot(years_test,ML_test_pred[istart,:]+10,':',color=plt.get_cmap('tab20b')(5), label= 'MLP - '+istart_label[istart])
    # else:
    #     ax_ts[1].plot(years_valid,ML_valid_pred[istart,:]+10,':',color=plt.get_cmap('tab20c')(7-ic), label= 'MLP - '+istart_label[istart])
    #     ax_ts[1].plot(years_test,ML_test_pred[istart,:]+10,':',color=plt.get_cmap('tab20c')(11-ic), label= 'MLP - '+istart_label[istart])

    # if istart== 5:
    #     ax_ts[1].plot(years_valid,ML_valid_pred[istart,:]+10,'o-',color=plt.get_cmap('Set1')(6), label= 'MLP - '+istart_label[istart])
    #     ax_ts[1].plot(years_test,ML_test_pred[istart,:]+10,'o-',color=plt.get_cmap('tab20b')(5), label= 'MLP - '+istart_label[istart])
    # else:
    #     ax_ts[1].plot(years_valid,ML_valid_pred[istart,:]+10,'o-',color=plt.get_cmap('tab20c')(7-ic), label= 'MLP - '+istart_label[istart])
    #     ax_ts[1].plot(years_test,ML_test_pred[istart,:]+10,'o-',color=plt.get_cmap('tab20c')(11-ic), label= 'MLP - '+istart_label[istart])


ax_ts[1].set_xlabel('Years')
ax_ts[0].set_ylabel('FUD (DOY)')
ax_ts[1].set_ylabel('FUD (DOY)')
ax_ts[0].set_ylim([337,375])
ax_ts[1].set_ylim([337,375])
# ax_ts[1].set_ylim([337,383])
ax_ts[1].set_xlim([2006.3,2019.8])

fig_ts.subplots_adjust(top=0.92,bottom=0.08,wspace=0.35,hspace=0.27,left=0.086,right=0.78)
ax_ts[0].text(2005.,378,'a) Multiple Linear Regression Forecasts',fontsize=12,fontweight='bold')
# ax_ts[1].text(2005.,378,'b) Machine Learning Forecasts',fontsize=12,fontweight='bold')
# ax_ts[0].text(2005.,385,'a) Multiple Linear Regression Forecasts',fontsize=12,fontweight='bold')
ax_ts[1].text(2005.,387,'b) Machine Learning Forecasts',fontsize=12,fontweight='bold')

ax_ts[1].text(2021.2,378,' Forecast\nstart date',fontsize=10)
ax_ts[1].text(2020.1,376,'valid.',fontsize=10,style='italic')
ax_ts[1].text(2023.,376,'test',fontsize=10,style='italic')
ax_ts[1].text(2021.3,373-3.5,'Nov. 3rd',fontsize=10)
ax_ts[1].text(2021.3,369-3.,'Nov. 10th',fontsize=10)
ax_ts[1].text(2021.3,365-2.5,'Nov. 17th',fontsize=10)
ax_ts[1].text(2021.3,361-2.,'Nov. 24th',fontsize=10)
ax_ts[1].text(2021.3,357-1.5,'Dec. 1st',fontsize=10)

ax_ts[0].text(2021.2,370,' Forecast\nstart date',fontsize=10)
ax_ts[0].text(2020.1,368,'valid.',fontsize=10,style='italic')
ax_ts[0].text(2023.,368,'test',fontsize=10,style='italic')
ax_ts[0].text(2021.3,367-4,'Nov. 1st',fontsize=10)
ax_ts[0].text(2021.3,364-4,'Dec. 1st',fontsize=10)
# ax_ts[1].legend(loc ='upper left', bbox_to_anchor=(1.05, 0.25, 0.5, 0.5))


# fig_ts.savefig('ML_and_MLR_predictions', dpi=700)

