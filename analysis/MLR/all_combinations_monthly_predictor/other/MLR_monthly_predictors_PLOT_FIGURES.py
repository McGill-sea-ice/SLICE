#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 12:34:16 2022

@author: amelie
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
import itertools
import calendar
import matplotlib.pyplot as plt
import statsmodels.api as sm
from functions import detect_FUD_from_Tw, detrend_ts
from functions_MLR import get_monthly_vars_from_daily, remove_collinear_features
from functions_MLR import find_models,eval_accuracy_multiple_models,make_metric_df
from functions_MLR import find_all_column_combinations
import sklearn.metrics as metrics
#%%
save_plots = False
save_folder = './output/all_coefficients_significant_05/'

valid_scheme = 'LOOk'
# file_name = './output/MLR_monthly_pred_Jan1st_maxpred5_valid_scheme_LOOk'
# file_name = './output/MLR_monthly_pred_Dec1st_maxpred5_valid_scheme_LOOk'
# file_name = './output/MLR_monthly_pred_Nov1st_maxpred6_valid_scheme_LOOk'
# file_name = './output/MLR_monthly_pred_p05_Jan1st_maxpred4_valid_scheme_LOOk'

file_name = save_folder +'MLR_monthly_pred_varslast6monthsp05only_Jan1st_maxpred4_valid_scheme_LOOk'
file_name = save_folder +'MLR_monthly_pred_varslast6months_Jan1st_maxpred4_valid_scheme_LOOk'



df_valid_all = pd.read_pickle(file_name+'_df_valid_all')
df_test_all = pd.read_pickle(file_name+'_df_test_all')
df_clim_valid_all = pd.read_pickle(file_name+'_df_clim_valid_all')
df_clim_test_all = pd.read_pickle(file_name+'_df_clim_test_all')
df_select_valid = pd.read_pickle(file_name+'_df_select_valid')
df_select_test = pd.read_pickle(file_name+'_df_select_test')
pred_df_clean = pd.read_pickle(file_name+'_pred_df_clean')


data = np.load(file_name+'.npz', allow_pickle=True)

valid_years=data['valid_years']
test_years=data['test_years']
plot_label = data['plot_label']
years = data['years']
avg_freezeup_doy = data['avg_freezeup_doy']
p_critical = data['p_critical']
date_ref = data['date_ref']
date_start = data['date_start']
date_end = data['date_end']
time = data['time']
start_yr = data['start_yr']
end_yr = data['end_yr']
train_yr_start = data['train_yr_start']
valid_yr_start = data['valid_yr_start']
test_yr_start = data['test_yr_start']
nsplits = data['nsplits']
nfolds = data['nfolds']
max_pred = data['max_pred']
valid_metric = data['valid_metric']


istart_labels = ['Sept. 1st', 'Oct. 1st','Nov. 1st','Dec. 1st','Jan. 1st']
istart_savelabels = ['Sept1st', 'Oct1st','Nov1st','Dec1st','Jan1st']
istart = 4
ind = 0

best_n = np.min((12,len(df_valid_all)))
#%%


#%%
# Plot forecasts:
fig,ax = plt.subplots()
ax.plot(years,avg_freezeup_doy ,'o-',color='k')
plot_label = istart_labels[istart]+': '
for p in range(len(df_select_test['predictors'])):
    plot_label += df_select_test['predictors'][p]+','

if (valid_scheme == 'standard')|(valid_scheme == 'rolling'):
    ax.plot(valid_years,df_select_valid['valid_predictions'],'o-',color=plt.get_cmap('tab20c')(7-(2*ind)))
ax.plot(test_years,df_select_test['test_predictions'],'o-',color=plt.get_cmap('tab20c')((2*ind)),label=plot_label)

avg = 0
for iyr in range(len(years)):
    avg += np.nanmean(np.delete(avg_freezeup_doy, iyr))/len(years)

ax.plot(test_years,np.ones(len(test_years))*avg, '--', color='gray')


ax.legend()
ax.set_ylabel('FUD (day of year)')
ax.set_xlabel('Year')
print('test MAE:',df_select_test['test_MAE'])
print('test RMSE:',df_select_test['test_RMSE'])
print('test R2:',df_select_test['test_R2'])
print('test R2_adj:',df_select_test['test_R2adj'])
print('test accuracy:',df_select_test['test_Acc'])
print('test 7-day accuracy:',np.sum(np.abs(avg_freezeup_doy[:]-df_select_test['test_predictions']) <= 7)/np.sum(~np.isnan(avg_freezeup_doy)))

#%%
# PLOT TOP N MODELS FOR RMSE AND MAE

if ~np.all(np.isnan(pd.to_numeric(df_valid_all['valid_RMSE']))):
    figMAE,axMAE = plt.subplots(figsize=[12.5,4.5])

    # axMAE.text(0.1,  1., 'A', transform=ax.transAxes,
    #       fontsize=16, fontweight='bold', va='top', ha='right')

    axMAE.plot(years,avg_freezeup_doy ,'o-',color='k')

    df_valid_MAE = pd.DataFrame(pd.to_numeric(df_valid_all['valid_MAE']))
    df_subset = df_valid_MAE.nsmallest(best_n, 'valid_MAE')
    for nmodel in range(np.min([len(df_subset),best_n])):
        nmodel = df_subset.index.values[nmodel]
        plot_label = ''
        for i in range(len(df_test_all.iloc[nmodel]['predictors'])):
            plot_label += df_test_all.iloc[nmodel]['predictors'][i]
            if i != (len(df_test_all.iloc[nmodel]['predictors'])-1):
                plot_label += ', '
        axMAE.plot(test_years,df_test_all.iloc[nmodel]['test_predictions'],'o-',label=plot_label)
    axMAE.legend()
    axMAE.set_ylabel('FUD')
    axMAE.set_xlabel('Year')

    # plt.title('Top '+str(best_n)+' MLR models according to '+ 'MAE' + ' - ' + valid_scheme + ' valid. ('+ istart_labels[istart] +')')
    # print(np.nanmean(np.abs(avg_freezeup_doy[:]-df_test_all.iloc[nmodel]['test_predictions'])))
    if save_plots:
        plt.savefig(save_folder+'best_MAE_MLR_models.png', bbox_inches='tight', dpi=600)


#%%
    figRMSE,axRMSE = plt.subplots(figsize=[12.5,4.5])
    axRMSE.plot(years,avg_freezeup_doy ,'o-',color='k')

    df_valid_RMSE = pd.DataFrame(pd.to_numeric(df_valid_all['valid_RMSE']))
    df_subset = df_valid_RMSE.nsmallest(best_n, 'valid_RMSE')
    for nmodel in range(np.min([len(df_subset),best_n])):
        nmodel = df_subset.index.values[nmodel]
        plot_label = df_test_all.iloc[nmodel]['predictors']
        axRMSE.plot(test_years,df_test_all.iloc[nmodel]['test_predictions'],'o-',label=plot_label)
    axRMSE.legend()
    axRMSE.set_ylabel('FUD')
    axRMSE.set_xlabel('Year')
    plt.title('Top '+str(best_n)+' MLR models according to '+ 'RMSE' + ' - ' + valid_scheme + ' valid. ('+ istart_labels[istart] +')')


    figss,axss = plt.subplots(figsize=[12.5,4.5])
    axss.plot(years,avg_freezeup_doy ,'o-',color='k')

    df_valid_RMSE = pd.DataFrame(pd.to_numeric(df_valid_all['valid_ss']))
    df_subset = df_valid_RMSE.nlargest(best_n, 'valid_ss')
    for nmodel in range(np.min([len(df_subset),best_n])):
        nmodel = df_subset.index.values[nmodel]
        plot_label = df_test_all.iloc[nmodel]['predictors']
        axss.plot(test_years,df_test_all.iloc[nmodel]['test_predictions'],'o-',label=plot_label)
    axss.legend()
    axss.set_ylabel('FUD')
    axss.set_xlabel('Year')
    plt.title('Top '+str(best_n)+' MLR models according to '+ 'Skill Score' + ' - ' + valid_scheme + ' valid. ('+ istart_labels[istart] +')')


#%%
# PLOT DISTRIBUTION OF VALIDATION MAE AND RMSE DURING K-FOLD CV
if (valid_scheme == 'LOOk') | (valid_scheme == 'standardk'):
    if ~np.all(np.isnan(pd.to_numeric(df_valid_all['valid_RMSE']))):
        figCV_MAE,axCV_MAE = plt.subplots(figsize=[12.5,4.5])
        df_valid_MAE = pd.DataFrame(pd.to_numeric(df_valid_all['valid_MAE']))
        df_clim_valid_MAE = pd.DataFrame(pd.to_numeric(df_clim_valid_all['clim_valid_MAE']))
        df_subset = df_valid_MAE.nsmallest(best_n, 'valid_MAE')

        axCV_MAE.plot([0,27],[df_clim_valid_all.iloc[0]['clim_valid_MAE'],df_clim_valid_all.iloc[0]['clim_valid_MAE']],'-',color=plt.get_cmap('tab20')(14),label='FUD climatology')
        axCV_MAE.fill_between([0,27],[df_clim_valid_all.iloc[0]['clim_valid_MAE']-df_clim_valid_all.iloc[0]['clim_valid_MAE_std'],df_clim_valid_all.iloc[0]['clim_valid_MAE']-df_clim_valid_all.iloc[0]['clim_valid_MAE_std']],[df_clim_valid_all.iloc[0]['clim_valid_MAE']+df_clim_valid_all.iloc[0]['clim_valid_MAE_std'],df_clim_valid_all.iloc[0]['clim_valid_MAE']+df_clim_valid_all.iloc[0]['clim_valid_MAE_std']],color=plt.get_cmap('tab20')(14),alpha=0.2)

        for imodel in range(np.min([len(df_subset),best_n])):
            nmodel = df_subset.index.values[imodel]
            plot_label = ''
            for i in range(len(df_test_all.iloc[nmodel]['predictors'])):
                plot_label += df_test_all.iloc[nmodel]['predictors'][i]
                if i != (len(df_test_all.iloc[nmodel]['predictors'])-1):
                    plot_label += ', '
            if valid_scheme == 'LOOk':

                for iyr in range(len(test_years)):
                    axCV_MAE.plot(((imodel+1)+32*(imodel+1))+(iyr*1),df_valid_all.iloc[nmodel]['valid_MAE_mean'][iyr],'.',color=plt.get_cmap('tab20')(imodel*2))
                    axCV_MAE.plot([((imodel+1)+32*(imodel+1))+(iyr*1),((imodel+1)+32*(imodel+1))+(iyr*1)],[df_valid_all.iloc[nmodel]['valid_MAE_min'][iyr],df_valid_all.iloc[nmodel]['valid_MAE_max'][iyr]],'-',color=plt.get_cmap('tab20')(imodel*2),linewidth=0.5)

                    axCV_MAE.plot((iyr*1),df_clim_valid_all.iloc[0]['clim_valid_MAE_mean'][iyr],'.',color=plt.get_cmap('tab20')(14))
                    axCV_MAE.plot([(iyr*1),(iyr*1)],[df_clim_valid_all.iloc[0]['clim_valid_MAE_min'][iyr],df_clim_valid_all.iloc[0]['clim_valid_MAE_max'][iyr]],'-',color=plt.get_cmap('tab20')(14),linewidth=0.5)

                axCV_MAE.plot([((imodel+1)+32*(imodel+1))+(0),((imodel+1)+32*(imodel+1))+(27)],[df_valid_all.iloc[nmodel]['valid_MAE'],df_valid_all.iloc[nmodel]['valid_MAE']],'-',color=plt.get_cmap('tab20')(imodel*2),label=plot_label)
                # axCV_MAE.plot([(imodel+32*imodel)+(0),(imodel+32*imodel)+(27)],[df_test_all.iloc[nmodel]['test_MAE'],df_test_all.iloc[nmodel]['test_MAE']],'--',color=plt.get_cmap('tab20')(imodel*2))
                axCV_MAE.fill_between([((imodel+1)+32*(imodel+1))+(0),((imodel+1)+32*(imodel+1))+(27)],[df_valid_all.iloc[nmodel]['valid_MAE']-df_valid_all.iloc[nmodel]['valid_MAE_std'],df_valid_all.iloc[nmodel]['valid_MAE']-df_valid_all.iloc[nmodel]['valid_MAE_std']],[df_valid_all.iloc[nmodel]['valid_MAE']+df_valid_all.iloc[nmodel]['valid_MAE_std'],df_valid_all.iloc[nmodel]['valid_MAE']+df_valid_all.iloc[nmodel]['valid_MAE_std']],color=plt.get_cmap('tab20')(imodel*2),alpha=0.2)

            if valid_scheme == 'standardk':
                axCV_MAE.plot((imodel+32*imodel),df_valid_all.iloc[nmodel]['valid_MAE_mean'],'x',color=plt.get_cmap('tab20')(imodel*2))
                axCV_MAE.plot([(imodel+32*imodel),(imodel+32*imodel)],[df_valid_all.iloc[nmodel]['valid_MAE_min'],df_valid_all.iloc[nmodel]['valid_MAE_max']],'-',color=plt.get_cmap('tab20')(imodel*2),linewidth=0.5)
                # axCV_MAE.plot([(imodel+32*imodel)+(-2),(imodel+32*imodel)+(2)],[df_test_all.iloc[nmodel]['test_MAE'],df_test_all.iloc[nmodel]['test_MAE']],'--',color=plt.get_cmap('tab20')(imodel*2))

        axCV_MAE.set_ylabel('MAE$_{CV}$ (days)')
        axCV_MAE.legend()
        axCV_MAE.set_xticks([])
        if istart_labels[istart] == 'Jan. 1st':
            plt.title('MLR Perfect Forecast Exp.')
        else:
            plt.title(istart_labels[istart])

        if save_plots:
            plt.savefig(save_folder+'valid_MAE_CV.png', bbox_inches='tight', dpi=600)


        figCV_RMSE,axCV_RMSE = plt.subplots(figsize=[12.5,4.5])
        df_valid_RMSE = pd.DataFrame(pd.to_numeric(df_valid_all['valid_RMSE']))
        df_subset = df_valid_RMSE.nsmallest(best_n, 'valid_RMSE')

        for imodel in range(np.min([len(df_subset),best_n])):
            nmodel = df_subset.index.values[imodel]
            plot_label = ''
            for i in range(len(df_test_all.iloc[nmodel]['predictors'])):
                plot_label += df_test_all.iloc[nmodel]['predictors'][i]
                if i != (len(df_test_all.iloc[nmodel]['predictors'])-1):
                    plot_label += ', '
            if valid_scheme == 'LOOk':
                for iyr in range(len(test_years)):
                    axCV_RMSE.plot((imodel+32*imodel)+(iyr*1),df_valid_all.iloc[nmodel]['valid_RMSE_mean'][iyr],'.',color=plt.get_cmap('tab20')(imodel*2))
                    axCV_RMSE.plot([(imodel+32*imodel)+(iyr*1),(imodel+32*imodel)+(iyr*1)],[df_valid_all.iloc[nmodel]['valid_RMSE_min'][iyr],df_valid_all.iloc[nmodel]['valid_RMSE_max'][iyr]],'-',color=plt.get_cmap('tab20')(imodel*2),linewidth=0.5)
                axCV_RMSE.plot([(imodel+32*imodel)+(0),(imodel+32*imodel)+(27)],[df_valid_all.iloc[nmodel]['valid_RMSE'],df_valid_all.iloc[nmodel]['valid_RMSE']],'-',color=plt.get_cmap('tab20')(imodel*2),label=plot_label)
                # axCV_RMSE.plot([(imodel+32*imodel)+(0),(imodel+32*imodel)+(27)],[df_test_all.iloc[nmodel]['test_RMSE'],df_test_all.iloc[nmodel]['test_RMSE']],'--',color=plt.get_cmap('tab20')(imodel*2))
                axCV_RMSE.fill_between([(imodel+32*imodel)+(0),(imodel+32*imodel)+(27)],[df_valid_all.iloc[nmodel]['valid_RMSE']-df_valid_all.iloc[nmodel]['valid_RMSE_std'],df_valid_all.iloc[nmodel]['valid_RMSE']-df_valid_all.iloc[nmodel]['valid_RMSE_std']],[df_valid_all.iloc[nmodel]['valid_RMSE']+df_valid_all.iloc[nmodel]['valid_RMSE_std'],df_valid_all.iloc[nmodel]['valid_RMSE']+df_valid_all.iloc[nmodel]['valid_RMSE_std']],color=plt.get_cmap('tab20')(imodel*2),alpha=0.2)
            if valid_scheme == 'standardk':
                axCV_RMSE.plot((imodel+32*imodel),df_valid_all.iloc[nmodel]['valid_RMSE_mean'],'x',color=plt.get_cmap('tab20')(imodel*2))
                axCV_RMSE.plot([(imodel+32*imodel),(imodel+32*imodel)],[df_valid_all.iloc[nmodel]['valid_RMSE_min'],df_valid_all.iloc[nmodel]['valid_RMSE_max']],'-',color=plt.get_cmap('tab20')(imodel*2),linewidth=0.5)
                # axCV_RMSE.plot([(imodel+32*imodel)+(-2),(imodel+32*imodel)+(2)],[df_test_all.iloc[nmodel]['test_RMSE'],df_test_all.iloc[nmodel]['test_RMSE']],'--',color=plt.get_cmap('tab20')(imodel*2))
        axCV_RMSE.set_ylabel('RMSE (days)')
        axCV_RMSE.legend()
        if istart_labels[istart] == 'Jan. 1st':
            plt.title('MLR Perfect Forecast Exp.')
        else:
            plt.title(istart_labels[istart])

        figCV_ss,axCV_ss = plt.subplots(figsize=[12.5,4.5])
        df_valid_ss = pd.DataFrame(pd.to_numeric(df_valid_all['valid_ss']))
        df_subset = df_valid_ss.nlargest(best_n, 'valid_ss')

        for imodel in range(np.min([len(df_subset),best_n])):
            nmodel = df_subset.index.values[imodel]
            plot_label = ''
            for i in range(len(df_test_all.iloc[nmodel]['predictors'])):
                plot_label += df_test_all.iloc[nmodel]['predictors'][i]
                if i != (len(df_test_all.iloc[nmodel]['predictors'])-1):
                    plot_label += ', '
            if valid_scheme == 'LOOk':
                for iyr in range(len(test_years)):
                    axCV_ss.plot((imodel+32*imodel)+(iyr*1),df_valid_all.iloc[nmodel]['valid_ss_mean'][iyr],'x',color=plt.get_cmap('tab20')(imodel*2))
                    axCV_ss.plot([(imodel+32*imodel)+(iyr*1),(imodel+32*imodel)+(iyr*1)],[df_valid_all.iloc[nmodel]['valid_ss_min'][iyr],df_valid_all.iloc[nmodel]['valid_ss_max'][iyr]],'-',color=plt.get_cmap('tab20')(imodel*2),linewidth=0.5)
                axCV_ss.plot([(imodel+32*imodel)+(0),(imodel+32*imodel)+(27)],[df_valid_all.iloc[nmodel]['valid_ss'],df_valid_all.iloc[nmodel]['valid_ss']],'-',color=plt.get_cmap('tab20')(imodel*2),label=plot_label)
                # axCV_ss.plot([(imodel+32*imodel)+(0),(imodel+32*imodel)+(27)],[df_test_all.iloc[nmodel]['test_ss'],df_test_all.iloc[nmodel]['test_ss']],'--',color=plt.get_cmap('tab20')(imodel*2))
                axCV_ss.fill_between([(imodel+32*imodel)+(0),(imodel+32*imodel)+(27)],[df_valid_all.iloc[nmodel]['valid_ss']-df_valid_all.iloc[nmodel]['valid_ss_std'],df_valid_all.iloc[nmodel]['valid_ss']-df_valid_all.iloc[nmodel]['valid_ss_std']],[df_valid_all.iloc[nmodel]['valid_ss']+df_valid_all.iloc[nmodel]['valid_ss_std'],df_valid_all.iloc[nmodel]['valid_ss']+df_valid_all.iloc[nmodel]['valid_ss_std']],color=plt.get_cmap('tab20')(imodel*2),alpha=0.2)
            if valid_scheme == 'standardk':
                axCV_ss.plot((imodel+32*imodel),df_valid_all.iloc[nmodel]['valid_ss_mean'],'x',color=plt.get_cmap('tab20')(imodel*2))
                axCV_ss.plot([(imodel+32*imodel),(imodel+32*imodel)],[df_valid_all.iloc[nmodel]['valid_ss_min'],df_valid_all.iloc[nmodel]['valid_ss_max']],'-',color=plt.get_cmap('tab20')(imodel*2),linewidth=0.5)
                # axCV_ss.plot([(imodel+32*imodel)+(-2),(imodel+32*imodel)+(2)],[df_test_all.iloc[nmodel]['test_ss'],df_test_all.iloc[nmodel]['test_ss']],'--',color=plt.get_cmap('tab20')(imodel*2))
            print('--------')
            print('MAE (valid, test):',df_valid_all.iloc[nmodel]['valid_MAE'],df_test_all.iloc[nmodel]['test_MAE'])
            print('RMSE (valid, test):',df_valid_all.iloc[nmodel]['valid_RMSE'],df_test_all.iloc[nmodel]['test_RMSE'])
            print('SS (valid, test):',df_valid_all.iloc[nmodel]['valid_ss'],df_test_all.iloc[nmodel]['test_ss'])
            print('Accuracy (valid, test):',df_valid_all.iloc[nmodel]['valid_Acc'],df_test_all.iloc[nmodel]['test_Acc'])
            print('Rsqr (test):',df_test_all.iloc[nmodel]['test_R2'])
            print('Rsqr_adj (test):',df_test_all.iloc[nmodel]['test_R2adj'])
        axCV_ss.set_ylabel('Skill Score')
        axCV_ss.legend()
        if istart_labels[istart] == 'Jan. 1st':
            plt.title('MLR Perfect Forecast Exp.')
        else:
            plt.title(istart_labels[istart])

#%%
# Load monthly predictor data
fpath_mp = local_path+'slice/data/monthly_predictors/'
monthly_pred = np.load(fpath_mp+'monthly_vars'+'_'+'Longueuil'+'.npz')
monthly_pred_data = monthly_pred['data']
pred_names = monthly_pred['labels']

# Keep only 1992 to 2020 (inclusive)
it_start = np.where(years == start_yr)[0][0]
it_end = np.where(years == end_yr)[0][0]

# years = years[it_start:it_end+1]
# avg_freezeup_doy = avg_freezeup_doy[it_start:it_end+1]
monthly_pred_data = monthly_pred_data[:,it_start:it_end+1,:]

p_list = ['Tot. precip.', 'Avg. SH (sfc)','Avg. LH (sfc)','Avg. SW down (sfc)', 'Avg. Ta_mean','Avg. Ta_mean','Avg. Ta_mean','Tot. FDD','Tot. TDD','Tot. TDD','Avg. SLP','Tot. snowfall','Avg. cloud cover','Avg. level Ottawa River','AO','NAO','NAO','PDO','PDO','Avg. windspeed','Tot. snowfall','Avg. Ta_mean','Tot. TDD','Tot. FDD','Avg. discharge St-L. River','Avg. level Ottawa River','NAO','AO'  ]
m_list = [9,               9,              9,              9,                    12,            5,             9,             11,        5,        11,         11,        11,             9,                 10,                       11,   9,    11,   1,    2,   10,              12,             12,            12,        12,        12,                          12,                       12,   12  ]

month_str = ['Jan. ', 'Feb. ', 'Mar. ', 'Apr. ', 'May ', 'Jun. ','Jul. ', 'Aug. ', 'Sep. ', 'Oct. ', 'Nov. ','Dec.']


pred_arr = np.zeros((monthly_pred_data.shape[1],len(p_list)))*np.nan
col = []
for i in range(len(p_list)):
    ipred = np.where(pred_names == p_list[i])[0][0]
    pred_arr[:,i] = monthly_pred_data[ipred,:,m_list[i]-1]
    col.append(month_str[m_list[i]-1]+p_list[i])
pred_df =  pd.DataFrame(pred_arr,columns=col)



#%%
# SeptNAO2022 = -0.702
# OctCloudCover2022 = 0.557011
# SeptSH2022 = -76754.74194477081
# # SeptCloudCover2022 = 0.66412

# var2022 = np.zeros((1,3))
# var2022[0,0]=OctCloudCover2022
# var2022[0,1]=SeptNAO2022
# var2022[0,2]=SeptSH2022


y = avg_freezeup_doy
y = pred_df['Sep. Avg. SW down (sfc)']
# y = pred_df['Sep. Avg. LH (sfc)']
y = pred_df['Sep. Avg. SH (sfc)']
y = pred_df['Sep. Tot. precip.']
# y = pred_df['Sep. Avg. Ta_mean']
X = pred_df[[
                    'Sep. Avg. cloud cover',
                    # 'Sep. NAO',
                    # 'Sep. Avg. SH (sfc)',
                    # 'Nov. Avg. Ta_mean',
                    # 'Nov. Tot. snowfall',
                    # 'Dec.Avg. Ta_mean',
                    # 'Dec. Tot. snowfall',
                    ]
                  ]
m = sm.OLS(y, sm.add_constant(X,has_constant='skip'), missing='drop').fit()
print(m.summary())

# y_pred = m.predict(sm.add_constant(pred_df_clean[[
#                     'Oct. Avg. cloud cover',
#                     'Sep. NAO',
#                     'Sep. Avg. SH (sfc)',
#                     'Nov. Avg. Ta_mean',
#                     'Nov. Tot. snowfall',
#                     'Dec. Avg. Ta_mean',
#                     'Dec. Tot. snowfall',
#                     ]
#                   ],has_constant='skip'))



# # y_pred_2022 = m.predict(sm.add_constant(pd.DataFrame(var2022),has_constant='add'))

# plt.figure()
# plt.plot(years,y, 'o-', color='k')
# plt.plot(years,y_pred, 'o-', color = plt.get_cmap('tab20')(0))
# # plt.plot(2022,y_pred_2022,'*', markersize = 15,color = plt.get_cmap('tab20')(1))
# # print(y_pred_2022)


# #%%



# i_5 = []
# i_4 = []
# i_3 = []
# i_2 = []
# i_1 = []

# for i in range(len(df_valid_all)):
#     if len(df_valid_all['predictors'].iloc[i]) == 5:
#         i_5.append(i)
#     if len(df_valid_all['predictors'].iloc[i]) == 4:
#         i_4.append(i)
#     if len(df_valid_all['predictors'].iloc[i]) == 3:
#         i_3.append(i)
#     if len(df_valid_all['predictors'].iloc[i]) == 2:
#         i_2.append(i)
#     if len(df_valid_all['predictors'].iloc[i]) == 1:
#         i_1.append(i)


# df_valid_all_5 = df_valid_all.iloc[i_5].copy()
# df_valid_all_4 = df_valid_all.iloc[i_4].copy()
# df_valid_all_3 = df_valid_all.iloc[i_3].copy()
# df_valid_all_2 = df_valid_all.iloc[i_2].copy()
# df_valid_all_1 = df_valid_all.iloc[i_1].copy()

# df_test_all_5 = df_test_all.iloc[i_5].copy()
# df_test_all_4 = df_test_all.iloc[i_4].copy()
# df_test_all_3 = df_test_all.iloc[i_3].copy()
# df_test_all_2 = df_test_all.iloc[i_2].copy()
# df_test_all_1 = df_test_all.iloc[i_1].copy()



#%%
f,ax = plt.subplots(nrows = 4, ncols = 1, sharex = True)
ax[0].plot(years, avg_freezeup_doy-df_select_test['test_predictions'],'*-')
ax[3].plot(years, pred_df['Sep. Avg. cloud cover'],'*-')
# ax[0].plot(years, pred_df['Dec. NAO'],'*-')
ax[1].plot(years, pred_df['Dec.Avg. Ta_mean'],'*-')
ax[2].plot(years, pred_df['Nov. Tot. snowfall'],'*-')


#%%
# PLOT RESIDUALS ANALYSIS
res = (avg_freezeup_doy-df_select_test['test_predictions'])
norm_res = (res-np.nanmean(res))/np.std(res)

fig,ax= plt.subplots(ncols=3,figsize=[12.5,4.2])
for i, label in enumerate(('A', 'B', 'C')):
    ax[i].text(-0.1, 1.1, label, transform=ax[i].transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

ax[0].scatter(df_select_test['test_predictions'], norm_res)
ax[1].boxplot(norm_res)
sm.qqplot(norm_res, line='45',ax=ax[2])

ax[0].set_ylabel('Normalized residuals')
ax[1].set_ylabel('Normalized residuals')
ax[0].set_xlabel('Predicted FUD')
ax[1].set_xticks([])
fig.tight_layout()
plt.savefig(save_folder+'test_residuals.png', dpi=600)

#%%
# PLOT RESIDUALS ANALYSIS
res = (avg_freezeup_doy-df_select_test['test_predictions'])
norm_res = (res-np.nanmean(res))/np.std(res)

fig,ax= plt.subplots(ncols=3,figsize=[12.5,4.2])
for i, label in enumerate(('A', 'B', 'C')):
    ax[i].text(-0.1, 1.1, label, transform=ax[i].transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

ax[0].scatter(df_select_test['test_predictions'], res)
ax[1].boxplot(res)
sm.qqplot(res, line='s',ax=ax[2])

ax[0].set_ylabel('Residual (days)')
ax[1].set_ylabel('Residual (days)')
ax[0].set_xlabel('Predicted FUD')
ax[1].set_xticks([])
fig.tight_layout()
plt.savefig(save_folder+'test_residuals.png', dpi=600)