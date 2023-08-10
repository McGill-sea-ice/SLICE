#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:09:08 2021

@author: Amelie
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 06:54:46 2021

@author: Amelie
"""
import numpy as np
import scipy
from functools import reduce

import pandas as pd
import statsmodels.api as sm

import datetime as dt
import calendar
import itertools
import matplotlib.pyplot as plt


import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)


from functions import running_nanmean,find_freezeup_Tw_all_yrs
from functions import linear_fit,r_confidence_interval, detrend_ts
from functions import get_window_monthly_vars, get_window_vars, deseasonalize_ts
from functions import predicted_r2
from functions_MLR import freezeup_multiple_linear_regression_model, MLR_model_analysis
# from functions_MLR import freezeup_multiple_linear_regression_model
#%%

years = [1991,1992,1993,1994,1995,1996,
          1997,1998,1999,2000,2001,
          2002,2003,2004,2005,2006,
          2007,2008,2009,2010,2011,
          2012,2013,2014,2015,2016,2017,
          2018,2019,2020,2021]

fp = '../../../data/processed/'

date_ref = dt.date(1900,1,1)
date_start = dt.date(1980,1,1)
date_end = dt.date(2021,12,31)
ndays = (date_end-date_start).days + 1

time = np.arange((date_start-date_ref).days, (date_end-date_ref).days+1)
years = np.array(years)
p_critical = 0.05


#%%
nyears = years.shape[0]-1
training_size = 14

rolling_training = True
anomaly = False

# xall = All years
# yall =  Observed FUD for all years
# xf = Years of forecast period
# yf_true = Observed FUD during forecast period
# yf = Predictions during forecast period
# xh = Years of hindcast period
# yh = Predictions during hindcast period

#%%
def MLR_model_analysis(fpath,npred,start_date,training_size,pc,show=True):

    if fpath == './MLR_save_v3_daily_NAO/':
        model_data = np.load(fpath+'models'+str(npred)+'_ts'+str(training_size)+'_'+start_date+'TEST.npz',allow_pickle=True)
    else:
        model_data = np.load(fpath+'models'+str(npred)+'_ts'+str(training_size)+'_'+start_date+'.npz',allow_pickle=True)

    models_h = model_data['models_h']
    x_models = model_data['x_models']
    yh_true_models = model_data['yh_true_models']
    yh_models = model_data['yh_models']
    xh_models = model_data['xh_models']
    xf_models = model_data['xf_models']
    yf_models = model_data['yf_models']
    yall = model_data['yall'][0]
    xall = model_data['xall'][0]

    mae_f_models = model_data['mae_f_models']
    rmse_f_models = model_data['rmse_f_models']
    Rsqr_f_models = model_data['Rsqr_f_models']
    Rsqr_adj_f_models = model_data['Rsqr_adj_f_models']
    std_f_models = model_data['std_f_models']
    mae_h_models = model_data['mae_h_models']
    rmse_h_models = model_data['rmse_h_models']
    Rsqr_h_models =  model_data['Rsqr_h_models']
    Rsqr_adj_h_models = model_data['Rsqr_adj_h_models']
    Rsqr_pred_h_models = model_data['Rsqr_pred_h_models']
    std_h_models = model_data['std_h_models']

    nmodels = x_models.shape[0]

    pred_list = []
    for p in range(npred):
        pred_list += ['Pred. '+str(p+1)]

    week_accuracy_f = []
    week_accuracy_h = []
    for i in range(nmodels):
        week_accuracy_f += [np.sum(np.abs(np.squeeze(yf_models[i])-yall[training_size:-1]) <= 7 )/ np.sum(~np.isnan(yall[training_size:-1]))]
        atmp = []
        for ih in range(Rsqr_h_models.shape[1]):
            atmp += [np.sum(np.abs(np.squeeze(yh_true_models[i,ih,:])-yh_models[i,ih,:]) <= 7 )/ np.sum(~np.isnan(yh_true_models[i,ih,:]))]
        week_accuracy_h += [atmp]

    week_accuracy_f = np.array(week_accuracy_f)
    week_accuracy_h = np.array(week_accuracy_h)

    arr1 = x_models
    arr2 = np.array([mae_f_models,rmse_f_models,Rsqr_f_models,Rsqr_adj_f_models,std_f_models,week_accuracy_f]).T
    arr_c = np.array(np.zeros((arr1.shape[0],arr1.shape[1]+arr2.shape[1])), dtype=object)
    arr_c[:,0:arr1.shape[1]] = arr1
    arr_c[:,arr1.shape[1]:arr1.shape[1]+arr2.shape[1]] = arr2
    stats_f_all = pd.DataFrame(arr_c,
                      columns=pred_list+['MAE', 'RMSE','Rsqr','Rsqr_adj','sig_err','week_accuracy'])

    arr1 = x_models
    arr2 = np.array([np.nanmean(mae_h_models,axis=1),np.nanmean(rmse_h_models,axis=1),np.nanmean(Rsqr_h_models,axis=1),np.nanmean(Rsqr_adj_h_models,axis=1),np.nanmean(std_h_models,axis=1),np.nanmean(week_accuracy_h,axis=1)]).T
    arr_c = np.array(np.zeros((arr1.shape[0],arr1.shape[1]+arr2.shape[1])), dtype=object)
    arr_c[:,0:arr1.shape[1]] = arr1
    arr_c[:,arr1.shape[1]:arr1.shape[1]+arr2.shape[1]] = arr2
    stats_h_all = pd.DataFrame(arr_c,
                      columns=pred_list+['MAE', 'RMSE','Rsqr','Rsqr_adj','sig_err','week_accuracy'])



    if show:
        fig, ax = plt.subplots()
        plt.title('Models '+start_date+' - '+str(npred)+' predictors')
        plt.ylim(-0.2,1)

        fig_FUD, ax_FUD = plt.subplots()
        plt.title('Models '+start_date+' - '+str(npred)+' predictors')
        ax_FUD.set_xlabel('Year')
        ax_FUD.set_ylabel('Freeze-up'+' Anomaly (days)'*anomaly+ ' DOY'*(not anomaly))

    p = 0
    arr_c_f = np.array(np.zeros((nmodels,npred+7)), dtype=object)
    arr_c_h = np.array(np.zeros((nmodels,npred+7)), dtype=object)

    for i in range(nmodels):
        xim = [x_models[i][m] for m in range(npred)]

        cond_ftest_h_all = (np.all([ (models_h[i][im].f_pvalue <= pc) for im in range(len(models_h[i]))]))
        cond_ftest_h_half = (np.sum([ (models_h[i][im].f_pvalue <= pc) for im in range(len(models_h[i]))]) >= len(models_h[i])/2.)
        cond_ftest_h_mean = (np.nanmean([ (models_h[i][im].f_pvalue) for im in range(len(models_h[i]))]) <= pc)

        cond_pred_pval_onepermodel = ( np.all([ np.any(models_h[i][im].pvalues[1:] <= pc) for im in range(len(models_h[i])) ]) )
        # Above: this test that there is at least one regression coefficient that is significantly different from zero (other than the constant) in each hindcast model.
        # IDEA: I could add another condition to check whether it is always the same variable that has a significant coefficient over all the hindcast models so that it is a robust signal.

        cond_Rsqr_adj_h_all = np.all(Rsqr_adj_h_models[i]>0)
        cond_Rsqr_adj_f_all = np.all(Rsqr_adj_f_models[i]>0)

        if start_date == 'Nov1':
            print(cond_ftest_h_all,cond_ftest_h_half,cond_ftest_h_mean,'|--|',cond_pred_pval_onepermodel,cond_Rsqr_adj_h_all,cond_Rsqr_adj_f_all)
        # print(cond_ftest_h_half & cond_Rsqr_adj_h_all & cond_Rsqr_adj_f_all)

        # if cond_ftest_h_all:
        #     print(cond_ftest_h_all,cond_pred_pval_onepermodel,cond_Rsqr_adj_h_all,cond_Rsqr_adj_f_all)

        # if i == 267:
        #     print(x_models[i])
        #     print(cond_ftest_h_all,cond_ftest_h_half,cond_ftest_h_mean,'|--|',cond_pred_pval_onepermodel,cond_Rsqr_adj_h_all,cond_Rsqr_adj_f_all)
        #     print([ (models_h[i][im].f_pvalue <= pc) for im in range(len(models_h[i]))])
        #     print([ np.any(models_h[i][im].pvalues[1:] <= pc) for im in range(len(models_h[i])) ])

        # if (np.all(Rsqr_adj_h_models[i]>0):
        # if (np.all(Rsqr_adj_h_models[i] > 0)) & (np.all(Rsqr_h_models[i] >= r_confidence_interval(0,0.05,training_size,tailed='two')[1]**2.)):
        # if (np.all(Rsqr_adj_h_models[i] > 0)) & (np.nanmean(Rsqr_h_models[i]) >= r_confidence_interval(0,0.05,training_size,tailed='two')[1]**2.):
        # if (np.nanmean(Rsqr_h_models[i]) >= r_confidence_interval(0,0.05,training_size,tailed='two')[1]**2.):
        # if (np.nanmean(Rsqr_adj_h_models[i]) > 0) & (np.nanmean(Rsqr_h_models[i]) >= r_confidence_interval(0,0.05,training_size,tailed='two')[1]**2.):
        # if np.all(Rsqr_pred_h_models[i]>0):
        # if np.all(Rsqr_adj_f_models[i]>0):
        # if (Rsqr_f_models[i] >= r_confidence_interval(0,0.05,xall[training_size+1:].shape[0],tailed='two')[1]**2.) & (np.all(Rsqr_h_models[i] >= r_confidence_interval(0,0.05,training_size,tailed='two')[1]**2.)):
        # if (Rsqr_f_models[i] >= r_confidence_interval(0,0.05,xall[training_size+1:].shape[0],tailed='two')[1]**2.) & (Rsqr_adj_f_models[i] > 0) & (np.all(Rsqr_h_models[i] >= r_confidence_interval(0,0.05,training_size,tailed='two')[1]**2.)) & (np.all(Rsqr_adj_h_models[i] > 0)):

        # condTOT = (cond_ftest_h_half & cond_Rsqr_adj_h_all & cond_Rsqr_adj_f_all)
        condTOT = (cond_ftest_h_all & cond_pred_pval_onepermodel & cond_Rsqr_adj_h_all & cond_Rsqr_adj_f_all)
        # condTOT = (cond_ftest_h_half & cond_pred_pval_onepermodel & cond_Rsqr_adj_h_all & cond_Rsqr_adj_f_all)

        if condTOT & (~np.any(['discharge' in xim[j] for j in range(npred)])):

            arr1_fi = x_models[i]
            arr2_fi = np.array([i,mae_f_models[i],rmse_f_models[i],Rsqr_f_models[i],Rsqr_adj_f_models[i],std_f_models[i],week_accuracy_f[i]]).T
            arr_c_f[p,0:npred] = arr1_fi
            arr_c_f[p,npred:npred+7] = arr2_fi
            # stats_f_all = pd.DataFrame(arr_c,
            #                   columns=pred_list+['MAE', 'RMSE','Rsqr','Rsqr_adj','sig_err'])

            arr1_hi = x_models[i]
            arr2_hi = np.array([i,np.nanmean(mae_h_models[i]),np.nanmean(rmse_h_models[i]),np.nanmean(Rsqr_h_models[i]),np.nanmean(Rsqr_adj_h_models[i]),np.nanmean(std_h_models[i]),np.nanmean(week_accuracy_h[i])]).T
            arr_c_h[p,0:npred] = arr1_hi
            arr_c_h[p,npred:npred+7] = arr2_hi

            if show:
                ax.plot(xall[training_size:-1],Rsqr_h_models[i],'-',color=plt.get_cmap('tab20')(p*2))
                ax.plot(xall[training_size:-1],Rsqr_adj_h_models[i],'--',color=plt.get_cmap('tab20')(p*2))
                ax.plot(xall[training_size:-1],Rsqr_pred_h_models[i],':',color=plt.get_cmap('tab20')(p*2))
                ax.text(2008,0.95-p*0.06,'Rsqr forecast = {:03.2f}'.format(Rsqr_f_models[i]),color=plt.get_cmap('tab20')(p*2))
                ax.plot(xall[training_size:-1],np.ones(len(Rsqr_h_models[i]))*Rsqr_f_models[i],'-',color=plt.get_cmap('tab20')(p*2+1))

                ax_FUD.plot(xall,yall,'o-',color='k')
                ax_FUD.plot(xf_models[i],yf_models[i], 'o:',color=plt.get_cmap('tab20')(p*2))

            p += 1

    arr_c_f = arr_c_f[0:p,:]
    arr_c_h = arr_c_h[0:p,:]

    stats_f_s = pd.DataFrame(arr_c_f,
                             columns=pred_list+['imodel','MAE','RMSE','Rsqr','Rsqr_adj','sig_err','week_accuracy'])
    stats_h_s = pd.DataFrame(arr_c_h,
                             columns=pred_list+['imodel','MAE','RMSE','Rsqr','Rsqr_adj','sig_err','week_accuracy'])

    return stats_h_all, stats_f_all, stats_h_s, stats_f_s



#%%
fpath = './MLR_save_v3/'
fpath = './MLR_save_v3_daily_NAO/'
show = False
training_size = 15

start_date = 'Dec1'

npred = 2
stats_h_Dec1_2pred_all15, stats_f_Dec1_2pred_all15, stats_h_Dec1_2pred_s15, stats_f_Dec1_2pred_s15  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# pred2_list15 = stats_f_Dec1_2pred_s15[['Pred. 1','Pred. 2']]
npred = 3
stats_h_Dec1_3pred_all15, stats_f_Dec1_3pred_all15, stats_h_Dec1_3pred_s15, stats_f_Dec1_3pred_s15  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# pred3_list15 = stats_f_Dec1_3pred_s15[['Pred. 1','Pred. 2','Pred. 3']]
npred = 4
stats_h_Dec1_4pred_all15, stats_f_Dec1_4pred_all15, stats_h_Dec1_4pred_s15, stats_f_Dec1_4pred_s15  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# pred4_list15 = stats_f_Dec1_4pred_s15[['Pred. 1','Pred. 2','Pred. 3','Pred. 4']]
# npred = 5
# stats_h_Dec1_5pred_all15, stats_f_Dec1_5pred_all15, stats_h_Dec1_5pred_s15, stats_f_Dec1_5pred_s15  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# # pred5_list15 = stats_f_Dec1_5pred_s15[['Pred. 1','Pred. 2','Pred. 3','Pred. 4','Pred. 5']]
#%%
start_date = 'Nov1'
npred = 2
stats_h_Nov1_2pred_all15, stats_f_Nov1_2pred_all15, stats_h_Nov1_2pred_s15, stats_f_Nov1_2pred_s15  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# pred2_list15 = stats_f_Nov1_2pred_s15[['Pred. 1','Pred. 2']]
npred = 3
stats_h_Nov1_3pred_all15, stats_f_Nov1_3pred_all15, stats_h_Nov1_3pred_s15, stats_f_Nov1_3pred_s15  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# pred3_list15 = stats_f_Nov1_3pred_s15[['Pred. 1','Pred. 2','Pred. 3']]
npred = 4
stats_h_Nov1_4pred_all15, stats_f_Nov1_4pred_all15, stats_h_Nov1_4pred_s15, stats_f_Nov1_4pred_s15  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# pred4_list15 = stats_f_Nov1_4pred_s15[['Pred. 1','Pred. 2','Pred. 3','Pred. 4']]
# npred = 5
# stats_h_Nov1_5pred_all15, stats_f_Nov1_5pred_all15, stats_h_Nov1_5pred_s15, stats_f_Nov1_5pred_s15  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# # pred5_list15 = stats_f_Nov1_5pred_s15[['Pred. 1','Pred. 2','Pred. 3','Pred. 4','Pred. 5']]

#%%
# fpath = './MLR_save_v2/'
# show = False
# training_size = 12

# start_date = 'Dec1'
# npred = 2
# stats_h_Dec1_2pred_all12, stats_f_Dec1_2pred_all12, stats_h_Dec1_2pred_s12, stats_f_Dec1_2pred_s12  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# # pred2_list12 = stats_f_Dec1_2pred_s12[['Pred. 1','Pred. 2']]
# npred = 3
# stats_h_Dec1_3pred_all12, stats_f_Dec1_3pred_all12, stats_h_Dec1_3pred_s12, stats_f_Dec1_3pred_s12  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# # pred3_list12 = stats_f_Dec1_3pred_s12[['Pred. 1','Pred. 2','Pred. 3']]
# npred = 4
# stats_h_Dec1_4pred_all12, stats_f_Dec1_4pred_all12, stats_h_Dec1_4pred_s12, stats_f_Dec1_4pred_s12  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# # pred4_list12 = stats_f_Dec1_4pred_s12[['Pred. 1','Pred. 2','Pred. 3','Pred. 4']]

# start_date = 'Nov1'
# npred = 2
# stats_h_Nov1_2pred_all12, stats_f_Nov1_2pred_all12, stats_h_Nov1_2pred_s12, stats_f_Nov1_2pred_s12  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# # pred2_list142 = stats_f_Nov1_2pred_s12[['Pred. 1','Pred. 2']]
# npred = 3
# stats_h_Nov1_3pred_all12, stats_f_Nov1_3pred_all12, stats_h_Nov1_3pred_s12, stats_f_Nov1_3pred_s12  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# # pred3_list12 = stats_f_Nov1_3pred_s12[['Pred. 1','Pred. 2','Pred. 3']]
# npred = 4
# stats_h_Nov1_4pred_all12, stats_f_Nov1_4pred_all12, stats_h_Nov1_4pred_s12, stats_f_Nov1_4pred_s12  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# # pred4_list12 = stats_f_Nov1_4pred_s12[['Pred. 1','Pred. 2','Pred. 3','Pred. 4']]


#%%
# fpath = './MLR_save_v3/'
# show = True
# training_size = 14

# start_date = 'Dec1'
# npred = 2
# stats_h_Dec1_2pred_all14, stats_f_Dec1_2pred_all14, stats_h_Dec1_2pred_s14, stats_f_Dec1_2pred_s14  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# # pred2_list14 = stats_f_Dec1_2pred_s14[['Pred. 1','Pred. 2']]
# npred = 3
# stats_h_Dec1_3pred_all14, stats_f_Dec1_3pred_all14, stats_h_Dec1_3pred_s14, stats_f_Dec1_3pred_s14  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# # pred3_list14 = stats_f_Dec1_3pred_s14[['Pred. 1','Pred. 2','Pred. 3']]
# npred = 4
# stats_h_Dec1_4pred_all14, stats_f_Dec1_4pred_all14, stats_h_Dec1_4pred_s14, stats_f_Dec1_4pred_s14  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# # pred4_list14 = stats_f_Dec1_4pred_s14[['Pred. 1','Pred. 2','Pred. 3','Pred. 4']]

# start_date = 'Nov1'
# npred = 2
# stats_h_Nov1_2pred_all14, stats_f_Nov1_2pred_all14, stats_h_Nov1_2pred_s14, stats_f_Nov1_2pred_s14  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# # pred2_list14 = stats_f_Nov1_2pred_s14[['Pred. 1','Pred. 2']]
# npred = 3
# stats_h_Nov1_3pred_all14, stats_f_Nov1_3pred_all14, stats_h_Nov1_3pred_s14, stats_f_Nov1_3pred_s14  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# # pred3_list14 = stats_f_Nov1_3pred_s14[['Pred. 1','Pred. 2','Pred. 3']]
# npred = 4
# stats_h_Nov1_4pred_all14, stats_f_Nov1_4pred_all14, stats_h_Nov1_4pred_s14, stats_f_Nov1_4pred_s14  = MLR_model_analysis(fpath,npred,start_date,training_size,p_critical,show=show)
# # pred4_list14 = stats_f_Nov1_4pred_s14[['Pred. 1','Pred. 2','Pred. 3','Pred. 4']]


#%%
# df_f = stats_f_Dec1_4pred_s15
# df_h = stats_h_Dec1_4pred_s15
# npred = 4
df_f = stats_f_Dec1_3pred_s15
df_h = stats_h_Dec1_3pred_s15
npred = 3

nhead = 5

if_MAE = np.array(df_f.sort_values('MAE',ascending=True).head(nhead )['imodel'],dtype=int)
if_RMSE = np.array(df_f.sort_values('RMSE',ascending=True).head(nhead )['imodel'],dtype=int)
if_Rsqr = np.array(df_f.sort_values('Rsqr',ascending=False).head(nhead )['imodel'],dtype=int)
if_Rsqr_adj = np.array(df_f.sort_values('Rsqr_adj',ascending=False).head(nhead )['imodel'],dtype=int)
if_sig_err = np.array(df_f.sort_values('sig_err',ascending=True).head(nhead)['imodel'],dtype=int)
if_weekly = np.array(df_f.sort_values('week_accuracy',ascending=False).head(nhead)['imodel'],dtype=int)

if_same = reduce(np.intersect1d, (if_MAE, if_RMSE, if_Rsqr, if_Rsqr_adj, if_sig_err, if_weekly))
if_unique = np.unique([if_MAE, if_RMSE, if_Rsqr, if_Rsqr_adj, if_sig_err,if_weekly])

ih_MAE = np.array(df_h.sort_values('MAE',ascending=True).head(nhead )['imodel'],dtype=int)
ih_RMSE = np.array(df_h.sort_values('RMSE',ascending=True).head(nhead )['imodel'],dtype=int)
ih_Rsqr = np.array(df_h.sort_values('Rsqr',ascending=False).head(nhead )['imodel'],dtype=int)
ih_Rsqr_adj = np.array(df_h.sort_values('Rsqr_adj',ascending=False).head(nhead )['imodel'],dtype=int)
ih_sig_err = np.array(df_h.sort_values('sig_err',ascending=True).head(nhead)['imodel'],dtype=int)
ih_weekly = np.array(df_h.sort_values('week_accuracy',ascending=False).head(nhead)['imodel'],dtype=int)

ih_same = reduce(np.intersect1d, (ih_MAE, ih_RMSE, ih_Rsqr, ih_Rsqr_adj, ih_sig_err,ih_weekly))
ih_unique = np.unique([ih_MAE, ih_RMSE, ih_Rsqr, ih_Rsqr_adj, ih_sig_err, ih_weekly])


ind_unique = np.unique(np.concatenate((ih_same, if_same)))
df_f_select_3pred = df_f.iloc[[df_f['imodel'][i] in ind_unique for i in range(df_f.shape[0])]]
df_h_select_3pred = df_h.iloc[[df_h['imodel'][i] in ind_unique for i in range(df_h.shape[0])]]

ind_unique = np.unique(np.concatenate((ih_unique, if_unique)))
df_f_select_3pred_unique = df_f.iloc[[df_f['imodel'][i] in ind_unique for i in range(df_f.shape[0])]]
df_h_select_3pred_unique = df_h.iloc[[df_h['imodel'][i] in ind_unique for i in range(df_h.shape[0])]]


#%%
df_f = stats_f_Dec1_4pred_s15
df_h = stats_h_Dec1_4pred_s15
npred = 4

nhead = 5

if_MAE = np.array(df_f.sort_values('MAE',ascending=True).head(nhead )['imodel'],dtype=int)
if_RMSE = np.array(df_f.sort_values('RMSE',ascending=True).head(nhead )['imodel'],dtype=int)
if_Rsqr = np.array(df_f.sort_values('Rsqr',ascending=False).head(nhead )['imodel'],dtype=int)
if_Rsqr_adj = np.array(df_f.sort_values('Rsqr_adj',ascending=False).head(nhead )['imodel'],dtype=int)
if_sig_err = np.array(df_f.sort_values('sig_err',ascending=True).head(nhead)['imodel'],dtype=int)
if_weekly = np.array(df_f.sort_values('week_accuracy',ascending=False).head(nhead)['imodel'],dtype=int)

if_same = reduce(np.intersect1d, (if_MAE, if_RMSE, if_Rsqr, if_Rsqr_adj, if_sig_err, if_weekly))
if_unique = np.unique([if_MAE, if_RMSE, if_Rsqr, if_Rsqr_adj, if_sig_err,if_weekly])

ih_MAE = np.array(df_h.sort_values('MAE',ascending=True).head(nhead )['imodel'],dtype=int)
ih_RMSE = np.array(df_h.sort_values('RMSE',ascending=True).head(nhead )['imodel'],dtype=int)
ih_Rsqr = np.array(df_h.sort_values('Rsqr',ascending=False).head(nhead )['imodel'],dtype=int)
ih_Rsqr_adj = np.array(df_h.sort_values('Rsqr_adj',ascending=False).head(nhead )['imodel'],dtype=int)
ih_sig_err = np.array(df_h.sort_values('sig_err',ascending=True).head(nhead)['imodel'],dtype=int)
ih_weekly = np.array(df_h.sort_values('week_accuracy',ascending=False).head(nhead)['imodel'],dtype=int)

ih_same = reduce(np.intersect1d, (ih_MAE, ih_RMSE, ih_Rsqr, ih_Rsqr_adj, ih_sig_err,ih_weekly))
ih_unique = np.unique([ih_MAE, ih_RMSE, ih_Rsqr, ih_Rsqr_adj, ih_sig_err, ih_weekly])

ind_unique = np.unique(np.concatenate((ih_same, if_same)))
df_f_select_4pred = df_f.iloc[[df_f['imodel'][i] in ind_unique for i in range(df_f.shape[0])]]
df_h_select_4pred = df_h.iloc[[df_h['imodel'][i] in ind_unique for i in range(df_h.shape[0])]]

ind_unique = np.unique(np.concatenate((ih_unique, if_unique)))
df_f_select_4pred_unique = df_f.iloc[[df_f['imodel'][i] in ind_unique for i in range(df_f.shape[0])]]
df_h_select_4pred_unique = df_h.iloc[[df_h['imodel'][i] in ind_unique for i in range(df_h.shape[0])]]



#%%

# fpath = './MLR_save_v3_daily_NAO/'
# npred = 3
# training_size = 15
# model_data = np.load(fpath+'models'+str(npred)+'_ts'+str(training_size)+'_Dec1.npz',allow_pickle=True)

# x_models = model_data['x_models']
# # yh_true_models = model_data['yh_true_models_'+str(npred)]
# yh_models = model_data['yh_models']
# xh_models = model_data['xh_models']
# Rsqr_f_models = model_data['Rsqr_f_models']
# Rsqr_adj_f_models = model_data['Rsqr_adj_f_models']
# mae_f_models = model_data['mae_f_models']
# rmse_f_models = model_data['rmse_f_models']
# Rsqr_h_models = model_data['Rsqr_h_models']
# Rsqr_adj_h_models = model_data['Rsqr_adj_h_models']
# mae_h_models = model_data['mae_h_models']
# rmse_h_models = model_data['rmse_h_models']

