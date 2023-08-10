#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 19:06:20 2022

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
import matplotlib.pyplot as plt
import statsmodels.api as sm
from functions import detect_FUD_from_Tw
from functions_MLR import remove_collinear_features
from functions_MLR import find_models,eval_accuracy_multiple_models
from functions_MLR import find_all_column_combinations

#%%
def select_best_model(valid_metric,df_valid_all,df_test_all):
    if (valid_metric == 'Acc'):
        df_select_tmp_valid= df_valid_all[df_valid_all['valid_Acc']==df_valid_all['valid_Acc'].max()]
        df_select_tmp_test= df_test_all[df_valid_all['valid_Acc']==df_valid_all['valid_Acc'].max()]
        valid_metric2 = 'R2adj'
        # valid_metric2 = 'MAE'
        if (valid_metric2 == 'R2') | (valid_metric2 == 'R2adj'):
            df_select_valid = df_select_tmp_valid.loc[pd.to_numeric(df_select_tmp_valid['valid_'+valid_metric2]).idxmax()]
            df_select_test = df_select_tmp_test.loc[pd.to_numeric(df_select_tmp_valid['valid_'+valid_metric2]).idxmax()]
        else:
            df_select_valid = df_select_tmp_valid.loc[pd.to_numeric(df_select_tmp_valid['valid_'+valid_metric2]).idxmin()]
            df_select_test = df_select_tmp_test.loc[pd.to_numeric(df_select_tmp_valid['valid_'+valid_metric2]).idxmin()]
    else:
        if (valid_metric == 'R2') | (valid_metric == 'R2adj'):
            df_select_valid = df_valid_all.loc[pd.to_numeric(df_valid_all['valid_'+valid_metric]).idxmax()]
            df_select_test = df_test_all.loc[pd.to_numeric(df_valid_all['valid_'+valid_metric]).idxmax()]
        else:
            df_select_valid = df_valid_all.loc[pd.to_numeric(df_valid_all['valid_'+valid_metric]).idxmin()]
            df_select_test = df_test_all.loc[pd.to_numeric(df_valid_all['valid_'+valid_metric]).idxmin()]

    return df_select_valid, df_select_test




def get_train_valid_test_metric_dfs(valid_scheme,pred_df_clean,targets,all_combinations,nsplits,nfolds,years,train_yr_start,valid_yr_start,test_yr_start,tercile1_FUD,tercile2_FUD,p_critical,max_pred):

    if valid_scheme == 'standard':
        it_train_start = np.where(years == train_yr_start)[0][0]
        it_train_end = np.where(years == valid_yr_start)[0][0]
        it_valid_start = np.where(years == valid_yr_start)[0][0]
        it_valid_end = np.where(years == test_yr_start)[0][0]

        it_test_start = np.where(years == test_yr_start)[0][0]
        it_test_end = np.where(years == 2020)[0][0]

        df_train = pred_df_clean[it_train_start:it_train_end].copy()
        df_valid = pred_df_clean[it_valid_start:it_valid_end].copy()
        df_test = pred_df_clean[it_test_start:it_test_end].copy()
        train_years = years[it_train_start:it_train_end]
        valid_years = years[it_valid_start:it_valid_end]
        test_years = years[it_test_start:it_test_end]

        target_train = targets[it_train_start:it_train_end].copy()
        target_valid = targets[it_valid_start:it_valid_end].copy()
        target_test = targets[it_test_start:it_test_end].copy()

        clim_valid = np.ones((len(target_valid)))*np.nanmean(targets[it_train_start:it_train_end])
        clim_test = np.ones((len(target_test)))*np.nanmean(targets[it_train_start:it_valid_end])


        # Get FUD categories for accuracy measure:
        FUD_cat_valid = np.zeros(target_valid.shape)*np.nan
        for iyr in range(FUD_cat_valid.shape[0]):
            if target_valid[iyr] <= tercile1_FUD:
                FUD_cat_valid[iyr] = -1
            elif target_valid[iyr] > tercile2_FUD:
                FUD_cat_valid[iyr] = 1
            else:
                FUD_cat_valid[iyr] = 0

        FUD_cat_test = np.zeros(target_test.shape)*np.nan
        for iyr in range(FUD_cat_test.shape[0]):
            if target_test[iyr] <= tercile1_FUD:
                FUD_cat_test[iyr] = -1
            elif target_test[iyr] > tercile2_FUD:
                FUD_cat_test[iyr] = 1
            else:
                FUD_cat_test[iyr] = 0

        models = []
        predictors = []
        valid_MAE = []
        valid_RMSE = []
        valid_R2 = []
        valid_R2adj = []
        valid_pval = []
        valid_ss = []
        test_MAE = []
        test_RMSE = []
        test_R2 = []
        test_R2adj = []
        test_pval = []
        test_ss = []
        train_predictions = []
        valid_predictions = []
        test_predictions = []
        valid_Acc = []
        test_Acc = []
        clim_valid_MAE = []
        clim_valid_RMSE = []
        clim_test_MAE = []
        clim_test_RMSE = []

        [train_predictions,valid_predictions,test_predictions,predictors,models,valid_MAE,valid_RMSE_1,valid_R2,valid_R2adj,valid_pval,valid_ss,test_MAE,test_RMSE,test_R2,test_R2adj,test_pval,test_ss]  = find_models(all_combinations,df_train,df_valid,df_test,clim_valid,clim_test,target_train,target_valid,target_test,train_predictions,valid_predictions,test_predictions,predictors,models,valid_MAE,valid_RMSE,valid_R2,valid_R2adj,valid_pval,valid_ss,test_MAE,test_RMSE,test_R2,test_R2adj,test_pval,test_ss,p_critical)

        valid_Acc, valid_cat = eval_accuracy_multiple_models(np.array(valid_predictions),target_valid,FUD_cat_valid,tercile1_FUD,tercile2_FUD)
        test_Acc, test_cat = eval_accuracy_multiple_models(np.array(test_predictions),target_test,FUD_cat_test,tercile1_FUD,tercile2_FUD)

        clim_valid_MAE = np.nanmean(np.abs(target_valid-clim_valid))
        clim_valid_RMSE = np.sqrt(np.nanmean((target_valid-clim_valid)**2.))
        clim_test_MAE = np.nanmean(np.abs(target_test-clim_test))
        clim_test_RMSE = np.sqrt(np.nanmean((target_test-clim_test)**2.))

        # Make Data Frames for output:
        valid_arr_col = ['predictors','valid_MAE','valid_RMSE','valid_Acc','valid_R2','valid_R2adj','valid_pval','valid_ss', 'valid_predictions']
        test_arr_col = ['predictors','test_MAE','test_RMSE','test_Acc','test_R2','test_R2adj','test_pval','test_ss', 'test_predictions']
        clim_test_arr_col = ['clim_test_MAE','clim_test_RMSE']
        clim_valid_arr_col = ['clim_valid_MAE','clim_valid_RMSE']

        arr_tmp_valid = np.zeros((len(predictors),len(valid_arr_col)),dtype= 'object')*np.nan
        for i in range(len(predictors)):
            arr_tmp_valid[i,0] = predictors[i][:]
            arr_tmp_valid[i,8] = np.array(valid_predictions)[i]
        arr_tmp_valid[:,1] = valid_MAE
        arr_tmp_valid[:,2] = valid_RMSE
        arr_tmp_valid[:,3] = valid_Acc
        arr_tmp_valid[:,4] = valid_R2
        arr_tmp_valid[:,5] = valid_R2adj
        arr_tmp_valid[:,6] = valid_pval
        arr_tmp_valid[:,7] = valid_ss
        df_valid_all = pd.DataFrame(arr_tmp_valid,columns=valid_arr_col)

        arr_tmp_test = np.zeros((len(predictors),len(test_arr_col)),dtype= 'object')*np.nan
        for i in range(len(predictors)):
            arr_tmp_test[i,0] = predictors[i][:]
            arr_tmp_test[i,8] = np.array(test_predictions)[i]
        arr_tmp_test[:,1] = test_MAE
        arr_tmp_test[:,2] = test_RMSE
        arr_tmp_test[:,3] = test_Acc
        arr_tmp_test[:,4] = test_R2
        arr_tmp_test[:,5] = test_R2adj
        arr_tmp_test[:,6] = test_pval
        arr_tmp_test[:,7] = test_ss
        df_test_all = pd.DataFrame(arr_tmp_test,columns=test_arr_col)

        clim_arr_tmp_valid = np.zeros((len(predictors),len(clim_valid_arr_col)),dtype= 'object')*np.nan
        clim_arr_tmp_valid[:,0] = clim_valid_MAE
        clim_arr_tmp_valid[:,1] = clim_valid_RMSE
        df_clim_valid_all = pd.DataFrame(clim_arr_tmp_valid,columns=clim_valid_arr_col)

        clim_arr_tmp_test = np.zeros((len(predictors),len(clim_test_arr_col)),dtype= 'object')*np.nan
        clim_arr_tmp_test[:,0] = clim_test_MAE
        clim_arr_tmp_test[:,1] = clim_test_RMSE
        df_clim_test_all = pd.DataFrame(clim_arr_tmp_test,columns=clim_test_arr_col)


    if valid_scheme == 'rolling':
        # Get rolling forecast for each combination and split:
        FUD_cat_valid =  np.zeros((nsplits))*np.nan
        clim =  np.zeros((nsplits))*np.nan
        pred_valid =  np.zeros((len(all_combinations),nsplits))*np.nan
        f_pvalue_train =  np.zeros((len(all_combinations),nsplits))*np.nan
        coeff_pvalues_train =  np.zeros((len(all_combinations),nsplits,max_pred))*np.nan

        for i_split in range(nsplits):
            it_train_start = np.where(years == train_yr_start)[0][0]
            it_train_end = np.where(years == valid_yr_start+i_split)[0][0]
            it_valid_start = np.where(years == valid_yr_start+i_split)[0][0]
            it_valid_end = np.where(years == valid_yr_start+i_split+1)[0][0]

            df_train = pred_df_clean[it_train_start:it_train_end].copy()
            df_valid = pred_df_clean[it_valid_start:it_valid_end].copy()

            target_train = targets[it_train_start:it_train_end].copy()
            target_valid = targets[it_valid_start:it_valid_end].copy()

            clim[i_split] = np.nanmean(targets[it_train_start:it_train_end])

            if target_valid <= tercile1_FUD:
                FUD_cat_valid[i_split] = -1
            elif target_valid > tercile2_FUD:
                FUD_cat_valid[i_split] = 1
            else:
                FUD_cat_valid[i_split] = 0

            # Get forecast for the year of the split
            for i in range(len(all_combinations)):
                x_model = [ df_train.columns[c] for c in all_combinations[i]]

                pred_train_select = df_train[x_model]
                pred_valid_select = df_valid[x_model]

                mlr_model_train = sm.OLS(target_train, sm.add_constant(pred_train_select,has_constant='skip'), missing='drop').fit()

                train_predictions_FUD = mlr_model_train.predict(sm.add_constant(pred_train_select,has_constant='skip'))
                valid_predictions_FUD = mlr_model_train.predict(sm.add_constant(pred_valid_select,has_constant='add'))
                pred_valid[i,i_split] = (valid_predictions_FUD)
                f_pvalue_train[i,i_split]= (mlr_model_train.f_pvalue)
                coeff_pvalues_train[i,i_split,0:len(x_model)]= (mlr_model_train.pvalues[1:])

        # Now redfine the valid and test interval in terms of splits
        split_valid_start = 0
        split_valid_end = test_yr_start-valid_yr_start
        split_test_start = test_yr_start-valid_yr_start
        split_test_end = nsplits

        it_valid_yr_start = np.where(years == valid_yr_start)[0][0]
        it_test_yr_start = np.where(years == test_yr_start)[0][0]
        it_test_yr_end = np.where(years == 2020)[0][0]

        target_valid = targets[it_valid_yr_start:it_test_yr_start]
        target_test = targets[it_test_yr_start:it_test_yr_end]

        clim_valid = clim[split_valid_start:split_valid_end]
        clim_test = clim[split_test_start:split_test_end]

        valid_years = years[it_valid_yr_start:it_test_yr_start]
        test_years = years[it_test_yr_start:it_test_yr_end]

        # Select model that have a significant pvalue over all splits,
        # and that have at least one significant regression coefficient
        # (other than the constant) for each training split up to the
        # test splits
        preselect_models = []
        for im in range(len(all_combinations)):
            if (np.all(f_pvalue_train[im,split_valid_start:split_valid_end] <= p_critical)) & (np.all(np.any(coeff_pvalues_train[im,split_valid_start:split_valid_end,:] <= p_critical, axis=1))):
                preselect_models.append(im)

        # Get all valid and test metrics for selected models
        predictors = []
        valid_MAE = []
        valid_RMSE = []
        valid_R2 = []
        valid_R2adj = []
        valid_pval = []
        valid_ss = []
        test_MAE = []
        test_RMSE = []
        test_R2 = []
        test_R2adj = []
        test_pval = []
        test_ss = []
        valid_predictions = []
        test_predictions = []
        valid_Acc = []
        test_Acc = []
        clim_valid_MAE = []
        clim_valid_RMSE = []
        clim_test_MAE = []
        clim_test_RMSE = []

        if len(preselect_models)>0:
            for s,im in enumerate(preselect_models):

                x_model = [ df_train.columns[c] for c in all_combinations[im]]

                valid_predictions.append(pred_valid[im,split_valid_start:split_valid_end])
                test_predictions.append(pred_valid[im,split_test_start:split_test_end])
                predictors.append(x_model)

                valid_MAE.append(np.nanmean(np.abs(target_valid-pred_valid[im,split_valid_start:split_valid_end])))
                valid_RMSE.append(np.sqrt(np.nanmean((target_valid-pred_valid[im,split_valid_start:split_valid_end])**2.)))
                mlr_model_valid = sm.OLS(target_valid, sm.add_constant(pred_valid[im,split_valid_start:split_valid_end],has_constant='skip'), missing='drop').fit()
                valid_R2.append(mlr_model_valid.rsquared)
                valid_R2adj.append(mlr_model_valid.rsquared_adj)
                valid_pval.append(mlr_model_valid.f_pvalue)
                valid_ss.append(1-( (np.nanmean((target_valid-pred_valid[im,split_valid_start:split_valid_end])**2.)) / (np.nanmean((target_valid-clim_valid)**2.)) ))

                test_MAE.append(np.nanmean(np.abs(target_test-pred_valid[im,split_test_start:split_test_end])))
                test_RMSE.append(np.sqrt(np.nanmean((target_test-pred_valid[im,split_test_start:split_test_end])**2.)))
                mlr_model_test = sm.OLS(target_test, sm.add_constant(pred_valid[im,split_test_start:split_test_end],has_constant='skip'), missing='drop').fit()
                test_R2.append(mlr_model_test.rsquared)
                test_R2adj.append(mlr_model_test.rsquared_adj)
                test_pval.append(mlr_model_test.f_pvalue)
                test_ss.append(1-( (np.nanmean((target_test-pred_valid[im,split_test_start:split_test_end])**2.)) / (np.nanmean((target_test-clim_test)**2.)) ))


            valid_Acc, valid_cat = eval_accuracy_multiple_models(valid_predictions,target_valid,FUD_cat_valid[split_valid_start:split_valid_end],tercile1_FUD,tercile2_FUD)
            test_Acc, test_cat = eval_accuracy_multiple_models(test_predictions,target_test,FUD_cat_valid[split_test_start:split_test_end],tercile1_FUD,tercile2_FUD)

            clim_valid_MAE = np.nanmean(np.abs(target_valid-clim_valid))
            clim_valid_RMSE = np.sqrt(np.nanmean((target_valid-clim_valid)**2.))
            clim_test_MAE = np.nanmean(np.abs(target_test-clim_test))
            clim_test_RMSE = np.sqrt(np.nanmean((target_test-clim_test)**2.))

            # Make Data Frames for output:
            valid_arr_col = ['predictors','valid_MAE','valid_RMSE','valid_Acc','valid_R2','valid_R2adj','valid_pval','valid_ss',  'valid_predictions']
            test_arr_col = ['predictors','test_MAE','test_RMSE','test_Acc','test_R2','test_R2adj','test_pval','test_ss', 'test_predictions']
            clim_test_arr_col = ['clim_test_MAE','clim_test_RMSE']
            clim_valid_arr_col = ['clim_valid_MAE','clim_valid_RMSE']

            arr_tmp_valid = np.zeros((len(predictors),len(valid_arr_col)),dtype= 'object')*np.nan
            for i in range(len(predictors)):
                arr_tmp_valid[i,0] = predictors[i][:]
                arr_tmp_valid[i,8] = valid_predictions[i]
            arr_tmp_valid[:,1] = valid_MAE
            arr_tmp_valid[:,2] = valid_RMSE
            arr_tmp_valid[:,3] = valid_Acc
            arr_tmp_valid[:,4] = valid_R2
            arr_tmp_valid[:,5] = valid_R2adj
            arr_tmp_valid[:,6] = valid_pval
            arr_tmp_valid[:,7] = valid_ss
            df_valid_all = pd.DataFrame(arr_tmp_valid,columns=valid_arr_col)

            arr_tmp_test = np.zeros((len(predictors),len(test_arr_col)),dtype= 'object')*np.nan
            for i in range(len(predictors)):
                arr_tmp_test[i,0] = predictors[i][:]
                arr_tmp_test[i,8] = test_predictions[i]
            arr_tmp_test[:,1] = test_MAE
            arr_tmp_test[:,2] = test_RMSE
            arr_tmp_test[:,3] = test_Acc
            arr_tmp_test[:,4] = test_R2
            arr_tmp_test[:,5] = test_R2adj
            arr_tmp_test[:,6] = test_pval
            arr_tmp_test[:,7] = test_ss
            df_test_all = pd.DataFrame(arr_tmp_test,columns=test_arr_col)

            clim_arr_tmp_valid = np.zeros((len(predictors),len(clim_valid_arr_col)),dtype= 'object')*np.nan
            clim_arr_tmp_valid[:,0] = clim_valid_MAE
            clim_arr_tmp_valid[:,1] = clim_valid_RMSE
            df_clim_valid_all = pd.DataFrame(clim_arr_tmp_valid,columns=clim_valid_arr_col)

            clim_arr_tmp_test = np.zeros((len(predictors),len(clim_test_arr_col)),dtype= 'object')*np.nan
            clim_arr_tmp_test[:,0] = clim_test_MAE
            clim_arr_tmp_test[:,1] = clim_test_RMSE
            df_clim_test_all = pd.DataFrame(clim_arr_tmp_test,columns=clim_test_arr_col)



    if valid_scheme == 'LOOk':

        from sklearn.model_selection import KFold
        it_2019 = np.where(years == 2019)[0][0]
        pred_df_clean = pred_df_clean[0:it_2019+1]
        targets = targets[0:it_2019+1]
        years = years[0:it_2019+1]
        valid_years = np.nan
        test_years = years

        kf = KFold(n_splits=nfolds)

        # 1.
        # For all validation and test periods,
        # and for all possible combinations of predictors:
        # Get FUD predictions, target FUD category, model f-pvalue and model's coefficients pvalues
        FUD_cat_valid = np.zeros((len(years)))*np.nan
        clim_test = np.zeros((len(years)))*np.nan
        pred_test = np.zeros((len(all_combinations),len(years)))*np.nan
        f_pvalue_train = np.zeros((len(all_combinations),len(years)))*np.nan
        coeff_pvalues_train = np.zeros((len(all_combinations),len(years),max_pred))*np.nan

        FUD_cat_valid_kfold = np.zeros((len(years),nfolds,np.max([(len(years)-1 )// nfolds + 1,(len(years)-1 )// nfolds])))*np.nan
        pred_valid_kfold = np.zeros((len(all_combinations),len(years),nfolds,np.max([(len(years)-1 )// nfolds + 1,(len(years)-1 )// nfolds])))*np.nan
        target_valid_kfold = np.zeros((len(all_combinations),len(years),nfolds,np.max([(len(years)-1 )// nfolds + 1,(len(years)-1 )// nfolds])))*np.nan
        clim_valid_kfold = np.zeros((len(all_combinations),len(years),nfolds,np.max([(len(years)-1 )// nfolds + 1,(len(years)-1 )// nfolds])))*np.nan
        f_pvalue_train_kfold = np.zeros((len(all_combinations),len(years),nfolds))*np.nan
        coeff_pvalues_train_kfold = np.zeros((len(all_combinations),len(years),nfolds,max_pred))*np.nan

        for iyr_test,yr_test in enumerate(years):
            # print('')
            # print(iyr_test,yr_test)

            mask_test = np.ones(pred_df_clean.shape[0], dtype=bool)
            mask_test[iyr_test] = False

            # First, get forecast using all years other than test year for training
            target_train = targets[mask_test]
            target_test = targets[~mask_test]
            df_train = pred_df_clean[mask_test]
            df_test = pred_df_clean[~mask_test]

            clim_test[iyr_test] = np.nanmean(target_train)

            if target_test <= tercile1_FUD:
                FUD_cat_valid[iyr_test] = -1
            elif target_test > tercile2_FUD:
                FUD_cat_valid[iyr_test] = 1
            else:
                FUD_cat_valid[iyr_test] = 0

            for i in range(len(all_combinations)):
                x_model = [ df_train.columns[c] for c in all_combinations[i]]

                pred_train_select = df_train[x_model]
                pred_test_select = df_test[x_model]

                mlr_model_train = sm.OLS(target_train, sm.add_constant(pred_train_select,has_constant='skip'), missing='drop').fit()

                train_predictions_FUD = mlr_model_train.predict(sm.add_constant(pred_train_select,has_constant='skip'))
                test_predictions_FUD = mlr_model_train.predict(sm.add_constant(pred_test_select,has_constant='add'))
                pred_test[i,iyr_test] = (test_predictions_FUD)
                f_pvalue_train[i,iyr_test]= (mlr_model_train.f_pvalue)
                coeff_pvalues_train[i,iyr_test,0:len(x_model)]= (mlr_model_train.pvalues[1:])



            # Then perform K-fold cross-validation with all years other than test year
            df_kfold = pred_df_clean[mask_test]
            target_kfold = targets[mask_test]
            years_kfold = years[mask_test]

            for ifold,[train_index, valid_index] in enumerate(kf.split(df_kfold.values)):
                df_train_fold, df_valid_fold = df_kfold.values[train_index], df_kfold.values[valid_index]
                target_train_fold, target_valid_fold = target_kfold[train_index], target_kfold[valid_index]

                df_train_fold = pd.DataFrame(df_train_fold,columns = df_kfold.columns)
                df_valid_fold = pd.DataFrame(df_valid_fold,columns = df_kfold.columns)

                for k in range(len(target_valid_fold)):
                    if target_valid_fold[k] <= tercile1_FUD:
                        FUD_cat_valid_kfold[iyr_test,ifold,k] = -1
                    elif target_valid_fold[k] > tercile2_FUD:
                        FUD_cat_valid_kfold[iyr_test,ifold,k] = 1
                    else:
                        FUD_cat_valid_kfold[iyr_test,ifold,k] = 0

                # Get forecast for all years in the valid fold
                for i in range(len(all_combinations)):
                    x_model = [ df_train_fold.columns[c] for c in all_combinations[i]]

                    pred_train_select_fold = df_train_fold[x_model]
                    pred_valid_select_fold = df_valid_fold[x_model]

                    mlr_model_train_fold = sm.OLS(target_train_fold, sm.add_constant(pred_train_select_fold,has_constant='skip'), missing='drop').fit()

                    train_predictions_FUD_fold = mlr_model_train_fold.predict(sm.add_constant(pred_train_select_fold,has_constant='skip'))
                    valid_predictions_FUD_fold = mlr_model_train_fold.predict(sm.add_constant(pred_valid_select_fold,has_constant='add'))
                    pred_valid_kfold[i,iyr_test,ifold,0:len(target_valid_fold)] = (valid_predictions_FUD_fold)
                    target_valid_kfold[i,iyr_test,ifold,0:len(target_valid_fold)] = target_valid_fold
                    clim_valid_kfold[i,iyr_test,ifold,0:len(target_valid_fold)] = np.ones((len(target_valid_fold)))*np.nanmean(target_train_fold)
                    f_pvalue_train_kfold[i,iyr_test,ifold]= (mlr_model_train_fold.f_pvalue)
                    coeff_pvalues_train_kfold[i,iyr_test,ifold,0:len(x_model)]= (mlr_model_train_fold.pvalues[1:])


        # 2.
        # Select model that have a significant pvalue over all folds,
        # for all years, and that have at least one significant regression
        # coefficient (other than the constant) for each split
        preselect_models_nofolds = []
        for im in range(len(all_combinations)):
            if (np.all(f_pvalue_train[im,:] <= p_critical)) & (np.all(np.any(coeff_pvalues_train[im,:,:] <= p_critical, axis=1))):
                preselect_models_nofolds.append(im)
            # if (np.all(f_pvalue_train[im,:] <= p_critical)) & (np.all(coeff_pvalues_train[im,:,:] <= p_critical)):
            #     preselect_models_nofolds.append(im)

        preselect_models_allfolds = []
        for im in range(len(all_combinations)):
            # !!!!! PROBLEM: My selection criteria appear to be too restrictive.
            #                So for start_date = Nov. 1st, there would be no model
            #                with f-pvalue < p_critical AND at least one coefficient
            #                that is significant FOR ALL OF THE FOLDS....
            # !!! FIX FOR NOW: Relax the selection to only models where the f-pvalue is
            #                  statistically significant for all folds, but that do not
            #                  necessarily have a significant regression coefficient.

            # if (np.all(f_pvalue_train_kfold[im,:,:] <= p_critical)) & (np.all(coeff_pvalues_train_kfold[im,:,:,:] <= p_critical)):
            if (np.all(f_pvalue_train_kfold[im,:,:] <= p_critical)) & (np.all(np.any(coeff_pvalues_train_kfold[im,:,:,:] <= p_critical, axis=2))):
            # if (np.all(f_pvalue_train_kfold[im,:,:] <= p_critical)):
                preselect_models_allfolds.append(im)

        preselect_models = list( set(preselect_models_nofolds).intersection(preselect_models_allfolds) )
        # !!! EVEN MORE STRONG FIX: USE ONLY FIRST CRITERION WHEN NO CV IS DONE
        # preselect_models = preselect_models_nofolds


        # 3.
        # Get all valid and test metrics for selected models
        predictors = []
        valid_MAE = []
        valid_MAE_mean = []
        valid_MAE_min = []
        valid_MAE_max = []
        valid_RMSE = []
        valid_RMSE_mean = []
        valid_RMSE_min = []
        valid_RMSE_max = []
        valid_Acc = []
        valid_Acc_mean = []
        valid_Acc_min = []
        valid_Acc_max = []
        valid_ss = []
        valid_ss_mean = []
        valid_ss_min = []
        valid_ss_max = []
        test_MAE = []
        test_RMSE = []
        test_R2 = []
        test_R2adj = []
        test_pval = []
        test_predictions = []
        test_Acc = []
        test_ss = []
        clim_valid_MAE = []
        clim_valid_MAE_mean = []
        clim_valid_MAE_min = []
        clim_valid_MAE_max = []
        clim_valid_RMSE = []
        clim_valid_RMSE_mean = []
        clim_valid_RMSE_min = []
        clim_valid_RMSE_max = []
        clim_test_MAE = []
        clim_test_RMSE = []

        valid_arr_col = ['predictors','valid_MAE','valid_MAE_mean','valid_MAE_min','valid_MAE_max','valid_RMSE','valid_RMSE_mean','valid_RMSE_min','valid_RMSE_max','valid_Acc','valid_Acc_mean','valid_Acc_min','valid_Acc_max','valid_ss','valid_ss_mean','valid_ss_min','valid_ss_max']
        test_arr_col = ['predictors','test_MAE','test_RMSE','test_Acc','test_R2','test_R2adj','test_pval','test_ss', 'test_predictions']
        clim_test_arr_col = ['clim_test_MAE','clim_test_RMSE']
        clim_valid_arr_col = ['clim_valid_MAE','clim_valid_MAE_mean','clim_valid_MAE_min','clim_valid_MAE_max','clim_valid_RMSE','clim_valid_RMSE_mean','clim_valid_RMSE_min','clim_valid_RMSE_max']
        print(len(preselect_models),len(preselect_models_nofolds),len(preselect_models_allfolds))
        if len(preselect_models)>0:
            for s,im in enumerate(preselect_models):
                x_model = [ df_train.columns[c] for c in all_combinations[im]]
                predictors.append(x_model)

                MAE_valid_kfolds = np.nanmean(np.abs(target_valid_kfold[im,:,:,:]-pred_valid_kfold[im,:,:,:]),axis=2)
                valid_MAE_mean.append(np.nanmean(MAE_valid_kfolds[:,:],axis=1))
                valid_MAE_min.append(np.nanmin(MAE_valid_kfolds[:,:],axis=1))
                valid_MAE_max.append(np.nanmax(MAE_valid_kfolds[:,:],axis=1))
                valid_MAE.append(np.nanmean(np.nanmean(MAE_valid_kfolds[:,:],axis=1)))

                RMSE_valid_kfolds = np.sqrt(np.nanmean((target_valid_kfold[im,:,:,:]-pred_valid_kfold[im,:,:,:])**2.,axis=2))
                valid_RMSE_mean.append(np.nanmean(RMSE_valid_kfolds[:,:],axis=1))
                valid_RMSE_min.append(np.nanmin(RMSE_valid_kfolds[:,:],axis=1))
                valid_RMSE_max.append(np.nanmax(RMSE_valid_kfolds[:,:],axis=1))
                valid_RMSE.append(np.nanmean(np.nanmean(RMSE_valid_kfolds[:,:],axis=1)))

                ss_valid_kfolds = 1-( (np.nanmean((target_valid_kfold[im,:,:,:]-pred_valid_kfold[im,:,:,:])**2.,axis=2)) / (np.nanmean((target_valid_kfold[im,:,:,:]-clim_valid_kfold[im,:,:,:])**2.,axis=2)) )
                valid_ss_mean.append(np.nanmean(ss_valid_kfolds[:,:],axis=1))
                valid_ss_min.append(np.nanmin(ss_valid_kfolds[:,:],axis=1))
                valid_ss_max.append(np.nanmax(ss_valid_kfolds[:,:],axis=1))
                valid_ss.append(np.nanmean(np.nanmean(ss_valid_kfolds[:,:],axis=1)))


                Acc_valid_kfolds = np.zeros((target_valid_kfold.shape[1],target_valid_kfold.shape[2]))*np.nan
                for iyr in range(target_valid_kfold[im,:,:,:].shape[0]):
                    for ifold in range(target_valid_kfold[im,:,:,:].shape[1]):
                        cat_tmp = np.zeros((target_valid_kfold[im,:,:,:].shape[2]))*np.nan

                        sum_acc = 0
                        for ik in range(target_valid_kfold[im,:,:,:].shape[2]):

                            if ~np.isnan(pred_valid_kfold[im,iyr,ifold,ik]):
                                if pred_valid_kfold[im,iyr,ifold,ik] <= tercile1_FUD:
                                    cat_tmp[ik] = -1
                                elif pred_valid_kfold[im,iyr,ifold,ik] > tercile2_FUD:
                                    cat_tmp[ik] = 1
                                else:
                                    cat_tmp[ik] = 0

                                if (cat_tmp[ik] == FUD_cat_valid_kfold[iyr,ifold,ik]):
                                    sum_acc += 1

                        Acc_valid_kfolds[iyr,ifold] = sum_acc/(len(~np.isnan(FUD_cat_valid_kfold[iyr,ifold,:])))

                valid_Acc_mean.append(np.nanmean(Acc_valid_kfolds[:,:],axis=1))
                valid_Acc_min.append(np.nanmin(Acc_valid_kfolds[:,:],axis=1))
                valid_Acc_max.append(np.nanmax(Acc_valid_kfolds[:,:],axis=1))
                valid_Acc.append(np.nanmean(Acc_valid_kfolds[:,:],axis=1))

                test_predictions.append(pred_test[im,:])
                test_MAE.append(np.nanmean(np.abs(targets-pred_test[im,:])))
                test_RMSE.append(np.sqrt(np.nanmean((targets-pred_test[im,:])**2.)))
                mlr_model_test = sm.OLS(targets, sm.add_constant(pred_test[im,:],has_constant='skip'), missing='drop').fit()
                test_R2.append(mlr_model_test.rsquared)
                test_R2adj.append(mlr_model_test.rsquared_adj)
                test_pval.append(mlr_model_test.f_pvalue)
                test_ss.append(1-( (np.nanmean((targets-pred_test[im,:])**2.)) / (np.nanmean((targets-clim_test)**2.)) ))

                clim_test_MAE=(np.nanmean(np.abs(targets-clim_test)))
                clim_test_RMSE=(np.sqrt(np.nanmean((targets-clim_test)**2.)))

                clim_MAE_valid_kfolds = np.nanmean(np.abs(target_valid_kfold[im,:,:,:]-clim_valid_kfold[im,:,:,:]),axis=2)
                clim_valid_MAE_mean.append(np.nanmean(clim_MAE_valid_kfolds[:,:],axis=1))
                clim_valid_MAE_min.append(np.nanmin(clim_MAE_valid_kfolds[:,:],axis=1))
                clim_valid_MAE_max.append(np.nanmax(clim_MAE_valid_kfolds[:,:],axis=1))
                clim_valid_MAE.append(np.nanmean(np.nanmean(clim_MAE_valid_kfolds[:,:],axis=1)))

                clim_RMSE_valid_kfolds = np.sqrt(np.nanmean((target_valid_kfold[im,:,:,:]-clim_valid_kfold[im,:,:,:])**2.,axis=2))
                clim_valid_RMSE_mean.append(np.nanmean(clim_RMSE_valid_kfolds[:,:],axis=1))
                clim_valid_RMSE_min.append(np.nanmin(clim_RMSE_valid_kfolds[:,:],axis=1))
                clim_valid_RMSE_max.append(np.nanmax(clim_RMSE_valid_kfolds[:,:],axis=1))
                clim_valid_RMSE.append(np.nanmean(np.nanmean(clim_RMSE_valid_kfolds[:,:],axis=1)))

            test_Acc, test_cat = eval_accuracy_multiple_models(test_predictions,targets,FUD_cat_valid,tercile1_FUD,tercile2_FUD)

            arr_tmp_valid = np.zeros((len(predictors),len(valid_arr_col)),dtype= 'object')*np.nan
            for i in range(len(predictors)):
                arr_tmp_valid[i,0] = predictors[i][:]
            arr_tmp_valid[:,1] = valid_MAE
            arr_tmp_valid[:,2] = valid_MAE_mean
            arr_tmp_valid[:,3] = valid_MAE_min
            arr_tmp_valid[:,4] = valid_MAE_max
            arr_tmp_valid[:,5] = valid_RMSE
            arr_tmp_valid[:,6] = valid_RMSE_mean
            arr_tmp_valid[:,7] = valid_RMSE_min
            arr_tmp_valid[:,8] = valid_RMSE_max
            arr_tmp_valid[:,9] = valid_Acc
            arr_tmp_valid[:,10] = valid_Acc_mean
            arr_tmp_valid[:,11] = valid_Acc_min
            arr_tmp_valid[:,12] = valid_Acc_max
            arr_tmp_valid[:,13] = valid_ss
            arr_tmp_valid[:,14] = valid_ss_mean
            arr_tmp_valid[:,15] = valid_ss_min
            arr_tmp_valid[:,16] = valid_ss_max
            df_valid_all = pd.DataFrame(arr_tmp_valid,columns=valid_arr_col)

            arr_tmp_test = np.zeros((len(predictors),len(test_arr_col)),dtype= 'object')*np.nan
            for i in range(len(predictors)):
                arr_tmp_test[i,0] = predictors[i][:]
                arr_tmp_test[i,8] = test_predictions[i]
            arr_tmp_test[:,1] = test_MAE
            arr_tmp_test[:,2] = test_RMSE
            arr_tmp_test[:,3] = test_Acc
            arr_tmp_test[:,4] = test_R2
            arr_tmp_test[:,5] = test_R2adj
            arr_tmp_test[:,6] = test_pval
            arr_tmp_test[:,7] = test_ss
            df_test_all = pd.DataFrame(arr_tmp_test,columns=test_arr_col)

            clim_arr_tmp_valid = np.zeros((len(predictors),len(clim_valid_arr_col)),dtype= 'object')*np.nan
            clim_arr_tmp_valid[:,0] = clim_valid_MAE
            clim_arr_tmp_valid[:,1] = clim_valid_MAE_mean
            clim_arr_tmp_valid[:,2] = clim_valid_MAE_min
            clim_arr_tmp_valid[:,3] = clim_valid_MAE_max
            clim_arr_tmp_valid[:,4] = clim_valid_RMSE
            clim_arr_tmp_valid[:,5] = clim_valid_RMSE_mean
            clim_arr_tmp_valid[:,6] = clim_valid_RMSE_min
            clim_arr_tmp_valid[:,7] = clim_valid_RMSE_max
            df_clim_valid_all = pd.DataFrame(clim_arr_tmp_valid,columns=clim_valid_arr_col)

            clim_arr_tmp_test = np.zeros((len(predictors),len(clim_test_arr_col)),dtype= 'object')*np.nan
            clim_arr_tmp_test[:,0] = clim_test_MAE
            clim_arr_tmp_test[:,1] = clim_test_RMSE
            df_clim_test_all = pd.DataFrame(clim_arr_tmp_test,columns=clim_test_arr_col)


        else:
            arr_tmp_valid = np.zeros((len(all_combinations),len(valid_arr_col)))*np.nan
            arr_tmp_test = np.zeros((len(all_combinations),len(test_arr_col)))*np.nan
            df_valid_all = pd.DataFrame(arr_tmp_valid,columns=valid_arr_col)
            df_test_all = pd.DataFrame(arr_tmp_test,columns=test_arr_col)

            clim_arr_tmp_valid = np.zeros((len(all_combinations),len(clim_valid_arr_col)))*np.nan
            clim_arr_tmp_test = np.zeros((len(all_combinations),len(clim_test_arr_col)))*np.nan
            df_clim_valid_all = pd.DataFrame(clim_arr_tmp_valid,columns=clim_valid_arr_col)
            df_clim_test_all = pd.DataFrame(clim_arr_tmp_test,columns=clim_test_arr_col)


    if valid_scheme == 'standardk':
        valid_years = np.nan
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=nfolds)

        it_train_start = np.where(years == train_yr_start)[0][0]
        it_train_end = np.where(years == test_yr_start)[0][0]
        it_valid_start = np.where(years == valid_yr_start)[0][0]
        it_test_start = np.where(years == test_yr_start)[0][0]
        it_test_end = np.where(years == 2020)[0][0]

        df_train = pred_df_clean[it_train_start:it_train_end].copy()
        df_test = pred_df_clean[it_test_start:it_test_end].copy()
        train_years = years[it_train_start:it_train_end]
        test_years = years[it_test_start:it_test_end]

        target_train = targets[it_train_start:it_train_end].copy()
        target_test = targets[it_test_start:it_test_end].copy()

        clim_test = np.ones((len(target_test)))*np.nanmean(targets[it_train_start:it_test_start])

        # 1.
        # Get FUD predictions, target FUD category, model f-pvalue and model's coefficients pvalues
        # for the test period, using the full training period
        FUD_cat_test = np.zeros(target_test.shape)*np.nan
        for iyr in range(FUD_cat_test.shape[0]):
            if target_test[iyr] <= tercile1_FUD:
                FUD_cat_test[iyr] = -1
            elif target_test[iyr] > tercile2_FUD:
                FUD_cat_test[iyr] = 1
            else:
                FUD_cat_test[iyr] = 0

        pred_test = np.zeros((len(all_combinations),len(test_years)))*np.nan
        f_pvalue_train = np.zeros((len(all_combinations)))*np.nan
        coeff_pvalues_train = np.zeros((len(all_combinations),max_pred))*np.nan
        for i in range(len(all_combinations)):
            x_model = [ df_train.columns[c] for c in all_combinations[i]]

            pred_train_select = df_train[x_model]
            pred_test_select = df_test[x_model]

            mlr_model_train = sm.OLS(target_train, sm.add_constant(pred_train_select,has_constant='skip'), missing='drop').fit()

            train_predictions_FUD = mlr_model_train.predict(sm.add_constant(pred_train_select,has_constant='skip'))
            test_predictions_FUD = mlr_model_train.predict(sm.add_constant(pred_test_select,has_constant='add'))
            pred_test[i] = (test_predictions_FUD)
            f_pvalue_train[i]= (mlr_model_train.f_pvalue)
            coeff_pvalues_train[i,0:len(x_model)]= (mlr_model_train.pvalues[1:])

        # 2.
        # Then perform K-fold cross-validation over all training years
        FUD_cat_valid_kfold = np.zeros((nfolds,np.max([(len(years)-1 )// nfolds + 1,(len(years)-1 )// nfolds])))*np.nan
        pred_valid_kfold = np.zeros((len(all_combinations),nfolds,np.max([(len(years)-1 )// nfolds + 1,(len(years)-1 )// nfolds])))*np.nan
        target_valid_kfold = np.zeros((len(all_combinations),nfolds,np.max([(len(years)-1 )// nfolds + 1,(len(years)-1 )// nfolds])))*np.nan
        clim_valid_kfold = np.zeros((len(all_combinations),nfolds,np.max([(len(years)-1 )// nfolds + 1,(len(years)-1 )// nfolds])))*np.nan
        f_pvalue_train_kfold = np.zeros((len(all_combinations),nfolds))*np.nan
        coeff_pvalues_train_kfold = np.zeros((len(all_combinations),nfolds,max_pred))*np.nan

        df_kfold = df_train
        target_kfold = target_train

        for ifold,[train_index, valid_index] in enumerate(kf.split(df_kfold.values)):
            df_train_fold, df_valid_fold = df_kfold.values[train_index], df_kfold.values[valid_index]
            target_train_fold, target_valid_fold = target_kfold[train_index], target_kfold[valid_index]

            df_train_fold = pd.DataFrame(df_train_fold,columns = df_kfold.columns)
            df_valid_fold = pd.DataFrame(df_valid_fold,columns = df_kfold.columns)

            for k in range(len(target_valid_fold)):
                if target_valid_fold[k] <= tercile1_FUD:
                    FUD_cat_valid_kfold[ifold,k] = -1
                elif target_valid_fold[k] > tercile2_FUD:
                    FUD_cat_valid_kfold[ifold,k] = 1
                else:
                    FUD_cat_valid_kfold[ifold,k] = 0

            # Get forecast for all years in the valid fold
            for i in range(len(all_combinations)):
                x_model = [ df_train_fold.columns[c] for c in all_combinations[i]]

                pred_train_select_fold = df_train_fold[x_model]
                pred_valid_select_fold = df_valid_fold[x_model]

                mlr_model_train_fold = sm.OLS(target_train_fold, sm.add_constant(pred_train_select_fold,has_constant='skip'), missing='drop').fit()

                train_predictions_FUD_fold = mlr_model_train_fold.predict(sm.add_constant(pred_train_select_fold,has_constant='skip'))
                valid_predictions_FUD_fold = mlr_model_train_fold.predict(sm.add_constant(pred_valid_select_fold,has_constant='add'))
                pred_valid_kfold[i,ifold,0:len(target_valid_fold)] = (valid_predictions_FUD_fold)
                target_valid_kfold[i,ifold,0:len(target_valid_fold)] = target_valid_fold
                clim_valid_kfold[i,ifold,0:len(target_valid_fold)] = np.ones((len(target_valid_fold)))*np.nanmean(target_train_fold)
                f_pvalue_train_kfold[i,ifold]= (mlr_model_train_fold.f_pvalue)
                coeff_pvalues_train_kfold[i,ifold,0:len(x_model)]= (mlr_model_train_fold.pvalues[1:])

        # 3.
        # Select model that have a significant pvalue over all folds,
        # for all years, and that have at least one significant regression
        # coefficient (other than the constant) for each split
        preselect_models_nofolds = []
        for im in range(len(all_combinations)):
            if (np.all(f_pvalue_train[im] <= p_critical)) & (np.all(np.any(coeff_pvalues_train[im,:] <= p_critical))):
                preselect_models_nofolds.append(im)

        preselect_models_allfolds = []
        for im in range(len(all_combinations)):
            # !!!!! PROBLEM: My selection criteria appear to be too restrictive.
            #                So for start_date = Nov. 1st, there would be no model
            #                with f-pvalue < p_critical AND at least one coefficient
            #                that is significant FOR ALL OF THE FOLDS....
            # !!! FIX FOR NOW: Relax the selection to only models where the f-pvalue is
            #                  statistically significant for all folds, but that do not
            #                  necessarily have a significant regression coefficient.

            if (np.all(f_pvalue_train_kfold[im,:] <= p_critical)) & (np.all(np.any(coeff_pvalues_train_kfold[im,:,:] <= p_critical, axis=1))):
            # if (np.all(f_pvalue_train_kfold[im,:] <= p_critical)):
                preselect_models_allfolds.append(im)

        preselect_models = list( set(preselect_models_nofolds).intersection(preselect_models_allfolds) )
        # !!! EVEN MORE STRONG FIX: USE ONLY FIRST CRITERION WHEN NO CV IS DONE
        # preselect_models = preselect_models_nofolds

        # 4.
        # Get all valid and test metrics for selected models
        predictors = []
        valid_MAE = []
        valid_MAE_mean = []
        valid_MAE_min = []
        valid_MAE_max = []
        valid_RMSE = []
        valid_RMSE_mean = []
        valid_RMSE_min = []
        valid_RMSE_max = []
        valid_Acc = []
        valid_Acc_mean = []
        valid_Acc_min = []
        valid_Acc_max = []
        valid_ss = []
        valid_ss_mean = []
        valid_ss_min = []
        valid_ss_max = []
        test_MAE = []
        test_RMSE = []
        test_R2 = []
        test_R2adj = []
        test_pval = []
        test_predictions = []
        test_Acc = []
        test_ss = []
        clim_valid_MAE = []
        clim_valid_MAE_mean = []
        clim_valid_MAE_min = []
        clim_valid_MAE_max = []
        clim_valid_RMSE = []
        clim_valid_RMSE_mean = []
        clim_valid_RMSE_min = []
        clim_valid_RMSE_max = []
        clim_test_MAE = []
        clim_test_RMSE = []

        valid_arr_col = ['predictors','valid_MAE','valid_MAE_mean','valid_MAE_min','valid_MAE_max','valid_RMSE','valid_RMSE_mean','valid_RMSE_min','valid_RMSE_max','valid_Acc','valid_Acc_mean','valid_Acc_min','valid_Acc_max','valid_ss','valid_ss_mean','valid_ss_min','valid_ss_max']
        test_arr_col = ['predictors','test_MAE','test_RMSE','test_Acc','test_R2','test_R2adj','test_pval','test_ss', 'test_predictions']
        clim_test_arr_col = ['clim_test_MAE','clim_test_RMSE']
        clim_valid_arr_col = ['clim_valid_MAE','clim_valid_MAE_mean','clim_valid_MAE_min','clim_valid_MAE_max','clim_valid_RMSE','clim_valid_RMSE_mean','clim_valid_RMSE_min','clim_valid_RMSE_max']
        print(len(preselect_models),len(preselect_models_nofolds),len(preselect_models_allfolds))

        if len(preselect_models)>0:
            for s,im in enumerate(preselect_models):
                x_model = [ df_train.columns[c] for c in all_combinations[im]]
                predictors.append(x_model)
                MAE_valid_kfolds = np.nanmean(np.abs(target_valid_kfold[im,:,:]-pred_valid_kfold[im,:,:]),axis=1)
                valid_MAE_mean.append(np.nanmean(MAE_valid_kfolds[:]))
                valid_MAE_min.append(np.nanmin(MAE_valid_kfolds[:]))
                valid_MAE_max.append(np.nanmax(MAE_valid_kfolds[:]))
                valid_MAE.append(np.nanmean(np.nanmean(MAE_valid_kfolds[:])))

                RMSE_valid_kfolds = np.sqrt(np.nanmean((target_valid_kfold[im,:,:]-pred_valid_kfold[im,:,:])**2.,axis=1))
                valid_RMSE_mean.append(np.nanmean(RMSE_valid_kfolds[:]))
                valid_RMSE_min.append(np.nanmin(RMSE_valid_kfolds[:]))
                valid_RMSE_max.append(np.nanmax(RMSE_valid_kfolds[:]))
                valid_RMSE.append(np.nanmean(np.nanmean(RMSE_valid_kfolds[:])))

                ss_valid_kfolds = 1-( (np.nanmean((target_valid_kfold[im,:,:]-pred_valid_kfold[im,:,:])**2.,axis=1)) / (np.nanmean((target_valid_kfold[im,:,:]-clim_valid_kfold[im,:,:])**2.,axis=1)) )
                valid_ss_mean.append(np.nanmean(ss_valid_kfolds[:]))
                valid_ss_min.append(np.nanmin(ss_valid_kfolds[:]))
                valid_ss_max.append(np.nanmax(ss_valid_kfolds[:]))
                valid_ss.append(np.nanmean(np.nanmean(ss_valid_kfolds[:])))

                # HERE: target_valid_kfold[i,ifold,0:len(target_valid_fold)]
                # OLD:  target_valid_kfold[i,iyr_test,ifold,0:len(target_valid_fold)]
                Acc_valid_kfolds = np.zeros((target_valid_kfold.shape[1]))*np.nan
                for ifold in range(target_valid_kfold[im,:,:].shape[0]):
                    cat_tmp = np.zeros((target_valid_kfold[im,:,:].shape[1]))*np.nan

                    sum_acc = 0
                    for ik in range(target_valid_kfold[im,:,:].shape[1]):

                        if ~np.isnan(pred_valid_kfold[im,ifold,ik]):
                            if pred_valid_kfold[im,ifold,ik] <= tercile1_FUD:
                                cat_tmp[ik] = -1
                            elif pred_valid_kfold[im,ifold,ik] > tercile2_FUD:
                                cat_tmp[ik] = 1
                            else:
                                cat_tmp[ik] = 0

                            if (cat_tmp[ik] == FUD_cat_valid_kfold[ifold,ik]):
                                sum_acc += 1

                    Acc_valid_kfolds[ifold] = sum_acc/(len(~np.isnan(FUD_cat_valid_kfold[ifold,:])))

                valid_Acc_mean.append(np.nanmean(Acc_valid_kfolds[:]))
                valid_Acc_min.append(np.nanmin(Acc_valid_kfolds[:]))
                valid_Acc_max.append(np.nanmax(Acc_valid_kfolds[:]))
                valid_Acc.append(np.nanmean(Acc_valid_kfolds[:]))

                test_predictions.append(pred_test[im,:])
                test_MAE.append(np.nanmean(np.abs(target_test-pred_test[im,:])))
                test_RMSE.append(np.sqrt(np.nanmean((target_test-pred_test[im,:])**2.)))
                mlr_model_test = sm.OLS(target_test, sm.add_constant(pred_test[im,:],has_constant='skip'), missing='drop').fit()
                test_R2.append(mlr_model_test.rsquared)
                test_R2adj.append(mlr_model_test.rsquared_adj)
                test_pval.append(mlr_model_test.f_pvalue)
                test_ss.append(1-( (np.nanmean((target_test-pred_test[im,:])**2.)) / (np.nanmean((target_test-clim_test)**2.)) ))

                clim_test_MAE=(np.nanmean(np.abs(target_test-clim_test)))
                clim_test_RMSE=(np.sqrt(np.nanmean((target_test-clim_test)**2.)))

                clim_MAE_valid_kfolds = np.nanmean(np.abs(target_valid_kfold[im,:,:]-clim_valid_kfold[im,:,:]),axis=1)
                clim_valid_MAE_mean.append(np.nanmean(clim_MAE_valid_kfolds[:]))
                clim_valid_MAE_min.append(np.nanmin(clim_MAE_valid_kfolds[:]))
                clim_valid_MAE_max.append(np.nanmax(clim_MAE_valid_kfolds[:]))
                clim_valid_MAE.append(np.nanmean(np.nanmean(clim_MAE_valid_kfolds[:])))

                clim_RMSE_valid_kfolds = np.sqrt(np.nanmean((target_valid_kfold[im,:,:]-clim_valid_kfold[im,:,:])**2.,axis=1))
                clim_valid_RMSE_mean.append(np.nanmean(clim_RMSE_valid_kfolds[:]))
                clim_valid_RMSE_min.append(np.nanmin(clim_RMSE_valid_kfolds[:]))
                clim_valid_RMSE_max.append(np.nanmax(clim_RMSE_valid_kfolds[:]))
                clim_valid_RMSE.append(np.nanmean(np.nanmean(clim_RMSE_valid_kfolds[:])))

            test_Acc, test_cat = eval_accuracy_multiple_models(test_predictions,target_test,FUD_cat_test,tercile1_FUD,tercile2_FUD)

            arr_tmp_valid = np.zeros((len(predictors),len(valid_arr_col)),dtype= 'object')*np.nan
            for i in range(len(predictors)):
                arr_tmp_valid[i,0] = predictors[i][:]
            arr_tmp_valid[:,1] = valid_MAE
            arr_tmp_valid[:,2] = valid_MAE_mean
            arr_tmp_valid[:,3] = valid_MAE_min
            arr_tmp_valid[:,4] = valid_MAE_max
            arr_tmp_valid[:,5] = valid_RMSE
            arr_tmp_valid[:,6] = valid_RMSE_mean
            arr_tmp_valid[:,7] = valid_RMSE_min
            arr_tmp_valid[:,8] = valid_RMSE_max
            arr_tmp_valid[:,9] = valid_Acc
            arr_tmp_valid[:,10] = valid_Acc_mean
            arr_tmp_valid[:,11] = valid_Acc_min
            arr_tmp_valid[:,12] = valid_Acc_max
            arr_tmp_valid[:,13] = valid_ss
            arr_tmp_valid[:,14] = valid_ss_mean
            arr_tmp_valid[:,15] = valid_ss_min
            arr_tmp_valid[:,16] = valid_ss_max
            df_valid_all = pd.DataFrame(arr_tmp_valid,columns=valid_arr_col)

            arr_tmp_test = np.zeros((len(predictors),len(test_arr_col)),dtype= 'object')*np.nan
            for i in range(len(predictors)):
                arr_tmp_test[i,0] = predictors[i][:]
                arr_tmp_test[i,8] = test_predictions[i]
            arr_tmp_test[:,1] = test_MAE
            arr_tmp_test[:,2] = test_RMSE
            arr_tmp_test[:,3] = test_Acc
            arr_tmp_test[:,4] = test_R2
            arr_tmp_test[:,5] = test_R2adj
            arr_tmp_test[:,6] = test_pval
            arr_tmp_test[:,7] = test_ss
            df_test_all = pd.DataFrame(arr_tmp_test,columns=test_arr_col)

            clim_arr_tmp_valid = np.zeros((len(predictors),len(clim_valid_arr_col)),dtype= 'object')*np.nan
            clim_arr_tmp_valid[:,0] = clim_valid_MAE
            clim_arr_tmp_valid[:,1] = clim_valid_MAE_mean
            clim_arr_tmp_valid[:,2] = clim_valid_MAE_min
            clim_arr_tmp_valid[:,3] = clim_valid_MAE_max
            clim_arr_tmp_valid[:,4] = clim_valid_RMSE
            clim_arr_tmp_valid[:,5] = clim_valid_RMSE_mean
            clim_arr_tmp_valid[:,6] = clim_valid_RMSE_min
            clim_arr_tmp_valid[:,7] = clim_valid_RMSE_max
            df_clim_valid_all = pd.DataFrame(clim_arr_tmp_valid,columns=clim_valid_arr_col)

            clim_arr_tmp_test = np.zeros((len(predictors),len(clim_test_arr_col)),dtype= 'object')*np.nan
            clim_arr_tmp_test[:,0] = clim_test_MAE
            clim_arr_tmp_test[:,1] = clim_test_RMSE
            df_clim_test_all = pd.DataFrame(clim_arr_tmp_test,columns=clim_test_arr_col)

        else:
            arr_tmp_valid = np.zeros((len(all_combinations),len(valid_arr_col)))*np.nan
            arr_tmp_test = np.zeros((len(all_combinations),len(test_arr_col)))*np.nan
            df_valid_all = pd.DataFrame(arr_tmp_valid,columns=valid_arr_col)
            df_test_all = pd.DataFrame(arr_tmp_test,columns=test_arr_col)

            clim_arr_tmp_valid = np.zeros((len(all_combinations),len(clim_valid_arr_col)))*np.nan
            clim_arr_tmp_test = np.zeros((len(all_combinations),len(clim_test_arr_col)))*np.nan
            df_clim_valid_all = pd.DataFrame(clim_arr_tmp_valid,columns=clim_valid_arr_col)
            df_clim_test_all = pd.DataFrame(clim_arr_tmp_test,columns=clim_test_arr_col)




    return df_valid_all,df_test_all,df_clim_valid_all,df_clim_test_all,valid_years,test_years




#%%

# p_critical = 0.01
p_critical = 0.05

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

plot_CV_metrics = False
plot_top5_models = False

valid_scheme = 'standard'
# valid_scheme = 'rolling'
# valid_scheme = 'LOOk'
# valid_scheme = 'standardk'
# valid_scheme = 'rollingk'

valid_scheme_list = ['standard','rolling','standardk','LOOk']
# valid_scheme_list = ['standard','rolling','standardk']
valid_scheme_list = ['LOOk']

train_yr_start = 1992 # [1992 - 2007] = 16 years
valid_yr_start = 2008 # [2008 - 2013] = 6 years
test_yr_start = 2014  # [2014 - 2019] = 6 years
nsplits = 2019-valid_yr_start+1
nfolds = 5


istart_show = [2]
max_pred = 3
valid_metric = 'RMSE'
# valid_metric = 'MAE'
# valid_metric = 'R2adj'
# valid_metric = 'Acc'

istart_labels = ['Sept. 1st', 'Oct. 1st','Nov. 1st','Dec. 1st']


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
Twater_mean[14269:14329] = 0.

# Average (and round) FUD from all locations:
avg_freezeup_doy = np.nanmean(freezeup_doy,axis=1)
avg_freezeup_doy = np.round(avg_freezeup_doy)

# Load monthly predictor data
fpath_mp = '/Volumes/SeagateUSB/McGill/Postdoc/slice/data/monthly_predictors/'
monthly_pred = np.load(fpath_mp+'monthly_vars.npz')
monthly_pred_data = monthly_pred['data']
pred_names = monthly_pred['labels']

# Replace zeros with nan for snowfall, FDD, TDD:
# monthly_pred_data[5,:,:][monthly_pred_data[5,:,:] == 0] = np.nan
# monthly_pred_data[10,:,:][monthly_pred_data[10,:,:] == 0] = np.nan
# monthly_pred_data[11,:,:][monthly_pred_data[11,:,:] == 0] = np.nan

# Remove data for December:
monthly_pred_data[:,:,11] = np.nan

# Select specific predictors and make dataframe:
month_str = ['Jan. ', 'Feb. ', 'Mar. ', 'Apr. ', 'May ', 'Jun. ','Jul. ', 'Aug. ', 'Sep. ', 'Oct. ', 'Nov. ']

# Choose targets: avg_freezeup_doy or synthetic data set
np.random.seed(seed=11)
noise = True

# Nov. windspeed x ( Nov. Tw -  Nov. Ta)
# targets = noise*np.random.normal(0,2,len(years)) +  monthly_pred_data[8,:,10]*(monthly_pred_data[15,:,10]-monthly_pred_data[0,:,10])

# (May Ta^4) + (Nov. windspeed x Nov. Tw) - (Nov. windspeed x  Nov. Ta)
term2 = monthly_pred_data[8,:,10]*(monthly_pred_data[15,:,10])
norm_term2 = (term2-np.nanmin(term2))/(np.nanmax(term2)-np.nanmin(term2))
term3 = -1*monthly_pred_data[8,:,10]*(monthly_pred_data[0,:,10])
norm_term3 = (term3-np.nanmin(term3))/(np.nanmax(term3)-np.nanmin(term3))
norm_Ta4_May = (monthly_pred_data[0,:,4]**4 - np.nanmin(monthly_pred_data[0,:,4]**4))/(np.nanmax(monthly_pred_data[0,:,4]**4)- np.nanmin(monthly_pred_data[0,:,4]**4))
targets = noise*np.random.normal(0,0.2,len(years)) + norm_Ta4_May + norm_term2 + norm_term3
targets = 355.6+10*targets
# plt.figure();plt.plot(years,targets,'o-')

targets = avg_freezeup_doy


# Get target categories for accuracy measure:
it_1992 = np.where(years == 1992)[0][0]
it_2008= np.where(years == 2008)[0][0]
mean_FUD = np.nanmean(targets[it_1992:it_2008])
std_FUD = np.nanstd(targets[it_1992:it_2008])
tercile1_FUD = np.nanpercentile(targets[it_1992:it_2008],(1/3.)*100)
tercile2_FUD = np.nanpercentile(targets[it_1992:it_2008],(2/3.)*100)

# Keep only 1991 to 2020 (inclusive)
it_1992 = np.where(years == 1992)[0][0]
it_2020 = np.where(years == 2020)[0][0]

years = years[it_1992:it_2020+1]
targets = targets[it_1992:it_2020+1]
monthly_pred_data = monthly_pred_data[:,it_1992:it_2020+1,:]

if freezeup_opt == 1:
    # PREDICTORS AVAILABLE SEPT 1st: FUD + Tw
    # p_sep1_list = ['Avg. Ta_mean','Tot. TDD','Tot. TDD','Tot. snowfall','Avg. cloud cover','PDO','PDO','Avg. Twater']
    # m_sep1_list = [5,             1,          5,         4,              4,                 1,    2,    5           ]
    # PREDICTORS AVAILABLE SEPT 1st: FUD ONLY
    p_sep1_list = ['Avg. Ta_mean','Tot. TDD','PDO','PDO']
    m_sep1_list = [5,             5,          1,    2]
if freezeup_opt == 2:
    # PREDICTORS AVAILABLE SEPT 1st: FUD ONLY
    p_sep1_list = ['AMO' ]
    m_sep1_list = [3     ]
pred_sep1_arr = np.zeros((monthly_pred_data.shape[1],len(p_sep1_list)))*np.nan
col_sep1 = []
for i in range(len(p_sep1_list)):
    ipred = np.where(pred_names == p_sep1_list[i])[0][0]
    pred_sep1_arr[:,i] = monthly_pred_data[ipred,:,m_sep1_list[i]-1]
    col_sep1.append(month_str[m_sep1_list[i]-1]+p_sep1_list[i])
pred_sep1_df =  pd.DataFrame(pred_sep1_arr,columns=col_sep1)

if freezeup_opt == 1:
    # PREDICTORS AVAILABLE OCT 1st: FUD + Tw
    # p_oct1_list = ['Avg. Ta_mean','Tot. TDD','Tot. TDD','Tot. snowfall','Avg. cloud cover','Avg. cloud cover','NAO','PDO','PDO','Avg. Twater']
    # m_oct1_list = [5,             1,          5,         4,              4,                9,                  9,    1,    2,    5           ]
    # PREDICTORS AVAILABLE OCT 1st: FUD ONLY
    p_oct1_list = ['Avg. Ta_mean','Tot. TDD','Avg. cloud cover','NAO','PDO','PDO']
    m_oct1_list = [5,             5,          9,                 9,    1,    2]
if freezeup_opt == 2:
    # PREDICTORS AVAILABLE OCT 1st: FUD ONLY
    p_oct1_list = ['AMO','NAO','PNA']
    m_oct1_list = [3,     9,   9,   ]

pred_oct1_arr = np.zeros((monthly_pred_data.shape[1],len(p_oct1_list)))*np.nan
col_oct1 = []
for i in range(len(p_oct1_list)):
    ipred = np.where(pred_names == p_oct1_list[i])[0][0]
    pred_oct1_arr[:,i] = monthly_pred_data[ipred,:,m_oct1_list[i]-1]
    col_oct1.append(month_str[m_oct1_list[i]-1]+p_oct1_list[i])
pred_oct1_df =  pd.DataFrame(pred_oct1_arr,columns=col_oct1)

if freezeup_opt == 1:
    # PREDICTORS AVAILABLE NOV 1st: FUD + Tw
    # p_nov1_list = ['Avg. Ta_mean','Tot. TDD','Tot. TDD','Tot. snowfall','Avg. cloud cover','Avg. cloud cover','Avg. level Ottawa River','NAO','PDO','PDO','Avg. Twater','Avg. windspeed']
    # m_nov1_list = [5,             1,          5,        4,              4,                 9,                 10,                        9,    1,    2,    5,         10              ]
    # PREDICTORS AVAILABLE NOV 1st: FUD ONLY
    p_nov1_list = ['Avg. Ta_mean','Tot. TDD','Avg. cloud cover','Avg. level Ottawa River','NAO','PDO','PDO','Avg. windspeed']
    m_nov1_list = [5,             5,          9,                 10,                       9,    1,    2,   10              ]
if freezeup_opt == 2:
    # PREDICTORS AVAILABLE NOV 1st: FUD ONLY
    p_nov1_list = ['AMO','NAO','PNA','Avg. level Ottawa River','SOI']
    m_nov1_list = [3,    9,     9,     10,                      10]

pred_nov1_arr = np.zeros((monthly_pred_data.shape[1],len(p_nov1_list)))*np.nan
col_nov1 = []
for i in range(len(p_nov1_list)):
    ipred = np.where(pred_names == p_nov1_list[i])[0][0]
    pred_nov1_arr[:,i] = monthly_pred_data[ipred,:,m_nov1_list[i]-1]
    col_nov1.append(month_str[m_nov1_list[i]-1]+p_nov1_list[i])
pred_nov1_df =  pd.DataFrame(pred_nov1_arr,columns=col_nov1)

if freezeup_opt == 1:
    # PREDICTORS AVAILABLE DEC 1st: FUD + Tw
    # p_dec1_list = ['Avg. Ta_mean','Avg. Ta_mean','Tot. FDD','Tot. TDD','Tot. TDD','Tot. precip.','Avg. SLP','Tot. snowfall','Tot. snowfall','Avg. cloud cover','Avg. cloud cover','Avg. cloud cover','Avg. level Ottawa River','AO','NAO','NAO','PDO','PDO','Avg. Twater','Avg. Twater','Avg. windspeed'  ]
    # m_dec1_list = [11,            5,             11,        1,          5,        11,            11,         4,              11,             4,                 9,                 11,                10,                       11,   9,    11,   1,    2,    5,            11         ,10                ]
    # PREDICTORS AVAILABLE DEC 1st: FUD ONLY
    p_dec1_list = ['Avg. Ta_mean','Avg. Ta_mean','Tot. FDD','Tot. TDD','Tot. TDD','Avg. SLP','Tot. snowfall','Avg. cloud cover','Avg. level Ottawa River','AO','NAO','NAO','PDO','PDO','Avg. windspeed'  ]
    m_dec1_list = [11,            5,             11,         5,        11,         11,        11,             9,                 10,                       11,   9,    11,   1,    2,   10               ]
if freezeup_opt == 2:
    # PREDICTORS AVAILABLE DEC 1st: FUD ONLY
    p_dec1_list = ['AMO','NAO','PNA','Avg. level Ottawa River','SOI','Avg. Ta_mean','Tot. FDD','Tot. TDD','Avg. SLP','Tot. snowfall','AO','NAO']
    m_dec1_list = [3,    9,     9,   10,                        10,   11,            11,         11,        11,        11,           11,    11 ]

pred_dec1_arr = np.zeros((monthly_pred_data.shape[1],len(p_dec1_list)))*np.nan
col_dec1 = []
for i in range(len(p_dec1_list)):
    ipred = np.where(pred_names == p_dec1_list[i])[0][0]
    pred_dec1_arr[:,i] = monthly_pred_data[ipred,:,m_dec1_list[i]-1]
    col_dec1.append(month_str[m_dec1_list[i]-1]+p_dec1_list[i])
pred_dec1_df =  pd.DataFrame(pred_dec1_arr,columns=col_dec1)


fig_ts,ax_ts = plt.subplots(nrows=2, ncols=2,sharex = True, sharey = True)
fig_ts.suptitle("Best MLR Models according to "+valid_metric)

fig_metrics,ax_metrics = plt.subplots(nrows=3, ncols=2,sharex = True, sharey = 'row')
ax_metrics[0,0].set_title('Validation (CV = '+ str(nfolds) +' folds)')
ax_metrics[0,1].set_title('Test')
ax_metrics[0,0].set_ylabel('MAE')
ax_metrics[1,0].set_ylabel('RMSE')
ax_metrics[2,0].set_ylabel('SS$_{MSE}$')

for ind,istart in enumerate(istart_show):

    if istart == 0:
        col = col_sep1; pred_df = pred_sep1_df
    if istart == 1:
        col = col_oct1; pred_df = pred_oct1_df
    if istart == 2:
        col = col_nov1; pred_df = pred_nov1_df
    if istart == 3:
        col = col_dec1; pred_df = pred_dec1_df

    # Check multicolinearity on the whole data set before making combinations of predictors:
    pred_df_clean, dropped_cols = remove_collinear_features(pred_df, targets, threshold=0.8, target_in_df = False)
    print("dropped columns: ")
    print(list(dropped_cols))

    # Then get all possible combination of predictors
    col = pred_df_clean.columns.tolist()
    all_combinations = []
    for p in range(max_pred):
        all_combinations += find_all_column_combinations(col,p+1)

    for v,valid_scheme in enumerate(valid_scheme_list):
        # Find all test and valid metrics for all or pre-selected models,
        # according to the specified validation scheme:
        df_valid_all,df_test_all,df_clim_valid_all,df_clim_test_all,valid_years,test_years  = get_train_valid_test_metric_dfs(valid_scheme,pred_df_clean,targets,all_combinations,nsplits,nfolds,years,train_yr_start,valid_yr_start,test_yr_start,tercile1_FUD,tercile2_FUD,p_critical,max_pred)

        if ~np.all(np.isnan(pd.to_numeric(df_valid_all['valid_RMSE']))):
            # Make selection of best model according to chosen validation metric:
            df_select_valid, df_select_test = select_best_model(valid_metric,df_valid_all,df_test_all)

            # Plot forecasts:
            plot_label = istart_labels[istart]+': '
            for p in range(len(df_select_test['predictors'])):
                plot_label += df_select_test['predictors'][p]+','

            ax_ts[v%2,v//2].plot(years,targets ,'o-',color='k')
            if (valid_scheme == 'standard')|(valid_scheme == 'rolling'):
                ax_ts[v%2,v//2].plot(valid_years,df_select_valid['valid_predictions'],'o-',color=plt.get_cmap('tab20c')(7-(2*ind)))
            ax_ts[v%2,v//2].plot(test_years,df_select_test['test_predictions'],'o-',color=plt.get_cmap('tab20c')(11-(2*ind)),label=plot_label)
            ax_ts[v%2,v//2].set_title(valid_scheme + ' valid.')
            ax_ts[v%2,v//2].legend()

            # Plot metrics for valid and test
            ax_metrics[0,0].plot(v+0.1*ind,df_select_valid['valid_MAE'],'o',color=plt.get_cmap('tab20')(2*v+(1-ind)))
            ax_metrics[1,0].plot(v+0.1*ind,df_select_valid['valid_RMSE'],'o',color=plt.get_cmap('tab20')(2*v+(1-ind)))
            ax_metrics[2,0].plot(v+0.1*ind,df_select_valid['valid_ss'],'o',color=plt.get_cmap('tab20')(2*v+(1-ind)))

            if (v == 0) :
                ax_metrics[0,1].plot(v+0.1*ind,df_select_test['test_MAE'],'o',color=plt.get_cmap('tab20')(2*v+(1-ind)), label=istart_labels[istart])
            else:
                ax_metrics[0,1].plot(v+0.1*ind,df_select_test['test_MAE'],'o',color=plt.get_cmap('tab20')(2*v+(1-ind)))
            ax_metrics[1,1].plot(v+0.1*ind,df_select_test['test_RMSE'],'o',color=plt.get_cmap('tab20')(2*v+(1-ind)))
            ax_metrics[2,1].plot(v+0.1*ind,df_select_test['test_ss'],'o',color=plt.get_cmap('tab20')(2*v+(1-ind)))

            # Add climatology metrics
            ax_metrics[0,0].plot([v-0.1,v+0.2],[df_clim_valid_all['clim_valid_MAE'].values[0],df_clim_valid_all['clim_valid_MAE'].values[0]],'-',color='gray')
            ax_metrics[1,0].plot([v-0.1,v+0.2],[df_clim_valid_all['clim_valid_RMSE'].values[0],df_clim_valid_all['clim_valid_RMSE'].values[0]],'-',color='gray')
            ax_metrics[1,1].plot([v-0.1,v+0.2],[df_clim_test_all['clim_test_RMSE'].values[0],df_clim_test_all['clim_test_RMSE'].values[0]],'-',color='gray')
            if (v == 0) :
                ax_metrics[0,1].plot([v-0.1,v+0.2],[df_clim_test_all['clim_test_MAE'].values[0],df_clim_test_all['clim_test_MAE'].values[0]],'-',color='gray', label='Clim.')
            else:
                ax_metrics[0,1].plot([v-0.1,v+0.2],[df_clim_test_all['clim_test_MAE'].values[0],df_clim_test_all['clim_test_MAE'].values[0]],'-',color='gray')

            # Add the CV min-max values
            if (valid_scheme == 'standardk'):
                ax_metrics[0,0].plot([v+0.1*ind,v+0.1*ind],[df_select_valid['valid_MAE_min'],df_select_valid['valid_MAE_max']],'-',color=plt.get_cmap('tab20')(2*v+(1-ind)),linewidth=0.75)
                ax_metrics[1,0].plot([v+0.1*ind,v+0.1*ind],[df_select_valid['valid_RMSE_min'],df_select_valid['valid_RMSE_max']],'-',color=plt.get_cmap('tab20')(2*v+(1-ind)),linewidth=0.75)
                ax_metrics[2,0].plot([v+0.1*ind,v+0.1*ind],[df_select_valid['valid_ss_min'],df_select_valid['valid_ss_max']],'-',color=plt.get_cmap('tab20')(2*v+(1-ind)),linewidth=0.75)

                ax_metrics[0,0].fill_between([v-0.1,v+0.2],[df_clim_valid_all['clim_valid_MAE_min'].values[0],df_clim_valid_all['clim_valid_MAE_min'].values[0]],[df_clim_valid_all['clim_valid_MAE_max'].values[0],df_clim_valid_all['clim_valid_MAE_max'].values[0]],color='gray',alpha=0.1)
                ax_metrics[1,0].fill_between([v-0.1,v+0.2],[df_clim_valid_all['clim_valid_RMSE_min'].values[0],df_clim_valid_all['clim_valid_RMSE_min'].values[0]],[df_clim_valid_all['clim_valid_RMSE_max'].values[0],df_clim_valid_all['clim_valid_RMSE_max'].values[0]],color='gray',alpha=0.1)

            if (valid_scheme == 'LOOk')|(valid_scheme == 'rollingk'):
                ax_metrics[0,0].plot([v+0.1*ind,v+0.1*ind],[np.nanmin(df_select_valid['valid_MAE_min']),np.nanmax(df_select_valid['valid_MAE_max'])],'-',color=plt.get_cmap('tab20')(2*v+(1-ind)),linewidth=0.75)
                ax_metrics[1,0].plot([v+0.1*ind,v+0.1*ind],[np.nanmin(df_select_valid['valid_RMSE_min']),np.nanmax(df_select_valid['valid_RMSE_max'])],'-',color=plt.get_cmap('tab20')(2*v+(1-ind)),linewidth=0.75)
                ax_metrics[2,0].plot([v+0.1*ind,v+0.1*ind],[np.nanmin(df_select_valid['valid_ss_min']),np.nanmax(df_select_valid['valid_ss_max'])],'-',color=plt.get_cmap('tab20')(2*v+(1-ind)),linewidth=0.75)

                ax_metrics[0,0].fill_between([v-0.1,v+0.2],[np.nanmin(df_clim_valid_all['clim_valid_MAE_min'].values[0]),np.nanmin(df_clim_valid_all['clim_valid_MAE_min'].values[0])],[np.nanmax(df_clim_valid_all['clim_valid_MAE_max'].values[0]),np.nanmax(df_clim_valid_all['clim_valid_MAE_max'].values[0])],color='gray',alpha=0.1)
                ax_metrics[1,0].fill_between([v-0.1,v+0.2],[np.nanmin(df_clim_valid_all['clim_valid_RMSE_min'].values[0]),np.nanmin(df_clim_valid_all['clim_valid_RMSE_min'].values[0])],[np.nanmax(df_clim_valid_all['clim_valid_RMSE_max'].values[0]),np.nanmax(df_clim_valid_all['clim_valid_RMSE_max'].values[0])],color='gray',alpha=0.1)


            # ax_metrics[0,1].legend()

#%%
# PLOT TOP N MODELS FOR RMSE AND MAE
best_n = np.min((20,len(df_valid_all)))
if ~np.all(np.isnan(pd.to_numeric(df_valid_all['valid_RMSE']))):
    figMAE,axMAE = plt.subplots()
    axMAE.plot(years,targets ,'o-',color='k')

    df_valid_MAE = pd.DataFrame(pd.to_numeric(df_valid_all['valid_MAE']))
    df_subset = df_valid_MAE.nsmallest(best_n, 'valid_MAE')
    for nmodel in range(np.min([len(df_subset),best_n])):
        nmodel = df_subset.index.values[nmodel]
        plot_label = df_test_all.iloc[nmodel]['predictors']
        axMAE.plot(test_years,df_test_all.iloc[nmodel]['test_predictions'],'o-',label=plot_label)
    axMAE.legend()
    plt.title('Top '+str(best_n)+' MLR models according to '+ 'MAE' + ' - ' + valid_scheme + ' valid. ('+ istart_labels[istart] +')')


    figRMSE,axRMSE = plt.subplots()
    axRMSE.plot(years,targets ,'o-',color='k')

    df_valid_RMSE = pd.DataFrame(pd.to_numeric(df_valid_all['valid_RMSE']))
    df_subset = df_valid_RMSE.nsmallest(best_n, 'valid_RMSE')
    for nmodel in range(np.min([len(df_subset),best_n])):
        nmodel = df_subset.index.values[nmodel]
        plot_label = df_test_all.iloc[nmodel]['predictors']
        axRMSE.plot(test_years,df_test_all.iloc[nmodel]['test_predictions'],'o-',label=plot_label)
    axRMSE.legend()
    plt.title('Top '+str(best_n)+' MLR models according to '+ 'RMSE' + ' - ' + valid_scheme + ' valid. ('+ istart_labels[istart] +')')


    figss,axss = plt.subplots()
    axss.plot(years,targets ,'o-',color='k')

    df_valid_RMSE = pd.DataFrame(pd.to_numeric(df_valid_all['valid_ss']))
    df_subset = df_valid_RMSE.nlargest(best_n, 'valid_ss')
    for nmodel in range(np.min([len(df_subset),best_n])):
        nmodel = df_subset.index.values[nmodel]
        plot_label = df_test_all.iloc[nmodel]['predictors']
        axss.plot(test_years,df_test_all.iloc[nmodel]['test_predictions'],'o-',label=plot_label)
    axss.legend()
    plt.title('Top '+str(best_n)+' MLR models according to '+ 'Skill Score' + ' - ' + valid_scheme + ' valid. ('+ istart_labels[istart] +')')



# PLOT DISTRIBUTION OF VALIDATION MAE AND RMSE DURING K-FOLD CV
if (valid_scheme == 'LOOk') | (valid_scheme == 'standardk'):
    if ~np.all(np.isnan(pd.to_numeric(df_valid_all['valid_RMSE']))):
        figCV_MAE,axCV_MAE = plt.subplots()
        df_valid_MAE = pd.DataFrame(pd.to_numeric(df_valid_all['valid_MAE']))
        df_subset = df_valid_MAE.nsmallest(best_n, 'valid_MAE')

        for imodel in range(np.min([len(df_subset),best_n])):
            nmodel = df_subset.index.values[imodel]
            plot_label = df_test_all.iloc[nmodel]['predictors']
            if valid_scheme == 'LOOk':
                for iyr in range(len(test_years)):
                    axCV_MAE.plot((imodel+32*imodel)+(iyr*1),df_valid_all.iloc[nmodel]['valid_MAE_mean'][iyr],'.',color=plt.get_cmap('tab20')(imodel*2))
                    axCV_MAE.plot([(imodel+32*imodel)+(iyr*1),(imodel+32*imodel)+(iyr*1)],[df_valid_all.iloc[nmodel]['valid_MAE_min'][iyr],df_valid_all.iloc[nmodel]['valid_MAE_max'][iyr]],'-',color=plt.get_cmap('tab20')(imodel*2),linewidth=0.5)
                axCV_MAE.plot([(imodel+32*imodel)+(0),(imodel+32*imodel)+(27)],[df_valid_all.iloc[nmodel]['valid_MAE'],df_valid_all.iloc[nmodel]['valid_MAE']],'-',color=plt.get_cmap('tab20')(imodel*2))
                # axCV_MAE.plot([(imodel+32*imodel)+(0),(imodel+32*imodel)+(27)],[df_test_all.iloc[nmodel]['test_MAE'],df_test_all.iloc[nmodel]['test_MAE']],'--',color=plt.get_cmap('tab20')(imodel*2))
                # axCV_MAE.fill_between([(imodel+32*imodel)+(0),(imodel+32*imodel)+(27)],[df_valid_all.iloc[nmodel]['valid_MAE']-df_valid_all.iloc[nmodel]['valid_MAE_std'],df_valid_all.iloc[nmodel]['valid_MAE']-df_valid_all.iloc[nmodel]['valid_MAE_std']],[df_valid_all.iloc[nmodel]['valid_MAE']+df_valid_all.iloc[nmodel]['valid_MAE_std'],df_valid_all.iloc[nmodel]['valid_MAE']+df_valid_all.iloc[nmodel]['valid_MAE_std']],color=plt.get_cmap('tab20')(imodel*2),alpha=0.2)
            if valid_scheme == 'standardk':
                axCV_MAE.plot((imodel+32*imodel),df_valid_all.iloc[nmodel]['valid_MAE_mean'],'x',color=plt.get_cmap('tab20')(imodel*2))
                axCV_MAE.plot([(imodel+32*imodel),(imodel+32*imodel)],[df_valid_all.iloc[nmodel]['valid_MAE_min'],df_valid_all.iloc[nmodel]['valid_MAE_max']],'-',color=plt.get_cmap('tab20')(imodel*2),linewidth=0.5)
                # axCV_MAE.plot([(imodel+32*imodel)+(-2),(imodel+32*imodel)+(2)],[df_test_all.iloc[nmodel]['test_MAE'],df_test_all.iloc[nmodel]['test_MAE']],'--',color=plt.get_cmap('tab20')(imodel*2))
        axCV_MAE.set_ylabel('MAE (days)')
        # axCV_MAE.legend()
        plt.title(istart_labels[istart])

        figCV_RMSE,axCV_RMSE = plt.subplots()
        df_valid_RMSE = pd.DataFrame(pd.to_numeric(df_valid_all['valid_RMSE']))
        df_subset = df_valid_RMSE.nsmallest(best_n, 'valid_RMSE')

        for imodel in range(np.min([len(df_subset),best_n])):
            nmodel = df_subset.index.values[imodel]
            plot_label = df_test_all.iloc[nmodel]['predictors']
            if valid_scheme == 'LOOk':
                for iyr in range(len(test_years)):
                    axCV_RMSE.plot((imodel+32*imodel)+(iyr*1),df_valid_all.iloc[nmodel]['valid_RMSE_mean'][iyr],'.',color=plt.get_cmap('tab20')(imodel*2))
                    axCV_RMSE.plot([(imodel+32*imodel)+(iyr*1),(imodel+32*imodel)+(iyr*1)],[df_valid_all.iloc[nmodel]['valid_RMSE_min'][iyr],df_valid_all.iloc[nmodel]['valid_RMSE_max'][iyr]],'-',color=plt.get_cmap('tab20')(imodel*2),linewidth=0.5)
                axCV_RMSE.plot([(imodel+32*imodel)+(0),(imodel+32*imodel)+(27)],[df_valid_all.iloc[nmodel]['valid_RMSE'],df_valid_all.iloc[nmodel]['valid_RMSE']],'-',color=plt.get_cmap('tab20')(imodel*2))
                # axCV_RMSE.plot([(imodel+32*imodel)+(0),(imodel+32*imodel)+(27)],[df_test_all.iloc[nmodel]['test_RMSE'],df_test_all.iloc[nmodel]['test_RMSE']],'--',color=plt.get_cmap('tab20')(imodel*2))
                # axCV_RMSE.fill_between([(imodel+32*imodel)+(0),(imodel+32*imodel)+(27)],[df_valid_all.iloc[nmodel]['valid_RMSE']-df_valid_all.iloc[nmodel]['valid_RMSE_std'],df_valid_all.iloc[nmodel]['valid_RMSE']-df_valid_all.iloc[nmodel]['valid_RMSE_std']],[df_valid_all.iloc[nmodel]['valid_RMSE']+df_valid_all.iloc[nmodel]['valid_RMSE_std'],df_valid_all.iloc[nmodel]['valid_RMSE']+df_valid_all.iloc[nmodel]['valid_RMSE_std']],color=plt.get_cmap('tab20')(imodel*2),alpha=0.2)
            if valid_scheme == 'standardk':
                axCV_RMSE.plot((imodel+32*imodel),df_valid_all.iloc[nmodel]['valid_RMSE_mean'],'x',color=plt.get_cmap('tab20')(imodel*2))
                axCV_RMSE.plot([(imodel+32*imodel),(imodel+32*imodel)],[df_valid_all.iloc[nmodel]['valid_RMSE_min'],df_valid_all.iloc[nmodel]['valid_RMSE_max']],'-',color=plt.get_cmap('tab20')(imodel*2),linewidth=0.5)
                # axCV_RMSE.plot([(imodel+32*imodel)+(-2),(imodel+32*imodel)+(2)],[df_test_all.iloc[nmodel]['test_RMSE'],df_test_all.iloc[nmodel]['test_RMSE']],'--',color=plt.get_cmap('tab20')(imodel*2))
        axCV_RMSE.set_ylabel('RMSE (days)')
        # axCV_RMSE.legend()
        plt.title(istart_labels[istart])

        figCV_ss,axCV_ss = plt.subplots()
        df_valid_ss = pd.DataFrame(pd.to_numeric(df_valid_all['valid_ss']))
        df_subset = df_valid_ss.nlargest(best_n, 'valid_ss')

        for imodel in range(np.min([len(df_subset),best_n])):
            nmodel = df_subset.index.values[imodel]
            plot_label = df_test_all.iloc[nmodel]['predictors']
            if valid_scheme == 'LOOk':
                for iyr in range(len(test_years)):
                    axCV_ss.plot((imodel+32*imodel)+(iyr*1),df_valid_all.iloc[nmodel]['valid_ss_mean'][iyr],'x',color=plt.get_cmap('tab20')(imodel*2))
                    axCV_ss.plot([(imodel+32*imodel)+(iyr*1),(imodel+32*imodel)+(iyr*1)],[df_valid_all.iloc[nmodel]['valid_ss_min'][iyr],df_valid_all.iloc[nmodel]['valid_ss_max'][iyr]],'-',color=plt.get_cmap('tab20')(imodel*2),linewidth=0.5)
                axCV_ss.plot([(imodel+32*imodel)+(0),(imodel+32*imodel)+(27)],[df_valid_all.iloc[nmodel]['valid_ss'],df_valid_all.iloc[nmodel]['valid_ss']],'-',color=plt.get_cmap('tab20')(imodel*2))
                # axCV_ss.plot([(imodel+32*imodel)+(0),(imodel+32*imodel)+(27)],[df_test_all.iloc[nmodel]['test_ss'],df_test_all.iloc[nmodel]['test_ss']],'--',color=plt.get_cmap('tab20')(imodel*2))
                # axCV_ss.fill_between([(imodel+32*imodel)+(0),(imodel+32*imodel)+(27)],[df_valid_all.iloc[nmodel]['valid_ss']-df_valid_all.iloc[nmodel]['valid_ss_std'],df_valid_all.iloc[nmodel]['valid_ss']-df_valid_all.iloc[nmodel]['valid_ss_std']],[df_valid_all.iloc[nmodel]['valid_ss']+df_valid_all.iloc[nmodel]['valid_ss_std'],df_valid_all.iloc[nmodel]['valid_ss']+df_valid_all.iloc[nmodel]['valid_ss_std']],color=plt.get_cmap('tab20')(imodel*2),alpha=0.2)
            if valid_scheme == 'standardk':
                axCV_ss.plot((imodel+32*imodel),df_valid_all.iloc[nmodel]['valid_ss_mean'],'x',color=plt.get_cmap('tab20')(imodel*2))
                axCV_ss.plot([(imodel+32*imodel),(imodel+32*imodel)],[df_valid_all.iloc[nmodel]['valid_ss_min'],df_valid_all.iloc[nmodel]['valid_ss_max']],'-',color=plt.get_cmap('tab20')(imodel*2),linewidth=0.5)
                # axCV_ss.plot([(imodel+32*imodel)+(-2),(imodel+32*imodel)+(2)],[df_test_all.iloc[nmodel]['test_ss'],df_test_all.iloc[nmodel]['test_ss']],'--',color=plt.get_cmap('tab20')(imodel*2))
        axCV_ss.set_ylabel('Skill Score')
        # axCV_ss.legend()
        plt.title(istart_labels[istart])


#%%
# Test with boostraping on a fixed number of years out of all available years
# to see how much the mean and std of my test and validation period is different
# from the rest of the FUD time series

# avg_freezeup_doy = np.array([360., 358., 364., 342., 365., 350., 365., 360., 343., 367.,
#                               339., 348., 354., 350., 381., 341., 347., 352., 357., 364., 358.,
#                               347., 365., 371., 351., 348., 356., 354.])
# years = np.arange(1992,2020)

# n_seq = 300
# seq_len = 6
# seq_arr = np.zeros((n_seq,seq_len))

# for n in range(n_seq):
#     i = np.random.randint(0,len(avg_freezeup_doy),size=seq_len)
#     seq_arr[n,:] = avg_freezeup_doy[i]

# mean = np.mean(seq_arr,axis=1)
# std = np.std(seq_arr,axis=1)

# plt.figure()
# plt.boxplot(mean,positions=[1])
# plt.plot(1,np.nanmean(avg_freezeup_doy[np.where(years==2008)[0][0]:np.where(years==2014)[0][0]]),'o',color='green',label='valid set')
# plt.plot(1,np.nanmean(avg_freezeup_doy[np.where(years==2014)[0][0]:]),'o',color='purple',label='test set')
# plt.figure()
# plt.boxplot(std,positions=[1])
# plt.plot(1,np.nanstd(avg_freezeup_doy[np.where(years==2008)[0][0]:np.where(years==2014)[0][0]]),'o',color='green',label='valid set')
# plt.plot(1,np.nanstd(avg_freezeup_doy[np.where(years==2014)[0][0]:]),'o',color='purple',label='test set')


