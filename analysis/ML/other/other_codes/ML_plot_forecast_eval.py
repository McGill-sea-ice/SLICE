#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 14:35:50 2021

@author: Amelie
"""
import numpy as np
import matplotlib.pyplot as plt
import cmocean

model_name = 'MLP'
input_width = 96
nepochs = 200
# horizon_arr = [8,16,24,32,42,64,96,128]
horizon_arr = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
# horizon_arr = [5,10,15,20,25,30]
suffix = ''
# suffix = '_withsnow'

# model_name = 'LSTM'
# horizon_arr = [5,10,15,20,25,30,35,40,45,50,55,60,65,70]
# input_width = 96
# nepochs = 100
# suffix = ''

plot_Tw_metric = np.zeros((len(horizon_arr),12))
plot_Tw_clim_metric = np.zeros((len(horizon_arr),12))
plot_fu_metric = np.zeros((len(horizon_arr),6))
plot_fu_clim_metric = np.zeros((len(horizon_arr),6))

for ih,h in enumerate(horizon_arr):
    metric_data = np.load('./eval_metrics/'+model_name+'_FORECAST_EVAL_horizon_'+str(h)+'_input_'+str(input_width)+'_nepochs_'+str(nepochs)+suffix+'.npz',allow_pickle='TRUE')
    Tw_eval_arr = np.squeeze(metric_data['Tw_eval_arr'])
    Tw_clim_eval_arr = np.squeeze(metric_data['Tw_clim_eval_arr'])
    fu_eval_arr = np.squeeze(metric_data['fu_eval_arr'])
    fu_clim_eval_arr = np.squeeze(metric_data['fu_clim_eval_arr'])

    plot_Tw_metric[ih,:] = Tw_eval_arr
    plot_Tw_clim_metric[ih,:] = Tw_clim_eval_arr
    plot_fu_metric[ih,:] = fu_eval_arr
    plot_fu_clim_metric[ih,:] = fu_clim_eval_arr


#%%
vmin = 0; vmax = 21; pivot = 7.77
cmap = cmocean.cm.curl
# cmap = cmocean.cm.balance
# cmap = cmocean.cm.diff
# cmap = cmocean.cm.tarn
crop_cmap = cmocean.tools.crop(cmap, vmin, vmax, pivot)
# newcmap = cmocean.tools.crop_by_percent(cmap, 30, which='both', N=None)

fig, axs = plt.subplots(1, 1, figsize=(6,4))
mappable = axs.pcolormesh(np.flipud(plot_fu_metric), cmap=crop_cmap, vmin=vmin, vmax=vmax)
axs.set_title('Freeze-up date MAE (days) - ' + model_name)
fig.colorbar(mappable, ax=axs)

fig_clim, axs_clim = plt.subplots(1, 1, figsize=(6,4))
mappable = axs_clim.pcolormesh(np.flipud(plot_fu_clim_metric), cmap=crop_cmap, vmin=vmin, vmax=vmax)
axs_clim.set_title('Freeze-up date MAE (days) - Climatology')
fig_clim.colorbar(mappable, ax=axs_clim)

vmin = -10; vmax = 10; pivot = 0
cmap = cmocean.cm.balance
# crop_cmap2 = cmocean.tools.crop(cmap, vmin, vmax, pivot)
fig_diff, axs_diff = plt.subplots(1, 1, figsize=(6,4))
mappable = axs_diff.pcolormesh(np.flipud(plot_fu_metric)-np.flipud(plot_fu_clim_metric), cmap=cmap, vmin=vmin, vmax=vmax)
axs_diff.set_title('FUD mean absolute error diff. (days)\n ' + model_name+' - Climatology')
fig_diff.colorbar(mappable, ax=axs_diff)



#%%
vmin = 0; vmax = 1.5; pivot = 1
cmap = cmocean.cm.tempo
cmap = cmocean.cm.dense
cmap = cmocean.cm.deep
cmap = cmocean.cm.thermal
cmap = plt.get_cmap('cividis')
cmap = plt.get_cmap('viridis')
cmap = plt.get_cmap('magma')
# crop_cmap = cmocean.tools.crop(cmap, vmin, vmax, pivot)
# newcmap = cmocean.tools.crop_by_percent(cmap, 30, which='both', N=None)

fig, axs = plt.subplots(1, 1, figsize=(6,4))
mappable = axs.pcolormesh(np.flipud(plot_Tw_metric), cmap=cmap, vmin=vmin, vmax=vmax)
axs.set_title('Water temp. MAE (deg. C) - ' + model_name)
fig.colorbar(mappable, ax=axs)

fig_clim, axs_clim = plt.subplots(1, 1, figsize=(6,4))
mappable = axs_clim.pcolormesh(np.flipud(plot_Tw_clim_metric), cmap=cmap, vmin=vmin, vmax=vmax)
axs_clim.set_title('Water temp. MAE (deg. C) - Climatology')
fig_clim.colorbar(mappable, ax=axs_clim)

vmin = -0.5; vmax = 0.5; pivot = 0
cmap = cmocean.cm.balance
# crop_cmap2 = cmocean.tools.crop(cmap, vmin, vmax, pivot)
fig_diff, axs_diff = plt.subplots(1, 1, figsize=(6,4))
mappable = axs_diff.pcolormesh(np.flipud(plot_Tw_metric)-np.flipud(plot_Tw_clim_metric), cmap=cmap, vmin=vmin, vmax=vmax)
axs_diff.set_title('$T_{w}$ mean absolute error diff. (deg. C)\n ' + model_name+' - Climatology')
fig_diff.colorbar(mappable, ax=axs_diff)

