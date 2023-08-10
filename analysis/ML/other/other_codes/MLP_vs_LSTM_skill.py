#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 20:28:32 2021

@author: Amelie
"""

import numpy as np
import matplotlib.pyplot as plt

# NAIVE FORECAST, ONE-STEP AHEAD -----------
# tensor(0.2226)
# tensor(0.2993)
# 0.9995109404621297
# CLIMATOLOGY, ONE-STEP AHEAD -----------
# tensor(1.0619)
# tensor(1.3848)
# 0.9919029173637587


# LINEAR REGRESSION, ONE-STEP AHEAD (200 epochs)-----------
# tensor(0.3338)
# tensor(0.4254)
# 0.9990199121868691
# MLP MODEL, ONE-STEP AHEAD (200 epochs)-----------
# tensor(0.2635)
# tensor(0.3433)
# 0.9993576358347769
# LSTM MODEL, ONE-STEP AHEAD (50 epochs)-----------
# tensor(0.2979)
# tensor(0.3865)
# 0.99921835922735

lead_times = [1,2,5,10,20,40]

clim_MAE = np.ones(len(lead_times))*1.0619
clim_MSE = np.ones(len(lead_times))*1.3848
clim_rsqr = np.ones(len(lead_times))*0.9919029173637587

# MLP RESULTS FOR 200 EPOCHS:
mlp_MAE = [0.2635,0.3140,0.4467,0.5740,0.7465,0.9327]
mlp_MSE = [0.3433,0.4081,0.5979,0.7739,0.9900,1.2280]
mlp_rsqr = [0.9993576358347769,0.9990874519640339,0.9980544440241976,0.9967450959438677,0.9947708420067675,0.9924094648389308]
# LSTM RESULTS FOR 50 EPOCHS:
lstm_MAE = [0.2979,0.3594,0.4895,0.6121,0.7668,0.9453]
lstm_MSE = [0.3865,0.4649,0.6390,0.8038,0.9990,1.2456]
lstm_rsqr = [0.99921835922735,0.9989094546267818,0.9978766309692997,0.9966046112667306,0.9947383039024306,0.992413861380699]

# plt.figure()
# plt.plot(lead_times,clim_MAE,'-',color='black',label='Climatology' )
# plt.plot(lead_times,mlp_MAE,'.-',color=plt.get_cmap('tab20')(0),label='MLP' )
# plt.plot(lead_times,lstm_MAE,'.--',color=plt.get_cmap('tab20')(0),label='LSTM' )
# plt.xlabel('Lead time (forecast window)')
# plt.ylabel('MAE')

plt.figure()
plt.plot(lead_times,clim_MSE,'-',color='black',label='Climatology')
plt.plot(lead_times,mlp_MSE,'.-',color=plt.get_cmap('tab20')(0),label='MLP')
plt.plot(lead_times,lstm_MSE,'.--',color=plt.get_cmap('tab20')(0),label='LSTM')
plt.xlabel('Lead time (forecast window) in days')
plt.ylabel('RMSE ($^{\circ}$C)')
plt.legend()

plt.figure()
plt.plot(lead_times,clim_rsqr,'-',color='black',label='Climatology')
plt.plot(lead_times,mlp_rsqr,'.-',color=plt.get_cmap('tab20')(2),label='MLP')
plt.plot(lead_times,lstm_rsqr,'.--',color=plt.get_cmap('tab20')(2),label='LSTM')
plt.xlabel('Lead time (forecast window) in days')
plt.ylabel('Rsqr')
plt.legend()


#%%

lead_times = [1,2,4,8,16,32,64]

clim_MAE = np.ones(len(lead_times))*1.0619
clim_MSE = np.ones(len(lead_times))*1.3848
clim_rsqr = np.ones(len(lead_times))*0.9919029173637587

# MLP RESULTS FOR 100 EPOCHS:
mlp_MAE = [0.3043,0.3565,0.4437,0.5601,0.6907,0.8748,1.0720]
mlp_MSE = [0.3960,0.4635,0.5882,0.7405,0.9130,1.1464,1.3998]
mlp_rsqr = [0.9991609850739418,0.9988722758958265,0.9981616564288995,
            0.9970567852096707,0.9956061569244633,0.9932335087487482,0.990586530301561]
# LSTM RESULTS FOR 100 EPOCHS:
lstm_MAE = [0.2500,0.3122,0.4116,0.5497,0.7120,0.8751,1.0902]
lstm_MSE = [0.3291,0.4153,0.5527,0.7356,0.9406,1.1514,1.4336]
lstm_rsqr = [0.9994154268318223,0.9990823990810515,0.998329394842195,
             0.9970490246055765,0.9952623770458824,0.9930947947779861,0.9898569219011286]


# plt.figure()
# plt.plot(lead_times,clim_MAE,'-',color='black',label='Climatology' )
# plt.plot(lead_times,mlp_MAE,'.-',color=plt.get_cmap('tab20')(0),label='MLP' )
# plt.plot(lead_times,lstm_MAE,'.--',color=plt.get_cmap('tab20')(0),label='LSTM' )
# plt.xlabel('Lead time (forecast window)')
# plt.ylabel('MAE')

plt.figure()
plt.plot(lead_times,clim_MSE,'-',color='black',label='Climatology')
plt.plot(lead_times,mlp_MSE,'.-',color=plt.get_cmap('tab20')(0),label='MLP')
plt.plot(lead_times,lstm_MSE,'.--',color=plt.get_cmap('tab20')(0),label='LSTM')
plt.xlabel('Lead time (forecast window) in days')
plt.ylabel('RMSE ($^{\circ}$C)')
plt.legend()

plt.figure()
plt.plot(lead_times,clim_rsqr,'-',color='black',label='Climatology')
plt.plot(lead_times,mlp_rsqr,'.-',color=plt.get_cmap('tab20')(2),label='MLP')
plt.plot(lead_times,lstm_rsqr,'.--',color=plt.get_cmap('tab20')(2),label='LSTM')
plt.xlabel('Lead time (forecast window) in days')
plt.ylabel('Rsqr')
plt.legend()



#%%
#FALL ONLY
lead_times = [1,2,4,8,16,32,64]

clim_MAE = np.ones(len(lead_times))*1.2695
clim_MSE = np.ones(len(lead_times))*1.5697
clim_rsqr = np.ones(len(lead_times))*0.943571267443411

# MLP RESULTS FOR 100 EPOCHS:
mlp_MAE = [0.3709,0.4365,0.5174,0.6270,0.6897,0.6588,0.5724]
mlp_MSE = [0.4612,0.5533,0.6984,0.8628,1.0129,0.9692,0.8281]
mlp_rsqr = [0.9936917872829909,0.990800585232653,0.9822761760135265,
            0.969633444369605,0.9515022048961245,0.9493625181425221,0.9424808764580564]
# LSTM RESULTS FOR 100 EPOCHS:
lstm_MAE = [0.2806,0.3809,0.4908,0.6185,0.7444,0.6692,0.5400]
lstm_MSE = [0.3717,0.4986,0.6549,0.8476,1.0463,0.9793,0.8042]
lstm_rsqr = [0.9956457260796281,0.9924514774311778,0.9856413215731105,
             0.9715758672163406,0.9478137158412335,0.9421459859862097,0.9365805289322707]


# plt.figure()
# plt.plot(lead_times,clim_MAE,'-',color='black',label='Climatology' )
# plt.plot(lead_times,mlp_MAE,'.-',color=plt.get_cmap('tab20')(0),label='MLP' )
# plt.plot(lead_times,lstm_MAE,'.--',color=plt.get_cmap('tab20')(0),label='LSTM' )
# plt.xlabel('Lead time (forecast window)')
# plt.ylabel('MAE')

plt.figure()
plt.plot(lead_times,clim_MSE,'-',color='black',label='Climatology')
plt.plot(lead_times,mlp_MSE,'.-',color=plt.get_cmap('tab20')(0),label='MLP')
plt.plot(lead_times,lstm_MSE,'.--',color=plt.get_cmap('tab20')(0),label='LSTM')
plt.xlabel('Lead time (forecast window) in days')
plt.ylabel('RMSE ($^{\circ}$C)')
plt.legend()
plt.title('Fall Only')

plt.figure()
plt.plot(lead_times,clim_rsqr,'-',color='black',label='Climatology')
plt.plot(lead_times,mlp_rsqr,'.-',color=plt.get_cmap('tab20')(2),label='MLP')
plt.plot(lead_times,lstm_rsqr,'.--',color=plt.get_cmap('tab20')(2),label='LSTM')
plt.xlabel('Lead time (forecast window) in days')
plt.ylabel('Rsqr')
plt.legend()
plt.title('Fall Only')




