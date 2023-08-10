#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:32:28 2020

@author: Amelie
"""

import numpy as np
import matplotlib.pyplot as plt



def g_R(t, D_R=13.06, T_air_mean=6.639,period=365):
    """
    Parameters
    ----------
    t : time of forecast, in days since t0 (in days)
    T_air_mean : annual mean air temperature (in Celsius);
    D_R : average depth of river (in meters);
    period : annual cycle of air temperature variation (in days);

    Returns
    -------
    A function of air temperature at time t used to compute river temperature (in Celsius)

    """
    rho_w = 1e3  # [kg/(m^3)]
    Cp_w = 4.182e3 #[J/(kg*C)]

    h_wa = 21.7 # [W/(m^2 degC)]
    k_R = (h_wa / (rho_w*Cp_w*D_R))*sec_per_day # [1/day]

    w_R = 2*np.pi/period # [rad/day]
    alpha_R = np.arctan(w_R/k_R) # [rad]

    C1 = 5.869 # [C]
    S1 = -13.093 # [C]
    a = np.sqrt(C1**2 + S1**2) # [C]
    theta = np.arctan(C1/S1)+ np.pi # [rad]

    return T_air_mean + a * np.cos(alpha_R) * np.sin(w_R*t+theta-alpha_R)


def g_L(t, D_L=23.56, T_air_mean=6.639,period=365):
    """
    Parameters
    ----------
    t : time of forecast, in days since t0 (in days)
    T_air_mean : annual mean air temperature (in Celsius);
    D-L : average depth of Lake Ontario (in meters);
    period : annual cycle of air temperature variation (in days);

    Returns
    -------
    A function of air temperature at time t used to compute lake temperature (in Celsius)

    """
    rho_w = 1e3  # [kg/(m^3)]
    Cp_w = 4.182e3 #[J/(kg*C)]

    h_wa = 21.7 # [W/(m^2 degC)]
    k_L = (h_wa / (rho_w*Cp_w*D_L))*sec_per_day # [1/day]

    w_L = 2*np.pi/period # [rad/day]
    alpha_L = np.arctan(w_L/k_L) #[rad]

    C1 = 5.869 # [C]
    S1 = -13.093 #[C]
    a = np.sqrt(C1**2 + S1**2) # [C]
    theta = np.arctan(C1/S1) + np.pi# [rad]

    return T_air_mean + a * np.cos(alpha_L) * np.sin(w_L*t+theta-alpha_L)



def T_Lake(t, travel_time, t0=1,D=23.56):
    """
    Parameters
    ----------
    t : time of forecast, in days since t0;
    travel_time : time to travel between the upstream and downstream location (in days);
    t0 : initial date of forecast, in days from Oct. 1st.
    D : average depth of river (in meters);

    Returns
    -------
    Forecasted Lake (river head) temperature at time t (in Celsius)

    """
    rho_w = 1e3  # [kg/(m^3)]
    Cp_w = 4.182e3 #[J/(kg*C)]

    h_wa = 21.7 # [W/(m^2 degC)]
    k_L = (h_wa / (rho_w*Cp_w*D))*sec_per_day # [1/day]

    T_Lake_t0 = 16.00 #[C]

    return g_L(t,D_L=D) + np.exp(-k_L*(t-t0))*(T_Lake_t0-g_L(t0,D_L=D))


def T_downstream(t, travel_time, D_R=13.06):
    """
    Parameters
    ----------
    t : time of forecast, in days since t0;
    travel_time : time to travel between the upstream and downstream location (in days);
    D : average depth of river (in meters);

    Returns
    -------
    Forecasted temperature at downstream location at time t (in Celsius)
    """
    rho_w = 1e3  # [kg/(m^3)]
    Cp_w = 4.182e3 #[J/(kg*C)]

    h_wa = 21.7 # [W/(m^2 degC)]
    k_R = (h_wa / (rho_w*Cp_w*D_R) )*sec_per_day # [1/day]

    return g_R(t,D_R=D_R)+ np.exp(-k_R*travel_time)*( T_Lake(t-travel_time,travel_time)-g_R(t-travel_time,D_R=D_R) )


def Ta(t, T_air_mean=6.639,period=365):

    w = 2*np.pi/period # [rad/day]

    C1 = 5.869 # [C]
    S1 =-13.093 #[C]
    a = np.sqrt(C1**2 + S1**2) # [C]
    theta =np.arctan(C1/S1)+np.pi  # [rad]

    return T_air_mean+ a*np.sin(w*t + theta)


#%%
#Lake Ontario discharge: https://ijc.org/fr/clofsl/bassin/debits

sec_per_day = 24*60*60 #[s/day]

dist = 160e3 # Kingston - Cornwall [m]
discharge_downstream = 7000 #[m^3/s]
avg_river_width =  1.5e3 #[m]
avg_river_depth = 13.04 #[m]

avg_cross_section = avg_river_width * avg_river_depth #[m^2]
flow_velocity = discharge_downstream / avg_cross_section #[m/s]
travel_time = (dist/flow_velocity)/sec_per_day #[day]
print('Travel time: ' +'%4.2f'%(travel_time)+' days')

t= np.arange(120)+1 #[days]
T_Massena = T_downstream(t,travel_time,avg_river_depth)
T_Massena[T_Massena<0] = 0

plt.figure()
plt.plot(t,T_Massena)

# plt.figure()
# plt.plot(t,T_Lake(t,travel_time))

# plt.figure()
# plt.plot(t,Ta(t))

# t= np.arange(365)+1
# plt.figure()
# plt.plot(t,Ta(t))

#%%

# FOR CORNWALL TO LAKE ST-FRANCIS - 1986
# discharge at Cornwall: https://wateroffice.ec.gc.ca/report/historical_e.html?stn=02MC002&dataType=Daily&parameterType=Flow&year=1986&mode=Table&page=historical&start_year=1850&end_year=2020
# depth: http://fishing-app.gpsnauticalcharts.com/i-boating-fishing-web-app/fishing-marine-charts-navigation.html?title=ST+LAWRENCE+RIVER+MORRISTOWN+NY+TO+BUTTERNUT+BAY+ONT+boating+app#13.28/45.1638/-74.3552
sec_per_day = 24*60*60 #[s/day]

dist = 50e3 # Cornwall- Lake St-Francis [m]
discharge_downstream = 9336.66 #[m^3/s] discharge at Cornwall 1986 Oct-Nov-Dec mean
avg_river_width =  4e3 #[m]
avg_river_depth = 4 #[m]

avg_cross_section = avg_river_width * avg_river_depth #[m^2]
flow_velocity = discharge_downstream / avg_cross_section #[m/s]
travel_time = (dist/flow_velocity)/sec_per_day #[day]
print('Travel time: ' +'%4.2f'%(travel_time)+' days')

# t= np.arange(120)+1 #[days]
# T_Massena = T_downstream(t,travel_time,avg_river_depth)
# T_Massena[T_Massena<0] = 0

# plt.figure()
# plt.plot(t,T_Massena)

#%%
# # plt.figure()
# t= np.arange(365)+1 #[daya]
# Ta_t=Ta(t)
# # plt.plot(t,Ta_t)

# t = 1.
# D_L = 23.54
# D_R = 13.06

# period = 365
# w = 2*np.pi/period # [rad/day]

# C1 = 5.869 # [C]
# S1 = -13.093 #[C]
# a = np.sqrt(C1**2 + S1**2) # [C]
# theta = np.arctan(C1/S1) # [rad]

# test=(a*np.sin(w*t + theta))


# rho_w = 1e3  # [kg/(m^3)]
# Cp_w = 4.182e3 #[J/(kg*K)]

# h_wa = 21.7 # [W/(m^2 degC)]
# k_L = (h_wa / (rho_w*Cp_w*D_L))*sec_per_day # [1/day]
# k_R = (h_wa / (rho_w*Cp_w*D_R))*sec_per_day# [1/day]

# alpha_L = np.arctan(w/k_L) #[rad]
# alpha_R = np.arctan(w/k_R) #[rad]

# T_Lake_t0 = 16.35 #[C]
# T_air_mean=6.639 #[C]

# TWO = T_Lake_t0
# TO = 1

# #%%
# # RIVER TEMP
# KL = 0.01910924 # [1/day]
# KR = 0.03452020 # [1/day]

# TIL = 2.72018 - np.arctan(w / KL)
# T2R = 2.72018 - np.arctan(w / KR)

# CL = 1 / np.sqrt(1 + (w / KL)**2)
# CR = 1 / np.sqrt(1 + (w/ KR)**2)

# Al = w * t + T2R
# A3 = w * (t-travel_time) + T2R
# ER = np.exp(- KR * travel_time)

# XD = t
# A2 = w * (t - travel_time) + TIL
# A4 = w * (TO) + TIL
# EL = np.exp( - KL * (t-travel_time - TO))
# A = T_air_mean
# B = (TWO - 6.639 ) * EL
# C = 14.348 * CL * np.sin(A2)
# D = 14.348 * EL * CL * np.sin(A4)

# TL = A + B + C - D



# A = 6.639
# B = (TL - 6.639 ) * ER
# C = 14.348 * CR * np.sin(Al)
# D = 14.348 * ER * CR

# TR = A + B + C - D

# #%%
# # # LAKE TEMP
# # XD = t + travel_time
# # A2 = w * (XD - travel_time) + TIL
# # A4 = w * (TO) + TIL
# # EL = np.exp( - KL * (XD-travel_time - TO))
# # A = T_air_mean
# # B = (TWO - 6.639 ) * EL
# # C = 14.348 * CL * np.sin(A2)
# # D = 14.348 * EL * CL * np.sin(A4)

# # TL = A + B + C - D

