#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:05:52 2022

@author: Amelie
"""

sec_per_day = 24*60*60 #[s/day]

dist = 250e3 # Kingston - Montreal [m]
discharge_downstream = 8000 #[m^3/s]
avg_river_width =  2e3 #[m]
avg_river_depth = 10 #[m]

avg_cross_section = avg_river_width * avg_river_depth #[m^2]
flow_velocity = discharge_downstream / avg_cross_section #[m/s]
travel_time = (dist/flow_velocity)/sec_per_day #[day]
print('Travel time: ' +'%4.2f'%(travel_time)+' days')


# t = L*(w*d)/Q












