#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:14:33 2022

@author: Amelie
"""
import os
from moviepy.editor import *
import moviepy.video.fx.all as vfx
import numpy as np
from natsort import natsorted

usb_path = '/run/media/amelie/Seagate Backup Plus Drive'
# usb_path = "/Volumes/Seagate Backup Plus Drive"

# loc = "IleCharron"
loc = "Longueuil"
# loc = "PontJacquesCartier"

speedx_all = 10
speedx_day = 50
speedx_month = 1

year_list = ['2021', '2022', '2022', '2022', '2022']
month_list = ['12',    '01',   '02',   '03',   '04']
ndays_list = [31,       31,     28,     31,     30 ]

#%%
# First combine all videos for one day at a time:
# for n in range(4,5):
#     year = year_list[n]
#     month = month_list[n]

#     day_end = ndays_list[n]
#     if month == '04':
#         day_start = 22
#     else:
#         day_start = 0
    

#     for day in range(day_start,day_end):
#         if (day+1) <  10:
#             day_str = '0'+str(day+1)
#         else:
#             day_str = str(day+1)

#         hourly_availability = np.zeros((24))

#         L =[]
#         for root, dirs, files in os.walk(usb_path+"/amelie/ice/videos/"+loc+"/"+year+month+day_str+"/"):
#             files = natsorted(files)
#             for hour in range(24):
#                 if hour < 10:
#                     hour_str = '0'+str(hour)
#                 else:
#                     hour_str = str(hour)

#                 for file in files:
#                     if (os.path.splitext(file)[0][-4:-2] == hour_str) & (hourly_availability[hour] == 0):
#                         if (os.path.splitext(file)[1] == '.mp4') & (os.path.splitext(file)[0][-4:] != '_all'):
#                             print(file)
#                             filePath = os.path.join(root, file)
#                             try:
#                                 video = VideoFileClip(filePath)
#                                 # print("fps: {}".format(video.fps))
#                                 L.append(video)
#                                 hourly_availability[hour] = 1
#                             except Exception as e:
#                                 print(e)
#                                 # Delete the erroneous file.
#                                 # if os.path.isfile(filePath):
#                                 #     os.remove(filePath)
#         if len(L)>0:
#             # Concatenate all clips
#             final_clip = concatenate_videoclips(L)
#             # Modify the FPS
#             final_clip = final_clip.set_fps(final_clip.fps * speedx_day)
#             # Apply speed up
#             final_clip = final_clip.fx(vfx.speedx, speedx_day)
#             print("fps: {}".format(final_clip.fps))
#             # Save video clip
#             final_clip.to_videofile(usb_path+"/amelie/ice/videos/"+loc+"/"+year+month+day_str+"/"+year+month+day_str+"_all.mp4", fps=final_clip.fps, remove_temp=False)

#             del L, video, final_clip


#%%
# Then combine all videos in a month
# for n in range(5):
#     year = year_list[n]
#     month = month_list[n]

#     L =[]
#     for day in range(ndays_list[n]):
#         if (day+1) <  10:
#             day_str = '0'+str(day+1)
#         else:
#             day_str = str(day+1)

#         for root, dirs, files in os.walk(os.path.join(usb_path+"/amelie/ice/videos/"+loc+"/"+year+month+day_str+"/")):

#             files = natsorted(files)
#             for file in files:
#                 if (os.path.splitext(file)[0][-4:] == '_all') & (os.path.splitext(file)[1] == '.mp4'):
#                     print(file)
#                     filePath = os.path.join(root, file)
#                     try:
#                         video = VideoFileClip(filePath)
#                         # print("fps: {}".format(video.fps))
#                         L.append(video)
#                     except Exception as e:
#                         print(e)

#     if len(L)>0:
#         # Concatenate all clips
#         final_clip = concatenate_videoclips(L)
#         # Modify the FPS
#         final_clip = final_clip.set_fps(final_clip.fps * speedx_month)
#         # Apply speed up
#         final_clip = final_clip.fx(vfx.speedx, speedx_month)
#         print(" ")
#         print("final fps: {}".format(final_clip.fps))
#         # Save video clip
#         final_clip.to_videofile(usb_path+"/amelie/ice/videos/"+loc+"/"+year+month+"_all.mp4", fps=final_clip.fps, remove_temp=False)

#         del L, video, final_clip
# print(" ")

#%%
# And finally combine all monthly videos in 1 file
L =[]

for root, dirs, files in os.walk(usb_path+"/amelie/ice/videos/"+loc+"/"):
    files = natsorted(files)
    for file in files:
        if (os.path.splitext(file)[0][-4:] == '_all') & (os.path.splitext(file)[1] == '.mp4') & (os.path.splitext(root)[0][-len(loc)-1:-1] == loc):
            print(file)
            filePath = os.path.join(root, file)
            try:
                video = VideoFileClip(filePath)
                # print("fps: {}".format(video.fps))
                L.append(video)
            except Exception as e:
                print(e)
                # Delete the erroneous file.
                # if os.path.isfile(filePath):
                #     os.remove(filePath)

if len(L)>0:
    # Concatenate all clips
    final_clip = concatenate_videoclips(L)
    # Modify the FPS
    final_clip = final_clip.set_fps(final_clip.fps * speedx_all)
    # Apply speed up
    final_clip = final_clip.fx(vfx.speedx, speedx_all)
    print(" ")
    print("final fps: {}".format(final_clip.fps))
    # Save video clip
    final_clip.to_videofile(usb_path+"/amelie/ice/videos/"+loc+"/"+loc+"_allwinter.mp4", fps=final_clip.fps, remove_temp=False)

    del L, video, final_clip
print(" ")







