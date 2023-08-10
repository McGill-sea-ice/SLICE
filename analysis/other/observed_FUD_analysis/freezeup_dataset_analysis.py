#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:12:42 2020

@author: Amelie
"""
import sys
import os
FCT_DIR = os.path.dirname(os.path.abspath('/Volumes/SeagateUSB/McGill/Postdoc/slice/prog/'+'/prog/'))
if not FCT_DIR in sys.path:
    sys.path.append(FCT_DIR)

from functions import read_csv
import numpy as np

import datetime as dt
import calendar

import matplotlib.pyplot as plt

# ==========================================================================

def clean_csv(arr,columns,nan_id,return_type=float):

    arr = arr[:,columns]
    arr[np.isin(arr,nan_id)] = np.nan

    clean_arr = arr

    return clean_arr.astype(return_type)


def day_of_year_arr(date_arr,flip_date_new_year):

    doy_arr=np.zeros((date_arr.shape[0]))*np.nan

    for i in range(date_arr.shape[0]):
        if np.all(~np.isnan(date_arr[i,:])):
            doy_arr[i] = (dt.datetime(int(date_arr[i,0]),int(date_arr[i,1]),int(date_arr[i,2])) - dt.datetime(int(date_arr[i,0]), 1, 1)).days + 1
            if calendar.isleap(int(date_arr[i,0])):
                doy_arr[i] -= 1 # Remove 1 for leap years so that
                                # e.g. Dec 1st is always DOY = 335

    if flip_date_new_year: doy_arr[doy_arr < 200] = doy_arr[doy_arr < 200]  + 365

    return doy_arr


def plot_DOY_timeseries(csv,title='',fig=None,axes=None,show_legend=False):
    fi = clean_csv(csv,[0,1,2],[999,888])
    si = clean_csv(csv,[3,4,5],[999,888])
    ci = clean_csv(csv,[9,10,11],[999,888])

    doy_fi = day_of_year_arr(fi,True)
    doy_si = day_of_year_arr(si,True)
    doy_ci = day_of_year_arr(ci,True)

    if fig is None:
        fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(7,4))
    plt.plot(np.arange(40),doy_fi,'.-',label=title+' - First ice')
    plt.plot(np.arange(40),doy_si,'.-',label=title+' - Stable ice')
    plt.plot(np.arange(40),doy_ci,'.-',label=title+' - First ice chart')
    plt.ylabel('Date'); plt.xlabel('Year')
    if show_legend: plt.legend(fontsize=8)
    axes.set_xlim([0,40])
    axes.set_ylim([320,412])
    plt.yticks([319,335,349,366,380,397,411],['Nov. 15','Dec. 1','Dec. 15','Jan. 1','Jan. 15','Feb. 1','Feb. 15'])
    axes.yaxis.grid(color=(0.9, 0.9, 0.9))



def plot_DOY_timeseries_one_plot(csv,label,fig,axes,show_legend):
    fi = clean_csv(csv,[0,1,2],[999,888])
    si = clean_csv(csv,[3,4,5],[999,888])
    ci = clean_csv(csv,[9,10,11],[999,888])

    doy_fi = day_of_year_arr(fi,True)
    doy_si = day_of_year_arr(si,True)
    doy_ci = day_of_year_arr(ci,True)

    h1 = axes.plot(np.arange(40),doy_fi,'.-',label='First ice',linewidth=1)
    h2 = axes.plot(np.arange(40),doy_si,'.-',label='Stable ice',linewidth=1)
    h3 = axes.plot(np.arange(40),doy_ci,'.-',label='First ice chart',linewidth=1)
    axes.set_ylabel(label,rotation=15,labelpad=35,fontsize=8)
    axes.yaxis.set_label_coords(-0.25,0.1)
    axes.set_xlim([0,40])
    axes.set_ylim([320,412])
    axes.set_yticks([335,366,397])
    axes.set_yticklabels(['Dec. 1','Jan. 1','Feb. 1'],fontsize=6)
    axes.set_xticks(np.arange(40+1))
    # axes.set_xticklabels(['','1982','','','','1986','','','','1990','',
    #                       '','','1994','','','','1998','','','',
    #                       '2002','','','','2006','','','','2010',
    #                       '','','','2014','','','','2018',''],fontsize=6)

    axes.yaxis.grid(color=(0.9, 0.9, 0.9))
    if show_legend:
        axes.legend(fontsize=5,framealpha=0.6)

def compare_DOY(csv,title=''):
    fi = clean_csv(csv,[0,1,2],[999,888])
    si = clean_csv(csv,[3,4,5],[999,888])
    ci = clean_csv(csv,[9,10,11],[999,888])

    doy_fi = day_of_year_arr(fi,True)
    doy_si = day_of_year_arr(si,True)
    doy_ci = day_of_year_arr(ci,True)

    plt.figure()
    plt.plot(doy_ci,doy_fi,'o', label='DOY of first ice')
    plt.plot(doy_ci,doy_si,'*', label='DOY of stable ice')
    plt.plot(np.arange(np.nanmin(doy_si),np.nanmax(doy_si)),np.arange(np.nanmin(doy_si),np.nanmax(doy_si)),':')
    plt.title(title)
    plt.ylabel('DOY - Seaway Corp.'); plt.xlabel('DOY - Charts')
    plt.legend()


def correlate_timeseries(csv1,csv2,loc1='',loc2=''):
    fi1 = clean_csv(csv1,[0,1,2],[999,888])
    si1 = clean_csv(csv1,[3,4,5],[999,888])
    ci1 = clean_csv(csv1,[9,10,11],[999,888])

    doy_fi1 = day_of_year_arr(fi1,True)
    doy_si1 = day_of_year_arr(si1,True)
    doy_ci1 = day_of_year_arr(ci1,True)

    fi2 = clean_csv(csv2,[0,1,2],[999,888])
    si2 = clean_csv(csv2,[3,4,5],[999,888])
    ci2 = clean_csv(csv2,[9,10,11],[999,888])

    doy_fi2 = day_of_year_arr(fi2,True)
    doy_si2 = day_of_year_arr(si2,True)
    doy_ci2 = day_of_year_arr(ci2,True)

    plt.figure()
    plt.plot(doy_fi1,doy_fi2,'o', label='First ice')
    plt.plot(doy_si1,doy_si2,'*', label='Stable ice')
    plt.plot(doy_ci1,doy_ci2,'+', label='Chart ice')
    plt.plot(np.arange(330,410),np.arange(330,410),':')
    plt.xlabel('DOY - '+loc1)
    plt.ylabel('DOY - '+loc2)
    plt.legend()


# ==========================================================================

fp = '/Users/Amelie/Dropbox/Postdoc/Projet_Fednav/Data/freezeup_dates_merged_from_SLSMC_and_Charts/'

iroquois_csv = read_csv(fp+'freezeup_iroquois.csv',skip=1)
summerstown_csv = read_csv(fp+'freezeup_summerstown.csv',skip=1)
lakestlawrence_csv = read_csv(fp+'freezeup_lakestlawrence.csv',skip=1)
lakestfrancis_csv = read_csv(fp+'freezeup_lakestfrancis.csv',skip=1)
beauharnois_csv = read_csv(fp+'freezeup_beauharnois.csv',skip=1)
lakestlouis_csv = read_csv(fp+'freezeup_lakestlouis.csv',skip=1)
southshore_csv = read_csv(fp+'freezeup_southshore.csv',skip=1)
montrealport_csv = read_csv(fp+'freezeup_montrealport.csv',skip=1)
varennes_contrecoeur_csv = read_csv(fp+'freezeup_varennes_contrecoeur.csv',skip=1)
contrecoeur_sorel_csv = read_csv(fp+'freezeup_contrecoeur_sorel.csv',skip=1)
lakestpierre_csv = read_csv(fp+'freezeup_lakestpierre.csv',skip=1)


#%%
plt.close('all')
correlate_timeseries(lakestfrancis_csv,iroquois_csv,'Lake St-Francis', 'Iroquois')
correlate_timeseries(lakestfrancis_csv,summerstown_csv,'Lake St-Francis', 'Summerstown')
correlate_timeseries(lakestfrancis_csv,lakestlawrence_csv,'Lake St-Francis', 'Lake St-Lawrence')


correlate_timeseries(beauharnois_csv,lakestfrancis_csv,'Beauharnois','Lake St-Francis')
correlate_timeseries(beauharnois_csv,lakestlouis_csv,'Beauharnois','Lake St-Louis')

correlate_timeseries(beauharnois_csv,southshore_csv,'Beauharnois','South Shore Canal')
correlate_timeseries(montrealport_csv,southshore_csv,'Montreal Port','South Shore Canal')
#%%
correlate_timeseries(lakestlouis_csv,southshore_csv,'Lake St-Louis','South Shore Canal')


#%%
# SCATTER PLOT OF CHART DATES VS FREEZEUP DATES FROM THE SLSWC

# compare_DOY(iroquois_csv,'Iroquois')
#compare_DOY(summerstown_csv,'Summerstown')
# compare_DOY(lakestlawrence_csv,'Lake St-Lawrence')
compare_DOY(lakestfrancis_csv,'Lake St-Francis - EAST')
compare_DOY(beauharnois_csv,'Beauharnois Canal')
compare_DOY(lakestlouis_csv,'Lake St-Louis')
compare_DOY(southshore_csv,'South Shore Canal')
#compare_DOY(montrealport_csv,'Montreal Port')
#compare_DOY(varennes_contrecoeur_csv,'Varennes-Contrecoeur')
#compare_DOY(contrecoeur_sorel_csv,'Contrecoeur-Sorel')
#compare_DOY(lakestpierre_csv,'Lake St-Pierre')

#%%
# PLOT ALL TIME SERIES IN ONE PANEL
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(7,4))
# plot_DOY_timeseries(iroquois_csv,'Iroquois',fig,ax)
# plot_DOY_timeseries(summerstown_csv,'Summerstown',fig,ax)
# plot_DOY_timeseries(lakestlawrence_csv,'Lake St-Lawrence',fig,ax)
plot_DOY_timeseries(lakestfrancis_csv,'Lake St-Francis - EAST',fig,ax)
plot_DOY_timeseries(beauharnois_csv,'Beauharnois Canal',fig,ax)
plot_DOY_timeseries(lakestlouis_csv,'Lake St-Louis',fig,ax)
plot_DOY_timeseries(southshore_csv,'South Shore Canal',fig,ax)
# plot_DOY_timeseries(montrealport_csv,'Montreal Port',fig,ax)
# plot_DOY_timeseries(varennes_contrecoeur_csv,'Varennes-Contrecoeur',fig,ax)
# plot_DOY_timeseries(contrecoeur_sorel_csv,'Contrecoeur-Sorel',fig,ax)
# plot_DOY_timeseries(lakestpierre_csv,'Lake St-Pierre',fig,ax)

#%%
# PLOT TIME SERIES IN ONE FIGURE, BUT IN SEPARATE PANELS
# csv_list = [iroquois_csv, summerstown_csv, lakestlawrence_csv ,
#               lakestfrancis_csv, beauharnois_csv ,lakestlouis_csv,
#               southshore_csv, montrealport_csv, varennes_contrecoeur_csv,
#               contrecoeur_sorel_csv, lakestpierre_csv ]

# labels_list = ['Iroquois','Summerstown','Lake\n St-Lawrence',
#                 'Lake\n St-Francis\n EAST','Beauharnois\n Canal','Lake\n St-Louis',
#                 'South Shore\n Canal','Montreal\n Port','Varennes-\nContrecoeur',
#                 'Contrecoeur-\nSorel','Lake\n St-Pierre']

csv_list =[lakestfrancis_csv, beauharnois_csv ,lakestlouis_csv,
southshore_csv]

labels_list = ['Lake\n St-Francis\n EAST','Beauharnois\n Canal','Lake\n St-Louis',
                'South Shore\n Canal']
fig,ax = plt.subplots(nrows=len(labels_list),ncols=1,figsize=(5,12),sharex=True,sharey=True)

for i in range(len(csv_list)):
    if i==0:
        plot_DOY_timeseries_one_plot(csv_list[i],labels_list[i],fig,ax[i],show_legend=True)
    else:
        plot_DOY_timeseries_one_plot(csv_list[i],labels_list[i],fig,ax[i],show_legend=False)


plt.subplots_adjust(top=0.95)
plt.subplots_adjust(right=0.95)
plt.subplots_adjust(left=0.3)
ax[10].set_xlabel('Year',fontsize=8)



#%%
# PLOT BOXPLOTS OF FREEZEUP DATES
csv_list = [iroquois_csv, summerstown_csv, lakestlawrence_csv ,
              lakestfrancis_csv, beauharnois_csv ,lakestlouis_csv,
              southshore_csv, montrealport_csv, varennes_contrecoeur_csv,
              contrecoeur_sorel_csv, lakestpierre_csv ]

labels_list = [ 'Iroquois',
                'Summerstown',
                'Lake\n St-Lawrence',
                'Lake\n St-Francis\n EAST',
                'Beauharnois\n Canal',
                'Lake\n St-Louis',
                'South Shore\n Canal',
                'Montreal\n Port',
                'Varennes-\nContrecoeur',
                'Contrecoeur-\nSorel',
                'Lake\n St-Pierre']


csv_list =[lakestfrancis_csv, beauharnois_csv ,lakestlouis_csv,
southshore_csv]

labels_list = ['Lake\n St-Francis\n EAST',
               'Beauharnois\n Canal',
               'Lake\n St-Louis',
               'South Shore\n Canal']


fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,3))
doy_fi = np.zeros((len(csv_list),40))*np.nan
doy_si = np.zeros((len(csv_list),40))*np.nan
doy_ci = np.zeros((len(csv_list),40))*np.nan
for i in range(len(csv_list)):

    csv = csv_list[i]

    fi = clean_csv(csv,[0,1,2],[999,888])
    si = clean_csv(csv,[3,4,5],[999,888])
    ci = clean_csv(csv,[9,10,11],[999,888])

    doy_fi[i] = day_of_year_arr(fi,True)
    doy_si[i] = day_of_year_arr(si,True)
    doy_ci[i] = day_of_year_arr(ci,True)

    diff = doy_si[i]-doy_fi[i]

    # plt.errorbar([i+1-0.15], np.nanmean(doy_fi), np.nanstd(doy_fi), fmt='xk', lw=1.5)
    bp1 = ax.boxplot(doy_fi[i][~np.isnan(doy_fi[i])], positions=[i+1-0.15],
                    sym='.', widths=0.12, patch_artist=True, whis= [10,90],
                    boxprops={'facecolor':'None','edgecolor':plt.get_cmap('tab20')(0),'linewidth':0.75},
                    whiskerprops={'color':plt.get_cmap('tab20')(0),'linewidth':1},
                    capprops={'color':plt.get_cmap('tab20')(0),'linewidth':1},
                    flierprops={'markerfacecolor':plt.get_cmap('tab20')(0),'markeredgecolor':'None'},
                    medianprops={'color':plt.get_cmap('tab20')(0),'linewidth':1.1},
                    showmeans=False,meanline=True, meanprops={'color':'gray', 'linestyle':'-'})
    for patch in bp1['boxes']:
            patch.set_facecolor('None')
            patch.set_edgecolor(plt.get_cmap('tab20')(0))

    # plt.errorbar([i+1], np.nanmean(doy_si), np.nanstd(doy_si), fmt='xr', lw=1.5)
    bp2 = ax.boxplot(doy_si[i][~np.isnan(doy_si[i])], positions=[i+1],
                    sym='.', widths=0.12, patch_artist=True, whis= [10,90],
                    boxprops={'facecolor':'None','edgecolor':plt.get_cmap('tab20')(2),'linewidth':0.75},
                    whiskerprops={'color':plt.get_cmap('tab20')(2),'linewidth':1},
                    capprops={'color':plt.get_cmap('tab20')(2),'linewidth':1},
                    flierprops={'markerfacecolor':plt.get_cmap('tab20')(2),'markeredgecolor':'None'},
                    medianprops={'color':plt.get_cmap('tab20')(2),'linewidth':1.1},
                    showmeans=False,meanline=True, meanprops={'color':'r', 'linestyle':'-'})
    for patch in bp2['boxes']:
            patch.set_facecolor('None')
            patch.set_edgecolor(plt.get_cmap('tab20')(2))

    # plt.errorbar([i+1+0.15], np.nanmean(doy_ci), np.nanstd(doy_ci), fmt='xb', lw=1.5)
    bp3 = ax.boxplot(doy_ci[i][~np.isnan(doy_ci[i])], positions=[i+1+0.15],
                    sym='.', widths=0.12, patch_artist=True, whis= [10,90],
                    boxprops={'facecolor':'None','edgecolor':plt.get_cmap('tab20')(4),'linewidth':0.75},
                    whiskerprops={'color':plt.get_cmap('tab20')(4),'linewidth':1},
                    capprops={'color':plt.get_cmap('tab20')(4),'linewidth':1},
                    flierprops={'markerfacecolor':plt.get_cmap('tab20')(4),'markeredgecolor':'None'},
                    medianprops={'color':plt.get_cmap('tab20')(4),'linewidth':1.1},
                    showmeans=False,meanline=True, meanprops={'color':'c', 'linestyle':'-'})
    for patch in bp3['boxes']:
            patch.set_facecolor('None')
            patch.set_edgecolor(plt.get_cmap('tab20')(4))

    print(np.nanmean(doy_fi[i][~np.isnan(doy_fi[i])]),np.nanmean(doy_si[i][~np.isnan(doy_si[i])]),np.nanmean(doy_ci[i][~np.isnan(doy_ci[i])]))


plt.legend([bp1['boxes'][0],bp2['boxes'][0],bp3['boxes'][0]],
           ['First ice','Stable ice','Chart ice'],
           fontsize=8, framealpha=0.4, loc='best')

ax.set_ylim([319,412])
plt.xticks(np.arange(len(csv_list))+1,labels_list,rotation=55, fontsize=8)
plt.yticks([319,335,349,366,380,397,411],['Nov. 15','Dec. 1','Dec. 15','Jan. 1','Jan. 15','Feb. 1','Feb. 15'],fontsize=8)
plt.ylabel('Date')
plt.subplots_adjust(top=0.9,bottom=0.3)
plt.subplots_adjust(left=0.1,right=0.95)
ax.yaxis.grid(color=(0.9, 0.9, 0.9))

#%%
# PLOT BOXPLOTS OF DIFFERENCE IN TIMING BETWEEN THE DIFFERENT STAGES
# csv_list =[lakestfrancis_csv, beauharnois_csv ,lakestlouis_csv,
# southshore_csv]

csv_list = [iroquois_csv, summerstown_csv, lakestlawrence_csv ,
              lakestfrancis_csv, beauharnois_csv ,lakestlouis_csv,
              southshore_csv, montrealport_csv, varennes_contrecoeur_csv,
              contrecoeur_sorel_csv, lakestpierre_csv ]

labels_list = ['Iroquois','Summerstown','Lake\n St-Lawrence',
                'Lake\n St-Francis\n EAST','Beauharnois\n Canal','Lake\n St-Louis',
                'South Shore\n Canal','Montreal\n Port','Varennes-\nContrecoeur',
                'Contrecoeur-\nSorel','Lake\n St-Pierre']

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,3))
for i in range(len(csv_list)):

    csv = csv_list[i]

    fi = clean_csv(csv,[0,1,2],[999,888])
    si = clean_csv(csv,[3,4,5],[999,888])
    ci = clean_csv(csv,[9,10,11],[999,888])

    doy_fi = day_of_year_arr(fi,True)
    doy_si = day_of_year_arr(si,True)
    doy_ci = day_of_year_arr(ci,True)

    diff_sifi = doy_si-doy_fi
    diff_cifi = doy_ci-doy_fi
    diff_cisi = doy_ci-doy_si


    bp1 = ax.boxplot(diff_sifi[~np.isnan(diff_sifi)], positions=[i+1-0.15],
                    sym='.', widths=0.12, patch_artist=True, whis= [10,90],
                    boxprops={'facecolor':'None','edgecolor':[0.8,0.8,0.8],'linewidth':0.75},
                    whiskerprops={'color':[0.8,0.8,0.8],'linewidth':1},
                    capprops={'color':[0.8,0.8,0.8],'linewidth':1},
                    flierprops={'markerfacecolor':[0.8,0.8,0.8],'markeredgecolor':'None'},
                    medianprops={'color':[0.8,0.8,0.8],'linewidth':1.1},
                    showmeans=False,meanline=True, meanprops={'color':'r', 'linestyle':'-'})
    for patch in bp1['boxes']:
            patch.set_facecolor('None')
            patch.set_edgecolor([0.8,0.8,0.8])

    bp2 = ax.boxplot(diff_cifi[~np.isnan(diff_cifi)], positions=[i+1],
                    sym='.', widths=0.12, patch_artist=True, whis= [10,90],
                    boxprops={'facecolor':'None','edgecolor':[0.5,0.5,0.5],'linewidth':0.75},
                    whiskerprops={'color':[0.5,0.5,0.5],'linewidth':1},
                    capprops={'color':[0.5,0.5,0.5],'linewidth':1},
                    flierprops={'markerfacecolor':[0.5,0.5,0.5],'markeredgecolor':'None'},
                    medianprops={'color':[0.5,0.5,0.5],'linewidth':1.1},
                    showmeans=False,meanline=True, meanprops={'color':'r', 'linestyle':'-'})
    for patch in bp2['boxes']:
            patch.set_facecolor('None')
            patch.set_edgecolor([0.5,0.5,0.5])

    bp3 = ax.boxplot(diff_cisi[~np.isnan(diff_cisi)], positions=[i+1+0.15],
                    sym='.', widths=0.12, patch_artist=True, whis= [10,90],
                    boxprops={'facecolor':'None','edgecolor':'black','linewidth':0.75},
                    whiskerprops={'color':'black','linewidth':1},
                    capprops={'color':'black','linewidth':1},
                    flierprops={'markerfacecolor':'black','markeredgecolor':'None'},
                    medianprops={'color':'black','linewidth':1.1},
                    showmeans=False,meanline=True, meanprops={'color':'c', 'linestyle':'-'},labels=['test'])
    for patch in bp3['boxes']:
            patch.set_facecolor('None')
            patch.set_edgecolor('black')


    print(labels_list[i],np.nanmean(diff_sifi[~np.isnan(diff_sifi)]),np.nanmean(diff_cifi[~np.isnan(diff_cifi)]),np.nanmean(diff_cisi[~np.isnan(diff_cisi)]))



plt.legend([bp1['boxes'][0],bp2['boxes'][0],bp3['boxes'][0]],
           ['Stable ice vs First ice','Chart ice vs First ice','Chart ice vs Stable ice'],
           fontsize=8)

plt.xticks(np.arange(len(csv_list))+1,labels_list,rotation=55, fontsize=8)
plt.yticks([-30,-20,-10,0,10,20,30,40],['','-20','','0','','20','','40'])
plt.ylabel('$\Delta$ t (days)')
plt.subplots_adjust(top=0.9,bottom=0.3)
plt.subplots_adjust(left=0.1,right=0.95)
ax.yaxis.gridcolor=((0.95, 0.95, 0.95))

#%%
# csv_list = [iroquois_csv, summerstown_csv, lakestlawrence_csv ,
#               lakestfrancis_csv, beauharnois_csv ,lakestlouis_csv,
#               southshore_csv, montrealport_csv, varennes_contrecoeur_csv,
#               contrecoeur_sorel_csv, lakestpierre_csv ]

# labels_list = ['Iroquois','Summerstown','Lake\n St-Lawrence',
#                 'Lake\n St-Francis\n EAST','Beauharnois\n Canal','Lake\n St-Louis',
#                 'South Shore\n Canal','Montreal\n Port','Varennes-\nContrecoeur',
#                 'Contrecoeur-\nSorel','Lake\n St-Pierre']

# fig,ax = plt.subplots(nrows=1,ncols=1)

# diff=[]
# for i in range(len(csv_list)):
#     csv = csv_list[i]

#     fi = clean_csv(csv,[0,1,2],[999,888])
#     si = clean_csv(csv,[3,4,5],[999,888])
#     ci = clean_csv(csv,[9,10,11],[999,888])

#     doy_fi = day_of_year_arr(fi,True)
#     doy_si = day_of_year_arr(si,True)
#     doy_ci = day_of_year_arr(ci,True)

#     diff.append(doy_si -doy_fi)

#     plt.scatter(doy_fi,doy_si)

# plt.plot(np.arange(320,410),np.arange(320,410),':',color='k')

# diff_median = np.nanmedian(np.array(diff))
# plt.plot(np.arange(320,410),np.arange(320,410)+diff_median,':',color=[0.7,0.7,0.7])

# diff_q75 = np.nanquantile(np.array(diff),0.75)
# plt.plot(np.arange(320,410),np.arange(320,410)+diff_q75,':',color=[0.8,0.8,0.8])

# diff_q95 = np.nanquantile(np.array(diff),0.95)
# plt.plot(np.arange(320,410),np.arange(320,410)+diff_q95,':',color=[0.9,0.9,0.9])


#%%

# csv1 = varennes_contrecoeur_csv
# csv2 = contrecoeur_sorel_csv

# fi1 = clean_csv(csv1,[0,1,2],[999,888])
# si1 = clean_csv(csv1,[3,4,5],[999,888])
# ci1 = clean_csv(csv1,[9,10,11],[999,888])

# doy_fi1 = day_of_year_arr(fi1,True)
# doy_si1 = day_of_year_arr(si1,True)
# doy_ci1 = day_of_year_arr(ci1,True)

# fi2 = clean_csv(csv2,[0,1,2],[999,888])
# si2 = clean_csv(csv2,[3,4,5],[999,888])
# ci2 = clean_csv(csv2,[9,10,11],[999,888])

# doy_fi2 = day_of_year_arr(fi2,True)
# doy_si2 = day_of_year_arr(si2,True)
# doy_ci2 = day_of_year_arr(ci2,True)

# diff = doy_ci1-doy_ci2
# print(np.nanmean(diff))

# test_ci1 = np.zeros(doy_ci1.shape)*np.nan
# test_ci2 = np.zeros(doy_ci1.shape)*np.nan

# for i in range(test_ci1.shape[0]):

#     if np.sum([~np.isnan(doy_ci1[i]),~np.isnan(doy_ci2[i])]) > 0 :
#         if (~np.isnan(doy_ci1[i]) & np.isnan(doy_ci2[i])):
#             test_ci2[i] = doy_ci1[i]-np.nanmean(diff)
#             test_ci1[i] = doy_ci1[i]
#         elif (np.isnan(doy_ci1[i]) & ~np.isnan(doy_ci2[i])):
#             test_ci1[i] = doy_ci2[i]+np.nanmean(diff)
#             test_ci2[i] = doy_ci2[i]
#         else:
#             test_ci1[i] = doy_ci1[i]
#             test_ci2[i] = doy_ci2[i]


# fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(6,3))
# plt.plot(np.arange(40),test_ci1,'.-',color='Magenta')
# plt.plot(np.arange(40),test_ci2,'.-',color='Black')
# plt.ylabel('DOY'); plt.xlabel('Year')
# plt.legend()
# axes.set_xlim([0,40])
# axes.set_ylim([330,410])

